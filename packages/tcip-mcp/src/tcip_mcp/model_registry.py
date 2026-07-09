"""Model registry — track trained models and their performance."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from tcip_mcp.utils.atomic_io import atomic_write_json, file_transaction


def _compute_sha256(filepath: str | Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelRegistry:
    """Simple file-based model registry in .tcip/models/."""

    def __init__(self, project_path: str) -> None:
        self.root = Path(project_path) / ".tcip" / "models"
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / "registry.json"
        self._index: list[dict] = self._load_index()

    def _load_index(self) -> list[dict]:
        if self._index_path.is_file():
            with open(self._index_path) as f:
                return json.load(f)
        return []

    def _save_index(self) -> None:
        atomic_write_json(self._index_path, self._index)

    def register_model(
        self,
        name: str,
        checkpoint_path: str,
        config: dict,
        metrics: dict | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        """Register a trained model with SHA-256 integrity checksum.

        Args:
            name: Model name (e.g. 'hazelnut_catkin_detector_v1').
            checkpoint_path: Path to the .pt checkpoint file.
            config: Training config dict.
            metrics: Evaluation metrics dict.
            tags: Optional tags for filtering.
        """
        ckpt = Path(checkpoint_path)
        sha256 = _compute_sha256(ckpt) if ckpt.is_file() else None
        file_size = ckpt.stat().st_size if ckpt.is_file() else None

        entry = {
            "name": name,
            "checkpoint_path": checkpoint_path,
            "sha256": sha256,
            "file_size_bytes": file_size,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "metrics": metrics or {},
            "tags": tags or [],
        }
        # Lock-guarded read-modify-write: re-read the on-disk index under the lock so a
        # concurrent writer's entries aren't clobbered, replace-by-name, then atomic save.
        with file_transaction(self._index_path):
            self._index = self._load_index()
            self._index = [e for e in self._index if e["name"] != name]
            self._index.append(entry)
            self._save_index()
        return entry

    def verify_model(self, name: str) -> dict:
        """Verify a model checkpoint's integrity against stored checksum.

        Returns:
            dict with 'valid' (bool), 'expected' (str), 'actual' (str|None), 'error' (str|None).
        """
        model = self.get_model(name)
        if model is None:
            return {"valid": False, "error": f"Model '{name}' not found in registry"}

        stored_hash = model.get("sha256")
        if stored_hash is None:
            return {"valid": False, "error": "No checksum stored for this model"}

        ckpt_path = Path(model["checkpoint_path"])
        if not ckpt_path.is_file():
            return {"valid": False, "expected": stored_hash, "actual": None, "error": "Checkpoint file not found"}

        actual_hash = _compute_sha256(ckpt_path)
        return {
            "valid": actual_hash == stored_hash,
            "expected": stored_hash,
            "actual": actual_hash,
            "error": None if actual_hash == stored_hash else "Checksum mismatch — file may be corrupted or modified",
        }

    def list_models(self, tag: str | None = None) -> list[dict]:
        """List registered models, optionally filtered by tag."""
        if tag is None:
            return self._index
        return [m for m in self._index if tag in m.get("tags", [])]

    def get_model(self, name: str) -> dict | None:
        """Get a model entry by name."""
        for m in self._index:
            if m["name"] == name:
                return m
        return None

    def best_model(
        self, metric_key: str = "val_map50", *, higher_is_better: bool | None = None
    ) -> dict | None:
        """Get the registered model with the best value for ``metric_key``.

        Only models that actually carry the metric are considered — a missing metric is
        skipped, not scored as a sentinel, so ``None`` cleanly means "no model has it".
        ``higher_is_better`` defaults to a name heuristic: loss/error keys rank ascending,
        everything else (map/f1/accuracy/…) descending.
        """
        if higher_is_better is None:
            lk = metric_key.lower()
            higher_is_better = not ("loss" in lk or "error" in lk)
        best = None
        best_val: float | None = None
        for m in self._index:
            val = m.get("metrics", {}).get(metric_key)
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                continue
            if best_val is None or (val > best_val if higher_is_better else val < best_val):
                best_val = float(val)
                best = m
        return best
