"""Artifact manager — track training outputs, checkpoints, predictions."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path


class ArtifactManager:
    """Manages training artifacts in .tcip/artifacts/."""

    def __init__(self, project_path: str) -> None:
        self.root = Path(project_path) / ".tcip" / "artifacts"
        self.root.mkdir(parents=True, exist_ok=True)

    def register(
        self,
        artifact_type: str,
        name: str,
        source_path: str,
        metadata: dict | None = None,
    ) -> dict:
        """Register a new artifact (copies file to managed storage).

        Args:
            artifact_type: Type ('checkpoint', 'prediction', 'export', 'config').
            name: Human-readable name.
            source_path: Path to the file to register.
            metadata: Optional metadata dict.
        """
        ts = int(time.time())
        dest_dir = self.root / artifact_type
        dest_dir.mkdir(parents=True, exist_ok=True)

        src = Path(source_path)
        dest = dest_dir / f"{ts}_{src.name}"
        shutil.copy2(src, dest)

        meta = {
            "name": name,
            "type": artifact_type,
            "path": str(dest),
            "original_path": source_path,
            "timestamp": ts,
            **(metadata or {}),
        }
        meta_path = dest.with_suffix(dest.suffix + ".meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return meta

    def list_artifacts(self, artifact_type: str | None = None) -> list[dict]:
        """List registered artifacts, optionally filtered by type."""
        results: list[dict] = []
        search_dirs = [self.root / artifact_type] if artifact_type else list(self.root.iterdir())

        for d in search_dirs:
            if not d.is_dir():
                continue
            for meta_file in sorted(d.glob("*.meta.json")):
                with open(meta_file) as f:
                    results.append(json.load(f))

        return results

    def get_artifact(self, artifact_path: str) -> dict | None:
        """Get metadata for a specific artifact."""
        meta_path = Path(artifact_path).with_suffix(Path(artifact_path).suffix + ".meta.json")
        if meta_path.is_file():
            with open(meta_path) as f:
                return json.load(f)
        return None
