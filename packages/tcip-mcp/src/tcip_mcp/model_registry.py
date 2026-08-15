"""Model registry, track trained models and their performance."""

from __future__ import annotations

import hashlib
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import tcip_store
from tcip_store import RECORD_JSON, Key, StoreDescriptor, check_json_value, register_store
from tcip_store.file_backend import RootedFileLocator

logger = logging.getLogger(__name__)

# ── the registry store ───────────────────────────────────────────────────────

_INDEX_DOC = RootedFileLocator(prefix=(".tcip", "models"), suffix=".json")
"""The registry index, one per project."""

MODEL_REGISTRY_STORE = "model_registry"
_INDEX_PARTS = ("registry",)
register_store(
    StoreDescriptor(
        name=MODEL_REGISTRY_STORE,
        kind="record",
        key_fields=("document",),
        codec=RECORD_JSON,
        concurrency="cas",
        locator=_INDEX_DOC,
    )
)


def registry_index_key(project_path: str | Path) -> Key:
    """The project's registered-model index.

    ``cas``: ``register_model`` re-reads the index under a lock, replaces one entry by name
    and writes the whole list back, so an unconditional write drops every model another
    writer registered in between.
    """
    return Key(MODEL_REGISTRY_STORE, str(project_path), _INDEX_PARTS)


def registry_index_path(project_path: str | Path) -> Path:
    """Where the project's registry index lives on disk."""
    return Path(project_path, *_INDEX_DOC.relative_path(str(project_path), _INDEX_PARTS).parts)


def read_registry_index(project_path: str | Path) -> list[dict]:
    """Every model entry the project has registered, in registration order.

    The read path for anything outside this module: a reader takes the entries from here
    rather than parsing the index document itself, so the entry keys have one owner and a
    key the writer stops emitting shows up as a reader going quiet. A project that has
    registered nothing reads as an empty list; an index that exists but does not decode
    raises ``DecodeError``, because a corrupt registry is not a project with no models.
    """
    return tcip_store.read(registry_index_key(project_path), default=[])


def _compute_sha256(filepath: str | Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# Process-level cache so a multi-GB checkpoint is hashed once per run, not once per delivery
# call. Keyed by (resolved path, size, mtime) so an edited/replaced file re-hashes.
_SHA_CACHE: dict[tuple[str, int, int], str] = {}


def checkpoint_sha256(filepath: str | Path) -> str | None:
    """Cached SHA-256 of a checkpoint file (``None`` if missing).

    Readers of a delivered phenotype resolve the producing checkpoint's identity through here so
    they carry its content hash without re-hashing the file on every call.
    """
    p = Path(filepath)
    if not p.is_file():
        return None
    st = p.stat()
    key = (str(p.resolve()), st.st_size, int(st.st_mtime))
    sha = _SHA_CACHE.get(key)
    if sha is None:
        sha = _compute_sha256(p)
        _SHA_CACHE[key] = sha
    return sha


def resolve_model_identity(
    checkpoint_path: str | Path,
    *,
    experiment_id: str | None = None,
    project_path: str | None = None,
) -> dict:
    """Best-effort producing-model identity for a checkpoint: ``{checkpoint, sha256, experiment_id}``.

    ``sha256`` is the cached content hash (never re-hashed per call). ``experiment_id`` resolves,
    in order: the caller's explicit value; the checkpoint payload's own stamped ``experiment_id``
    (every checkpoint saved through the audited envelope carries one via ``stamp_model_ref``,
    without this, the ordinary train-then-calibrate workflow silently resolved to ``None`` and
    bypassed the train-disjointness gate entirely, not just genuinely-foreign checkpoints); then a
    best-effort registry lookup (a checkpoint registered but never carrying the stamp, e.g. from
    before the stamp existed). A raw/foreign checkpoint legitimately has no experiment, the
    identity records the sha and leaves ``experiment_id`` ``None`` rather than failing.

    Reads the checkpoint with ``weights_only=True``, sufficient to read the stamped ``str`` (a
    payload of tensors/dicts/basic Python types, which is what ``stamp_model_ref`` produces and
    what this reads back) without executing arbitrary pickle content or loading the full weights
    just to read one field. A checkpoint that fails even this safe load is logged and treated as
    "no stamp", the same ``experiment_id=None`` outcome as a genuinely foreign checkpoint, but a
    distinct, visible log line rather than an indistinguishable silent ``pass``, since "the stamp
    couldn't be read" and "there was never a stamp" are materially different situations.
    """
    from tcip_mcp.project_paths import project_root

    ckpt = Path(checkpoint_path)
    sha = checkpoint_sha256(ckpt)
    exp = experiment_id
    if exp is None and ckpt.is_file():
        try:
            import torch  # local checkpoint the caller is deliberately identifying

            payload = torch.load(ckpt, map_location="cpu", weights_only=True)
            if isinstance(payload, dict):
                stamped = payload.get("experiment_id")
                if isinstance(stamped, str) and stamped:
                    exp = stamped
        except Exception as exc:
            logger.warning(
                "could not read a stamped experiment_id from checkpoint %s (%s); treating it as "
                "unstamped for this identity resolution (falls through to the registry lookup, "
                "then a foreign/no-provenance identity), which is distinct from a checkpoint that "
                "genuinely never carried a stamp.", ckpt, exc,
            )
    if exp is None and sha is not None:
        try:
            registry = ModelRegistry(project_path or str(project_root()))
            for m in registry.list_models():
                if m.get("sha256") == sha or (
                    m.get("checkpoint_path") and Path(m["checkpoint_path"]) == ckpt
                ):
                    for tag in m.get("tags", []):
                        if isinstance(tag, str) and tag.startswith("experiment:"):
                            exp = tag.split(":", 1)[1]
                            break
                    if exp:
                        break
        except Exception:
            exp = None
    return {"checkpoint": ckpt.stem, "sha256": sha, "experiment_id": exp}


def read_checkpoint_data_config(checkpoint_path: str | Path) -> dict:
    """The checkpoint's own stamped ``config["data"]`` (tiling/subject/attribute/id_map/...).

    Read via ``weights_only=True`` (the same safe partial read :func:`resolve_model_identity` uses
    for ``experiment_id``), without a full model rebuild: a light config sniff for a caller that
    only needs the training-run facts already stamped on the checkpoint, not its weights. ``{}`` for
    a checkpoint with no ``config`` key, or one that fails even this safe read (logged, not raised):
    a foreign/raw checkpoint legitimately carries neither.
    """
    ckpt = Path(checkpoint_path)
    if not ckpt.is_file():
        return {}
    try:
        import torch  # local checkpoint the caller is deliberately reading

        payload = torch.load(ckpt, map_location="cpu", weights_only=True)
        if isinstance(payload, dict):
            data_cfg = (payload.get("config") or {}).get("data")
            if isinstance(data_cfg, dict):
                return data_cfg
    except Exception as exc:
        logger.warning(
            "could not read config[\"data\"] from checkpoint %s (%s); treating it as carrying "
            "none, the same honest fallback a genuinely-foreign checkpoint gets.", ckpt, exc,
        )
    return {}


class ModelRegistry:
    """Simple file-based model registry in .tcip/models/."""

    def __init__(self, project_path: str) -> None:
        self._project_path = project_path
        self._key = registry_index_key(project_path)
        self.root = registry_index_path(project_path).parent
        self.root.mkdir(parents=True, exist_ok=True)
        self._index: list[dict] = self._load_index()

    def _load_index(self) -> list[dict]:
        return read_registry_index(self._project_path)

    def register_model(
        self,
        name: str,
        checkpoint_path: str,
        config: dict,
        metrics: dict | None = None,
        tags: list[str] | None = None,
        kind: str | None = None,
    ) -> dict:
        """Register a trained model with SHA-256 integrity checksum.

        Args:
            name: Model name (e.g. '<crop>_<trait>_detector_v1').
            checkpoint_path: Path to the .pt checkpoint file.
            config: Training config dict.
            metrics: Evaluation metrics dict.
            tags: Optional tags for filtering.
            kind: Model kind (``tcip_module``; open to a future foreign kind) so the GUI + agent
                know how to run it; ``build_predictor`` can still sniff it at inference time.

        Raises:
            FileNotFoundError: ``checkpoint_path`` does not exist, refuses to register a
                phantom deliverable rather than silently storing a null-checksum entry.
            TypeError / ValueError: ``config`` or ``metrics`` holds something JSON cannot
                carry. Both arrive from a caller (an agent's own dict, or a checkpoint's
                stamped metrics), so the offending field is named before anything is stored.
        """
        check_json_value(config, path="config")
        check_json_value(metrics or {}, path="metrics")
        ckpt = Path(checkpoint_path)
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"register_model: checkpoint_path {checkpoint_path!r} does not exist, "
                "refusing to register a phantom registry entry."
            )
        sha256 = _compute_sha256(ckpt)
        file_size = ckpt.stat().st_size

        entry = {
            "name": name,
            "checkpoint_path": checkpoint_path,
            "kind": kind,
            "sha256": sha256,
            "file_size_bytes": file_size,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "metrics": metrics or {},
            "tags": tags or [],
        }
        # Lock-guarded read-modify-write: re-read the stored index under the lock so a
        # concurrent writer's entries aren't clobbered, then replace by name.
        with tcip_store.transaction(self._key) as txn:
            index = txn.read(self._key, default=[])
            superseded = next((e for e in index if e["name"] == name), None)
            index = [e for e in index if e["name"] != name]
            index.append(entry)
            txn.write(self._key, index)
        self._index = index
        # A replace-by-name that actually changes content (a resumed or re-registered run under the
        # same name) needs a record, or the prior sha256/config/metrics are destroyed silently.
        # Purely additive: nothing here changes what get_model/best_model/
        # list_registered_models return, only whether the supersession is durably auditable.
        if superseded is not None and superseded.get("sha256") != sha256:
            from tcip_mcp.audit import record_event

            record_event("model_registry_replace", {
                "name": name, "superseded_sha256": superseded.get("sha256"),
                "superseded_tags": superseded.get("tags"), "new_sha256": sha256,
            })
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
            "error": None if actual_hash == stored_hash else "Checksum mismatch, file may be corrupted or modified",
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

        Only models that actually carry the metric are considered, a missing metric is
        skipped, not scored as a sentinel, so ``None`` cleanly means "no model has it". A
        value that is not a finite number is skipped for the same reason: it compares false
        against every candidate, so an incumbent holding one could never be displaced.
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
            if not math.isfinite(val):
                continue
            if best_val is None or (val > best_val if higher_is_better else val < best_val):
                best_val = float(val)
                best = m
        return best
