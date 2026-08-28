"""Model registry, track trained models and their performance."""

from __future__ import annotations

import hashlib
import io
import logging
import math
from dataclasses import dataclass
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


def _sha256_of_bytes(data: bytes) -> str:
    """The one digest function every bytes-to-hash caller in this module shares."""
    return hashlib.sha256(data).hexdigest()


def _compute_sha256(filepath: str | Path) -> str:
    """Compute SHA-256 checksum of a file, reading it once."""
    with open(filepath, "rb") as f:
        return _sha256_of_bytes(f.read())


_PERIODIC_RESUME_PREFIX = "checkpoint_epoch_"
"""The trainer's own periodic-checkpoint naming convention (``generic_trainer.py``): a resume
artifact the trainer's own resume path reads, never a deliverable to register."""


def _unregistered_checkpoint_error(checkpoint_path: Path, digest: str, root: str) -> "UnregisteredCheckpoint":
    if checkpoint_path.stem.startswith(_PERIODIC_RESUME_PREFIX):
        return UnregisteredCheckpoint(
            f"{checkpoint_path} (sha256 {digest}) is not named by any entry in the registry at "
            f"{root!r}: its name marks it a periodic resume checkpoint "
            f"({_PERIODIC_RESUME_PREFIX}*), read only by the trainer's own resume path, not a "
            "deliverable to register."
        )
    return UnregisteredCheckpoint(
        f"{checkpoint_path} (sha256 {digest}) is not named by any entry in the registry at "
        f"{root!r}: register it with register_model under a name of its own (explicit mode; a "
        "completed run registers its own final weights on completion under the run's id, and a "
        "second checkpoint of the same run -- model_final beside a model_best, or a bespoke tag "
        "-- is registered in explicit mode under a distinct name, since experiment mode names "
        "the entry after the run and replaces by name)."
    )


def _resolve_producer(entries: tuple[dict, ...], *, checkpoint_path: Path, digest: str) -> str | None:
    """The one producer ``entries`` (already matched by ``sha256``) agree on.

    Today's producer is the ``experiment:`` tag; ``tags`` is caller-asserted until the model
    registry's own tag-binding family lands, so this is bounded to what the entries assert, not a
    verified fact. An entry naming none is ignored (not a vote for ``None``); every entry that
    does name one must name the same value, or the load refuses rather than guess a first match.
    """
    producers = {
        tag.split(":", 1)[1]
        for e in entries
        for tag in (e.get("tags") or [])
        if isinstance(tag, str) and tag.startswith("experiment:")
    }
    if len(producers) > 1:
        raise UnregisteredCheckpoint(
            f"{checkpoint_path} (sha256 {digest}) is named by registry entries naming different "
            f"producers ({sorted(producers)!r}): the producer a stamp will carry is a required "
            "fact, not a first-match guess. Name the conflicting entries distinctly or supersede "
            "the stale one before this checkpoint can be loaded."
        )
    return next(iter(producers), None)


@dataclass(frozen=True)
class VerifiedCheckpoint:
    """A checkpoint :func:`load_registered_checkpoint` read, hashed and matched against the
    registry before anything in it was unpickled.

    Holding an instance is evidence of verification only because no other constructor exists in
    production: every predictor build and every measurement-path checkpoint load in this platform
    takes this object from :func:`load_registered_checkpoint`, and a test that stubs a load stubs
    that function.
    """

    path: str
    """The path the caller named, as given."""
    sha256: str
    """The digest of the exact bytes ``payload`` was unpickled from."""
    payload: dict
    """The loaded checkpoint, read with ``weights_only=True``."""
    entries: tuple[dict, ...]
    """Every registry entry whose ``sha256`` equals this checkpoint's digest."""
    producer: str | None
    """The one producer ``entries`` agree names, ``None`` when none of them names one."""

    @property
    def data_config(self) -> dict:
        """The checkpoint's own stamped ``config["data"]``, ``{}`` for a checkpoint carrying
        none (a foreign checkpoint's documented answer)."""
        data_cfg = (self.payload.get("config") or {}).get("data")
        return data_cfg if isinstance(data_cfg, dict) else {}


class UnregisteredCheckpoint(ValueError):
    """A checkpoint the registry names no entry for, or whose registered entries disagree on
    producer, or whose payload cannot be trusted to unpickle under ``weights_only=True``."""


def load_registered_checkpoint(
    checkpoint_path: str | Path, *, project_path: str | None = None,
) -> VerifiedCheckpoint:
    """Read a checkpoint's bytes once, hash them, and refuse unless the registry names that hash.

    Closes the family of forgeries option B rules out: a checkpoint dropped at any path with no
    registry entry, and a checkpoint whose file is replaced (in place or by rename) between a
    caller checking its identity and a caller loading its weights. In order: the file is read
    into one ``bytes`` object; the digest is taken over that exact object through
    :func:`_sha256_of_bytes` (the same function :meth:`ModelRegistry.register_model` hashes with,
    so registration and this load are literally the same bytes-to-digest code); the registry
    index of ``project_path`` (or, unset, :func:`~tcip_mcp.project_paths.project_root`) is read
    and every entry whose ``sha256`` equals the digest is collected; none of them raises
    :class:`UnregisteredCheckpoint`, naming the path, the digest, the root searched, and the
    remedy. Several entries naming one digest must agree on producer or the load refuses (see
    :func:`_resolve_producer`). Only once the registry has answered is the payload unpickled,
    with ``weights_only=True``, a narrower sink than the ``weights_only=False`` sniff a predictor
    build used to run: every deliverable checkpoint this platform's own trainer writes (a state
    dict, JSON config, numeric metrics and the three stamps) loads under it; a payload that will
    not (a periodic resume checkpoint's RNG/optimizer state, or a bespoke loop's own arbitrary
    object) refuses naming the reason, since registration verifies identity, never payload shape.
    A payload that does not unpickle to a dict raises the same ``ValueError`` a kind sniff raises
    for the same fact. A missing file raises ``FileNotFoundError`` before any read.

    The digest and the load are over one immutable byte string, so no replacement of the file, in
    place or by rename, on either platform, can separate them.
    """
    from tcip_mcp.pipelines.inference.predictor import _require_dict_payload
    from tcip_mcp.project_paths import project_root

    ckpt = Path(checkpoint_path)
    root = project_path or str(project_root())
    with open(ckpt, "rb") as f:
        data = f.read()
    digest = _sha256_of_bytes(data)
    index = read_registry_index(root)
    entries = tuple(e for e in index if e.get("sha256") == digest)
    if not entries:
        raise _unregistered_checkpoint_error(ckpt, digest, root)
    producer = _resolve_producer(entries, checkpoint_path=ckpt, digest=digest)

    import torch  # local checkpoint the registry already named; unpickling it is the point

    try:
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise UnregisteredCheckpoint(
            f"{ckpt} (sha256 {digest}) is registered but could not be loaded with "
            f"weights_only=True ({exc}): registration verifies identity, not payload shape, and "
            "this payload carries something outside a platform-written deliverable checkpoint's "
            "contract (a resume checkpoint's RNG/optimizer state is the trainer's own resume "
            "path to read; a bespoke loop's own arbitrary state is outside the contract)."
        ) from exc
    payload = _require_dict_payload(payload, str(ckpt))
    return VerifiedCheckpoint(
        path=str(checkpoint_path), sha256=digest, payload=payload, entries=entries,
        producer=producer,
    )


def resolve_model_identity(checkpoint: VerifiedCheckpoint, *, experiment_id: str | None = None) -> dict:
    """Producing-model identity for a verified checkpoint: ``{checkpoint, sha256, experiment_id}``.

    ``sha256`` and the ``experiment_id`` this returns both come off ``checkpoint``, already loaded
    and matched against the registry by :func:`load_registered_checkpoint`; this function reads no
    file and hashes nothing itself. ``experiment_id`` resolves, in order: the caller's explicit
    value; the checkpoint payload's own stamped ``experiment_id`` (every checkpoint saved through
    the audited envelope carries one via ``stamp_model_ref``); then the checkpoint's own resolved
    ``producer`` (the registry's ``experiment:`` tag on the entries that matched its digest). A
    raw/foreign but registered checkpoint legitimately has none of the three, and the identity
    records the sha with ``experiment_id`` left ``None`` rather than failing.
    """
    exp = experiment_id
    if exp is None:
        stamped = checkpoint.payload.get("experiment_id")
        if isinstance(stamped, str) and stamped:
            exp = stamped
    if exp is None:
        exp = checkpoint.producer
    return {"checkpoint": Path(checkpoint.path).stem, "sha256": checkpoint.sha256, "experiment_id": exp}


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
        *,
        metrics_source: str | None,
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
            metrics_source: Which path produced ``metrics``: ``"trainer"`` (the platform's own
                ``default_train``, which measured them), ``"training_source"`` (a bespoke loop's
                own saved state, unverified), ``"caller"`` (an explicit-mode argument,
                unverified), or ``None`` when ``metrics`` is empty. Required, never derived here:
                the two production callers (``register_model_from_experiment`` and the
                ``register_model`` tool) compute it from which path actually produced the run.

        Raises:
            FileNotFoundError: ``checkpoint_path`` does not exist, refuses to register a
                phantom deliverable rather than silently storing a null-checksum entry.
            TypeError / ValueError: ``config`` or ``metrics`` holds something JSON cannot
                carry. Both arrive from a caller (an agent's own dict, or a checkpoint's
                stamped metrics), so the offending field is named before anything is stored.
                ``ValueError`` also covers ``metrics_source`` disagreeing with whether
                ``metrics`` is empty.
        """
        check_json_value(config, path="config")
        check_json_value(metrics or {}, path="metrics")
        has_metrics = bool(metrics)
        if has_metrics and metrics_source is None:
            raise ValueError(
                "register_model: metrics is non-empty but metrics_source is None; name the path "
                "that produced these numbers ('trainer', 'training_source', or 'caller')."
            )
        if not has_metrics and metrics_source is not None:
            raise ValueError(
                f"register_model: metrics_source={metrics_source!r} but metrics is empty; a "
                "source with nothing to source is not a real pairing."
            )
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
            "metrics_source": metrics_source,
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
        self, metric_key: str, *, higher_is_better: bool, include_unverified: bool = False,
    ) -> dict | None:
        """Get the registered model with the best value for ``metric_key``.

        Only models that actually carry the metric are considered, a missing metric is
        skipped, not scored as a sentinel, so ``None`` cleanly means "no model has it". A
        value that is not a finite number is skipped for the same reason: it compares false
        against every candidate, so an incumbent holding one could never be displaced.
        ``metric_key`` and ``higher_is_better`` are both required: this reader guesses neither
        the metric nor its direction from the key's spelling. By default only an entry whose
        ``metrics_source`` is ``"trainer"``, the one path the platform itself measured, is
        ranked; ``include_unverified=True`` also ranks ``"training_source"`` and ``"caller"``
        entries, whose numbers nothing here verified. An entry carrying no ``metrics_source`` key
        at all is a malformed record predating the field, refused by name rather than silently
        treated as just another unverified entry: conform the registry first with
        ``scripts/conform_registry_metrics_source.py``. A present ``metrics_source`` of ``None``
        is not malformed, it is the honest pairing for an entry with no metrics.
        """
        malformed = [m.get("name") for m in self._index if "metrics_source" not in m]
        if malformed:
            raise ValueError(
                f"registry entries {malformed} carry no metrics_source key (they predate the "
                "field); conform this registry with scripts/conform_registry_metrics_source.py "
                "before ranking it."
            )
        best = None
        best_val: float | None = None
        for m in self._index:
            source = m["metrics_source"]
            if source != "trainer" and not include_unverified:
                continue
            val = m.get("metrics", {}).get(metric_key)
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                continue
            if not math.isfinite(val):
                continue
            if best_val is None or (val > best_val if higher_is_better else val < best_val):
                best_val = float(val)
                best = m
        return best
