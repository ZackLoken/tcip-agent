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

    The producer is the entry's own ``experiment_id`` field, written only by a run's own
    completion through the experiment-mode binding (:func:`register_entry`) and ``None`` for an
    explicit-mode entry: a verified fact, not a caller assertion (``tags`` carries no producer
    claim any more). Every matched entry must carry the key at all, or the load refuses by name,
    as ``best_model`` refuses a pre-``metrics_source`` entry: conform the registry with
    ``scripts/conform_registry_experiment_id.py`` first. An entry naming ``None`` is ignored (not
    a vote for ``None``); every entry that does name a producer must name the same one, or the
    load refuses rather than guess a first match.
    """
    missing = [str(e.get("name")) for e in entries if "experiment_id" not in e]
    if missing:
        raise UnregisteredCheckpoint(
            f"{checkpoint_path} (sha256 {digest}) is named by registry entries {sorted(missing)!r} "
            "that carry no experiment_id key (they predate the producer-binding field): conform "
            "this registry with scripts/conform_registry_experiment_id.py before this checkpoint "
            "can be loaded."
        )
    producers = {e["experiment_id"] for e in entries if e["experiment_id"] is not None}
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
    ``producer`` (the entries that matched its digest, agreeing on their own ``experiment_id``
    field, the binding a run's completion writes). A raw/foreign but registered checkpoint
    legitimately has none of the three, and the identity records the sha with ``experiment_id``
    left ``None`` rather than failing.
    """
    exp = experiment_id
    if exp is None:
        stamped = checkpoint.payload.get("experiment_id")
        if isinstance(stamped, str) and stamped:
            exp = stamped
    if exp is None:
        exp = checkpoint.producer
    return {"checkpoint": Path(checkpoint.path).stem, "sha256": checkpoint.sha256, "experiment_id": exp}


class EntryOwnedByRun(ValueError):
    """A registry entry a run's completion bound cannot be superseded by anything but that run."""


def _refuse_if_owned_by_another_run(superseded: dict, *, name: str, new_experiment_id: str | None) -> None:
    """The eviction rail: a name a run bound is that run's for good.

    Reads ``experiment_id`` directly, never through ``.get()``, so a pre-field entry (predating
    the producer-binding field) is not silently evictable: the key's absence itself refuses,
    naming the conform script, exactly as a present-but-different owner does.
    """
    try:
        owner = superseded["experiment_id"]
    except KeyError:
        raise EntryOwnedByRun(
            f"registry entry {name!r} carries no experiment_id key (it predates the "
            "producer-binding field): conform this registry with "
            "scripts/conform_registry_experiment_id.py before replacing it."
        ) from None
    if owner is not None and owner != new_experiment_id:
        raise EntryOwnedByRun(
            f"registry entry {name!r} is bound to experiment {owner!r} (its recorded producer); "
            "only that run may replace it. To give its weights a second name, register them "
            "again in experiment mode with a new name and the recorded bytes."
        )


def _write_registry_entry(txn: tcip_store.Txn, key: Key, entry: dict) -> dict | None:
    """Replace-by-name inside ``txn``'s already-open transaction over the index key.

    Returns the superseded entry (``None`` for a first registration under this name), for the
    caller to audit once the transaction has closed. Refuses (:class:`EntryOwnedByRun`) a replace
    whose superseded entry names a run other than this write's own.
    """
    index = txn.read(key, default=[])
    superseded = next((e for e in index if e["name"] == entry["name"]), None)
    if superseded is not None:
        _refuse_if_owned_by_another_run(
            superseded, name=entry["name"], new_experiment_id=entry.get("experiment_id"))
    index = [e for e in index if e["name"] != entry["name"]]
    index.append(entry)
    txn.write(key, index)
    return superseded


def _audit_entry_replace(name: str, superseded: dict | None, entry: dict) -> None:
    """Emit ``model_registry_replace`` once the transaction that changed ``name`` has closed, and
    only when the replace actually changed content (not an idempotent same-digest re-registration).
    """
    if superseded is None or superseded.get("sha256") == entry["sha256"]:
        return
    from tcip_mcp.audit import record_event

    record_event("model_registry_replace", {
        "name": name, "superseded_sha256": superseded.get("sha256"),
        "superseded_tags": superseded.get("tags"),
        "superseded_experiment_id": superseded.get("experiment_id"),
        "new_sha256": entry["sha256"],
    })


def _audit_refused(op: str, detail: dict) -> None:
    from tcip_mcp.audit import record_event_or_raise

    record_event_or_raise("model_registry_write_refused", {"op": op, **detail}, status="refused")


def register_entry(
    project_path: str,
    *,
    name: str,
    checkpoint_path: str,
    config: dict,
    metrics: dict | None,
    tags: list[str] | None,
    kind: str | None,
    metrics_source: str | None,
    experiment_id: str | None,
    sha256: str | None = None,
) -> dict:
    """The one write behind both registration modes: replace-by-name inside one transaction over
    the project's index key, then audit the replacement (or the refusal) once it has closed.

    ``sha256``, when given (experiment mode: the digest a run's completion already recorded, and
    the caller's own file already verified against it), is written as-is, no second hash of the
    path; ``None`` (explicit mode) hashes ``checkpoint_path`` here, the one production hash
    explicit-mode registration takes. ``experiment_id`` is the entry's own producer-binding field:
    the run that bound it in experiment mode, ``None`` in explicit mode.

    Raises ``FileNotFoundError`` for a missing ``checkpoint_path`` (explicit mode only; experiment
    mode's caller has already confirmed the file), ``ValueError``/``TypeError`` for a
    ``config``/``metrics`` JSON cannot hold or a ``metrics_source`` pairing that disagrees with
    whether ``metrics`` is empty, and :class:`EntryOwnedByRun` for a replace the eviction rail
    refuses.
    """
    check_json_value(config, path="config")
    check_json_value(metrics or {}, path="metrics")
    has_metrics = bool(metrics)
    if has_metrics and metrics_source is None:
        raise ValueError(
            "register_entry: metrics is non-empty but metrics_source is None; name the path "
            "that produced these numbers ('trainer', 'training_source', or 'caller')."
        )
    if not has_metrics and metrics_source is not None:
        raise ValueError(
            f"register_entry: metrics_source={metrics_source!r} but metrics is empty; a "
            "source with nothing to source is not a real pairing."
        )
    ckpt = Path(checkpoint_path)
    file_size: int | None
    if sha256 is None:
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"register_model: checkpoint_path {checkpoint_path!r} does not exist, "
                "refusing to register a phantom registry entry."
            )
        sha256 = _compute_sha256(ckpt)
        file_size = ckpt.stat().st_size
    else:
        file_size = ckpt.stat().st_size if ckpt.is_file() else None

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
        "experiment_id": experiment_id,
    }
    key = registry_index_key(project_path)
    try:
        with tcip_store.transaction(key) as txn:
            superseded = _write_registry_entry(txn, key, entry)
    except EntryOwnedByRun as exc:
        _audit_refused("register_entry", {
            "name": name, "attempted_experiment_id": experiment_id, "reason": str(exc),
        })
        raise
    _audit_entry_replace(name, superseded, entry)
    return entry


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
        """Register a trained model with SHA-256 integrity checksum, in explicit mode.

        Explicit mode's own entry point: wraps :func:`register_entry` with ``experiment_id=None``
        (no run bound this weights file, whatever ``tags`` the caller passes) and no ``sha256``
        (this call's own hash of ``checkpoint_path`` is the only production hash explicit mode
        takes). Experiment mode (``register_model_from_experiment``) calls :func:`register_entry`
        directly, with the digest completion already recorded in hand.

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
            EntryOwnedByRun: ``name`` already names an entry a run's completion bound to a
                different (or, for a pre-field entry, an unrecorded) run.
        """
        entry = register_entry(
            self._project_path, name=name, checkpoint_path=checkpoint_path, config=config,
            metrics=metrics, tags=tags, kind=kind, metrics_source=metrics_source,
            experiment_id=None,
        )
        self._index = self._load_index()
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
        treated as just another unverified entry: re-register it through ``register_model`` (its
        checkpoint on disk is unchanged; the call re-hashes and rewrites the entry with the field
        present). A present ``metrics_source`` of ``None`` is not malformed, it is the honest
        pairing for an entry with no metrics.
        """
        malformed = [m.get("name") for m in self._index if "metrics_source" not in m]
        if malformed:
            raise ValueError(
                f"registry entries {malformed} carry no metrics_source key (they predate the "
                "field); re-register each through register_model before ranking this registry."
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
