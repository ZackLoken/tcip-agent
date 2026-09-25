"""Model registry, track trained models and their performance.

The index document is ``{entries: [...]}``. A stored ``checkpoint_path`` is relative POSIX exactly
when the checkpoint lives under the registry's own scope root, absolute exactly when it does not.
Every response surface answers the resolved absolute path on a copy.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import tcip_store
from tcip_store import (
    RECORD_JSON,
    Key,
    SchemaVersionRefused,
    StoreDescriptor,
    check_json_value,
    check_schema_version,
    get_descriptor,
    register_store,
)
from tcip_store.file_backend import RootedFileLocator

from tcip_mcp.registry_paths import (
    RegistryPathEmpty,
    RegistryPathTraversal,
    checkpoint_registry_path_for,
    is_at_or_under,
    is_external_form,
    resolved_registry_path,
)

logger = logging.getLogger(__name__)

# ── the registry store ───────────────────────────────────────────────────────

_INDEX_DOC = RootedFileLocator(prefix=(".tcip", "models"), suffix=".json")
"""The registry index, one per project."""

MODEL_REGISTRY_STORE = "model_registry"
_INDEX_PARTS = ("registry",)
REGISTRY_SCHEMA_VERSION = 1
"""The store's registered version, per ``frozen-formats.json``; never written into a document."""
register_store(
    StoreDescriptor(
        name=MODEL_REGISTRY_STORE,
        kind="record",
        key_fields=("document",),
        frozen=True,
        schema_version=REGISTRY_SCHEMA_VERSION,
        codec=RECORD_JSON,
        concurrency="cas",
        locator=_INDEX_DOC,
    )
)


class RegistryVersionRefused(ValueError):
    """The registry index document is not the ``{entries: [...]}`` mapping the writer writes."""


def _read_registry_document(raw: object) -> dict:
    """The registry's entries-mapping document from what the store handed back.

    ``None`` (first use) answers the empty document. Anything but the mapping
    :func:`_write_registry_document` writes raises :class:`RegistryVersionRefused`.
    """
    if raw is None:
        return {"entries": []}
    if not isinstance(raw, dict) or set(raw) != {"entries"} or not isinstance(raw["entries"], list):
        raise RegistryVersionRefused(
            f"the model registry index is not a recognized entries-mapping document: {raw!r}"
        )
    return raw


def _write_registry_document(entries: list[dict]) -> dict:
    """The document a write puts on disk for ``entries``."""
    return {"entries": entries}


def registry_index_key(project_path: str | Path) -> Key:
    """The project's registered-model index, written compare-and-set."""
    return Key(MODEL_REGISTRY_STORE, str(project_path), _INDEX_PARTS)


def registry_index_path(project_path: str | Path) -> Path:
    """Where the project's registry index lives on disk."""
    return Path(project_path, *_INDEX_DOC.relative_path(str(project_path), _INDEX_PARTS).parts)


def read_registry_index(project_path: str | Path) -> list[dict]:
    """Every model entry the project has registered, in registration order.

    A project that has registered nothing reads as an empty list; an index that does not decode
    raises ``DecodeError``, and one that is not the entries mapping raises
    :class:`RegistryVersionRefused`.
    """
    raw = tcip_store.read(registry_index_key(project_path), default=None)
    return _read_registry_document(raw)["entries"]


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
        "second checkpoint of the same run (model_final beside a model_best, or a bespoke tag) "
        "is registered in explicit mode under a distinct name, since experiment mode names "
        "the entry after the run and replaces by name)."
    )


def _resolve_producer(entries: tuple[dict, ...], *, checkpoint_path: Path, digest: str) -> str | None:
    """The one producer ``entries`` (already matched by ``sha256``) agree on.

    The producer is each entry's ``experiment_id``; an entry naming ``None`` casts no vote.
    Entries naming different producers raise :class:`UnregisteredCheckpoint`.
    """
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

    @property
    def task(self) -> str:
        """The task the checkpoint's model is for, :func:`run_task` over its own config with its
        ``model_source`` in place; raises ``ValueError`` when neither states one."""
        from tcip_mcp.pipelines.model_build import MODEL_SOURCE_KEY, run_task

        return run_task({**(self.payload.get("config") or {}),
                         MODEL_SOURCE_KEY: self.payload.get(MODEL_SOURCE_KEY) or {}})


class UnregisteredCheckpoint(ValueError):
    """A checkpoint the registry names no entry for, whose registered entries disagree on
    producer, whose payload cannot be trusted to unpickle under ``weights_only=True``, or whose
    payload carries a ``schema_version`` this reader does not accept: whatever identity check
    already passed (a registry-name match, or a completion's own recorded digest), none of these
    is a payload this reader can act on."""


def _load_verified_payload(data: bytes, *, source: str) -> dict:
    """Unpickle already identity-verified checkpoint bytes, refusing a payload this reader cannot
    act on.

    Unpickles with ``weights_only=True`` and checks the ``schema_version`` ceiling. ``source``
    names the checkpoint (path and digest) in every raised message.

    Raises :class:`UnregisteredCheckpoint` for a payload that will not unpickle under
    ``weights_only=True``, or that carries a ``schema_version`` this reader does not accept. Raises
    a bare ``ValueError`` (:func:`~tcip_mcp.pipelines.inference.predictor._require_dict_payload`)
    for a payload that does not unpickle to a dict.
    """
    import torch  # local checkpoint an identity check already named; unpickling it is the point

    from tcip_mcp.pipelines.inference.predictor import _require_dict_payload

    try:
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise UnregisteredCheckpoint(
            f"{source} could not be loaded with weights_only=True ({exc}): registration "
            "verifies identity, not payload shape, and this payload carries something outside "
            "a platform-written deliverable checkpoint's contract (a resume checkpoint's "
            "RNG/optimizer state is the trainer's own resume path to read; a bespoke loop's own "
            "arbitrary state is outside the contract)."
        ) from exc
    payload = _require_dict_payload(payload, source)
    from tcip_mcp.pipelines.training.generic_trainer import RUN_CHECKPOINT_STORE

    try:
        check_schema_version(get_descriptor(RUN_CHECKPOINT_STORE), payload)
    except SchemaVersionRefused as exc:
        raise UnregisteredCheckpoint(f"{source}: {exc}") from exc
    return payload


def load_registered_checkpoint(
    checkpoint_path: str | Path, *, project_path: str | None = None,
) -> VerifiedCheckpoint:
    """Read a checkpoint's bytes once, hash them, and refuse unless the registry names that hash.

    Refuses two forgeries: a checkpoint dropped at any path with no registry entry, and a
    checkpoint whose file is replaced (in place or by rename) between a caller checking its
    identity and a caller loading its weights. In order: the file is read into one ``bytes``
    object; the digest is taken over that exact object through :func:`_sha256_of_bytes`; the
    registry index of ``project_path`` (or, unset,
    :func:`~tcip_mcp.project_paths.platform_state_root`) is read and every entry whose ``sha256``
    equals the digest is collected; none of them raises :class:`UnregisteredCheckpoint`, naming the
    path, the digest, the root searched, and the remedy. Several entries naming one digest must
    agree on producer or the load refuses (see :func:`_resolve_producer`). Only once the registry
    has answered is the payload unpickled and version-checked, through
    :func:`_load_verified_payload`; a payload that will not unpickle (a periodic resume
    checkpoint's RNG/optimizer state, or a bespoke loop's own arbitrary object), or that carries a
    ``schema_version`` this reader does not accept, refuses naming the reason. A payload that does
    not unpickle to a dict raises ``ValueError``. A missing file raises ``FileNotFoundError``
    before any read.

    The digest and the load are over one immutable byte string.
    """
    from tcip_mcp.project_paths import platform_state_root

    ckpt = Path(checkpoint_path)
    root = project_path or str(platform_state_root())
    with open(ckpt, "rb") as f:
        data = f.read()
    digest = _sha256_of_bytes(data)
    index = read_registry_index(root)
    entries = tuple(e for e in index if e["sha256"] == digest)
    if not entries:
        raise _unregistered_checkpoint_error(ckpt, digest, root)
    producer = _resolve_producer(entries, checkpoint_path=ckpt, digest=digest)
    payload = _load_verified_payload(data, source=f"{ckpt} (sha256 {digest})")
    return VerifiedCheckpoint(
        path=str(checkpoint_path), sha256=digest, payload=payload, entries=entries,
        producer=producer,
    )


def resolve_model_identity(checkpoint: VerifiedCheckpoint, *, experiment_id: str | None = None) -> dict:
    """Producing-model identity for a verified checkpoint: ``{checkpoint, sha256, experiment_id}``.

    ``sha256`` and the ``experiment_id`` this returns both come off ``checkpoint``, already loaded
    and matched against the registry by :func:`load_registered_checkpoint`; this function reads no
    file and hashes nothing itself. ``experiment_id`` resolves, in order: the caller's explicit
    value; the checkpoint payload's own stamped ``experiment_id``; then the checkpoint's own
    resolved ``producer``. A registered checkpoint with none of the three records the sha with
    ``experiment_id`` left ``None``.
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
    """Raise :class:`EntryOwnedByRun` when ``superseded`` names a producer other than
    ``new_experiment_id``."""
    owner = superseded["experiment_id"]
    if owner is not None and owner != new_experiment_id:
        raise EntryOwnedByRun(
            f"registry entry {name!r} is bound to experiment {owner!r} (its recorded producer); "
            "only that run may replace it. To give its weights a second name, register them "
            "again in experiment mode with a new name and the recorded bytes."
        )


def _write_registry_entry(txn: tcip_store.Txn, key: Key,
                          entry: dict) -> tuple[dict | None, dict]:
    """Replace-by-name inside ``txn``'s already-open transaction over the index key.

    Returns ``(superseded, stored)``: the entry replaced (``None`` for a first registration under
    this name), and the entry the registry holds after the call, ``entry`` itself when it was
    written and the superseded entry when that already holds every field but ``registered_at``,
    which is no change and so no write. Refuses (:class:`EntryOwnedByRun`) a replace whose
    superseded entry names a run other than this write's own.
    """
    index = _read_registry_document(txn.read(key, default=None))["entries"]
    superseded = next((e for e in index if e["name"] == entry["name"]), None)
    if superseded is not None:
        _refuse_if_owned_by_another_run(
            superseded, name=entry["name"], new_experiment_id=entry["experiment_id"])
        if ({k: v for k, v in superseded.items() if k != "registered_at"}
                == {k: v for k, v in entry.items() if k != "registered_at"}):
            return superseded, superseded
    index = [e for e in index if e["name"] != entry["name"]]
    index.append(entry)
    txn.write(key, _write_registry_document(index))
    return superseded, entry


def _audit_entry_write(name: str, superseded: dict | None, entry: dict) -> None:
    """Emit ``model_registered`` once the transaction that wrote ``name`` has closed, naming the
    entry it superseded. Called only for a write that happened.

    A failed append raises ``AuditEntryNotWritten``.
    """
    from tcip_mcp.audit import record_event_or_raise

    replaced = {} if superseded is None else {
        "superseded_sha256": superseded["sha256"], "superseded_tags": superseded["tags"],
        "superseded_experiment_id": superseded["experiment_id"],
    }
    record_event_or_raise("model_registered", {
        "name": name, "new_sha256": entry["sha256"], "experiment_id": entry["experiment_id"],
        **replaced,
    })


def _register_entry(
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
    """The write behind both registration modes: replace-by-name inside one transaction over the
    project's index key, then audit the write once it has closed.

    ``sha256``, when given, is written as-is, no second hash of the path; ``None`` hashes
    ``checkpoint_path`` here. ``experiment_id`` is the entry's own producer-binding field: the run
    that bound it in experiment mode, ``None`` in explicit mode.

    The stored ``checkpoint_path`` is spelled through
    :func:`~tcip_mcp.registry_paths.checkpoint_registry_path_for` against this project's own root:
    relative POSIX when the checkpoint resolves under it, absolute when it does not. A checkpoint
    outside the project (``sha256`` given but the file gone by the time this runs) keeps the
    caller's own string verbatim.

    Raises ``FileNotFoundError`` for a missing ``checkpoint_path`` when ``sha256`` is not given,
    ``ValueError``/``TypeError`` for a ``config``/``metrics`` JSON cannot hold or a
    ``metrics_source`` pairing that disagrees with whether ``metrics`` is empty,
    :class:`EntryOwnedByRun` for a replace the eviction rail refuses, and ``AuditEntryNotWritten``
    after a committed write, when ``_audit_entry_write`` cannot append its own line.
    """
    check_json_value(config, path="config")
    check_json_value(metrics or {}, path="metrics")
    has_metrics = bool(metrics)
    if has_metrics and metrics_source is None:
        raise ValueError(
            "_register_entry: metrics is non-empty but metrics_source is None; name the path "
            "that produced these numbers ('trainer', 'training_source', or 'caller')."
        )
    if not has_metrics and metrics_source is not None:
        raise ValueError(
            f"_register_entry: metrics_source={metrics_source!r} but metrics is empty; a "
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

    stored_checkpoint_path = (
        checkpoint_registry_path_for(ckpt, project_path) if ckpt.is_file() else checkpoint_path
    )

    entry = {
        "name": name,
        "checkpoint_path": stored_checkpoint_path,
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
    with tcip_store.transaction(key) as txn:
        superseded, stored = _write_registry_entry(txn, key, entry)
    if stored is entry:  # only a write that happened leaves a line
        _audit_entry_write(name, superseded, entry)
    return stored


def _candidate_checkpoint_paths(root: Path, raw: str) -> set[Path]:
    """Every file under ``root`` worth hashing while relocating one entry's checkpoint: the
    ``.tcip/models`` tree recursively, the ``.tcip/experiments`` tree's checkpoint files, and any
    file under ``root`` the entry's own relative suffix (everything from its last ``.tcip`` segment
    onward) or basename names.
    """
    found: set[Path] = set()
    models_dir = root / ".tcip" / "models"
    if models_dir.is_dir():
        found.update(p for p in models_dir.rglob("*") if p.is_file())
    experiments_dir = root / ".tcip" / "experiments"
    if experiments_dir.is_dir():
        found.update(p for p in experiments_dir.rglob("*.pt") if p.is_file())
    raw_parts = PurePosixPath(Path(raw).as_posix()).parts
    if ".tcip" in raw_parts:
        last_tcip = len(raw_parts) - 1 - raw_parts[::-1].index(".tcip")
        suffix = Path(root, *raw_parts[last_tcip:])
        if suffix.is_file():
            found.add(suffix)
    basename = Path(raw).name
    if basename:
        found.update(p for p in root.rglob(basename) if p.is_file())
    return {p.resolve() for p in found}


def _cached_sha256(hash_cache: dict[Path, str], path: Path) -> str:
    """``hash_cache``'s own digest for ``path``, computed once per run."""
    if path not in hash_cache:
        hash_cache[path] = _compute_sha256(path)
    return hash_cache[path]


def _pick_duplicate(candidates: list[Path], *, root: Path, original_basename: str) -> Path:
    """The deterministic choice among more than one byte-identical relocation candidate: a
    basename match to the entry's own original spelling first, then the sorted
    project-relative path."""
    basename_matches = [c for c in candidates if c.name == original_basename]
    pool = basename_matches or candidates
    return sorted(pool, key=lambda c: c.relative_to(root).as_posix())[0]


def _conform_entries(
    entries: list[dict], root: Path, *, hash_cache: dict[Path, str],
) -> tuple[list[dict], list[str]]:
    """Respell every entry's ``checkpoint_path`` relative to ``root``, per
    :func:`conform_registry_paths_on_disk`'s own rule. Returns the conformed entries and one
    outcome line per entry examined."""
    conformed: list[dict] = []
    lines: list[str] = []
    for entry in entries:
        raw = entry["checkpoint_path"]
        expected_sha = entry["sha256"]
        direct = Path(raw) if is_external_form(raw) else root.joinpath(*PurePosixPath(raw).parts)
        direct_resolved = direct.resolve()
        if (
            is_at_or_under(direct_resolved, root)
            and direct_resolved.is_file()
            and _cached_sha256(hash_cache, direct_resolved) == expected_sha
        ):
            respelled = checkpoint_registry_path_for(direct_resolved, root)
            if respelled != raw:
                lines.append(f"{entry['name']}: respelled {raw!r} to {respelled!r}")
                conformed.append({**entry, "checkpoint_path": respelled})
            else:
                conformed.append(entry)
            continue

        matches = []
        for candidate in sorted(_candidate_checkpoint_paths(root, raw)):
            size = entry["file_size_bytes"]
            if size is not None and candidate.stat().st_size != size:
                continue
            if _cached_sha256(hash_cache, candidate) == expected_sha:
                matches.append(candidate)

        if len(matches) == 1:
            respelled = checkpoint_registry_path_for(matches[0], root)
            lines.append(f"{entry['name']}: respelled {raw!r} to {respelled!r} (relocated)")
            conformed.append({**entry, "checkpoint_path": respelled})
        elif len(matches) > 1:
            chosen = _pick_duplicate(matches, root=root, original_basename=Path(raw).name)
            respelled = checkpoint_registry_path_for(chosen, root)
            named = sorted(m.relative_to(root).as_posix() for m in matches)
            lines.append(f"{entry['name']}: respelled {raw!r} to {respelled!r}, ambiguous "
                         f"among {len(matches)} byte-identical candidates {named}, picked by "
                         "basename match then sorted path")
            conformed.append({**entry, "checkpoint_path": respelled})
        elif is_external_form(raw):
            # The stored value is itself the external claim; a host-resolved respelling of it
            # fabricates a path this machine derived, not one the writer stated.
            lines.append(f"{entry['name']}: {raw!r} kept as stored "
                         f"(external-or-missing, exists={direct_resolved.is_file()})")
            conformed.append(entry)
        else:
            # A relative entry keeps its spelling: writing an absolute path here would fabricate a designed-external claim.
            lines.append(f"{entry['name']}: {raw!r} stays unresolved, no matching digest "
                         f"found under root (exists={direct_resolved.is_file()})")
            conformed.append(entry)
    return conformed, lines


def conform_registry_paths_on_disk(root: str | Path) -> list[str]:
    """Respell every entry's ``checkpoint_path`` in ``root``'s registry index relative to ``root``,
    reading and writing the index file directly rather than through the storage seam, for a
    loose-file tree the caller holds exclusively.

    Per entry: a stored path that resolves under ``root`` with a matching ``sha256`` is respelled
    through the checkpoint speller. A path that does not (outside ``root``, missing, or a hash
    mismatch) is relocated: every file under ``.tcip/models``, every checkpoint-shaped file under
    ``.tcip/experiments``, and any file the entry's own stored suffix or basename names
    (:func:`_candidate_checkpoint_paths`), prefiltered by the entry's recorded ``file_size_bytes``,
    is hashed once (each hash cached across every entry this run examines) and matched against the
    entry's own ``sha256``. Exactly one match respells to it; more than one (byte-identical files)
    picks deterministically (:func:`_pick_duplicate`) and is disclosed as ambiguous; none leaves an
    already-external entry absolute, classified external-or-missing by existence in the outcome
    line, and leaves an already-relative entry's spelling untouched.

    Idempotent. Raises :class:`RegistryVersionRefused` for a document that is not the entries
    mapping.
    """
    root_path = Path(root).resolve()
    path = registry_index_path(root_path)
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise RegistryVersionRefused(f"{path} will not decode as JSON: {exc}") from exc
    entries = _read_registry_document(raw)["entries"]
    conformed, lines = _conform_entries(entries, root_path, hash_cache={})
    path.write_bytes(RECORD_JSON.encode(_write_registry_document(conformed)))
    return lines


def _resolve_entry_checkpoint(project_path: str, entry: dict) -> dict:
    """A shallow copy of ``entry`` with ``checkpoint_path`` resolved to an absolute path.

    A stored value this reader cannot resolve (empty, or a relative form carrying ``..``) keeps its
    stored spelling and carries the reason in ``checkpoint_path_error``.
    """
    copy = dict(entry)
    stored = entry["checkpoint_path"]
    try:
        copy["checkpoint_path"] = str(resolved_registry_path(project_path, stored))
    except (RegistryPathEmpty, RegistryPathTraversal) as exc:
        copy["checkpoint_path_error"] = str(exc)
    return copy


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

        Wraps :func:`_register_entry` with ``experiment_id=None`` and no ``sha256``, so this call
        hashes ``checkpoint_path`` itself.

        Args:
            name: Model name (e.g. '<crop>_<trait>_detector_v1').
            checkpoint_path: Path to the .pt checkpoint file.
            config: Training config dict.
            metrics: Evaluation metrics dict.
            tags: Optional tags for filtering.
            kind: Model kind (``tcip_module``; open to a future foreign kind) so the GUI + agent
                know how to run it.
            metrics_source: Which path produced ``metrics``: ``"trainer"`` (the platform's own
                ``default_train``, which measured them), ``"training_source"`` (a bespoke loop's
                own saved state, unverified), ``"caller"`` (an explicit-mode argument, unverified),
                or ``None`` when ``metrics`` is empty. Required, never derived here.

        Raises:
            FileNotFoundError: ``checkpoint_path`` does not exist.
            TypeError / ValueError: ``config`` or ``metrics`` holds something JSON cannot carry,
            named before anything is stored. ``ValueError`` also covers ``metrics_source``
            disagreeing with whether ``metrics`` is empty.
            EntryOwnedByRun: ``name`` already names an entry a run's completion bound to a
                different run.
            AuditEntryNotWritten: the write committed but its own audit line could not be appended.
        """
        entry = _register_entry(
            self._project_path, name=name, checkpoint_path=checkpoint_path, config=config,
            metrics=metrics, tags=tags, kind=kind, metrics_source=metrics_source,
            experiment_id=None,
        )
        self._index = self._load_index()
        return _resolve_entry_checkpoint(self._project_path, entry)

    def verify_model(self, name: str) -> dict:
        """Verify a model checkpoint's integrity against stored checksum.

        Returns:
            dict with 'valid' (bool), 'expected' (str), 'actual' (str|None), 'error' (str|None).
        """
        model = self.get_model(name)
        if model is None:
            return {"valid": False, "error": f"Model '{name}' not found in registry"}

        stored_hash = model["sha256"]
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
        """List registered models, optionally filtered by tag.

        Every returned entry is a copy carrying the resolved absolute ``checkpoint_path``
        (:func:`_resolve_entry_checkpoint`).
        """
        entries = self._index if tag is None else [m for m in self._index if tag in m["tags"]]
        return [_resolve_entry_checkpoint(self._project_path, m) for m in entries]

    def get_model(self, name: str) -> dict | None:
        """Get a model entry by name, with its checkpoint_path resolved absolute."""
        for m in self._index:
            if m["name"] == name:
                return _resolve_entry_checkpoint(self._project_path, m)
        return None

    def best_model(
        self, metric_key: str, *, higher_is_better: bool, include_unverified: bool = False,
        experiment_ids: list[str] | None = None,
    ) -> dict | None:
        """Get the registered model with the best value for ``metric_key``.

        Only models that actually carry the metric as a finite number are considered, so ``None``
        means "no model has it". ``metric_key`` and ``higher_is_better`` are both required. By
        default only an entry whose ``metrics_source`` is ``"trainer"`` is ranked;
        ``include_unverified=True`` also ranks ``"training_source"`` and ``"caller"`` entries.
        ``experiment_ids``, when given, narrows ranking to entries whose own ``experiment_id`` is
        in the set; ``None`` (the default) ranks the whole registry.
        """
        best = None
        best_val: float | None = None
        for m in self._index:
            if experiment_ids is not None and m["experiment_id"] not in experiment_ids:
                continue
            source = m["metrics_source"]
            if source != "trainer" and not include_unverified:
                continue
            val = m["metrics"].get(metric_key)
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                continue
            if not math.isfinite(val):
                continue
            if best_val is None or (val > best_val if higher_is_better else val < best_val):
                best_val = float(val)
                best = m
        return _resolve_entry_checkpoint(self._project_path, best) if best is not None else None
