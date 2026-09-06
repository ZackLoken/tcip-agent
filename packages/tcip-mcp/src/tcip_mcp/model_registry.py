"""Model registry, track trained models and their performance.

The index document is ``{entries: [...]}``, no ``schema_version`` field until this store's first
bump (absence is the frozen version 1): a stored ``checkpoint_path`` is relative POSIX exactly
when the checkpoint lives under the registry's own scope root, absolute exactly when it does not,
so a pre-family absolute-under-root spelling can never read as a designed-external claim. Every
response surface (``list_models``, ``get_model``, ``best_model``, both ``register_model`` returns)
answers the resolved absolute path on a copy, never this internal storage spelling.
A bare top-level array (the shape this store carried before the family that wrapped it) is
never accepted for reading; no operator door rewraps a live project's registry in place, and
this registry predates the entries-mapping shape the platform writes, so nothing repairs it in
place. The only door that wraps one into the mapping shape and respells every entry is
``import_project``'s own staging conform
(:func:`~tcip_mcp.model_registry.conform_registry_paths_on_disk`), which reads and writes a
project's staging tree directly rather than through this module's own read path, so it never
meets the refusal above. A document neither shape refuses through :class:`RegistryVersionRefused`
naming what it found.
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
"""The ceiling this reader knows, per ``frozen-formats.json``. Never written into a document: the
index carries no ``schema_version`` field for as long as it stays at this ceiling (absence is the
frozen default), and this constant exists only for the store's registration and for accepting a
document that does carry an explicit ``1``."""
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
    """The registry index document is not a shape this reader accepts: a bare top-level array
    (the shape this store carried before the family that wrapped it), or a mapping whose
    ``schema_version``/``entries`` shape this reader does not recognize.

    Deliberately not a :class:`~tcip_store.StoreError`: a blanket ``StoreError`` catch (bundle's,
    for one) would otherwise swallow this into an empty answer, and an archive of an unconformed
    project would silently pack zero weights. Every caller states its own behavior on this
    exception rather than treating it like any other read failure.
    """


def _read_registry_document(raw: object) -> dict:
    """The registry's entries-mapping document, decoded from whatever the store handed back.

    ``raw`` absent (``None``, first use) answers the empty document: absence stays legitimate
    first use, never a refusal. A present bare list is the shape this store carried before the
    family that wrapped it, refused by name with the remedy rather than accepted or treated as
    unknown. A mapping is accepted with no ``schema_version`` key or with one equal to the frozen
    default; any other shape (a ``schema_version`` above the ceiling, ``entries`` missing or not a
    list) refuses naming what it found.
    """
    if raw is None:
        return {"entries": []}
    if isinstance(raw, list):
        raise RegistryVersionRefused(
            "the model registry index is a top-level JSON array, the shape this store carried "
            "before the family that wrapped it into an entries mapping; no operator door "
            "rewraps a live project's registry in place, and this registry predates the "
            "entries-mapping shape the platform writes, so nothing repairs it in place"
        )
    if (
        not isinstance(raw, dict)
        or raw.get("schema_version") not in (None, REGISTRY_SCHEMA_VERSION)
        or not isinstance(raw.get("entries"), list)
    ):
        raise RegistryVersionRefused(
            f"the model registry index is not a recognized entries-mapping document: {raw!r}"
        )
    return raw


def _write_registry_document(entries: list[dict]) -> dict:
    """The document a write puts on disk for ``entries``. Carries no ``schema_version`` field:
    absence is the frozen default, and the first writer of the field is whichever future change
    bumps this format."""
    return {"entries": entries}


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
    Raises :class:`RegistryVersionRefused` for a document this reader does not recognize (a
    bare top-level array included); see the module docstring for the archive/import remedy.
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
        "second checkpoint of the same run -- model_final beside a model_best, or a bespoke tag "
        "-- is registered in explicit mode under a distinct name, since experiment mode names "
        "the entry after the run and replaces by name)."
    )


def _resolve_producer(entries: tuple[dict, ...], *, checkpoint_path: Path, digest: str) -> str | None:
    """The one producer ``entries`` (already matched by ``sha256``) agree on.

    The producer is the entry's own ``experiment_id`` field, written only by a run's own
    completion through the experiment-mode binding (:func:`_register_entry`) and ``None`` for an
    explicit-mode entry: a verified fact, not a caller assertion (``tags`` carries no producer
    claim any more). Every matched entry must carry the key at all, or the load refuses by name,
    as ``best_model`` refuses a pre-``metrics_source`` entry: no operator door adds the missing
    key to an existing entry. An entry naming ``None`` is ignored (not a vote for ``None``);
    every entry that does name a producer must name the same one, or the load refuses rather
    than guess a first match.
    """
    missing = [str(e.get("name")) for e in entries if "experiment_id" not in e]
    if missing:
        raise UnregisteredCheckpoint(
            f"{checkpoint_path} (sha256 {digest}) is named by registry entries {sorted(missing)!r} "
            "that carry no experiment_id key (they predate the producer-binding field); no "
            "operator door adds the missing key to an existing entry, so this checkpoint cannot "
            "be loaded until those entries are corrected."
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
    """A checkpoint the registry names no entry for, whose registered entries disagree on
    producer, whose payload cannot be trusted to unpickle under ``weights_only=True``, or whose
    payload carries a ``schema_version`` this reader does not accept: whatever identity check
    already passed (a registry-name match, or a completion's own recorded digest), none of these
    is a payload this reader can act on."""


def _load_verified_payload(data: bytes, *, source: str) -> dict:
    """Unpickle already identity-verified checkpoint bytes, refusing a payload this reader
    cannot act on.

    The one implementation :func:`load_registered_checkpoint` (identity verified by a registry
    name match) and ``experiments.register_model_from_experiment`` (identity verified by the
    run's own recorded completion digest) both call, once their own identity check has passed,
    so the unpickle discipline (``weights_only=True``) and the ``schema_version`` ceiling can
    never drift between the two doors a checkpoint's bytes reach this platform's own reader
    through. ``source`` names the checkpoint (path and digest) in every raised message.

    Raises :class:`UnregisteredCheckpoint` for a payload that will not unpickle under
    ``weights_only=True``, or that carries a ``schema_version`` this reader does not accept.
    Raises a bare ``ValueError`` (:func:`~tcip_mcp.pipelines.inference.predictor._require_dict_payload`)
    for a payload that does not unpickle to a dict, the same fact the kind sniff raises.
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

    Closes the family of forgeries option B rules out: a checkpoint dropped at any path with no
    registry entry, and a checkpoint whose file is replaced (in place or by rename) between a
    caller checking its identity and a caller loading its weights. In order: the file is read
    into one ``bytes`` object; the digest is taken over that exact object through
    :func:`_sha256_of_bytes` (the same function :meth:`ModelRegistry.register_model` hashes with,
    so registration and this load are literally the same bytes-to-digest code); the registry
    index of ``project_path`` (or, unset, :func:`~tcip_mcp.project_paths.platform_state_root`) is read
    and every entry whose ``sha256`` equals the digest is collected; none of them raises
    :class:`UnregisteredCheckpoint`, naming the path, the digest, the root searched, and the
    remedy. Several entries naming one digest must agree on producer or the load refuses (see
    :func:`_resolve_producer`). Only once the registry has answered is the payload unpickled and
    version-checked, through :func:`_load_verified_payload`, a narrower sink than the
    ``weights_only=False`` sniff a predictor build used to run: every deliverable checkpoint this
    platform's own trainer writes (a state dict, JSON config, numeric metrics and the three
    stamps) loads under it; a payload that will not (a periodic resume checkpoint's RNG/optimizer
    state, or a bespoke loop's own arbitrary object), or that carries a ``schema_version`` this
    reader does not accept, refuses naming the reason, since registration verifies identity,
    never payload shape. A payload that does not unpickle to a dict raises the same ``ValueError``
    a kind sniff raises for the same fact. A missing file raises ``FileNotFoundError`` before any
    read.

    The digest and the load are over one immutable byte string, so no replacement of the file, in
    place or by rename, on either platform, can separate them.
    """
    from tcip_mcp.project_paths import platform_state_root

    ckpt = Path(checkpoint_path)
    root = project_path or str(platform_state_root())
    with open(ckpt, "rb") as f:
        data = f.read()
    digest = _sha256_of_bytes(data)
    index = read_registry_index(root)
    entries = tuple(e for e in index if e.get("sha256") == digest)
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


class RegistryEntryPredatesMetricsSource(ValueError):
    """A registry entry carries no ``metrics_source`` key at all: a malformed record predating
    the field, refused by :meth:`ModelRegistry.best_model` rather than ranked as just another
    unverified entry. Typed distinctly from a bare ``ValueError`` so a caller (the comparison
    route's 409 mapping) can catch this one refusal and let every other ``ValueError`` propagate."""


def _refuse_if_owned_by_another_run(superseded: dict, *, name: str, new_experiment_id: str | None) -> None:
    """The eviction rail: a name a run bound is that run's for good.

    Reads ``experiment_id`` directly, never through ``.get()``, so a pre-field entry (predating
    the producer-binding field) is not silently evictable: the key's absence itself refuses,
    stating the fact plainly, exactly as a present-but-different owner does.
    """
    try:
        owner = superseded["experiment_id"]
    except KeyError:
        raise EntryOwnedByRun(
            f"registry entry {name!r} carries no experiment_id key (it predates the "
            "producer-binding field); no operator door adds the missing key to an existing "
            "entry, so it cannot be replaced by name until it is corrected."
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
    whose superseded entry names a run other than this write's own. Reads and writes through the
    entries-mapping document pair: an unwrapped bare-array document raises
    :class:`RegistryVersionRefused` naming what it found before anything is written.
    """
    index = _read_registry_document(txn.read(key, default=None))["entries"]
    superseded = next((e for e in index if e["name"] == entry["name"]), None)
    if superseded is not None:
        _refuse_if_owned_by_another_run(
            superseded, name=entry["name"], new_experiment_id=entry.get("experiment_id"))
    index = [e for e in index if e["name"] != entry["name"]]
    index.append(entry)
    txn.write(key, _write_registry_document(index))
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
    """The one write behind both registration modes: replace-by-name inside one transaction over
    the project's index key, then audit the replacement (or the refusal) once it has closed.

    ``sha256``, when given (experiment mode: the digest a run's completion already recorded, and
    the caller's own file already verified against it), is written as-is, no second hash of the
    path; ``None`` (explicit mode) hashes ``checkpoint_path`` here, the one production hash
    explicit-mode registration takes. ``experiment_id`` is the entry's own producer-binding field:
    the run that bound it in experiment mode, ``None`` in explicit mode.

    The stored ``checkpoint_path`` is spelled through
    :func:`~tcip_mcp.registry_paths.checkpoint_registry_path_for` against this project's own root
    (the registry's scope by definition): relative POSIX when the checkpoint resolves under it,
    absolute when it does not, the one carrier of internal versus external. A checkpoint
    outside the project (``sha256`` given but the file gone by the time this runs) keeps the
    caller's own string verbatim, since there is nothing to resolve or spell against.

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
    try:
        with tcip_store.transaction(key) as txn:
            superseded = _write_registry_entry(txn, key, entry)
    except EntryOwnedByRun as exc:
        _audit_refused("_register_entry", {
            "name": name, "attempted_experiment_id": experiment_id, "reason": str(exc),
        })
        raise
    _audit_entry_replace(name, superseded, entry)
    return entry


_STRAY_SCHEMA_VERSION_TWO = 2
"""A registry index's own stray ``schema_version`` value from before this store's version-1
reset: this conform's one-time acceptance window, dropped at the store's first real bump. The
seam's own read-side ceiling refuses this value outright (``REGISTRY_SCHEMA_VERSION`` is 1), so
only :func:`conform_registry_paths_on_disk`'s raw-bytes read ever reaches
:func:`_document_entries_for_conform` carrying it; once this store's ceiling moves again, a
document actually stamped with the store's new second version must refuse here exactly as the
seam already refuses it, never be silently downgraded by this conform reusing this same literal."""


def _document_entries_for_conform(raw: object) -> tuple[list[dict], bool, bool]:
    """(entries, was_already_wrapped, had_stray_schema_version_two) for this conform's own
    read.

    Unlike :func:`_read_registry_document`, two dev-era shapes are accepted here rather than
    refused, since wrapping and respelling one of them is exactly this conform's own purpose: a
    bare top-level array (the shape this store carried before the family that wrapped it), and a
    mapping still carrying a stray ``schema_version: 2`` from before this store's version-1 reset
    (:func:`conform_registry_paths_on_disk` reads such a document directly, bypassing the seam's
    own ceiling refusal, precisely to reach this function; the seam's own read otherwise refuses a
    stray 2 outright, so this function is never reached carrying one except through that bypass).
    Both rewrite through :func:`_write_registry_document`, which carries no ``schema_version`` field, so
    the field is dropped on the same write that wraps or respells; ``had_stray_schema_version_two``
    tells the caller that drop actually happened, so it can be disclosed as its own outcome line
    rather than silently folded into "already wrapped, nothing to say". Anything else refuses,
    naming what was found rather than the whole document, since a caller of this function
    (``import_project``'s own conform step) surfaces the refusal message to a remote caller.
    """
    if raw is None:
        return [], True, False
    if isinstance(raw, list):
        return raw, False, False
    if isinstance(raw, dict):
        version = raw.get("schema_version")
        entries = raw.get("entries")
        if version in (None, REGISTRY_SCHEMA_VERSION, _STRAY_SCHEMA_VERSION_TWO) and isinstance(entries, list):
            return entries, True, version == _STRAY_SCHEMA_VERSION_TWO
        if version not in (None, REGISTRY_SCHEMA_VERSION, _STRAY_SCHEMA_VERSION_TWO):
            raise RegistryVersionRefused(
                f"the model registry index carries schema_version={version!r}, not one this "
                "conform recognizes"
            )
        raise RegistryVersionRefused(
            f"the model registry index's entries field is {type(entries).__name__}, not a list"
        )
    raise RegistryVersionRefused(
        f"the model registry index is a {type(raw).__name__}, neither a bare array nor an "
        "entries mapping this conform recognizes"
    )


def _candidate_checkpoint_paths(root: Path, raw: str) -> set[Path]:
    """Every file under ``root`` worth hashing while relocating one entry's checkpoint.

    A purpose-built enumeration, never bundle's ``_blob_files`` (whose sweep hashes a project's
    imagery and whose models glob is non-recursive): the ``.tcip/models`` tree recursively, the
    ``.tcip/experiments`` tree's checkpoint files, and any file under ``root`` the entry's own
    relative suffix (everything from its last ``.tcip`` segment onward) or basename names.
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
    """``hash_cache``'s own digest for ``path``, computed once per run.

    ``dict.setdefault(key, _compute_sha256(path))`` evaluates its default argument eagerly, so a
    cache built that way rehashes every candidate on every lookup rather than only the first; this
    is the one place the whole file's digest is actually cached.
    """
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
    entries: list[dict], root: Path, *, plan: bool, hash_cache: dict[Path, str],
) -> tuple[list[dict], list[str]]:
    """Respell every entry's ``checkpoint_path`` relative to ``root``, per
    :func:`conform_registry_paths_on_disk`'s own rule. Returns the conformed entries and one
    outcome line per entry actually changed (or, under ``plan``, that would change)."""
    verb = "would respell" if plan else "respelled"
    conformed: list[dict] = []
    lines: list[str] = []
    for entry in entries:
        raw = entry.get("checkpoint_path")
        if not raw:
            conformed.append(entry)
            continue
        expected_sha = entry.get("sha256")
        direct = Path(raw) if is_external_form(str(raw)) else root.joinpath(*PurePosixPath(raw).parts)
        direct_resolved = direct.resolve()
        if (
            is_at_or_under(direct_resolved, root)
            and direct_resolved.is_file()
            and expected_sha is not None
            and _cached_sha256(hash_cache, direct_resolved) == expected_sha
        ):
            respelled = checkpoint_registry_path_for(direct_resolved, root)
            if respelled != raw:
                lines.append(f"{entry.get('name')}: {verb} {raw!r} to {respelled!r}")
                conformed.append({**entry, "checkpoint_path": respelled})
            else:
                conformed.append(entry)
            continue

        candidates = sorted(_candidate_checkpoint_paths(root, str(raw)))
        matches = []
        if expected_sha is not None:
            for candidate in candidates:
                size = entry.get("file_size_bytes")
                if size is not None and candidate.stat().st_size != size:
                    continue
                if _cached_sha256(hash_cache, candidate) == expected_sha:
                    matches.append(candidate)

        if len(matches) == 1:
            respelled = checkpoint_registry_path_for(matches[0], root)
            lines.append(f"{entry.get('name')}: {verb} {raw!r} to {respelled!r} (relocated)")
            conformed.append({**entry, "checkpoint_path": respelled})
        elif len(matches) > 1:
            chosen = _pick_duplicate(matches, root=root, original_basename=Path(str(raw)).name)
            respelled = checkpoint_registry_path_for(chosen, root)
            named = sorted(m.relative_to(root).as_posix() for m in matches)
            lines.append(f"{entry.get('name')}: {verb} {raw!r} to {respelled!r}, ambiguous "
                         f"among {len(matches)} byte-identical candidates {named}, picked by "
                         "basename match then sorted path")
            conformed.append({**entry, "checkpoint_path": respelled})
        elif is_external_form(str(raw)):
            # The stored value is itself the external claim; a host-resolved respelling of it
            # fabricates a path this machine derived, not one the writer stated.
            lines.append(f"{entry.get('name')}: {raw!r} kept as stored "
                         f"(external-or-missing, exists={direct_resolved.is_file()})")
            conformed.append(entry)
        else:
            # A relative entry keeps its spelling: writing an absolute path here would fabricate a designed-external claim.
            lines.append(f"{entry.get('name')}: {raw!r} stays unresolved, no matching digest "
                         f"found under root (exists={direct_resolved.is_file()})")
            conformed.append(entry)
    return conformed, lines


def _wrap_and_drop_lines(*, already_wrapped: bool, had_stray_two: bool, plan: bool) -> list[str]:
    """The outcome lines :func:`_document_entries_for_conform`'s own findings earn, in the tense
    ``plan`` calls for: a bare array wrapped into the entries mapping, a stray
    ``schema_version: 2`` dropped, both, or neither."""
    lines = []
    if not already_wrapped:
        lines.append(("wrapping" if plan else "wrapped") + " the registry index into the entries mapping")
    if had_stray_two:
        lines.append(("dropping" if plan else "dropped") + f" a stray schema_version: {_STRAY_SCHEMA_VERSION_TWO}")
    return lines


def conform_registry_paths_on_disk(root: str | Path) -> list[str]:
    """Wrap ``root``'s registry index into the entries mapping and respell every entry's
    ``checkpoint_path`` relative to ``root``, reading and writing the registry index file
    directly rather than through the storage seam.

    For a tree the caller already holds exclusive access to but that is not yet a database
    (``import_project``'s own staging tree, always loose files until adoption runs): a seam
    transaction there either refuses outright (the database backend refuses a write to an
    unconformed root) or, once adopted, leaves a cached connection open on a directory the door
    is about to rename, which Windows refuses. This is the registry's only conform: no operator
    door repairs a live, already-adopted project's registry in place.

    Per entry: a stored path that resolves under ``root`` with a matching ``sha256`` is
    respelled through the checkpoint speller (existence alone never blesses a replaced file). A
    path that does not (outside ``root``, missing, or a hash mismatch) is relocated: every file
    under ``.tcip/models``, every checkpoint-shaped file under ``.tcip/experiments``, and any
    file the entry's own stored suffix or basename names (:func:`_candidate_checkpoint_paths`),
    prefiltered by the entry's recorded ``file_size_bytes``, is hashed once (each hash cached
    across every entry this run examines) and matched against the entry's own ``sha256``.
    Exactly one match respells to it; more than one (byte-identical files) picks
    deterministically (:func:`_pick_duplicate`) and is disclosed as ambiguous; none leaves an
    already-external entry absolute, classified external-or-missing by existence in the outcome
    line, and leaves an already-relative entry's spelling untouched (the truthful claim stays
    internal-but-absent) rather than ever writing a path under the conform root.

    Reading the raw bytes directly, bypassing the seam's own schema_version ceiling check, is
    what lets this route accept a document still carrying a stray ``schema_version: 2`` from
    before this store's version-1 reset (the seam's own entry point refuses that value outright,
    on read as on write): this is the one conform that accepts it, since ``import_project`` is
    the only door this function serves.

    Idempotent: a second run changes nothing, since an already-correctly-spelled entry's
    respelled form always equals its stored one. Raises :class:`RegistryVersionRefused` for a
    document neither a bare array nor a recognized entries mapping.
    """
    root_path = Path(root).resolve()
    path = registry_index_path(root_path)
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise RegistryVersionRefused(f"{path} will not decode as JSON: {exc}") from exc
    entries, already_wrapped, had_stray_two = _document_entries_for_conform(raw)
    conformed, lines = _conform_entries(entries, root_path, plan=False, hash_cache={})
    path.write_bytes(RECORD_JSON.encode(_write_registry_document(conformed)))
    prefix = _wrap_and_drop_lines(already_wrapped=already_wrapped, had_stray_two=had_stray_two, plan=False)
    return prefix + lines


def _resolve_entry_checkpoint(project_path: str, entry: dict) -> dict:
    """A shallow copy of ``entry`` with ``checkpoint_path`` resolved to an absolute path.

    Every response surface hands the caller something they can open, or feed back into a
    path-taking door, directly, never the registry's own internal-vs-external storage
    spelling: a resolve can correct case, separators, or a caller's cwd-relative process root.
    A stored value this reader cannot resolve (empty, or a relative form carrying ``..``) is
    never raised through a listing: the row keeps its stored spelling and carries the reason in
    ``checkpoint_path_error`` instead, so one malformed entry never hides the rest of the
    registry from a caller listing every model.
    """
    copy = dict(entry)
    stored = entry.get("checkpoint_path") or ""
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

        Explicit mode's own entry point: wraps :func:`_register_entry` with ``experiment_id=None``
        (no run bound this weights file, whatever ``tags`` the caller passes) and no ``sha256``
        (this call's own hash of ``checkpoint_path`` is the only production hash explicit mode
        takes). Experiment mode (``register_model_from_experiment``) calls :func:`_register_entry`
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
        """List registered models, optionally filtered by tag.

        Every returned entry is a copy carrying the resolved absolute ``checkpoint_path``
        (:func:`_resolve_entry_checkpoint`), never the internal registry's own stored spelling,
        and never the live ``self._index`` list itself (a caller mutating one row must not
        corrupt the registry's own in-memory state).
        """
        entries = self._index if tag is None else [m for m in self._index if tag in m.get("tags", [])]
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

        Only models that actually carry the metric are considered, a missing metric is
        skipped, not scored as a sentinel, so ``None`` cleanly means "no model has it". A
        value that is not a finite number is skipped for the same reason: it compares false
        against every candidate, so an incumbent holding one could never be displaced.
        ``metric_key`` and ``higher_is_better`` are both required: this reader guesses neither
        the metric nor its direction from the key's spelling. By default only an entry whose
        ``metrics_source`` is ``"trainer"``, the one path the platform itself measured, is
        ranked; ``include_unverified=True`` also ranks ``"training_source"`` and ``"caller"``
        entries, whose numbers nothing here verified. An entry carrying no ``metrics_source`` key
        at all is a malformed record predating the field (and, for the same reason, predating
        ``experiment_id``), refused by name rather than silently treated as just another
        unverified entry: no operator door adds either missing field to an existing entry, so a
        registry carrying one must be corrected before ranking. A present
        ``metrics_source`` of ``None`` is not malformed, it is the honest pairing for an entry with
        no metrics. ``experiment_ids``, when given, narrows ranking to entries whose own
        ``experiment_id`` is in the set (an explicit-mode entry, whose ``experiment_id`` is
        ``None``, is never in it); ``None`` (the default) ranks the whole registry, unchanged.
        """
        malformed = [m.get("name") for m in self._index if "metrics_source" not in m]
        if malformed:
            raise RegistryEntryPredatesMetricsSource(
                f"registry entries {malformed} carry no metrics_source key (they predate the "
                "field, and the producer-binding field beside it); no operator door adds either "
                "missing field to an existing entry, so this registry cannot be ranked until "
                "they are corrected."
            )
        best = None
        best_val: float | None = None
        for m in self._index:
            if experiment_ids is not None and m.get("experiment_id") not in experiment_ids:
                continue
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
        return _resolve_entry_checkpoint(self._project_path, best) if best is not None else None
