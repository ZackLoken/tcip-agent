"""Model registry: the checkpoints a project can load, and what is known of each.

A checkpoint a run of this project produced is registered by that run's completed final status
(``experiments.RunObservation.checkpoint``). The index document, ``{entries: [...]}``, holds
foreign checkpoints only, one entry per sha256, each registered in explicit mode. A stored
``checkpoint_path`` is relative POSIX exactly when the checkpoint lives under the registry's own
scope root, absolute exactly when it does not, and is read as written. Every response surface
answers the resolved absolute path on a copy.
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import tcip_store
from tcip_store import RECORD_JSON, Key, StoreDescriptor, check_json_value, register_store
from tcip_store.file_backend import RootedFileLocator

from tcip_mcp.registry_paths import (
    RegistryPathEmpty,
    RegistryPathTraversal,
    checkpoint_registry_path_for,
    resolved_registry_path,
)

if TYPE_CHECKING:
    from tcip_mcp.experiments import RunObservation

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
    """Every foreign checkpoint the project has registered, in registration order.

    A project that has registered nothing reads as an empty list; an index that does not decode
    raises ``DecodeError``, and one that is not the entries mapping raises
    :class:`RegistryVersionRefused`.
    """
    raw = tcip_store.read(registry_index_key(project_path), default=None)
    return _read_registry_document(raw)["entries"]


def run_entry(observation: "RunObservation") -> dict | None:
    """The registry entry a completed training run's final status stands for, named by the run's
    id and carrying it as ``experiment_id``, or ``None`` for a run that has not completed."""
    checkpoint = observation.checkpoint
    if checkpoint is None:
        return None
    assert observation.final is not None, "a completed checkpoint is named by a final status"
    return {
        "name": observation.directory.name, "checkpoint_path": checkpoint["path"],
        "sha256": checkpoint["sha256"], "registered_at": observation.final["ended"],
        "config": observation.record["config"], "tags": [],
        "experiment_id": observation.directory.name,
    }


def registered_entries(project_path: str | Path) -> list[dict]:
    """Every checkpoint the project can load, one entry per sha256, its one owner: the completed
    training run whose final status names those bytes (:func:`run_entry`; the earliest to
    complete when two do), else the foreign entry of the index registering them, with
    ``experiment_id`` ``None``. Runs' entries come first, in completion order."""
    from tcip_mcp.experiments import run_observations

    runs = sorted((entry for entry in map(run_entry, run_observations(project_path))
                   if entry is not None), key=lambda entry: entry["registered_at"])
    owners: dict[str, dict] = {}
    for entry in [*runs, *({**e, "experiment_id": None} for e in read_registry_index(project_path))]:
        owners.setdefault(entry["sha256"], entry)
    return list(owners.values())


def entry_facts(entry: dict) -> dict:
    """What ``entry``'s checkpoint says of itself, read from its payload
    (:func:`checkpoint_payload`): the ``metrics`` a ranking reads with their ``metrics_source``. A run's metrics are the ones its payload carries, sourced ``"trainer"``,
    or ``"training_source"`` for a bespoke loop's; a foreign entry's are the ones its registration
    stated, sourced ``"caller"``. No metrics carry no source."""
    from tcip_mcp.pipelines.model_build import TRAINING_SOURCE_KEY

    payload = checkpoint_payload(entry["checkpoint_path"], entry["sha256"])
    if entry["experiment_id"] is None:
        metrics, source = entry["metrics"], "caller"
    else:
        metrics = payload.get("metrics") or {}
        source = "training_source" if entry["config"].get(TRAINING_SOURCE_KEY) else "trainer"
    return {"metrics": metrics, "metrics_source": source if metrics else None}


def _sha256_of_bytes(data: bytes) -> str:
    """The sha256 hex digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def _unregistered_checkpoint_error(checkpoint_path: Path, digest: str, root: str) -> "UnregisteredCheckpoint":
    return UnregisteredCheckpoint(
        f"{checkpoint_path} (sha256 {digest}) is not named by any completed run's final status or "
        f"entry in the registry at {root!r}: a completed run registers its own final weights, and "
        "a foreign checkpoint, or a second checkpoint of a run (model_final beside a model_best, "
        "or a bespoke tag), is registered with register_model under a name of its own."
    )


@dataclass(frozen=True)
class VerifiedCheckpoint:
    """A checkpoint read, hashed and matched against the registry before anything in it was
    unpickled.
    """

    path: str
    """The path the caller named, as given."""
    sha256: str
    """The digest of the exact bytes ``payload`` was unpickled from."""
    payload: dict
    """The loaded checkpoint, read with ``weights_only=True``."""
    entry: dict
    """The one registered entry of this digest (:func:`registered_entries`)."""

    @property
    def experiment_id(self) -> str | None:
        """The run whose final status names this checkpoint, ``None`` for a foreign one."""
        return self.entry["experiment_id"]

    @property
    def producer(self) -> dict[str, str | None]:
        """The producer every record of this checkpoint's output names: its sha256 and the run
        behind it."""
        return {"checkpoint_sha256": self.sha256, "experiment_id": self.experiment_id}

    @property
    def data_config(self) -> dict:
        """The checkpoint's own stamped ``config["data"]``, ``{}`` for a checkpoint carrying
        none (a foreign checkpoint's documented answer)."""
        data_cfg = (self.payload.get("config") or {}).get("data")
        return data_cfg if isinstance(data_cfg, dict) else {}

    @property
    def task(self) -> str:
        """The task the checkpoint's model is for, :func:`run_task` over its own config; raises
        ``ValueError`` when that states none."""
        from tcip_mcp.pipelines.model_build import run_task

        return run_task(self.payload.get("config") or {})


class UnregisteredCheckpoint(ValueError):
    """A checkpoint no completed run and no registry entry names, or one whose payload cannot be
    trusted to unpickle under ``weights_only=True``."""


def _load_verified_payload(data: bytes, *, source: str) -> dict:
    """Unpickle already identity-verified checkpoint bytes with ``weights_only=True``. ``source``
    names the checkpoint (path and digest) in every raised message.

    Raises :class:`UnregisteredCheckpoint` for a payload that will not unpickle under
    ``weights_only=True``, and a bare ``ValueError`` for a payload that does not unpickle to a
    dict or carries no weights under ``model_build``'s ``STATE_DICT_KEY``.
    """
    import torch  # local checkpoint an identity check already named; unpickling it is the point

    from tcip_mcp.pipelines.model_build import STATE_DICT_KEY

    try:
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise UnregisteredCheckpoint(
            f"{source} could not be loaded with weights_only=True ({exc}): this payload carries "
            "something outside a platform-written deliverable checkpoint's contract (a resume "
            "checkpoint's RNG/optimizer state is the trainer's own resume path to read; a bespoke "
            "loop's own arbitrary state is outside the contract)."
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{source} did not unpickle to a dict (got {type(payload).__name__}), so "
                         "it carries none of the config and weights this platform's checkpoints "
                         "do.")
    if STATE_DICT_KEY not in payload:
        raise ValueError(f"{source} carries no {STATE_DICT_KEY!r}: a checkpoint is admitted for "
                         "the weights it loads, and this one holds none.")
    return payload


def admitted_digest(checkpoint_path: str | Path) -> str:
    """The sha256 of the checkpoint file at ``checkpoint_path``, read once, returned only once the
    verified reader (:func:`_load_verified_payload`) admits the payload of those same bytes; its
    refusals raise, and an unreadable file raises ``OSError``."""
    data = Path(checkpoint_path).read_bytes()
    digest = _sha256_of_bytes(data)
    _load_verified_payload(data, source=f"{checkpoint_path} (sha256 {digest})")
    return digest


def checkpoint_payload(checkpoint_path: str | Path, sha256: str) -> dict:
    """The payload of the checkpoint at ``checkpoint_path`` (:func:`_load_verified_payload`), read
    only once its bytes are the ones ``sha256`` names. Bytes hashing otherwise raise
    :class:`UnregisteredCheckpoint` naming both digests."""
    data = Path(checkpoint_path).read_bytes()
    digest = _sha256_of_bytes(data)
    if digest != sha256:
        raise UnregisteredCheckpoint(
            f"{checkpoint_path} now hashes to {digest}, not the {sha256} its record names: the "
            "file was replaced after it was recorded.")
    return _load_verified_payload(data, source=f"{checkpoint_path} (sha256 {digest})")


def load_registered_checkpoint(checkpoint_path: str | Path, *, project: Path) -> VerifiedCheckpoint:
    """Read a checkpoint's bytes once, hash them, and refuse unless ``project``'s registry names
    that hash.

    Refuses two forgeries: a checkpoint dropped at any path that nothing registered, and a
    checkpoint whose file is replaced (in place or by rename) between a caller checking its
    identity and a caller loading its weights. In order: the file is read into one ``bytes``
    object; the digest is taken over that exact object through :func:`_sha256_of_bytes` and
    looked up among ``project``'s :func:`registered_entries`, none raising
    :class:`UnregisteredCheckpoint` naming the path, the digest, the root searched and the remedy.
    Only then is the payload unpickled (:func:`_load_verified_payload`). A missing file raises
    ``FileNotFoundError`` before any read.

    The digest and the load are over one immutable byte string.
    """
    ckpt = Path(checkpoint_path)
    root = str(project)
    with open(ckpt, "rb") as f:
        data = f.read()
    digest = _sha256_of_bytes(data)
    entry = next((e for e in registered_entries(root) if e["sha256"] == digest), None)
    if entry is None:
        raise _unregistered_checkpoint_error(ckpt, digest, root)
    payload = _load_verified_payload(data, source=f"{ckpt} (sha256 {digest})")
    return VerifiedCheckpoint(path=str(checkpoint_path), sha256=digest, payload=payload,
                              entry=entry)


def _write_registry_entry(txn: tcip_store.Txn, key: Key,
                          entry: dict) -> tuple[dict | None, dict]:
    """Write ``entry`` as the one foreign entry of its sha256 inside ``txn``'s already-open
    transaction over the index key.

    Returns ``(superseded, stored)``: the entry of that sha256 replaced (``None`` for its first
    registration), and the entry the registry holds after the call, ``entry`` itself when it was
    written and the superseded entry when that already holds every field but ``registered_at``,
    which is no change and so no write.
    """
    index = _read_registry_document(txn.read(key, default=None))["entries"]
    superseded = next((e for e in index if e["sha256"] == entry["sha256"]), None)
    if superseded is not None:
        if ({k: v for k, v in superseded.items() if k != "registered_at"}
                == {k: v for k, v in entry.items() if k != "registered_at"}):
            return superseded, superseded
    index = [e for e in index if e["sha256"] != entry["sha256"]]
    index.append(entry)
    txn.write(key, _write_registry_document(index))
    return superseded, entry


def _audit_entry_write(project_path: str, superseded: dict | None, entry: dict) -> None:
    """Emit ``model_registered`` in the project's log for a written ``entry``, naming the entry of
    the same sha256 it superseded. A failed append raises ``AuditEntryNotWritten``."""
    from tcip_mcp.audit import record_event_or_raise

    replaced = {} if superseded is None else {
        "superseded_name": superseded["name"], "superseded_tags": superseded["tags"],
    }
    record_event_or_raise("model_registered",
                          {"name": entry["name"], "new_sha256": entry["sha256"], **replaced},
                          scope=project_path)


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
    """A project's registered checkpoints (:func:`registered_entries`), each read at the call that
    asks, and the explicit-mode door that registers a foreign one."""

    def __init__(self, project_path: str) -> None:
        self._project_path = project_path
        self.root = registry_index_path(project_path).parent
        self.root.mkdir(parents=True, exist_ok=True)

    def register_model(
        self,
        name: str,
        checkpoint_path: str,
        config: dict,
        metrics: dict | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        """Register a checkpoint by its sha256 (:func:`admitted_digest`), under ``name`` as its
        presentation, record the write, and return the owner :func:`registered_entries` resolves
        those bytes to: this entry, or the completed run whose final status names the same bytes.
        A second registration of the same bytes replaces the first. Its ``metrics`` are asserted by
        the caller, never verified. The stored ``checkpoint_path`` is spelled through
        :func:`~tcip_mcp.registry_paths.checkpoint_registry_path_for` against the project root.

        Args:
            name: Model name (e.g. '<crop>_<trait>_detector_v1').
            checkpoint_path: Path to the .pt checkpoint file.
            config: Training config dict.
            metrics: Evaluation metrics dict.
            tags: Optional tags for filtering.

        Raises:
            FileNotFoundError: ``checkpoint_path`` does not exist.
            TypeError / ValueError: ``config`` or ``metrics`` holds something JSON cannot carry,
                or the checkpoint's payload is one the verified reader refuses, named before
                anything is stored.
            AuditEntryNotWritten: the write committed but its own audit line could not be appended.
        """
        check_json_value(config, path="config")
        check_json_value(metrics or {}, path="metrics")
        ckpt = Path(checkpoint_path)
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"register_model: checkpoint_path {checkpoint_path!r} does not exist, refusing to "
                "register a phantom registry entry.")
        entry = {
            "name": name,
            "checkpoint_path": checkpoint_registry_path_for(ckpt, self._project_path),
            "sha256": admitted_digest(ckpt),
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "metrics": metrics or {},
            "tags": tags or [],
        }
        key = registry_index_key(self._project_path)
        with tcip_store.transaction(key) as txn:
            superseded, stored = _write_registry_entry(txn, key, entry)
        if stored is entry:
            _audit_entry_write(self._project_path, superseded, entry)
        owner = next(e for e in registered_entries(self._project_path)
                     if e["sha256"] == entry["sha256"])
        return _resolve_entry_checkpoint(self._project_path, owner)

    def list_models(self, tag: str | None = None) -> list[dict]:
        """Every registered checkpoint carrying ``tag`` (every one when ``tag`` is ``None``), each
        a copy carrying the resolved absolute ``checkpoint_path``
        (:func:`_resolve_entry_checkpoint`) and what its payload says of it (:func:`entry_facts`).
        """
        resolved = [_resolve_entry_checkpoint(self._project_path, m)
                    for m in registered_entries(self._project_path)
                    if tag is None or tag in m["tags"]]
        # A path the resolution above refused names no file to read facts from; its error says so.
        return [m if "checkpoint_path_error" in m else {**m, **entry_facts(m)} for m in resolved]


def best_model(
    models: list[dict], metric_key: str, *, higher_is_better: bool,
    include_unverified: bool = False,
) -> dict | None:
    """The one of ``models`` (:meth:`ModelRegistry.list_models`'s entries) with the best finite
    value for ``metric_key`` in the stated direction, or ``None`` when none carries it. Only an
    entry whose ``metrics_source`` is ``"trainer"`` is ranked unless ``include_unverified``, which
    also ranks ``"training_source"`` and ``"caller"`` entries."""
    best = None
    best_val: float | None = None
    for m in models:
        if m["metrics_source"] != "trainer" and not include_unverified:
            continue
        val = m["metrics"].get(metric_key)
        if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(val):
            continue
        if best_val is None or (val > best_val if higher_is_better else val < best_val):
            best_val = float(val)
            best = m
    return best
