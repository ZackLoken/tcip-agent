"""Model registry: the checkpoints a project can load, and what is known of each.

A checkpoint a run of this project produced is registered by that run's completed final status
(``experiments.RunObservation.checkpoint``). The index document, ``{entries: [...]}``, holds
foreign checkpoints only, one entry per sha256, each registered in explicit mode. A stored
``checkpoint_path`` is relative POSIX exactly when the checkpoint lives under the registry's own
scope root, absolute exactly when it does not, and is read as written.
"""

from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SerializerFunctionWrapHandler,
    ValidationError,
    model_serializer,
)

import tcip_store
from tcip_mcp.audit import now_iso
from tcip_mcp.pipelines.model_build import (
    CONFIG_KEY,
    GEOMETRY_KEY,
    METRICS_KEY,
    SNAPSHOT_KEY,
)
from tcip_mcp.pipelines.schemas import TrainGeometry
from tcip_mcp.registry_paths import (
    checkpoint_registry_path_for,
    is_external_form,
    located,
    resolved_registry_path,
    runtime_paths,
)
from tcip_store import (
    FINITE_JSON_OBJECT,
    FiniteJsonObject,
    Key,
    finite_number,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tcip_mcp.experiments import RunObservation, RunRecord
    from tcip_mcp.pipelines.model_build import SourceLayout
    from tcip_mcp.pipelines.schemas import TrainConfigSchema

logger = logging.getLogger(__name__)

MODEL_REGISTRY_STORE = "model_registry"
_INDEX_PARTS = ("registry",)


def registry_index_key(project_path: str | Path) -> Key:
    """Return the key of the project's registered-model index, a :class:`RegistryIndex`."""
    return Key(MODEL_REGISTRY_STORE, str(project_path), _INDEX_PARTS)


class RegistryEntry(BaseModel):
    """One checkpoint a project can load, under the name and tags it was registered with.

    ``checkpoint_path`` is as stored (:func:`~tcip_mcp.registry_paths.checkpoint_registry_path_for`)
    and ``sha256`` the digest of its bytes. ``metrics`` are the ones its registration asserted: a
    caller's for a foreign checkpoint, none for a run's own entry, whose metrics its checkpoint
    carries (:func:`entry_facts`). ``experiment_id`` is the run whose final status names it,
    ``None`` for a foreign checkpoint.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    checkpoint_path: str
    sha256: str
    registered_at: str
    metrics: FiniteJsonObject
    tags: list[str]
    experiment_id: str | None


class RegistryIndex(BaseModel):
    """The project's registered-model index: its foreign checkpoints, one entry per sha256."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entries: list[RegistryEntry]


def _index_entries(stored: JsonValue) -> list[RegistryEntry]:
    """Return the entries of a stored index, validated as a :class:`RegistryIndex`."""
    return RegistryIndex.model_validate(stored).entries


def read_registry_index(project_path: str | Path) -> list[RegistryEntry]:
    """Return every foreign checkpoint the project has registered, in registration order.

    A project that has registered nothing reads as an empty list; an index that does not decode
    raises ``DecodeError``, and one that does not validate ``pydantic.ValidationError``.
    """
    return _index_entries(
        tcip_store.read(
            registry_index_key(project_path), default={"entries": []}
        )
    )


def _checkpoint_entry_file(project_path: Path, raw: str) -> Path | None:
    """Return the existing file under ``project_path`` a stored registry ``checkpoint_path`` names.

    ``raw`` is the path as stored. Returns ``None`` for an entry naming another tree or a relative
    one with nothing at it; a stored path the resolver refuses (empty, or a relative form carrying
    ``..``) raises its refusal (:func:`~tcip_mcp.registry_paths.resolved_registry_path`).
    """
    resolved = resolved_registry_path(project_path, raw).resolve()
    return (
        resolved
        if resolved.is_file() and resolved.is_relative_to(project_path)
        else None
    )


def checkpoint_files(root: Path) -> frozenset[Path]:
    """Return every checkpoint file under the established project root ``root``.

    These are each file in ``.tcip/models``, each file a registry entry names under the project,
    and each ``.pt`` file a run directory (``experiments.run_dirs``, a sweep's trials included)
    holds beside its run record. A stored entry path the resolver refuses raises its refusal.
    """
    from tcip_mcp.experiments import run_dirs

    found = {p for p in (root / ".tcip" / "models").glob("*") if p.is_file()}
    found |= {
        path
        for path in (
            _checkpoint_entry_file(root, e.checkpoint_path)
            for e in read_registry_index(root)
        )
        if path is not None
    }
    found |= {p for run_dir in run_dirs(root) for p in run_dir.glob("*.pt")}
    return frozenset(found)


def registry_checkpoint_disclosures(root: Path) -> dict[str, list]:
    """Return what the registry index of project root ``root`` says of checkpoints outside it.

    ``root`` is the established project root. ``checkpoint_paths_unresolved`` is every stored
    path that is not a designed-external claim (:func:`~tcip_mcp.registry_paths.is_external_form`)
    and names no file under the project, as stored; ``external_checkpoints`` is every
    designed-external path with whether it exists. A stored path the resolver refuses raises its
    refusal.
    """
    stored = sorted(e.checkpoint_path for e in read_registry_index(root))
    return {
        "checkpoint_paths_unresolved": [
            raw
            for raw in stored
            if not is_external_form(raw)
            and _checkpoint_entry_file(root, raw) is None
        ],
        "external_checkpoints": [
            {"checkpoint_path": raw, "exists": Path(raw).is_file()}
            for raw in stored
            if is_external_form(raw)
        ],
    }


def run_entry(
    observation: RunObservation[RunRecord],
) -> RegistryEntry | None:
    """Return the registry entry a completed training run's final status stands for.

    The entry is named by the run's id and carries it as ``experiment_id``; ``None`` for a run
    that has not completed.
    """
    checkpoint = observation.checkpoint
    if checkpoint is None:
        return None
    assert observation.final is not None, (
        "a completed checkpoint is named by a final status"
    )
    return RegistryEntry(
        name=observation.directory.name,
        checkpoint_path=checkpoint.path,
        sha256=checkpoint.sha256,
        registered_at=observation.final.ended,
        metrics={},
        tags=[],
        experiment_id=observation.directory.name,
    )


def registered_entries(project_path: str | Path) -> list[RegistryEntry]:
    """Return every checkpoint the project can load, one entry per sha256, from its one owner.

    The owner is the completed training run whose final status names those bytes
    (:func:`run_entry`), else the foreign entry of the index registering them. Runs' entries come
    first, in completion order.
    """
    from tcip_mcp.experiments import run_observations

    runs = sorted(
        (
            entry
            for entry in map(run_entry, run_observations(project_path))
            if entry is not None
        ),
        key=lambda entry: entry.registered_at,
    )
    owned = {entry.sha256 for entry in runs}
    return [
        *runs,
        *(
            e
            for e in read_registry_index(project_path)
            if e.sha256 not in owned
        ),
    ]


type MetricsSource = Literal["trainer", "training_source", "caller"]
"""Who measured a registered checkpoint's metrics: its trainer, a bespoke loop, or the caller."""


class CheckpointFacts(BaseModel):
    """What a registered checkpoint says of itself: its ``metrics`` and their ``metrics_source``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metrics: FiniteJsonObject
    metrics_source: MetricsSource | None


def checkpoint_metrics(payload: Mapping[str, Any]) -> dict[str, JsonValue]:
    """Return the metrics a checkpoint ``payload`` carries, an empty mapping when it carries none.

    The metrics under ``model_build.METRICS_KEY`` are a ``tcip_store.FiniteJsonObject``; anything
    else refuses (``pydantic.ValidationError``) naming the position.
    """
    if METRICS_KEY not in payload:
        return {}
    return FINITE_JSON_OBJECT.validate_python(payload[METRICS_KEY])


def entry_facts(entry: RegistryEntry, project: Path) -> CheckpointFacts:
    """Return what ``entry``'s checkpoint, a checkpoint of ``project``, says of itself.

    ``entry``'s ``checkpoint_path`` is the location the checkpoint is read at. It is read from
    its payload (:func:`checkpoint_payload`): the ``metrics`` with their ``metrics_source``. A
    run's metrics are the ones its payload carries, sourced ``"trainer"``, or
    ``"training_source"`` when the payload's config (:attr:`VerifiedCheckpoint.spec`) names a
    bespoke loop; a foreign entry's are the ones its registration stated, sourced ``"caller"``.
    No metrics carry no source. The metrics are admitted as :class:`CheckpointFacts` constructs
    them, whatever the entry they came from now holds.
    """
    checkpoint = VerifiedCheckpoint(
        path=entry.checkpoint_path,
        entry=entry,
        project=project,
        payload=checkpoint_payload(entry.checkpoint_path, entry.sha256),
    )
    source: MetricsSource
    if entry.experiment_id is None:
        metrics, source = entry.metrics, "caller"
    else:
        metrics = checkpoint_metrics(checkpoint.payload)
        source = (
            "training_source" if checkpoint.spec.training_source else "trainer"
        )
    return CheckpointFacts(
        metrics=metrics, metrics_source=source if metrics else None
    )


def _sha256_of_bytes(data: bytes) -> str:
    """Return the sha256 hex digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def _unregistered_checkpoint_error(
    checkpoint_path: Path, digest: str, root: str
) -> UnregisteredCheckpointError:
    return UnregisteredCheckpointError(
        f"{checkpoint_path} (sha256 {digest}) is not named by any completed run's final status or "
        f"entry in the registry at {root!r}: a completed run registers its own final weights, and "
        "a foreign checkpoint, or a second checkpoint of a run (model_final beside a model_best, "
        "or a bespoke tag), is registered with register_model under a name of its own."
    )


class CheckpointStamp(BaseModel):
    """The stamp a checkpoint carries beside its weights, as ``checkpoint_stamp`` writes it.

    That is ``generic_trainer.checkpoint_stamp``: its run's ``config`` as it records itself, its
    paths stored against the run's project, the digest of the source ``snapshot`` its run took
    (``SourceLayout.snapshot``), and the ``geometry`` its run read off its training frames.
    """

    model_config = ConfigDict(frozen=True)

    config: dict[str, JsonValue] = Field(
        validation_alias=CONFIG_KEY, serialization_alias=CONFIG_KEY
    )
    snapshot: str = Field(
        validation_alias=SNAPSHOT_KEY, serialization_alias=SNAPSHOT_KEY
    )
    geometry: TrainGeometry = Field(
        validation_alias=GEOMETRY_KEY, serialization_alias=GEOMETRY_KEY
    )


class Producer(BaseModel):
    """The checkpoint and the run behind a model's predictions.

    It is what :attr:`VerifiedCheckpoint.producer` states.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint_sha256: str
    experiment_id: str | None


@dataclass(frozen=True)
class VerifiedCheckpoint:
    """A checkpoint read, hashed and matched against the registry before anything was unpickled."""

    path: str
    """The established location the checkpoint was read from."""
    payload: dict
    """The loaded checkpoint, read with ``weights_only=True``."""
    entry: RegistryEntry
    """The one registered entry of the digest of the bytes ``payload`` was unpickled from
    (:func:`registered_entries`)."""
    project: Path
    """The project the checkpoint was loaded for, the root its config's stored paths read
    against."""

    @property
    def sha256(self) -> str:
        """The digest of the exact bytes ``payload`` was unpickled from, its entry's."""
        return self.entry.sha256

    @property
    def experiment_id(self) -> str | None:
        """The run whose final status names this checkpoint, ``None`` for a foreign one."""
        return self.entry.experiment_id

    @property
    def producer(self) -> Producer:
        """The checkpoint's producer identity: its sha256 and the run behind it."""
        return Producer(
            checkpoint_sha256=self.sha256, experiment_id=self.experiment_id
        )

    @cached_property
    def stamp(self) -> CheckpointStamp:
        """The payload's :class:`CheckpointStamp`, validated once.

        Refuses (``ValueError``) a payload that does not carry it, naming each key it lacks or
        holds invalid and the stamp's writer.
        """
        try:
            return CheckpointStamp.model_validate(self.payload)
        except ValidationError as exc:
            keys = ", ".join(sorted({repr(e["loc"][0]) for e in exc.errors()}))
            raise ValueError(
                f"{self.path} carries no {keys} as generic_trainer.checkpoint_stamp writes "
                "it, as the default trainer and ctx.save_checkpoint save it, so it was written "
                "past that stamp. Save it through a run (launch_training, or "
                f"ctx.save_checkpoint in its training body). {exc}"
            ) from exc

    @cached_property
    def spec(self) -> TrainConfigSchema:
        """The run config the checkpoint's :attr:`stamp` carries, validated once.

        Its stored paths (``experiments.CONFIG_PATHS``) are read against :attr:`project`, and it
        is validated by ``schemas.train_config``, which refuses (``ValueError``) one that is
        invalid: a config stating its ``model_source`` and ``data`` validates with no training
        regime. A source file it declares (``model_build.source_seams``') that does not exist
        refuses (``ValueError``) naming it and how to carry the run whole, before anything is
        imported.
        """
        from tcip_mcp.experiments import CONFIG_PATHS
        from tcip_mcp.pipelines.model_build import source_seams
        from tcip_mcp.pipelines.schemas import train_config

        spec = train_config(
            runtime_paths(self.stamp.config, CONFIG_PATHS, self.project)
        )
        absent = [
            file
            for _field, _dotted, files in source_seams(spec)
            for file in files or []
            if not Path(file).is_file()
        ]
        if absent:
            raise ValueError(
                f"{self.path} binds its source to {absent}, which {self.project} does not hold: a "
                "checkpoint carries its run's source snapshot and loads only beside that run. "
                "Load it in the project holding the run directory those paths name, or carry that "
                "run whole into this project (archive_project, then import_project)."
            )
        return spec

    @property
    def geometry(self) -> TrainGeometry:
        """The geometry its run read off its training frames, as its :attr:`stamp` records it."""
        return self.stamp.geometry

    @cached_property
    def layout(self) -> SourceLayout:
        """The run snapshot the checkpoint's model imports from.

        It is the one its :attr:`stamp`'s snapshot digest names, as the record of the run of
        :attr:`project` that took it states it (``model_build.owning_run_layout``, which refuses
        a config declaring no file, or a snapshot and files no run of :attr:`project` took).
        """
        from tcip_mcp.pipelines.model_build import owning_run_layout

        return owning_run_layout(self.spec, self.stamp.snapshot, self.project)

    @property
    def task(self) -> str:
        """The task the checkpoint's model is for, its validated config's ``model_source.task``.

        Raises ``ValueError`` for a config that does not validate.
        """
        return self.spec.model_source.task


class CheckpointNotFoundError(FileNotFoundError):
    """No file is at the location a checkpoint was named by."""


class UnregisteredCheckpointError(ValueError):
    """A checkpoint nothing registers, or one whose payload cannot be trusted to unpickle.

    Nothing registers it when no completed run and no registry entry names it; a payload that
    cannot be trusted is one that will not unpickle under ``weights_only=True``.
    """


def _load_verified_payload(data: bytes, *, source: str) -> dict:
    """Unpickle already identity-verified checkpoint bytes with ``weights_only=True``.

    ``source`` names the checkpoint (path and digest) in every raised message.

    Raises :class:`UnregisteredCheckpointError` for a payload that will not unpickle under
    ``weights_only=True``, and a bare ``ValueError`` for a payload that does not unpickle to a
    dict or carries no weights under ``model_build``'s ``STATE_DICT_KEY``.
    """
    import torch  # local checkpoint an identity check already named; unpickling it is the point

    from tcip_mcp.pipelines.model_build import STATE_DICT_KEY

    try:
        payload = torch.load(
            io.BytesIO(data), map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise UnregisteredCheckpointError(
            f"{source} could not be loaded with weights_only=True ({exc}): this payload carries "
            "something outside a platform-written deliverable checkpoint's contract (a resume "
            "checkpoint's RNG/optimizer state is the trainer's own resume path to read; a bespoke "
            "loop's own arbitrary state is outside the contract)."
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"{source} did not unpickle to a dict (got {type(payload).__name__}), so "
            "it carries none of the config and weights this platform's checkpoints "
            "do."
        )
    if STATE_DICT_KEY not in payload:
        raise ValueError(
            f"{source} carries no {STATE_DICT_KEY!r}: a checkpoint is admitted for "
            "the weights it loads, and this one holds none."
        )
    return payload


def admitted_digest(checkpoint_path: str | Path) -> str:
    """Return the sha256 of the checkpoint file at ``checkpoint_path``, read once.

    The digest is returned only once the verified reader (:func:`_load_verified_payload`) admits
    the payload of those same bytes; its refusals raise, and an unreadable file raises
    ``OSError``.
    """
    data = Path(checkpoint_path).read_bytes()
    digest = _sha256_of_bytes(data)
    _load_verified_payload(data, source=f"{checkpoint_path} (sha256 {digest})")
    return digest


def checkpoint_payload(checkpoint_path: str | Path, sha256: str) -> dict:
    """Return the payload of the checkpoint at ``checkpoint_path`` once its bytes match ``sha256``.

    The payload is read through :func:`_load_verified_payload` only once its bytes are the ones
    ``sha256`` names. Bytes hashing otherwise raise :class:`UnregisteredCheckpointError` naming
    both digests.
    """
    data = Path(checkpoint_path).read_bytes()
    digest = _sha256_of_bytes(data)
    if digest != sha256:
        raise UnregisteredCheckpointError(
            f"{checkpoint_path} now hashes to {digest}, not the {sha256} its record names: the "
            "file was replaced after it was recorded."
        )
    return _load_verified_payload(
        data, source=f"{checkpoint_path} (sha256 {digest})"
    )


def load_registered_checkpoint(
    checkpoint_path: Path, *, project: Path
) -> VerifiedCheckpoint:
    """Return the checkpoint at the established location ``checkpoint_path``, verified.

    It is read once and unpickled from the same bytes its digest is taken over, when
    ``project``'s :func:`registered_entries` name that digest.

    Raises :class:`UnregisteredCheckpointError` naming the path, the digest and the root searched
    when none does, and :class:`CheckpointNotFoundError` when no file is at the location.
    """
    ckpt = Path(checkpoint_path)
    root = str(project)
    try:
        data = ckpt.read_bytes()
    except FileNotFoundError as exc:
        raise CheckpointNotFoundError(f"Checkpoint not found: {ckpt}") from exc
    digest = _sha256_of_bytes(data)
    entry = next(
        (e for e in registered_entries(root) if e.sha256 == digest), None
    )
    if entry is None:
        raise _unregistered_checkpoint_error(ckpt, digest, root)
    payload = _load_verified_payload(data, source=f"{ckpt} (sha256 {digest})")
    return VerifiedCheckpoint(
        path=str(ckpt), payload=payload, entry=entry, project=Path(project)
    )


def _write_registry_entry(
    txn: tcip_store.Txn, key: Key, entry: RegistryEntry
) -> tuple[RegistryEntry | None, RegistryEntry]:
    """Write ``entry`` as the one foreign entry of its sha256 inside ``txn``.

    ``txn`` is an already-open transaction over the index key.

    Returns ``(superseded, stored)``: the entry of that sha256 replaced (``None`` for its first
    registration), and the entry the registry holds after the call, ``entry`` itself when it was
    written and the superseded entry when that already holds every field but ``registered_at``,
    which is no change and so no write.
    """
    index = _index_entries(txn.read(key, default={"entries": []}))
    superseded = next((e for e in index if e.sha256 == entry.sha256), None)
    if superseded is not None and entry == superseded.model_copy(
        update={"registered_at": entry.registered_at}
    ):
        return superseded, superseded
    index = [*(e for e in index if e.sha256 != entry.sha256), entry]
    txn.write(key, RegistryIndex(entries=index).model_dump(mode="json"))
    return superseded, entry


def _audit_entry_write(
    project_path: str,
    superseded: RegistryEntry | None,
    entry: RegistryEntry,
    actor: str | None,
) -> None:
    """Emit ``model_registered``, by ``actor``, in the project's log for a written ``entry``.

    The line names the entry of the same sha256 it superseded, if any. A failed append raises
    ``AuditEntryNotWrittenError``.
    """
    from tcip_mcp.audit import record_event_or_raise

    replaced: dict[str, JsonValue] = (
        {}
        if superseded is None
        else {
            "superseded_name": superseded.name,
            "superseded_tags": list(superseded.tags),
        }
    )
    record_event_or_raise(
        "model_registered",
        {"name": entry.name, "new_sha256": entry.sha256, **replaced},
        actor=actor,
        scope=project_path,
    )


class Listed(BaseModel):
    """A registered ``entry`` as listed, its ``checkpoint_path`` resolved, beside its ``facts``.

    ``facts`` are what its checkpoint says of it (:func:`entry_facts`). It serializes flat: the
    entry's fields with the facts' ``metrics`` and ``metrics_source`` laid over them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    entry: RegistryEntry
    facts: CheckpointFacts

    @model_serializer(mode="wrap")
    def _flat(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Return the entry's serialized fields with the facts' laid over them."""
        dumped = handler(self)
        return dumped["entry"] | dumped["facts"]


def _listed(project_path: str, entry: RegistryEntry) -> Listed:
    """Return ``entry`` as listed, its ``checkpoint_path`` resolved to an absolute path.

    A stored path the resolver refuses (empty, or a relative form carrying ``..``) raises its
    refusal (:func:`~tcip_mcp.registry_paths.resolved_registry_path`).
    """
    resolved = entry.model_copy(
        update={
            "checkpoint_path": str(
                resolved_registry_path(project_path, entry.checkpoint_path)
            )
        }
    )
    return Listed(
        entry=resolved, facts=entry_facts(resolved, Path(project_path))
    )


class ModelRegistry:
    """A project's registered checkpoints, and the explicit-mode door that registers a foreign one.

    The checkpoints are :func:`registered_entries`, each read at the call that asks.
    """

    def __init__(self, project_path: str) -> None:
        """Take the project root whose registry this reads and writes."""
        self._project_path = project_path

    def register_model(
        self,
        name: str,
        checkpoint_path: str,
        *,
        metrics: dict | None = None,
        tags: list[str] | None = None,
        actor: str | None,
    ) -> Listed:
        """Register a checkpoint by its sha256 under ``name``, and return the owner of its bytes.

        The digest is :func:`admitted_digest`'s and ``name`` is the entry's presentation; the
        write is recorded as ``actor``'s, and the owner is the one :func:`registered_entries`
        resolves those bytes to: this entry, or the completed run whose final status names the
        same bytes. A second registration of the same bytes replaces the first. Its ``metrics``
        are asserted by the caller, never verified. The stored ``checkpoint_path`` is spelled
        through :func:`~tcip_mcp.registry_paths.checkpoint_registry_path_for` against the
        project root.

        Args:
            name: Model name (e.g. '<crop>_<trait>_detector_v1').
            checkpoint_path: Path to the .pt checkpoint file
                (:func:`~tcip_mcp.registry_paths.located` against the project); its config is the
                one its payload carries (:attr:`VerifiedCheckpoint.spec`).
            metrics: Evaluation metrics, a ``tcip_store.FiniteJsonObject``; omitted (``None``),
                the entry carries none.
            tags: Optional tags for filtering.

        Raises:
            FileNotFoundError: ``checkpoint_path`` does not exist.
            pydantic.ValidationError: ``metrics`` is given and is not a
                ``tcip_store.FiniteJsonObject``, named before anything is stored.
            ValueError: the checkpoint's payload is one the verified reader refuses.
            AuditEntryNotWrittenError: the write committed but its own audit line could not be
                appended.

        """
        ckpt = located(checkpoint_path, self._project_path)
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"register_model: checkpoint_path {checkpoint_path!r} does not exist, refusing to "
                "register a phantom registry entry."
            )
        entry = RegistryEntry(
            name=name,
            checkpoint_path=checkpoint_registry_path_for(
                ckpt, self._project_path
            ),
            sha256=admitted_digest(ckpt),
            registered_at=now_iso(),
            metrics={} if metrics is None else metrics,
            tags=tags or [],
            experiment_id=None,
        )
        key = registry_index_key(self._project_path)
        with tcip_store.transaction(key) as txn:
            superseded, stored = _write_registry_entry(txn, key, entry)
        if stored is entry:
            _audit_entry_write(self._project_path, superseded, entry, actor)
        owner = next(
            e
            for e in registered_entries(self._project_path)
            if e.sha256 == entry.sha256
        )
        return _listed(self._project_path, owner)

    def list_models(self, tag: str | None = None) -> list[Listed]:
        """Return every registered checkpoint carrying ``tag``, every one when ``tag`` is ``None``.

        Each is listed (:func:`_listed`).
        """
        return [
            _listed(self._project_path, m)
            for m in registered_entries(self._project_path)
            if tag is None or tag in m.tags
        ]


def verified(entry: Listed) -> bool:
    """Return whether a registry entry's metrics were measured by the trainer, not asserted."""
    return entry.facts.metrics_source == "trainer"


def best_model(
    models: list[Listed],
    metric_key: str,
    *,
    higher_is_better: bool,
    include_unverified: bool = False,
) -> Listed | None:
    """Return the one of ``models`` with the best finite ``metric_key`` value, or ``None``.

    ``models`` are :meth:`ModelRegistry.list_models`'s entries; best is in the stated direction,
    and ``None`` is returned when none carries it. Only a :func:`verified` entry is ranked unless
    ``include_unverified``.
    """
    best = None
    best_val: float | None = None
    for m in models:
        if not (include_unverified or verified(m)):
            continue
        val = m.facts.metrics.get(metric_key)
        if not finite_number(val):
            continue
        if best_val is None or (
            val > best_val if higher_is_better else val < best_val
        ):
            best_val = float(val)
            best = m
    return best
