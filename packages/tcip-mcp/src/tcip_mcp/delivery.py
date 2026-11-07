"""Delivering a measurement: the gate, the result an acknowledgment binds to, and the writer.

The writer writes a delivery's CSV and its delivery event; the acknowledgment is the breeder's,
recorded. The gate reads each bucket's record and the assessment it was published under, each
assessment once, and refuses outright what no acknowledgment can ship; a delivery with any other
finding ships only under an acknowledgment recorded for the digest of exactly the rows and
disclosure the writer is about to write.
"""

from __future__ import annotations

import csv
import hashlib
import io
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

import tcip_store
from tcip_mcp.assessment import (
    Assessment,
    CheckpointAssessment,
    assessment_dir,
    read_assessment,
)
from tcip_mcp.audit import append_or_raise, audit_line
from tcip_mcp.model_registry import Producer
from tcip_mcp.pipelines.delivery_events_schema import (
    Acknowledgment,
    BucketFinding,
    DeliveryEventRecord,
    PlantMapping,
    Shipped,
    ShippedAcknowledged,
    ShippedValidated,
    UnvalidatedBucket,
    ValidatedBucket,
)
from tcip_mcp.project_paths import project_state_dir
from tcip_mcp.registry_paths import PathFields, recorded_paths, runtime_paths
from tcip_store import Key, encode_record

_DISCLOSURE: TypeAdapter[PlantMapping | None] = TypeAdapter(
    PlantMapping | None
)

if TYPE_CHECKING:
    from tcip_mcp.buckets import Bucket
    from tcip_mcp.traits import TraitRevision


class Acknowledgeable(BaseModel):
    """A refused delivery an acknowledgment can ship: the digest of the result it would ship."""

    kind: Literal["acknowledgeable"] = "acknowledgeable"
    result_sha256: str


class Unacknowledgeable(BaseModel):
    """A refused delivery no acknowledgment can ship."""

    kind: Literal["unacknowledgeable"] = "unacknowledgeable"


type Shipping = Annotated[
    Acknowledgeable | Unacknowledgeable, Field(discriminator="kind")
]
"""Whether an acknowledgment can ship a refused delivery, told apart by its ``kind``."""


class DeliveryRefusedError(ValueError):
    """A delivery the gate refused, one sentence per reason, and whether an acknowledgment ships it.

    ``shipping`` is the digest of the result an acknowledgment would ship (:func:`result_digest`),
    or that none can; a refusal stating no digest is one no acknowledgment can ship.
    """

    def __init__(
        self, message: str, shipping: Shipping = Unacknowledgeable()
    ) -> None:
        """Take the refusal's message and whether an acknowledgment ships the delivery."""
        super().__init__(message)
        self.shipping = shipping

    def as_detail(self) -> dict[str, Any]:
        """Return the structured refusal body, :class:`DeliveryRefusal` dumped."""
        return DeliveryRefusal(
            message=str(self), shipping=self.shipping
        ).model_dump()


class DeliveryRefusal(BaseModel):
    """A delivery door's refusal: the gate's sentence and whether an acknowledgment ships it."""

    kind: Literal["delivery"] = "delivery"
    message: str
    shipping: Shipping


class ProjectionDigests(BaseModel):
    """Each phenology projection's digest a breeder's acknowledgment of exporting it binds to."""

    curves: str
    milestones: str


class PhenologyCleared(BaseModel):
    """A phenology measurement the delivery gate cleared."""

    kind: Literal["cleared"] = "cleared"


class PhenologyRefusal(BaseModel):
    """The gate's refusal of a phenology measurement shown unvalidated, with each digest."""

    kind: Literal["shown_unvalidated"] = "shown_unvalidated"
    message: str
    digests: ProjectionDigests


type PhenologyValidation = Annotated[
    PhenologyCleared | PhenologyRefusal, Field(discriminator="kind")
]
"""What the delivery gate found of a phenology measurement, told apart by its ``kind``."""


class ConfAdmitted(BaseModel):
    """The conf a bucket's assessment admits its predictions at."""

    kind: Literal["conf_admitted"] = "conf_admitted"
    conf: float


class ConfRefused(BaseModel):
    """A bucket whose predictions no conf is admitted for, and why not."""

    kind: Literal["conf_refused"] = "conf_refused"
    reason: str


type OperatingPoint = Annotated[
    ConfAdmitted | ConfRefused, Field(discriminator="kind")
]
"""What a bucket's assessment admits its predictions at, told apart by its ``kind``."""


DELIVERY_COLUMNS = (
    "validated",
    "trait",
    "trait_revision",
    "trait_revision_sha256",
    "delivery_event_id",
)
"""The columns every delivered CSV carries beside its measurement, the same on every row."""


@dataclass(frozen=True)
class Clearance:
    """What the gate found.

    It holds each bucket's own finding, the buckets' producer, the scale assessment, and how the
    delivery ships: validated, or under an acknowledgment.
    """

    buckets: list[BucketFinding]
    producer: Producer
    scale_assessment_id: str | None
    shipped: Shipped


@dataclass(frozen=True)
class Result:
    """A delivery's measured result and disclosure.

    It holds the CSV's columns and rows, its population and its plant-mapping disclosure.
    """

    columns: Sequence[str]
    rows: Sequence[dict]
    population: Sequence[str]
    plant_mapping: PlantMapping | None = None

    def encoded(self, cells: dict[str, Any]) -> bytes:
        """Return the CSV of the rows under the columns.

        Each row carries ``cells`` beside its own.
        """
        text = io.StringIO(newline="")
        writer = csv.DictWriter(
            text, fieldnames=list(self.columns), extrasaction="ignore"
        )
        writer.writeheader()
        for row in self.rows:
            writer.writerow({**row, **cells})
        return text.getvalue().encode("utf-8")


@dataclass(frozen=True)
class _Answers:
    """An assessment read that answers under the revision it was read under."""

    assessment: Assessment
    kind: Literal["answers"] = "answers"


@dataclass(frozen=True)
class _Validates:
    """The checkpoint's assessment validating a bucket."""

    assessment: CheckpointAssessment
    kind: Literal["validates"] = "validates"


@dataclass(frozen=True)
class _Unanswered:
    """Why an assessment id answers for nothing, its assessment read or not."""

    reason: str
    kind: Literal["unanswered"] = "unanswered"


def _assessment_finding(
    project: Path, assessment: Assessment, *, revision: TraitRevision
) -> _Read:
    """Return whether ``assessment`` answers under ``revision``, or why not in one sentence.

    It answers when it passed, under this revision whole (number and content hash), over a
    reference that has not changed since.
    """
    aid = assessment.assessment_id
    if not assessment.passed:
        return _Unanswered(
            f"assessment {aid} did not pass ({', '.join(assessment.failures)})."
        )
    judged = assessment.revision
    if judged != revision.ref:
        return _Unanswered(
            f"assessment {aid} was judged under {judged.trait!r} revision "
            f"{judged.trait_revision} ({judged.trait_revision_sha256[:12]}), "
            f"and this delivery ships under revision {revision.number} "
            f"({revision.entry_sha256[:12]}); assess again under the revision delivered."
        )
    moved = assessment.reference.moved(assessment_dir(project, aid))
    if moved:
        return _Unanswered(
            f"the reference assessment {aid} measured has changed since ({', '.join(moved)}), "
            "so the number it earned no longer answers for it; assess again."
        )
    return _Answers(assessment)


type _Read = Annotated[_Answers | _Unanswered, Field(discriminator="kind")]
"""An assessment read that answers, or why its id answers for nothing."""

_SCALE: TypeAdapter[dict[str, _Read]] = TypeAdapter(dict[str, _Read])


def _at_kind(read: _Read, delivery_kind: str) -> _Read:
    """Return ``read``, or why its assessment measured something else than a ``delivery_kind``."""
    if read.kind == "unanswered":
        return read
    assessment = read.assessment
    if assessment.kind == "physical_scale":
        measured = "a physical scale"
    elif assessment.delivery_kind == delivery_kind:
        return read
    else:
        measured = f"a {assessment.delivery_kind} delivery"
    return _Unanswered(
        f"assessment {assessment.assessment_id} measured {measured}, not a "
        f"{delivery_kind} one."
    )


def _at_scale(
    read: _Read, *, unit: str | None, buckets: Sequence[Bucket]
) -> _Read:
    """Return ``read``, or why its assessment answers for no scale in ``unit`` over ``buckets``.

    It answers when it is a physical scale in ``unit`` whose reference covers every bucket.
    """
    if read.kind == "unanswered":
        return read
    scale = read.assessment
    aid = scale.assessment_id
    if scale.kind != "physical_scale":
        return _Unanswered(
            f"assessment {aid} measured a {scale.delivery_kind} delivery, not a "
            "physical scale."
        )
    if scale.scale["unit"] != unit:
        return _Unanswered(
            f"scale assessment {aid} is in {scale.scale['unit']!r}, and this "
            f"delivery is in {unit!r}."
        )
    if not all(scale.reference.covers(b) for b in buckets):
        return _Unanswered(
            f"scale assessment {aid}'s reference does not cover every delivered "
            "capture."
        )
    return read


def _validating(
    bucket: Bucket, found: Mapping[str, _Read]
) -> _Validates | _Unanswered:
    """Return the assessment validating ``bucket``, or why none does.

    ``found`` holds the assessment it was published under, as :func:`_assessments` read it. It
    validates when it answers and measured the bucket's own producer under its own execution
    record over a reference covering its capture.
    """
    aid = bucket.assessment_id
    if aid is None:
        return _Unanswered(
            f"no assessment answers for bucket {bucket.name!r}: assess its checkpoint "
            "against a held-out reference selection and publish its predictions under that "
            "assessment."
        )
    read = found[aid]
    if read.kind == "unanswered":
        return read
    assessment = read.assessment
    if assessment.kind != "checkpoint" or (
        assessment.producer,
        assessment.execution,
    ) != (bucket.producer, bucket.execution):
        return _Unanswered(
            f"bucket {bucket.name!r} was not produced by the checkpoint and execution "
            f"record assessment {aid} measured."
        )
    if not assessment.reference.covers(bucket):
        return _Unanswered(
            f"assessment {aid}'s reference does not cover bucket {bucket.name!r}'s "
            f"capture ({bucket.dataset_id}, {bucket.date})."
        )
    return _Validates(assessment)


def _bucket_finding(
    bucket: Bucket, found: Mapping[str, _Read]
) -> BucketFinding:
    """Return what the gate finds for ``bucket``: validated (:func:`_validating`), or why not."""
    read = _validating(bucket, found)
    if read.kind == "validates":
        return ValidatedBucket(
            dataset_root=str(bucket.root),
            bucket=bucket.name,
            date=bucket.date,
            assessment_id=read.assessment.assessment_id,
        )
    return UnvalidatedBucket(
        dataset_root=str(bucket.root),
        bucket=bucket.name,
        date=bucket.date,
        assessment_id=bucket.assessment_id,
        reason=read.reason,
    )


def _assessments(
    project: Path,
    ids: Sequence[str | None],
    *,
    revision: TraitRevision | None,
) -> dict[str, _Read]:
    """Read each distinct assessment id among ``ids`` once, as :func:`_assessment_finding` finds it.

    The finding is under ``revision``, or the assessment's own trait's latest confirmed
    revision for ``None``; an id naming none is the sentence saying so, and ``None`` among
    ``ids`` is skipped.
    """
    from tcip_mcp.operationalization import latest_confirmed
    from tcip_mcp.traits import TraitUnknownError

    out: dict[str, _Read] = {}
    for aid in dict.fromkeys(ids):
        if aid is None:
            continue
        try:
            assessment = read_assessment(project, aid)
            out[aid] = _assessment_finding(
                project,
                assessment,
                revision=revision
                or latest_confirmed(assessment.revision.trait, project),
            )
        except (ValueError, TraitUnknownError) as exc:
            out[aid] = _Unanswered(str(exc))
    return out


def admitted_conf(project: Path, bucket: Bucket) -> OperatingPoint:
    """Return the conf ``bucket``'s assessment admits its predictions at, or why none is.

    The conf is the conf of the execution record the assessment it was published under measured,
    when that assessment validates it (:func:`_validating`) under that trait's latest confirmed
    revision.
    """
    read = _validating(
        bucket, _assessments(project, [bucket.assessment_id], revision=None)
    )
    if read.kind == "unanswered":
        return ConfRefused(reason=read.reason)
    conf = read.assessment.execution.conf
    if conf is None:
        return ConfRefused(
            reason=f"bucket {bucket.name!r} holds no detector's predictions, which no conf "
            "admits."
        )
    return ConfAdmitted(conf=conf)


def result_digest(
    result: Result,
    *,
    delivery_kind: str,
    revision: TraitRevision,
    producer: Producer,
    buckets: Sequence[BucketFinding],
    scale: dict[str, _Read],
) -> str:
    """Return the digest of the result a breeder's acknowledgment binds to.

    It covers ``result``'s rows as the CSV writes them (without the delivery's own cells, which
    are receipt), its population and plant-mapping disclosure, beside the kind, the trait
    revision, the producer, each bucket's finding and the named scale assessment's
    (:func:`_at_scale`), by its id.
    """
    return hashlib.sha256(
        encode_record(
            {
                "rows": result.encoded({}).decode("utf-8"),
                "population": list(result.population),
                "plant_mapping": _DISCLOSURE.dump_python(
                    result.plant_mapping, mode="json"
                ),
                "delivery_kind": delivery_kind,
                **revision.ref.model_dump(),
                "producer": producer.model_dump(),
                "buckets": [b.model_dump() for b in buckets],
                "scale": _SCALE.dump_python(scale, mode="json"),
            }
        )
    ).hexdigest()


def gate(
    project: Path,
    buckets: Sequence[Bucket],
    *,
    delivery_kind: str,
    revision: TraitRevision,
    result: Result,
    acknowledgment_id: str | None = None,
    scale_assessment_id: str | None = None,
    unit: str | None = None,
) -> Clearance:
    """Clear ``buckets`` for a ``delivery_kind`` delivery under ``revision``.

    ``unit`` is the linear physical unit the delivered values are in (``None`` for none); a
    detector delivery in a unit rests on the physical-scale assessment ``scale_assessment_id``,
    which must have passed in ``unit``, under ``revision``, over a reference that has not changed,
    covering every bucket's capture.

    Refuses (:class:`DeliveryRefusedError`) outright no buckets at all, buckets naming more than
    one producer or proposing rather than predicting, a state-crossing delivery over a bucket that
    classifies no positive state, a scale assessment named for a value in no unit, and a detector
    delivery in a unit naming none; refuses (``ValueError``) buckets under more than one dataset
    root (:func:`~tcip_mcp.buckets.shared_root`), and
    (:class:`~tcip_mcp.operationalization.OperationalizationRefusedError`) a detector delivery
    whose buckets do not count the operationalization's measured subject. A delivery with any
    unvalidated finding ships only under ``acknowledgment_id``, a recorded acknowledgment
    (:func:`read_acknowledgment`) of exactly ``result``, the rows and disclosure about to be
    written (:func:`result_digest`); otherwise it refuses with one sentence per finding, carrying
    the result's digest.
    """
    from tcip_mcp.buckets import shared_root
    from tcip_mcp.operationalization import bind
    from tcip_mcp.traits import DETECTOR_KINDS, STATE_CROSSING_DATES

    if not buckets:
        raise DeliveryRefusedError(
            "a delivery names no prediction bucket, so nothing states what its "
            "numbers were measured from; name the published buckets it ships."
        )
    shared_root(buckets)
    by_model = {
        b.name: b.producer for b in buckets if b.producer.kind == "checkpoint"
    }
    proposed = [b.name for b in buckets if b.name not in by_model]
    if proposed:
        raise DeliveryRefusedError(
            f"{proposed} hold staged proposals, which no model predicted and "
            "no assessment measures: a delivery ships a model's predictions."
        )
    producers = set(by_model.values())
    if len(producers) > 1:
        raise DeliveryRefusedError(
            "the delivered buckets were produced by more than one checkpoint or run ("
            + "; ".join(
                f"{name}: {p.checkpoint_sha256}"
                for name, p in by_model.items()
            )
            + "): one delivered series names one producer, so deliver the buckets one producer "
            "made."
        )
    if delivery_kind in DETECTOR_KINDS:
        bind(
            revision,
            delivery_kind,
            buckets={b.name: b.scope.subject for b in buckets},
        )
    if delivery_kind == STATE_CROSSING_DATES:
        state = revision.entry.positive_state
        unclassified = [
            b.name for b in buckets if b.scope.state_ids(state) is None
        ]
        if unclassified:
            raise DeliveryRefusedError(
                f"{unclassified} classify no {state}: the classifier that produced them never "
                "assessed this trait's positive state, so a positive fraction over them is not a "
                "measurement."
            )
    if scale_assessment_id is not None and unit is None:
        raise DeliveryRefusedError(
            "a physical scale answers for a dimensional value, and the "
            "delivered values are in no physical unit."
        )
    if (
        delivery_kind in DETECTOR_KINDS
        and unit is not None
        and scale_assessment_id is None
    ):
        raise DeliveryRefusedError(
            f"the delivered values are in {unit!r} from a detection or "
            "segmentation bucket, and nothing answers for that unit: name the "
            "physical-scale assessment (calibrate_physical_scale) they were "
            "scaled under."
        )
    found = {
        aid: _at_kind(read, delivery_kind)
        for aid, read in _assessments(
            project, [b.assessment_id for b in buckets], revision=revision
        ).items()
    }
    rows = [_bucket_finding(bucket, found) for bucket in buckets]
    scale = {
        aid: _at_scale(read, unit=unit, buckets=buckets)
        for aid, read in _assessments(
            project, [scale_assessment_id], revision=revision
        ).items()
    }
    producer = next(iter(producers))
    unvalidated = [
        *(r.reason for r in rows if r.kind == "unvalidated"),
        *(r.reason for r in scale.values() if r.kind == "unanswered"),
    ]
    shipped: Shipped = ShippedValidated()
    if unvalidated:
        digest = result_digest(
            result,
            delivery_kind=delivery_kind,
            revision=revision,
            producer=producer,
            buckets=rows,
            scale=scale,
        )
        if acknowledgment_id is None:
            raise DeliveryRefusedError(
                " ".join(unvalidated),
                Acknowledgeable(result_sha256=digest),
            )
        acknowledgment = read_acknowledgment(project, acknowledgment_id)
        if acknowledgment.result_sha256 != digest:
            raise DeliveryRefusedError(
                f"acknowledgment {acknowledgment_id} was given for another result than the one "
                f"this delivery computed: {' '.join(unvalidated)}",
                Acknowledgeable(result_sha256=digest),
            )
        shipped = ShippedAcknowledged(acknowledgment=acknowledgment)
    return Clearance(
        buckets=rows,
        producer=producer,
        scale_assessment_id=scale_assessment_id,
        shipped=shipped,
    )


# ── the breeder's acknowledgment ────────────────────────────────────────────

ACKNOWLEDGMENTS_STORE = "delivery_acknowledgments"
"""One record per breeder acknowledgment of an unvalidated result, each written exactly once."""


def _acknowledgment_key(project: str | Path, acknowledgment_id: str) -> Key:
    return Key(
        ACKNOWLEDGMENTS_STORE,
        str(project_state_dir(project)),
        (acknowledgment_id,),
    )


def record_acknowledgment(
    project: Path, *, acknowledged_by: str, reason: str, result_sha256: str
) -> Acknowledgment:
    """Record ``acknowledged_by``'s acknowledgment of shipping an unvalidated result, once.

    The result is the one ``result_sha256`` names (:func:`result_digest`); the acknowledgment's
    one audit line by that person, ``delivery_acknowledged``, lands in the project's log. A blank
    name or reason refuses (``ValueError``).
    """
    acknowledgment_id = uuid.uuid4().hex
    line = audit_line(
        "delivery_acknowledged",
        {
            "acknowledgment_id": acknowledgment_id,
            "result_sha256": result_sha256,
        },
        acknowledged_by,
        "ok",
    )
    acknowledgment = Acknowledgment(
        acknowledgment_id=acknowledgment_id,
        acknowledged_by=acknowledged_by,
        reason=reason,
        result_sha256=result_sha256,
        recorded_at=line.timestamp,
    )
    tcip_store.replace(
        _acknowledgment_key(project, acknowledgment_id),
        acknowledgment.model_dump(mode="json"),
        expect=tcip_store.Version.ABSENT,
    )
    append_or_raise(line, scope=project)
    return acknowledgment


def read_acknowledgment(
    project: Path, acknowledgment_id: str
) -> Acknowledgment:
    """Return the acknowledgment ``acknowledgment_id`` names.

    An id under which none is recorded refuses (:class:`DeliveryRefusedError`), and a record that
    does not validate raises ``tcip_store.ReadRefusedError`` naming the acknowledgment.
    """
    raw = tcip_store.read(
        _acknowledgment_key(project, acknowledgment_id), default=None
    )
    if raw is None:
        raise DeliveryRefusedError(
            f"no acknowledgment {acknowledgment_id!r} is recorded: a breeder "
            "acknowledges shipping an unvalidated result in the Results tab."
        )
    return tcip_store.validated(
        Acknowledgment, raw, f"acknowledgment {acknowledgment_id}"
    )


# ── the delivery event ──────────────────────────────────────────────────────

DELIVERY_EVENTS_STORE = "delivery_events"
"""One record per completed delivery, keyed by its own id, each written exactly once."""

DELIVERY_EVENT_PATHS: PathFields = (
    ("output_path",),
    ("plant_mapping", "dataset_root"),
    ("buckets", "[]", "dataset_root"),
)
"""The fields of a delivery event record that name a path, stored against its project."""


def delivery_event_key(project: str | Path, event_id: str) -> Key:
    """Return the delivery event's record key."""
    return Key(
        DELIVERY_EVENTS_STORE, str(project_state_dir(project)), (event_id,)
    )


class Delivered(BaseModel):
    """What a delivery shipped, each field its one delivery event's (:meth:`of`).

    It holds the delivered file, how it shipped, the event's id, the trait revision and the
    producer it shipped under, and its plant-mapping disclosure.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    output_path: str
    event_id: str
    shipped: Shipped
    trait_revision: int
    producer: Producer
    plant_mapping: PlantMapping | None

    @classmethod
    def of(cls, event: DeliveryEventRecord) -> Delivered:
        """Return what the delivery ``event`` records was delivered, each field under its name."""
        return cls.model_validate(event, from_attributes=True)


class PerImageCountsDelivered(Delivered):
    """A per-image count delivery: its image and detection totals."""

    image_count: int
    total_detections: int


class PerPlantDelivered(Delivered):
    """A per-plant delivery: what it shipped and how many plants its population holds."""

    n_plants: int


class OrthomosaicDelivered(PerPlantDelivered):
    """A per-plant count delivery from a whole raster, with the detections it attributed."""

    n_detections: int


class PhenologyDelivered(PerPlantDelivered):
    """A phenology projection's delivery, with its row count and its columns in order."""

    n_rows: int
    columns: list[str]


def deliver_csv(
    project: Path,
    output_path: str | Path,
    result: Result,
    *,
    clearance: Clearance,
    revision: TraitRevision,
    door: str,
    delivery_kind: str,
    actor: str | None,
) -> Delivered:
    """Write ``result`` as a CSV at ``output_path`` with its one delivery event and audit line.

    ``result`` is composed as a CSV (each row carrying the :data:`DELIVERY_COLUMNS` cells of this
    delivery) and its one delivery event (the door, the kind, the trait revision it ships under,
    the delivered file and its sha256, the producer, each bucket's finding, the scale assessment,
    how it shipped, and the result's population and plant-mapping disclosure), the event
    validated before anything is written; then the CSV is written at ``output_path``, the event,
    and its one audit line by ``actor``, ``delivery_event``, in the log of the dataset root the
    buckets are published under. Returns what the event records was delivered, the event read
    back from the store with its paths resolved against ``project`` as the event reader serves
    them.
    """
    event_id = uuid.uuid4().hex
    data = result.encoded(
        {
            "validated": clearance.shipped.kind == "validated",
            **revision.ref.model_dump(),
            "delivery_event_id": event_id,
        }
    )
    path = Path(output_path)
    line = audit_line("delivery_event", {"event_id": event_id}, actor, "ok")
    event = DeliveryEventRecord(
        **dict(revision.ref),
        event_id=event_id,
        door=door,
        delivery_kind=delivery_kind,
        output_path=str(path),
        output_sha256=hashlib.sha256(data).hexdigest(),
        producer=clearance.producer,
        buckets=clearance.buckets,
        scale_assessment_id=clearance.scale_assessment_id,
        shipped=clearance.shipped,
        population=list(result.population),
        plant_mapping=result.plant_mapping,
        produced_at=line.timestamp,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    tcip_store.replace(
        delivery_event_key(project, event_id),
        recorded_paths(
            event.model_dump(mode="json"), DELIVERY_EVENT_PATHS, project
        ),
    )
    append_or_raise(line, scope=clearance.buckets[0].dataset_root)
    return Delivered.of(
        _delivery_event(delivery_event_key(project, event_id), project)
    )


def read_delivery_events(project: str | Path) -> list[DeliveryEventRecord]:
    """Return every delivery event recorded under ``project``, decoded through its declared shape.

    Its paths are resolved against the project; the events stand in the order produced. An event
    its shape refuses, or one that will not decode or sits in a database that will not open or
    answer or that another holder keeps locked past the wait, raises
    ``tcip_store.ReadRefusedError`` naming the event's id: the shape's issues, or the store's
    sentence.
    """
    scope = str(project_state_dir(project))
    with tcip_store.unreadable(
        f"the delivery events under {scope}", *tcip_store.STORE_READ_REFUSALS
    ):
        keys = tcip_store.keys(DELIVERY_EVENTS_STORE, scope)
    return sorted(
        (_delivery_event(key, project) for key in keys),
        key=lambda event: event.produced_at,
    )


def _delivery_event(key: Key, project: str | Path) -> DeliveryEventRecord:
    """Return the delivery event ``key`` names, its paths resolved against ``project``."""
    where = f"delivery event {key.parts[0]}"
    return tcip_store.validated(
        DeliveryEventRecord,
        runtime_paths(
            tcip_store.read_raw(tcip_store.read, key, where),
            DELIVERY_EVENT_PATHS,
            project,
        ),
        where,
    )
