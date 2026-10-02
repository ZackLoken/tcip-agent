"""Delivering a measurement: the one gate every delivery function calls, the result a breeder's
recorded acknowledgment binds to, and the one writer each delivers its CSV and delivery
event through.

The gate reads each bucket's record and the assessment it was published under, each assessment
once, and refuses outright what no acknowledgment can ship; a delivery with any other finding
ships only under an acknowledgment recorded for the digest of exactly the rows and disclosure the
writer is about to write.
"""

from __future__ import annotations

import csv
import hashlib
import io
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tcip_store
from pydantic import TypeAdapter
from tcip_store import RECORD_JSON, Key, StoreDescriptor, register_store
from tcip_store.file_backend import RootedFileLocator

from tcip_mcp.pipelines.delivery_events_schema import (
    Acknowledgment, BucketFinding, DeliveryEventRecord, Producer,
)
from tcip_mcp.project_paths import project_state_dir
from tcip_mcp.registry_paths import PathFields, recorded_paths, runtime_paths

_DISCLOSURE: TypeAdapter[Any] = TypeAdapter(
    DeliveryEventRecord.model_fields["plant_mapping"].annotation)

if TYPE_CHECKING:
    from tcip_mcp.assessment import Assessment
    from tcip_mcp.buckets import Bucket
    from tcip_mcp.traits import TraitRevision


class DeliveryRefused(ValueError):
    """A delivery the gate refused, one sentence per reason. ``result_sha256`` is the digest of
    the result an acknowledgment would ship (:func:`result_digest`), ``None`` for a delivery no
    acknowledgment can ship."""

    def __init__(self, message: str, result_sha256: str | None = None) -> None:
        super().__init__(message)
        self.result_sha256 = result_sha256

    def as_detail(self) -> dict[str, Any]:
        """The structured refusal body a web door raises."""
        return {"kind": "delivery", "message": str(self), "result_sha256": self.result_sha256}


DELIVERY_COLUMNS = ("validated", "trait", "trait_revision", "trait_revision_sha256",
                    "delivery_event_id")
"""The columns every delivered CSV carries beside its measurement, the same on every row."""


@dataclass(frozen=True)
class Clearance:
    """What the gate found: whether every bucket is validated, each bucket's own finding, the one
    producer, the scale assessment, and the acknowledgment the delivery ships under when it is not
    validated."""

    validated: bool
    buckets: list[BucketFinding]
    producer: Producer
    scale_assessment_id: str | None
    acknowledgment: Acknowledgment | None


@dataclass(frozen=True)
class Result:
    """A delivery's measured result and disclosure as it is shown and written: the CSV's columns
    and rows, its population, its missingness rule and its plant-mapping disclosure."""

    columns: Sequence[str]
    rows: Sequence[dict]
    population: Sequence[str]
    require_all_dates_complete: bool | None = None
    plant_mapping: dict | None = None

    def encoded(self, cells: dict[str, Any]) -> bytes:
        """The CSV of the rows under the columns, each row carrying ``cells`` beside its own."""
        text = io.StringIO(newline="")
        writer = csv.DictWriter(text, fieldnames=list(self.columns), extrasaction="ignore")
        writer.writeheader()
        for row in self.rows:
            writer.writerow({**row, **cells})
        return text.getvalue().encode("utf-8")


def _assessment_finding(project: Path, assessment: Assessment, *, revision: TraitRevision,
                        delivery_kind: str | None) -> str | None:
    """Why ``assessment`` answers for no delivery of ``delivery_kind`` under ``revision``, in one
    sentence, or ``None`` when it answers: it passed, at this kind, under this revision whole
    (number and content hash), over a reference that has not changed since."""
    from tcip_mcp.assessment import assessment_dir

    aid = assessment.assessment_id
    if not assessment.passed:
        return f"assessment {aid} did not pass ({', '.join(assessment.failures)})."
    if assessment.delivery_kind != delivery_kind:
        return (f"assessment {aid} measured a {assessment.delivery_kind} delivery, not a "
                f"{delivery_kind} one.")
    judged = assessment.revision
    if judged != revision.ref:
        return (f"assessment {aid} was judged under {judged['trait']!r} revision "
                f"{judged['trait_revision']} ({judged['trait_revision_sha256'][:12]}), "
                f"and this delivery ships under revision {revision.number} "
                f"({revision.entry_sha256[:12]}); assess again under the revision delivered.")
    moved = assessment.reference.moved(assessment_dir(project, aid))
    if moved:
        return (f"the reference assessment {aid} measured has changed since ({', '.join(moved)}), "
                "so the number it earned no longer answers for it; assess again.")
    return None


def _covers(assessment: Assessment, bucket: Bucket) -> bool:
    """Whether ``assessment``'s reference spans ``bucket``'s capture."""
    return assessment.reference.covers(dataset_id=bucket.dataset_id, date=bucket.date,
                                       raster_identity=bucket.raster_identity)


def _bucket_reason(bucket: Bucket, assessment: Assessment | None, finding: str | None,
                   ) -> str | None:
    """Why ``bucket`` is not validated, given the assessment it was published under (``None``
    for none or one that could not be read, ``finding`` then saying why) and that assessment's
    own :func:`_assessment_finding`; ``None`` when it is validated: the assessment answers, and it
    measured the bucket's own producer under its own execution record over a reference covering
    its capture."""
    if bucket.assessment_id is None:
        return (f"no assessment answers for {bucket.path}: assess its checkpoint against a "
                "held-out reference selection and publish its predictions under that "
                "assessment.")
    if assessment is None or finding is not None:
        return finding
    if (assessment.producer, assessment.execution) != (bucket.producer, bucket.execution):
        return (f"{bucket.path} was not produced by the checkpoint and execution record "
                f"assessment {assessment.assessment_id} measured.")
    if not _covers(assessment, bucket):
        return (f"assessment {assessment.assessment_id}'s reference does not cover "
                f"{bucket.path}'s capture ({bucket.dataset_id}, {bucket.date}).")
    return None


def _assessments(project: Path, ids: Sequence[str | None], *, revision: TraitRevision | None,
                 delivery_kind: str | None
                 ) -> dict[str | None, tuple[Assessment | None, str | None]]:
    """Each distinct assessment id among ``ids`` read once, beside its
    :func:`_assessment_finding` under ``revision`` (the assessment's own trait's latest confirmed
    revision for ``None``) at ``delivery_kind`` (its own kind for ``None``); an id naming none
    carries the sentence saying so, and ``None`` itself no assessment."""
    from tcip_mcp.assessment import read_assessment
    from tcip_mcp.operationalization import latest_confirmed
    from tcip_mcp.traits import TraitUnknownError

    out: dict[str | None, tuple[Assessment | None, str | None]] = {None: (None, None)}
    for aid in dict.fromkeys(ids):
        if aid is None:
            continue
        try:
            assessment = read_assessment(project, aid)
            finding = _assessment_finding(
                project, assessment,
                revision=revision or latest_confirmed(assessment.revision["trait"], project),
                delivery_kind=delivery_kind or assessment.delivery_kind)
        except (ValueError, TraitUnknownError) as exc:
            out[aid] = (None, str(exc))
            continue
        out[aid] = (assessment, finding)
    return out


def admitted_conf(project: Path, bucket: Bucket) -> tuple[float | None, str]:
    """The conf a review may accept ``bucket``'s predictions at on its assessment's authority: its
    execution record's conf when the assessment it was published under validates it at its own
    kind under that trait's latest confirmed revision, else ``None`` and the sentence saying
    why."""
    assessment, finding = _assessments(project, [bucket.assessment_id], revision=None,
                                       delivery_kind=None)[bucket.assessment_id]
    reason = _bucket_reason(bucket, assessment, finding)
    if reason is not None:
        return None, reason
    if bucket.execution is None or bucket.execution.conf is None:
        return None, f"{bucket.path} holds no detector's predictions, which no conf admits."
    return bucket.execution.conf, ""


def result_digest(result: Result, *, delivery_kind: str, revision: TraitRevision,
                  producer: Producer, buckets: Sequence[BucketFinding],
                  scale_finding: str | None) -> str:
    """The digest of the result a breeder's acknowledgment binds to: ``result``'s rows as the CSV
    writes them (without the delivery's own cells, which are receipt), its population,
    missingness rule and plant-mapping disclosure, beside the kind, the trait revision, the
    producer, each bucket's finding and the scale assessment's."""
    return hashlib.sha256(RECORD_JSON.encode({
        "rows": result.encoded({}).decode("utf-8"), "population": list(result.population),
        "require_all_dates_complete": result.require_all_dates_complete,
        "plant_mapping": _DISCLOSURE.dump_python(_DISCLOSURE.validate_python(
            result.plant_mapping), mode="json"),
        "delivery_kind": delivery_kind, **revision.ref, "producer": producer.model_dump(),
        "buckets": [b.model_dump() for b in buckets], "scale_finding": scale_finding,
    })).hexdigest()


def gate(
    project: Path, buckets: Sequence[Bucket], *, delivery_kind: str, revision: TraitRevision,
    result: Result, acknowledgment_id: str | None = None,
    scale_assessment_id: str | None = None, unit: str | None = None,
) -> Clearance:
    """Clear ``buckets`` for a ``delivery_kind`` delivery under ``revision``. ``unit`` is the
    linear physical unit the delivered values are in (``None`` for none); a detector delivery in a
    unit rests on the physical-scale assessment ``scale_assessment_id``, which must have passed in
    ``unit``, under ``revision``, over a reference that has not changed, covering every bucket's
    capture.

    Refuses (:class:`DeliveryRefused`) outright no buckets at all, buckets naming more than one
    producer or proposing rather than predicting, a state-crossing delivery over a bucket that
    classifies no positive state, a scale assessment named for a value in no unit, and a detector
    delivery in a unit naming none; refuses
    (:class:`~tcip_mcp.operationalization.OperationalizationRefused`) a detector delivery whose
    buckets do not count the operationalization's measured subject. A delivery with any
    unvalidated finding ships only under ``acknowledgment_id``, a recorded acknowledgment
    (:func:`read_acknowledgment`) of exactly ``result``, the rows and disclosure about to be
    written (:func:`result_digest`); otherwise it refuses with one sentence per finding, carrying
    the result's digest.
    """
    from tcip_mcp.operationalization import bind
    from tcip_mcp.traits import DETECTOR_KINDS, STATE_CROSSING_DATES

    if not buckets:
        raise DeliveryRefused("a delivery names no prediction bucket, so nothing states what its "
                              "numbers were measured from; name the published buckets it ships.")
    proposed = [str(b.path) for b in buckets if "checkpoint_sha256" not in b.producer]
    if proposed:
        raise DeliveryRefused(f"{proposed} hold staged proposals, which no model predicted and "
                              "no assessment measures: a delivery ships a model's predictions.")
    producers = {Producer.model_validate(b.producer) for b in buckets}
    if len(producers) > 1:
        raise DeliveryRefused(
            "the delivered buckets were produced by more than one checkpoint or run ("
            + "; ".join(f"{b.path}: {b.producer['checkpoint_sha256']} / "
                        f"{b.producer['experiment_id']}" for b in buckets)
            + "): one delivered series names one producer, so deliver the buckets one producer "
            "made.")
    if delivery_kind in DETECTOR_KINDS:
        bind(revision, delivery_kind, buckets={str(b.path): b.scope.subject for b in buckets})
    if delivery_kind == STATE_CROSSING_DATES:
        state = revision.entry.positive_state
        unclassified = [str(b.path) for b in buckets if b.scope.state_ids(state) is None]
        if unclassified:
            raise DeliveryRefused(
                f"{unclassified} classify no {state}: the classifier that produced them never "
                "assessed this trait's positive state, so a positive fraction over them is not a "
                "measurement.")
    if scale_assessment_id is not None and unit is None:
        raise DeliveryRefused("a physical scale answers for a dimensional value, and the "
                              "delivered values are in no physical unit.")
    if delivery_kind in DETECTOR_KINDS and unit is not None and scale_assessment_id is None:
        raise DeliveryRefused(f"the delivered values are in {unit!r} from a detection or "
                              "segmentation bucket, and nothing answers for that unit: name the "
                              "physical-scale assessment (calibrate_physical_scale) they were "
                              "scaled under.")
    found = _assessments(project, [b.assessment_id for b in buckets], revision=revision,
                         delivery_kind=delivery_kind)
    rows = []
    for bucket in buckets:
        reason = _bucket_reason(bucket, *found[bucket.assessment_id])
        rows.append(BucketFinding(path=str(bucket.path), date=bucket.date,
                                  assessment_id=bucket.assessment_id, validated=reason is None,
                                  reason=reason))
    scale_finding = None
    if scale_assessment_id is not None:
        scale, scale_finding = _assessments(project, [scale_assessment_id], revision=revision,
                                            delivery_kind=None)[scale_assessment_id]
        if scale is not None and scale_finding is None:
            assert scale.scale is not None, "a physical-scale assessment records its scale"
            if scale.scale["unit"] != unit:
                scale_finding = (f"scale assessment {scale_assessment_id} is in "
                                 f"{scale.scale['unit']!r}, and this delivery is in {unit!r}.")
            elif not all(_covers(scale, b) for b in buckets):
                scale_finding = (f"scale assessment {scale_assessment_id}'s reference does not "
                                 "cover every delivered capture.")
    producer = next(iter(producers))
    unvalidated = [*(r.reason for r in rows if r.reason is not None),
                   *([scale_finding] if scale_finding is not None else [])]
    acknowledgment = None
    if unvalidated:
        digest = result_digest(result, delivery_kind=delivery_kind, revision=revision,
                               producer=producer, buckets=rows, scale_finding=scale_finding)
        if acknowledgment_id is None:
            raise DeliveryRefused(" ".join(unvalidated), result_sha256=digest)
        acknowledgment = read_acknowledgment(project, acknowledgment_id)
        if acknowledgment.result_sha256 != digest:
            raise DeliveryRefused(
                f"acknowledgment {acknowledgment_id} was given for another result than the one "
                f"this delivery computed: {' '.join(unvalidated)}", result_sha256=digest)
    return Clearance(validated=not unvalidated, buckets=rows, producer=producer,
                     scale_assessment_id=scale_assessment_id, acknowledgment=acknowledgment)


# ── the breeder's acknowledgment ────────────────────────────────────────────

ACKNOWLEDGMENTS_STORE = "delivery_acknowledgments"
register_store(
    StoreDescriptor(
        name=ACKNOWLEDGMENTS_STORE,
        kind="record",
        key_fields=("acknowledgment_id",),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="cas",
        enumerable=True,
        locator=RootedFileLocator(prefix=("delivery_acknowledgments",), suffix=".json"),
    )
)
"""One record per breeder acknowledgment of an unvalidated result, each written exactly once."""


def _acknowledgment_key(project: str | Path, acknowledgment_id: str) -> Key:
    return Key(ACKNOWLEDGMENTS_STORE, str(project_state_dir(project)), (acknowledgment_id,))


def record_acknowledgment(project: Path, *, acknowledged_by: str, reason: str,
                          result_sha256: str) -> Acknowledgment:
    """Record ``acknowledged_by``'s acknowledgment of shipping the unvalidated result
    ``result_sha256`` names (:func:`result_digest`), once, with its one audit line, ``delivery_acknowledged``, in the
    project's log; a blank name or reason refuses (``ValueError``)."""
    from tcip_mcp.audit import record_event_or_raise

    acknowledgment = Acknowledgment(
        acknowledgment_id=uuid.uuid4().hex, acknowledged_by=acknowledged_by, reason=reason,
        result_sha256=result_sha256, recorded_at=datetime.now(timezone.utc).isoformat())
    tcip_store.replace(_acknowledgment_key(project, acknowledgment.acknowledgment_id),
                       acknowledgment.model_dump(), expect=tcip_store.Version.ABSENT)
    record_event_or_raise("delivery_acknowledged",
                          {"acknowledgment_id": acknowledgment.acknowledgment_id,
                           "acknowledged_by": acknowledged_by, "result_sha256": result_sha256},
                          scope=project)
    return acknowledgment


def read_acknowledgment(project: Path, acknowledgment_id: str) -> Acknowledgment:
    """The acknowledgment ``acknowledgment_id`` names; one recorded under no such id refuses
    (:class:`DeliveryRefused`)."""
    raw = tcip_store.read(_acknowledgment_key(project, acknowledgment_id), default=None)
    if raw is None:
        raise DeliveryRefused(f"no acknowledgment {acknowledgment_id!r} is recorded: a breeder "
                              "acknowledges shipping an unvalidated result in the Results tab.")
    return Acknowledgment.model_validate(raw)


# ── the delivery event ──────────────────────────────────────────────────────

DELIVERY_EVENTS_STORE = "delivery_events"
register_store(
    StoreDescriptor(
        name=DELIVERY_EVENTS_STORE,
        kind="record",
        key_fields=("event_id",),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="last_writer_wins",
        enumerable=True,
        locator=RootedFileLocator(prefix=("delivery_events",), suffix=".json"),
    )
)
"""One record per completed delivery, keyed by its own id, each written exactly once."""

DELIVERY_EVENT_PATHS: PathFields = (
    ("output_path",), ("plant_mapping", "dataset_root"), ("buckets", "[]", "path"))
"""The fields of a delivery event record that name a path, stored against its project."""


def delivery_event_key(project: str | Path, event_id: str) -> Key:
    """One delivery event's record."""
    return Key(DELIVERY_EVENTS_STORE, str(project_state_dir(project)), (event_id,))


def deliver_csv(
    project: Path, output_path: str | Path, result: Result, *, clearance: Clearance,
    revision: TraitRevision, door: str, delivery_kind: str,
) -> dict[str, Any]:
    """Compose ``result`` as a CSV (each row carrying the :data:`DELIVERY_COLUMNS` cells of this
    delivery) and its one delivery event (the door, the kind, the trait revision it ships under,
    the delivered file and its sha256, the producer, each bucket's finding, the scale assessment,
    whether it is validated, the acknowledgment it ships under, and the result's population,
    missingness rule and plant-mapping disclosure), validating the event before anything is
    written; then write the CSV at ``output_path``, the event, and its one audit line,
    ``delivery_event``, in the log of the dataset the buckets sit in (the project's when they sit
    in none). Returns what was delivered: ``csv_path``, ``validated``, ``delivery_event_id``,
    ``trait_revision``, ``producer`` and ``acknowledged_by``."""
    from tcip_mcp.audit import record_event_or_raise
    from tcip_mcp.subject_registry import distinct_dataset_root

    event_id = uuid.uuid4().hex
    data = result.encoded(
        {"validated": clearance.validated, **revision.ref, "delivery_event_id": event_id})
    path = Path(output_path)
    event = DeliveryEventRecord.model_validate({
        "event_id": event_id, "door": door, "delivery_kind": delivery_kind, **revision.ref,
        "output_path": str(path), "output_sha256": hashlib.sha256(data).hexdigest(),
        "producer": clearance.producer, "buckets": clearance.buckets,
        "scale_assessment_id": clearance.scale_assessment_id, "validated": clearance.validated,
        "acknowledgment": clearance.acknowledgment, "population": list(result.population),
        "require_all_dates_complete": result.require_all_dates_complete,
        "plant_mapping": result.plant_mapping, "produced_at": datetime.now(timezone.utc).isoformat()})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    tcip_store.replace(delivery_event_key(project, event_id),
                       recorded_paths(event.model_dump(mode="json"), DELIVERY_EVENT_PATHS, project))
    record_event_or_raise("delivery_event", {"event_id": event_id},
                          scope=distinct_dataset_root([b.path for b in clearance.buckets])
                          or project)
    ack = clearance.acknowledgment
    return {"csv_path": str(path), "validated": clearance.validated,
            "delivery_event_id": event_id, "trait_revision": revision.number,
            "producer": clearance.producer.model_dump(),
            "acknowledged_by": ack.acknowledged_by if ack is not None else None}


def read_delivery_events(project: str | Path) -> list[DeliveryEventRecord]:
    """Every delivery event recorded under ``project``, decoded through its declared shape, its
    paths resolved against the project."""
    scope = project_state_dir(project)
    return [DeliveryEventRecord.model_validate(
                runtime_paths(tcip_store.read(key), DELIVERY_EVENT_PATHS, project))
            for key in tcip_store.keys(DELIVERY_EVENTS_STORE, str(scope))]
