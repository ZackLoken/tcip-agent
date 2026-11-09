"""A delivery's event records the scale assessment the gate read, and why it could not be used.

A dimensional delivery names a physical-scale assessment. The gate reads it into an answer or a
reason, and the event it writes stores that read beside the bucket findings: the assessment's id
when it answers, its id and the gate's own sentence when it does not, nothing when the delivery
names none. The refusal and the event carry the same sentence.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import tcip_store
from tcip_mcp.assessment import (
    ASSESSMENT_FILE,
    assessment_dir,
    read_assessment,
)
from tcip_mcp.buckets import by_recorded_date
from tcip_mcp.delivery import (
    DeliveryRefusedError,
    delivery_event_key,
    read_delivery_events,
    record_acknowledgment,
)
from tcip_mcp.pipelines.postprocessing.aggregation import (
    deliver_per_plant_aggregate,
)
from tcip_mcp.pipelines.postprocessing.plant_mapping import (
    plant_mapping_disclosure,
    read_images,
)
from tcip_mcp.traits import PER_PLANT_COUNT_AGGREGATE
from tests import _trait_fixtures as fx
from tests._chain_fixtures import (
    SUBJECT,
    deliver_acknowledged,
    digest_of,
    mapped_results,
    unassessed_bucket,
)
from tests._mapping_fixtures import COUNT_CROP
from tests.test_physical_scale_assessment import (
    SUBJECT as REFERENCE_SUBJECT,
    _calibrate,
    _reference,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from tcip_mcp.buckets import Bucket
    from tcip_mcp.delivery import PerPlantDelivered
    from tcip_mcp.pipelines.delivery_events_schema import (
        DeliveryEventRecord,
        UnansweredScale,
    )
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        MappingResolution,
    )

TRAIT = "plant_surface_area"
PLANTS = ["PLANT_A"]
UNPASSING_EXTENTS = [5.0, 10.0, 15.0, 50.0, 5.0, 10.0, 15.0, 50.0]
"""Reference extents whose implied scales span a factor of ten, so the scale assessment fails."""
ACKNOWLEDGED_FOR = "the scale assessment is under review"


def _confirm_area(project: Path, subject: str) -> None:
    """Confirm the area trait with a per-plant operationalization of ``subject`` in mm2."""
    fx.seed_delivery_traits(project)
    fx.seed_confirmed_aggregate(
        project, TRAIT, value_keys=["area_mm2"], measured_subject=subject
    )


def _validated_bucket(project: Path) -> Bucket:
    """Return a bucket published under a passing per-plant assessment of the area trait.

    The area trait is confirmed first, measuring the chain's subject, and the captures are mapped
    to :data:`PLANTS` by the mapping ``valley``.
    """
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.tools.calibration_tools import assess_checkpoint
    from tcip_mcp.tools.inference_tools import run_inference
    from tests._chain_fixtures import (
        DATE,
        draw_reference_selection,
        plant_captures,
        synthetic_capture,
        train_on,
    )
    from tests._mapping_fixtures import map_plants
    from tests._training_values import (
        DISPERSION_QUANTILE,
        EQUIVALENCE_Z,
        RATIONALE,
        collection_pass,
    )

    _confirm_area(project, SUBJECT)
    root = project / "ds"
    images_dir = synthetic_capture(
        root, positions=plant_captures(dict.fromkeys(PLANTS, 1))
    )
    draw_reference_selection(project, root, project / "selection")
    checkpoint = train_on(project / "selection", project, "exp-scale-event")
    assessment = assess_checkpoint(
        project,
        checkpoint_path=checkpoint,
        trait=TRAIT,
        delivery_kind=PER_PLANT_COUNT_AGGREGATE,
        selection_dir=str(project / "selection"),
        equivalence_z=EQUIVALENCE_Z,
        dispersion_quantile=DISPERSION_QUANTILE,
        rationale=RATIONALE,
        stated=collection_pass(),
        actor=None,
    )
    assert read_assessment(project, assessment["assessment_id"]).passed, (
        assessment
    )
    bucket = f"validated/{DATE}"
    published = run_inference(
        project,
        checkpoint_path=checkpoint,
        images_dir=str(images_dir),
        bucket=bucket,
        assessment_id=assessment["assessment_id"],
        stated=Stated(),
        actor=None,
    )
    assert "error" not in published, published
    map_plants(project, root, PLANTS, crop=COUNT_CROP)
    return read_bucket(root, bucket)


def _scale(project: Path, extents_mm: list[float] | None = None) -> str:
    """Return the id of a physical-scale assessment over a reference dataset beside the others.

    The reference's bars are ``extents_mm`` long, else the fixed scale's.
    """
    selection, csv_path = _reference(
        project, name="-scale", extents_mm=extents_mm
    )
    return _calibrate(project, selection, csv_path)["assessment_id"]


def _areas(mapping: MappingResolution, plants: list[str]) -> list:
    """Return a 12.5 mm2 result for every image ``mapping``'s delivery reads of ``plants``."""
    return mapped_results(
        mapping,
        {
            plant: [12.5] * len(images)
            for plant, images in read_images(mapping, plants).items()
        },
    )


def _deliver(
    project: Path,
    bucket: Bucket,
    *,
    scale: str,
    acknowledgment: str | None = None,
    plants: list[str] = PLANTS,
) -> PerPlantDelivered:
    """Deliver the area of ``plants`` over ``bucket`` under the scale assessment ``scale``."""
    resolved = plant_mapping_disclosure(
        project, "valley", by_recorded_date([bucket]), plants
    )
    return deliver_per_plant_aggregate(
        project,
        _areas(resolved, plants),
        str(project / "area.csv"),
        value_key="area_mm2",
        delivered_phenotype=TRAIT,
        delivery_kind=PER_PLANT_COUNT_AGGREGATE,
        buckets=[bucket],
        plants=plants,
        mapping=resolved,
        door="test_scale_on_event",
        scale_assessment_id=scale,
        acknowledgment_id=acknowledgment,
        actor=None,
    )


def _refused(
    project: Path, bucket: Bucket, *, scale: str, plants: list[str] = PLANTS
) -> DeliveryRefusedError:
    """Return the refusal of delivering over ``bucket`` under ``scale`` with no acknowledgment."""
    with pytest.raises(DeliveryRefusedError) as refused:
        _deliver(project, bucket, scale=scale, plants=plants)
    return refused.value


def _acknowledged(project: Path, refusal: DeliveryRefusedError) -> str:
    """Return the id of a recorded acknowledgment of the result ``refusal`` says it would ship."""
    return record_acknowledgment(
        project,
        acknowledged_by="user:breeder",
        reason=ACKNOWLEDGED_FOR,
        result_sha256=digest_of(refusal),
    ).acknowledgment_id


def _delivered_acknowledged(
    project: Path, bucket: Bucket, *, scale: str
) -> tuple[DeliveryRefusedError, DeliveryEventRecord]:
    """Return the gate's refusal of a delivery and the event of it shipped under an acknowledgment.

    The acknowledged delivery goes through ``deliver_per_plant_csv``, the door an agent calls.
    """
    from tcip_mcp.tools.delivery_tools import deliver_per_plant_csv

    refusal = _refused(project, bucket, scale=scale)
    resolved = plant_mapping_disclosure(
        project, "valley", by_recorded_date([bucket]), PLANTS
    )
    answer = deliver_per_plant_csv(
        project,
        _areas(resolved, PLANTS),
        str(project / "area.csv"),
        TRAIT,
        PER_PLANT_COUNT_AGGREGATE,
        PLANTS,
        str(bucket.root),
        [bucket.name],
        value_key="area_mm2",
        plant_mapping="valley",
        scale_assessment_id=scale,
        acknowledgment_id=_acknowledged(project, refusal),
        actor=None,
    )
    assert "error" not in answer, answer
    (event,) = read_delivery_events(project)
    return refusal, event


def _unanswered(event: DeliveryEventRecord) -> UnansweredScale:
    """Return the event's scale read, which must be an unanswered one."""
    scale = event.scale
    assert scale is not None and scale.kind == "unanswered", scale
    return scale


def test_an_acknowledged_delivery_records_why_its_scale_assessment_did_not_pass(
    tmp_path: Path,
) -> None:
    """The event stores the failing scale assessment's id and the gate's sentence for it.

    The buckets validate, so the scale's reason is the delivery's only unvalidated finding.
    """
    bucket = _validated_bucket(tmp_path)
    scale = _scale(tmp_path, extents_mm=UNPASSING_EXTENTS)
    assert not read_assessment(tmp_path, scale).passed

    _, event = _delivered_acknowledged(tmp_path, bucket, scale=scale)

    assert event.shipped.kind == "acknowledged"
    assert [b.kind for b in event.buckets] == ["validated"]
    unanswered = _unanswered(event)
    assert unanswered.assessment_id == scale
    assert "did not pass" in unanswered.reason


def test_the_event_stores_the_sentence_the_refusal_gave_for_the_scale_assessment(
    tmp_path: Path,
) -> None:
    """Over validated buckets, the refusal's whole message is the event's scale reason."""
    bucket = _validated_bucket(tmp_path)
    scale = _scale(tmp_path, extents_mm=UNPASSING_EXTENTS)

    refusal, event = _delivered_acknowledged(tmp_path, bucket, scale=scale)

    unanswered = _unanswered(event)
    assert str(refusal) == unanswered.reason


def test_an_acknowledged_delivery_records_a_scale_assessment_that_will_not_read(
    tmp_path: Path,
) -> None:
    """A damaged scale assessment record is the event's scale reason, under the requested id."""
    bucket = _validated_bucket(tmp_path)
    scale = _scale(tmp_path)
    record = assessment_dir(tmp_path, scale) / ASSESSMENT_FILE
    record.write_bytes(b"\xff\xfe not json")

    refusal, event = _delivered_acknowledged(tmp_path, bucket, scale=scale)

    unanswered = _unanswered(event)
    assert unanswered.assessment_id == scale
    assert str(record) in unanswered.reason
    assert str(refusal) == unanswered.reason


def test_an_unvalidated_bucket_and_an_unanswered_scale_each_keep_their_own_reason(
    tmp_path: Path,
) -> None:
    """The refusal joins the bucket's reason and the scale's, and the event holds each apart."""
    _confirm_area(tmp_path, SUBJECT)
    bucket = unassessed_bucket(
        tmp_path, experiment_id="exp-mixed", images=dict.fromkeys(PLANTS, 1)
    )
    scale = _scale(tmp_path, extents_mm=UNPASSING_EXTENTS)
    refusal = _refused(tmp_path, bucket, scale=scale)

    _deliver(
        tmp_path,
        bucket,
        scale=scale,
        acknowledgment=_acknowledged(tmp_path, refusal),
    )

    (event,) = read_delivery_events(tmp_path)
    (finding,) = event.buckets
    assert finding.kind == "unvalidated", finding
    unanswered = _unanswered(event)
    assert unanswered.reason != finding.reason
    assert str(refusal) == f"{finding.reason} {unanswered.reason}"


def _answered_scene(project: Path) -> tuple[str, Bucket]:
    """Return a scale assessment that answers and an unassessed bucket its reference covers.

    The reference is the dataset ``ds`` of bars the scale is measured on, the bucket a publication
    of the first of its frames under no assessment, mapped to the plant ``p1``.
    """
    from tcip_mcp.dataset_layout import image_dir
    from tests._chain_fixtures import predicted, published
    from tests._mapping_fixtures import map_plants
    from tests.test_physical_scale_assessment import DATE

    _confirm_area(project, REFERENCE_SUBJECT)
    selection, csv_path = _reference(project)
    scale = _calibrate(project, selection, csv_path)["assessment_id"]
    assert read_assessment(project, scale).passed
    bucket = published(
        project,
        f"m/{DATE}",
        [
            predicted(
                image_dir(project / "ds", DATE) / "r0.png",
                [REFERENCE_SUBJECT],
            )
        ],
        scope={"subject": REFERENCE_SUBJECT},
    )
    map_plants(project, project / "ds", ["p1"], crop=COUNT_CROP)
    return scale, bucket


def test_an_acknowledgment_stops_binding_once_the_answering_scale_assessment_changes(
    tmp_path: Path,
) -> None:
    """The acknowledgment's digest covers the answered scale assessment's whole record.

    Beside an unvalidated bucket, so the acknowledgment stays necessary, a scale assessment that
    still answers but is recorded at another instant makes the earlier acknowledgment refuse as
    given for another result.
    """
    from tcip_mcp.experiments import read_record, write_once

    scale, bucket = _answered_scene(tmp_path)
    acknowledgment = _acknowledged(
        tmp_path, _refused(tmp_path, bucket, scale=scale, plants=["p1"])
    )
    record = assessment_dir(tmp_path, scale) / ASSESSMENT_FILE
    stored = read_record(record)
    assert isinstance(stored, dict)
    recorded = read_assessment(tmp_path, scale).recorded_at
    record.unlink()
    write_once(
        record,
        {
            **stored,
            "recorded_at": (recorded + timedelta(seconds=1)).isoformat(),
        },
    )
    assert read_assessment(tmp_path, scale).passed

    with pytest.raises(DeliveryRefusedError, match="given for another result"):
        _deliver(
            tmp_path,
            bucket,
            scale=scale,
            acknowledgment=acknowledgment,
            plants=["p1"],
        )


def test_a_delivery_naming_no_scale_assessment_records_none(
    tmp_path: Path,
) -> None:
    """A count delivery, which no unit makes dimensional, names no scale and stores none."""
    fx.seed_delivery_traits(tmp_path)
    fx.seed_confirmed_aggregate(
        tmp_path, "stem_count", value_keys=["count"], measured_subject="bud"
    )

    deliver_acknowledged(
        tmp_path,
        {"PLANT_A": [3]},
        tmp_path / "count.csv",
        "stem_count",
        value_key="count",
        delivery_kind=PER_PLANT_COUNT_AGGREGATE,
    )

    (event,) = read_delivery_events(tmp_path)
    assert event.scale is None


def _drop_scale(stored: dict) -> None:
    del stored["scale"]


def _carry_the_replaced_key(stored: dict) -> None:
    stored["scale_assessment_id"] = stored["scale"]["assessment_id"]


def _drop_the_assessment_id(stored: dict) -> None:
    del stored["scale"]["assessment_id"]


def _drop_the_reason(stored: dict) -> None:
    del stored["scale"]["reason"]


def _answer_with_a_reason(stored: dict) -> None:
    stored["scale"]["kind"] = "answers"


@pytest.mark.parametrize(
    "damage",
    [
        _drop_scale,
        _carry_the_replaced_key,
        _drop_the_assessment_id,
        _drop_the_reason,
        _answer_with_a_reason,
    ],
)
def test_a_stored_event_missing_a_scale_field_or_carrying_the_replaced_key_refuses_on_read(
    tmp_path: Path, damage: Callable[[dict], None]
) -> None:
    """The reader refuses a stored event that lacks the scale or one of its parts.

    It refuses one that still carries ``scale_assessment_id``, and an answered read carrying a
    reason, each naming the event.
    """
    bucket = _validated_bucket(tmp_path)
    scale = _scale(tmp_path, extents_mm=UNPASSING_EXTENTS)
    _, event = _delivered_acknowledged(tmp_path, bucket, scale=scale)
    key = delivery_event_key(tmp_path, event.event_id)
    stored = tcip_store.read(key)
    assert isinstance(stored, dict)
    damage(stored)
    tcip_store.replace(key, stored)

    with pytest.raises(tcip_store.ReadRefusedError, match=event.event_id):
        read_delivery_events(tmp_path)
