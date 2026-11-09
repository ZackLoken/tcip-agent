"""Per-plant aggregation over a cohort of several plants, through to the delivery CSV.

The fixtures give each plant statistics distinct from the other plants' and from the cohort's,
and the tests compare each plant's value, observation count and identity provenance with its own
records. The gate cases deliver a per-plant count under a per-plant and under a per-image count
assessment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tcip_mcp.assessment import read_assessment
from tcip_mcp.pipelines.execution import Stated
from tcip_mcp.pipelines.postprocessing.aggregation import (
    PlantObservation,
    _per_plant,
)
from tcip_mcp.traits import PER_IMAGE_COUNT, PER_PLANT_COUNT_AGGREGATE
from tests import _trait_fixtures as fx, csv_rows
from tests._training_values import (
    DISPERSION_QUANTILE,
    EQUIVALENCE_Z,
    RATIONALE,
    collection_pass,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tcip_mcp.buckets import Bucket
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        MappingResolution,
    )


@pytest.fixture(autouse=True)
def _recorded_meaning(tmp_path: Path) -> None:
    """Confirm the delivery traits and ``stem_count``'s per-plant count meaning of ``bud``."""
    fx.seed_delivery_traits(tmp_path)
    # The count measures what the chain's unassessed bucket detects.
    fx.seed_confirmed_aggregate(
        tmp_path, "stem_count", value_keys=["count"], measured_subject="bud"
    )


def _by_plant(rows: list[dict]) -> dict[str, dict]:
    """Index summary dicts (or CSV rows) by plant_id, asserting more than one distinct plant.

    The assertion also requires every row's plant_id to be distinct.
    """
    indexed = {r["plant_id"]: r for r in rows}
    assert len(indexed) == len(rows) > 1
    return indexed


# -- each plant's summary comes from that plant's own records ----------------


def test_each_plants_summary_describes_only_that_plants_records() -> None:
    """Each plant's median and image count are computed over that plant's own records.

    Three plants have deliberately different count distributions, none of which equals the
    cohort-wide statistic.
    """
    results = [
        PlantObservation(plant_id=plant, value=count)
        for plant, count in (
            ("PLANT_A", 2),
            ("PLANT_A", 4),
            ("PLANT_A", 9),
            ("PLANT_B", 10),
            ("PLANT_B", 20),
            ("PLANT_C", 7),
        )
    ]
    out = _by_plant(_per_plant(results, plant_aggregation="median"))

    assert out["PLANT_A"]["value"] == 4
    assert out["PLANT_B"]["value"] == 15.0
    assert out["PLANT_C"]["value"] == 7
    assert [out[p]["n_images"] for p in ("PLANT_A", "PLANT_B", "PLANT_C")] == [
        3,
        2,
        1,
    ]


def test_identity_provenance_is_summarized_per_plant_not_across_the_cohort() -> (
    None
):
    """Each plant's identity provenance is summarized over its own records.

    One plant is resolved from a single source, another from two, a third with no provenance at
    all: each reports its own source and its own worst assignment distance, and the third
    reports neither.
    """
    results = [
        PlantObservation(
            plant_id="PLANT_A", value=2, source="gnss_sequence", distance_m=0.4
        ),
        PlantObservation(
            plant_id="PLANT_A", value=4, source="gnss_sequence", distance_m=1.9
        ),
        PlantObservation(
            plant_id="PLANT_B",
            value=10,
            source="gnss_sequence",
            distance_m=6.5,
        ),
        PlantObservation(plant_id="PLANT_B", value=20, source="qr_code"),
        PlantObservation(plant_id="PLANT_C", value=7),
    ]
    out = _by_plant(_per_plant(results, plant_aggregation="median"))

    assert out["PLANT_A"]["plant_id_source"] == "gnss_sequence"
    assert out["PLANT_A"]["plant_id_distance_m_max"] == pytest.approx(1.9)
    assert out["PLANT_B"]["plant_id_source"] == "mixed"
    assert out["PLANT_B"]["plant_id_distance_m_max"] == pytest.approx(6.5)
    assert "plant_id_source" not in out["PLANT_C"]
    assert "plant_id_distance_m_max" not in out["PLANT_C"]


def test_delivery_csv_carries_each_plants_own_value_and_image_count(
    tmp_path: Path,
) -> None:
    """Each delivery CSV row carries its own plant's value and image count.

    The delivery runs from per-image records to the CSV; each of three rows carries its plant's
    value and n_images, PLANT_A its maximum distance, PLANT_B its source and an empty distance,
    and PLANT_C an empty source.
    """
    pytest.importorskip("torch")
    from tests._chain_fixtures import deliver_acknowledged

    results = [
        PlantObservation(
            plant_id="PLANT_A", value=2, source="gnss_sequence", distance_m=0.4
        ),
        PlantObservation(
            plant_id="PLANT_A", value=4, source="gnss_sequence", distance_m=1.9
        ),
        PlantObservation(plant_id="PLANT_A", value=9, source="gnss_sequence"),
        PlantObservation(plant_id="PLANT_B", value=10, source="qr_code"),
        PlantObservation(plant_id="PLANT_B", value=20, source="qr_code"),
        PlantObservation(plant_id="PLANT_C", value=7),
    ]
    out_path = tmp_path / "per_plant.csv"
    deliver_acknowledged(
        tmp_path,
        results,
        out_path,
        "stem_count",
        value_key="count",
        delivery_kind=PER_PLANT_COUNT_AGGREGATE,
    )
    rows = _by_plant(csv_rows(out_path))

    assert [
        int(rows[p]["n_images"]) for p in ("PLANT_A", "PLANT_B", "PLANT_C")
    ] == [3, 2, 1]
    assert float(rows["PLANT_A"]["value"]) == 4
    assert float(rows["PLANT_B"]["value"]) == 15.0
    assert float(rows["PLANT_C"]["value"]) == 7
    assert rows["PLANT_A"]["plant_id_distance_m_max"] == "1.9"
    assert rows["PLANT_B"]["plant_id_source"] == "qr_code"
    assert rows["PLANT_B"]["plant_id_distance_m_max"] == ""
    assert rows["PLANT_C"]["plant_id_source"] == ""


# -- a plant's mean is its own -------------------------------------------------


def test_each_plants_mean_describes_only_that_plants_records() -> None:
    """Each plant's mean is computed over that plant's own records.

    The samples are skewed, each plant's mean and median several units apart.
    """
    results = [
        PlantObservation(plant_id=plant, value=value)
        for plant, value in (
            ("PLANT_A", 1.0),
            ("PLANT_A", 2.0),
            ("PLANT_A", 9.0),
            ("PLANT_B", 10.0),
            ("PLANT_B", 11.0),
            ("PLANT_B", 30.0),
        )
    ]
    out = _by_plant(_per_plant(results, plant_aggregation="mean"))

    assert out["PLANT_A"]["value"] == pytest.approx(4.0)
    assert out["PLANT_B"]["value"] == pytest.approx(17.0)


def test_summed_areas_stay_within_their_own_plant() -> None:
    """Each plant's summed area is taken over its own images.

    The plants have different numbers of contributing images, and each reports its own image
    count.
    """
    results = [
        PlantObservation(plant_id="PLANT_A", value=100.0),
        PlantObservation(plant_id="PLANT_A", value=250.0),
        PlantObservation(plant_id="PLANT_B", value=40.0),
    ]
    out = _by_plant(_per_plant(results, plant_aggregation="sum"))

    assert out["PLANT_A"]["value"] == pytest.approx(350.0)
    assert out["PLANT_A"]["n_images"] == 2
    assert out["PLANT_B"]["value"] == pytest.approx(40.0)
    assert out["PLANT_B"]["n_images"] == 1


# -- an assessment answers for the kind it measured, and no other -------


def _assessed(project: Path, kind: str, experiment_id: str) -> Bucket:
    """Return a bucket published under an assessment of ``kind``, as recorded.

    The count trait is confirmed first with both its per-image and its per-plant count
    operationalizations, and the captures are mapped to :data:`_PLANTS` by the mapping
    ``valley`` once the bucket is published.
    """
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.tools.calibration_tools import assess_checkpoint
    from tcip_mcp.tools.inference_tools import run_inference
    from tests._chain_fixtures import (
        DATE,
        SUBJECT,
        draw_reference_selection,
        plant_captures,
        synthetic_capture,
        train_on,
    )
    from tests._mapping_fixtures import COUNT_CROP, map_plants

    entry = fx.with_operationalization(
        fx.COUNT_SPEC, PER_IMAGE_COUNT, measured_subject=SUBJECT
    )
    fx.propose_and_confirm(
        project,
        fx.with_operationalization(
            entry,
            PER_PLANT_COUNT_AGGREGATE,
            measured_subject=SUBJECT,
            delivered_phenotypes=("stem_count",),
            delivered_value_keys=("count",),
        ),
    )
    root = project / "ds"
    images_dir = synthetic_capture(root, positions=plant_captures(_PLANTS))
    draw_reference_selection(project, root, project / "selection")
    checkpoint = train_on(project / "selection", project, experiment_id)
    assessment = assess_checkpoint(
        project,
        checkpoint_path=checkpoint,
        trait=fx.COUNT_TRAIT,
        delivery_kind=kind,
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
    bucket = f"{kind}/{DATE}"
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
    map_plants(project, root, _PLANTS, crop=COUNT_CROP)
    return read_bucket(root, bucket)


def _resolved(project: Path, bucket: Bucket) -> MappingResolution:
    """Return the resolved mapping of delivering :data:`_PLANTS` over ``bucket``."""
    from tcip_mcp.buckets import by_recorded_date
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        plant_mapping_disclosure,
    )

    return plant_mapping_disclosure(
        project, "valley", by_recorded_date([bucket]), _PLANTS
    )


_PLANTS = ["PLANT_A"]
_RECORDS = [
    PlantObservation(plant_id="PLANT_A", value=count) for count in (2, 4, 9)
]


def test_a_per_plant_count_assessment_validates_a_per_plant_count_delivery(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.postprocessing.aggregation import (
        deliver_per_plant_aggregate,
    )

    bucket = _assessed(tmp_path, PER_PLANT_COUNT_AGGREGATE, "exp-kind-matched")
    resolved = _resolved(tmp_path, bucket)
    delivered = deliver_per_plant_aggregate(
        tmp_path,
        _RECORDS,
        str(tmp_path / "matched.csv"),
        value_key="count",
        delivered_phenotype="stem_count",
        delivery_kind=PER_PLANT_COUNT_AGGREGATE,
        buckets=[bucket],
        plants=_PLANTS,
        crop=resolved.verified.crop,
        door="test",
        plant_mapping=resolved.disclosure,
        actor=None,
    )

    assert delivered.shipped.kind == "validated"


def test_a_per_image_count_assessment_never_answers_for_a_per_plant_delivery(
    tmp_path: Path,
) -> None:
    """A per-plant delivery under a per-image count assessment is refused, naming both kinds.

    The per-image count the assessment measured is not the per-plant number the delivery ships:
    the gate refuses, and no file is written.
    """
    pytest.importorskip("torch")
    from tcip_mcp.delivery import DeliveryRefusedError
    from tcip_mcp.pipelines.postprocessing.aggregation import (
        deliver_per_plant_aggregate,
    )

    bucket = _assessed(tmp_path, PER_IMAGE_COUNT, "exp-kind-crossed")
    resolved = _resolved(tmp_path, bucket)
    with pytest.raises(
        DeliveryRefusedError,
        match=f"{PER_IMAGE_COUNT} delivery, not a "
        f"{PER_PLANT_COUNT_AGGREGATE} one",
    ):
        deliver_per_plant_aggregate(
            tmp_path,
            _RECORDS,
            str(tmp_path / "crossed.csv"),
            value_key="count",
            delivered_phenotype="stem_count",
            delivery_kind=PER_PLANT_COUNT_AGGREGATE,
            buckets=[bucket],
            plants=_PLANTS,
            crop=resolved.verified.crop,
            door="test",
            plant_mapping=resolved.disclosure,
            actor=None,
        )
    assert not (tmp_path / "crossed.csv").exists()
