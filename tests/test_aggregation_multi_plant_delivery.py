"""Per-plant aggregation over a cohort of several plants, through to the delivery CSV.

The fixtures give each plant statistics distinct from the other plants' and from the cohort's,
and the tests compare each plant's value, observation count and identity provenance with its own
images. The gate cases deliver a per-plant count under a per-plant and under a per-image count
assessment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tcip_mcp.assessment import read_assessment
from tcip_mcp.pipelines.execution import Stated
from tcip_mcp.pipelines.postprocessing.aggregation import _per_plant
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
        Assignment,
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


# -- each plant's summary comes from that plant's own images ----------------

COHORT = ("PLANT_A", "PLANT_B", "PLANT_C")
COHORT_DATES = ["2026-02-11", "2026-02-25", "2026-03-11"]


def _cohort(tmp_path: Path) -> dict[str, list[Assignment]]:
    """Return each plant's assignments of a mapping of :data:`COHORT` over :data:`COHORT_DATES`.

    The mapping is ``build_mapping``'s (``mapping_over``) of one geolocated capture of each plant
    on each date, each at its plant's place.
    """
    from tests._mapping_fixtures import (
        geolocated_captures,
        mapping_over,
        plant_positions,
        plant_rows,
        write_plots_csv,
    )

    places = plant_positions(COHORT)
    root = tmp_path / "ds"
    geolocated_captures(root, COHORT_DATES, plant_rows(places))
    build = mapping_over(
        root,
        [write_plots_csv(tmp_path / "cohort.csv", places)],
        tmp_path,
        dates=COHORT_DATES,
    )
    return {
        plant: [
            a
            for date in COHORT_DATES
            for a in build.assignments[date]
            if a.plant_id == plant
        ]
        for plant in COHORT
    }


def _observed(
    tmp_path: Path, values: dict[str, list[float]]
) -> list[tuple[Assignment, float]]:
    """Return each plant's first images of :func:`_cohort`, valued by ``values``."""
    assigned = _cohort(tmp_path)
    return [
        (a, v)
        for plant, plant_values in values.items()
        for a, v in zip(assigned[plant], plant_values)
    ]


def test_each_plants_summary_describes_only_that_plants_images(
    tmp_path: Path,
) -> None:
    """Each plant's median and image count are computed over that plant's own images.

    Three plants have deliberately different count distributions, none of which equals the
    cohort-wide statistic.
    """
    observed = _observed(
        tmp_path, {"PLANT_A": [2, 4, 9], "PLANT_B": [10, 20], "PLANT_C": [7]}
    )
    out = _by_plant(_per_plant(observed, plant_aggregation="median"))

    assert out["PLANT_A"]["value"] == 4
    assert out["PLANT_B"]["value"] == 15.0
    assert out["PLANT_C"]["value"] == 7
    assert [out[p]["n_images"] for p in COHORT] == [3, 2, 1]


def test_identity_provenance_is_summarized_per_plant_not_across_the_cohort(
    tmp_path: Path,
) -> None:
    """Each plant's identity provenance is summarized over its own images' assignments.

    One plant is assigned by a single source, another by two: each reports its own source and
    its own worst assignment distance.
    """
    from dataclasses import replace

    assigned = _cohort(tmp_path)
    a1, a2, _ = assigned["PLANT_A"]
    b1, b2, _ = assigned["PLANT_B"]
    observed = [
        (replace(a1, distance_m=0.4), 2),
        (replace(a2, distance_m=0.9), 4),
        (replace(b1, distance_m=0.7), 10),
        (replace(b2, source="nearest_neighbor", distance_m=0.2), 20),
    ]
    out = _by_plant(_per_plant(observed, plant_aggregation="median"))

    assert out["PLANT_A"]["plant_id_source"] == "sequence"
    assert out["PLANT_A"]["plant_id_distance_m_max"] == pytest.approx(0.9)
    assert out["PLANT_B"]["plant_id_source"] == "mixed"
    assert out["PLANT_B"]["plant_id_distance_m_max"] == pytest.approx(0.7)


def test_delivery_csv_carries_each_plants_own_value_and_image_count(
    tmp_path: Path,
) -> None:
    """Each delivery CSV row carries its own plant's value, image count and assignments.

    The delivery runs from per-image values to the CSV; each of three rows carries its plant's
    value and n_images, and the source and worst distance of the mapping's assignments of its
    images: the plants imaged more than once find their place claimed after the first image.
    """
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.postprocessing.plant_mapping import load_mapping
    from tests._chain_fixtures import DATE, deliver_acknowledged

    out_path = tmp_path / "per_plant.csv"
    deliver_acknowledged(
        tmp_path,
        {"PLANT_A": [2, 4, 9], "PLANT_B": [10, 20], "PLANT_C": [7]},
        out_path,
        "stem_count",
        value_key="count",
        delivery_kind=PER_PLANT_COUNT_AGGREGATE,
    )
    rows = _by_plant(csv_rows(out_path))
    build = load_mapping(tmp_path, "valley")
    assert build is not None

    assert [int(rows[p]["n_images"]) for p in COHORT] == [3, 2, 1]
    assert float(rows["PLANT_A"]["value"]) == 4
    assert float(rows["PLANT_B"]["value"]) == 15.0
    assert float(rows["PLANT_C"]["value"]) == 7
    assert [rows[p]["plant_id_source"] for p in COHORT] == [
        "mixed",
        "mixed",
        "sequence",
    ]
    for plant in COHORT:
        assert float(rows[plant]["plant_id_distance_m_max"]) == pytest.approx(
            max(
                a.distance_m
                for a in build.assignments[DATE]
                if a.plant_id == plant and a.distance_m is not None
            )
        )


# -- a plant's mean is its own -------------------------------------------------


def test_each_plants_mean_describes_only_that_plants_images(
    tmp_path: Path,
) -> None:
    """Each plant's mean is computed over that plant's own images.

    The samples are skewed, each plant's mean and median several units apart.
    """
    observed = _observed(
        tmp_path, {"PLANT_A": [1.0, 2.0, 9.0], "PLANT_B": [10.0, 11.0, 30.0]}
    )
    out = _by_plant(_per_plant(observed, plant_aggregation="mean"))

    assert out["PLANT_A"]["value"] == pytest.approx(4.0)
    assert out["PLANT_B"]["value"] == pytest.approx(17.0)


def test_summed_areas_stay_within_their_own_plant(tmp_path: Path) -> None:
    """Each plant's summed area is taken over its own images.

    The plants have different numbers of contributing images, and each reports its own image
    count.
    """
    observed = _observed(
        tmp_path, {"PLANT_A": [100.0, 250.0], "PLANT_B": [40.0]}
    )
    out = _by_plant(_per_plant(observed, plant_aggregation="sum"))

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
    images_dir = synthetic_capture(
        root,
        positions=plant_captures({p: len(v) for p, v in _VALUES.items()}),
    )
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


_VALUES = {"PLANT_A": [2, 4, 9]}
"""Each delivered plant's per-image values, one per capture :func:`_assessed` takes at it."""
_PLANTS = list(_VALUES)


def test_a_per_plant_count_assessment_validates_a_per_plant_count_delivery(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.postprocessing.aggregation import (
        deliver_per_plant_aggregate,
    )
    from tests._chain_fixtures import mapped_results

    bucket = _assessed(tmp_path, PER_PLANT_COUNT_AGGREGATE, "exp-kind-matched")
    resolved = _resolved(tmp_path, bucket)
    delivered = deliver_per_plant_aggregate(
        tmp_path,
        mapped_results(resolved, _VALUES),
        str(tmp_path / "matched.csv"),
        value_key="count",
        delivered_phenotype="stem_count",
        delivery_kind=PER_PLANT_COUNT_AGGREGATE,
        buckets=[bucket],
        plants=_PLANTS,
        mapping=resolved,
        door="test",
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
    from tests._chain_fixtures import mapped_results

    bucket = _assessed(tmp_path, PER_IMAGE_COUNT, "exp-kind-crossed")
    resolved = _resolved(tmp_path, bucket)
    with pytest.raises(
        DeliveryRefusedError,
        match=f"{PER_IMAGE_COUNT} delivery, not a "
        f"{PER_PLANT_COUNT_AGGREGATE} one",
    ):
        deliver_per_plant_aggregate(
            tmp_path,
            mapped_results(resolved, _VALUES),
            str(tmp_path / "crossed.csv"),
            value_key="count",
            delivered_phenotype="stem_count",
            delivery_kind=PER_PLANT_COUNT_AGGREGATE,
            buckets=[bucket],
            plants=_PLANTS,
            mapping=resolved,
            door="test",
            actor=None,
        )
    assert not (tmp_path / "crossed.csv").exists()
