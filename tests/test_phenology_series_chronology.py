"""Tests of phenology milestones on curves that are not tidy.

Covered: noisy series, exact-on-target first observations, a positive value past an unused one,
and a bucket written by the real prediction writer.

A crossing is computed on the series in capture order, and a date the observations bound but do
not fix is delivered as a bound, by the module and by the delivery tool.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox
from tcip_mcp import subject_registry as cr
from tcip_mcp.dataset_layout import image_dir
from tcip_mcp.pipelines.postprocessing import phenology
from tests import csv_rows
from tests._producer_fixtures import mapping_of, registry_of, scope_over
from tests._trait_fixtures import BUD_OPENING

# An attribute whose positive value sits at id 2 with no record ever carrying id 1.
SPARSE = cr.Attribute("opening", "categorical", ("closed", "partial", "open"))
REGISTRY = registry_of("bud", SPARSE)


def _states(n_positive: int, n_negative: int) -> list[str]:
    return ["open"] * n_positive + ["closed"] * n_negative


# -- crossings on a curve that rises and falls ----------------------------


def test_crossing_uses_the_bracket_that_exists_in_time() -> None:
    """A crossing is read off the bracket of captures that straddles the target in calendar order.

    The crossing is read off the two neighboring captures that straddle the target in calendar
    order, so over a series whose fraction order differs from its capture order its date lies
    inside that bracket and the gap it reports is the bracket's four days.
    """
    series = [
        ("2026-03-09", 0.30),
        ("2026-03-01", 0.10),
        ("2026-03-13", 0.90),
        ("2026-03-05", 0.70),
    ]

    c = phenology.crossing_date(series, 0.50, BUD_OPENING.crossing_rule)

    assert c.bound == "interpolated"
    assert c.date == "2026-03-04"
    assert "2026-03-01" < c.date < "2026-03-05"
    assert c.gap_days == 4


def test_crossing_above_target_at_the_first_capture_stays_left_censored_when_the_curve_dips() -> (
    None
):
    """A curve past the target at its first capture stays left-censored when it later dips.

    A curve whose first capture is already past the target, and which later falls back below it
    and recovers, crosses at the first capture, left-censored.
    """
    series = [("2026-03-01", 0.60), ("2026-03-05", 0.20), ("2026-03-09", 0.80)]

    c = phenology.crossing_date(series, 0.50, BUD_OPENING.crossing_rule)

    assert c.bound == "left_censored"
    assert c.date == "2026-03-01"


def test_positive_onset_is_the_earliest_capture_with_any_positive_observation() -> (
    None
):
    """Onset is the first date in calendar order carrying a positive observation.

    It is not the date carrying the smallest positive fraction.
    """
    series = [("2026-03-01", 0.0), ("2026-03-05", 0.40), ("2026-03-09", 0.10)]

    assert phenology.positive_onset_date(series) == "2026-03-05"


def test_first_capture_exactly_on_the_target_is_an_upper_bound() -> None:
    """A first capture exactly on the target is an upper bound with no gap.

    The date is the first capture's, left-censored, with no gap.
    """
    series = [("2026-03-01", 0.50), ("2026-03-05", 0.90)]

    c = phenology.crossing_date(series, 0.50, BUD_OPENING.crossing_rule)

    assert c.bound == "left_censored"
    assert c.date == "2026-03-01"
    assert c.gap_days is None


def test_first_capture_exactly_on_the_target_keeps_its_date_when_the_curve_falls_back() -> (
    None
):
    """A first capture exactly on the target keeps its date when the curve falls back.

    The crossing is still bounded by the first observation, not moved forward to a later
    re-crossing.
    """
    series = [("2026-03-01", 0.50), ("2026-03-05", 0.20), ("2026-03-09", 0.80)]

    c = phenology.crossing_date(series, 0.50, BUD_OPENING.crossing_rule)

    assert c.bound == "left_censored"
    assert c.date == "2026-03-01"


def test_milestones_of_a_noisy_plant_and_a_steady_plant_are_each_read_in_capture_order(
    tmp_path: Path,
) -> None:
    """A noisy plant's and a steady plant's milestones are each read in capture order.

    Two plants observed on the same four dates, one whose fraction dips mid-season and one that
    rises steadily, get their milestones from their own series. The mapping's dates are supplied
    out of order, as an assembled mapping file has them.
    """
    pytest.importorskip("torch")
    from tests._chain_fixtures import predicted, published

    dates = ["2026-03-01", "2026-03-05", "2026-03-09", "2026-03-13"]
    counts = {
        "2026-03-01": {"P1": (1, 9), "P2": (0, 4)},
        "2026-03-05": {"P1": (7, 3), "P2": (1, 3)},
        "2026-03-09": {"P1": (3, 7), "P2": (3, 1)},
        "2026-03-13": {"P1": (9, 1), "P2": (4, 0)},
    }
    buckets = {
        d: published(
            tmp_path,
            f"run/{d}",
            [
                predicted(
                    image_dir(tmp_path / "ds", d) / f"{plant}_{d}.jpg",
                    _states(pos, neg),
                    (SPARSE,),
                )
                for plant, (pos, neg) in counts[d].items()
            ],
            scope={"subject": "bud"},
            registry=REGISTRY,
        )
        for d in dates
    }
    mapping = mapping_of(
        {
            d: [
                (f"P1_{d}.jpg", ("P1", "acc-noisy"), 0.0),
                (f"P2_{d}.jpg", ("P2", "acc-steady"), 0.0),
            ]
            for d in ["2026-03-09", "2026-03-01", "2026-03-13", "2026-03-05"]
        }
    )

    out = phenology.per_plant_phenology(
        mapping, buckets, BUD_OPENING, ["P1", "P2"]
    )

    rows = {r["plant_id"]: r for r in out["rows"]}
    assert set(rows) == {"P1", "P2"}
    assert rows["P1"]["n_dates"] == 4
    assert rows["P1"]["n_dates_unclassified"] == 0
    # The noisy plant reaches half its buds between the first two captures, before the dip, and
    # never reaches 95 per cent inside the observed window.
    assert rows["P1"]["bud_50per_date"] == "2026-03-04"
    assert rows["P1"]["bud_50per_date_bound"] == "interpolated"
    assert rows["P1"]["bud_95per_date_bound"] == "right_censored"
    # The steady plant crosses later than the noisy one despite ending higher.
    assert rows["P2"]["bud_50per_date"] == "2026-03-07"
    assert rows["P2"]["bud_95per_date"] == "2026-03-12"
    assert rows["P1"]["bud_50per_date"] < rows["P2"]["bud_50per_date"]


# -- the counts the fraction is built from --------------------------------


def test_positive_detections_are_the_named_value_not_a_fixed_position(
    tmp_path: Path,
) -> None:
    """Positive detections are the value the trait names, wherever it sits in declared order.

    An attribute whose positive value sits at id 2 with no record at id 1 counts two positive
    detections of three, none unclassified.
    """
    from tcip_mcp.dataset_layout import prediction_key

    p = prediction_key(tmp_path, "run/2026-03-05", "img")
    json_io.write_label_document(
        p,
        [
            Annotation(
                subject="bud",
                geometry=BBox(1, 1, 4, 9),
                score=0.9,
                attributes={"opening": "open"},
            ),
            Annotation(
                subject="bud",
                geometry=BBox(6, 1, 9, 4),
                score=0.8,
                attributes={"opening": "closed"},
            ),
            Annotation(
                subject="bud",
                geometry=BBox(11, 2, 14, 12),
                score=0.7,
                attributes={"opening": "open"},
            ),
        ],
        40,
        24,
    )

    scope = scope_over(tmp_path, REGISTRY, "bud")
    total, positive, unclassified = phenology.count_by_class(
        p, BUD_OPENING.positive_state, scope=scope
    )

    assert (total, positive, unclassified) == (3, 2, 0)


def test_a_bucket_the_prediction_writer_produced_reads_back_with_its_own_classes(
    tmp_path: Path,
) -> None:
    """A bucket the real publication produced reads back with the values the run decoded.

    The bucket is published through the real publication and read back through the readers here:
    its record declares the state's attribute, and its detections count three, two positive, none
    unclassified.
    """
    pytest.importorskip("torch")
    from tests._chain_fixtures import predicted, published

    image = image_dir(tmp_path / "ds", "2026-03-05") / "P1_2026-03-05.jpg"
    bucket = published(
        tmp_path,
        "run/2026-03-05",
        [predicted(image, ["open", "closed", "open"], (SPARSE,))],
        scope={"subject": "bud"},
        registry=REGISTRY,
    )

    assert bucket.scope.attributes == (SPARSE,)
    counts = phenology.count_by_class(
        bucket.document_key(image.stem),
        BUD_OPENING.positive_state,
        scope=bucket.scope,
    )
    assert counts == (3, 2, 0)


# -- the delivered CSV ----------------------------------------------------


def test_delivered_csv_marks_a_milestone_the_first_capture_only_bounds(
    tmp_path: Path,
) -> None:
    """The delivered CSV marks a milestone met on the first capture as left-censored.

    A plant already at half its buds on the first capture ships that date with its left-censored
    bound, and the later dip below the target does not move the delivered date.
    """
    pytest.importorskip("torch")
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones
    from tests._chain_fixtures import attributed_series

    body = attributed_series(tmp_path, fractions=(0.5, 0.25, 1.0)).body()
    out_csv = tmp_path / "out" / "bud_phenology.csv"

    res = deliver_phenology_milestones(
        tmp_path,
        trait=body["trait"],
        mapping_name=body["mapping_name"],
        plants=["PLANT_A"],
        dataset_root=body["dataset_root"],
        buckets=body["buckets"],
        output_csv_path=str(out_csv),
        actor=None,
    )

    assert "error" not in res, res
    rows = csv_rows(out_csv)
    assert len(rows) == 1
    assert rows[0]["plant_id"] == "PLANT_A"
    assert rows[0]["bud_50per_date"] == "2026-02-11"
    assert rows[0]["bud_50per_date_bound"] == "left_censored"
