"""Phenology milestones on curves that are not tidy: noisy series, exact-on-target first
observations, a positive value past an unused one, and a bucket written by the real prediction
writer.

A positive-fraction curve measured from real imagery moves up and down: weather, occlusion and
sampling noise all push a plant's fraction back down between captures. The milestone a breeder
reads off such a curve is only defensible if the crossing was computed on the series as it happened
in time, and if a date the observations only bound is delivered as a bound rather than as a
measurement. These pin both, at the module's own boundary and through the delivery tool, plus the
agreement between the writer that produces a prediction bucket and the readers here that count it.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox
from tcip_mcp import subject_registry as cr
from tcip_mcp.pipelines.postprocessing import phenology
from tests._trait_fixtures import BUD_OPENING

# An attribute whose positive value sits at id 2 with no record ever carrying id 1.
SPARSE = cr.Attribute("opening", "categorical", ("closed", "partial", "open"))
REGISTRY = cr.SubjectRegistry(subjects=(cr.Subject(name="bud", attributes=(SPARSE,)),))


class _Assignment:
    def __init__(self, stem: str, plot_name: str, accession_name: str) -> None:
        self.stem = stem
        self.plot_name = plot_name
        self.accession_name = accession_name


def _states(n_positive: int, n_negative: int) -> list[str]:
    return ["open"] * n_positive + ["closed"] * n_negative


# -- crossings on a curve that rises and falls ----------------------------


def test_crossing_uses_the_bracket_that_exists_in_time():
    """The crossing is read off the two neighboring captures that straddle the target in calendar
    order, so its date lies inside that bracket and the gap it reports is the real number of days
    between them. A series whose fraction order differs from its capture order (the ordinary case
    for a noisy curve) must not be re-ordered into a bracket that never happened.
    """
    series = [("2026-03-09", 0.30), ("2026-03-01", 0.10), ("2026-03-13", 0.90), ("2026-03-05", 0.70)]

    c = phenology.crossing_date(series, 0.50)

    assert c.bound == "interpolated"
    assert c.date == "2026-03-04"
    assert "2026-03-01" < c.date < "2026-03-05"
    assert c.gap_days == 4


def test_crossing_above_target_at_the_first_capture_stays_left_censored_when_the_curve_dips():
    """A curve whose first capture is already past the target and which later falls back below it
    still crosses only once, before the watching began. The later dip and recovery is not a second,
    later crossing to deliver in its place.
    """
    series = [("2026-03-01", 0.60), ("2026-03-05", 0.20), ("2026-03-09", 0.80)]

    c = phenology.crossing_date(series, 0.50)

    assert c.bound == "left_censored"
    assert c.date == "2026-03-01"


def test_positive_onset_is_the_earliest_capture_with_any_positive_observation():
    """Onset is the first date in calendar order carrying a positive observation, not the date
    carrying the smallest positive fraction.
    """
    series = [("2026-03-01", 0.0), ("2026-03-05", 0.40), ("2026-03-09", 0.10)]

    assert phenology.positive_onset_date(series) == "2026-03-05"


def test_first_capture_exactly_on_the_target_is_an_upper_bound():
    """A first observation sitting exactly on the target met it before anyone looked, so the date is
    an upper bound on the true crossing and carries no bracket to report a gap for. Delivering it as
    an interpolated or exact crossing claims a precision the observations do not support.
    """
    series = [("2026-03-01", 0.50), ("2026-03-05", 0.90)]

    c = phenology.crossing_date(series, 0.50)

    assert c.bound == "left_censored"
    assert c.date == "2026-03-01"
    assert c.gap_days is None


def test_first_capture_exactly_on_the_target_keeps_its_date_when_the_curve_falls_back():
    """The upper-bound reading holds when the fraction drops after that first capture: the crossing
    is still bounded by the first observation, never moved forward to a later re-crossing.
    """
    series = [("2026-03-01", 0.50), ("2026-03-05", 0.20), ("2026-03-09", 0.80)]

    c = phenology.crossing_date(series, 0.50)

    assert c.bound == "left_censored"
    assert c.date == "2026-03-01"


def test_milestones_of_a_noisy_plant_and_a_steady_plant_are_each_read_in_capture_order(tmp_path):
    """Two plants observed on the same four dates, one whose fraction dips mid-season and one that
    rises steadily, get their milestones from their own series in capture order. The mapping's dates
    are supplied out of order, as an assembled mapping file has them, which changes no result.
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
    buckets = {d: published(tmp_path, tmp_path / "ds" / "predictions" / "run" / d, [
        predicted(f"{plant}_{d}", _states(pos, neg), (SPARSE,))
        for plant, (pos, neg) in counts[d].items()], scope={"subject": "bud"},
        registry=REGISTRY) for d in dates}
    mapping = {d: [_Assignment(f"P1_{d}", "P1", "acc-noisy"),
                   _Assignment(f"P2_{d}", "P2", "acc-steady")]
               for d in ["2026-03-09", "2026-03-01", "2026-03-13", "2026-03-05"]}

    out = phenology.per_plant_phenology(mapping, buckets, BUD_OPENING, ["P1", "P2"],
                                        require_all_dates_complete=phenology.REQUIRE_ALL_DATES_COMPLETE)

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


def test_positive_detections_are_the_named_value_not_a_fixed_position(tmp_path):
    """The positive state is whichever value the trait names, wherever it sits in its attribute's
    declared order. An attribute whose positive value sits at id 2 with no record at id 1 counts
    the same as one whose values are all in use.
    """
    from tcip_mcp.pipelines.data.label_queries import registry_scope
    from tests._producer_fixtures import registry_over

    p = tmp_path / "img.json"
    json_io.write_annotations(
        p,
        [Annotation(subject="bud", geometry=BBox(1, 1, 4, 9), score=0.9,
                   attributes={"opening": "open"}),
         Annotation(subject="bud", geometry=BBox(6, 1, 9, 4), score=0.8,
                   attributes={"opening": "closed"}),
         Annotation(subject="bud", geometry=BBox(11, 2, 14, 12), score=0.7,
                   attributes={"opening": "open"})],
        40, 24,
    )

    registry_over(tmp_path, REGISTRY)
    (tmp_path / "annotations").mkdir()
    scope = registry_scope(tmp_path / "annotations", "bud")
    total, positive, unclassified = phenology.count_by_class(p, BUD_OPENING.positive_state,
                                                             scope=scope)

    assert (total, positive, unclassified) == (3, 2, 0)


def test_a_bucket_the_prediction_writer_produced_reads_back_with_its_own_classes(tmp_path):
    """A bucket published through the real publication reads back through the readers here with
    the values the run decoded through: the record's attribute and each detection's own decoded
    value have to line up, or a bucket whose every detection carries the state's attribute counts
    as unassessed.
    """
    pytest.importorskip("torch")
    from tests._chain_fixtures import predicted, published

    bucket = published(tmp_path, tmp_path / "ds" / "predictions" / "run" / "2026-03-05",
                       [predicted("P1_2026-03-05", ["open", "closed", "open"], (SPARSE,))],
                       scope={"subject": "bud"}, registry=REGISTRY)

    assert bucket.scope.attributes == (SPARSE,)
    counts = phenology.count_by_class(bucket.path / "P1_2026-03-05.json",
                                      BUD_OPENING.positive_state, scope=bucket.scope)
    assert counts == (3, 2, 0)


# -- the delivered CSV ----------------------------------------------------


def test_delivered_csv_marks_a_milestone_the_first_capture_only_bounds(tmp_path: Path):
    """A plant already at half its buds on the first capture ships that date with its bound, so a
    breeder reading the CSV can tell an upper bound from a date the observations measured. The later
    dip below the target does not move the delivered date to the re-crossing.
    """
    pytest.importorskip("torch")
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones
    from tests._chain_fixtures import attributed_series

    body = attributed_series(tmp_path, fractions=(0.5, 0.25, 1.0)).body()
    out_csv = tmp_path / "out" / "bud_phenology.csv"

    res = deliver_phenology_milestones(
        tmp_path, trait=body["trait"], mapping_name=body["mapping_name"], plants=["PLANT_A"],
        buckets=body["buckets"], output_csv_path=str(out_csv))

    assert "error" not in res, res
    with out_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["plant_id"] == "PLANT_A"
    assert rows[0]["bud_50per_date"] == "2026-02-11"
    assert rows[0]["bud_50per_date_bound"] == "left_censored"
