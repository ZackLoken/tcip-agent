"""Tests for the canonical phenology measurement module.

The positive-state fraction is the share of a plant's detected objects that are in the trait's
positive/measured state, where that state is a classifier call (never a geometric proxy). These
tests pin the authoritative trait definitions:

    catkin_05/50/95per_date  = dates the elongated fraction crosses 5/50/95%
    catkin_elongation_date   = date most catkins have elongated (crops.yml) = the 95% crossing

and the id_map-key-membership coverage rule that decides whether a prediction
bucket ever assessed the trait's positive class at all, and the per-detection membership check
within it.
"""

from __future__ import annotations

import json
import sys

import pytest
from pathlib import Path

_MCP_SRC = Path(__file__).resolve().parents[1] / "packages" / "tcip-mcp" / "src"
if str(_MCP_SRC) not in sys.path:
    sys.path.insert(0, str(_MCP_SRC))

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox  # noqa: E402
from tcip_mcp.pipelines.postprocessing import phenology  # noqa: E402
from tcip_mcp.traits import TraitSpec  # noqa: E402
from tests._trait_fixtures import CATKIN  # noqa: E402


def _sidecar(dir_path: Path, id_map: dict | None) -> None:
    """Write a bucket's operating_point.json exactly the way export_predictions does: the only
    fact count_by_class reads to decide whether/how a bucket was classified."""
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "operating_point.json").write_text(json.dumps({"id_map": id_map}), encoding="utf-8")


def _preds(dir_path: Path, stem: str, subjects: list[str]) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    anns = [Annotation(subject=s, geometry=BBox(1.0, 1.0, 3.0, 3.0), score=0.9) for s in subjects]
    json_io.write_annotations(dir_path / f"{stem}.json", anns, 8, 8)


# ── date helpers (unchanged) ──────────────────────────────────────────────


def test_date_key_orders_chronologically():
    assert phenology.date_key("2024-05-01") < phenology.date_key("2024-05-15")
    assert phenology.date_key("2024-05-15") < phenology.date_key("2024-06-01")


def test_date_key_malformed_sorts_first():
    assert phenology.date_key("undated") == (0, 0, 0)
    assert phenology.date_key("2024-13") == (0, 0, 0)
    assert phenology.date_key("not-a-date-x") == (0, 0, 0)


def test_date_key_rejects_out_of_range_dates():
    assert phenology.date_key("2026-13-01") == (0, 0, 0)
    assert phenology.date_key("2026-02-30") == (0, 0, 0)


def test_crossing_date_does_not_crash_on_malformed_date():
    series = [
        ("2026-02-01", 0.0),
        ("2026-13-01", 0.9),  # out-of-range month, excluded
        ("2026-02-15", 1.0),
    ]
    c = phenology.crossing_date(series, 0.50)
    assert c.date == "2026-02-08"
    assert c.bound == "interpolated"
    assert phenology.positive_onset_date(series) == "2026-02-15"


def test_real_points_drops_undated_and_sorts():
    series = [("2024-05-15", 0.5), ("undated", 0.9), ("2024-05-01", 0.1)]
    pts = phenology._real_points(series)
    assert [d for d, _ in pts] == ["2024-05-01", "2024-05-15"]


# ── crossings (censoring bound disclosed) ─────────────────────────────


def test_crossing_interpolates_between_dates():
    series = [("2024-05-01", 0.0), ("2024-05-11", 1.0)]
    c = phenology.crossing_date(series, 0.50)
    assert c.date == "2024-05-06"
    assert c.bound == "interpolated"
    assert c.gap_days == 10


def test_crossing_first_point_already_at_target_is_left_censored():
    # A left-censored crossing must be flagged, not returned as a bare tuple indistinguishable
    # from a real single-date measurement.
    series = [("2024-05-01", 0.2), ("2024-05-05", 0.8)]
    c = phenology.crossing_date(series, 0.05)
    assert c.date == "2024-05-01"
    assert c.bound == "left_censored"


def test_crossing_exact_match_is_not_censored():
    series = [("2024-05-01", 0.0), ("2024-05-05", 0.50)]
    c = phenology.crossing_date(series, 0.50)
    assert c.date == "2024-05-05"
    assert c.bound == "exact"


def test_crossing_never_reached_is_right_censored():
    # The last observed point still hasn't met the target -> the true
    # crossing, if it happens at all, is after this date. Distinguishable from "no observations at
    # all" (which stays None, see below).
    series = [("2024-05-01", 0.0), ("2024-05-05", 0.3)]
    c = phenology.crossing_date(series, 0.95)
    assert c.date == "2024-05-05"
    assert c.bound == "right_censored"
    assert c.gap_days is None


def test_crossing_no_observations_is_none():
    assert phenology.crossing_date([], 0.95) is None


def test_elongation_onset_is_first_nonzero_date():
    series = [("2024-05-01", 0.0), ("2024-05-05", 0.0), ("2024-05-09", 0.10), ("2024-05-13", 0.60)]
    assert phenology.positive_onset_date(series) == "2024-05-09"


def test_elongation_onset_none_when_all_zero():
    series = [("2024-05-01", 0.0), ("2024-05-05", 0.0)]
    assert phenology.positive_onset_date(series) is None


def test_plant_milestones_returns_four_dates_and_bounds():
    series = [("2024-05-01", 0.0), ("2024-05-06", 0.04), ("2024-05-11", 0.50), ("2024-05-21", 1.0)]
    m = phenology.plant_milestones(series, CATKIN)
    assert {"catkin_05per_date", "catkin_50per_date", "catkin_95per_date", "catkin_elongation_date"} <= set(m)
    assert m["catkin_elongation_date"] == m["catkin_95per_date"]
    assert m["catkin_50per_date"] == "2024-05-11"
    assert m["catkin_50per_date_bound"] == "exact"


def test_plant_milestones_requires_spec_no_catkin_fallback():
    # No silent default: a caller that forgets to pass spec must fail loudly.
    import pytest

    with pytest.raises(TypeError):
        phenology.plant_milestones([("2024-05-01", 0.5)])


def test_milestone_date_columns_is_proper_subset_of_full_columns():
    cols = phenology.phenology_csv_columns(CATKIN)
    milestone_cols = phenology.milestone_date_columns(CATKIN)
    assert set(milestone_cols) <= set(cols)
    assert "plant_id" not in milestone_cols
    assert "positive_state_classifier_validated" not in milestone_cols


# ── count_by_class: the coverage mechanism ─────────────────────


def test_count_by_class_bare_detector_bucket_refuses_never_full_coverage(tmp_path):
    # A single-class detector's id_map is {"catkin": 0}, no
    # attribute axis at all. Every prediction decodes to subject="catkin". This must not be scored
    # classified: "catkin" is not the trait's positive value.
    p = tmp_path / "img.json"
    json_io.write_annotations(
        p, [Annotation(subject="catkin", geometry=BBox(1, 1, 3, 3), score=0.9),
            Annotation(subject="catkin", geometry=BBox(4, 4, 6, 6), score=0.8)],
        8, 8,
    )
    id_map = {"catkin": 0}
    total, positive, unclassified = phenology.count_by_class(p, id_map, "elongated")
    assert (total, positive, unclassified) == (2, 0, 2)  # whole bucket unclassified, not full coverage


def test_count_by_class_wrong_axis_bucket_refuses(tmp_path):
    # A run classified on a different attribute of the same subject (e.g. damage severity); its
    # value set doesn't include the trait's positive value, so it must refuse, not be miscounted.
    p = tmp_path / "img.json"
    json_io.write_annotations(
        p, [Annotation(subject="mild", geometry=BBox(1, 1, 3, 3), score=0.9)], 8, 8,
    )
    id_map = {"none": 0, "mild": 1, "severe": 2}
    total, positive, unclassified = phenology.count_by_class(p, id_map, "elongated")
    assert (total, positive, unclassified) == (1, 0, 1)


def test_count_by_class_absent_id_map_refuses(tmp_path):
    p = tmp_path / "img.json"
    json_io.write_annotations(
        p, [Annotation(subject="elongated", geometry=BBox(1, 1, 3, 3), score=0.9)], 8, 8,
    )
    total, positive, unclassified = phenology.count_by_class(p, None, "elongated")
    assert (total, positive, unclassified) == (1, 0, 1)


def test_count_by_class_classified_bucket_splits_positive_negative(tmp_path):
    p = tmp_path / "img.json"
    json_io.write_annotations(
        p, [Annotation(subject="elongated", geometry=BBox(1, 1, 3, 3), score=0.9),
            Annotation(subject="dormant", geometry=BBox(4, 4, 6, 6), score=0.8),
            Annotation(subject="elongated", geometry=BBox(1, 1, 3, 3), score=0.7)],
        8, 8,
    )
    id_map = {"dormant": 0, "elongated": 1}
    total, positive, unclassified = phenology.count_by_class(p, id_map, "elongated")
    assert (total, positive, unclassified) == (3, 2, 0)


def test_count_by_class_foreign_subject_within_classified_bucket_is_per_detection_unclassified(tmp_path):
    # A classified bucket (id_map has the positive value) must still not
    # silently coerce a detection whose own subject isn't a key of that map (a stale file from a
    # prior run, or a raw-index decode fallback) into a classified negative.
    p = tmp_path / "img.json"
    json_io.write_annotations(
        p, [Annotation(subject="elongated", geometry=BBox(1, 1, 3, 3), score=0.9),
            Annotation(subject="catkin", geometry=BBox(4, 4, 6, 6), score=0.8),  # stale bare-detector file
            Annotation(subject="2", geometry=BBox(1, 1, 3, 3), score=0.7)],  # raw-index fallback
        8, 8,
    )
    id_map = {"dormant": 0, "elongated": 1}
    total, positive, unclassified = phenology.count_by_class(p, id_map, "elongated")
    assert (total, positive, unclassified) == (3, 1, 2)


def test_count_by_class_missing_file_reads_as_empty():
    # count_by_class itself degrades gracefully (json_io.read_annotations on a missing path returns
    # []), but per_plant_series never actually calls it on a missing path; it checks is_file() and
    # tracks n_missing itself instead (see test_per_plant_phenology_missing_image_is_disclosed_not_a_zero),
    # so a missing observation is disclosed there, not silently zero'd here.
    total, positive, unclassified = phenology.count_by_class(
        Path("does-not-exist.json"), {"elongated": 1}, "elongated")
    assert (total, positive, unclassified) == (0, 0, 0)


# ── per_plant_series / per_plant_phenology: bucket-level + expected-coverage ────────────────────


class _Assignment:
    def __init__(self, stem, plot_name, accession_name):
        self.stem = stem
        self.plot_name = plot_name
        self.accession_name = accession_name


def test_per_plant_phenology_builds_fraction_series_when_classified(tmp_path):
    d1 = tmp_path / "2024-05-01"
    d2 = tmp_path / "2024-05-15"
    _preds(d1, "P1_a", ["dormant", "dormant"])
    _sidecar(d1, {"dormant": 0, "elongated": 1})
    _preds(d2, "P1_b", ["elongated", "elongated"])
    _sidecar(d2, {"dormant": 0, "elongated": 1})
    mapping = {
        "2024-05-01": [_Assignment("P1_a", "P1", "acc-9")],
        "2024-05-15": [_Assignment("P1_b", "P1", "acc-9")],
    }
    preds = {"2024-05-01": str(d1), "2024-05-15": str(d2)}

    out = phenology.per_plant_phenology(mapping, preds, positive_class_name="elongated", spec=CATKIN)

    assert out["positive_class_assessed"] is True
    row = out["rows"][0]
    assert row["plant_id"] == "P1"
    assert row["accession"] == "acc-9"
    assert row["n_dates"] == 2
    assert row["n_dates_unclassified"] == 0
    assert row["n_dates_missing_images"] == 0
    assert [s["n_positive"] for s in row["series"]] == [0, 2]
    assert row["catkin_95per_date"] is not None


def test_per_plant_phenology_bare_detector_bucket_refuses_whole_delivery(tmp_path):
    d1 = tmp_path / "2024-05-01"
    _preds(d1, "P1_a", ["catkin"])
    _sidecar(d1, {"catkin": 0})
    mapping = {"2024-05-01": [_Assignment("P1_a", "P1", "acc-9")]}
    preds = {"2024-05-01": str(d1)}

    out = phenology.per_plant_phenology(mapping, preds, positive_class_name="elongated", spec=CATKIN)

    assert out["positive_class_assessed"] is False
    row = out["rows"][0]
    assert row["n_dates_unclassified"] == 1
    assert row["catkin_50per_date"] is None


def test_per_plant_phenology_missing_image_is_disclosed_not_a_zero(tmp_path):
    # A stem the mapping names with no prediction file must not read as an
    # observed zero (which would count as "classified, 0/0" and silently pass coverage).
    d1 = tmp_path / "2024-05-01"
    d1.mkdir(parents=True, exist_ok=True)
    _sidecar(d1, {"dormant": 0, "elongated": 1})
    # no P1_a.json written: the mapping names it but nothing was ever inferred for it
    mapping = {"2024-05-01": [_Assignment("P1_a", "P1", "acc-9")]}
    preds = {"2024-05-01": str(d1)}

    out = phenology.per_plant_phenology(mapping, preds, positive_class_name="elongated", spec=CATKIN)

    row = out["rows"][0]
    assert row["series"][0]["n_missing"] == 1
    assert row["series"][0]["ratio"] is None
    assert row["n_dates_missing_images"] == 1
    assert out["positive_class_assessed"] is False  # nothing anywhere was actually classified
    assert row["catkin_50per_date"] is None


def test_per_plant_phenology_multi_date_and_excludes_plant_with_one_bad_date(tmp_path):
    # A plant's milestones require every date to be classified; one
    # unclassified or missing date excludes the whole plant's milestones, disclosed, not silently
    # computed from the subset that happened to be usable.
    d1 = tmp_path / "2024-05-01"
    d2 = tmp_path / "2024-05-15"
    _preds(d1, "P1_a", ["elongated"])
    _sidecar(d1, {"dormant": 0, "elongated": 1})
    _preds(d2, "P1_b", ["catkin"])  # bare-detector date, unclassified
    _sidecar(d2, {"catkin": 0})
    mapping = {
        "2024-05-01": [_Assignment("P1_a", "P1", "acc-9")],
        "2024-05-15": [_Assignment("P1_b", "P1", "acc-9")],
    }
    preds = {"2024-05-01": str(d1), "2024-05-15": str(d2)}

    out = phenology.per_plant_phenology(mapping, preds, positive_class_name="elongated", spec=CATKIN)

    row = out["rows"][0]
    assert row["n_dates"] == 2
    assert row["n_dates_unclassified"] == 1
    # The whole plant's milestones are None: one bad date excludes it, not a partial computation.
    assert row["catkin_05per_date"] is None
    assert row["catkin_95per_date"] is None
    # But at least one date elsewhere was classified, so the delivery-level flag is still True,
    # distinguishing "wired, some gaps" from "never wired at all".
    assert out["positive_class_assessed"] is True


def test_per_plant_series_accepts_dict_assignments(tmp_path):
    d1 = tmp_path / "2024-05-01"
    _preds(d1, "P1_a", ["elongated"])
    _sidecar(d1, {"dormant": 0, "elongated": 1})
    mapping = {"2024-05-01": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}]}
    preds = {"2024-05-01": str(d1)}
    per_plant = phenology.per_plant_series(mapping, preds, positive_class_name="elongated")
    assert "P1" in per_plant
    assert per_plant["P1"]["accession"] == "acc-9"
    assert per_plant["P1"]["series"][0][:3] == ("2024-05-01", 1, 1)  # total=1, positive=1


# ── resolve_positive_class_id ─────────────────────────────────────────────


def test_resolve_positive_class_id_from_bucket_id_map(tmp_path):
    d1 = tmp_path / "2024-05-01"
    _sidecar(d1, {"dormant": 0, "elongated": 1})
    cid, msg = phenology.resolve_positive_class_id(CATKIN, {"2024-05-01": str(d1)})
    assert cid == 1
    assert "resolved" in msg


def test_resolve_positive_class_id_no_bucket_has_it_refuses(tmp_path):
    d1 = tmp_path / "2024-05-01"
    _sidecar(d1, {"catkin": 0})
    cid, msg = phenology.resolve_positive_class_id(CATKIN, {"2024-05-01": str(d1)})
    assert cid is None
    assert "never assessed" in msg


# ── write_phenology_csv raises on an unknown stamp key ────────────────────


def test_write_phenology_csv_raises_on_unknown_stamp_key(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        phenology.write_phenology_csv([], tmp_path / "out.csv", CATKIN, stamp={"bogus_key": "x"})


def test_write_phenology_csv_carries_every_milestone_bound(tmp_path):
    """plant_milestones emits a `_bound` beside each milestone date
    (exact / interpolated / left_censored); those must be in phenology_csv_columns, or the writer's
    extrasaction="ignore" drops them all. A left-censored crossing (the first observation already
    met the target, so the date is only an upper bound) must not ship indistinguishable from a
    measured one, which would be a precision claim the data does not support."""
    # First observed point already at 100% -> every crossing is left-censored.
    series = [("2026-02-11", 1.0), ("2026-02-20", 1.0)]
    milestones = phenology.plant_milestones(series, CATKIN)
    assert milestones["catkin_05per_date_bound"] == "left_censored"

    row = {"plant_id": "P1", "n_dates": 2, "n_observed_dates": 2, **milestones}
    out = phenology.write_phenology_csv([row], tmp_path / "out.csv", CATKIN)
    written = Path(out).read_text(encoding="utf-8")
    header = written.splitlines()[0].split(",")
    values = written.splitlines()[1].split(",")

    for col in phenology.milestone_date_columns(CATKIN):
        assert f"{col}_bound" in header, col
        assert values[header.index(f"{col}_bound")] == "left_censored", col


_SPEC_SHAPES = [
    # The shape a config-authored trait produces by simply omitting both majority fields, both at
    # their dataclass defaults, which made the phantom names doubly malformed (`b__date`).
    TraitSpec(name="b", milestone_fractions=(0.1, 0.9), phenology_prefix="b"),
    # A majority label with no majority milestone: a column must not be built from the label
    # alone without a milestone to source it.
    TraitSpec(name="b", milestone_fractions=(0.1, 0.9), phenology_prefix="b",
              majority_label="peak"),
    # A majority milestone naming a crossing the trait does not compute: the column is real (the
    # spec names it) and its value is honestly None, which is not the same as a phantom.
    TraitSpec(name="b", milestone_fractions=(0.1, 0.9), phenology_prefix="b",
              majority_milestone="95per", majority_label="peak"),
    CATKIN,
]


@pytest.mark.parametrize("spec", _SPEC_SHAPES, ids=lambda s: f"{s.name}-{s.majority_milestone or 'nomajority'}")
def test_phenology_csv_columns_name_no_column_without_a_producer(spec):
    """Every trait-prefixed column the schema names must be filled by a producer, or the delivered
    CSV carries a permanently-blank column. The schema and the producer share ``_milestone_columns``,
    so this holds by construction for any spec shape, including one whose ``majority_milestone``
    is unset (unlike CATKIN's, which is always set).
    """
    series = [("2026-02-01", 0.0), ("2026-02-10", 0.5), ("2026-02-20", 1.0)]
    produced = set(phenology.plant_milestones(series, spec))
    schema = set(phenology.phenology_csv_columns(spec))
    prefixed = {c for c in schema if c.startswith(spec.phenology_prefix + "_")}
    # The provisional marker is stamped by the delivery tool rather than computed here, and exists
    # only when the spec names a majority alias for it to qualify.
    stamped = ({f"{spec.phenology_prefix}_{spec.majority_label}_provisional"}
               if spec.majority_milestone else set())
    assert prefixed - produced - stamped == set()
    assert produced - schema == set()  # and nothing computed is silently dropped
    assert not any(c.startswith(f"{spec.phenology_prefix}__") for c in schema)


def test_excluded_plant_carries_the_same_milestone_keys_as_an_included_one(tmp_path):
    """A plant excluded from milestone computation must carry the same row shape as an included one,
    including each milestone date's ``*_date_bound`` companion, not just ``milestone_date_columns``'s
    bare dates.
    """
    d1, d2 = tmp_path / "2026-02-11", tmp_path / "2026-03-09"
    id_map = {"dormant": 0, "elongated": 1}
    for d in (d1, d2):
        _sidecar(d, id_map)
    _preds(d1, "GOOD", ["elongated", "dormant"])
    _preds(d2, "GOOD", ["elongated", "elongated"])
    _preds(d1, "BAD", ["elongated", "mystery"])  # unclassifiable -> plant excluded
    _preds(d2, "BAD", ["elongated", "elongated"])
    mapping = {
        "2026-02-11": [_Assignment("GOOD", "GOOD", "a"), _Assignment("BAD", "BAD", "b")],
        "2026-03-09": [_Assignment("GOOD", "GOOD", "a"), _Assignment("BAD", "BAD", "b")],
    }
    res = phenology.per_plant_phenology(
        mapping, {"2026-02-11": str(d1), "2026-03-09": str(d2)},
        positive_class_name="elongated", spec=CATKIN)
    by_plant = {r["plant_id"]: r for r in res["rows"]}
    assert by_plant["BAD"]["n_dates_unclassified"] == 1  # genuinely excluded
    assert set(by_plant["GOOD"]) == set(by_plant["BAD"])
    assert "catkin_95per_date_bound" in by_plant["BAD"]


def test_per_plant_series_counts_the_images_the_mapping_names(tmp_path):
    """``n_images`` is derived from every image the mapping names for a (plant, date), not asserted
    by a consumer, so a breeder auditing coverage can tell a well-sampled plant from a single-photo
    one.
    """
    d = tmp_path / "2026-02-11"
    _sidecar(d, {"dormant": 0, "elongated": 1})
    for i in range(3):
        _preds(d, f"IMG{i}", ["elongated", "dormant"])
    mapping = {"2026-02-11": [_Assignment(f"IMG{i}", "P1", "a") for i in range(3)]
               + [_Assignment("GONE", "P1", "a")]}  # named, no prediction file
    series = phenology.per_plant_series(mapping, {"2026-02-11": str(d)},
                                        positive_class_name="elongated")["P1"]["series"]
    (_date, total, positive, unclassified, missing, n_images) = series[0]
    assert (total, positive, unclassified, missing) == (6, 3, 0, 1)
    # 4 images named for this (plant, date), the coverage the entry summarises, of which one is
    # missing, not 3 (the files that happened to exist).
    assert n_images == 4


def test_write_phenology_csv_carries_n_observed_dates(tmp_path):
    # per_plant_phenology's row already computes n_observed_dates (a plant fully
    # classified/observed but with zero real detections on every date must be distinguishable from
    # one with real detection data). write_phenology_csv is the real MCP delivery door, not the GUI's
    # own per-row rendering in routes/results.py, which already carries this field, so this column
    # must reach the CSV too.
    row = {"plant_id": "P1", "accession": "acc-9", "n_dates": 2, "n_observed_dates": 1,
          "n_dates_unclassified": 0, "n_dates_missing_images": 0}
    out = phenology.write_phenology_csv([row], tmp_path / "out.csv", CATKIN)
    written = Path(out).read_text(encoding="utf-8")
    header = written.splitlines()[0].split(",")
    assert "n_observed_dates" in header
    data_row = written.splitlines()[1].split(",")
    assert data_row[header.index("n_observed_dates")] == "1"
