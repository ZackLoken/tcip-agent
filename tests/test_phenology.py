"""Tests for the canonical bloom phenology module.

Bloom is the fraction of a plant's detected objects that are in the trait's positive/measured
state, where that state is a classifier call (never a geometric proxy). These tests pin the
authoritative trait definitions:

    catkin_05/50/95per_date  = dates the elongated fraction crosses 5/50/95%
    catkin_elongation_date   = date most catkins have elongated (crops.yml) = the 95% crossing

and — the highest-scrutiny part of this module (Group A's K4/K5, five rounds of adversarial
design review) — the id_map-key-membership coverage rule that decides whether a prediction
bucket ever assessed the trait's positive class at all, and the per-detection membership check
within it. See docs/decisions/cluster-map.md's Group A history for the four prior formulations
that broke and why.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_MCP_SRC = Path(__file__).resolve().parents[1] / "packages" / "tcip-mcp" / "src"
if str(_MCP_SRC) not in sys.path:
    sys.path.insert(0, str(_MCP_SRC))

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox  # noqa: E402
from tcip_mcp.pipelines.postprocessing import phenology  # noqa: E402
from tcip_mcp.traits import CATKIN  # noqa: E402


def _sidecar(dir_path: Path, id_map: dict | None) -> None:
    """Write a bucket's operating_point.json exactly the way export_predictions does — the only
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
        ("2026-13-01", 0.9),  # out-of-range month — excluded
        ("2026-02-15", 1.0),
    ]
    c = phenology.crossing_date(series, 0.50)
    assert c.date == "2026-02-08"
    assert c.bound == "interpolated"
    assert phenology.elongation_onset_date(series) == "2026-02-15"


def test_real_points_drops_undated_and_sorts():
    series = [("2024-05-15", 0.5), ("undated", 0.9), ("2024-05-01", 0.1)]
    pts = phenology._real_points(series)
    assert [d for d, _ in pts] == ["2024-05-01", "2024-05-15"]


# ── crossings (K4: censoring bound disclosed) ─────────────────────────────


def test_crossing_interpolates_between_dates():
    series = [("2024-05-01", 0.0), ("2024-05-11", 1.0)]
    c = phenology.crossing_date(series, 0.50)
    assert c.date == "2024-05-06"
    assert c.bound == "interpolated"
    assert c.gap_days == 10


def test_crossing_first_point_already_at_target_is_left_censored():
    # K4: the prior bare-tuple return made a left-censored crossing indistinguishable from a real
    # single-date measurement. It must be flagged.
    series = [("2024-05-01", 0.2), ("2024-05-05", 0.8)]
    c = phenology.crossing_date(series, 0.05)
    assert c.date == "2024-05-01"
    assert c.bound == "left_censored"


def test_crossing_exact_match_is_not_censored():
    series = [("2024-05-01", 0.0), ("2024-05-05", 0.50)]
    c = phenology.crossing_date(series, 0.50)
    assert c.date == "2024-05-05"
    assert c.bound == "exact"


def test_crossing_never_reached_returns_none():
    series = [("2024-05-01", 0.0), ("2024-05-05", 0.3)]
    assert phenology.crossing_date(series, 0.95) is None


def test_elongation_onset_is_first_nonzero_date():
    series = [("2024-05-01", 0.0), ("2024-05-05", 0.0), ("2024-05-09", 0.10), ("2024-05-13", 0.60)]
    assert phenology.elongation_onset_date(series) == "2024-05-09"


def test_elongation_onset_none_when_all_zero():
    series = [("2024-05-01", 0.0), ("2024-05-05", 0.0)]
    assert phenology.elongation_onset_date(series) is None


def test_plant_milestones_returns_four_dates_and_bounds():
    series = [("2024-05-01", 0.0), ("2024-05-06", 0.04), ("2024-05-11", 0.50), ("2024-05-21", 1.0)]
    m = phenology.plant_milestones(series, CATKIN)
    assert {"catkin_05per_date", "catkin_50per_date", "catkin_95per_date", "catkin_elongation_date"} <= set(m)
    assert m["catkin_elongation_date"] == m["catkin_95per_date"]
    assert m["catkin_50per_date"] == "2024-05-11"
    assert m["catkin_50per_date_bound"] == "exact"


def test_plant_milestones_requires_spec_no_catkin_fallback():
    # K4/K5: no silent default — a caller that forgets to pass spec must fail loudly.
    import pytest

    with pytest.raises(TypeError):
        phenology.plant_milestones([("2024-05-01", 0.5)])


def test_milestone_date_columns_is_proper_subset_of_full_columns():
    cols = phenology.phenology_csv_columns(CATKIN)
    milestone_cols = phenology.milestone_date_columns(CATKIN)
    assert set(milestone_cols) <= set(cols)
    assert "plant_id" not in milestone_cols
    assert "positive_state_classifier_validated" not in milestone_cols


# ── count_by_class: the coverage mechanism (5 rounds of adversarial review) ─────────────────────


def test_count_by_class_bare_detector_bucket_refuses_never_full_coverage(tmp_path):
    # THE round-3/round-4 inversion case: a single-class detector's id_map is {"catkin": 0} — no
    # attribute axis at all. Every prediction decodes to subject="catkin". This must NOT be scored
    # classified — "catkin" is not the trait's positive value.
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
    # A run classified on a DIFFERENT attribute of the same subject (e.g. damage severity) — its
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
    # round-5's severe finding: a classified bucket (id_map has the positive value) must still not
    # silently coerce a detection whose OWN subject isn't a key of that map (a stale file from a
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
    # []) — but per_plant_series never actually calls it on a missing path; it checks is_file() and
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

    assert out["elongation_classified"] is True
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

    assert out["elongation_classified"] is False
    row = out["rows"][0]
    assert row["n_dates_unclassified"] == 1
    assert row["catkin_50per_date"] is None


def test_per_plant_phenology_missing_image_is_disclosed_not_a_zero(tmp_path):
    # round-5 finding A-NEW-2: a stem the mapping names with no prediction file must not read as an
    # observed zero (which would count as "classified, 0/0" and silently pass coverage).
    d1 = tmp_path / "2024-05-01"
    d1.mkdir(parents=True, exist_ok=True)
    _sidecar(d1, {"dormant": 0, "elongated": 1})
    # no P1_a.json written — the mapping names it but nothing was ever inferred for it
    mapping = {"2024-05-01": [_Assignment("P1_a", "P1", "acc-9")]}
    preds = {"2024-05-01": str(d1)}

    out = phenology.per_plant_phenology(mapping, preds, positive_class_name="elongated", spec=CATKIN)

    row = out["rows"][0]
    assert row["series"][0]["n_missing"] == 1
    assert row["series"][0]["ratio"] is None
    assert row["n_dates_missing_images"] == 1
    assert out["elongation_classified"] is False  # nothing anywhere was actually classified
    assert row["catkin_50per_date"] is None


def test_per_plant_phenology_multi_date_and_excludes_plant_with_one_bad_date(tmp_path):
    # round-5 finding A6: a plant's milestones require EVERY date to be classified — one
    # unclassified or missing date excludes the whole plant's milestones, disclosed, not silently
    # computed from the subset that happened to be usable.
    d1 = tmp_path / "2024-05-01"
    d2 = tmp_path / "2024-05-15"
    _preds(d1, "P1_a", ["elongated"])
    _sidecar(d1, {"dormant": 0, "elongated": 1})
    _preds(d2, "P1_b", ["catkin"])  # bare-detector date — unclassified
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
    # The whole plant's milestones are None — one bad date excludes it, not a partial computation.
    assert row["catkin_05per_date"] is None
    assert row["catkin_95per_date"] is None
    # But at least one date elsewhere WAS classified, so the delivery-level flag is still True —
    # distinguishing "wired, some gaps" from "never wired at all".
    assert out["elongation_classified"] is True


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
    """Round-4 review: plant_milestones has always emitted a `_bound` beside each milestone date
    (exact / interpolated / left_censored), but none were in phenology_csv_columns, so the writer's
    extrasaction="ignore" dropped them all. A LEFT_CENSORED crossing — the first observation already
    met the target, so the date is only an UPPER BOUND — then shipped indistinguishable from a
    measured one, which is a precision claim the data does not support."""
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


def test_phenology_csv_columns_name_no_column_without_a_producer(tmp_path):
    """The other half of the bound fix: every trait-prefixed column the schema names must actually
    be filled by plant_milestones, or the CSV ships a permanently-blank column — the same phantom
    the round-4 review found claimed (a `gap_days` column no producer emits)."""
    series = [("2026-02-01", 0.0), ("2026-02-10", 0.5), ("2026-02-20", 1.0)]
    produced = set(phenology.plant_milestones(series, CATKIN))
    schema = set(phenology.phenology_csv_columns(CATKIN))
    prefixed = {c for c in schema if c.startswith(CATKIN.phenology_prefix + "_")}
    provisional = f"{CATKIN.phenology_prefix}_{CATKIN.majority_label}_provisional"  # stamped, not computed
    assert prefixed - produced - {provisional} == set()
    assert produced - schema == set()  # and nothing computed is silently dropped


def test_write_phenology_csv_carries_n_observed_dates(tmp_path):
    # Round-4 review N6: per_plant_phenology's row already computes n_observed_dates (a plant fully
    # classified/observed but with zero real detections on every date must be distinguishable from
    # one with real bloom data) — it was silently dropped by extrasaction="ignore" for not being in
    # phenology_csv_columns. This is the real MCP delivery door (write_phenology_csv), not the GUI's
    # own per-row rendering in routes/results.py, which already carried this field.
    row = {"plant_id": "P1", "accession": "acc-9", "n_dates": 2, "n_observed_dates": 1,
          "n_dates_unclassified": 0, "n_dates_missing_images": 0}
    out = phenology.write_phenology_csv([row], tmp_path / "out.csv", CATKIN)
    written = Path(out).read_text(encoding="utf-8")
    header = written.splitlines()[0].split(",")
    assert "n_observed_dates" in header
    data_row = written.splitlines()[1].split(",")
    assert data_row[header.index("n_observed_dates")] == "1"
