"""Tests for the canonical catkin bloom phenology module.

Bloom is the fraction of a plant's detected catkins that are *elongated*, where
"elongated" is a classifier class (never a geometric proxy). These tests pin the
authoritative trait definitions:

    catkin_05/50/95per_date  = dates the elongated fraction crosses 5/50/95%
    catkin_elongation_date   = first date any elongation appears (fraction > 0)

and the guard that makes an unclassified prediction set impossible to pass off as a
bloom measurement (elongation_classified == False).
"""

from __future__ import annotations

import sys
from pathlib import Path

_MCP_SRC = Path(__file__).resolve().parents[1] / "packages" / "tcip-mcp" / "src"
if str(_MCP_SRC) not in sys.path:
    sys.path.insert(0, str(_MCP_SRC))

from tcip_mcp.pipelines.postprocessing import phenology  # noqa: E402


# ── date helpers ─────────────────────────────────────────────────────────


def test_date_key_orders_chronologically():
    assert phenology.date_key("2024-05-01") < phenology.date_key("2024-05-15")
    assert phenology.date_key("2024-05-15") < phenology.date_key("2024-06-01")


def test_date_key_malformed_sorts_first():
    # The undated/ bucket (and anything non-ISO) sorts before real dates and is
    # excluded from milestone math.
    assert phenology.date_key("undated") == (0, 0, 0)
    assert phenology.date_key("2024-13") == (0, 0, 0)
    assert phenology.date_key("not-a-date-x") == (0, 0, 0)


def test_date_key_rejects_out_of_range_dates():
    # Three integers but not a calendar-legal date — must collapse to the sentinel,
    # not survive into crossing_date and raise date(2026, 13, 1).
    assert phenology.date_key("2026-13-01") == (0, 0, 0)
    assert phenology.date_key("2026-02-30") == (0, 0, 0)


def test_crossing_date_does_not_crash_on_malformed_date():
    # A bogus date folder among real ones must be dropped, not crash interpolation.
    series = [
        ("2026-02-01", 0.0),
        ("2026-13-01", 0.9),  # out-of-range month — excluded
        ("2026-02-15", 1.0),
    ]
    assert phenology.crossing_date(series, 0.50) == "2026-02-08"
    assert phenology.elongation_onset_date(series) == "2026-02-15"


def test_real_points_drops_undated_and_sorts():
    series = [
        ("2024-05-15", 0.5),
        ("undated", 0.9),
        ("2024-05-01", 0.1),
    ]
    pts = phenology._real_points(series)
    assert [d for d, _ in pts] == ["2024-05-01", "2024-05-15"]


# ── crossings ────────────────────────────────────────────────────────────


def test_crossing_interpolates_between_dates():
    # 0.0 on May 1, 1.0 on May 11 → 50% crossing lands at the midpoint, May 6.
    series = [("2024-05-01", 0.0), ("2024-05-11", 1.0)]
    assert phenology.crossing_date(series, 0.50) == "2024-05-06"


def test_crossing_first_point_already_at_target():
    series = [("2024-05-01", 0.2), ("2024-05-05", 0.8)]
    assert phenology.crossing_date(series, 0.05) == "2024-05-01"


def test_crossing_never_reached_returns_none():
    series = [("2024-05-01", 0.0), ("2024-05-05", 0.3)]
    assert phenology.crossing_date(series, 0.95) is None


def test_elongation_onset_is_first_nonzero_date():
    series = [
        ("2024-05-01", 0.0),
        ("2024-05-05", 0.0),
        ("2024-05-09", 0.10),
        ("2024-05-13", 0.60),
    ]
    assert phenology.elongation_onset_date(series) == "2024-05-09"


def test_elongation_onset_none_when_all_zero():
    series = [("2024-05-01", 0.0), ("2024-05-05", 0.0)]
    assert phenology.elongation_onset_date(series) is None


def test_plant_milestones_returns_four_dates():
    series = [
        ("2024-05-01", 0.0),
        ("2024-05-06", 0.04),
        ("2024-05-11", 0.50),
        ("2024-05-21", 1.0),
    ]
    m = phenology.plant_milestones(series)
    assert set(m) == {
        "catkin_05per_date",
        "catkin_50per_date",
        "catkin_95per_date",
        "catkin_elongation_date",
    }
    # onset = first date with any elongation (discrete observation). The 05/50/95 crossings
    # interpolate. Here onset (first >0 is the 0.04 point) equals the 05% crossing because
    # the fraction rises gradually; with a sparse 0→1 jump the discrete onset can land LATER
    # than the interpolated 05% crossing — see test_onset_can_trail_interpolated_05per.
    assert m["catkin_elongation_date"] == "2024-05-06"
    assert m["catkin_50per_date"] == "2024-05-11"


def test_onset_can_trail_interpolated_05per():
    # Definitional subtlety (locked by Zack): onset is the first OBSERVED elongation, the
    # crossings are interpolated ESTIMATES. On a sparse 0→1 jump the estimate lands earlier
    # than the first observation, so catkin_elongation_date > catkin_05per_date. This is
    # intended, not a bug — the two are different quantities.
    series = [("2024-05-01", 0.0), ("2024-05-15", 1.0)]
    m = phenology.plant_milestones(series)
    assert m["catkin_elongation_date"] == "2024-05-15"  # first observed
    assert m["catkin_05per_date"] == "2024-05-02"  # interpolated estimate (earlier)


# ── elongated-fraction from classified predictions ───────────────────────


def test_count_by_class_counts_elongated_by_class_not_geometry(tmp_path):
    # Two detections: class 0 (not elongated) and class 1 (elongated). Geometry
    # (the last four columns) is irrelevant — elongation is the class id.
    p = tmp_path / "img.txt"
    p.write_text(
        "0 0.9 0.5 0.5 0.02 0.30\n"  # tall box, but class 0 → NOT elongated
        "1 0.8 0.4 0.4 0.40 0.02\n",  # short box, but class 1 → elongated
        encoding="utf-8",
    )
    total, elongated, classes_seen = phenology.count_by_class(p, elongated_class_id=1)
    assert (total, elongated) == (2, 1)
    assert classes_seen == {0, 1}


def test_count_by_class_missing_file_is_empty():
    total, elongated, classes_seen = phenology.count_by_class(
        Path("does-not-exist.txt"), elongated_class_id=1
    )
    assert (total, elongated, classes_seen) == (0, 0, set())


class _Assignment:
    """Mimics the plant_mapping Assignment record (attribute access)."""

    def __init__(self, stem, plot_name, accession_name):
        self.stem = stem
        self.plot_name = plot_name
        self.accession_name = accession_name


def _write_preds(dir_path: Path, stem: str, lines: list[str]) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_per_plant_phenology_builds_fraction_series(tmp_path):
    # Plant P1: early date 0/2 elongated, later date 2/2 elongated.
    d1 = tmp_path / "2024-05-01"
    d2 = tmp_path / "2024-05-15"
    _write_preds(d1, "P1_a", ["0 0.9 0.5 0.5 0.1 0.1", "0 0.8 0.4 0.4 0.1 0.1"])
    _write_preds(d2, "P1_b", ["1 0.9 0.5 0.5 0.1 0.1", "1 0.8 0.4 0.4 0.1 0.1"])
    mapping = {
        "2024-05-01": [_Assignment("P1_a", "P1", "acc-9")],
        "2024-05-15": [_Assignment("P1_b", "P1", "acc-9")],
    }
    preds = {"2024-05-01": str(d1), "2024-05-15": str(d2)}

    out = phenology.per_plant_phenology(mapping, preds, elongated_class_id=1)

    assert out["elongation_classified"] is True
    assert out["classes_seen"] == [0, 1]
    assert len(out["rows"]) == 1
    row = out["rows"][0]
    assert row["plant_id"] == "P1"
    assert row["accession"] == "acc-9"
    assert row["n_dates"] == 2
    # Fraction rises 0.0 (May 1) → 1.0 (May 15). Onset is the first non-zero
    # observation (May 15); the crossings interpolate between the two dates:
    # 50% at the midpoint (May 8), 95% near the top (May 14).
    assert row["catkin_elongation_date"] == "2024-05-15"
    assert row["catkin_50per_date"] == "2024-05-08"
    assert row["catkin_95per_date"] == "2024-05-14"


def test_per_plant_phenology_excludes_zero_detection_date_from_milestones(tmp_path):
    # A date with total==0 (nothing detected) isn't an observation of the elongated fraction
    # (pre-emergence or a detection gap), so it's excluded from the milestone series and kept in
    # the raw series with ratio=None. A total>0/elongated==0 date is a real 0% and is kept.
    # Excluding this date moves the 50% crossing to the first real observation (05-15).
    d0 = tmp_path / "2024-05-01"
    d1 = tmp_path / "2024-05-15"
    d0.mkdir(parents=True, exist_ok=True)
    (d0 / "P1_a.txt").write_text("", encoding="utf-8")  # zero detections on this date
    _write_preds(d1, "P1_b", ["1 0.9 0.5 0.5 0.1 0.1", "1 0.8 0.4 0.4 0.1 0.1"])
    mapping = {
        "2024-05-01": [_Assignment("P1_a", "P1", "acc-9")],
        "2024-05-15": [_Assignment("P1_b", "P1", "acc-9")],
    }
    preds = {"2024-05-01": str(d0), "2024-05-15": str(d1)}

    out = phenology.per_plant_phenology(mapping, preds, elongated_class_id=1)
    row = out["rows"][0]
    # the zero-detection date is kept in the raw series but flagged as no-observation (ratio None)
    assert row["series"][0] == {"date": "2024-05-01", "n_total": 0, "n_elongated": 0, "ratio": None}
    assert row["n_dates"] == 2 and row["n_observed_dates"] == 1
    # ...and it does not feed the milestones: onset + 50% crossing are the first real observation.
    assert row["catkin_elongation_date"] == "2024-05-15"
    assert row["catkin_50per_date"] == "2024-05-15"


def test_per_plant_phenology_flags_unclassified_predictions(tmp_path):
    # Predictions carry only class 0 — no elongation class anywhere. The fraction
    # is meaningless as bloom, and the guard must say so.
    d1 = tmp_path / "2024-05-01"
    _write_preds(d1, "P1_a", ["0 0.9 0.5 0.5 0.1 0.1"])
    mapping = {"2024-05-01": [_Assignment("P1_a", "P1", "acc-9")]}
    preds = {"2024-05-01": str(d1)}

    out = phenology.per_plant_phenology(mapping, preds, elongated_class_id=1)

    assert out["elongation_classified"] is False
    assert out["classes_seen"] == [0]


def test_per_plant_series_accepts_dict_assignments(tmp_path):
    # Assignments may be plain dicts (e.g. loaded from persisted mapping.json).
    d1 = tmp_path / "2024-05-01"
    _write_preds(d1, "P1_a", ["1 0.9 0.5 0.5 0.1 0.1"])
    mapping = {
        "2024-05-01": [
            {"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}
        ]
    }
    preds = {"2024-05-01": str(d1)}
    per_plant, classes = phenology.per_plant_series(mapping, preds, elongated_class_id=1)
    assert "P1" in per_plant
    assert per_plant["P1"]["accession"] == "acc-9"
    assert classes == {1}
