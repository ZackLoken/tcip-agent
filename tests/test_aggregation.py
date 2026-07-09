"""Tests for per-plant aggregation postprocessing.

Covers the plant_id extraction fallback (the source of a delivery-CSV
fragmentation bug), the explicit plant_id_key / plant_id_fn override paths, the
aggregation strategies, and the sigmoid time-axis extraction.
"""

from __future__ import annotations

import csv
import logging

import pytest

from tcip_mcp.pipelines.postprocessing.aggregation import (
    _extract_plant_id,
    aggregate_per_plant,
    export_aggregated_csv,
)


# ── _extract_plant_id (the guessing fallback) ───────────────────────────────


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        # Two-token flight suffix: heuristic recovers the intended id.
        ("bush_42_flight_3", "bush_42"),
        ("bush_42_flight_2", "bush_42"),
        # Three-token YYYY_MM_DD date: only two tokens are stripped, so a year
        # token is retained. This documents the (intended, minimal) behavior —
        # the plant-year fragmentation the caller warns about.
        ("PLANT_001_2024_05_15", "PLANT_001_2024"),
        # Single token: nothing to strip, whole stem returned.
        ("tree12", "tree12"),
        # Filename with extension is reduced to its stem first.
        ("bush_42_flight_3.jpg", "bush_42"),
    ],
)
def test_extract_plant_id_matches_documented_behavior(stem, expected):
    assert _extract_plant_id(stem) == expected


def test_extract_plant_id_two_tokens():
    # 'PLANT_001' → rsplit('_', 2) yields ['PLANT', '001']; strips to 'PLANT'.
    # Documents that a bare id with a numeric suffix is NOT preserved by the
    # fallback — callers with such names must pass plant_id_fn / plant_id key.
    assert _extract_plant_id("PLANT_001") == "PLANT"


# ── grouping paths ──────────────────────────────────────────────────────────


def test_explicit_plant_id_key_takes_precedence():
    results = [
        {"image": "PLANT_001_2024_05_15", "plant_id": "PLANT_001", "count": 3},
        {"image": "PLANT_001_2024_06_20", "plant_id": "PLANT_001", "count": 5},
    ]
    out = aggregate_per_plant(results, strategy="count", value_key="count")
    assert len(out) == 1
    assert out[0]["plant_id"] == "PLANT_001"
    assert out[0]["observations"] == 2


def test_plant_id_fn_override_keeps_series_together():
    # A caller-supplied fn that strips the trailing YYYY_MM_DD date keeps one
    # plant's multi-date series in a single group (no plant-year fragmentation).
    def strip_date(image_name: str) -> str:
        return image_name.rsplit("_", 3)[0]

    results = [
        {"image": "PLANT_001_2024_05_15", "count": 1},
        {"image": "PLANT_001_2024_06_20", "count": 4},
        {"image": "PLANT_001_2025_05_15", "count": 2},
    ]
    out = aggregate_per_plant(
        results, strategy="count", value_key="count", plant_id_fn=strip_date
    )
    assert len(out) == 1
    assert out[0]["plant_id"] == "PLANT_001"
    assert out[0]["observations"] == 3


def test_fallback_fragments_multiyear_series_and_warns(caplog):
    # Without an override, the YYYY_MM_DD naming fragments one plant into two
    # plant-year groups — the documented delivery-CSV bug. The caller must warn.
    results = [
        {"image": "PLANT_001_2024_05_15", "count": 1},
        {"image": "PLANT_001_2024_06_20", "count": 4},
        {"image": "PLANT_001_2025_05_15", "count": 2},
    ]
    with caplog.at_level(logging.WARNING):
        out = aggregate_per_plant(results, strategy="count", value_key="count")

    plant_ids = sorted(r["plant_id"] for r in out)
    assert plant_ids == ["PLANT_001_2024", "PLANT_001_2025"]
    assert any("guessed" in rec.message for rec in caplog.records)


def test_fallback_no_warning_when_key_present(caplog):
    results = [{"image": "x", "plant_id": "P1", "count": 1}]
    with caplog.at_level(logging.WARNING):
        aggregate_per_plant(results, strategy="count", value_key="count")
    assert not any("guessed" in rec.message for rec in caplog.records)


def test_missing_image_key_uses_unknown_bucket():
    results = [{"count": 1}, {"count": 3}]
    out = aggregate_per_plant(results, strategy="count", value_key="count")
    assert len(out) == 1
    assert out[0]["plant_id"] == "unknown"


# ── strategies ──────────────────────────────────────────────────────────────


def test_count_strategy_median():
    results = [
        {"image": "bush_1_f_1", "count": 2},
        {"image": "bush_1_f_2", "count": 4},
        {"image": "bush_1_f_3", "count": 9},
    ]
    out = aggregate_per_plant(results, strategy="count", value_key="count")
    assert out[0]["value"] == 4
    assert out[0]["min_count"] == 2
    assert out[0]["max_count"] == 9


def test_mean_strategy():
    results = [
        {"image": "bush_1_f_1", "value": 1.0},
        {"image": "bush_1_f_2", "value": 3.0},
    ]
    out = aggregate_per_plant(results, strategy="mean", value_key="value")
    assert out[0]["value"] == 2.0


def test_mode_strategy():
    results = [
        {"image": "bush_1_f_1", "grade": 2},
        {"image": "bush_1_f_2", "grade": 2},
        {"image": "bush_1_f_3", "grade": 5},
    ]
    out = aggregate_per_plant(results, strategy="mode", value_key="grade")
    assert out[0]["value"] == 2
    assert out[0]["agreement"] == pytest.approx(2 / 3, abs=1e-4)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown aggregation strategy"):
        aggregate_per_plant([{"image": "a_b_c", "count": 1}], strategy="nope")


# ── sigmoid time-axis extraction ────────────────────────────────────────────


def test_sigmoid_milestones_basic():
    # A monotonic rise across four time points; 50% crossing interpolated.
    results = [
        {"image": "p_1_1", "day_of_year": 100, "count": 0.0},
        {"image": "p_1_2", "day_of_year": 110, "count": 2.0},
        {"image": "p_1_3", "day_of_year": 120, "count": 8.0},
        {"image": "p_1_4", "day_of_year": 130, "count": 10.0},
    ]
    out = aggregate_per_plant(results, strategy="sigmoid", value_key="count")
    fit = out[0]["fit"]
    assert fit["n_points"] == 4
    assert fit["milestones"]["50per"] is not None


def test_sigmoid_time_index_zero_is_not_dropped():
    # time_index=0 is a valid first time point; the explicit is-not-None
    # precedence must include it (an ``or``-chain would drop the falsy 0).
    results = [
        {"image": "p_1_1", "time_index": 0, "count": 0.0},
        {"image": "p_1_2", "time_index": 1, "count": 5.0},
        {"image": "p_1_3", "time_index": 2, "count": 10.0},
    ]
    out = aggregate_per_plant(results, strategy="sigmoid", value_key="count")
    assert out[0]["fit"]["n_points"] == 3


def test_sigmoid_day_of_year_zero_is_not_dropped():
    # A falsy 0 in the first (highest-precedence) time key must survive.
    results = [
        {"image": "p_1_1", "day_of_year": 0, "count": 0.0},
        {"image": "p_1_2", "day_of_year": 5, "count": 5.0},
        {"image": "p_1_3", "day_of_year": 10, "count": 10.0},
    ]
    out = aggregate_per_plant(results, strategy="sigmoid", value_key="count")
    assert out[0]["fit"]["n_points"] == 3


def test_sigmoid_insufficient_data():
    results = [
        {"image": "p_1_1", "day_of_year": 100, "count": 1.0},
        {"image": "p_1_2", "day_of_year": 110, "count": 2.0},
    ]
    out = aggregate_per_plant(results, strategy="sigmoid", value_key="count")
    assert out[0]["fit"]["error"] == "insufficient_data"


# ── CSV export ──────────────────────────────────────────────────────────────


def test_export_aggregated_csv(tmp_path):
    results = [
        {"plant_id": "PLANT_001", "value": 7, "observations": 3},
        {"plant_id": "PLANT_002", "value": 4, "observations": 2},
    ]
    out_path = tmp_path / "out" / "aggregated.csv"
    export_aggregated_csv(
        results, str(out_path), trait_name="catkin_count", crop="hazelnut"
    )

    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert [r["plant_id"] for r in rows] == ["PLANT_001", "PLANT_002"]
    assert rows[0]["crop"] == "hazelnut"
    assert rows[0]["trait_name"] == "catkin_count"
    assert rows[0]["n_images"] == "3"
