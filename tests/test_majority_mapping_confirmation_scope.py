"""Each trait's phenology delivery carries its own majority-crossing column, named from its own
prefix and label and computed at the crossing its entry states."""

from __future__ import annotations

from tcip_mcp.pipelines.postprocessing.phenology import milestone_date_columns, plant_milestones
from tests._trait_fixtures import entry

BUD = entry(
    "bud", ["leaf_out_05per_date", "leaf_out_50per_date"], positive_value="open",
    milestone_fractions=[0.05, 0.5, 0.95], milestone_on="positive_fraction",
    majority_milestone="95per", phenology_prefix="bud", majority_label="opening")

PISTILLATE = entry(
    "pistillate", ["pistillate_50per_date", "pistillate_flowering_date"], positive_value="open",
    milestone_fractions=[0.5], milestone_on="positive_fraction", majority_milestone="50per",
    phenology_prefix="pistillate", majority_label="flowering")

SERIES = [("2026-02-11", 0.0), ("2026-02-25", 0.4), ("2026-03-09", 0.6), ("2026-03-23", 1.0)]


def test_each_trait_carries_its_own_majority_crossing_column():
    """bud's majority date is its 95% crossing, pistillate's its 50%: each delivery carries its own
    prefix and label, equal to the crossing its entry names, and no column of the other's."""
    bud = plant_milestones(SERIES, BUD)
    pistillate = plant_milestones(SERIES, PISTILLATE)

    assert bud["bud_opening_date"] == bud["bud_95per_date"]
    assert bud["bud_opening_date"] != bud["bud_50per_date"]
    assert pistillate["pistillate_flowering_date"] == pistillate["pistillate_50per_date"]
    assert "pistillate_flowering_date" not in milestone_date_columns(BUD)
    assert "bud_opening_date" not in milestone_date_columns(PISTILLATE)
