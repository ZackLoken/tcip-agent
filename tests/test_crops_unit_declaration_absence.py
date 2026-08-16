"""A trait that declares no physical unit stays absent from the unit mapping.

``crops_units()`` is the unit authority two deliveries read: the aggregated CSV's units column
cross-check, and the vocabulary of trailing tokens a value_key may imply a unit with. A date,
ordinal, ratio, or count trait has no physical unit, and giving it one there both labels a
delivered number with a unit nobody declared and widens the token vocabulary every value_key is
matched against.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tcip_mcp.pipelines.postprocessing.aggregation import (
    _resolve_units,
    _unit_from_value_key,
    export_aggregated_csv,
)
from tcip_mcp.traits import crops_units
from tests import _operationalization_fixtures as fx


@pytest.fixture(autouse=True)
def _recorded_meaning(tmp_path: Path):
    """Every export below ships under a trait whose delivered number has a confirmed meaning."""
    fx.seed_delivery_traits(tmp_path)
    fx.seed_confirmed_aggregate(tmp_path, "stem_count",
                                value_keys=["detections_count"])
    fx.seed_confirmed_aggregate(tmp_path, "plant_surface_area", value_keys=["area_mm2"])

# crops.yml traits that declare a physical unit, one per declared unit shape it uses.
DECLARED_UNITS = {
    "bark_thickness": "mm",
    "efb_canker_length": "cm",
    "dbh": "m",
    "kernel_weight": "g",
    "plant_yield": "kg",
    "soluble_tannins": "ug/g",
}

# crops.yml traits that declare no unit at all: a date, an ordinal rating, a ratio, a count, and a
# dimensional trait whose unit was simply never authored.
UNDECLARED = [
    "catkin_95per_date",
    "astringency",
    "leaf_length_width_ratio",
    "stem_count",
    "plant_surface_area",
]


def test_declared_units_are_read_verbatim_from_crops_yml():
    """The mapping carries each declaring trait's own unit, several distinct ones, so a later
    assertion that some trait is absent cannot pass merely because the mapping is empty."""
    units = crops_units()
    assert units, "crops.yml units should be loadable in the repo checkout"
    for name, unit in DECLARED_UNITS.items():
        assert units[name] == unit


@pytest.mark.parametrize("trait_name", UNDECLARED)
def test_a_trait_declaring_no_unit_is_absent_rather_than_given_one(trait_name: str):
    """Absent from the mapping, not mapped to a plausible-looking stand-in. A guessed unit here is
    a fabricated measurement claim: it would be cross-checked against, and shipped in a delivered
    units column, as though a breeder had declared it."""
    from tcip_mcp import traits

    assert trait_name in traits._crops_vocab(), trait_name
    assert trait_name not in crops_units()


@pytest.mark.parametrize("value_key", ["detections_count", "burrs_count", "seeds_count"])
def test_a_count_suffixed_value_key_implies_no_unit(value_key: str):
    """A count is not a physical unit, so a value_key ending in it implies none. The recognized
    tokens are exactly crops.yml's declared units and their squared forms, so a trailing token that
    entered the vocabulary through an undeclared trait would start labeling counts as measured."""
    assert _unit_from_value_key(value_key) is None
    # the same key shape with a real declared unit is still recognized, so this is exactness, not
    # a blanket refusal of the pattern
    assert _unit_from_value_key("detections_mm") == ("mm", "mm")


def test_a_count_valued_delivery_ships_with_a_blank_units_column(tmp_path: Path):
    """End to end: a per-plant count, delivered for a trait crops.yml declares no unit for, carries
    an empty units column. A unit in that column would read as a physical measurement of a number
    that is a tally."""
    results = [
        {"plant_id": "PLANT_001", "value": 7, "observations": 3, "value_key": "detections_count"},
        {"plant_id": "PLANT_014", "value": 2, "observations": 1, "value_key": "detections_count"},
    ]
    out_path = tmp_path / "counts.csv"
    export_aggregated_csv(results, str(out_path), trait_name="stem_count",
                          acknowledge_unvalidated=True)
    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["units"] for r in rows] == ["", ""]
    assert [r["value"] for r in rows] == ["7", "2"]


def test_a_dimensional_value_ships_for_a_trait_crops_yml_declares_no_unit_for(tmp_path: Path):
    """crops.yml declaring no unit for a trait means there is nothing to cross-check against, not
    that the measurement is refused: an mm-keyed area still resolves to its own squared label. A
    stand-in unit in the mapping would turn this legitimate delivery into a mismatch refusal."""
    results = [{"plant_id": "PLANT_001", "value": 812.5, "observations": 2,
                "value_key": "area_mm2"}]
    assert _resolve_units("plant_surface_area", results) == "mm2"

    out_path = tmp_path / "area.csv"
    export_aggregated_csv(results, str(out_path), trait_name="plant_surface_area",
                          acknowledge_unvalidated=True)
    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["units"] == "mm2"


def test_a_declared_unit_still_cross_checks_the_value_keys_own_unit(tmp_path: Path):
    """The absence rule above must not cost the check it exists to serve: a trait crops.yml does
    declare a unit for still refuses a value whose own key implies a different one."""
    results = [{"plant_id": "PLANT_001", "value": 3.4, "observations": 1,
                "value_key": "thickness_cm"}]
    assert crops_units()["bark_thickness"] == "mm"
    with pytest.raises(ValueError, match="declared units"):
        export_aggregated_csv(results, str(tmp_path / "mismatch.csv"),
                              trait_name="bark_thickness", acknowledge_unvalidated=True)
