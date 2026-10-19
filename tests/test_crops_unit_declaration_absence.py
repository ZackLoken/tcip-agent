"""A trait that declares no physical unit stays absent from the unit mapping.

A date, ordinal, ratio, or count trait in crops.yml declares no physical unit, and
``crops_units()`` maps none for it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_mcp.pipelines.measurement.mask_geometry import unit_from_value_key
from tcip_mcp.pipelines.postprocessing.aggregation import _resolve_units
from tcip_mcp.traits import (
    PER_PLANT_COUNT_AGGREGATE,
    PER_PLANT_REGRESSION_AGGREGATE,
    crops_units,
)
from tests import _trait_fixtures as fx, csv_rows


@pytest.fixture(autouse=True)
def _recorded_meaning(tmp_path: Path) -> None:
    """Confirm ``stem_count`` and ``plant_surface_area`` delivered meanings in the project."""
    fx.seed_delivery_traits(tmp_path)
    fx.seed_confirmed_aggregate(
        tmp_path,
        "stem_count",
        value_keys=["detections_count"],
        measured_subject="bud",
    )
    fx.seed_confirmed_aggregate(
        tmp_path,
        "plant_surface_area",
        value_keys=["area_mm2"],
        delivery_kind=PER_PLANT_REGRESSION_AGGREGATE,
    )


def _records(value_key: str, *values: tuple[str, float]) -> list[dict]:
    """Return one per-image record under ``value_key`` per ``(plant_id, value)``."""
    return [
        {"plant_id": plant, value_key: value, "plant_attribution": "image"}
        for plant, value in values
    ]


def _deliver(
    project: Path,
    records: list[dict],
    out: Path,
    delivered_phenotype: str,
    *,
    value_key: str,
    delivery_kind: str,
    buckets: list | None = None,
) -> dict:
    """Run the acknowledged per-plant delivery (``_chain_fixtures.deliver_acknowledged``)."""
    if buckets is None:
        pytest.importorskip("torch")
    from tests._chain_fixtures import deliver_acknowledged

    return deliver_acknowledged(
        project,
        records,
        out,
        delivered_phenotype,
        value_key=value_key,
        delivery_kind=delivery_kind,
        buckets=buckets,
    )


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


def test_declared_units_are_read_verbatim_from_crops_yml() -> None:
    """The declared units are read verbatim from crops.yml.

    The mapping is not empty and carries each declaring trait's own unit, several distinct ones.
    """
    units = crops_units()
    assert units, "crops.yml units should be loadable in the repo checkout"
    for name, unit in DECLARED_UNITS.items():
        assert units[name] == unit


@pytest.mark.parametrize("trait_name", UNDECLARED)
def test_a_trait_declaring_no_unit_is_absent_rather_than_given_one(
    trait_name: str,
) -> None:
    """A crops.yml trait declaring no unit is absent from the unit mapping."""
    from tcip_mcp import traits

    assert trait_name in {t["name"] for t in traits._crops_traits()}, (
        trait_name
    )
    assert trait_name not in crops_units()


@pytest.mark.parametrize(
    "value_key", ["detections_count", "burrs_count", "seeds_count"]
)
def test_a_count_suffixed_value_key_implies_no_unit(value_key: str) -> None:
    """A value_key ending in a count implies no unit, since a count is not a physical unit.

    A key ending in a declared unit, ``detections_mm``, resolves to that unit.
    """
    assert unit_from_value_key(value_key) is None
    # the same key shape with a real declared unit is still recognized, so this is exactness, not
    # a blanket refusal of the pattern
    assert unit_from_value_key("detections_mm") == ("mm", "mm")


def test_a_count_valued_delivery_ships_with_a_blank_units_column(
    tmp_path: Path,
) -> None:
    """A per-plant count delivered for a trait declaring no unit ships with a blank units column.

    The delivery runs end to end and keeps the counted values.
    """
    out_path = tmp_path / "counts.csv"
    _deliver(
        tmp_path,
        _records("detections_count", ("PLANT_001", 7), ("PLANT_014", 2)),
        out_path,
        "stem_count",
        value_key="detections_count",
        delivery_kind=PER_PLANT_COUNT_AGGREGATE,
    )
    rows = csv_rows(out_path)
    assert [r["units"] for r in rows] == ["", ""]
    assert [r["value"] for r in rows] == ["7", "2"]


def test_a_dimensional_value_ships_for_a_trait_crops_yml_declares_no_unit_for(
    tmp_path: Path,
) -> None:
    """A dimensional value ships for a trait crops.yml declares no unit for.

    With no unit declared for the trait, an mm-keyed area resolves to its own squared label,
    and the regression aggregate ships with ``mm2`` in its units column.
    """
    assert _resolve_units("plant_surface_area", "area_mm2", True)[:2] == (
        "mm2",
        "mm",
    )

    out_path = tmp_path / "area.csv"
    _deliver(
        tmp_path,
        _records("area_mm2", ("PLANT_001", 812.5)),
        out_path,
        "plant_surface_area",
        value_key="area_mm2",
        delivery_kind=PER_PLANT_REGRESSION_AGGREGATE,
    )
    rows = csv_rows(out_path)
    assert rows[0]["units"] == "mm2"


def test_a_declared_unit_still_cross_checks_the_value_keys_own_unit(
    tmp_path: Path,
) -> None:
    """A trait declaring ``mm`` refuses a value whose own key implies ``cm``."""
    assert crops_units()["bark_thickness"] == "mm"
    with pytest.raises(ValueError, match="declared units"):
        _deliver(
            tmp_path,
            _records("thickness_cm", ("PLANT_001", 3.4)),
            tmp_path / "mismatch.csv",
            "bark_thickness",
            value_key="thickness_cm",
            delivery_kind=PER_PLANT_COUNT_AGGREGATE,
            buckets=[],
        )
