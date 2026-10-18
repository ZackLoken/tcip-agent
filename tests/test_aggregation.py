"""Per-plant aggregation and its delivery.

A record must carry its ``plant_id``, the assignment rows' ``source``/``distance_m`` pass
through, the units column comes from ``crops.yml``, each statistic aggregates, and a delivery
aggregates per-image records under its trait's ``plant_aggregation``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tcip_mcp.pipelines.postprocessing.aggregation import (
    _per_plant,
    deliver_per_plant_aggregate,
)
from tcip_mcp.traits import (
    PER_PLANT_COUNT_AGGREGATE,
    PER_PLANT_REGRESSION_AGGREGATE,
)
from tests import _trait_fixtures as fx, csv_rows


@pytest.fixture(autouse=True)
def _recorded_meaning(tmp_path):
    """Confirm the delivery traits and three per-plant aggregate meanings.

    They are ``stem_count``'s count of ``bud``, and ``plant_surface_area``'s and
    ``bark_thickness``'s regressions.
    """
    fx.seed_delivery_traits(tmp_path)
    # The count measures what the chain's unassessed bucket detects.
    fx.seed_confirmed_aggregate(
        tmp_path, "stem_count", value_keys=["count"], measured_subject="bud"
    )
    fx.seed_confirmed_aggregate(
        tmp_path,
        "plant_surface_area",
        value_keys=["area_mm2"],
        delivery_kind=PER_PLANT_REGRESSION_AGGREGATE,
    )
    fx.seed_confirmed_aggregate(
        tmp_path,
        "bark_thickness",
        value_keys=["principal_axis_extent_px"],
        delivery_kind=PER_PLANT_REGRESSION_AGGREGATE,
    )


def _deliver(
    project: Path,
    records: list[dict],
    out: Path,
    delivered_phenotype: str,
    *,
    value_key: str,
    delivery_kind: str = PER_PLANT_COUNT_AGGREGATE,
    buckets=None,
    **kwargs,
):
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
        **kwargs,
    )


# ── plant identity: no guessing, ever ──────────────────────


@pytest.mark.parametrize(
    "record",
    [
        {"image": "bush_42_flight_3", "count": 1},
        {"count": 1},
        {"image": "x", "plant_id": "", "count": 1},
        {"image": "unmapped_img", "plant_id": None, "count": 2},
    ],
)
def test_a_record_carrying_no_plant_id_refuses_naming_the_mapping(record):
    """A record carrying no ``plant_id`` is refused naming the primitive that assigns plants.

    A record whose ``plant_id`` is absent, empty or ``None`` (an unmapped assignment) never
    groups under a missing key.
    """
    with pytest.raises(ValueError, match="build_plant_mapping"):
        _per_plant([record], plant_aggregation="median", value_key="count")


def test_records_group_by_their_plant_id():
    records = [
        {"image": "PLANT_001_2024_05_15", "plant_id": "PLANT_001", "count": 3},
        {"image": "PLANT_001_2024_06_20", "plant_id": "PLANT_001", "count": 5},
    ]
    out = _per_plant(records, plant_aggregation="median", value_key="count")
    assert len(out) == 1
    assert out[0]["plant_id"] == "PLANT_001"
    assert out[0]["n_images"] == 2


# ── identity provenance pass-through (build_plant_mapping's honest signals) ──


def _mapped_rows(project, *assigned: tuple[str, float]) -> list[dict]:
    """Return per-image count rows carrying the assignment rows of a produced mapping.

    The rows are the ones ``MappingBuild.rows`` produces for a mapping built through the
    platform's own producers (``map_captures``, one capture of each of its two plots), each
    produced assignment restated with one ``(source, distance_m)`` of ``assigned`` and counted
    toward one plant.
    """
    from dataclasses import replace

    from tcip_mcp.pipelines.postprocessing.plant_mapping import load_mapping
    from tests._mapping_fixtures import map_captures
    from tests._web_fixtures import named_project

    date = "2026-02-11"
    named_project(project, "Orchard")
    map_captures(project, project / "ds", [date])
    build = load_mapping(project, "valley")
    assert build is not None and len(build.assignments[date]) == len(assigned)
    restated = replace(
        build,
        assignments={
            date: [
                replace(a, source=source, distance_m=distance)
                for a, (source, distance) in zip(
                    build.assignments[date], assigned
                )
            ]
        },
    )
    return [
        {**row, "plant_id": "P1", "count": 1} for row in restated.rows()[date]
    ]


def test_plant_id_source_passes_through_when_uniform(tmp_path):
    out = _per_plant(
        _mapped_rows(tmp_path, ("sequence", 1.0), ("sequence", 2.0)),
        plant_aggregation="median",
        value_key="count",
    )
    assert out[0]["plant_id_source"] == "sequence"


def test_plant_id_source_reports_mixed_when_not_uniform(tmp_path):
    out = _per_plant(
        _mapped_rows(tmp_path, ("sequence", 1.0), ("nearest_neighbor", 2.0)),
        plant_aggregation="median",
        value_key="count",
    )
    assert out[0]["plant_id_source"] == "mixed"


def test_plant_id_distance_m_max_tracked(tmp_path):
    out = _per_plant(
        _mapped_rows(tmp_path, ("sequence", 1.5), ("sequence", 4.2)),
        plant_aggregation="median",
        value_key="count",
    )
    assert out[0]["plant_id_distance_m_max"] == pytest.approx(4.2)


def test_plant_id_source_absent_when_never_supplied():
    records = [{"image": "a", "plant_id": "P1", "count": 1}]
    out = _per_plant(records, plant_aggregation="median", value_key="count")
    assert "plant_id_source" not in out[0]


# ── statistics ─


@pytest.mark.parametrize(
    "plant_aggregation, value",
    [("median", 6.5), ("mean", 6), ("mode", 9), ("sum", 24)],
)
def test_each_statistic_is_its_plants_value_over_the_records_carrying_one(
    plant_aggregation, value
):
    """Each statistic is taken over the records carrying a value, the rest counted as images.

    A record missing the value key is a missing observation: it counts toward the plant's
    observations and never enters the statistic as a zero.
    """
    records = [
        {"image": f"bush_1_f_{i}", "count": c, "plant_id": "bush_1"}
        for i, c in enumerate((2, 4, 9, 9))
    ] + [{"image": "bush_1_f_4", "plant_id": "bush_1"}]
    (out,) = _per_plant(
        records, plant_aggregation=plant_aggregation, value_key="count"
    )
    assert out["value"] == value
    assert out["n_images"] == 5


# ── CSV export ────────────────────────────────────────────────────────────


def test_the_per_plant_delivery_names_its_phenotype_in_the_crop_vocabulary_sense():
    """The per-plant delivery takes ``delivered_phenotype`` and no ``trait_name`` keyword."""
    import inspect

    params = inspect.signature(deliver_per_plant_aggregate).parameters
    assert "delivered_phenotype" in params
    assert "trait_name" not in params


@pytest.mark.parametrize("door", ["delivery", "tool"])
def test_a_per_plant_delivery_names_its_value_key_with_no_default(door):
    """Both per-plant delivery doors require ``value_key`` with no default.

    The key every record's value sits under is required by both doors' signatures: a call
    omitting it is a ``TypeError`` naming ``value_key``.
    """
    import inspect

    from tcip_mcp.tools.delivery_tools import deliver_per_plant_csv

    signature = inspect.signature(
        deliver_per_plant_aggregate
        if door == "delivery"
        else deliver_per_plant_csv
    )
    assert signature.parameters["value_key"].default is inspect.Parameter.empty
    arguments = {
        name: None
        for name, p in signature.parameters.items()
        if p.default is inspect.Parameter.empty and name != "value_key"
    }
    with pytest.raises(TypeError, match="value_key"):
        signature.bind(**arguments)


def test_a_per_plant_count_delivery_writes_its_population_one_row_each(
    tmp_path,
):
    records = [
        {"plant_id": plant, "count": count, "plant_attribution": "image"}
        for plant, count, n in (("PLANT_001", 7, 3), ("PLANT_002", 4, 2))
        for _ in range(n)
    ]
    out_path = tmp_path / "out" / "aggregated.csv"
    delivered = _deliver(
        tmp_path,
        records,
        out_path,
        "stem_count",
        value_key="count",
        crop="currant",
    )

    with open(out_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert "trait_name" not in (reader.fieldnames or [])
        assert "plant_aggregation" not in (reader.fieldnames or [])
    assert [r["plant_id"] for r in rows] == ["PLANT_001", "PLANT_002"]
    assert rows[0]["crop"] == "currant"
    assert rows[0]["delivered_phenotype"] == "stem_count"
    assert rows[0]["n_images"] == "3"
    assert rows[0]["units"] == ""
    assert rows[0]["value_key"] == "count"
    assert {r["validated"] for r in rows} == {"False"}
    assert delivered["acknowledged_by"] == "user:breeder"


@pytest.mark.parametrize("statistic", ["mean", "median"])
def test_a_delivery_aggregates_under_the_statistic_its_trait_confirms(
    tmp_path, statistic
):
    """A delivery aggregates under the statistic its trait's confirmed revision states.

    The per-plant number is the confirmed revision's ``plant_aggregation`` over the plant's
    per-image values: records whose mean and median differ deliver the one the entry states.
    """
    from tcip_mcp import traits

    trait = fx.DELIVERY_TRAIT_BY_PHENOTYPE["stem_count"]
    fx.propose_and_confirm(
        tmp_path,
        fx.with_fields(
            traits.read_trait(trait, tmp_path).latest_confirmed.entry,
            plant_aggregation=statistic,
        ),
    )
    values = (1, 2, 9)
    out_path = tmp_path / "out.csv"
    _deliver(
        tmp_path,
        [
            {"plant_id": "P1", "count": v, "plant_attribution": "image"}
            for v in values
        ],
        out_path,
        "stem_count",
        value_key="count",
    )
    expected = {"mean": 4, "median": 2}[statistic]
    assert float(csv_rows(out_path)[0]["value"]) == expected


@pytest.mark.parametrize("attributions", [(None,), ("image", "detection")])
def test_a_delivery_refuses_rows_that_omit_or_disagree_on_plant_attribution(
    tmp_path, attributions
):
    records = [
        {
            "plant_id": f"P{i}",
            "count": 7,
            **({} if a is None else {"plant_attribution": a}),
        }
        for i, a in enumerate(attributions)
    ]
    with pytest.raises(
        ValueError, match="disagree on or omit plant_attribution"
    ):
        _deliver(
            tmp_path,
            records,
            tmp_path / "out.csv",
            "stem_count",
            value_key="count",
            buckets=[],
        )


def test_a_dimensional_value_key_labels_its_units_squared_for_an_area(
    tmp_path,
):
    """An area value key labels the delivery's units column with its squared unit.

    A record valued under ``area_mm2`` delivers with mm2 in the units column.
    """
    out_path = tmp_path / "out.csv"
    _deliver(
        tmp_path,
        [
            {
                "plant_id": "PLANT_001",
                "area_mm2": 12.5,
                "plant_attribution": "image",
            }
        ],
        out_path,
        "plant_surface_area",
        value_key="area_mm2",
        delivery_kind=PER_PLANT_REGRESSION_AGGREGATE,
    )
    rows = csv_rows(out_path)
    assert rows[0]["units"] == "mm2"


def test_a_count_delivery_of_a_dimensional_value_refuses_without_a_physical_scale(
    tmp_path,
):
    """A count delivery of a physical-unit value with no scale assessment is refused, unwritten.

    The gate refuses it through the acknowledged delivery, naming ``calibrate_physical_scale``.
    """
    fx.seed_confirmed_aggregate(
        tmp_path,
        "plant_surface_area",
        value_keys=["area_mm2"],
        measured_subject="bud",
    )
    records = [
        {
            "plant_id": "PLANT_001",
            "area_mm2": 12.5,
            "plant_attribution": "image",
        }
    ]
    with pytest.raises(ValueError, match="calibrate_physical_scale"):
        _deliver(
            tmp_path,
            records,
            tmp_path / "out.csv",
            "plant_surface_area",
            value_key="area_mm2",
        )
    assert not (tmp_path / "out.csv").exists()


def test_a_delivery_refuses_a_unit_mismatched_against_crops_yml(tmp_path):
    """A delivery whose value key's unit differs from the unit crops.yml declares is refused."""
    from tcip_mcp.traits import crops_units

    units = crops_units()
    # Find any real crops.yml trait declared in a unit other than mm, to construct a genuine
    # mismatch against an mm-suffixed value_key. Skip if crops.yml is unreadable in this environment
    # rather than asserting on a fixture that can't exist.
    mismatched_trait = next(
        (name for name, u in units.items() if u != "mm"), None
    )
    if mismatched_trait is None:
        pytest.skip(
            "no non-mm trait found in crops.yml to construct a mismatch against"
        )

    with pytest.raises(ValueError, match="declared units|refusing"):
        _deliver(
            tmp_path,
            [
                {
                    "plant_id": "P1",
                    "area_mm2": 1.0,
                    "plant_attribution": "image",
                }
            ],
            tmp_path / "out.csv",
            mismatched_trait,
            value_key="area_mm2",
            buckets=[],
        )


def test_a_delivery_never_labels_a_pixel_value_with_crops_yml_units(tmp_path):
    """A delivery of a px-suffixed value leaves its units column blank.

    A ``principal_axis_extent_px`` value of 124.0 for a trait crops.yml declares in mm delivers
    with a blank units column and the value 124.0.
    """
    from tcip_mcp.traits import crops_units

    assert crops_units()["bark_thickness"] == "mm"
    # A scalar head's delivery, since a count delivery refuses a unit-declared trait whose
    # value_key implies no physical unit; a scalar head's declared unit may legitimately not appear.
    out_path = tmp_path / "out.csv"
    _deliver(
        tmp_path,
        [
            {
                "plant_id": "P1",
                "principal_axis_extent_px": 124.0,
                "plant_attribution": "image",
            }
        ],
        out_path,
        "bark_thickness",
        value_key="principal_axis_extent_px",
        delivery_kind=PER_PLANT_REGRESSION_AGGREGATE,
    )
    rows = csv_rows(out_path)
    assert rows[0]["units"] == ""
    assert rows[0]["value"] == "124.0"


def test_units_never_fall_back_with_an_empty_value_key():
    """An empty value key resolves no unit outside a scalar delivery.

    crops.yml's declared unit is not inherited when nothing names the quantity it would
    describe.
    """
    from tcip_mcp.pipelines.postprocessing.aggregation import _resolve_units
    from tcip_mcp.traits import crops_units

    assert crops_units()["bark_thickness"] == "mm"
    assert _resolve_units("bark_thickness", "", False) == ("", None, "mm")


@pytest.mark.parametrize(
    "value_key", ["plant_id", "detections_total", "open_fraction", "pct_open"]
)
def test_unit_from_value_key_never_fabricates_from_an_unrelated_key(value_key):
    """``unit_from_value_key`` reads no unit off a key whose trailing token is no declared unit.

    ``plant_id``, ``detections_total``, ``open_fraction`` and ``pct_open`` each resolve to no
    unit.
    """
    from tcip_mcp.pipelines.measurement.mask_geometry import (
        unit_from_value_key,
    )

    assert unit_from_value_key(value_key) is None


def test_unit_from_value_key_is_vocabulary_driven_not_a_field_name_whitelist():
    """``unit_from_value_key`` reads a unit off bespoke keys as it does off mask_geometry's own.

    Keys such as ``length_cm``, ``width_m``, ``nut_diameter_mm``, ``arc_length_cm`` and
    ``leaf_area_mm2`` resolve to their units as mask_geometry's ``area_mm2`` and
    ``perimeter_mm`` do, and a ``_px`` key resolves to none.
    """
    from tcip_mcp.pipelines.measurement.mask_geometry import (
        unit_from_value_key,
    )

    assert unit_from_value_key("area_mm2") == ("mm2", "mm")
    assert unit_from_value_key("principal_axis_extent_cm") == ("cm", "cm")
    assert unit_from_value_key("secondary_axis_extent_m") == ("m", "m")
    assert unit_from_value_key("perimeter_mm") == ("mm", "mm")
    assert unit_from_value_key("principal_axis_extent_px") is None
    # bespoke, non-mask_geometry keys: now recognized on the same basis as mask_geometry's own
    assert unit_from_value_key("length_cm") == ("cm", "cm")
    assert unit_from_value_key("width_m") == ("m", "m")
    assert unit_from_value_key("nut_diameter_mm") == ("mm", "mm")
    assert unit_from_value_key("arc_length_cm") == ("cm", "cm")
    assert unit_from_value_key("leaf_area_mm2") == ("mm2", "mm")


def test_unit_from_value_key_refuses_an_area_key_missing_its_squared_suffix():
    """``unit_from_value_key`` refuses an area key whose trailing unit is not squared.

    ``area_mm`` and ``leaf_area_cm`` each raise ``ValueError``.
    """
    from tcip_mcp.pipelines.measurement.mask_geometry import (
        unit_from_value_key,
    )

    with pytest.raises(ValueError, match="area.*squared|squared.*area"):
        unit_from_value_key("area_mm")
    with pytest.raises(ValueError):
        unit_from_value_key(
            "leaf_area_cm"
        )  # 'area' buried mid-key still catches it


def test_resolve_units_squares_area_but_cross_checks_the_linear_declared_unit():
    """``_resolve_units`` squares an area's label but cross-checks its linear unit.

    ``area_mm2`` resolves to the label 'mm2' and the linear basis 'mm' and a linear key stays
    'mm'; for a trait crops.yml declares in mm, when one exists, the declared 'mm' is returned
    beside the squared label.
    """
    from tcip_mcp.pipelines.postprocessing.aggregation import _resolve_units

    # no crops.yml entry -> no cross-check
    assert _resolve_units("__no_such_trait__", "area_mm2", False) == (
        "mm2",
        "mm",
        None,
    )
    # non-area stays unsquared
    assert _resolve_units(
        "__no_such_trait__", "principal_axis_extent_mm", False
    ) == (
        "mm",
        "mm",
        None,
    )

    # A real mm-declared trait: the cross-check compares crops.yml's linear "mm" against area_mm2's
    # own linear basis ("mm", not "mm2") and passes; the returned label is still squared.
    from tcip_mcp.traits import crops_units

    units = crops_units()
    mm_trait = next((name for name, u in units.items() if u == "mm"), None)
    if mm_trait is not None:
        assert _resolve_units(mm_trait, "area_mm2", False) == (
            "mm2",
            "mm",
            "mm",
        )


def test_resolve_units_recognizes_a_bespoke_non_mask_geometry_value_key():
    """``_resolve_units`` resolves the unit of a bespoke value key outside mask_geometry.

    A bespoke value key named '{name}_{unit}' in a crops.yml unit (``nut_diameter_mm``, a
    caliper-style diameter) resolves to that unit.
    """
    from tcip_mcp.pipelines.postprocessing.aggregation import _resolve_units

    assert _resolve_units("__no_such_trait__", "nut_diameter_mm", False) == (
        "mm",
        "mm",
        None,
    )


def test_resolve_units_propagates_the_area_squared_mismatch_refusal():
    """``_resolve_units`` raises the unsquared-area refusal ``unit_from_value_key`` raises."""
    from tcip_mcp.pipelines.postprocessing.aggregation import _resolve_units

    with pytest.raises(ValueError):
        _resolve_units("__no_such_trait__", "area_mm", False)


def test_delivery_skill_documents_the_real_csv_schema(tmp_path):
    """The delivery skill's Per-Plant CSV Schema table lists the header the writer writes.

    The table's columns equal the header of a CSV the per-plant delivery writes.
    """
    from tcip_mcp.knowledge import document_path

    out_path = tmp_path / "schema.csv"
    _deliver(
        tmp_path,
        [{"plant_id": "P1", "count": 1.0, "plant_attribution": "image"}],
        out_path,
        "stem_count",
        value_key="count",
    )
    with open(out_path, newline="") as f:
        written = next(csv.reader(f))

    skill = document_path("delivery")
    lines = skill.read_text(encoding="utf-8").splitlines()
    start = next(
        i
        for i, ln in enumerate(lines)
        if ln.startswith("## Per-Plant CSV Schema")
    )
    documented = []
    for ln in lines[start:]:
        if ln.startswith("## ") and not ln.startswith(
            "## Per-Plant CSV Schema"
        ):
            break
        if (
            ln.startswith("|")
            and not ln.startswith("|---")
            and not ln.startswith("| Column")
        ):
            documented.append(ln.split("|")[1].strip())

    assert documented == written
