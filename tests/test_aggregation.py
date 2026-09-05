"""Tests for per-plant aggregation postprocessing.

Covers the plant_id hard requirement (plant identity is never guessed from a
filename; a record must carry an explicit plant_id_key value or one plant_id_fn resolves),
identity-provenance pass-through (plant_id_source/plant_id_distance_m, mirroring
build_plant_mapping's own real Assignment fields), the crops.yml-derived units column, and the
aggregation strategies (count / mean / mode / sum). Phenology milestones are not here; they
are the positive-fraction crossing, tested in test_phenology.py.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tcip_mcp.pipelines.postprocessing.aggregation import (
    aggregate_per_plant,
    export_aggregated_csv,
)
from tests import _operationalization_fixtures as fx


@pytest.fixture(autouse=True)
def _recorded_meaning(tmp_path):
    """Every export below ships under a trait whose delivered number has a confirmed meaning."""
    fx.seed_delivery_traits(tmp_path)
    fx.seed_confirmed_aggregate(tmp_path, "stem_count", value_keys=["count"])
    fx.seed_confirmed_aggregate(tmp_path, "plant_surface_area", value_keys=["area_mm2"])
    fx.seed_confirmed_aggregate(tmp_path, "bark_thickness",
                                value_keys=["principal_axis_extent_px"],
                                measurement_document="regression_operating_point")


def _identity_fn(image_name: str) -> str:
    """A trivial plant_id_fn for tests that don't care about grouping specifics: every image maps
    to a plant_id equal to its own stem-derived group key."""
    return image_name.rsplit("_", 1)[0]


def _validated_bucket(tmp_path, trait: str, *, document: str = "operating_point", tag: str = "a") -> str:
    """A prediction bucket whose sidecar carries a genuine held-out-validated claim for ``trait``,
    for a test whose subject is the CSV's shape or arithmetic rather than the delivery gate itself."""
    from tcip_mcp.pipelines.resolution import VALIDATED_HELD_OUT
    from tests._binding_fixtures import write_bound_sidecar, write_prediction

    root = tmp_path / f"ds_{tag}"
    bucket = root / "predictions" / "preds"
    write_prediction(bucket, "img_a")
    param_key = {"operating_point": "conf", "regression_operating_point": "regression"}[document]
    stamp = {
        "validated": True, "trait": trait,
        "operating_point": {param_key: {"value": 0.4, "requires_validation": True,
                                        "validation_kind": "annotations",
                                        "validated_against": VALIDATED_HELD_OUT}},
    }
    if document == "operating_point":
        stamp["subject"], stamp["attribute"] = trait, None
    write_bound_sidecar(bucket, stamp, document=document, dataset_root=root,
                        experiment_id=f"exp-validated-{tag}")
    return str(bucket)


def _add_validated_scale(bucket: str, trait: str, *, unit: str = "mm", tag: str = "a") -> str:
    """Stamp ``bucket`` (already carrying a validated operating_point.json) with a genuine
    physical-measurement-validated resolve_scale.json, and return its images_dir."""
    from PIL import Image

    from tcip_mcp.pipelines.resolution import VALIDATED_PHYSICAL_MEASUREMENT
    from tcip_mcp.prediction_buckets import bucket_stems
    from tests._binding_fixtures import write_bound_sidecar

    root = Path(bucket).parents[1]
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for stem in bucket_stems(bucket):
        Image.new("RGB", (8, 8), (120, 120, 120)).save(images_dir / f"{stem}.png")
    stamp = {
        "validated": True, "trait": trait,
        "operating_point": {"scale": {
            "value": 0.1, "unit": unit, "capture_id": None,
            "requires_validation": True, "validation_kind": "physical",
            "validated_against": VALIDATED_PHYSICAL_MEASUREMENT,
        }},
    }
    write_bound_sidecar(bucket, stamp, document="resolve_scale", dataset_root=root,
                        images_dir=images_dir, experiment_id=f"exp-scale-{tag}")
    return str(images_dir)


# ── plant identity: no guessing, ever ──────────────────────


def test_no_extract_plant_id_helper_exists_anymore():
    """The filename-guessing fallback is deleted, not just unused; this locks its absence."""
    import tcip_mcp.pipelines.postprocessing.aggregation as agg_module

    assert not hasattr(agg_module, "_extract_plant_id")


def test_missing_plant_id_and_no_fn_raises():
    results = [{"image": "bush_42_flight_3", "count": 1}]
    with pytest.raises(ValueError, match="plant_id_fn|build_plant_mapping"):
        aggregate_per_plant(results, strategy="count", value_key="count")


def test_missing_image_key_and_no_plant_id_raises():
    """The old fallback silently bucketed a keyless record under 'unknown'; now it raises, same as
    any other unresolved-identity record."""
    results = [{"count": 1}]
    with pytest.raises(ValueError, match="plant_id_fn|build_plant_mapping"):
        aggregate_per_plant(results, strategy="count", value_key="count")


def test_plant_id_fn_returning_none_raises_not_groups_under_none():
    """build_plant_mapping's own plant_id_fn can legitimately return None for an unmapped image
    (plot_name=None, source='unmapped'). That must raise cleanly here, not silently group under a
    None key and crash later when aggregate_per_plant sorts groups against a real string key."""
    def sometimes_none(image_name: str) -> str | None:
        return None if "unmapped" in image_name else "PLANT_001"

    results = [
        {"image": "mapped_img", "count": 1},
        {"image": "unmapped_img", "count": 2},
    ]
    with pytest.raises(ValueError, match="plant_id_fn|build_plant_mapping"):
        aggregate_per_plant(results, strategy="count", value_key="count", plant_id_fn=sometimes_none)


def test_empty_string_plant_id_value_raises():
    """A membership test alone (`plant_id_key in r`) would pass for an empty-string value: the
    check must be on the value, not just presence of the key."""
    results = [{"image": "x", "plant_id": "", "count": 1}]
    with pytest.raises(ValueError, match="plant_id_fn|build_plant_mapping"):
        aggregate_per_plant(results, strategy="count", value_key="count")


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


# ── identity provenance pass-through (build_plant_mapping's honest signals) ──


def test_plant_id_source_passes_through_when_uniform():
    results = [
        {"image": "a", "plant_id": "P1", "count": 1, "plant_id_source": "sequence"},
        {"image": "b", "plant_id": "P1", "count": 2, "plant_id_source": "sequence"},
    ]
    out = aggregate_per_plant(results, strategy="count", value_key="count")
    assert out[0]["plant_id_source"] == "sequence"


def test_plant_id_source_reports_mixed_when_not_uniform():
    results = [
        {"image": "a", "plant_id": "P1", "count": 1, "plant_id_source": "sequence"},
        {"image": "b", "plant_id": "P1", "count": 2, "plant_id_source": "nearest_neighbour"},
    ]
    out = aggregate_per_plant(results, strategy="count", value_key="count")
    assert out[0]["plant_id_source"] == "mixed"


def test_plant_id_distance_m_max_tracked():
    results = [
        {"image": "a", "plant_id": "P1", "count": 1, "plant_id_distance_m": 1.5},
        {"image": "b", "plant_id": "P1", "count": 2, "plant_id_distance_m": 4.2},
    ]
    out = aggregate_per_plant(results, strategy="count", value_key="count")
    assert out[0]["plant_id_distance_m_max"] == pytest.approx(4.2)


def test_plant_id_source_absent_when_never_supplied():
    results = [{"image": "a", "plant_id": "P1", "count": 1}]
    out = aggregate_per_plant(results, strategy="count", value_key="count")
    assert "plant_id_source" not in out[0]


# ── strategies (all now supply plant_id_fn, no bare-filename grouping left) ─


def test_count_strategy_median():
    results = [
        {"image": "bush_1_f_1", "count": 2, "plant_id": "bush_1"},
        {"image": "bush_1_f_2", "count": 4, "plant_id": "bush_1"},
        {"image": "bush_1_f_3", "count": 9, "plant_id": "bush_1"},
    ]
    out = aggregate_per_plant(results, strategy="count", value_key="count")
    assert out[0]["value"] == 4
    assert out[0]["min_count"] == 2
    assert out[0]["max_count"] == 9


def test_mean_strategy():
    results = [
        {"image": "bush_1_f_1", "value": 1.0, "plant_id": "bush_1"},
        {"image": "bush_1_f_2", "value": 3.0, "plant_id": "bush_1"},
    ]
    out = aggregate_per_plant(results, strategy="mean", value_key="value")
    assert out[0]["value"] == 2.0


def test_mode_strategy():
    results = [
        {"image": "bush_1_f_1", "grade": 2, "plant_id": "bush_1"},
        {"image": "bush_1_f_2", "grade": 2, "plant_id": "bush_1"},
        {"image": "bush_1_f_3", "grade": 5, "plant_id": "bush_1"},
    ]
    out = aggregate_per_plant(results, strategy="mode", value_key="grade")
    assert out[0]["value"] == 2
    assert out[0]["agreement"] == pytest.approx(2 / 3, abs=1e-4)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown aggregation strategy"):
        aggregate_per_plant([{"image": "a_b_c", "count": 1, "plant_id": "a"}], strategy="nope")


# ── CSV export ────────────────────────────────────────────────────────────


def test_export_aggregated_csv_signature_carries_delivered_phenotype_not_trait_name():
    """The vocabulary-sense parameter is delivered_phenotype; trait_name (the registry sense the
    rest of the trait-facing surface keeps) is no longer a valid keyword here."""
    import inspect

    params = inspect.signature(export_aggregated_csv).parameters
    assert "delivered_phenotype" in params
    assert "trait_name" not in params


def test_export_aggregated_csv_signature_carries_operating_point_validated_not_measurement_validated():
    """The unified dimension key names its own parameter and column: measurement_validated is a
    retired spelling of the same fact operating_point_validated already carries."""
    import inspect

    params = inspect.signature(export_aggregated_csv).parameters
    assert "operating_point_validated" in params
    assert "measurement_validated" not in params


def test_export_aggregated_csv(tmp_path):
    results = [
        {"plant_id": "PLANT_001", "value": 7, "observations": 3, "value_key": "count",
         "plant_attribution": "image", "measurement_document": "operating_point"},
        {"plant_id": "PLANT_002", "value": 4, "observations": 2, "value_key": "count",
         "plant_attribution": "image", "measurement_document": "operating_point"},
    ]
    out_path = tmp_path / "out" / "aggregated.csv"
    bucket = _validated_bucket(tmp_path, "stem", tag="export")
    export_aggregated_csv(
        results, str(out_path), delivered_phenotype="stem_count", crop="currant",
        pred_dirs=[bucket],
    )

    with open(out_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert "trait_name" not in (reader.fieldnames or [])

    assert [r["plant_id"] for r in rows] == ["PLANT_001", "PLANT_002"]
    assert rows[0]["crop"] == "currant"
    assert rows[0]["delivered_phenotype"] == "stem_count"
    assert rows[0]["n_images"] == "3"


def test_export_aggregated_csv_refuses_a_record_set_with_no_plant_attribution(tmp_path):
    results = [{"plant_id": "PLANT_001", "value": 7, "observations": 3, "value_key": "count",
               "measurement_document": "operating_point"}]
    out_path = tmp_path / "out.csv"
    with pytest.raises(ValueError, match="disagree on or omit plant_attribution"):
        export_aggregated_csv(results, str(out_path), delivered_phenotype="stem_count")


def test_export_aggregated_csv_refuses_when_records_disagree_on_plant_attribution(tmp_path):
    results = [
        {"plant_id": "PLANT_001", "value": 7, "observations": 3, "value_key": "count",
         "plant_attribution": "image", "measurement_document": "operating_point"},
        {"plant_id": "PLANT_002", "value": 4, "observations": 2, "value_key": "count",
         "plant_attribution": "detection", "measurement_document": "operating_point"},
    ]
    out_path = tmp_path / "out.csv"
    with pytest.raises(ValueError, match="disagree on or omit plant_attribution"):
        export_aggregated_csv(results, str(out_path), delivered_phenotype="stem_count")


def test_export_aggregated_csv_header_carries_operating_point_validated_not_measurement_validated(
    tmp_path,
):
    """The delivered per-plant CSV's validity column is operating_point_validated; the retired
    spelling measurement_validated must not reappear in its header."""
    results = [{"plant_id": "PLANT_001", "value": 7, "observations": 3, "value_key": "count",
               "plant_attribution": "image", "measurement_document": "operating_point"}]
    out_path = tmp_path / "aggregated.csv"
    bucket = _validated_bucket(tmp_path, "stem", tag="header")
    export_aggregated_csv(results, str(out_path), delivered_phenotype="stem_count",
                          pred_dirs=[bucket])

    with open(out_path, newline="") as f:
        fieldnames = csv.DictReader(f).fieldnames or []
    assert "operating_point_validated" in fieldnames
    assert "measurement_validated" not in fieldnames


def test_export_aggregated_csv_units_derived_from_value_key(tmp_path):
    """A dimensional value_key (mask_geometry-style, area_mm2) must label the units column mm2,
    derived from the key that produced the number, never a caller-asserted string. Squared, not the
    bare linear unit: an area labeled "mm" understates its own dimensionality."""
    results = [
        {"plant_id": "PLANT_001", "value": 12.5, "observations": 1, "value_key": "area_mm2",
         "plant_attribution": "image", "measurement_document": "operating_point",
         "scale_document": "resolve_scale"},
    ]
    out_path = tmp_path / "out.csv"
    bucket = _validated_bucket(tmp_path, "plant_surface_area", tag="units")
    images_dir = _add_validated_scale(bucket, "plant_surface_area", tag="units")
    export_aggregated_csv(
        results, str(out_path), delivered_phenotype="plant_surface_area",
        pred_dirs=[bucket], images_dir=images_dir,
    )
    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["units"] == "mm2"


def test_export_aggregated_csv_count_trait_has_blank_units(tmp_path):
    results = [{"plant_id": "PLANT_001", "value": 4, "observations": 3, "value_key": "count",
                "plant_attribution": "image", "measurement_document": "operating_point"}]
    out_path = tmp_path / "out.csv"
    bucket = _validated_bucket(tmp_path, "stem", tag="blank-units")
    export_aggregated_csv(results, str(out_path), delivered_phenotype="stem_count",
                          pred_dirs=[bucket])
    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["units"] == ""


def test_export_aggregated_csv_refuses_unit_mismatch_against_crops_yml(tmp_path):
    """A trait crops.yml declares in one unit must not ship under a different unit implied by the
    aggregated values' own key: that's the exact failure mode this column exists to prevent."""
    from tcip_mcp.traits import crops_units

    units = crops_units()
    # Find any real crops.yml trait declared in a unit other than mm, to construct a genuine
    # mismatch against an mm-suffixed value_key. Skip if crops.yml is unreadable in this environment
    # rather than asserting on a fixture that can't exist.
    mismatched_trait = next((name for name, u in units.items() if u != "mm"), None)
    if mismatched_trait is None:
        pytest.skip("no non-mm trait found in crops.yml to construct a mismatch against")

    results = [{"plant_id": "P1", "value": 1.0, "observations": 1, "value_key": "area_mm2",
                "plant_attribution": "image", "measurement_document": "operating_point"}]
    out_path = tmp_path / "out.csv"
    with pytest.raises(ValueError, match="declared units|refusing"):
        export_aggregated_csv(results, str(out_path), delivered_phenotype=mismatched_trait)


def test_export_aggregated_csv_never_labels_a_pixel_value_with_crops_yml_units(tmp_path):
    """A px-suffixed value_key must not inherit crops.yml's declared physical unit as a fallback:
    that shipped a 124-pixel measurement labeled 'mm' under a real mm-declared trait. The units
    column must be blank, not the declared unit, whenever the value's own key implies no physical
    unit at all."""
    from tcip_mcp.traits import crops_units

    assert crops_units()["bark_thickness"] == "mm"
    # A scalar-head document, since operating_point refuses a unit-declared trait whose value_key
    # implies no physical unit; a scalar head's declared unit may legitimately not appear in value.
    results = [{"plant_id": "P1", "value": 124.0, "observations": 1,
                "value_key": "principal_axis_extent_px",
                "plant_attribution": "image", "measurement_document": "regression_operating_point"}]
    out_path = tmp_path / "out.csv"
    bucket = _validated_bucket(tmp_path, "bark_thickness", document="regression_operating_point",
                              tag="pixel")
    export_aggregated_csv(results, str(out_path), delivered_phenotype="bark_thickness",
                          pred_dirs=[bucket])
    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["units"] == ""
    assert rows[0]["value"] == "124.0"


def test_units_never_fall_back_with_no_value_key_at_all():
    """A results list not produced by aggregate_per_plant (no value_key present) must not inherit
    crops.yml's declared unit either; there is nothing to cross-check it against.

    Asserted on the resolution rather than on a delivered file, because a row that names no
    quantity is refused at the door now: there is nothing for a confirmed operationalization to
    check it against.
    """
    from tcip_mcp.pipelines.postprocessing.aggregation import _resolve_units
    from tcip_mcp.traits import crops_units

    assert crops_units()["bark_thickness"] == "mm"
    results = [{"plant_id": "P1", "value": 1.0, "observations": 1}]
    assert _resolve_units("bark_thickness", results, "operating_point") == ("", None)


@pytest.mark.parametrize("value_key", ["plant_id", "detections_total", "open_fraction", "pct_open"])
def test_unit_from_value_key_never_fabricates_from_an_unrelated_key(value_key):
    """The unit-suffix regex used to match any trailing underscore-word, so 'detections_total' read
    as unit='total' and 'plant_id' as unit='id'. Only a trailing token that is one of crops.yml's own
    declared units (or a mechanically-squared form of one) may imply a unit: 'id'/'total'/'fraction'/
    'open' are none of those, regardless of what precedes them."""
    from tcip_mcp.pipelines.postprocessing.aggregation import _unit_from_value_key

    assert _unit_from_value_key(value_key) is None


def test_unit_from_value_key_is_vocabulary_driven_not_a_field_name_whitelist():
    """The unit vocabulary is driven by crops.yml's own declared units, not a hardcoded whitelist of
    mask_geometry field names, so a bespoke agent-composed measurement (arc length, a landmark
    distance, anything outside mask_geometry) is still recognized when its value_key is a sensible
    '{name}_{unit}'. Any key ending in one of crops.yml's declared units (or its squared form) is
    recognized, regardless of what module produced it."""
    from tcip_mcp.pipelines.postprocessing.aggregation import _unit_from_value_key

    assert _unit_from_value_key("area_mm2") == ("mm2", "mm")
    assert _unit_from_value_key("principal_axis_extent_cm") == ("cm", "cm")
    assert _unit_from_value_key("secondary_axis_extent_m") == ("m", "m")
    assert _unit_from_value_key("perimeter_mm") == ("mm", "mm")
    assert _unit_from_value_key("principal_axis_extent_px") is None
    # bespoke, non-mask_geometry keys: now recognized on the same basis as mask_geometry's own
    assert _unit_from_value_key("length_cm") == ("cm", "cm")
    assert _unit_from_value_key("width_m") == ("m", "m")
    assert _unit_from_value_key("nut_diameter_mm") == ("mm", "mm")
    assert _unit_from_value_key("arc_length_cm") == ("cm", "cm")
    assert _unit_from_value_key("leaf_area_mm2") == ("mm2", "mm")


def test_unit_from_value_key_refuses_an_area_key_missing_its_squared_suffix():
    """A value_key that names 'area' but whose trailing unit isn't squared (area_mm instead of
    area_mm2) is a real dimensional-mismatch bug in the producing code, not a case to guess through.
    An area is length^2, and silently labeling it with a bare linear unit is exactly the kind of wrong
    number this function exists to prevent from shipping quietly."""
    from tcip_mcp.pipelines.postprocessing.aggregation import _unit_from_value_key

    with pytest.raises(ValueError, match="area.*squared|squared.*area"):
        _unit_from_value_key("area_mm")
    with pytest.raises(ValueError):
        _unit_from_value_key("leaf_area_cm")  # 'area' buried mid-key still catches it


def test_resolve_units_squares_area_but_cross_checks_the_linear_declared_unit():
    """area_mm2 must resolve to units='mm2' in the CSV, but crops.yml's cross-check declares only
    the linear unit ('mm'), so the cross-check itself must still compare against that linear unit.
    Comparing 'mm2' to 'mm' directly would never match."""
    from tcip_mcp.pipelines.postprocessing.aggregation import _resolve_units

    results = [{"plant_id": "P1", "value": 800.0, "observations": 1, "value_key": "area_mm2"}]
    # no crops.yml entry -> no cross-check
    assert _resolve_units("__no_such_trait__", results, "operating_point") == ("mm2", "mm")

    results_linear = [{"plant_id": "P1", "value": 12.0, "observations": 1,
                       "value_key": "principal_axis_extent_mm"}]
    # non-area stays unsquared
    assert _resolve_units("__no_such_trait__", results_linear, "operating_point") == ("mm", "mm")

    # A real mm-declared trait: the cross-check compares crops.yml's linear "mm" against area_mm2's
    # own linear basis ("mm", not "mm2") and passes; the returned label is still squared.
    from tcip_mcp.traits import crops_units

    units = crops_units()
    mm_trait = next((name for name, u in units.items() if u == "mm"), None)
    if mm_trait is not None:
        assert _resolve_units(mm_trait, results, "operating_point") == ("mm2", "mm")


def test_resolve_units_recognizes_a_bespoke_non_mask_geometry_value_key():
    """The generalized vocabulary win end to end: a trait the agent measured with its own bespoke
    code (a caliper-style diameter, never mask_geometry's field names) still gets a real units column
    instead of shipping blank, as long as it names itself '{name}_{unit}' in a crops.yml-real unit."""
    from tcip_mcp.pipelines.postprocessing.aggregation import _resolve_units

    results = [{"plant_id": "P1", "value": 14.2, "observations": 1, "value_key": "nut_diameter_mm"}]
    assert _resolve_units("__no_such_trait__", results, "operating_point") == ("mm", "mm")


def test_resolve_units_propagates_the_area_squared_mismatch_refusal():
    """The refusal in unit_from_value_key must reach export_aggregated_csv's own caller, not be
    swallowed: a delivery CSV must never ship silently mislabeled because of a naming bug upstream."""
    from tcip_mcp.pipelines.postprocessing.aggregation import _resolve_units

    results = [{"plant_id": "P1", "value": 800.0, "observations": 1, "value_key": "area_mm"}]
    with pytest.raises(ValueError):
        _resolve_units("__no_such_trait__", results, "operating_point")


def test_export_aggregated_csv_writes_value_key_column(tmp_path):
    """value_key is a real CSV column, not just an internal field: it's the one thing that lets a
    reader independently detect a px/mm mismatch themselves."""
    results = [{"plant_id": "P1", "value": 1.0, "observations": 1, "value_key": "area_mm2",
                "plant_attribution": "image", "measurement_document": "operating_point",
                "scale_document": "resolve_scale"}]
    out_path = tmp_path / "out.csv"
    bucket = _validated_bucket(tmp_path, "plant_surface_area", tag="value-key")
    images_dir = _add_validated_scale(bucket, "plant_surface_area", tag="value-key")
    export_aggregated_csv(results, str(out_path), delivered_phenotype="plant_surface_area",
                          pred_dirs=[bucket], images_dir=images_dir)
    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["value_key"] == "area_mm2"


def test_delivery_skill_documents_the_real_csv_schema(tmp_path):
    """The delivery skill's Per-Plant CSV Schema table must be the schema the writer actually writes.

    The table is what the agent reads before building a deliverable; a stale one (it listed seven of
    the columns for a while) teaches a schema the breeder's file does not have. Compared against the
    real written header, not against a second copy of the list.
    """
    from tcip_mcp.knowledge import document_path

    out_path = tmp_path / "schema.csv"
    bucket = _validated_bucket(tmp_path, "stem", tag="schema")
    export_aggregated_csv(
        [{"plant_id": "P1", "value": 1.0, "observations": 1, "value_key": "count",
          "plant_attribution": "image", "measurement_document": "operating_point"}],
        str(out_path), delivered_phenotype="stem_count", pred_dirs=[bucket])
    with open(out_path, newline="") as f:
        written = next(csv.reader(f))

    skill = document_path("delivery")
    lines = skill.read_text(encoding="utf-8").splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("## Per-Plant CSV Schema"))
    documented = []
    for ln in lines[start:]:
        if ln.startswith("## ") and not ln.startswith("## Per-Plant CSV Schema"):
            break
        if ln.startswith("|") and not ln.startswith("|---") and not ln.startswith("| Column"):
            documented.append(ln.split("|")[1].strip())

    assert documented == written
