"""``scripts/conform_delivery_events.py``: the one-off check for a project's stored
``delivery_events`` records against the current ``DeliveryEventRecord`` shape. Unlike
``conform_view_coverage_viewing.py`` this script rewrites almost nothing: none of the three
``plant_mapping`` disclosure keys a refused record lacks has a value derivable from the rest of
that record. The one exception is ``acknowledged_by``/``acknowledgement_reason``, whose true value
for a record predating this pair is ``null``, derivable from the record's own age alone; a record
missing only that pair is write-forwarded rather than merely named.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import tcip_store as ts
from tcip_store.binding import bind_default

from tcip_mcp.pipelines import resolution
from tcip_mcp.pipelines.resolution import record_delivery_binding_event

SCRIPT = Path(__file__).parent.parent / "scripts" / "conform_delivery_events.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("conform_delivery_events_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _events(root: Path) -> dict[str, dict]:
    scope = resolution.delivery_events_scope(root)
    keys = ts.keys(resolution.DELIVERY_EVENTS_STORE, str(scope))
    return {key.parts[0]: ts.read(key) for key in keys}


def _write_valid_event(root: Path) -> None:
    record_delivery_binding_event(
        "test_door", None, [], {}, measurement_documents=["operating_point"],
        scale_document=None, acknowledgement=None, trait="astringency",
        delivery_kind="state_crossing_dates", project_root=root, plant_mapping=None,
    )


def _write_old_shaped_event(root: Path, event_id: str = "old-shaped") -> None:
    """A ``plant_mapping`` written before ``dates_delivered``, ``images_unattributed`` and
    ``plant_attribution`` existed: exactly the gap this script names rather than fills."""
    key = resolution.delivery_event_key(resolution.delivery_events_scope(root), event_id)
    ts.replace(
        key,
        {
            "event_id": event_id,
            "trait": "astringency",
            "delivery_kind": "state_crossing_dates",
            "door": "compute_phenology",  # a stored value written before the rename
            "output_path": None,
            "measurement_documents": ["operating_point"],
            "scale_document": None,
            "plant_mapping": {
                "name": "valley",
                "project_root": str(root),
                "dataset_id": "ds-1",
                "dataset_root": "C:/data",
                "built_at": datetime.now(timezone.utc).isoformat(),
                "record_sha256": "0" * 64,
                "nn_tolerance_m": {"value": 3, "source": "stated"},
                "capture_identity": {},
                "captures_unverified": [],
                "plant_csvs_unverified": [],
                "images_unattributed_scope": "delivered_dates",
            },
            "documents": {},
            "produced_at": datetime.now(timezone.utc).isoformat(),
        },
        expect=ts.Version.ABSENT,
    )


def _write_event_missing_output_sha256(root: Path, event_id: str = "no-digest") -> None:
    """A ``delivery_events`` record written before ``output_sha256`` existed: every other key
    present, that one alone missing, so the refusal names exactly this gap."""
    key = resolution.delivery_event_key(resolution.delivery_events_scope(root), event_id)
    ts.replace(
        key,
        {
            "event_id": event_id, "trait": "astringency", "delivery_kind": "state_crossing_dates",
            "door": "test_door", "output_path": None, "measurement_documents": ["operating_point"],
            "scale_document": None, "plant_mapping": None, "documents": {},
            "produced_at": datetime.now(timezone.utc).isoformat(),
        },
        expect=ts.Version.ABSENT,
    )


def _write_event_missing_acknowledgement(root: Path, event_id: str = "no-ack") -> None:
    """A fully-shaped ``delivery_events`` record written before ``acknowledged_by``/
    ``acknowledgement_reason`` existed: every other key present, those two alone missing, so
    forwarding them to ``null`` is enough to validate."""
    key = resolution.delivery_event_key(resolution.delivery_events_scope(root), event_id)
    ts.replace(
        key,
        {
            "event_id": event_id, "trait": "astringency", "delivery_kind": "state_crossing_dates",
            "door": "test_door", "output_path": None, "output_sha256": None,
            "measurement_documents": ["operating_point"], "scale_document": None,
            "plant_mapping": None, "documents": {},
            "produced_at": datetime.now(timezone.utc).isoformat(),
        },
        expect=ts.Version.ABSENT,
    )


def test_check_root_write_forwards_a_record_missing_only_acknowledgement_keys(tmp_path: Path):
    bind_default()
    module = _load_script()
    _write_event_missing_acknowledgement(tmp_path)

    outcomes, refused = module.check_root(tmp_path)

    assert refused is False
    assert "write-forwarded acknowledged_by/acknowledgement_reason to null, validates" in outcomes[0]
    stored = _events(tmp_path)["no-ack"]
    assert stored["acknowledged_by"] is None
    assert stored["acknowledgement_reason"] is None
    # The write-forward is the only change: every other key survives untouched.
    assert stored["event_id"] == "no-ack"
    assert stored["door"] == "test_door"


def test_check_root_plan_mode_names_the_forwardable_gap_and_writes_nothing(tmp_path: Path):
    bind_default()
    module = _load_script()
    _write_event_missing_acknowledgement(tmp_path)
    before = _events(tmp_path)

    outcomes, refused = module.check_root(tmp_path, plan=True)

    assert refused is True
    assert "would write-forward" in outcomes[0]
    assert _events(tmp_path) == before


def test_check_root_names_the_gap_when_output_sha256_predates_this_record(tmp_path: Path):
    bind_default()
    module = _load_script()
    _write_event_missing_output_sha256(tmp_path)

    outcomes, refused = module.check_root(tmp_path)

    assert refused is True
    assert "output_sha256" in outcomes[0]
    assert "re-deliver, or remove the record by hand" in outcomes[0]


def test_a_root_with_nothing_stored_reports_no_outcomes_and_is_not_refused(tmp_path: Path):
    bind_default()
    module = _load_script()
    (tmp_path / ".tcip").mkdir()

    outcomes, refused = module.check_root(tmp_path)

    assert outcomes == []
    assert refused is False


def test_a_root_holding_no_tcip_directory_is_refused_by_name(tmp_path: Path):
    bind_default()
    module = _load_script()

    outcomes, refused = module.check_root(tmp_path)

    assert refused is True
    assert len(outcomes) == 1
    assert "no .tcip directory" in outcomes[0]


def test_a_nonexistent_root_is_refused_the_same_way_as_one_missing_tcip(tmp_path: Path):
    bind_default()
    module = _load_script()
    missing = tmp_path / "does-not-exist"

    outcomes, refused = module.check_root(missing)

    assert refused is True
    assert "no .tcip directory" in outcomes[0]


def _write_event_with_neither_plant_mapping_shape(root: Path, event_id: str = "neither-shape") -> None:
    """A ``plant_mapping`` naming neither ``PlantMappingDisclosure``'s nor
    ``PlantRegistryDisclosure``'s own key set: the union resolves to nothing, so this refuses."""
    key = resolution.delivery_event_key(resolution.delivery_events_scope(root), event_id)
    ts.replace(
        key,
        {
            "event_id": event_id, "trait": "astringency", "delivery_kind": "state_crossing_dates",
            "door": "test_door", "output_path": None, "output_sha256": None,
            "measurement_documents": ["operating_point"], "scale_document": None,
            "acknowledged_by": None, "acknowledgement_reason": None,
            "plant_mapping": {"garbage": True}, "documents": {},
            "produced_at": datetime.now(timezone.utc).isoformat(),
        },
        expect=ts.Version.ABSENT,
    )


def test_check_root_names_a_plant_mapping_of_neither_disclosure_shape(tmp_path: Path):
    bind_default()
    module = _load_script()
    _write_event_with_neither_plant_mapping_shape(tmp_path)

    outcomes, refused = module.check_root(tmp_path)

    assert refused is True
    assert len(outcomes) == 1
    assert "neither-shape: refused" in outcomes[0]
    assert "plant_mapping" in outcomes[0]


def test_check_root_names_a_valid_record_and_refuses_an_old_shaped_one(tmp_path: Path):
    bind_default()
    module = _load_script()
    _write_valid_event(tmp_path)
    _write_old_shaped_event(tmp_path)

    outcomes, refused = module.check_root(tmp_path)

    assert refused is True
    assert len(outcomes) == 2
    assert any(o.endswith(": validates, unchanged") for o in outcomes)
    refusal = next(o for o in outcomes if o.startswith("old-shaped: refused"))
    assert "dates_delivered" in refusal
    assert "re-deliver, or remove the record by hand" in refusal


def test_main_over_an_empty_root_exits_zero_and_reports_nothing_stored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    bind_default()
    module = _load_script()
    (tmp_path / ".tcip").mkdir()

    monkeypatch.setattr(sys, "argv", ["conform_delivery_events.py", str(tmp_path)])
    exit_code = module.main()
    output = capsys.readouterr().out

    assert exit_code == 0
    assert f"{tmp_path.resolve()}: nothing stored" in output


def test_main_over_a_root_with_no_tcip_directory_exits_two_and_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    bind_default()
    module = _load_script()
    missing = tmp_path / "does-not-exist"

    monkeypatch.setattr(sys, "argv", ["conform_delivery_events.py", str(missing)])
    exit_code = module.main()
    output = capsys.readouterr().out

    assert exit_code == 2
    assert f"{missing.resolve()}: refused, no .tcip directory found" in output


def _geo_transform_dict() -> dict:
    return {
        "tiepoint_pixel_x": 0.0, "tiepoint_pixel_y": 0.0,
        "tiepoint_native_x": 500_000.0, "tiepoint_native_y": 4_800_000.0,
        "pixel_scale_x": 0.5, "pixel_scale_y": 0.5, "epsg": 32615,
    }


def _register_two_plants(tmp_path: Path, *, name: str = "reg") -> tuple[dict, tuple[float, float]]:
    """A real registered plant registry, one plant inside a 64x64 raster frame and one outside
    it, projected through :func:`_geo_transform_dict`'s own affine; returns the registered record
    and the (lat, lon) an in-frame detection's own projected position would resolve to."""
    from tcip_mcp.pipelines.postprocessing.orthomosaic_mapping import (
        GeoTransform, OrthomosaicGeoreference,
    )
    from tcip_mcp.pipelines.postprocessing.plant_mapping import register_plant_registry_record

    georef = OrthomosaicGeoreference(GeoTransform(**_geo_transform_dict()))
    in_lat, in_lon = georef.pixel_to_wgs84(10.0, 10.0)
    out_lat, out_lon = georef.pixel_to_wgs84(200.0, 10.0)  # column 200 >= width 64: outside

    plant_csv = tmp_path / "plants.csv"
    plant_csv.write_text(
        "plot_name,accession_name,WGS84_centroid_x,WGS84_centroid_y\n"
        f"plot_in,acc_in,{in_lon},{in_lat}\n"
        f"plot_out,acc_out,{out_lon},{out_lat}\n",
        encoding="utf-8",
    )
    registry = register_plant_registry_record(
        tmp_path, name, [plant_csv], crop="chestnut", site="orchard", registered_by="test")
    return registry, (in_lat, in_lon)


def _write_registry_event_missing_plants_outside_raster(
    tmp_path: Path, *, event_id: str, registry_digest: str,
) -> None:
    key = resolution.delivery_event_key(resolution.delivery_events_scope(tmp_path), event_id)
    ts.replace(
        key,
        {
            "event_id": event_id, "trait": "astringency", "delivery_kind": "state_crossing_dates",
            "door": "deliver_orthomosaic_plant_counts", "output_path": None, "output_sha256": None,
            "measurement_documents": ["operating_point"], "scale_document": None,
            "acknowledged_by": None, "acknowledgement_reason": None,
            "plant_mapping": {
                "plant_registry": {"name": "reg", "digest": registry_digest},
                "project_root": str(tmp_path),
                "raster_identity": {"width": 64, "height": 64, "geotransform": _geo_transform_dict()},
                "nn_tolerance_m": {"value": 1.0, "source": "stated"},
                "detections_unattributed": 0,
                "detections_unattributed_scope": "delivered_raster",
                "plant_attribution": "detection",
            },
            "documents": {},
            "produced_at": datetime.now(timezone.utc).isoformat(),
        },
        expect=ts.Version.ABSENT,
    )


def test_check_root_write_forwards_plants_outside_raster_for_a_registry_event_lacking_it(
    tmp_path: Path,
) -> None:
    bind_default()
    module = _load_script()
    registry, _in_position = _register_two_plants(tmp_path)
    _write_registry_event_missing_plants_outside_raster(
        tmp_path, event_id="registry-no-outside", registry_digest=registry["digest"])

    outcomes, refused = module.check_root(tmp_path)

    assert refused is False
    assert "write-forwarded plants_outside_raster, validates" in outcomes[0]
    stored = _events(tmp_path)["registry-no-outside"]
    assert stored["plant_mapping"]["plants_outside_raster"] == ["plot_out"]


def test_check_root_refuses_a_registry_event_whose_registry_moved(tmp_path: Path) -> None:
    bind_default()
    module = _load_script()
    _register_two_plants(tmp_path)
    _write_registry_event_missing_plants_outside_raster(
        tmp_path, event_id="registry-moved", registry_digest="0" * 64)

    outcomes, refused = module.check_root(tmp_path)

    assert refused is True
    assert "has moved" in outcomes[0]
    assert "plants_outside_raster" in outcomes[0]


def test_check_root_refuses_a_registry_event_whose_registry_no_longer_loads(tmp_path: Path) -> None:
    bind_default()
    module = _load_script()
    _write_registry_event_missing_plants_outside_raster(
        tmp_path, event_id="registry-gone", registry_digest="0" * 64)

    outcomes, refused = module.check_root(tmp_path)

    assert refused is True
    assert "no longer loads" in outcomes[0]


def test_check_root_refuses_a_registry_event_whose_csv_was_rewritten_on_disk(tmp_path: Path) -> None:
    """A registered CSV's own registration-time digest does not move when the file is rewritten
    afterward: only reading and hashing the file's current bytes catches this."""
    bind_default()
    module = _load_script()
    registry, _in_position = _register_two_plants(tmp_path)
    _write_registry_event_missing_plants_outside_raster(
        tmp_path, event_id="registry-csv-rewritten", registry_digest=registry["digest"])
    csv_path = Path(registry["csvs"][0]["path"])
    csv_path.write_text(
        "plot_name,accession_name,WGS84_centroid_x,WGS84_centroid_y\n"
        "plot_new,acc_new,0.0,0.0\n",
        encoding="utf-8",
    )

    outcomes, refused = module.check_root(tmp_path)

    assert refused is True
    assert len(outcomes) == 1
    assert "registry-csv-rewritten" in outcomes[0]
    assert str(csv_path) in outcomes[0]
    assert "rewritten" in outcomes[0]
    stored = _events(tmp_path)["registry-csv-rewritten"]
    assert "plants_outside_raster" not in stored["plant_mapping"]


def test_check_root_refuses_a_registry_event_whose_csv_file_is_missing(tmp_path: Path) -> None:
    bind_default()
    module = _load_script()
    registry, _in_position = _register_two_plants(tmp_path)
    _write_registry_event_missing_plants_outside_raster(
        tmp_path, event_id="registry-csv-missing", registry_digest=registry["digest"])
    csv_path = Path(registry["csvs"][0]["path"])
    csv_path.unlink()

    outcomes, refused = module.check_root(tmp_path)

    assert refused is True
    assert len(outcomes) == 1
    assert "registry-csv-missing" in outcomes[0]
    assert csv_path.name in outcomes[0]
    stored = _events(tmp_path)["registry-csv-missing"]
    assert "plants_outside_raster" not in stored["plant_mapping"]


def test_main_plan_mode_over_a_mixed_root_prints_both_lines_exits_two_and_rewrites_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    bind_default()
    module = _load_script()
    _write_valid_event(tmp_path)
    _write_old_shaped_event(tmp_path)
    before = _events(tmp_path)

    monkeypatch.setattr(sys, "argv", ["conform_delivery_events.py", "--plan", str(tmp_path)])
    exit_code = module.main()
    output = capsys.readouterr().out

    assert exit_code == 2
    assert "validates, unchanged" in output
    assert "old-shaped: refused" in output
    assert _events(tmp_path) == before
