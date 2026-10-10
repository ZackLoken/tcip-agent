"""A refusal leaves no audit row, whichever door or library refuses and however deep.

Every mutating door leaves one line per act it made, the decorator's or its library's, and a
refusal made none: an undecorated door refused before it acts, a decorated door returning its
error dict, and a library refusing a write (the registry's ownership rail) alike. The admitting
halves (one row per act when the door does act) live beside each door's own tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tcip_store as ts


def _rows(tmp_path: Path) -> list[dict]:
    from tcip_mcp.audit import audit_log_key

    return list(ts.read_log(audit_log_key(tmp_path)).records)


def _register_model(tmp_path: Path):
    from tcip_mcp.tools.model_tools import register_model

    return register_model(tmp_path, name="m", checkpoint_path=str(tmp_path / "absent.pt"))


def _deliver_per_image_counts(tmp_path: Path):
    from tcip_mcp.tools.inference_tools import deliver_per_image_counts

    return deliver_per_image_counts(tmp_path, str(tmp_path), "preds/2025-09-14",
                                    str(tmp_path / "o.csv"), trait="stem")


def _import_coco(tmp_path: Path):
    from tcip_mcp.tools.ingest_tools import import_coco

    document = tmp_path / "external.json"
    document.write_text('{"images": [], "annotations": [], "categories": []}', encoding="utf-8")
    return import_coco(tmp_path, str(document), str(tmp_path), "2025-09-14")


def _build_plant_mapping(tmp_path: Path):
    from tcip_mcp.tools.phenology_tools import build_plant_mapping

    return build_plant_mapping(tmp_path, name="../no", images_root=str(tmp_path),
                               plant_registry=str(tmp_path / "plants.csv"))


def _deliver_per_plant_csv(tmp_path: Path):
    from tcip_mcp.tools.delivery_tools import deliver_per_plant_csv

    return deliver_per_plant_csv(tmp_path, [], str(tmp_path / "out.csv"), "cyme count",
                                 "per_plant_count_aggregate", ["p1"], str(tmp_path),
                                 ["preds/2025-09-14"])


def _deliver_orthomosaic_plant_counts(tmp_path: Path):
    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    return deliver_orthomosaic_plant_counts(
        tmp_path, str(tmp_path), "preds/2025-09-14", "no-registry", str(tmp_path / "out.csv"),
        "cyme count", ["p1"])


def _deliver_phenology_milestones(tmp_path: Path):
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones

    return deliver_phenology_milestones(tmp_path, "no-such-trait", "mapping", str(tmp_path), [],
                                        str(tmp_path / "o.csv"), ["p1"])


def _assess_an_unregistered_checkpoint(tmp_path: Path):
    from tcip_mcp.tools.calibration_tools import assess_checkpoint

    return assess_checkpoint(tmp_path, str(tmp_path / "m.pt"), "cyme count", "per_image_count",
                             str(tmp_path / "selection"))


def _assess_the_reserved_regions_of_an_unregistered_checkpoint(tmp_path: Path):
    from tcip_mcp.tools.calibration_tools import assess_reserved_regions

    return assess_reserved_regions(tmp_path, str(tmp_path / "m.pt"), "cyme count",
                                   "per_image_count")


def _calibrate_scale_for_no_such_trait(tmp_path: Path):
    from tcip_mcp.tools.calibration_tools import calibrate_physical_scale

    return calibrate_physical_scale(tmp_path, "no-such-trait", str(tmp_path / "selection"),
                                    str(tmp_path / "ref.csv"), "mm", "ruler")


DOORS = [_register_model, _deliver_per_image_counts, _import_coco, _build_plant_mapping,
         _deliver_per_plant_csv, _deliver_orthomosaic_plant_counts,
         _deliver_phenology_milestones, _assess_an_unregistered_checkpoint,
         _assess_the_reserved_regions_of_an_unregistered_checkpoint,
         _calibrate_scale_for_no_such_trait]


@pytest.mark.parametrize("door", DOORS, ids=[d.__name__.lstrip("_") for d in DOORS])
def test_a_door_refused_before_it_acts_leaves_no_row(tmp_path: Path, door):
    try:
        result = door(tmp_path)
    except (ValueError, FileNotFoundError):
        result = {"error": "raised"}
    assert "error" in result, result
    assert _rows(tmp_path) == []


def test_a_save_on_a_missing_image_leaves_no_row(bound) -> None:
    from tcip_mcp.tools.annotation_tools import save_annotations

    before = _rows(bound.root)
    result = save_annotations(bound, str(bound.root / "images" / "absent.png"),
                              annotations=[{"subject": "cyme", "bbox": [1, 1, 5, 5]}])

    assert "error" in result, result
    assert _rows(bound.root) == before
