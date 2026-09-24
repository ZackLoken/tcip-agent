"""A refusal leaves no audit row, whichever door or library refuses and however deep.

Every mutating door leaves one line per act it made, the decorator's or its library's, and a
refusal made none: an undecorated door refused before it acts, a decorated door returning its
error dict, and a library refusing a write (the registry's ownership rail, the experiment
record's terminal lock) alike. A body that raises keeps its exception line. The admitting halves
(one row per act when the door does act) live beside each door's own tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tcip_store as ts


def _rows(tmp_path: Path) -> list[dict]:
    from tcip_mcp.audit import audit_log_key

    return [row for key in dict.fromkeys((audit_log_key(), audit_log_key(tmp_path)))
            for row in ts.read_log(key).records]


def _register_model(tmp_path: Path):
    from tcip_mcp.tools.model_tools import register_model

    return register_model(name="m", checkpoint_path=str(tmp_path / "absent.pt"), config={"a": 1},
                          experiment_id="exp-1")


def _deliver_per_image_counts(tmp_path: Path):
    from tcip_mcp.tools.inference_tools import deliver_per_image_counts

    return deliver_per_image_counts(predictions_dir=str(tmp_path / "preds"), trait="stem")


def _import_coco(tmp_path: Path):
    from tcip_mcp.tools.ingest_tools import import_coco

    document = tmp_path / "external.json"
    document.write_text('{"images": [], "annotations": [], "categories": []}', encoding="utf-8")
    return import_coco(str(document), str(tmp_path), "2025-09-14")


def _build_plant_mapping(tmp_path: Path):
    from tcip_mcp.tools.phenology_tools import build_plant_mapping

    return build_plant_mapping(name="../no", images_root=str(tmp_path),
                               plant_registry=str(tmp_path / "plants.csv"))


def _redraw_calibration_holdout(tmp_path: Path):
    from tcip_mcp.tools.calibration_tools import redraw_calibration_holdout

    return redraw_calibration_holdout(str(tmp_path), reason="")


def _deliver_per_plant_csv(tmp_path: Path):
    from tcip_mcp.tools.delivery_tools import deliver_per_plant_csv

    return deliver_per_plant_csv([], str(tmp_path / "out.csv"), "cyme count")


def _deliver_orthomosaic_plant_counts(tmp_path: Path):
    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    return deliver_orthomosaic_plant_counts(
        str(tmp_path / "preds"), str(tmp_path / "absent.tif"), str(tmp_path / "plants.csv"),
        str(tmp_path / "out.csv"), "cyme count")


def _deliver_phenology_milestones(tmp_path: Path):
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones

    return deliver_phenology_milestones("no-such-trait", "mapping", {}, str(tmp_path / "o.csv"),
                                        ["p1"])


def _save_annotations_on_a_missing_image(tmp_path: Path):
    from tcip_mcp.tools.annotation_tools import save_annotations

    return save_annotations(str(tmp_path / "images" / "absent.png"),
                            annotations=[{"subject": "cyme", "bbox": [1, 1, 5, 5]}])


def _calibrate_count_over_an_unstamped_bucket(tmp_path: Path):
    from tcip_mcp.tools.calibration_tools import calibrate_count_operating_point

    bucket = tmp_path / "predictions" / "m" / "2025-09-14"
    bucket.mkdir(parents=True)
    return calibrate_count_operating_point(
        str(tmp_path / "m.pt"), "cyme count", str(tmp_path / "labels"), str(tmp_path / "images"),
        str(tmp_path), str(bucket))


def _calibrate_scalar_with_no_such_task(tmp_path: Path):
    from tcip_mcp.tools.calibration_tools import calibrate_scalar_operating_point

    return calibrate_scalar_operating_point(
        "cyme count", "histogram", str(tmp_path / "m.pt"), str(tmp_path / "images"),
        str(tmp_path / "t.csv"), "mae", str(tmp_path / "out"), str(tmp_path))


def _calibrate_classifier_for_no_such_trait(tmp_path: Path):
    from tcip_mcp.tools.phenology_tools import calibrate_classifier_operating_point

    labels = str(tmp_path / "labels")
    return calibrate_classifier_operating_point(
        "no-such-trait", "cyme", "stage", labels, labels, labels, labels,
        str(tmp_path / "out"), str(tmp_path))


def _calibrate_scale_for_no_such_trait(tmp_path: Path):
    from tcip_mcp.tools.scale_tools import calibrate_physical_scale

    return calibrate_physical_scale(
        "no-such-trait", str(tmp_path / "preds"), str(tmp_path), str(tmp_path / "images"), "mm",
        "ruler", str(tmp_path / "labels"), str(tmp_path / "ref.csv"))


def _terminal_status_refusal(tmp_path: Path):
    from tcip_mcp.experiments import create_experiment, update_status

    create_experiment("exp-closed", {"model_source": {"builder": "x:y"}})
    update_status("exp-closed", "completed")
    return update_status("exp-closed", "failed")


DOORS = [_register_model, _deliver_per_image_counts, _import_coco, _build_plant_mapping,
         _redraw_calibration_holdout, _deliver_per_plant_csv, _deliver_orthomosaic_plant_counts,
         _deliver_phenology_milestones, _save_annotations_on_a_missing_image,
         _calibrate_count_over_an_unstamped_bucket, _calibrate_scalar_with_no_such_task,
         _calibrate_classifier_for_no_such_trait, _calibrate_scale_for_no_such_trait,
         _terminal_status_refusal]


@pytest.mark.parametrize("door", DOORS, ids=[d.__name__.lstrip("_") for d in DOORS])
def test_a_door_refused_before_it_acts_leaves_no_row(tmp_path: Path, door):
    try:
        result = door(tmp_path)
    except ValueError:
        result = {"error": "raised"}
    assert "error" in result, result
    assert _rows(tmp_path) == []


def _completed_run(tmp_path: Path, experiment_id: str) -> Path:
    """A run completed through the experiment record's own producers, its weights on disk."""
    from tcip_mcp.experiments import complete_run, create_experiment, update_status

    weights = tmp_path / f"{experiment_id}.pt"
    weights.write_bytes(f"{experiment_id} weights".encode())
    create_experiment(experiment_id, {"model_source": {"builder": "x:y"}})
    update_status(experiment_id, "running")
    assert "error" not in complete_run(experiment_id, str(weights))
    return weights


def test_the_registrys_ownership_refusal_leaves_no_row(tmp_path: Path, monkeypatch):
    """A second run registering a name the first run's completion bound is refused by the
    registry's ownership rail: the one row is the first registration's."""
    from tcip_mcp.experiments import register_model_from_experiment

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    first = register_model_from_experiment(
        "exp-first", str(_completed_run(tmp_path, "exp-first")), name="shared")
    assert "error" not in first, first
    rows = _rows(tmp_path)
    assert [r["tool"] for r in rows] == ["model_registered"]

    refused = register_model_from_experiment(
        "exp-second", str(_completed_run(tmp_path, "exp-second")), name="shared")
    assert "exp-first" in refused["error"]
    assert _rows(tmp_path) == rows


def test_a_decorated_door_whose_body_raises_keeps_its_exception_row(tmp_path: Path, monkeypatch):
    from PIL import Image

    import tcip_mcp.tools.annotation_tools as annotation_tools

    image = tmp_path / "images" / "2025-09-14" / "a.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(image)

    def _refused_write(*args, **kwargs):
        raise OSError("the label store refused the write")

    monkeypatch.setattr(annotation_tools, "write_annotations", _refused_write)
    with pytest.raises(OSError):
        annotation_tools.save_annotations(str(image), annotations=[{"subject": "cyme",
                                                                     "bbox": [1, 1, 5, 5]}])
    assert [(r["tool"], r["status"]) for r in _rows(tmp_path)] == [("save_annotations", "exception")]
