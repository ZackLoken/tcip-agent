"""Unit tests for the orthomosaic MCP tools: tiled inference over a whole raster, then per-plant
delivery from the persisted predictions + a plant-locations CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from tests import _operationalization_fixtures as fx

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

# UTM zone 15N: the same real projected CRS test_orthomosaic_mapping.py uses.
UTM_15N_EPSG = 32615
TIEPOINT_NATIVE_X = 500_000.0
TIEPOINT_NATIVE_Y = 4_800_000.0
PIXEL_SCALE = 0.5  # native-CRS units (m) per pixel

TILE = 32



@pytest.fixture(autouse=True)
def _recorded_meaning(tmp_path):
    """Every per-plant delivery below ships under a trait whose meaning is confirmed.

    Seeded into the project these tests pin as well as the one the autouse pin names, so a
    delivery reads the same registry whichever of the two it resolves against.
    """
    for project_root in (tmp_path, tmp_path / "proj"):
        fx.seed_delivery_traits(project_root)
        fx.seed_confirmed_aggregate(project_root, "stem_count", value_keys=["count"])


def _geokeys() -> tuple[int, ...]:
    entries = [1024, 0, 1, 1, 3072, 0, 1, UTM_15N_EPSG]  # GTModelType=Projected, ProjectedCSType
    return (1, 1, 0, len(entries) // 4, *entries)


def _write_geo_raster(path: Path, *, height: int = 64, width: int = 64, channels: int = 3,
                      rowsperstrip: int = 8, tiepoint_x: float = TIEPOINT_NATIVE_X,
                      seed: int = 0) -> np.ndarray:
    """A raster carrying both real georeferencing tags and real (random, decodable) pixel
    content, so it works for both :func:`read_geotransform` and a tiling inference pass.

    ``tiepoint_x``/``seed`` vary the two halves independently, so a caller can write a
    pixel-identical copy at a moved tiepoint, or different content at the same one."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(height, width, channels), dtype=np.uint8)
    extratags = [
        (33550, "d", 3, (PIXEL_SCALE, PIXEL_SCALE, 0.0), False),
        (33922, "d", 6, (0.0, 0.0, 0.0, tiepoint_x, TIEPOINT_NATIVE_Y, 0.0), False),
        (34735, "H", len(_geokeys()), _geokeys(), False),
    ]
    tifffile.imwrite(str(path), arr, rowsperstrip=rowsperstrip, extratags=extratags)
    return arr


def _bespoke_detection_checkpoint(tmp_path: Path, *, in_chans: int = 3, tile_size: int = TILE) -> str:
    from tcip_mcp.pipelines.model_build import build_model

    model_source = {"builder": "tests.bespoke_models:build_bespoke_detection",
                    "builder_kwargs": {"num_classes": 1, "in_chans": in_chans,
                                      "min_size": tile_size, "max_size": tile_size * 2},
                    "task": "detection", "in_chans": in_chans}
    model = build_model({"model_source": model_source})
    ckpt = tmp_path / "model_best.pt"
    torch.save({"model_source": model_source, "model_state_dict": model.state_dict()}, str(ckpt))
    return str(ckpt)


def _pixel_to_wgs84(raster_path: Path, px: float, py: float) -> tuple[float, float]:
    from tcip_mcp.pipelines.postprocessing.orthomosaic_mapping import OrthomosaicGeoreference

    return OrthomosaicGeoreference.from_file(raster_path).pixel_to_wgs84(px, py)


def _write_plant_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["plot_name", "accession_name", "plot_number", "row_number", "col_number",
                  "WGS84_centroid_y", "WGS84_centroid_x"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plant_grid_csv(tmp_path: Path, raster_path: Path,
                    plant_pixels: list[tuple[float, float]]) -> Path:
    rows = []
    for i, (px, py) in enumerate(plant_pixels):
        lat, lon = _pixel_to_wgs84(raster_path, px, py)
        rows.append({
            "plot_name": f"plot{i}", "accession_name": f"acc{i}",
            "plot_number": i, "row_number": i // 2, "col_number": i % 2,
            "WGS84_centroid_y": lat, "WGS84_centroid_x": lon,
        })
    csv_path = tmp_path / "plants.csv"
    _write_plant_csv(csv_path, rows)
    return csv_path


# A 2x2 plant grid, 40px apart, mirroring test_orthomosaic_mapping.py's own layout convention.
_PLANT_PIXELS = [(10.0, 10.0), (10.0, 50.0), (50.0, 10.0), (50.0, 50.0)]


# ── export_predictions (raster_path regime) ──────────────────────────────


def test_export_predictions_raster_writes_bucket_with_explicit_tile_size(tmp_path, monkeypatch):
    """An explicit tile_size clears the tile_size gate on its own (no acknowledgement needed);
    the persisted bucket carries one prediction file for the whole raster plus a real
    operating_point.json sidecar in the same shape every other bucket writes."""
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    ckpt = _bespoke_detection_checkpoint(tmp_path)

    from tcip_mcp.tools.inference_tools import export_predictions

    out_dir = tmp_path / "preds"
    result = export_predictions(
        ckpt, output_dir=str(out_dir), raster_path=str(raster_path), conf_threshold=0.0,
        tile_size=TILE, overlap=0.2)

    assert "error" not in result
    assert result["tiles"] > 1
    assert Path(result["output_dir"]) == out_dir
    pred_path = Path(result["files"][0])
    assert pred_path.is_file()
    assert pred_path.name == "mosaic.json"

    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

    sidecar = read_operating_point_sidecar(out_dir)
    assert sidecar["operating_point"]["tile_size"]["value"] == TILE
    assert sidecar["operating_point"]["tile_size"]["validated_against"] == "explicit_caller_stated_geometry"
    assert sidecar["tile_size_validated"] == "explicit_caller_stated_geometry"
    # conf is never validated at this raw-persist step.
    assert sidecar["operating_point"]["conf"]["validated_against"] == "false"
    assert sidecar["validated"] is False
    assert sidecar["checkpoint_sha256"]
    # Which raster produced this bucket is a provenance fact, the same as images_dir is for the
    # ordinary regime: a reviewer must be able to reconstruct how a number was produced.
    assert sidecar["raster_path"] == str(raster_path)

    data = json.loads(pred_path.read_text())
    assert data["width"] == 64 and data["height"] == 64
    assert isinstance(data["annotations"], list)


def test_a_second_orthomosaic_export_against_a_completed_experiment_refuses_before_writing(
        tmp_path, monkeypatch):
    """A pointer is checked before its write: a second export against an experiment whose
    lineage.predictions is already populated and terminal refuses by name, before the raster
    pass runs, so no second bucket is written to orphan."""
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)

    from tcip_mcp.experiments import create_experiment, update_status
    create_experiment("expOrtho", {"model_source": {"builder": "x:y"}})
    update_status("expOrtho", "running")

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    ckpt = _bespoke_detection_checkpoint(tmp_path)

    from tcip_mcp.tools.inference_tools import export_predictions

    out1 = tmp_path / "preds1"
    r1 = export_predictions(ckpt, output_dir=str(out1), raster_path=str(raster_path),
                            conf_threshold=0.0, tile_size=TILE, overlap=0.2,
                            experiment_id="expOrtho")
    assert "error" not in r1, r1
    update_status("expOrtho", "completed")

    out2 = tmp_path / "preds2"
    r2 = export_predictions(ckpt, output_dir=str(out2), raster_path=str(raster_path),
                            conf_threshold=0.0, tile_size=TILE, overlap=0.2,
                            experiment_id="expOrtho")
    assert "error" in r2
    assert not out2.exists()


def test_a_same_path_orthomosaic_export_against_a_completed_experiment_admits_the_second_run(
        tmp_path, monkeypatch):
    """A rail must admit valid work: re-exporting into the same raster bucket path records the
    same lineage value the completed experiment already holds, so the additive lock's same-value
    conjunct admits it rather than refusing a legitimate re-run."""
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)

    from tcip_mcp.experiments import create_experiment, update_status
    create_experiment("expOrthoSame", {"model_source": {"builder": "x:y"}})
    update_status("expOrthoSame", "running")

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    ckpt = _bespoke_detection_checkpoint(tmp_path)

    from tcip_mcp.tools.inference_tools import export_predictions

    out = tmp_path / "preds"
    r1 = export_predictions(ckpt, output_dir=str(out), raster_path=str(raster_path),
                            conf_threshold=0.0, tile_size=TILE, overlap=0.2,
                            experiment_id="expOrthoSame")
    assert "error" not in r1, r1
    update_status("expOrthoSame", "completed")

    r2 = export_predictions(ckpt, output_dir=str(out), raster_path=str(raster_path),
                            conf_threshold=0.0, tile_size=TILE, overlap=0.2,
                            experiment_id="expOrthoSame")
    assert "error" not in r2, r2


def test_export_predictions_raster_refuses_missing_checkpoint_cleanly(tmp_path, monkeypatch):
    """A missing checkpoint must return an honest {"error": ...}, the same shape every other
    refusal in this door uses, never an uncaught FileNotFoundError out of the MCP tool: the
    raster regime builds its own predictor directly rather than going through run_inference's own
    existence check."""
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)

    from tcip_mcp.tools.inference_tools import export_predictions

    out_dir = tmp_path / "preds"
    r = export_predictions(str(tmp_path / "missing.pt"), output_dir=str(out_dir),
                           raster_path=str(raster_path), conf_threshold=0.0)
    assert "error" in r
    assert "Checkpoint not found" in r["error"]
    assert not out_dir.exists()


def test_export_predictions_raster_refuses_missing_raster_cleanly(tmp_path, monkeypatch):
    """Same shape for a missing raster_path."""
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)
    ckpt = _bespoke_detection_checkpoint(tmp_path)

    from tcip_mcp.tools.inference_tools import export_predictions

    out_dir = tmp_path / "preds"
    r = export_predictions(ckpt, output_dir=str(out_dir),
                           raster_path=str(tmp_path / "missing.tif"), conf_threshold=0.0)
    assert "error" in r
    assert "raster_path not found" in r["error"]
    assert not out_dir.exists()


def test_export_predictions_raster_refuses_when_tile_size_has_no_real_basis(tmp_path, monkeypatch):
    """No persisted training geometry and no explicit tile_size -> no real basis to tile at, at
    all, unlike the ordinary images_dir regime this always-tiled regime has no untiled fallback to
    run instead, so this refusal is unconditional: acknowledge_unvalidated=True cannot un-stick it
    either, since there is no value to provisionally proceed with, never crashing mid-pass on a
    ``None`` tile_size."""
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    ckpt = _bespoke_detection_checkpoint(tmp_path)  # no persisted tiling geometry in its config

    from tcip_mcp.tools.inference_tools import export_predictions

    out_dir = tmp_path / "preds"
    refused = export_predictions(
        ckpt, output_dir=str(out_dir), raster_path=str(raster_path), conf_threshold=0.0)
    assert "error" in refused
    assert not out_dir.exists()  # refused before ever writing the bucket

    still_refused = export_predictions(
        ckpt, output_dir=str(out_dir), raster_path=str(raster_path), conf_threshold=0.0,
        acknowledge_unvalidated=True)
    assert "error" in still_refused
    assert not out_dir.exists()


def test_export_predictions_raster_bucket_immutability(tmp_path, monkeypatch):
    """A bucket with a recorded review verdict is never silently overwritten, the same
    immutability the images_dir regime already enforces, shared rather than reimplemented."""
    platform_root = tmp_path / "platform"
    (platform_root / ".tcip" / "state").mkdir(parents=True)
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(platform_root))

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    ckpt = _bespoke_detection_checkpoint(tmp_path)

    from tcip_mcp.tools.inference_tools import export_predictions

    # Inside a dataset, where the verdicts that freeze a bucket are recorded.
    dataset_root = tmp_path / "dataset"
    out_dir = dataset_root / "predictions" / "preds"
    first = export_predictions(
        ckpt, output_dir=str(out_dir), raster_path=str(raster_path), conf_threshold=0.0,
        tile_size=TILE)
    assert "error" not in first

    from tcip_annotation.review_engine import ReviewContext, ReviewDetection, ReviewEngine
    from tcip_annotation.state import Annotation, BBox

    from tcip_mcp.prediction_buckets import bucket_key_of

    engine = ReviewEngine(dataset_root / ".tcip" / "state")
    ctx = ReviewContext(img_name="mosaic", img_width=64, img_height=64,
                        preds=[Annotation(subject="0", geometry=BBox(1.0, 1.0, 5.0, 5.0), score=0.9)])
    det = ReviewDetection(det_type="fp", class_name="0", conf=0.9, iou=None, gt_idx=None,
                          pred_idx=0, bbox=(1.0, 1.0, 5.0, 5.0))
    engine.record_detection_action(bucket_key_of(out_dir), det, ctx, action="accepted")

    overwrite_attempt = export_predictions(
        ckpt, output_dir=str(out_dir), raster_path=str(raster_path), conf_threshold=0.0,
        tile_size=TILE, overwrite=True)
    assert "error" in overwrite_attempt and overwrite_attempt["verdict_count"] == 1
    assert Path(overwrite_attempt["suggested_bucket"]).name == "preds@r2"

    redirected = export_predictions(
        ckpt, output_dir=str(out_dir), raster_path=str(raster_path), conf_threshold=0.0,
        tile_size=TILE)
    assert "error" not in redirected
    assert redirected["bucket_redirected"] is True
    assert Path(redirected["output_dir"]).name == "preds@r2"


# ── deliver_orthomosaic_plant_counts ─────────────────────────────────────


def _run_bucket(tmp_path, monkeypatch, raster_path: Path) -> tuple[Path, str]:
    """A real bucket via export_predictions's raster_path regime (explicit tile_size, so it
    writes without acknowledgement); returns (bucket dir, prediction stem)."""
    from tcip_mcp.tools.inference_tools import export_predictions

    ckpt = _bespoke_detection_checkpoint(tmp_path)
    out_dir = tmp_path / "preds"
    result = export_predictions(
        ckpt, output_dir=str(out_dir), raster_path=str(raster_path), conf_threshold=0.0,
        tile_size=TILE)
    assert "error" not in result
    return out_dir, Path(result["files"][0]).stem


def _replace_boxes(pred_path: Path, boxes: list[tuple[float, float, float, float]]) -> None:
    """Overwrite a persisted prediction file's boxes with deterministic ones (the model's own
    random-weight output is not something a delivery-math test should depend on)."""
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    anns = [Annotation(subject="0", geometry=BBox(*b), score=0.9) for b in boxes]
    data = json.loads(pred_path.read_text())
    json_io.write_annotations(str(pred_path), anns, data["width"], data["height"], keep_empty=True)


def test_deliver_orthomosaic_plant_counts_refuses_unacknowledged_then_admits(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    bucket_dir, stem = _run_bucket(tmp_path, monkeypatch, raster_path)

    # Two detections near plant0, one near plant2, none near plant1/plant3.
    _replace_boxes(bucket_dir / f"{stem}.json", [
        (8.0, 8.0, 12.0, 12.0),
        (9.0, 9.0, 11.0, 11.0),
        (48.0, 8.0, 52.0, 12.0),
    ])
    plant_csv = _plant_grid_csv(tmp_path, raster_path, _PLANT_PIXELS)

    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    out_csv = tmp_path / "counts.csv"
    refused = deliver_orthomosaic_plant_counts(
        str(bucket_dir), str(raster_path), [str(plant_csv)], str(out_csv), trait_name="stem_count")
    assert "error" in refused
    assert not out_csv.exists()

    admitted = deliver_orthomosaic_plant_counts(
        str(bucket_dir), str(raster_path), [str(plant_csv)], str(out_csv), trait_name="stem_count",
        acknowledge_unvalidated=True)
    assert "error" not in admitted
    assert admitted["measurement_validated"] == "false"
    assert admitted["n_detections"] == 3
    assert admitted["n_mapped"] == 3
    assert admitted["n_unmapped"] == 0
    assert admitted["n_plants"] == 4  # every plant in the grid gets a row
    assert admitted["n_plants_zero_count"] == 2  # plant1, plant3

    rows = {r["plant_id"]: r for r in csv.DictReader(out_csv.open(newline=""))}
    assert rows["plot0"]["value"] == "2"
    assert rows["plot2"]["value"] == "1"
    assert rows["plot1"]["value"] == "0"
    assert rows["plot3"]["value"] == "0"
    for r in rows.values():
        assert r["measurement_validated"] == "false"
        assert r["trait_name"] == "stem_count"


def test_deliver_orthomosaic_plant_counts_floors_a_stamp_earned_for_a_different_trait(
    tmp_path, monkeypatch,
):
    """A count stamp validated for one trait must not answer for a per-plant delivery under a
    different trait: the refusal names the sidecar and both traits."""
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)

    from tcip_mcp.tools.inference_tools import export_predictions

    ckpt = _bespoke_detection_checkpoint(tmp_path)
    dataset_root = tmp_path / "ds"
    bucket_dir = dataset_root / "predictions" / "run1"
    result = export_predictions(
        ckpt, output_dir=str(bucket_dir), raster_path=str(raster_path), conf_threshold=0.0,
        tile_size=TILE)
    assert "error" not in result
    stem = Path(result["files"][0]).stem
    _replace_boxes(bucket_dir / f"{stem}.json", [(8.0, 8.0, 12.0, 12.0)])
    plant_csv = _plant_grid_csv(tmp_path, raster_path, _PLANT_PIXELS)

    from tcip_mcp.pipelines.resolution import VALIDATED_HELD_OUT, read_operating_point_sidecar
    from tests._binding_fixtures import write_bound_sidecar

    stamped = read_operating_point_sidecar(bucket_dir)
    assert stamped is not None
    op = dict(stamped["operating_point"])
    op["conf"] = {**op["conf"], "validated_against": VALIDATED_HELD_OUT}
    validated_stamp = {**stamped, "operating_point": op, "trait": "astringency", "validated": True}
    write_bound_sidecar(bucket_dir, validated_stamp, dataset_root=dataset_root,
                        experiment_id="exp-mismatched-trait")

    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    out_csv = tmp_path / "counts.csv"
    refused = deliver_orthomosaic_plant_counts(
        str(bucket_dir), str(raster_path), [str(plant_csv)], str(out_csv), trait_name="stem_count")
    assert "error" in refused
    assert not out_csv.exists()
    assert str(bucket_dir) in refused["error"]
    assert "astringency" in refused["error"] and fx.COUNT_TRAIT in refused["error"]


def test_deliver_orthomosaic_plant_counts_far_detection_is_unmapped(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    bucket_dir, stem = _run_bucket(tmp_path, monkeypatch, raster_path)

    _replace_boxes(bucket_dir / f"{stem}.json", [
        (8.0, 8.0, 12.0, 12.0),      # near plant0
        (3990.0, 3990.0, 4010.0, 4010.0),  # ~2 km away, no plant anywhere near
    ])
    plant_csv = _plant_grid_csv(tmp_path, raster_path, _PLANT_PIXELS)

    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    out_csv = tmp_path / "counts.csv"
    result = deliver_orthomosaic_plant_counts(
        str(bucket_dir), str(raster_path), [str(plant_csv)], str(out_csv), trait_name="stem_count",
        acknowledge_unvalidated=True)
    assert "error" not in result
    assert result["n_detections"] == 2
    assert result["n_mapped"] == 1
    assert result["n_unmapped"] == 1

    rows = {r["plant_id"]: r for r in csv.DictReader(out_csv.open(newline=""))}
    assert rows["plot0"]["value"] == "1"
    assert sum(int(r["value"]) for r in rows.values()) == 1  # the far detection contributes nowhere


def test_deliver_orthomosaic_plant_counts_rotated_raster_refuses_cleanly(tmp_path, monkeypatch):
    """A raster this module can't georeference (here: a ModelTransformationTag, refused by
    OrthomosaicGeoreference itself) surfaces as a clean error, not an uncaught exception."""
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)

    raster_path = tmp_path / "mosaic.tif"
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    transform = (
        PIXEL_SCALE, 0.0, 0.0, TIEPOINT_NATIVE_X,
        0.0, -PIXEL_SCALE, 0.0, TIEPOINT_NATIVE_Y,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    extratags = [
        (33550, "d", 3, (PIXEL_SCALE, PIXEL_SCALE, 0.0), False),
        (33922, "d", 6, (0.0, 0.0, 0.0, TIEPOINT_NATIVE_X, TIEPOINT_NATIVE_Y, 0.0), False),
        (34735, "H", len(_geokeys()), _geokeys(), False),
        (34264, "d", 16, transform, False),
    ]
    tifffile.imwrite(str(raster_path), arr, rowsperstrip=8, extratags=extratags)

    bucket_dir = tmp_path / "preds"
    bucket_dir.mkdir()
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    json_io.write_annotations(
        str(bucket_dir / "mosaic.json"),
        [Annotation(subject="0", geometry=BBox(1.0, 1.0, 5.0, 5.0), score=0.9)], 64, 64,
        keep_empty=True)
    # The bucket records the rotated raster's own identity so the delivery's identity check
    # passes and the refusal under test is the georeferencing one, not a missing identity.
    import dataclasses

    from tcip_mcp.pipelines.raster_source import (
        CONTENT_IDENTITY_MAX_WINDOWS,
        CONTENT_IDENTITY_SEED,
        CONTENT_IDENTITY_WINDOW_SIZE,
        raster_content_identity,
    )

    identity = dataclasses.asdict(raster_content_identity(
        str(raster_path), 3, seed=CONTENT_IDENTITY_SEED,
        window_size=CONTENT_IDENTITY_WINDOW_SIZE, max_windows=CONTENT_IDENTITY_MAX_WINDOWS))
    from tcip_mcp.pipelines.resolution import write_sidecar

    write_sidecar(bucket_dir, {"validated": False, "raster_content_identity": identity},
                 "operating_point")

    # Arbitrary geolocation, never derived from the rotated raster itself (which refuses to
    # resolve any pixel -> real-world coordinate at all): the point of this test is that
    # deliver_orthomosaic_plant_counts refuses cleanly before it ever needs one.
    plant_csv = tmp_path / "plants.csv"
    _write_plant_csv(plant_csv, [{
        "plot_name": "plot0", "accession_name": "acc0", "plot_number": 0, "row_number": 0,
        "col_number": 0, "WGS84_centroid_y": 42.0, "WGS84_centroid_x": -93.0,
    }])

    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    result = deliver_orthomosaic_plant_counts(
        str(bucket_dir), str(raster_path), [str(plant_csv)], str(tmp_path / "counts.csv"),
        trait_name="stem_count", acknowledge_unvalidated=True)
    assert "error" in result
    assert "ModelTransformationTag" in result["error"]


def test_deliver_orthomosaic_keeps_a_bespoke_producer_checkpoint(tmp_path, monkeypatch):
    """A bucket a bespoke checkpoint produced belongs to no experiment, so its checkpoint hash
    stands on its own and travels into the provisional delivery. The claim it carries is another
    matter: nothing validated the operating point, so it names no validation record."""
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    bucket_dir, stem = _run_bucket(tmp_path, monkeypatch, raster_path)
    _replace_boxes(bucket_dir / f"{stem}.json", [(8.0, 8.0, 12.0, 12.0)])
    plant_csv = _plant_grid_csv(tmp_path, raster_path, _PLANT_PIXELS)

    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    out_csv = tmp_path / "counts.csv"
    result = deliver_orthomosaic_plant_counts(
        str(bucket_dir), str(raster_path), [str(plant_csv)], str(out_csv),
        trait_name="stem_count", acknowledge_unvalidated=True)

    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

    assert "error" not in result
    stamped = read_operating_point_sidecar(bucket_dir)
    assert result["checkpoint_sha256"] == stamped["checkpoint_sha256"]
    assert result["experiment_id"] is None
    assert result["validation_record"] == ""
    row = next(csv.DictReader(out_csv.open(newline="")))
    assert row["producer_model_sha256"] == stamped["checkpoint_sha256"]
    assert row["validation_record"] == ""

    import tcip_store as ts
    from tcip_mcp.audit import audit_log_key

    emitted = ts.read_log(audit_log_key(tmp_path / "proj")).records
    door = [e for e in emitted if e["tool"] == "deliver_orthomosaic_plant_counts"
            and "verified_buckets" in e]
    assert len(door) == 1, emitted
    assert door[0]["verified_buckets"][str(bucket_dir)]["record"] == ""
    assert door[0]["record_digests"] == []


def test_deliver_orthomosaic_drops_a_producer_no_experiment_answers_for(tmp_path, monkeypatch):
    """The recorded route's own shape: a stamp naming a run the experiment store never held. The
    provisional CSV says the producer is unknown instead of naming an experiment that never ran."""
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    bucket_dir, stem = _run_bucket(tmp_path, monkeypatch, raster_path)
    _replace_boxes(bucket_dir / f"{stem}.json", [(8.0, 8.0, 12.0, 12.0)])
    plant_csv = _plant_grid_csv(tmp_path, raster_path, _PLANT_PIXELS)

    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar, write_sidecar

    stamped = read_operating_point_sidecar(bucket_dir)
    assert stamped is not None
    write_sidecar(bucket_dir, {**stamped, "checkpoint_sha256": "0" * 64,
                              "experiment_id": "exp_that_never_ran"}, "operating_point")

    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    out_csv = tmp_path / "counts.csv"
    result = deliver_orthomosaic_plant_counts(
        str(bucket_dir), str(raster_path), [str(plant_csv)], str(out_csv),
        trait_name="stem_count", acknowledge_unvalidated=True)

    assert "error" not in result
    assert result["checkpoint_sha256"] is None
    assert result["experiment_id"] is None
    row = next(csv.DictReader(out_csv.open(newline="")))
    assert row["producer_model_sha256"] == ""
    assert row["experiment_id"] == ""
    assert row["validation_record"] == ""
