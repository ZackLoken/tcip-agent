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
    """Write a bespoke detection checkpoint and register it in explicit mode against the
    platform state root the caller has already pinned (``TCIP_STATE_ROOT``), so a caller can
    hand its bare path to a door that resolves the registry itself."""
    from tcip_mcp.pipelines.model_build import build_model
    from tcip_mcp.tools.model_tools import register_model

    model_source = {"builder": "tests.bespoke_models:build_bespoke_detection",
                    "builder_kwargs": {"num_classes": 1, "in_chans": in_chans,
                                      "min_size": tile_size, "max_size": tile_size * 2},
                    "task": "detection", "in_chans": in_chans}
    model = build_model({"model_source": model_source})
    ckpt = tmp_path / "model_best.pt"
    torch.save({"model_source": model_source, "model_state_dict": model.state_dict()}, str(ckpt))
    result = register_model(name="test-model", checkpoint_path=str(ckpt), config={})
    assert "error" not in result, result
    return str(ckpt)


def _pixel_to_wgs84(raster_path: Path, px: float, py: float) -> tuple[float, float]:
    from tcip_mcp.pipelines.postprocessing.orthomosaic_mapping import OrthomosaicGeoreference

    return OrthomosaicGeoreference.from_file(raster_path).pixel_to_wgs84(px, py)


def _plant_registry(plant_csv: Path, *, name: str = "reg") -> str:
    from tests._binding_fixtures import register_plant_registry_for

    return register_plant_registry_for([plant_csv], name=name)


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


# ── run_inference (raster_path regime) ──────────────────────────────


def test_run_inference_raster_writes_bucket_with_explicit_tile_size(tmp_path, monkeypatch):
    """An explicit tile_size clears the tile_size gate on its own (no acknowledgement needed);
    the persisted bucket carries one prediction file for the whole raster plus a real
    operating_point.json sidecar in the same shape every other bucket writes."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    ckpt = _bespoke_detection_checkpoint(tmp_path)

    from tcip_mcp.tools.inference_tools import run_inference

    out_dir = tmp_path / "preds"
    result = run_inference(
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
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)

    from tcip_mcp.experiments import create_experiment, update_status
    create_experiment("expOrtho", {"model_source": {"builder": "x:y"}})
    update_status("expOrtho", "running")

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    ckpt = _bespoke_detection_checkpoint(tmp_path)

    from tcip_mcp.tools.inference_tools import run_inference

    out1 = tmp_path / "preds1"
    r1 = run_inference(ckpt, output_dir=str(out1), raster_path=str(raster_path),
                            conf_threshold=0.0, tile_size=TILE, overlap=0.2,
                            experiment_id="expOrtho")
    assert "error" not in r1, r1
    update_status("expOrtho", "completed")

    out2 = tmp_path / "preds2"
    r2 = run_inference(ckpt, output_dir=str(out2), raster_path=str(raster_path),
                            conf_threshold=0.0, tile_size=TILE, overlap=0.2,
                            experiment_id="expOrtho")
    assert "error" in r2
    assert not out2.exists()


def test_a_same_path_orthomosaic_export_against_a_completed_experiment_admits_the_second_run(
        tmp_path, monkeypatch):
    """A rail must admit valid work: re-exporting into the same raster bucket path records the
    same lineage value the completed experiment already holds, so the additive lock's same-value
    conjunct admits it rather than refusing a legitimate re-run."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)

    from tcip_mcp.experiments import create_experiment, update_status
    create_experiment("expOrthoSame", {"model_source": {"builder": "x:y"}})
    update_status("expOrthoSame", "running")

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    ckpt = _bespoke_detection_checkpoint(tmp_path)

    from tcip_mcp.tools.inference_tools import run_inference

    out = tmp_path / "preds"
    r1 = run_inference(ckpt, output_dir=str(out), raster_path=str(raster_path),
                            conf_threshold=0.0, tile_size=TILE, overlap=0.2,
                            experiment_id="expOrthoSame")
    assert "error" not in r1, r1
    update_status("expOrthoSame", "completed")

    r2 = run_inference(ckpt, output_dir=str(out), raster_path=str(raster_path),
                            conf_threshold=0.0, tile_size=TILE, overlap=0.2,
                            experiment_id="expOrthoSame")
    assert "error" not in r2, r2


def test_run_inference_raster_refuses_missing_checkpoint_cleanly(tmp_path, monkeypatch):
    """A missing checkpoint must return an honest {"error": ...}, the same shape every other
    refusal in this door uses, never an uncaught FileNotFoundError out of the MCP tool: the
    raster regime builds its own predictor directly rather than going through run_inference's own
    existence check."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)

    from tcip_mcp.tools.inference_tools import run_inference

    out_dir = tmp_path / "preds"
    r = run_inference(str(tmp_path / "missing.pt"), output_dir=str(out_dir),
                           raster_path=str(raster_path), conf_threshold=0.0)
    assert "error" in r
    assert "Checkpoint not found" in r["error"]
    assert not out_dir.exists()


def test_run_inference_raster_refuses_missing_raster_cleanly(tmp_path, monkeypatch):
    """Same shape for a missing raster_path."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)
    ckpt = _bespoke_detection_checkpoint(tmp_path)

    from tcip_mcp.tools.inference_tools import run_inference

    out_dir = tmp_path / "preds"
    r = run_inference(ckpt, output_dir=str(out_dir),
                           raster_path=str(tmp_path / "missing.tif"), conf_threshold=0.0)
    assert "error" in r
    assert "raster_path not found" in r["error"]
    assert not out_dir.exists()


def test_run_inference_raster_refuses_when_tile_size_has_no_real_basis(tmp_path, monkeypatch):
    """No persisted training geometry and no explicit tile_size -> no real basis to tile at, at
    all, unlike the ordinary images_dir regime this always-tiled regime has no untiled fallback to
    run instead, so this refusal is unconditional: allow_unvalidated_staging=True cannot un-stick it
    either, since there is no value to provisionally proceed with, never crashing mid-pass on a
    ``None`` tile_size."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    ckpt = _bespoke_detection_checkpoint(tmp_path)  # no persisted tiling geometry in its config

    from tcip_mcp.tools.inference_tools import run_inference

    out_dir = tmp_path / "preds"
    refused = run_inference(
        ckpt, output_dir=str(out_dir), raster_path=str(raster_path), conf_threshold=0.0)
    assert "error" in refused
    assert not out_dir.exists()  # refused before ever writing the bucket

    still_refused = run_inference(
        ckpt, output_dir=str(out_dir), raster_path=str(raster_path), conf_threshold=0.0,
        allow_unvalidated_staging=True)
    assert "error" in still_refused
    assert not out_dir.exists()


def test_run_inference_raster_bucket_immutability(tmp_path, monkeypatch):
    """A bucket with a recorded review verdict is never silently overwritten, the same
    immutability the images_dir regime already enforces, shared rather than reimplemented."""
    platform_root = tmp_path / "platform"
    (platform_root / ".tcip" / "state").mkdir(parents=True)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(platform_root))

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    ckpt = _bespoke_detection_checkpoint(tmp_path)

    from tcip_mcp.tools.inference_tools import run_inference

    # Inside a dataset, where the verdicts that freeze a bucket are recorded.
    dataset_root = tmp_path / "dataset"
    out_dir = dataset_root / "predictions" / "preds"
    first = run_inference(
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

    overwrite_attempt = run_inference(
        ckpt, output_dir=str(out_dir), raster_path=str(raster_path), conf_threshold=0.0,
        tile_size=TILE, overwrite=True)
    assert "error" in overwrite_attempt and overwrite_attempt["verdict_count"] == 1
    assert Path(overwrite_attempt["suggested_bucket"]).name == "preds@r2"

    redirected = run_inference(
        ckpt, output_dir=str(out_dir), raster_path=str(raster_path), conf_threshold=0.0,
        tile_size=TILE)
    assert "error" not in redirected
    assert redirected["bucket_redirected"] is True
    assert Path(redirected["output_dir"]).name == "preds@r2"


# ── deliver_orthomosaic_plant_counts ─────────────────────────────────────


def test_deliver_orthomosaic_plant_counts_signature_carries_delivered_phenotype_not_trait_name():
    """The vocabulary-sense parameter is delivered_phenotype; trait_name (the registry sense
    the rest of the tool surface keeps) is no longer a valid keyword here."""
    import inspect

    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    params = inspect.signature(deliver_orthomosaic_plant_counts).parameters
    assert "delivered_phenotype" in params
    assert "trait_name" not in params


def _run_bucket(tmp_path, monkeypatch, raster_path: Path) -> tuple[Path, str]:
    """A real bucket via run_inference's raster_path regime (explicit tile_size, so it
    writes without acknowledgement); returns (bucket dir, prediction stem). Written under a
    canonical dataset layout so a caller can promote its conf claim afterward (verify_stamp_binding's
    covered-bucket check needs a real dataset root to key the bucket under)."""
    from tcip_mcp.tools.inference_tools import run_inference

    ckpt = _bespoke_detection_checkpoint(tmp_path)
    out_dir = tmp_path / "ds" / "predictions" / "preds"
    result = run_inference(
        ckpt, output_dir=str(out_dir), raster_path=str(raster_path), conf_threshold=0.0,
        tile_size=TILE)
    assert "error" not in result
    return out_dir, Path(result["files"][0]).stem


def _promote_bucket_conf(
    bucket_dir: Path, dataset_root: Path, *, trait: str, tag: str = "a",
    producing_experiment_id: str | None = None,
) -> None:
    """Promote a raw bucket's conf dimension to a genuine held-out-validated claim, over the
    bucket's content exactly as it stands now: call after any prediction-file edits, never before,
    since the record covers the bucket's bytes at filing time.

    ``producing_experiment_id`` names the run that produced the predictions, ``None`` by default
    (the ordinary bespoke-checkpoint case every caller here uses): distinct from the calibration
    experiment this claim is filed under, which always names the fixture's own promotion, never a
    training run.
    """
    from tcip_mcp.pipelines.resolution import VALIDATED_HELD_OUT, read_operating_point_sidecar
    from tests._binding_fixtures import write_bound_sidecar

    sidecar = read_operating_point_sidecar(bucket_dir) or {}
    op = dict(sidecar.get("operating_point") or {})
    op["conf"] = {**op.get("conf", {}), "validated_against": VALIDATED_HELD_OUT}
    stamp = {**sidecar, "validated": True, "trait": trait, "operating_point": op}
    write_bound_sidecar(bucket_dir, stamp, dataset_root=dataset_root, experiment_id=f"exp-promoted-{tag}",
                        producing_experiment_id=producing_experiment_id)


def _replace_boxes(pred_path: Path, boxes: list[tuple[float, float, float, float]]) -> None:
    """Overwrite a persisted prediction file's boxes with deterministic ones (the model's own
    random-weight output is not something a delivery-math test should depend on)."""
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    anns = [Annotation(subject="0", geometry=BBox(*b), score=0.9) for b in boxes]
    data = json.loads(pred_path.read_text())
    json_io.write_annotations(str(pred_path), anns, data["width"], data["height"], keep_empty=True)


def test_deliver_orthomosaic_plant_counts_refuses_unvalidated_then_delivers_once_validated(
    tmp_path, monkeypatch,
):
    """This door takes no acknowledgement, so a bare unvalidated count always refuses (and the
    retired escape hatch is gone outright, a TypeError rather than a quieter admission); the same
    delivery ships once the bucket earns a real reference."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "proj"))
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
        str(bucket_dir), str(raster_path), _plant_registry(plant_csv), str(out_csv), delivered_phenotype="stem_count")
    assert "error" in refused
    assert refused["unvalidated_dimensions"] == "operating_point"
    assert refused["operating_point_validated"] == "false"
    assert refused["n_detections"] == 3 and refused["n_mapped"] == 3
    assert not out_csv.exists()

    with pytest.raises(TypeError):
        deliver_orthomosaic_plant_counts(
            str(bucket_dir), str(raster_path), _plant_registry(plant_csv), str(out_csv),
            delivered_phenotype="stem_count", acknowledge_unvalidated=True)
    assert not out_csv.exists()

    from tcip_mcp.pipelines.resolution import VALIDATED_HELD_OUT

    _promote_bucket_conf(bucket_dir, bucket_dir.parents[1], trait=fx.COUNT_TRAIT)
    delivered = deliver_orthomosaic_plant_counts(
        str(bucket_dir), str(raster_path), _plant_registry(plant_csv), str(out_csv),
        delivered_phenotype="stem_count")
    assert "error" not in delivered
    assert delivered["operating_point_validated"] == VALIDATED_HELD_OUT
    assert delivered["unvalidated_dimensions"] == ""
    assert delivered["n_detections"] == 3
    assert delivered["n_mapped"] == 3
    assert delivered["n_unmapped"] == 0
    assert delivered["n_plants"] == 4  # every plant in the grid gets a row
    assert delivered["n_plants_zero_count"] == 2  # plant1, plant3

    reader = csv.DictReader(out_csv.open(newline=""))
    rows = {r["plant_id"]: r for r in reader}
    assert "trait_name" not in (reader.fieldnames or [])
    assert "operating_point_validated" in (reader.fieldnames or [])
    assert "measurement_validated" not in (reader.fieldnames or [])
    assert rows["plot0"]["value"] == "2"
    assert rows["plot2"]["value"] == "1"
    assert rows["plot1"]["value"] == "0"
    assert rows["plot3"]["value"] == "0"
    for r in rows.values():
        assert r["operating_point_validated"] == VALIDATED_HELD_OUT
        assert r["delivered_phenotype"] == "stem_count"


def test_deliver_orthomosaic_plant_counts_csv_carries_detection_level_attribution(
    tmp_path, monkeypatch,
):
    """Orthomosaic delivery attributes objects to plants at the raw-detection level, never a
    per-image aggregate: the CSV's plant_attribution column names that granularity."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    bucket_dir, stem = _run_bucket(tmp_path, monkeypatch, raster_path)
    _replace_boxes(bucket_dir / f"{stem}.json", [(8.0, 8.0, 12.0, 12.0)])
    _promote_bucket_conf(bucket_dir, bucket_dir.parents[1], trait=fx.COUNT_TRAIT)
    plant_csv = _plant_grid_csv(tmp_path, raster_path, _PLANT_PIXELS)

    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    out_csv = tmp_path / "counts.csv"
    result = deliver_orthomosaic_plant_counts(
        str(bucket_dir), str(raster_path), _plant_registry(plant_csv), str(out_csv),
        delivered_phenotype="stem_count")
    assert "error" not in result, result

    rows = list(csv.DictReader(out_csv.open(newline="")))
    assert rows
    assert all(r["plant_attribution"] == "detection" for r in rows)


def test_deliver_orthomosaic_plant_counts_records_exactly_one_delivery_event(tmp_path, monkeypatch):
    """One orthomosaic export writes exactly one delivery event, carrying this door's own name."""
    import tcip_store as ts

    from tcip_mcp.pipelines import resolution

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    bucket_dir, stem = _run_bucket(tmp_path, monkeypatch, raster_path)
    _replace_boxes(bucket_dir / f"{stem}.json", [(8.0, 8.0, 12.0, 12.0)])
    _promote_bucket_conf(bucket_dir, bucket_dir.parents[1], trait=fx.COUNT_TRAIT)
    plant_csv = _plant_grid_csv(tmp_path, raster_path, _PLANT_PIXELS)

    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    out_csv = tmp_path / "counts.csv"
    result = deliver_orthomosaic_plant_counts(
        str(bucket_dir), str(raster_path), _plant_registry(plant_csv), str(out_csv),
        delivered_phenotype="stem_count")
    assert "error" not in result, result

    scope = resolution.delivery_events_scope(tmp_path / "proj")
    events = [ts.read(k) for k in ts.keys(resolution.DELIVERY_EVENTS_STORE, str(scope))]
    assert len(events) == 1
    assert events[0]["door"] == "deliver_orthomosaic_plant_counts"


def test_deliver_orthomosaic_plant_counts_excludes_a_point_from_the_count(tmp_path, monkeypatch):
    """A Point annotation in a reviewed bucket names no detection to georeference or count
    (bbox_of has no box for one by design): the door must read the bucket through the same
    real-detections predicate count_by_class already shares, excluding it, rather than crash on
    the first Point it meets."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    bucket_dir, stem = _run_bucket(tmp_path, monkeypatch, raster_path)

    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox, Point

    pred_path = bucket_dir / f"{stem}.json"
    data = json.loads(pred_path.read_text())
    anns = [
        Annotation(subject="0", geometry=BBox(8.0, 8.0, 12.0, 12.0), score=0.9),
        Annotation(subject="0", geometry=Point(60.0, 60.0), score=0.8),
    ]
    json_io.write_annotations(str(pred_path), anns, data["width"], data["height"], keep_empty=True)
    _promote_bucket_conf(bucket_dir, bucket_dir.parents[1], trait=fx.COUNT_TRAIT)

    plant_csv = _plant_grid_csv(tmp_path, raster_path, _PLANT_PIXELS)

    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    out_csv = tmp_path / "counts.csv"
    delivered = deliver_orthomosaic_plant_counts(
        str(bucket_dir), str(raster_path), _plant_registry(plant_csv), str(out_csv),
        delivered_phenotype="stem_count")

    assert "error" not in delivered, delivered
    assert delivered["n_detections"] == 1  # the Point excluded, never read as a boxless detection
    assert delivered["n_mapped"] == 1
    rows = {r["plant_id"]: r for r in csv.DictReader(out_csv.open(newline=""))}
    assert rows["plot0"]["value"] == "1"


def test_deliver_orthomosaic_plant_counts_floors_a_stamp_earned_for_a_different_trait(
    tmp_path, monkeypatch,
):
    """A count stamp validated for one trait must not answer for a per-plant delivery under a
    different trait: the refusal names the sidecar and both traits."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)

    from tcip_mcp.tools.inference_tools import run_inference

    ckpt = _bespoke_detection_checkpoint(tmp_path)
    dataset_root = tmp_path / "ds"
    bucket_dir = dataset_root / "predictions" / "run1"
    result = run_inference(
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
        str(bucket_dir), str(raster_path), _plant_registry(plant_csv), str(out_csv), delivered_phenotype="stem_count")
    assert "error" in refused
    assert not out_csv.exists()
    assert str(bucket_dir) in refused["error"]
    assert "astringency" in refused["error"] and fx.COUNT_TRAIT in refused["error"]


def test_deliver_orthomosaic_plant_counts_sibling_tile_floor_despite_valid_conf(tmp_path, monkeypatch):
    """A per-plant delivery whose count operating point genuinely validated still refuses when a
    sibling gated dimension (tile_size here) has no real basis, since this door takes no
    acknowledgement: the refusal names tile_size as the actual floorer, and separately reports the
    operating_point dimension's own cleared reference rather than folding it into the floor."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)

    from tcip_mcp.tools.inference_tools import run_inference

    ckpt = _bespoke_detection_checkpoint(tmp_path)
    dataset_root = tmp_path / "ds"
    bucket_dir = dataset_root / "predictions" / "run1"
    result = run_inference(
        ckpt, output_dir=str(bucket_dir), raster_path=str(raster_path), conf_threshold=0.0,
        tile_size=TILE)
    assert "error" not in result
    stem = Path(result["files"][0]).stem
    _replace_boxes(bucket_dir / f"{stem}.json", [(8.0, 8.0, 12.0, 12.0)])
    plant_csv = _plant_grid_csv(tmp_path, raster_path, _PLANT_PIXELS)

    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE, VALIDATED_HELD_OUT, read_operating_point_sidecar
    from tests._binding_fixtures import write_bound_sidecar

    stamped = read_operating_point_sidecar(bucket_dir)
    assert stamped is not None
    op = dict(stamped["operating_point"])
    op["conf"] = {**op["conf"], "validated_against": VALIDATED_HELD_OUT}
    # Patched here rather than published this way: run_inference's own persist-time gate
    # refuses an unfounded tile_size unconditionally, so no real bucket can carry one directly.
    op["tile_size"] = {**op["tile_size"], "validated_against": VALIDATED_FALSE}
    validated_stamp = {**stamped, "operating_point": op, "trait": fx.COUNT_TRAIT, "validated": True}
    write_bound_sidecar(bucket_dir, validated_stamp, dataset_root=dataset_root,
                        experiment_id="exp-sibling-tile-floor")

    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    out_csv = tmp_path / "counts.csv"
    refused = deliver_orthomosaic_plant_counts(
        str(bucket_dir), str(raster_path), _plant_registry(plant_csv), str(out_csv),
        delivered_phenotype="stem_count")
    assert "error" in refused
    assert not out_csv.exists()
    # The genuinely-cleared operating_point reference is reported on its own, never folded into
    # the tile_size floor that actually blocks this delivery.
    assert refused["operating_point_validated"] == VALIDATED_HELD_OUT
    assert refused["unvalidated_dimensions"] == "tile_size"


def test_deliver_orthomosaic_plant_counts_far_detection_is_unmapped(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    bucket_dir, stem = _run_bucket(tmp_path, monkeypatch, raster_path)

    _replace_boxes(bucket_dir / f"{stem}.json", [
        (8.0, 8.0, 12.0, 12.0),      # near plant0
        (3990.0, 3990.0, 4010.0, 4010.0),  # ~2 km away, no plant anywhere near
    ])
    _promote_bucket_conf(bucket_dir, bucket_dir.parents[1], trait=fx.COUNT_TRAIT)
    plant_csv = _plant_grid_csv(tmp_path, raster_path, _PLANT_PIXELS)

    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    out_csv = tmp_path / "counts.csv"
    result = deliver_orthomosaic_plant_counts(
        str(bucket_dir), str(raster_path), _plant_registry(plant_csv), str(out_csv),
        delivered_phenotype="stem_count")
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
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "proj"))
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
        str(bucket_dir), str(raster_path), _plant_registry(plant_csv), str(tmp_path / "counts.csv"),
        delivered_phenotype="stem_count")
    assert "error" in result
    assert "ModelTransformationTag" in result["error"]


def test_deliver_orthomosaic_keeps_a_bespoke_producer_checkpoint(tmp_path, monkeypatch):
    """A bucket a bespoke checkpoint produced belongs to no experiment, so its checkpoint hash
    stands on its own and travels into the delivered CSV, even though the validated claim itself
    was earned by a separate calibration record rather than by any training run."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    bucket_dir, stem = _run_bucket(tmp_path, monkeypatch, raster_path)
    _replace_boxes(bucket_dir / f"{stem}.json", [(8.0, 8.0, 12.0, 12.0)])
    _promote_bucket_conf(bucket_dir, bucket_dir.parents[1], trait=fx.COUNT_TRAIT)
    plant_csv = _plant_grid_csv(tmp_path, raster_path, _PLANT_PIXELS)

    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    out_csv = tmp_path / "counts.csv"
    result = deliver_orthomosaic_plant_counts(
        str(bucket_dir), str(raster_path), _plant_registry(plant_csv), str(out_csv),
        delivered_phenotype="stem_count")

    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

    assert "error" not in result
    stamped = read_operating_point_sidecar(bucket_dir)
    assert result["checkpoint_sha256"] == stamped["checkpoint_sha256"]
    assert result["producing_experiment_id"] is None
    assert result["validation_record"] != ""
    row = next(csv.DictReader(out_csv.open(newline="")))
    assert row["producer_model_sha256"] == stamped["checkpoint_sha256"]
    assert row["validation_record"] != ""

    import tcip_store as ts
    from tcip_mcp.audit import audit_log_key

    # The bucket now sits under a real dataset root, so this event files in the dataset's own
    # log (record_delivery_binding_event's own scoping), not the project's.
    emitted = ts.read_log(audit_log_key(bucket_dir.parents[1])).records
    door = [e for e in emitted if e["tool"] == "deliver_orthomosaic_plant_counts"
            and "verified_buckets" in e]
    assert len(door) == 1, emitted
    assert door[0]["verified_buckets"][str(bucket_dir)]["record"] != ""
    assert door[0]["record_digests"] != []


def test_deliver_orthomosaic_drops_a_producer_no_experiment_answers_for(tmp_path, monkeypatch):
    """The recorded route's own shape: a validated claim naming a producing run the experiment
    store never held reports the producer unknown rather than naming a run that never ran, even
    though the count claim's own validation record (a separate calibration record) still holds."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    bucket_dir, stem = _run_bucket(tmp_path, monkeypatch, raster_path)
    _replace_boxes(bucket_dir / f"{stem}.json", [(8.0, 8.0, 12.0, 12.0)])
    _promote_bucket_conf(bucket_dir, bucket_dir.parents[1], trait=fx.COUNT_TRAIT,
                         producing_experiment_id="exp_that_never_ran")
    plant_csv = _plant_grid_csv(tmp_path, raster_path, _PLANT_PIXELS)

    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    out_csv = tmp_path / "counts.csv"
    result = deliver_orthomosaic_plant_counts(
        str(bucket_dir), str(raster_path), _plant_registry(plant_csv), str(out_csv),
        delivered_phenotype="stem_count")

    assert "error" not in result
    assert result["checkpoint_sha256"] is None
    assert result["producing_experiment_id"] is None
    row = next(csv.DictReader(out_csv.open(newline="")))
    assert row["producer_model_sha256"] == ""
    assert row["producing_experiment_id"] == ""
    assert row["validation_record"] != ""


# ── plant_mapping disclosure (PlantRegistryDisclosure) ───────────────────


def test_deliver_orthomosaic_plant_counts_records_a_registry_disclosure_that_reads_back_validated(
    tmp_path, monkeypatch,
):
    """A whole-raster frame carries no walked MappingBuild, so this door's own delivery event
    names the registry it read, the raster identity, the matched tolerance and its source, and
    this delivery's own unattributed-detection count, and that disclosure reads back through
    read_delivery_events, which already validates every record against DeliveryEventRecord
    (PlantRegistryDisclosure). A second delivery under an explicit nn_tolerance_m records that
    value under source "stated" rather than the derived "grid_pitch" the first delivery gets."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    bucket_dir, stem = _run_bucket(tmp_path, monkeypatch, raster_path)
    _replace_boxes(bucket_dir / f"{stem}.json", [
        (8.0, 8.0, 12.0, 12.0),
        (3990.0, 3990.0, 4010.0, 4010.0),  # far from every plant: contributes to n_unmapped
    ])
    _promote_bucket_conf(bucket_dir, bucket_dir.parents[1], trait=fx.COUNT_TRAIT)
    plant_csv = _plant_grid_csv(tmp_path, raster_path, _PLANT_PIXELS)
    registry_name = _plant_registry(plant_csv)

    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    out_csv = tmp_path / "counts.csv"
    result = deliver_orthomosaic_plant_counts(
        str(bucket_dir), str(raster_path), registry_name, str(out_csv),
        delivered_phenotype="stem_count")
    assert "error" not in result, result
    assert result["n_unmapped"] == 1

    from tcip_mcp.pipelines.postprocessing.plant_mapping import grid_pitch_m, read_plant_csvs
    from tcip_mcp.pipelines.resolution import read_delivery_events, read_operating_point_sidecar

    events = read_delivery_events(tmp_path / "proj")
    assert len(events) == 1
    event = events[0]

    pm = event["plant_mapping"]
    assert pm["plant_registry"]["name"] == registry_name
    assert pm["detections_unattributed"] == 1
    assert pm["detections_unattributed_scope"] == "delivered_raster"
    assert pm["plant_attribution"] == "detection"
    plants = read_plant_csvs([plant_csv])
    assert pm["nn_tolerance_m"] == {"value": grid_pitch_m(plants) / 6, "source": "grid_pitch"}
    sidecar = read_operating_point_sidecar(bucket_dir)
    assert pm["raster_identity"] == sidecar["raster_content_identity"]
    assert "dates_delivered" not in pm  # the walked-mapping form's own keys, never named here
    assert "record_sha256" not in pm

    stated_tolerance = 5.0
    out_csv_stated = tmp_path / "counts_stated.csv"
    result_stated = deliver_orthomosaic_plant_counts(
        str(bucket_dir), str(raster_path), registry_name, str(out_csv_stated),
        delivered_phenotype="stem_count", nn_tolerance_m=stated_tolerance)
    assert "error" not in result_stated, result_stated

    existing_ids = {e["event_id"] for e in events}
    new_events = [
        e for e in read_delivery_events(tmp_path / "proj") if e["event_id"] not in existing_ids]
    assert len(new_events) == 1
    assert new_events[0]["plant_mapping"]["nn_tolerance_m"] == {
        "value": stated_tolerance, "source": "stated"}


def test_deliver_orthomosaic_plant_counts_refuses_a_registry_csv_rewritten_after_registration(
    tmp_path, monkeypatch,
):
    """The registry byte check this door now runs (verify_registry_csv_bytes, shared with the
    walked-mapping verifier) refuses by name when a registered plant CSV's bytes moved since
    registration, before any plant or prediction is read: every CSV this delivery reads is
    verified or the delivery never happens. The refused sentence names the fact
    (verify_registry_csv_bytes' own words) plus this door's own composed remedy, never the
    walked-mapping verifier's different one."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    bucket_dir, stem = _run_bucket(tmp_path, monkeypatch, raster_path)
    _replace_boxes(bucket_dir / f"{stem}.json", [(8.0, 8.0, 12.0, 12.0)])
    _promote_bucket_conf(bucket_dir, bucket_dir.parents[1], trait=fx.COUNT_TRAIT)
    plant_csv = _plant_grid_csv(tmp_path, raster_path, _PLANT_PIXELS)
    registry_name = _plant_registry(plant_csv)

    plant_csv.write_text(
        "plot_name,accession_name,WGS84_centroid_x,WGS84_centroid_y\n"
        "plot0,acc0,-93.0,42.0\n", encoding="utf-8")

    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    out_csv = tmp_path / "counts.csv"
    refused = deliver_orthomosaic_plant_counts(
        str(bucket_dir), str(raster_path), registry_name, str(out_csv),
        delivered_phenotype="stem_count")

    assert "error" in refused
    assert str(plant_csv) in refused["error"]
    assert "was rewritten since it was registered" in refused["error"]
    assert "register the current file under a new registry name" in refused["error"]
    assert not out_csv.exists()


def test_deliver_orthomosaic_plant_counts_refuses_a_registered_csv_deleted_after_registration(
    tmp_path, monkeypatch,
):
    """The same registry byte check refuses by name when a registered plant CSV is missing
    entirely, a separate outcome from a rewritten one (verify_registry_csv_bytes' own missing
    list, not its rewritten-file fact)."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    bucket_dir, stem = _run_bucket(tmp_path, monkeypatch, raster_path)
    _replace_boxes(bucket_dir / f"{stem}.json", [(8.0, 8.0, 12.0, 12.0)])
    _promote_bucket_conf(bucket_dir, bucket_dir.parents[1], trait=fx.COUNT_TRAIT)
    plant_csv = _plant_grid_csv(tmp_path, raster_path, _PLANT_PIXELS)
    registry_name = _plant_registry(plant_csv)

    plant_csv.unlink()

    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    out_csv = tmp_path / "counts.csv"
    refused = deliver_orthomosaic_plant_counts(
        str(bucket_dir), str(raster_path), registry_name, str(out_csv),
        delivered_phenotype="stem_count")

    assert "error" in refused
    assert plant_csv.name in refused["error"]
    assert not out_csv.exists()


def test_deliver_orthomosaic_plant_counts_delivers_once_registry_csv_bytes_verify(
    tmp_path, monkeypatch,
):
    """The rail admits valid work: an unmodified registry CSV still delivers, so the byte check
    above refuses only a genuinely rewritten file, never a legitimately unchanged one."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    bucket_dir, stem = _run_bucket(tmp_path, monkeypatch, raster_path)
    _replace_boxes(bucket_dir / f"{stem}.json", [(8.0, 8.0, 12.0, 12.0)])
    _promote_bucket_conf(bucket_dir, bucket_dir.parents[1], trait=fx.COUNT_TRAIT)
    plant_csv = _plant_grid_csv(tmp_path, raster_path, _PLANT_PIXELS)
    registry_name = _plant_registry(plant_csv)

    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    out_csv = tmp_path / "counts.csv"
    delivered = deliver_orthomosaic_plant_counts(
        str(bucket_dir), str(raster_path), registry_name, str(out_csv),
        delivered_phenotype="stem_count")

    assert "error" not in delivered, delivered
    assert out_csv.exists()


# ── nearest-neighbour path: the in-frame correction ───────────────────────


def _plants_csv_at(tmp_path: Path, raster_path: Path, rows: list[tuple[str, float, float]]) -> Path:
    """A registry CSV naming ``rows`` (plot_name, pixel_x, pixel_y) each projected to WGS84
    through this raster's own georeferencing, for a test that needs plants at arbitrary pixel
    positions rather than the shared ``_PLANT_PIXELS`` grid."""
    entries = []
    for i, (name, px, py) in enumerate(rows):
        lat, lon = _pixel_to_wgs84(raster_path, px, py)
        entries.append({
            "plot_name": name, "accession_name": f"acc-{name}", "plot_number": i,
            "row_number": 0, "col_number": i, "WGS84_centroid_y": lat, "WGS84_centroid_x": lon,
        })
    csv_path = tmp_path / f"plants_{len(rows)}.csv"
    _write_plant_csv(csv_path, entries)
    return csv_path


def test_deliver_orthomosaic_plant_counts_names_an_outside_raster_plant_and_leaves_a_near_edge_detection_unmapped(
    tmp_path, monkeypatch,
):
    """A registry plant outside the raster's own frame gets no row and is named
    (plants_outside_raster); a detection at the raster's edge, nearer to that outside plant than
    to any in-frame plant, stays unmapped rather than attributed to it: without the in-frame
    guard, the outside plant would have been the nearest candidate and well inside the fallback
    tolerance, mapping the edge detection to a plant this raster never pictures."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)  # 64x64
    bucket_dir, stem = _run_bucket(tmp_path, monkeypatch, raster_path)

    plant_csv = _plants_csv_at(tmp_path, raster_path, [
        ("plot_in", 10.0, 10.0), ("plot_out", 70.0, 10.0),  # column 70 >= width 64: outside
    ])
    _replace_boxes(bucket_dir / f"{stem}.json", [(61.0, 8.0, 65.0, 12.0)])  # centroid (63, 10)
    _promote_bucket_conf(bucket_dir, bucket_dir.parents[1], trait=fx.COUNT_TRAIT)

    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    out_csv = tmp_path / "counts.csv"
    result = deliver_orthomosaic_plant_counts(
        str(bucket_dir), str(raster_path), _plant_registry(plant_csv), str(out_csv),
        delivered_phenotype="stem_count")

    assert "error" not in result, result
    assert result["n_unmapped"] == 1
    assert result["plants_outside_raster"] == ["plot_out"]

    rows = {r["plant_id"]: r for r in csv.DictReader(out_csv.open(newline=""))}
    assert rows["plot_in"]["value"] == "0"
    assert "plot_out" not in rows


# ── canopy_subject: attribution by segment containment ────────────────────


def _canopy_setup(tmp_path, monkeypatch) -> tuple[Path, Path, Path, str]:
    """A raster and its prediction bucket under one shared, registered dataset root, for the
    canopy_subject regime's own dataset-binding check."""
    from tests._geotiff_fixtures import write_canonical_dataset_raster

    dataset_root = tmp_path / "ds"
    raster_path = write_canonical_dataset_raster(dataset_root, width=64, height=64)
    bucket_dir, stem = _run_bucket(tmp_path, monkeypatch, raster_path)
    return dataset_root, raster_path, bucket_dir, stem


def _write_canopy_document(raster_path: Path, boxes: list[tuple[float, float, float, float]],
                          *, subject: str = "canopy") -> None:
    """A hand-traced canopy boundary document at ``raster_path``'s own canonical label position,
    one rectangle per ``boxes`` entry, under a person's identity."""
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, Polygon
    from tcip_mcp.dataset_layout import annotation_path_for_image

    anns = [
        Annotation(
            subject=subject,
            geometry=Polygon(rings=[[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]),
            created_by="user:breeder", created_at="2024-01-01T00:00:00+00:00",
        )
        for (x0, y0, x1, y1) in boxes
    ]
    doc_path = annotation_path_for_image(raster_path)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    json_io.write_annotations(str(doc_path), anns, 64, 64, keep_empty=True)


def test_deliver_orthomosaic_plant_counts_canopy_subject_delivers_ties_and_names_the_gaps(
    tmp_path, monkeypatch,
):
    dataset_root, raster_path, bucket_dir, stem = _canopy_setup(tmp_path, monkeypatch)
    plant_csv = _plants_csv_at(tmp_path, raster_path, [
        ("plot0", 10.0, 10.0), ("plot1", 10.0, 50.0),
        ("plot2", 50.0, 10.0), ("plot3", 50.0, 50.0),
    ])
    registry_name = _plant_registry(plant_csv)
    _write_canopy_document(raster_path, [
        (5.0, 5.0, 15.0, 15.0),    # segment 0: ties to plot0
        (45.0, 5.0, 55.0, 15.0),   # segment 1: ties to plot2
        (0.0, 55.0, 5.0, 60.0),    # segment 2: no plant inside, untied
    ])
    _replace_boxes(bucket_dir / f"{stem}.json", [
        (8.0, 8.0, 12.0, 12.0),    # inside segment 0: attributed to plot0
        (48.0, 8.0, 52.0, 12.0),   # inside segment 1: attributed to plot2
        (1.0, 56.0, 3.0, 58.0),    # inside segment 2: segment_without_plant
        (30.0, 30.0, 32.0, 32.0),  # inside no segment: outside_segments
    ])
    _promote_bucket_conf(bucket_dir, dataset_root, trait=fx.COUNT_TRAIT)

    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    out_csv = tmp_path / "counts.csv"
    result = deliver_orthomosaic_plant_counts(
        str(bucket_dir), str(raster_path), registry_name, str(out_csv),
        delivered_phenotype="stem_count", canopy_subject="canopy")

    assert "error" not in result, result
    assert result["n_segments"] == 3
    assert result["n_segments_without_plant"] == 1
    assert sorted(result["plants_without_segment"]) == ["plot1", "plot3"]
    assert result["plants_with_ambiguous_detections"] == []
    assert result["plants_outside_raster"] == []
    assert result["n_mapped"] == 2
    assert result["n_unmapped"] == 2

    rows = {r["plant_id"]: r for r in csv.DictReader(out_csv.open(newline=""))}
    assert rows["plot0"]["value"] == "1"
    assert rows["plot2"]["value"] == "1"
    assert "plot1" not in rows
    assert "plot3" not in rows
    for r in rows.values():
        assert r["plant_attribution"] == "segment"
        assert r["plant_id_source"] == "segment_containment"
        assert r["plant_id_distance_m_max"] == ""

    from tcip_mcp.pipelines.resolution import read_delivery_events

    events = read_delivery_events(tmp_path)
    canopy_events = [e for e in events if "canopy_segments" in (e["plant_mapping"] or {})]
    assert len(canopy_events) == 1
    pm = canopy_events[0]["plant_mapping"]
    assert pm["canopy_segments"]["n_segments"] == 3
    assert pm["canopy_segments"]["subject"] == "canopy"
    assert {t["plot_name"] for t in pm["segment_ties"]} == {"plot0", "plot2"}
    assert all(t["clearance_m"] > 0 for t in pm["segment_ties"])
    assert pm["plant_attribution"] == "segment"


def test_deliver_orthomosaic_plant_counts_canopy_subject_drops_an_ambiguous_overlap(
    tmp_path, monkeypatch,
):
    """Two overlapping segments with a detection in the overlap: unattributed under
    overlapping_segments, both implicated plants absent and named, a third unimplicated plant
    delivered."""
    dataset_root, raster_path, bucket_dir, stem = _canopy_setup(tmp_path, monkeypatch)
    plant_csv = _plants_csv_at(tmp_path, raster_path, [
        ("plot0", 10.0, 10.0), ("plot1", 25.0, 10.0), ("plot2", 50.0, 50.0),
    ])
    registry_name = _plant_registry(plant_csv)
    _write_canopy_document(raster_path, [
        (0.0, 0.0, 20.0, 20.0),    # segment 0: ties to plot0, overlaps segment 1 on x in [15,20]
        (15.0, 0.0, 35.0, 20.0),   # segment 1: ties to plot1
        (45.0, 45.0, 55.0, 55.0),  # segment 2: ties to plot2, unimplicated
    ])
    _replace_boxes(bucket_dir / f"{stem}.json", [
        (15.0, 8.0, 19.0, 12.0),   # centroid (17, 10): inside both segment 0 and segment 1
        (48.0, 48.0, 52.0, 52.0),  # inside segment 2 alone: attributed to plot2
    ])
    _promote_bucket_conf(bucket_dir, dataset_root, trait=fx.COUNT_TRAIT)

    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    out_csv = tmp_path / "counts.csv"
    result = deliver_orthomosaic_plant_counts(
        str(bucket_dir), str(raster_path), registry_name, str(out_csv),
        delivered_phenotype="stem_count", canopy_subject="canopy")

    assert "error" not in result, result
    assert sorted(result["plants_with_ambiguous_detections"]) == ["plot0", "plot1"]

    rows = {r["plant_id"]: r for r in csv.DictReader(out_csv.open(newline=""))}
    assert "plot0" not in rows
    assert "plot1" not in rows
    assert rows["plot2"]["value"] == "1"


def test_deliver_orthomosaic_plant_counts_refuses_nn_tolerance_m_beside_canopy_subject(
    tmp_path, monkeypatch,
):
    dataset_root, raster_path, bucket_dir, stem = _canopy_setup(tmp_path, monkeypatch)
    plant_csv = _plants_csv_at(tmp_path, raster_path, [("plot0", 10.0, 10.0)])
    registry_name = _plant_registry(plant_csv)
    _write_canopy_document(raster_path, [(5.0, 5.0, 15.0, 15.0)])

    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    result = deliver_orthomosaic_plant_counts(
        str(bucket_dir), str(raster_path), registry_name, str(tmp_path / "counts.csv"),
        delivered_phenotype="stem_count", canopy_subject="canopy", nn_tolerance_m=5.0)

    assert "error" in result
    assert "nn_tolerance_m" in result["error"]
    assert not (tmp_path / "counts.csv").exists()


def test_deliver_orthomosaic_plant_counts_canopy_subject_refuses_a_raster_outside_a_registered_dataset(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    bucket_dir, stem = _run_bucket(tmp_path, monkeypatch, raster_path)
    plant_csv = _plants_csv_at(tmp_path, raster_path, [("plot0", 10.0, 10.0)])
    registry_name = _plant_registry(plant_csv)

    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    result = deliver_orthomosaic_plant_counts(
        str(bucket_dir), str(raster_path), registry_name, str(tmp_path / "counts.csv"),
        delivered_phenotype="stem_count", canopy_subject="canopy")

    assert "error" in result
    assert "registered dataset" in result["error"]


def test_deliver_orthomosaic_plant_counts_canopy_subject_refuses_a_raster_under_a_different_dataset_than_the_bucket(
    tmp_path, monkeypatch,
):
    from tests._geotiff_fixtures import write_canonical_dataset_raster

    dataset_root, raster_path, bucket_dir, stem = _canopy_setup(tmp_path, monkeypatch)
    other_dataset_root = tmp_path / "other_ds"
    other_raster_path = write_canonical_dataset_raster(other_dataset_root, width=64, height=64)
    plant_csv = _plants_csv_at(tmp_path, raster_path, [("plot0", 10.0, 10.0)])
    registry_name = _plant_registry(plant_csv)

    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    result = deliver_orthomosaic_plant_counts(
        str(bucket_dir), str(other_raster_path), registry_name, str(tmp_path / "counts.csv"),
        delivered_phenotype="stem_count", canopy_subject="canopy")

    assert "error" in result
    assert "different" in result["error"]


def test_deliver_orthomosaic_plant_counts_canopy_subject_refuses_a_missing_document(
    tmp_path, monkeypatch,
):
    dataset_root, raster_path, bucket_dir, stem = _canopy_setup(tmp_path, monkeypatch)
    plant_csv = _plants_csv_at(tmp_path, raster_path, [("plot0", 10.0, 10.0)])
    registry_name = _plant_registry(plant_csv)

    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    result = deliver_orthomosaic_plant_counts(
        str(bucket_dir), str(raster_path), registry_name, str(tmp_path / "counts.csv"),
        delivered_phenotype="stem_count", canopy_subject="canopy")

    assert "error" in result
    assert "no label document" in result["error"]


def test_deliver_orthomosaic_plant_counts_canopy_subject_refuses_a_raster_one_level_deeper_than_canonical(
    tmp_path, monkeypatch,
):
    """A raster resolves under its registered dataset (dataset_root_of only checks for an
    images/, predictions/, annotations/, or labels/ segment somewhere in the path) but does not
    sit at either canonical position parse_image_path recognizes
    (<root>/images/<date>/<stem> or <root>/images/<stem>); the door refuses by name instead of
    raising an uncaught ValueError out of annotation_path_for_image."""
    from tcip_mcp.tools.project_tools import register_dataset

    dataset_root = tmp_path / "ds"
    raster_path = dataset_root / "images" / "extra" / "2024-06-01" / "mosaic.tif"
    raster_path.parent.mkdir(parents=True, exist_ok=True)
    _write_geo_raster(raster_path)
    reg = register_dataset(str(dataset_root), crop="chestnut", project_root=str(dataset_root))
    assert "error" not in reg, reg

    bucket_dir, stem = _run_bucket(tmp_path, monkeypatch, raster_path)
    plant_csv = _plants_csv_at(tmp_path, raster_path, [("plot0", 10.0, 10.0)])
    registry_name = _plant_registry(plant_csv)

    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    out_csv = tmp_path / "counts.csv"
    result = deliver_orthomosaic_plant_counts(
        str(bucket_dir), str(raster_path), registry_name, str(out_csv),
        delivered_phenotype="stem_count", canopy_subject="canopy")

    assert "error" in result
    assert "not under a recognized dataset image tree" in result["error"]
    assert not out_csv.exists()


def test_deliver_orthomosaic_plant_counts_canopy_subject_refuses_when_no_plant_would_receive_a_row(
    tmp_path, monkeypatch,
):
    """Every tied segment's own detection is ambiguous, so no plant would receive a row: the
    door refuses before export_aggregated_csv's own generic 'results is empty' refusal, naming
    which plants were dropped and why, rather than reaching that refusal's unnamed message."""
    dataset_root, raster_path, bucket_dir, stem = _canopy_setup(tmp_path, monkeypatch)
    plant_csv = _plants_csv_at(tmp_path, raster_path, [
        ("plot0", 10.0, 10.0), ("plot1", 25.0, 10.0),
    ])
    registry_name = _plant_registry(plant_csv)
    _write_canopy_document(raster_path, [
        (0.0, 0.0, 20.0, 20.0),    # segment 0: ties to plot0, overlaps segment 1
        (15.0, 0.0, 35.0, 20.0),   # segment 1: ties to plot1
    ])
    _replace_boxes(bucket_dir / f"{stem}.json", [
        (15.0, 8.0, 19.0, 12.0),   # centroid (17, 10): inside both segments, ambiguous
    ])

    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    out_csv = tmp_path / "counts.csv"
    result = deliver_orthomosaic_plant_counts(
        str(bucket_dir), str(raster_path), registry_name, str(out_csv),
        delivered_phenotype="stem_count", canopy_subject="canopy")

    assert "error" in result
    assert "plot0" in result["error"] and "plot1" in result["error"]
    assert not out_csv.exists()


def test_deliver_orthomosaic_plant_counts_refuses_a_duplicate_plot_name_in_the_registry(
    tmp_path, monkeypatch,
):
    """The nearest-neighbour regime refuses a duplicated plot_name by name too, through the same
    check the canopy regime already runs over its own registry (require_named_plants): two rows
    sharing one identity would otherwise merge two trees' detections into one aggregation row."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    bucket_dir, stem = _run_bucket(tmp_path, monkeypatch, raster_path)
    plant_csv = _plants_csv_at(tmp_path, raster_path, [
        ("plotA", 10.0, 10.0), ("plotA", 50.0, 50.0),
    ])
    registry_name = _plant_registry(plant_csv)

    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    out_csv = tmp_path / "counts.csv"
    result = deliver_orthomosaic_plant_counts(
        str(bucket_dir), str(raster_path), registry_name, str(out_csv),
        delivered_phenotype="stem_count")

    assert "error" in result
    assert "duplicate plot_name" in result["error"]
    assert not out_csv.exists()


def test_deliver_orthomosaic_plant_counts_refuses_a_blank_plot_name_in_the_registry(
    tmp_path, monkeypatch,
):
    """Same guard, a blank identity instead of a duplicated one."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    bucket_dir, stem = _run_bucket(tmp_path, monkeypatch, raster_path)
    plant_csv = _plants_csv_at(tmp_path, raster_path, [("", 10.0, 10.0)])
    registry_name = _plant_registry(plant_csv)

    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    out_csv = tmp_path / "counts.csv"
    result = deliver_orthomosaic_plant_counts(
        str(bucket_dir), str(raster_path), registry_name, str(out_csv),
        delivered_phenotype="stem_count")

    assert "error" in result
    assert "blank plot_name" in result["error"]
    assert not out_csv.exists()
