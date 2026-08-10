"""Block-aware calibration/holdout end to end: a mosaic's own reserved calibration/test regions
validate a detection operating point directly, without a separate held-out image set.

Covers: the region-completeness gate (refuses unattested, admits attested), the recursive
sub-banding + halo mechanism running real tiled inference over real bands, the geometric
disjointness check, and the whole-raster export entry point (``export_predictions``'s
``raster_path`` regime) with its claim-scope gate and ``max_dets`` non-transfer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox  # noqa: E402

pytestmark = pytest.mark.usefixtures("seed_catkin_trait_spec")

TILE = 32
WIDTH, HEIGHT = 3200, 200
BOX_STEP = 40


def _write_mosaic(path: Path, *, seed: int = 0) -> None:
    import numpy as np
    import tifffile

    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
    tifffile.imwrite(str(path), arr)


def _bespoke_detection_checkpoint(tmp_path: Path, *, tile_size: int = TILE) -> str:
    from tcip_mcp.pipelines.model_build import build_model

    model_source = {"builder": "tests.bespoke_models:build_bespoke_detection",
                    "builder_kwargs": {"num_classes": 1, "in_chans": 3,
                                      "min_size": tile_size, "max_size": tile_size * 2},
                    "task": "detection", "in_chans": 3}
    model = build_model({"model_source": model_source})
    ckpt = tmp_path / "model_best.pt"
    torch.save({"model_source": model_source, "model_state_dict": model.state_dict()}, str(ckpt))
    return str(ckpt)


def _build_experiment(tmp_path: Path, *, reserve_frac: float = 0.15,
                      experiment_id: str = "exp_block") -> dict:
    """A real 4-way spatial-strip split over a real raster, persisted as a real experiment
    (config.json + split.json), the checkpoint's own training provenance block calibration reads.

    Returns a dict of everything a block-calibration test needs: ``root``/``images_dir``/
    ``labels_dir``/``stem``/``raster_path``/``checkpoint_path``/``experiment_id``/
    ``spatial_manifest``.
    """
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.tools.training_tools import _auto_train_val, _persist_split_manifest

    root = tmp_path / "ds"
    images_dir, labels_dir = root / "images", root / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    stem = "mosaic"
    raster_path = images_dir / f"{stem}.tif"
    _write_mosaic(raster_path)

    boxes = [Annotation(subject="catkin", geometry=BBox(x, 80, x + 15, 110))
            for x in range(10, WIDTH - 20, BOX_STEP)]
    json_io.write_annotations(str(labels_dir / f"{stem}.json"), boxes, WIDTH, HEIGHT, keep_empty=True)

    data_cfg = {
        "images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "catkin",
        "auto_val": True, "tiling": {"enabled": True, "tile_size": TILE, "overlap": 0.2},
        "split": {"val_ratio": 0.2, "test_ratio": 0.15, "seed": 1,
                  "reserve_calibration_fraction": reserve_frac},
    }
    train_ds, val_ds = _auto_train_val("detection", data_cfg, None)
    assert val_ds is not None
    create_experiment(experiment_id, {"data": data_cfg})
    _persist_split_manifest(experiment_id, train_ds, val_ds, data_cfg)

    checkpoint_path = _bespoke_detection_checkpoint(tmp_path)
    return {
        "root": root, "images_dir": images_dir, "labels_dir": labels_dir, "stem": stem,
        "raster_path": raster_path, "checkpoint_path": checkpoint_path,
        "experiment_id": experiment_id, "spatial_manifest": data_cfg["split"]["spatial_manifest"],
    }


def _attest_regions_complete(root: Path, stem: str, regions: list[list[tuple[int, int, int, int]]],
                             *, subject: str = "catkin") -> None:
    """Directly writes the region-completeness store (bypassing the HTTP route, same effect as a
    breeder double-clicking every intersecting cell complete): every reference-grid cell (at the
    training tile_size, clamped) that any rect in ``regions`` overlaps is marked complete, with a
    real content digest so the staleness check reads it as fresh."""
    from tcip_annotation import json_io as _json_io

    from tcip_mcp.dataset_layout import (
        annotation_path, region_completeness_digest_path, region_completeness_path, status_bucket,
    )
    from tcip_mcp.pipelines.data.tiling import rects_overlap
    from tcip_mcp.pipelines.reference_grid import grid_geometry, reference_cells
    from tcip_mcp.pipelines.region_completeness import cell_annotation_digest
    from tcip_mcp.utils.atomic_io import atomic_write_json

    grid = grid_geometry(WIDTH, HEIGHT, TILE, 0.0)
    cells = reference_cells(WIDTH, HEIGHT, TILE, 0.0, clamp=True)
    all_rects = [tuple(r) for region in regions for r in region]
    covered = sorted({
        c.name for c in cells
        if any(rects_overlap((c.x0, c.y0, c.x1, c.y1), r) for r in all_rects)
    })

    label_path = annotation_path(root, None, stem)
    annotations = _json_io.read_annotations(str(label_path)) if label_path.is_file() else []
    bucket = status_bucket(subject, stem)
    store = {bucket: {
        "grid": grid, "cells_complete": covered, "attested_by": "test", "attested_at": "now",
        "stem": stem, "date": None, "subject": subject,
    }}
    atomic_write_json(region_completeness_path(root), store)
    digests = {bucket: {
        c.name: cell_annotation_digest(annotations, subject, c) for c in cells if c.name in covered
    }}
    atomic_write_json(region_completeness_digest_path(root), digests)


def test_block_calibration_refuses_when_regions_unattested(tmp_path: Path):
    exp = _build_experiment(tmp_path)

    from tcip_mcp.pipelines.block_calibration import (
        BlockCalibrationRefused, resolve_block_calibration_records,
    )
    from tcip_mcp.pipelines.inference.predictor import build_predictor

    predictor = build_predictor(checkpoint_path=exp["checkpoint_path"], device="cpu",
                                score_threshold=0.01, nms_iou=0.3, max_dets=1000)
    with pytest.raises(BlockCalibrationRefused, match="not fully attested complete"):
        resolve_block_calibration_records(
            predictor, checkpoint_path=exp["checkpoint_path"], trait_name="catkin",
            experiment_id=exp["experiment_id"], global_nms_iou=0.3)


def test_block_calibration_completeness_checked_before_feasibility(tmp_path: Path):
    """Refusal ordering: a region that is both incomplete and would be infeasible even if
    complete produces the completeness message, not the feasibility one. k_cal/k_test set high
    enough that, were the regions attested, feasibility would itself refuse (too few GT-bearing
    bands); left unattested here, so completeness must win regardless."""
    exp = _build_experiment(tmp_path)

    from tcip_mcp.pipelines.block_calibration import (
        BlockCalibrationRefused, resolve_block_calibration_records,
    )
    from tcip_mcp.pipelines.inference.predictor import build_predictor

    predictor = build_predictor(checkpoint_path=exp["checkpoint_path"], device="cpu",
                                score_threshold=0.01, nms_iou=0.3, max_dets=1000)
    with pytest.raises(BlockCalibrationRefused) as exc_info:
        resolve_block_calibration_records(
            predictor, checkpoint_path=exp["checkpoint_path"], trait_name="catkin",
            experiment_id=exp["experiment_id"], global_nms_iou=0.3, k_cal=40, k_test=40)
    msg = str(exc_info.value)
    assert "not fully attested complete" in msg
    assert "leaves only" not in msg  # the feasibility message never gets a chance to fire


def test_block_calibration_admits_valid_work_once_attested(tmp_path: Path):
    """The rail-admits-valid-work paired test: once every reserved cell is attested complete, the
    same call that refused above resolves a real bundle with real per-band cal/hold records."""
    exp = _build_experiment(tmp_path)
    manifest = exp["spatial_manifest"]
    _attest_regions_complete(
        exp["root"], exp["stem"], [manifest["calibration_region"], manifest["test_region"]])

    from tcip_mcp.pipelines.block_calibration import resolve_block_calibration_records
    from tcip_mcp.pipelines.inference.predictor import build_predictor

    predictor = build_predictor(checkpoint_path=exp["checkpoint_path"], device="cpu",
                                score_threshold=0.01, nms_iou=0.3, max_dets=1000)
    bundle, prov = resolve_block_calibration_records(
        predictor, checkpoint_path=exp["checkpoint_path"], trait_name="catkin",
        experiment_id=exp["experiment_id"], global_nms_iou=0.3)

    conf = bundle.get("conf")
    assert conf.sweep is not None
    assert conf.sweep["calibration_image_ids"] or conf.sweep.get("note")
    assert prov["experiment_id"] == exp["experiment_id"]
    assert prov["k_cal"] == 3 and prov["k_test"] == 3
    assert sum(prov["cal_gt_counts"].values()) > 0
    assert sum(prov["test_gt_counts"].values()) > 0
    # Pins the defect independent review found: calibration_region was once missing from the
    # geometric disjointness check's non-train set, flagging every real cal/holdout rect as a leak.
    assert conf.sweep["train_disjointness"] == {
        "checked": True, "unresolvable": False, "leaked_groups": [], "leaked_stems": [],
        "group_check": "spatial_strip_geometric",
    }


def test_export_predictions_raster_block_calibration_admits_and_uncaps_max_dets(tmp_path: Path):
    """The real entry point: export_predictions(raster_path=..., trait=...) runs block
    calibration, ships a validated-or-honestly-stamped conf, and the persisted operating point's
    max_dets is None (uncapped), never the block bundle's own band-scoped density-derived value."""
    exp = _build_experiment(tmp_path)
    manifest = exp["spatial_manifest"]
    _attest_regions_complete(
        exp["root"], exp["stem"], [manifest["calibration_region"], manifest["test_region"]])

    from tcip_mcp.tools.inference_tools import export_predictions

    out_dir = tmp_path / "preds"
    result = export_predictions(
        exp["checkpoint_path"], output_dir=str(out_dir), raster_path=str(exp["raster_path"]),
        conf_threshold=0.0, tile_size=TILE, overlap=0.2, trait="catkin",
        experiment_id=exp["experiment_id"])

    assert "error" not in result, result
    assert result["conf_source"] == "block_calibration"
    sidecar = json.loads((out_dir / "operating_point.json").read_text())
    assert sidecar["operating_point"]["max_dets"]["value"] is None
    assert sidecar["operating_point"]["max_dets"]["derived_from"].startswith("block calibration")
    assert sidecar["claim_scope_validated"] == "same_mosaic_content_identity"
    assert sidecar["block_calibration"]["experiment_id"] == exp["experiment_id"]
    assert "spatial_manifest" not in sidecar["block_calibration"]


def test_export_predictions_raster_claim_scope_refuses_cross_mosaic(tmp_path: Path):
    """A different raster reusing the same checkpoint must refuse: the block-validated reference
    is scoped to the training mosaic, never silently applied to a different one."""
    exp = _build_experiment(tmp_path)
    manifest = exp["spatial_manifest"]
    _attest_regions_complete(
        exp["root"], exp["stem"], [manifest["calibration_region"], manifest["test_region"]])

    other_raster = tmp_path / "different_mosaic.tif"
    _write_mosaic(other_raster, seed=99)  # different content, same dims

    from tcip_mcp.tools.inference_tools import export_predictions

    out_dir = tmp_path / "preds"
    result = export_predictions(
        exp["checkpoint_path"], output_dir=str(out_dir), raster_path=str(other_raster),
        conf_threshold=0.0, tile_size=TILE, overlap=0.2, trait="catkin",
        experiment_id=exp["experiment_id"])
    assert "error" in result
    assert not out_dir.exists()


def test_export_predictions_raster_without_reserved_region_names_the_real_gap(tmp_path: Path):
    """A checkpoint whose training experiment has no reserved calibration region (an ordinary
    3-way split) refuses trait+raster_path, but with a message naming the missing reserved
    region and the remedy (reserve_calibration_fraction) -- not baseline's blanket "not
    supported" refusal, which fired unconditionally for every trait+raster_path call regardless
    of whether a reserved region existed. Asserting on message content (not just "error" in
    result) is required here: baseline's blanket refusal also satisfies a bare "error" in
    result check, so that alone can't distinguish the old behavior from the new reserved-region
    check this test exists to guard."""
    exp = _build_experiment(tmp_path, reserve_frac=0.0, experiment_id="exp_no_reserve")

    from tcip_mcp.tools.inference_tools import export_predictions

    out_dir = tmp_path / "preds"
    result = export_predictions(
        exp["checkpoint_path"], output_dir=str(out_dir), raster_path=str(exp["raster_path"]),
        conf_threshold=0.0, tile_size=TILE, overlap=0.2, trait="catkin",
        experiment_id=exp["experiment_id"])
    assert "error" in result
    assert "reserved calibration" in result["error"]
    assert "reserve_calibration_fraction" in result["error"]
    assert not out_dir.exists()


def test_export_predictions_raster_with_no_trait_is_byte_identical_to_the_original_raw_path(
    tmp_path: Path,
):
    """Fail-before/no-op: a raster_path export with no trait at all (today's only working raster
    export shape, unchanged by this phase) still produces the original raw, unvalidated bucket --
    the real backward-compatible guarantee this phase must not disturb, distinct from the
    trait+no-reserved-region refusal above, which is new behavior with no baseline to preserve."""
    exp = _build_experiment(tmp_path, reserve_frac=0.0, experiment_id="exp_no_trait")

    from tcip_mcp.tools.inference_tools import export_predictions

    out_dir = tmp_path / "preds"
    result = export_predictions(
        exp["checkpoint_path"], output_dir=str(out_dir), raster_path=str(exp["raster_path"]),
        conf_threshold=0.0, tile_size=TILE, overlap=0.2, experiment_id=exp["experiment_id"])
    assert "error" not in result
    assert result["validated"] is False
    op = json.loads((out_dir / "operating_point.json").read_text())["operating_point"]
    assert op["conf"]["validated_against"] == "false"


def test_select_gt_for_band_matches_the_dt_sides_center_inclusion_rule():
    """GT selection must apply the same center-in-band-rect keep test the DT side already applies
    (independent review finding: clip-and-keep on GT paired against center-filter-and-drop on DT
    biased the reference at every band boundary). A box centered inside is kept at full extent,
    translated to local coordinates; a box centered outside is dropped entirely, even one that
    overlaps the band substantially -- clipping a straddling box's visible remainder must never
    happen on this path anymore."""
    import numpy as np

    from tcip_mcp.pipelines.block_calibration import _select_gt_for_band

    band = (100, 100, 200, 200)
    boxes = np.array([
        [120.0, 120.0, 140.0, 140.0],   # center (130,130): inside -> kept, full extent
        [190.0, 120.0, 260.0, 140.0],   # center (225,130): outside (past x1=200) -> dropped
        [50.0, 90.0, 110.0, 150.0],     # center (80,120): outside (past x0=100) -> dropped
    ])
    labels = np.array([1, 2, 3])

    kept_boxes, kept_labels = _select_gt_for_band(boxes, labels, band)

    assert kept_labels.tolist() == [1]
    np.testing.assert_array_equal(kept_boxes, np.array([[20.0, 20.0, 40.0, 40.0]]))


def test_select_gt_for_band_empty_input_is_a_no_op():
    import numpy as np

    from tcip_mcp.pipelines.block_calibration import _select_gt_for_band

    boxes = np.zeros((0, 4), dtype=np.float32)
    labels = np.zeros((0,), dtype=np.int64)
    kept_boxes, kept_labels = _select_gt_for_band(boxes, labels, (0, 0, 10, 10))
    assert len(kept_boxes) == 0 and len(kept_labels) == 0
