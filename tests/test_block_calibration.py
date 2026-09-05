"""Block-aware calibration/holdout end to end: a mosaic's own reserved calibration/test regions
validate a detection operating point directly, without a separate held-out image set.

Covers: the region-completeness gate (refuses unattested, admits attested), the recursive
sub-banding + halo mechanism running real tiled inference over real bands, the geometric
disjointness check, and the whole-raster export entry point (``run_inference``'s
``raster_path`` regime) with its claim-scope gate and ``max_dets`` non-transfer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox  # noqa: E402

pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")

TILE = 32
WIDTH, HEIGHT = 3200, 200
BOX_STEP = 40


def _write_mosaic(path: Path, *, seed: int = 0, georeferenced: bool = False,
                  tiepoint_x: float = 500_000.0) -> None:
    import numpy as np
    import tifffile

    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
    if not georeferenced:
        tifffile.imwrite(str(path), arr)
        return
    geokeys = (1, 1, 0, 2, 1024, 0, 1, 1, 3072, 0, 1, 32615)  # UTM zone 15N
    tifffile.imwrite(
        str(path), arr, photometric="rgb",
        extratags=[
            (33550, "d", 3, (1.0, 1.0, 0.0), False),
            (33922, "d", 6, (0.0, 0.0, 0.0, tiepoint_x, 4_800_000.0, 0.0), False),
            (34735, "H", len(geokeys), geokeys, False),
        ],
    )


def _write_plant_csv(path: Path) -> None:
    import csv

    # Two plants ~10m apart, independent of the mosaic's own pixel geometry: the plant path only
    # needs the raster's real geotransform to convert this real-world spacing to pixels.
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["plot_name", "accession_name", "WGS84_centroid_x", "WGS84_centroid_y"])
        w.writerow(["P1", "acc-A", -93.0, 45.0])
        w.writerow(["P2", "acc-B", -93.0001, 45.0])


def _bespoke_detection_checkpoint(tmp_path: Path, *, tile_size: int = TILE) -> str:
    from tcip_mcp.pipelines.model_build import build_model
    from tcip_mcp.tools.model_tools import register_model

    model_source = {"builder": "tests.bespoke_models:build_bespoke_detection",
                    "builder_kwargs": {"num_classes": 1, "in_chans": 3,
                                      "min_size": tile_size, "max_size": tile_size * 2},
                    "task": "detection", "in_chans": 3}
    model = build_model({"model_source": model_source})
    ckpt = tmp_path / "model_best.pt"
    torch.save({"model_source": model_source, "model_state_dict": model.state_dict()}, str(ckpt))
    result = register_model(name="block-calibration-bespoke", checkpoint_path=str(ckpt),
                            config={}, project_path=str(tmp_path))
    assert "error" not in result, result
    return str(ckpt)


def _build_experiment(tmp_path: Path, *, reserve_frac: float = 0.15,
                      experiment_id: str = "exp_block",
                      plant_csv_paths: list[str] | None = None) -> dict:
    """A real 4-way spatial-strip split over a real raster, persisted as a real experiment
    (config.json + split.json), the checkpoint's own training provenance block calibration reads.

    ``plant_csv_paths`` (when given) writes a georeferenced mosaic (the plant-pitch derivation
    needs a real geotransform to convert real-world plant spacing to pixels) and threads the
    paths into ``data.plant_csv_paths``, the config field ``resolve_block_calibration_records``
    reads to prefer plant-pitch over GT-object-spacing.

    Returns a dict of everything a block-calibration test needs: ``root``/``images_dir``/
    ``labels_dir``/``stem``/``raster_path``/``checkpoint_path``/``experiment_id``/
    ``spatial_manifest``.
    """
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.pipelines.data.split_construction import auto_train_val, persist_split_manifest

    root = tmp_path / "ds"
    images_dir, labels_dir = root / "images", root / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    stem = "mosaic"
    raster_path = images_dir / f"{stem}.tif"
    _write_mosaic(raster_path, georeferenced=bool(plant_csv_paths))

    boxes = [Annotation(subject="bud", geometry=BBox(x, 80, x + 15, 110))
            for x in range(10, WIDTH - 20, BOX_STEP)]
    json_io.write_annotations(str(labels_dir / f"{stem}.json"), boxes, WIDTH, HEIGHT, keep_empty=True)

    data_cfg = {
        "images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud",
        "auto_val": True, "tiling": {"enabled": True, "tile_size": TILE, "overlap": 0.2},
        "split": {"val_ratio": 0.2, "test_ratio": 0.15, "seed": 1,
                  "reserve_calibration_fraction": reserve_frac},
    }
    if plant_csv_paths:
        data_cfg["plant_csv_paths"] = plant_csv_paths
    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)
    assert val_ds is not None
    create_experiment(experiment_id, {"data": data_cfg})
    persist_split_manifest(experiment_id, train_ds, val_ds, data_cfg)

    checkpoint_path = _bespoke_detection_checkpoint(tmp_path)
    return {
        "root": root, "images_dir": images_dir, "labels_dir": labels_dir, "stem": stem,
        "raster_path": raster_path, "checkpoint_path": checkpoint_path,
        "experiment_id": experiment_id, "spatial_manifest": data_cfg["split"]["spatial_manifest"],
    }


def _attest_regions_complete(root: Path, stem: str, regions: list[list[tuple[int, int, int, int]]],
                             *, subject: str = "bud") -> None:
    """Directly writes the region-completeness store (bypassing the HTTP route, same effect as a
    breeder attesting every intersecting cell complete): every reference-grid cell (at the
    training tile_size, clamped) that any rect in ``regions`` overlaps is marked complete, with a
    real content digest so the staleness check reads it as fresh."""
    from tcip_annotation import json_io as _json_io

    from tcip_mcp.dataset_layout import (
        annotation_path, region_completeness_digest_key, region_completeness_key, status_bucket,
    )
    from tcip_mcp.pipelines.data.tiling import rects_overlap
    from tcip_mcp.pipelines.reference_grid import grid_geometry, reference_cells
    from tcip_mcp.pipelines.region_completeness import cell_annotation_digest
    from tcip_store import transaction

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
        "stem": stem, "date": None, "subject": subject, "cells_attested_view": {},
    }}
    digests = {bucket: {
        c.name: cell_annotation_digest(annotations, subject, c) for c in cells if c.name in covered
    }}
    # Digest first, the order the coverage route commits them in: an attestation with no digest
    # beside it reads as stale.
    digest_key, completeness_key = region_completeness_digest_key(root), region_completeness_key(root)
    with transaction(digest_key, completeness_key) as txn:
        txn.write(digest_key, digests)
        txn.write(completeness_key, store)


def test_block_calibration_refuses_when_regions_unattested(tmp_path: Path):
    exp = _build_experiment(tmp_path)

    from tcip_mcp.pipelines.block_calibration import (
        BlockCalibrationRefused, resolve_block_calibration_records,
    )
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.predictor import build_predictor

    checkpoint = load_registered_checkpoint(exp["checkpoint_path"], project_path=str(tmp_path))
    predictor = build_predictor(checkpoint, device="cpu",
                                score_threshold=0.01, nms_iou=0.3, max_dets=1000)
    with pytest.raises(BlockCalibrationRefused, match="not fully attested complete"):
        resolve_block_calibration_records(
            predictor, trait_name="bud_opening",
            experiment_id=exp["experiment_id"], global_nms_iou=0.3, export_tile_size=TILE)


def test_block_calibration_completeness_checked_before_feasibility(tmp_path: Path):
    """Refusal ordering: a region that is both incomplete and would be infeasible even if
    complete produces the completeness message, not the feasibility one. k_cal/k_test set high
    enough that, were the regions attested, feasibility would itself refuse (too few GT-bearing
    bands); left unattested here, so completeness must win regardless."""
    exp = _build_experiment(tmp_path)

    from tcip_mcp.pipelines.block_calibration import (
        BlockCalibrationRefused, resolve_block_calibration_records,
    )
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.predictor import build_predictor

    checkpoint = load_registered_checkpoint(exp["checkpoint_path"], project_path=str(tmp_path))
    predictor = build_predictor(checkpoint, device="cpu",
                                score_threshold=0.01, nms_iou=0.3, max_dets=1000)
    with pytest.raises(BlockCalibrationRefused) as exc_info:
        resolve_block_calibration_records(
            predictor, trait_name="bud_opening",
            experiment_id=exp["experiment_id"], global_nms_iou=0.3, k_cal=40, k_test=40, export_tile_size=TILE)
    msg = str(exc_info.value)
    assert "not fully attested complete" in msg
    assert "leaves only" not in msg  # the feasibility message never gets a chance to fire


def test_block_calibration_refuses_when_export_tile_size_differs_from_manifest(tmp_path: Path):
    """The reserved-region claim and the exported bucket must be tiled at one regime: an export
    resolved to a different tile edge than the split manifest's own tile_size refuses, naming
    both, before completeness or feasibility even run."""
    exp = _build_experiment(tmp_path)

    from tcip_mcp.pipelines.block_calibration import (
        BlockCalibrationRefused, resolve_block_calibration_records,
    )
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.predictor import build_predictor

    checkpoint = load_registered_checkpoint(exp["checkpoint_path"], project_path=str(tmp_path))
    predictor = build_predictor(checkpoint, device="cpu",
                                score_threshold=0.01, nms_iou=0.3, max_dets=1000)
    with pytest.raises(BlockCalibrationRefused) as exc_info:
        resolve_block_calibration_records(
            predictor, trait_name="bud_opening",
            experiment_id=exp["experiment_id"], global_nms_iou=0.3, export_tile_size=TILE * 2)
    msg = str(exc_info.value)
    assert f"{TILE}px" in msg and f"{TILE * 2}px" in msg
    assert "not fully attested complete" not in msg  # the tile-size check runs first


def test_block_calibration_admits_valid_work_once_attested(tmp_path: Path):
    """The rail-admits-valid-work paired test: once every reserved cell is attested complete, the
    same call that refused above resolves a real bundle with real per-band cal/hold records."""
    exp = _build_experiment(tmp_path)
    manifest = exp["spatial_manifest"]
    _attest_regions_complete(
        exp["root"], exp["stem"], [manifest["calibration_region"], manifest["test_region"]])

    from tcip_mcp.pipelines.block_calibration import resolve_block_calibration_records
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.predictor import build_predictor

    checkpoint = load_registered_checkpoint(exp["checkpoint_path"], project_path=str(tmp_path))
    predictor = build_predictor(checkpoint, device="cpu",
                                score_threshold=0.01, nms_iou=0.3, max_dets=1000)
    bundle, prov, _evidence = resolve_block_calibration_records(
        predictor, trait_name="bud_opening",
        experiment_id=exp["experiment_id"], global_nms_iou=0.3, export_tile_size=TILE)

    conf = bundle.get("conf")
    assert conf.gate_evidence is not None
    assert conf.gate_evidence["calibration_image_ids"] or conf.gate_evidence.get("note")
    assert prov["experiment_id"] == exp["experiment_id"]
    assert prov["k_cal"] == 3 and prov["k_test"] == 3
    assert sum(prov["cal_gt_counts"].values()) > 0
    assert sum(prov["test_gt_counts"].values()) > 0
    # Pins the defect independent review found: calibration_region was once missing from the
    # geometric disjointness check's non-train set, flagging every real cal/holdout rect as a leak.
    assert conf.gate_evidence["train_disjointness"] == {
        "checked": True, "unresolvable": False, "leaked_groups": [], "leaked_stems": [],
        "group_check": "spatial_strip_geometric",
    }


def test_block_calibration_prefers_plant_pitch_over_gt_spacing_when_configured(tmp_path: Path):
    """A training experiment whose config.json carries data.plant_csv_paths (report 9814, real
    threading of plant-CSV data into block calibration) resolves the block scale from the real
    planting-grid pitch, not the GT-object-spacing fallback -- report 9814's whole point: the
    plant-pitch derivation path existed but had no production caller before this fix."""
    plant_csv = tmp_path / "plants.csv"
    _write_plant_csv(plant_csv)
    exp = _build_experiment(tmp_path, plant_csv_paths=[str(plant_csv)])
    manifest = exp["spatial_manifest"]
    _attest_regions_complete(
        exp["root"], exp["stem"], [manifest["calibration_region"], manifest["test_region"]])

    from tcip_mcp.pipelines.block_calibration import resolve_block_calibration_records
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.predictor import build_predictor

    checkpoint = load_registered_checkpoint(exp["checkpoint_path"], project_path=str(tmp_path))
    predictor = build_predictor(checkpoint, device="cpu",
                                score_threshold=0.01, nms_iou=0.3, max_dets=1000)
    _bundle, prov, _evidence = resolve_block_calibration_records(
        predictor, trait_name="bud_opening",
        experiment_id=exp["experiment_id"], global_nms_iou=0.3, export_tile_size=TILE)

    assert prov["block_scale_source"].startswith("plant grid pitch"), prov["block_scale_source"]


def test_block_calibration_falls_back_to_gt_spacing_with_no_plant_csv_configured(tmp_path: Path):
    """No data.plant_csv_paths (the ordinary case, every other test in this file) still resolves
    the GT-object-spacing fallback exactly as before this feature existed."""
    exp = _build_experiment(tmp_path)
    manifest = exp["spatial_manifest"]
    _attest_regions_complete(
        exp["root"], exp["stem"], [manifest["calibration_region"], manifest["test_region"]])

    from tcip_mcp.pipelines.block_calibration import resolve_block_calibration_records
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.predictor import build_predictor

    checkpoint = load_registered_checkpoint(exp["checkpoint_path"], project_path=str(tmp_path))
    predictor = build_predictor(checkpoint, device="cpu",
                                score_threshold=0.01, nms_iou=0.3, max_dets=1000)
    _bundle, prov, _evidence = resolve_block_calibration_records(
        predictor, trait_name="bud_opening",
        experiment_id=exp["experiment_id"], global_nms_iou=0.3, export_tile_size=TILE)

    assert prov["block_scale_source"].startswith("GT object-spacing"), prov["block_scale_source"]


def test_a_saturated_band_cap_surfaces_as_cap_saturated_frac_provenance(
    tmp_path: Path, monkeypatch,
):
    """A band whose raw detection count exceeds the applied per-image cap is invisible in block
    calibration's own provenance today: ``_band_records`` builds every record via
    ``build_coco_image_record``, which never stamps ``cap_hit``, so
    ``operating_point._cap_saturated_frac`` (already wired into
    ``calibration_cap_saturated_frac``/``holdout_cap_saturated_frac``) always reads ``None`` for a
    block-calibrated bundle regardless of real saturation. Forces the density-derived cap down to
    1 (monkeypatching ``derive_max_dets_from_counts``, since the real cap is derived from GT counts
    this test does not otherwise control) so every band with more than one raw detection is
    genuinely truncated."""
    exp = _build_experiment(tmp_path)
    manifest = exp["spatial_manifest"]
    _attest_regions_complete(
        exp["root"], exp["stem"], [manifest["calibration_region"], manifest["test_region"]])

    import tcip_mcp.pipelines.operating_point as operating_point_module

    monkeypatch.setattr(operating_point_module, "derive_max_dets_from_counts", lambda *a, **k: 1)

    from tcip_mcp.pipelines.block_calibration import resolve_block_calibration_records
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.predictor import build_predictor

    checkpoint = load_registered_checkpoint(exp["checkpoint_path"], project_path=str(tmp_path))
    predictor = build_predictor(checkpoint, device="cpu",
                                score_threshold=0.01, nms_iou=0.3, max_dets=1000)
    bundle, prov, _evidence = resolve_block_calibration_records(
        predictor, trait_name="bud_opening",
        experiment_id=exp["experiment_id"], global_nms_iou=0.3, export_tile_size=TILE)

    conf = bundle.get("conf")
    assert conf.gate_evidence is not None
    cal_frac = conf.gate_evidence.get("calibration_cap_saturated_frac")
    hold_frac = conf.gate_evidence.get("holdout_cap_saturated_frac")
    assert cal_frac is not None and cal_frac > 0.0
    assert hold_frac is not None and hold_frac > 0.0


def _rewrite_split_manifest_dims(root: Path, experiment_id: str, *, width: int, height: int) -> None:
    """Hand-edits a persisted experiment's split manifest ``spatial.width``/``spatial.height``,
    simulating a manifest recorded against a raster that was later replaced or truncated."""
    import tcip_store

    from tcip_mcp.experiments import read_split_manifest, split_key

    split = read_split_manifest(experiment_id)
    split["spatial"]["width"] = width
    split["spatial"]["height"] = height
    tcip_store.replace(split_key(experiment_id), split)


def test_block_calibration_refuses_by_name_when_manifest_dims_exceed_the_real_raster(tmp_path: Path):
    """The split manifest's recorded mosaic dimensions must be cross-checked against the real
    raster's own current dimensions before block calibration trusts the reserved regions'
    geometry: a manifest recording larger-than-real dims must refuse by name
    (``BlockCalibrationRefused``), not let a too-large haloed rect reach ``_RegionView.__init__``'s
    bare ``ValueError``."""
    exp = _build_experiment(tmp_path)
    manifest = exp["spatial_manifest"]
    _attest_regions_complete(
        exp["root"], exp["stem"], [manifest["calibration_region"], manifest["test_region"]])
    _rewrite_split_manifest_dims(
        tmp_path, exp["experiment_id"], width=WIDTH + 500, height=HEIGHT)

    from tcip_mcp.pipelines.block_calibration import (
        BlockCalibrationRefused, resolve_block_calibration_records,
    )
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.predictor import build_predictor

    checkpoint = load_registered_checkpoint(exp["checkpoint_path"], project_path=str(tmp_path))
    predictor = build_predictor(checkpoint, device="cpu",
                                score_threshold=0.01, nms_iou=0.3, max_dets=1000)
    with pytest.raises(BlockCalibrationRefused, match="do not match"):
        resolve_block_calibration_records(
            predictor, trait_name="bud_opening",
            experiment_id=exp["experiment_id"], global_nms_iou=0.3, export_tile_size=TILE)


def test_block_calibration_refuses_by_name_when_manifest_dims_are_smaller_than_the_real_raster(
    tmp_path: Path,
):
    """The inverse mismatch, a manifest recording smaller-than-real dims, must also refuse by
    name, closing the silent-partial-scoring behavior (band rects resolved against a smaller
    mosaic than the one actually being read would silently score only a sub-area of the real
    raster) rather than proceeding as if the reserved regions still described the whole image."""
    exp = _build_experiment(tmp_path)
    manifest = exp["spatial_manifest"]
    _attest_regions_complete(
        exp["root"], exp["stem"], [manifest["calibration_region"], manifest["test_region"]])
    _rewrite_split_manifest_dims(
        tmp_path, exp["experiment_id"], width=WIDTH - 500, height=HEIGHT)

    from tcip_mcp.pipelines.block_calibration import (
        BlockCalibrationRefused, resolve_block_calibration_records,
    )
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.predictor import build_predictor

    checkpoint = load_registered_checkpoint(exp["checkpoint_path"], project_path=str(tmp_path))
    predictor = build_predictor(checkpoint, device="cpu",
                                score_threshold=0.01, nms_iou=0.3, max_dets=1000)
    with pytest.raises(BlockCalibrationRefused, match="do not match"):
        resolve_block_calibration_records(
            predictor, trait_name="bud_opening",
            experiment_id=exp["experiment_id"], global_nms_iou=0.3, export_tile_size=TILE)


def test_max_dets_stamp_reflects_the_pooled_cal_and_test_density_cap(tmp_path: Path):
    """The calibration bundle's own persisted max_dets must equal the real cap applied to the
    model during the band passes (density_cap, pooled across cal+test bands), not
    resolve_operating_point's internal cal-only fallback. The uniform BOX_STEP layout alone makes
    cal-only and pooled cal+test density agree by symmetry, so this inflates GT density inside the
    reserved test region only, to force a genuine divergence between the two derivations."""
    exp = _build_experiment(tmp_path)
    manifest = exp["spatial_manifest"]

    label_path = exp["labels_dir"] / f"{exp['stem']}.json"
    existing = json_io.read_annotations(str(label_path))
    tx0, _ty0, tx1, _ty1 = manifest["test_region"][0]
    dense = [Annotation(subject="bud", geometry=BBox(x, 80, x + 15, 110))
            for x in range(int(tx0) + 5, int(tx1) - 20, 2)]
    json_io.write_annotations(str(label_path), existing + dense, WIDTH, HEIGHT, keep_empty=True)

    _attest_regions_complete(
        exp["root"], exp["stem"], [manifest["calibration_region"], manifest["test_region"]])

    from tcip_mcp.pipelines.block_calibration import resolve_block_calibration_records
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.predictor import build_predictor
    from tcip_mcp.pipelines.operating_point import derive_max_dets_from_counts

    checkpoint = load_registered_checkpoint(exp["checkpoint_path"], project_path=str(tmp_path))
    predictor = build_predictor(checkpoint, device="cpu",
                                score_threshold=0.01, nms_iou=0.3, max_dets=1000)
    bundle, prov, _evidence = resolve_block_calibration_records(
        predictor, trait_name="bud_opening",
        experiment_id=exp["experiment_id"], global_nms_iou=0.3, export_tile_size=TILE)

    density_cap = derive_max_dets_from_counts(
        list(prov["cal_gt_counts"].values()) + list(prov["test_gt_counts"].values()))
    cal_only_cap = derive_max_dets_from_counts(list(prov["cal_gt_counts"].values()))
    # The dense test-side injection must actually produce a real divergence from the cal-only
    # fallback (otherwise this proves nothing).
    assert density_cap != cal_only_cap
    assert bundle.get("max_dets")._raw == density_cap
    assert bundle.get("max_dets")._raw != cal_only_cap


def test_run_inference_raster_block_calibration_admits_and_uncaps_max_dets(tmp_path: Path):
    """The real entry point: run_inference(raster_path=..., trait=...) runs block
    calibration, ships a validated-or-honestly-stamped conf, and the persisted operating point's
    max_dets is None (uncapped), never the block bundle's own band-scoped density-derived value."""
    exp = _build_experiment(tmp_path)
    manifest = exp["spatial_manifest"]
    _attest_regions_complete(
        exp["root"], exp["stem"], [manifest["calibration_region"], manifest["test_region"]])

    from tcip_mcp.tools.inference_tools import run_inference

    out_dir = tmp_path / "preds"
    result = run_inference(
        exp["checkpoint_path"], output_dir=str(out_dir), raster_path=str(exp["raster_path"]),
        conf_threshold=0.0, tile_size=TILE, overlap=0.2, trait="bud_opening",
        experiment_id=exp["experiment_id"])

    assert "error" not in result, result
    assert result["conf_source"] == "block_calibration"
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

    sidecar = read_operating_point_sidecar(out_dir)
    assert sidecar["operating_point"]["max_dets"]["value"] is None
    assert sidecar["operating_point"]["max_dets"]["derived_from"].startswith("block calibration")
    # This fixture's mosaic carries no geotransform, so the claim is content-only: the
    # georeference-aware token is reserved for a training identity that recorded one.
    assert sidecar["claim_scope_validated"] == "same_mosaic_content_identity"
    assert sidecar["block_calibration"]["experiment_id"] == exp["experiment_id"]
    assert "spatial_manifest" not in sidecar["block_calibration"]


def test_run_inference_raster_earns_the_record_behind_a_validated_block_calibrated_count(
    tmp_path: Path, monkeypatch,
):
    """The raster door's own admit case for a validated count: when the block calibration's own
    bundle is shippable and the export target is the mosaic it was calibrated against, the bucket
    is stamped validated and the stamp names a record the reader's verification confirms.

    The band passes over this fixture's small synthetic mosaic do not clear a held-out reference,
    so the bundle they resolve is stood in for by one resolved over a dense reference that does,
    with the evidence behind it, which is what the door reopens the gate over.
    """
    from tests._dense_op_fixtures import dense_records

    exp = _build_experiment(tmp_path)
    manifest = exp["spatial_manifest"]
    _attest_regions_complete(
        exp["root"], exp["stem"], [manifest["calibration_region"], manifest["test_region"]])

    import tcip_mcp.pipelines.block_calibration as block_calibration_module

    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    real_resolve = block_calibration_module.resolve_block_calibration_records
    n_images = 20
    inputs = {
        "dataset_hash": "H",
        "calibration_records": dense_records(n_images=n_images, objects_per_image=80,
                                             id_prefix="c", fp_pattern=[1] * n_images, score=0.9,
                                             fp_score=0.05),
        "holdout_records": dense_records(n_images=n_images, objects_per_image=80, id_prefix="h",
                                         shift=5.0, fp_pattern=[1] * n_images, score=0.9,
                                         fp_score=0.05),
        "tiled": True, "staged_conf_floor": 0.01,
    }

    def _held_out_block_calibration(*args, **kwargs):
        _bundle, prov, _evidence = real_resolve(*args, **kwargs)
        bundle = resolve_operating_point("bud_opening", experiment_id=exp["experiment_id"], **inputs)
        evidence = {"resolver": "resolve_operating_point", "inputs": inputs,
                    "reference_inputs": {"label_dirs": {"reserved_regions": str(exp["labels_dir"])},
                                         "stated_values": {"stem": exp["stem"]}}}
        return bundle, prov, evidence

    monkeypatch.setattr(
        block_calibration_module, "resolve_block_calibration_records", _held_out_block_calibration)

    from tcip_mcp.pipelines.resolution import (
        VALIDATED_HELD_OUT, read_operating_point_sidecar, reconcile_operating_point_validity,
        verify_stamp_binding,
    )
    from tcip_mcp.tools.inference_tools import run_inference

    out_dir = exp["root"] / "predictions" / "baseline" / "2026-01-01"
    result = run_inference(
        exp["checkpoint_path"], output_dir=str(out_dir), raster_path=str(exp["raster_path"]),
        conf_threshold=0.0, tile_size=TILE, overlap=0.2, trait="bud_opening",
        experiment_id=exp["experiment_id"])

    assert "error" not in result, result
    assert result["validated"] is True
    stamp = read_operating_point_sidecar(out_dir)
    assert verify_stamp_binding(stamp, out_dir, document="operating_point", trait="bud_opening").ok
    assert reconcile_operating_point_validity(
        [str(out_dir)], trait="bud_opening")["validated"] == VALIDATED_HELD_OUT


def test_run_inference_raster_applies_a_legitimate_zero_cross_tile_nms(
    tmp_path: Path, monkeypatch,
):
    """A block-calibrated cross_tile_nms of exactly 0.0 is a real, legitimate value (never None by
    construction, see resolve_operating_point), so the raster export pass must use it verbatim, not
    silently fall back to global_nms_iou because 0.0 reads as falsy. Deriving a genuine 0.0 through
    the real GT neighbor-IoU pipeline is not possible (derive_cross_tile_nms clamps to [0.2, 0.8]),
    so this forces the block bundle's own cross_tile_nms param to 0.0 directly, the same shape
    resolve_operating_point actually constructs it in (requires_validation=False)."""
    exp = _build_experiment(tmp_path)
    manifest = exp["spatial_manifest"]
    _attest_regions_complete(
        exp["root"], exp["stem"], [manifest["calibration_region"], manifest["test_region"]])

    import tcip_mcp.pipelines.block_calibration as block_calibration_module
    from tcip_mcp.pipelines.resolution import ResolvedParam

    real_resolve = block_calibration_module.resolve_block_calibration_records

    def _zeroed_cross_tile_nms(*args, **kwargs):
        bundle, prov, evidence = real_resolve(*args, **kwargs)
        bundle.params["cross_tile_nms"] = ResolvedParam(
            "cross_tile_nms", 0.0, source="derived", derived_from="test override: forced zero",
            requires_validation=False)
        return bundle, prov, evidence

    monkeypatch.setattr(
        block_calibration_module, "resolve_block_calibration_records", _zeroed_cross_tile_nms)

    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor

    real_predict_tiled = GenericPredictor.predict_tiled
    captured: dict = {}

    def _capture_predict_tiled(self, source, **kwargs):
        captured["global_nms_iou"] = kwargs.get("global_nms_iou")
        return real_predict_tiled(self, source, **kwargs)

    monkeypatch.setattr(GenericPredictor, "predict_tiled", _capture_predict_tiled)

    from tcip_mcp.tools.inference_tools import run_inference

    out_dir = tmp_path / "preds"
    result = run_inference(
        exp["checkpoint_path"], output_dir=str(out_dir), raster_path=str(exp["raster_path"]),
        conf_threshold=0.0, tile_size=TILE, overlap=0.2, trait="bud_opening",
        experiment_id=exp["experiment_id"], global_nms_iou=0.3)

    assert "error" not in result, result
    # The last predict_tiled call is the raster export's own final full-mosaic pass (the band
    # passes inside resolve_block_calibration_records all run first).
    assert captured["global_nms_iou"] == 0.0
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

    sidecar = read_operating_point_sidecar(out_dir)
    assert sidecar["operating_point"]["cross_tile_nms"]["value"] == 0.0


def test_run_inference_raster_claim_scope_refuses_cross_mosaic(tmp_path: Path):
    """A different raster reusing the same checkpoint must refuse: the block-validated reference
    is scoped to the training mosaic, never silently applied to a different one."""
    exp = _build_experiment(tmp_path)
    manifest = exp["spatial_manifest"]
    _attest_regions_complete(
        exp["root"], exp["stem"], [manifest["calibration_region"], manifest["test_region"]])

    other_raster = tmp_path / "different_mosaic.tif"
    _write_mosaic(other_raster, seed=99)  # different content, same dims

    from tcip_mcp.tools.inference_tools import run_inference

    out_dir = tmp_path / "preds"
    result = run_inference(
        exp["checkpoint_path"], output_dir=str(out_dir), raster_path=str(other_raster),
        conf_threshold=0.0, tile_size=TILE, overlap=0.2, trait="bud_opening",
        experiment_id=exp["experiment_id"])
    assert "error" in result
    assert not out_dir.exists()


def test_run_inference_raster_claim_scope_admits_a_georeferenced_self_export(
    tmp_path: Path,
):
    """The rail must admit valid work: exporting back over the exact training mosaic, which now
    carries a real geotransform, clears the claim-scope gate on content and georeferencing both
    and stamps the georeference-aware token."""
    plant_csv = tmp_path / "plants.csv"
    _write_plant_csv(plant_csv)
    exp = _build_experiment(tmp_path, plant_csv_paths=[str(plant_csv)])
    manifest = exp["spatial_manifest"]
    _attest_regions_complete(
        exp["root"], exp["stem"], [manifest["calibration_region"], manifest["test_region"]])

    from tcip_mcp.tools.inference_tools import run_inference

    out_dir = tmp_path / "preds"
    result = run_inference(
        exp["checkpoint_path"], output_dir=str(out_dir), raster_path=str(exp["raster_path"]),
        conf_threshold=0.0, tile_size=TILE, overlap=0.2, trait="bud_opening",
        experiment_id=exp["experiment_id"])

    assert "error" not in result, result
    assert result["claim_scope_validated"] == "same_mosaic_georeferenced_identity"


def test_run_inference_raster_claim_scope_refuses_a_moved_tiepoint_copy(tmp_path: Path):
    """A pixel-identical copy of the training mosaic with a moved tiepoint refuses (a different
    raster to a consumer that resolves pixels through the georeferencing), and the staging escape
    still ships it, stamped false exactly as a content mismatch does today."""
    plant_csv = tmp_path / "plants.csv"
    _write_plant_csv(plant_csv)
    exp = _build_experiment(tmp_path, plant_csv_paths=[str(plant_csv)])
    manifest = exp["spatial_manifest"]
    _attest_regions_complete(
        exp["root"], exp["stem"], [manifest["calibration_region"], manifest["test_region"]])

    moved = tmp_path / "moved_tiepoint.tif"
    _write_mosaic(moved, seed=0, georeferenced=True, tiepoint_x=500_020.0)

    from tcip_mcp.tools.inference_tools import run_inference
    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE

    out_dir = tmp_path / "preds"
    refused = run_inference(
        exp["checkpoint_path"], output_dir=str(out_dir), raster_path=str(moved),
        conf_threshold=0.0, tile_size=TILE, overlap=0.2, trait="bud_opening",
        experiment_id=exp["experiment_id"])
    assert "error" in refused
    assert "georeferencing mismatch" in refused["error"]
    assert not out_dir.exists()

    acknowledged = run_inference(
        exp["checkpoint_path"], output_dir=str(out_dir), raster_path=str(moved),
        conf_threshold=0.0, tile_size=TILE, overlap=0.2, trait="bud_opening",
        experiment_id=exp["experiment_id"], allow_unvalidated_staging=True)
    assert "error" not in acknowledged, acknowledged
    assert acknowledged["claim_scope_validated"] == VALIDATED_FALSE
    assert "georeferencing mismatch" in acknowledged["claim_scope_note"]


def test_run_inference_raster_claim_scope_admits_a_band_group_trained_export_over_a_georeferenced_target(
    tmp_path: Path,
):
    """A checkpoint trained on a mosaic with no readable geotransform (a band-group source, or an
    unprojected raster) exported over a target that does carry one clears the claim-scope gate on
    content alone and stamps the content-only token, never the georeferenced one no comparison
    for it ever ran."""
    exp = _build_experiment(tmp_path)
    manifest = exp["spatial_manifest"]
    _attest_regions_complete(
        exp["root"], exp["stem"], [manifest["calibration_region"], manifest["test_region"]])

    georef_copy = tmp_path / "georef_copy.tif"
    _write_mosaic(georef_copy, seed=0, georeferenced=True)

    from tcip_mcp.tools.inference_tools import run_inference

    out_dir = tmp_path / "preds"
    result = run_inference(
        exp["checkpoint_path"], output_dir=str(out_dir), raster_path=str(georef_copy),
        conf_threshold=0.0, tile_size=TILE, overlap=0.2, trait="bud_opening",
        experiment_id=exp["experiment_id"])

    assert "error" not in result, result
    assert result["claim_scope_validated"] == "same_mosaic_content_identity"


def test_run_inference_raster_without_reserved_region_names_the_real_gap(tmp_path: Path):
    """A checkpoint whose training experiment has no reserved calibration region (an ordinary
    3-way split) refuses trait+raster_path, but with a message naming the missing reserved
    region and the remedy (reserve_calibration_fraction) -- not baseline's blanket "not
    supported" refusal, which fired unconditionally for every trait+raster_path call regardless
    of whether a reserved region existed. Asserting on message content (not just "error" in
    result) is required here: baseline's blanket refusal also satisfies a bare "error" in
    result check, so that alone can't distinguish the old behavior from the new reserved-region
    check this test exists to guard."""
    exp = _build_experiment(tmp_path, reserve_frac=0.0, experiment_id="exp_no_reserve")

    from tcip_mcp.tools.inference_tools import run_inference

    out_dir = tmp_path / "preds"
    result = run_inference(
        exp["checkpoint_path"], output_dir=str(out_dir), raster_path=str(exp["raster_path"]),
        conf_threshold=0.0, tile_size=TILE, overlap=0.2, trait="bud_opening",
        experiment_id=exp["experiment_id"])
    assert "error" in result
    assert "reserved calibration" in result["error"]
    assert "reserve_calibration_fraction" in result["error"]
    assert not out_dir.exists()


def test_run_inference_raster_with_no_trait_is_byte_identical_to_the_original_raw_path(
    tmp_path: Path,
):
    """Fail-before/no-op: a raster_path export with no trait at all (today's only working raster
    export shape, unchanged by this phase) still produces the original raw, unvalidated bucket --
    the real backward-compatible guarantee this phase must not disturb, distinct from the
    trait+no-reserved-region refusal above, which is new behavior with no baseline to preserve."""
    exp = _build_experiment(tmp_path, reserve_frac=0.0, experiment_id="exp_no_trait")

    from tcip_mcp.tools.inference_tools import run_inference

    out_dir = tmp_path / "preds"
    result = run_inference(
        exp["checkpoint_path"], output_dir=str(out_dir), raster_path=str(exp["raster_path"]),
        conf_threshold=0.0, tile_size=TILE, overlap=0.2, experiment_id=exp["experiment_id"])
    assert "error" not in result
    assert result["validated"] is False
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

    op = read_operating_point_sidecar(out_dir)["operating_point"]
    assert op["conf"]["validated_against"] == "false"


def test_run_inference_raster_raw_path_stamps_explicit_conf_source_at_the_default(
    tmp_path: Path,
):
    """A caller-stated conf equal to the platform default is stamped 'explicit' on the raster
    export's own raw path, never laundered into 'default'."""
    exp = _build_experiment(tmp_path, reserve_frac=0.0, experiment_id="exp_conf_explicit")

    from tcip_mcp.pipelines.resolution import DEFAULT_CONF
    from tcip_mcp.tools.inference_tools import run_inference

    out_dir = tmp_path / "preds"
    result = run_inference(
        exp["checkpoint_path"], output_dir=str(out_dir), raster_path=str(exp["raster_path"]),
        conf_threshold=DEFAULT_CONF, tile_size=TILE, overlap=0.2,
        experiment_id=exp["experiment_id"])
    assert "error" not in result, result

    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

    op = read_operating_point_sidecar(out_dir)["operating_point"]
    assert op["conf"]["source"] == "explicit"


def test_run_inference_raster_raw_path_stamps_default_conf_source_when_omitted(
    tmp_path: Path,
):
    """The rail must admit the ordinary, unstated call: an omitted conf on the raster export's raw
    path still runs at the platform default, stamped 'default'."""
    exp = _build_experiment(tmp_path, reserve_frac=0.0, experiment_id="exp_conf_default")

    from tcip_mcp.pipelines.resolution import DEFAULT_CONF
    from tcip_mcp.tools.inference_tools import run_inference

    out_dir = tmp_path / "preds"
    result = run_inference(
        exp["checkpoint_path"], output_dir=str(out_dir), raster_path=str(exp["raster_path"]),
        tile_size=TILE, overlap=0.2, experiment_id=exp["experiment_id"])
    assert "error" not in result, result

    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

    op = read_operating_point_sidecar(out_dir)["operating_point"]
    assert op["conf"]["source"] == "default"
    assert op["conf"]["value"] == DEFAULT_CONF


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


def test_band_rects_are_reported_in_full_mosaic_coordinates():
    """Sub-banding recurses the strip split over the region's own local extent, so every returned
    rect must be translated back by the region's origin before it can name real mosaic pixels. A
    region that does not start at (0, 0) is the only fixture that can tell the two apart: bands
    left in local coordinates still look like plausible rects, while addressing a different part
    of the raster than the one the breeder attested."""
    from tcip_mcp.pipelines.block_calibration import _band_rects

    region = (500, 60, 1300, 260)
    rx0, ry0, rx1, ry1 = region
    bands = _band_rects(region, 3, TILE, 0.2, buffer_px=40, seed=0, name_prefix="cal")

    assert len(bands) == 3
    for name, (bx0, by0, bx1, by1) in bands.items():
        assert rx0 <= bx0 < bx1 <= rx1, name
        assert ry0 <= by0 < by1 <= ry1, name
    ordered = sorted(bands.values())
    for (_x0, _y0, x1, _y1), (nx0, _ny0, _nx1, _ny1) in zip(ordered, ordered[1:]):
        assert x1 <= nx0


def test_density_outlier_bands_are_flagged_against_their_own_siblings():
    """The density smoke check names the bands whose GT count is a stark outlier in either
    direction: a band far sparser than its siblings is the attestation error worth catching (a
    region marked complete after only part of it was annotated), so a one-sided check on the dense
    end would miss the case the flag exists for. Empty bands carry no density signal and are never
    flagged; the feasibility gate is what speaks for them."""
    from tcip_mcp.pipelines.block_calibration import _density_uniformity_flags

    skewed = {"cal_0": 10, "cal_1": 11, "cal_2": 1, "cal_3": 60, "cal_4": 0}
    assert _density_uniformity_flags(skewed) == ["cal_2", "cal_3"]
    assert _density_uniformity_flags({"cal_0": 10, "cal_1": 11, "cal_2": 9}) == []


def test_feasibility_counts_only_bands_that_carry_ground_truth():
    """A band with no GT at all cannot contribute to an equivalence check, so it never counts
    toward the two present bands the check needs. The refusal names the side and the shortfall;
    a layout that genuinely has two GT-bearing bands passes untouched."""
    from tcip_mcp.pipelines.block_calibration import BlockCalibrationRefused, _check_feasibility

    with pytest.raises(BlockCalibrationRefused, match="leaves only 1 band"):
        _check_feasibility({"test_0": 0, "test_1": 7, "test_2": 0}, side="test")
    _check_feasibility({"cal_0": 4, "cal_1": 0, "cal_2": 9}, side="cal")


def test_the_band_passes_run_under_the_max_dets_the_bundle_stamps(tmp_path: Path, monkeypatch):
    """The cap the bundle carries must be the cap its own evidence was collected under, on both
    surfaces that can truncate a band: the in-model ``detections_per_img`` and ``predict_tiled``'s
    own post-merge cap. A bundle stamping a cap no band pass ran under describes a run that never
    happened, and the density-derived cap here is deliberately not the one the predictor was
    constructed with, so agreement cannot come from the construction default."""
    exp = _build_experiment(tmp_path)
    manifest = exp["spatial_manifest"]
    _attest_regions_complete(
        exp["root"], exp["stem"], [manifest["calibration_region"], manifest["test_region"]])

    from tcip_mcp.pipelines.block_calibration import resolve_block_calibration_records
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.predictor import build_predictor
    from tcip_mcp.pipelines.operating_point import _current_detections_cap

    real_predict_tiled = GenericPredictor.predict_tiled
    caps: list[tuple[int | None, int | None]] = []

    def _capture_caps(self, source, **kwargs):
        caps.append((_current_detections_cap(self.model), self.max_dets))
        return real_predict_tiled(self, source, **kwargs)

    monkeypatch.setattr(GenericPredictor, "predict_tiled", _capture_caps)

    constructed_max_dets = 1000
    checkpoint = load_registered_checkpoint(exp["checkpoint_path"], project_path=str(tmp_path))
    predictor = build_predictor(checkpoint, device="cpu",
                                score_threshold=0.01, nms_iou=0.3,
                                max_dets=constructed_max_dets)
    bundle, prov, _evidence = resolve_block_calibration_records(
        predictor, trait_name="bud_opening",
        experiment_id=exp["experiment_id"], global_nms_iou=0.3, export_tile_size=TILE)

    stamped = bundle.get("max_dets")._raw
    assert len(caps) == prov["k_cal"] + prov["k_test"]
    assert {in_model for in_model, _merge in caps} == {stamped}
    assert {merge for _in_model, merge in caps} == {stamped}
    assert stamped != constructed_max_dets


def test_the_recorded_staged_conf_floor_is_the_floor_the_band_passes_ran_under(
    tmp_path: Path, monkeypatch,
):
    """``resolve_operating_point``'s conf-censoring guard only means anything if the floor recorded
    beside the sweep is the floor the band detections were really kept at. Block calibration stages
    its own low floor so hesitant detections survive to be swept, so a predictor handed a higher
    threshold must be lowered to that floor before the bands run: leaving it filtering at the
    caller's threshold censors the very evidence the recorded floor claims was kept."""
    exp = _build_experiment(tmp_path)
    manifest = exp["spatial_manifest"]
    _attest_regions_complete(
        exp["root"], exp["stem"], [manifest["calibration_region"], manifest["test_region"]])

    from tcip_mcp.pipelines.block_calibration import resolve_block_calibration_records
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.predictor import build_predictor

    real_predict_tiled = GenericPredictor.predict_tiled
    floors: list[float] = []

    def _capture_floor(self, source, **kwargs):
        floors.append(self.score_threshold)
        return real_predict_tiled(self, source, **kwargs)

    monkeypatch.setattr(GenericPredictor, "predict_tiled", _capture_floor)

    constructed_threshold = 0.4
    checkpoint = load_registered_checkpoint(exp["checkpoint_path"], project_path=str(tmp_path))
    predictor = build_predictor(checkpoint, device="cpu",
                                score_threshold=constructed_threshold, nms_iou=0.3, max_dets=1000)
    bundle, prov, _evidence = resolve_block_calibration_records(
        predictor, trait_name="bud_opening",
        experiment_id=exp["experiment_id"], global_nms_iou=0.3, export_tile_size=TILE)

    recorded_floor = bundle.get("conf").gate_evidence["staged_conf_floor"]
    assert len(floors) == prov["k_cal"] + prov["k_test"]
    assert recorded_floor is not None and recorded_floor < constructed_threshold
    assert set(floors) == {recorded_floor}


def _build_attribute_scoped_experiment(
    tmp_path: Path, *, trained_values: tuple[str, ...], reordered_values: tuple[str, ...],
    labeled_value: str, experiment_id: str = "exp_block_attribute",
) -> dict:
    """A block-calibration experiment whose subject is scoped by a categorical attribute, with the
    dataset's registry reordered after the run resolved and stamped its own name->id map.

    ``trained_values`` is the attribute-value order declared while the run resolved its map (the
    map ``subprocess_worker`` stamps onto ``config['data']['id_map']``, which every checkpoint
    embeds and ``GenericPredictor`` reads back as ``predictor.config``); ``reordered_values`` is
    the order the registry on disk declares now. ``labeled_value`` is the value every annotation
    carries. Returns the same keys ``_build_experiment`` does plus ``recorded_id_map``.
    """
    from tcip_mcp.class_registry import Attribute, ClassRegistry, Subject, write_registry
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.pipelines.model_build import build_model
    from tcip_mcp.pipelines.training.subprocess_worker import _resolve_run_id_map
    from tcip_mcp.tools.model_tools import register_model
    from tcip_mcp.pipelines.data.split_construction import auto_train_val, persist_split_manifest

    def _write_registry(values: tuple[str, ...]) -> None:
        write_registry(root / "classes.json", ClassRegistry(subjects=(
            Subject(name="bud", attributes=(
                Attribute(name="stage", type="categorical", values=values),)),)))

    root = tmp_path / "ds_attribute"
    images_dir, labels_dir = root / "images", root / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    stem = "mosaic"
    _write_mosaic(images_dir / f"{stem}.tif")
    _write_registry(trained_values)

    boxes = [Annotation(subject="bud", geometry=BBox(x, 80, x + 15, 110),
                        attributes={"stage": labeled_value})
            for x in range(10, WIDTH - 20, BOX_STEP)]
    json_io.write_annotations(str(labels_dir / f"{stem}.json"), boxes, WIDTH, HEIGHT, keep_empty=True)

    data_cfg = {
        "images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud",
        "attribute": "stage", "auto_val": True,
        "tiling": {"enabled": True, "tile_size": TILE, "overlap": 0.2},
        "split": {"val_ratio": 0.2, "test_ratio": 0.15, "seed": 1,
                  "reserve_calibration_fraction": 0.15},
    }
    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)
    assert val_ds is not None
    _subject, _attribute, recorded_id_map = _resolve_run_id_map("detection", data_cfg)
    data_cfg["id_map"] = dict(recorded_id_map)
    create_experiment(experiment_id, {"data": data_cfg})
    persist_split_manifest(experiment_id, train_ds, val_ds, data_cfg)

    _write_registry(reordered_values)

    model_source = {"builder": "tests.bespoke_models:build_bespoke_detection",
                    "builder_kwargs": {"num_classes": 1, "in_chans": 3,
                                      "min_size": TILE, "max_size": TILE * 2},
                    "task": "detection", "in_chans": 3}
    checkpoint_path = tmp_path / "model_best_attribute.pt"
    torch.save({"model_source": model_source, "config": {"data": data_cfg},
                "model_state_dict": build_model({"model_source": model_source}).state_dict()},
               str(checkpoint_path))
    result = register_model(name=experiment_id, checkpoint_path=str(checkpoint_path),
                            config={}, project_path=str(tmp_path))
    assert "error" not in result, result
    return {
        "root": root, "images_dir": images_dir, "labels_dir": labels_dir, "stem": stem,
        "checkpoint_path": str(checkpoint_path), "experiment_id": experiment_id,
        "spatial_manifest": data_cfg["split"]["spatial_manifest"],
        "recorded_id_map": dict(recorded_id_map),
    }


def test_ground_truth_decodes_through_the_checkpoints_own_recorded_id_map(tmp_path: Path):
    """Block calibration reads the mosaic's GT through the map the training run recorded, never
    through a live re-derivation off the dataset's registry: a registry whose declared
    attribute-value order changed after training assigns the same value a different id, and
    decoding the reference under that order silently scores every real object as a class the model
    was never trained to emit. The reserved regions carry only one attribute value, so the two
    orders put the whole reference in two different classes and cannot agree by accident."""
    exp = _build_attribute_scoped_experiment(
        tmp_path, trained_values=("closed", "open", "shed"),
        reordered_values=("open", "closed", "shed"), labeled_value="open")
    manifest = exp["spatial_manifest"]
    _attest_regions_complete(
        exp["root"], exp["stem"], [manifest["calibration_region"], manifest["test_region"]])

    from tcip_mcp.class_registry import assign_class_ids, read_registry
    from tcip_mcp.pipelines.block_calibration import resolve_block_calibration_records
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.predictor import build_predictor

    live_id_map = assign_class_ids(read_registry(exp["root"] / "classes.json"), "bud", "stage")
    recorded_category = exp["recorded_id_map"]["open"] + 1
    live_category = live_id_map["open"] + 1
    assert recorded_category != live_category

    checkpoint = load_registered_checkpoint(exp["checkpoint_path"], project_path=str(tmp_path))
    predictor = build_predictor(checkpoint, device="cpu",
                                score_threshold=0.01, nms_iou=0.3, max_dets=1000)
    bundle, _prov, _evidence = resolve_block_calibration_records(
        predictor, trait_name="bud_opening",
        experiment_id=exp["experiment_id"], global_nms_iou=0.3, export_tile_size=TILE)

    per_class = bundle.get("conf").gate_evidence["holdout_bias"]["per_class"]
    recorded_entry = per_class.get(str(recorded_category))
    assert recorded_entry is not None
    assert recorded_entry["tp"] + recorded_entry["fn"] > 0
    live_entry = per_class.get(str(live_category)) or {"tp": 0, "fn": 0}
    assert live_entry["tp"] + live_entry["fn"] == 0


def _drop_the_checkpoints_recorded_id_map(checkpoint_path: str, *, project_root: Path) -> None:
    """Strip ``config['data']['id_map']`` from a saved checkpoint, leaving a run whose decode map
    can only come from the dataset's registry, and re-register the rewritten bytes under their own
    new digest so the mutated file is still a checkpoint the registry names."""
    from tcip_mcp.tools.model_tools import register_model

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload["config"]["data"].pop("id_map", None)
    torch.save(payload, checkpoint_path)
    result = register_model(name="block-calibration-no-id-map", checkpoint_path=checkpoint_path,
                            config={}, project_path=str(project_root))
    assert "error" not in result, result


def test_block_calibration_refuses_when_no_id_map_can_be_resolved(tmp_path: Path):
    """An attribute-scoped run whose checkpoint recorded no map and whose dataset has no registry
    has nothing to decode the mosaic's ground truth with. The decode resolver every
    prediction-writing door already calls carries that precondition, so this path refuses by name
    instead of restating the prefer-recorded-else-derive rule and reaching the registry read."""
    exp = _build_attribute_scoped_experiment(
        tmp_path, trained_values=("closed", "open", "shed"),
        reordered_values=("closed", "open", "shed"), labeled_value="open",
        experiment_id="exp_block_no_id_map")
    _drop_the_checkpoints_recorded_id_map(exp["checkpoint_path"], project_root=tmp_path)
    (exp["root"] / "classes.json").unlink()

    from tcip_mcp.pipelines.block_calibration import (
        BlockCalibrationRefused, resolve_block_calibration_records,
    )
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.predictor import build_predictor

    checkpoint = load_registered_checkpoint(exp["checkpoint_path"], project_path=str(tmp_path))
    predictor = build_predictor(checkpoint, device="cpu",
                                score_threshold=0.01, nms_iou=0.3, max_dets=1000)
    with pytest.raises(BlockCalibrationRefused, match="records no name->id map"):
        resolve_block_calibration_records(
            predictor, trait_name="bud_opening",
            experiment_id=exp["experiment_id"], global_nms_iou=0.3, export_tile_size=TILE)


def test_block_calibration_runs_on_a_recorded_id_map_with_no_registry_on_disk(tmp_path: Path):
    """The refusal above must not swallow the legitimate case: a checkpoint that carries its own
    recorded map needs no registry at all, so calibration resolves with classes.json gone."""
    exp = _build_attribute_scoped_experiment(
        tmp_path, trained_values=("closed", "open", "shed"),
        reordered_values=("closed", "open", "shed"), labeled_value="open",
        experiment_id="exp_block_recorded_no_registry")
    manifest = exp["spatial_manifest"]
    _attest_regions_complete(
        exp["root"], exp["stem"], [manifest["calibration_region"], manifest["test_region"]])
    (exp["root"] / "classes.json").unlink()

    from tcip_mcp.pipelines.block_calibration import resolve_block_calibration_records
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.predictor import build_predictor

    checkpoint = load_registered_checkpoint(exp["checkpoint_path"], project_path=str(tmp_path))
    predictor = build_predictor(checkpoint, device="cpu",
                                score_threshold=0.01, nms_iou=0.3, max_dets=1000)
    bundle, prov, _evidence = resolve_block_calibration_records(
        predictor, trait_name="bud_opening",
        experiment_id=exp["experiment_id"], global_nms_iou=0.3, export_tile_size=TILE)

    assert sum(prov["cal_gt_counts"].values()) > 0
    assert bundle.get("conf").gate_evidence["calibration_image_ids"]


def _attest_regions_complete_through_the_coverage_route(
    client, image_path: str, regions: list[list[tuple[int, int, int, int]]],
    *, subject: str = "bud",
) -> list[str]:
    """Attest every reference-grid cell the given regions touch through the coverage route the
    Annotate canvas's Attest control posts to, one cell per request, and return the attested
    cell names. The grid posted is the one the grid route serves, the same lattice the browser
    draws, with the cell list and derivation line split off the way the browser's grid hook
    splits them before posting."""
    from tcip_mcp.pipelines.data.tiling import rects_overlap

    grid_resp = client.get("/api/coverage/grid", params={"path": image_path, "tile_size": TILE})
    assert grid_resp.status_code == 200, grid_resp.text
    served = grid_resp.json()["grid"]
    cells = served["cells"]
    grid = {key: value for key, value in served.items() if key not in ("cells", "derivation")}
    all_rects = [tuple(r) for region in regions for r in region]
    covered = sorted(
        c["name"] for c in cells
        if any(rects_overlap((c["x0"], c["y0"], c["x1"], c["y1"]), r) for r in all_rects)
    )
    for name in covered:
        resp = client.post("/api/coverage/completeness", json={
            "image_path": image_path, "subject": subject, "grid": grid, "cell": name,
            "complete": True, "user": "breeder", "view_scale": None})
        assert resp.status_code == 200, resp.text
        assert resp.json()["complete"] is True
    return covered


def test_regions_attested_through_the_coverage_route_admit_block_calibration(tmp_path: Path):
    """The completeness gate must read the record the breeder's own attestation actually writes.
    Every reserved cell here is toggled complete through the coverage route, so the record shape
    the route produces (its bucket key, its grid, its per-cell digest stamp) is what the gate
    resolves, rather than a store the test wrote in the shape the gate expects."""
    exp = _build_experiment(tmp_path)
    manifest = exp["spatial_manifest"]

    from fastapi.testclient import TestClient

    from tcip_web.app import app

    client = TestClient(app, base_url="http://127.0.0.1")
    attested = _attest_regions_complete_through_the_coverage_route(
        client, str(exp["raster_path"]),
        [manifest["calibration_region"], manifest["test_region"]])
    assert attested

    from tcip_mcp.pipelines.block_calibration import resolve_block_calibration_records
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.predictor import build_predictor
    from tcip_mcp.pipelines.region_completeness import incomplete_cells_for_rect

    for region in (manifest["calibration_region"], manifest["test_region"]):
        assert incomplete_cells_for_rect(
            str(exp["root"]), "bud", exp["stem"], tuple(region[0])) == []

    checkpoint = load_registered_checkpoint(exp["checkpoint_path"], project_path=str(tmp_path))
    predictor = build_predictor(checkpoint, device="cpu",
                                score_threshold=0.01, nms_iou=0.3, max_dets=1000)
    bundle, prov, _evidence = resolve_block_calibration_records(
        predictor, trait_name="bud_opening",
        experiment_id=exp["experiment_id"], global_nms_iou=0.3, export_tile_size=TILE)

    assert sum(prov["cal_gt_counts"].values()) > 0
    assert bundle.get("conf").gate_evidence["calibration_image_ids"]
