"""Sliced inference through SAHI: one lattice, one merge, both source kinds, and one execution
regime from calibration through the stamp a count claim is sealed over."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sahi")

TILE = 128
OVERLAP = 0.25
# A 200px frame at 128px tiles and 0.25 overlap is a 2x2 lattice whose second column and row start
# at 72: the seam blob crosses x=128, and the wide blob is wider than any slice.
FRAME = 200
SEAM_BLOB = (90, 20, 130, 60)
INNER_BLOB = (150, 140, 180, 170)
WIDE_BLOB = (20, 95, 190, 120)
BLOBS = (SEAM_BLOB, INNER_BLOB, WIDE_BLOB)


def _checkpoint(tmp_path: Path, *, in_chans: int = 3, with_masks: bool = False,
                classes_by_channel: bool = False, tiling: dict | None = None,
                data: dict | None = None):
    """A registered bright-blob checkpoint, loaded through the platform's own loader."""
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.model_build import build_model
    from tcip_mcp.tools.model_tools import register_model

    task = "instance_seg" if with_masks else "detection"
    model_source = {"builder": "tests.bespoke_models:build_bright_blob_detector",
                    "builder_kwargs": {"in_chans": in_chans, "with_masks": with_masks,
                                       "classes_by_channel": classes_by_channel},
                    "task": task}
    model = build_model({"model_source": model_source})
    ckpt = tmp_path / "model_best.pt"
    data_cfg = {**(data or {}), **({"tiling": tiling} if tiling else {})}
    config = {"model_source": model_source, "data": data_cfg} if data_cfg else {}
    torch.save({"model_source": model_source, "model_state_dict": model.state_dict(),
                "config": config}, str(ckpt))
    result = register_model(name="blob", checkpoint_path=str(ckpt), config={},
                            project_path=str(tmp_path))
    assert "error" not in result, result
    return str(ckpt), load_registered_checkpoint(str(ckpt), project_path=str(tmp_path))


def _frame(bands: int = 3, *, value=255, blobs=BLOBS) -> np.ndarray:
    arr = np.zeros((FRAME, FRAME, bands), dtype=np.uint8)
    for x0, y0, x1, y1 in blobs:
        arr[y0:y1, x0:x1] = value
    return arr


def _predictor(checkpoint):
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor

    return GenericPredictor(checkpoint, device="cpu", score_threshold=0.0, max_dets=None)


def _sliced(pred, source, **overrides):
    kwargs = dict(tile_size=TILE, overlap=OVERLAP, postprocess="nmm", cross_tile_nms=0.5,
                  tile_batch_size=2, tile_resize=None, require_masks=True)
    kwargs.update(overrides)
    return pred.predict_sliced(source, **kwargs)


def _png(directory: Path, arr: np.ndarray, name: str = "frame.png") -> str:
    from PIL import Image

    path = directory / name
    Image.fromarray(arr).save(path)
    return str(path)


@pytest.mark.parametrize("height,width,tile,overlap", [
    (FRAME, 333, TILE, OVERLAP),
    (200, 200, 64, 0.2),  # 64 * 0.2 is fractional, so rounding the stride either way differs
])
def test_slice_lattice_states_sahis_own_boxes(height, width, tile, overlap):
    from sahi.slicing import get_slice_bboxes

    from tcip_mcp.pipelines.slicing import slice_lattice

    ours = slice_lattice(height, width, tile, overlap)
    theirs = get_slice_bboxes(image_height=height, image_width=width, slice_height=tile,
                              slice_width=tile, auto_slice_resolution=False,
                              overlap_height_ratio=overlap, overlap_width_ratio=overlap)
    assert ours == [tuple(b) for b in theirs]


def test_objects_across_a_seam_predict_once_through_both_source_kinds(tmp_path):
    """The whole-decode path and the windowed path merge to the same boxes, the wide blob no slice
    holds whole predicted once over its full extent."""
    from tcip_mcp.pipelines.raster_source import open_raster

    _path, checkpoint = _checkpoint(tmp_path)
    pred = _predictor(checkpoint)
    npy = tmp_path / "frame.npy"
    np.save(npy, _frame())

    whole = _sliced(pred, _png(tmp_path, _frame()))
    with open_raster(str(npy), 3) as reader:
        windowed = _sliced(pred, reader, source_label="frame")

    expected = sorted(list(map(float, b)) for b in BLOBS)
    assert whole["tiles"] == windowed["tiles"] == 4
    assert sorted(whole["boxes"]) == expected
    assert sorted(windowed["boxes"]) == expected


def test_a_five_band_slice_reaches_the_model_with_every_band(tmp_path):
    """Every band reaches the model with its own values, never one band repeated or dropped."""
    from tcip_mcp.pipelines.raster_source import open_raster

    band_values = [255, 200, 150, 100, 60]
    _path, checkpoint = _checkpoint(tmp_path, in_chans=5)
    pred = _predictor(checkpoint)
    npy = tmp_path / "five.npy"
    np.save(npy, _frame(bands=5, value=np.asarray(band_values, dtype=np.uint8)))

    whole = _sliced(pred, str(npy))
    with open_raster(str(npy), 5) as reader:
        _sliced(pred, reader, source_label="five")

    assert pred.model.seen_channels == [5] * 8
    expected = pytest.approx([v / 255 for v in band_values], abs=1e-6)
    assert any(peaks == expected for peaks in pred.model.seen_band_peaks)
    assert len(whole["boxes"]) == len(BLOBS)


def _mask_extents(record: dict) -> list[tuple]:
    extents = []
    for mask in record["masks"]:
        assert len(mask["segmentation"]) == 1
        xs, ys = mask["segmentation"][0][0::2], mask["segmentation"][0][1::2]
        extents.append((min(xs), min(ys), max(xs), max(ys)))
    return sorted(extents)


def test_instance_masks_merged_across_a_seam_are_one_polygon_per_object_and_class(tmp_path):
    """Two objects of two classes over one footprint wider than any slice: each class's partial
    masks merge into one polygon spanning the whole object, and the two classes never merge."""
    _path, checkpoint = _checkpoint(tmp_path, with_masks=True, classes_by_channel=True)
    arr = np.zeros((FRAME, FRAME, 3), dtype=np.uint8)
    x0, y0, x1, y1 = WIDE_BLOB
    arr[y0:y1, x0:x1, 0] = 255
    arr[y0:y1, x0:x1, 1] = 255

    result = _sliced(_predictor(checkpoint), _png(tmp_path, arr))

    assert sorted(result["labels"]) == [1, 2]
    # Contours run through pixel centers, so a blob's polygon ends one pixel inside its box.
    assert _mask_extents(result) == [(x0, y0, x1 - 1, y1 - 1)] * 2


def test_an_untiled_record_carries_the_polygons_the_sliced_record_does(tmp_path):
    _path, checkpoint = _checkpoint(tmp_path, with_masks=True)
    pred = _predictor(checkpoint)
    path = _png(tmp_path, _frame())

    whole = pred.predict(path)
    sliced = _sliced(pred, path)

    assert whole["count"] == sliced["count"] == len(BLOBS)
    assert _mask_extents(whole) == _mask_extents(sliced)
    assert sorted(whole["boxes"]) == sorted(sliced["boxes"])
    assert whole["cap_hit"] is sliced["cap_hit"] is False
    assert whole["mask_binarize"] == sliced["mask_binarize"]


def test_a_predicted_mask_carries_the_rings_the_shared_extractor_draws(tmp_path):
    """A predictor's masks and SAM-assisted ground truth describe one object through one
    extractor."""
    from tcip_annotation.mask_contours import mask_to_polygon_rings

    _path, checkpoint = _checkpoint(tmp_path, with_masks=True)
    # A disk, whose contour the shared extractor simplifies where a rectangle's has nothing to drop.
    yy, xx = np.mgrid[:FRAME, :FRAME]
    disk = ((xx - 100) ** 2 + (yy - 100) ** 2 <= 40 ** 2).astype(np.uint8)
    path = _png(tmp_path, np.repeat(disk[..., None] * 255, 3, axis=2), "disk.png")
    expected = [[c for point in ring for c in point] for ring in mask_to_polygon_rings(disk)]

    record = _predictor(checkpoint).predict(path)

    assert [m["segmentation"] for m in record["masks"]] == [expected]


def test_masks_are_cut_at_the_platform_binarize_threshold_the_record_names(tmp_path, monkeypatch):
    import importlib

    _path, checkpoint = _checkpoint(tmp_path, with_masks=True)
    pred = _predictor(checkpoint)
    path = _png(tmp_path, _frame())
    # The package attribute of this name is a same-named function, so the module comes from sys.modules.
    mask_geometry = importlib.import_module("tcip_mcp.pipelines.measurement.mask_geometry")
    real = mask_geometry.resolve_binarize_threshold

    assert all(m["segmentation"] for m in pred.predict(path)["masks"])
    monkeypatch.setattr(mask_geometry, "resolve_binarize_threshold", lambda *a, **k: real(1.5))
    record = pred.predict(path)
    assert [m["segmentation"] for m in record["masks"]] == [[]] * len(BLOBS)
    assert record["mask_binarize"]["value"] == 1.5


def test_an_untiled_detector_record_carries_no_masks(tmp_path):
    _path, checkpoint = _checkpoint(tmp_path)

    record = _predictor(checkpoint).predict(_png(tmp_path, _frame()))

    assert "masks" not in record and record["count"] == len(BLOBS)


@pytest.mark.parametrize("with_masks", [False, True])
def test_a_windowed_pass_resumed_mid_raster_matches_an_uninterrupted_one(tmp_path, with_masks):
    """The recorded progress carries each slice's shifted predictions whole, so a resumed pass
    merges the seeded predictions, masks included, exactly as the uninterrupted pass does."""
    from tcip_mcp.pipelines.raster_source import open_raster

    _path, checkpoint = _checkpoint(tmp_path, with_masks=with_masks)
    pred = _predictor(checkpoint)
    npy = tmp_path / "frame.npy"
    np.save(npy, _frame())
    recorded: list[dict] = []

    class Interrupted(Exception):
        pass

    def record_then_stop(_start, _end, batch):
        recorded.append(batch)
        raise Interrupted

    with open_raster(str(npy), 3) as reader:
        uninterrupted = _sliced(pred, reader)
        with pytest.raises(Interrupted):
            _sliced(pred, reader, progress=record_then_stop)
        prior = {field: [v for b in recorded for v in b[field]]
                 for field in ("slices", "predictions")}
        seen_before = len(pred.model.seen_channels)
        resumed = _sliced(pred, reader, prior=prior)

    assert len(prior["slices"]) == 2
    assert len(pred.model.seen_channels) - seen_before == 2
    assert resumed == uninterrupted
    if with_masks:
        assert all(m["segmentation"] for m in resumed["masks"])


def test_training_and_inference_slice_one_frame_on_one_lattice(tmp_path):
    """The tiled training dataset's slices and a sliced inference pass's recorded slices over the
    same frame, each read whole off its own route's record."""
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    from tcip_mcp.pipelines.raster_source import open_raster
    from tests._producer_fixtures import dataset_over

    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    np.save(images / "frame.npy", _frame())
    json_io.write_annotations(str(labels / "frame.json"),
                              [Annotation(subject="bud", geometry=BBox(*SEAM_BLOB))],
                              FRAME, FRAME, keep_empty=True)
    ds = dataset_over("detection", str(images), str(labels), subject="bud",
                      stated={"num_channels": 3},
                      tiling={"enabled": True, "tile_size": TILE, "overlap": OVERLAP})

    _path, checkpoint = _checkpoint(tmp_path)
    recorded: list[list[int]] = []
    with open_raster(str(images / "frame.npy"), 3) as reader:
        _sliced(_predictor(checkpoint), reader, require_masks=False,
                progress=lambda _s, _e, batch: recorded.extend(batch["slices"]))

    assert [box for _stem, box in ds.tile_entries] == [tuple(s) for s in recorded]


def test_a_postprocess_outside_the_vocabulary_refuses_by_name(tmp_path):
    _path, checkpoint = _checkpoint(tmp_path)
    pred = _predictor(checkpoint)
    path = _png(tmp_path, _frame())

    with pytest.raises(ValueError, match="greedynmm"):
        _sliced(pred, path, postprocess="soft-nms")
    assert pred.model.seen_channels == []
    assert len(_sliced(pred, path, postprocess="greedynmm")["boxes"]) == len(BLOBS)


def test_the_merge_leaves_the_process_environment_as_it_found_it(tmp_path, monkeypatch):
    """SAHI's torchvision backend selects its device by writing ``CUDA_VISIBLE_DEVICES`` into the
    process, which a training launch's child would inherit; the platform's merge never does."""
    import os

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _path, checkpoint = _checkpoint(tmp_path)

    _sliced(_predictor(checkpoint), _png(tmp_path, _frame()), require_masks=False)

    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_a_stated_merge_threshold_never_reaches_the_detectors_own_nms(tmp_path):
    from tcip_mcp.tools.inference_tools import _prepare_pass

    _path, checkpoint = _checkpoint(tmp_path)
    p = _prepare_pass(
        checkpoint, images_dir=None, conf_threshold=0.0, device="cpu", tile=True,
        tile_size=TILE, overlap=OVERLAP, cross_tile_nms=0.9, max_dets=None, postprocess="nms",
        experiment_id=None, tile_batch_size=2)

    assert not isinstance(p, str), p
    assert (p.cross_tile_nms.value, p.cross_tile_nms.source) == (0.9, "explicit")
    assert p.predictor.model.nms_thresh == 0.45


def test_each_merge_runs_at_the_threshold_its_own_metric_derives():
    """A box nested in another overlaps it at IoU 0.25 and IoS 1: suppression by IoU derives a
    threshold the pair stays apart under, merging by IoS one it joins under, each merge running at
    its own metric's value and labeled with that metric."""
    from sahi.prediction import ObjectPrediction

    from tcip_mcp.pipelines.derivations import CROSS_TILE_NMS_DERIVATIONS
    from tcip_mcp.pipelines.resolution import resolve_cross_tile_nms
    from tcip_mcp.pipelines.slicing import cross_tile_merge, slicing_record

    nested = [[[0.0, 0.0, 20.0, 20.0], [5.0, 5.0, 10.0, 10.0]]]
    merged = {}
    for postprocess, metric in (("nms", "IOU"), ("nmm", "IOS")):
        param = resolve_cross_tile_nms(None, slicing_record(OVERLAP, None, postprocess), nested)
        assert param.derived_from == CROSS_TILE_NMS_DERIVATIONS[metric]
        predictions = [ObjectPrediction(bbox=[0, 0, 20, 20], category_id=1, score=0.9),
                       ObjectPrediction(bbox=[5, 5, 15, 15], category_id=1, score=0.8)]
        merged[postprocess] = (param.value,
                               len(cross_tile_merge(postprocess, param.value)(predictions)))

    assert merged["nms"] == (pytest.approx(0.30), 2)
    assert merged["nmm"] == (pytest.approx(0.80), 1)
    assert "provisional" in CROSS_TILE_NMS_DERIVATIONS["IOS"]


def _dry_and_real(tmp_path, **stated):
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar
    from tcip_mcp.tools.inference_tools import run_inference

    ckpt, _checkpoint_record = _checkpoint(tmp_path)
    images = tmp_path / "images"
    images.mkdir()
    _png(images, _frame())
    out = tmp_path / "bucket"
    dry = run_inference(ckpt, images_dir=str(images), output_dir=str(out), dry_run=True, **stated)
    real = run_inference(ckpt, images_dir=str(images), output_dir=str(out), **stated)
    assert "error" not in dry and "error" not in real, (dry, real)
    return dry, read_operating_point_sidecar(out)


def test_the_dry_run_reports_the_record_the_run_stamps(tmp_path):
    dry, stamp = _dry_and_real(
        tmp_path, tile=True, tile_size=TILE, overlap=OVERLAP, conf_threshold=0.2,
        cross_tile_nms=0.4, max_dets=50, postprocess="greedynmm", allow_unvalidated_staging=True)

    assert dry["operating_point"] == stamp["operating_point"]
    assert dry["slicing"] == stamp["slicing"]


# --- one execution regime from calibration through the sealed claim ---

N_CALIBRATION_IMAGES = 20


def _blob_calibration_dataset(root: Path) -> tuple[Path, Path]:
    """Frames of 20px blobs in three rows of four, each image shifted a pixel further right so no
    two share ground truth, labeled with 32px boxes that overlap their row neighbors by 10px (a
    neighbor tail every metric derives a threshold from); columns cross the lattice's seam."""
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    images, labels = root / "images", root / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for i in range(N_CALIBRATION_IMAGES):
        arr = np.zeros((FRAME, FRAME, 3), dtype=np.uint8)
        boxes = []
        for y0 in (20, 80, 140):
            for col in range(4):
                x0 = 40 + col * 22 + i
                arr[y0:y0 + 20, x0:x0 + 20] = 255
                boxes.append(BBox(x0 - 6, y0 - 6, x0 + 26, y0 + 26))
        _png(images, arr, f"img{i:02d}.png")
        json_io.write_annotations(str(labels / f"img{i:02d}.json"),
                                  [Annotation(subject="bud", geometry=b) for b in boxes],
                                  FRAME, FRAME, keep_empty=True)
    return images, labels


def test_a_calibrated_pass_collects_exports_previews_and_seals_one_regime(
        tmp_path, monkeypatch, seed_bud_trait_spec):
    """A tiled calibration through run_inference: the merge threshold is resolved from the
    calibration GT in the metric its postprocess compares over before any image is predicted,
    every calibration and export slice is merged at it, the dry run reports the stamp the real
    call writes, and the record sealed over the bucket answers for the regime it ran."""
    from tcip_mcp.pipelines.derivations import CROSS_TILE_NMS_DERIVATIONS
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar, verify_stamp_binding
    from tcip_mcp.tools.inference_tools import run_inference

    dataset = tmp_path / "ds"
    images, labels = _blob_calibration_dataset(dataset)
    ckpt, _record = _checkpoint(tmp_path, data={"subject": "bud", "id_map": {"bud": 0}})
    merged_at: list[float] = []
    real_predict_sliced = GenericPredictor.predict_sliced

    def _capture(self, source, **kwargs):
        merged_at.append(kwargs["cross_tile_nms"])
        return real_predict_sliced(self, source, **kwargs)

    monkeypatch.setattr(GenericPredictor, "predict_sliced", _capture)
    out = dataset / "predictions" / "blob" / "2026-01-01"
    call = dict(images_dir=str(images), output_dir=str(out), device="cpu", tile=True,
                tile_size=TILE, overlap=OVERLAP, postprocess="nmm", trait="bud_opening",
                calibration_labels_dir=str(labels), group_by="stem",
                allow_unvalidated_staging=True)

    dry = run_inference(ckpt, dry_run=True, **call)
    assert "error" not in dry, dry
    assert not out.exists()
    real = run_inference(ckpt, **call)
    assert "error" not in real, real

    stamp = read_operating_point_sidecar(out)
    merge = stamp["operating_point"]["cross_tile_nms"]
    assert merge["derived_from"] == CROSS_TILE_NMS_DERIVATIONS["IOS"]
    assert len(merged_at) > N_CALIBRATION_IMAGES and set(merged_at) == {merge["value"]}
    assert (dry["operating_point"], dry["slicing"]) == (stamp["operating_point"], stamp["slicing"])
    assert stamp["validated"] is True
    assert verify_stamp_binding(stamp, out, document="operating_point", trait="bud_opening").ok


def _earned_result(tmp_path, *, slicing_postprocess: str):
    """A run whose calibration evidence was collected tiled under ``nms``, its result stating the
    slicing record a pass under ``slicing_postprocess`` carries."""
    from tcip_mcp.pipelines.slicing import slicing_record
    from tests._binding_fixtures import calibrated_run_fields, run_result

    fields = calibrated_run_fields(labels_dir=tmp_path, checkpoint_sha256="deadbeef",
                                   postprocess="nms", tile_size=TILE, tile_size_source="derived")
    fields["slicing"] = slicing_record(fields["slicing"]["overlap"], None, slicing_postprocess)
    return run_result(
        results=[{"image": "a.png", "width": 100, "height": 100, "boxes": [[10.0, 10.0, 30.0, 30.0]],
                  "scores": [0.9], "labels": [1], "count": 1}], **fields)


def _calibrated_bucket_under(tmp_path, postprocess: str):
    """A bucket a calibrated tiled run published under ``postprocess``, its record earned by the
    publisher itself."""
    from tcip_mcp.dataset_layout import bucket_dataset_root
    from tcip_mcp.tools.inference_tools import publish_bucket

    out = tmp_path / "ds" / "predictions" / "baseline" / "2026-01-01"
    pub = publish_bucket(_earned_result(tmp_path, slicing_postprocess=postprocess), out=out,
                         trait="bud_opening", dataset_root=bucket_dataset_root(out),
                         allow_unvalidated_staging=False)
    assert pub["refusal"] is None and pub["op_stamp"]["validated"], pub
    return out


def test_a_bucket_asserting_a_regime_its_calibration_never_ran_under_refuses_by_name(
        tmp_path, seed_bud_trait_spec):
    with pytest.raises(ValueError, match="slicing"):
        _calibrated_bucket_under(tmp_path, "nmm")


def test_a_claim_stating_no_slicing_record_refuses_by_name(tmp_path, seed_bud_trait_spec):
    from tcip_mcp.dataset_layout import bucket_dataset_root
    from tcip_mcp.pipelines.resolution import open_validation, seal_validation
    from tcip_mcp.tools.inference_tools import _calibration_evidence
    from tests._binding_fixtures import calibrated_run_fields, write_prediction

    fields = calibrated_run_fields(labels_dir=tmp_path, checkpoint_sha256="deadbeef",
                                   postprocess="nms", tile_size=TILE, tile_size_source="derived")
    out = tmp_path / "ds" / "predictions" / "baseline" / "2026-01-01"
    write_prediction(out, "a")
    root = bucket_dataset_root(out)
    evidence = _calibration_evidence(fields)
    draft = open_validation(
        document="operating_point",
        evidence={"resolver": evidence["resolver"], "inputs": evidence["inputs"]},
        trait="bud_opening", checkpoint_sha256="deadbeef", producing_experiment_id=None,
        reference_inputs={**evidence["reference_inputs"], "dataset_root": str(root)})
    stamp = {"validated": True, "operating_point": fields["operating_point"]}

    with pytest.raises(ValueError, match="slicing"):
        seal_validation(draft, dataset_root=root, bucket_dirs=[out], stamp_body=stamp)


def _delivery(out: Path):
    from tcip_mcp.pipelines.resolution import (
        check_delivery_gate, reconcile_operating_point_validity,
    )

    recon = reconcile_operating_point_validity([str(out)], trait="bud_opening")
    return recon, check_delivery_gate({"operating_point": recon["validated"]})


def test_a_bucket_stamped_under_its_calibrations_slicing_delivers(tmp_path, seed_bud_trait_spec):
    out = _calibrated_bucket_under(tmp_path, "nms")

    _recon, gate = _delivery(out)

    assert gate.ok, gate.reason


def test_a_bucket_restamped_under_another_merge_than_its_calibration_refuses_by_name(
        tmp_path, seed_bud_trait_spec):
    from tcip_mcp.pipelines.resolution import update_sidecar
    from tcip_mcp.pipelines.slicing import slicing_record

    out = _calibrated_bucket_under(tmp_path, "nms")
    assert update_sidecar(out, lambda s: {**s, "slicing": slicing_record(OVERLAP, None, "nmm")})

    recon, gate = _delivery(out)

    assert not gate.ok
    assert "slicing disagree" in recon["binding_notes"][str(out)]


def test_the_stamp_records_the_slice_geometry_the_operating_point_derived(tmp_path):
    import sahi

    from tcip_mcp.pipelines.inference.predictor import resolve_tile_geometry
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar
    from tcip_mcp.tools.inference_tools import run_inference

    ckpt, checkpoint = _checkpoint(tmp_path, tiling={"tile_size": TILE, "overlap": OVERLAP})
    images = tmp_path / "images"
    images.mkdir()
    _png(images, _frame())

    out = tmp_path / "bucket"
    response = run_inference(ckpt, images_dir=str(images), output_dir=str(out),
                             conf_threshold=0.0, postprocess="nmm", allow_unvalidated_staging=True)
    assert "error" not in response, response
    stamp = read_operating_point_sidecar(out)

    geometry = resolve_tile_geometry(_predictor(checkpoint), tiled=True, tile_size=None,
                                     overlap=None)
    assert stamp["operating_point"]["tile_size"]["value"] == geometry.tile_size == TILE
    assert stamp["slicing"]["overlap"] == geometry.overlap == OVERLAP
    assert stamp["slicing"]["postprocess"] == "nmm"
    assert (stamp["slicing"]["merge_type"], stamp["slicing"]["match_metric"]) == ("NMM", "IOS")
    assert stamp["slicing"]["sahi_version"] == sahi.__version__
