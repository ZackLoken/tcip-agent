"""Sliced inference through SAHI: one lattice, one merge, both source kinds, and one execution
record from the assessment through the bucket published under it."""

from __future__ import annotations

from tcip_mcp.dataset_layout import UNDATED_BUCKET

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tcip_mcp.pipelines.model_build import CONFIG_KEY, STATE_DICT_KEY  # noqa: E402

from tcip_mcp.pipelines.execution import Stated  # noqa: E402
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


COLOR = {"name": "color", "type": "categorical", "values": ["red", "green", "blue"]}
"""An attribute whose value is the band a blob is bright in, as the blob detector calls it."""


def _checkpoint(tmp_path: Path, *, in_chans: int = 3, with_masks: bool = False,
                by_band: bool = False, tiling: dict | None = None,
                data: dict | None = None):
    """A registered bright-blob checkpoint, loaded through the platform's own loader; with
    ``by_band`` its recorded scope declares :data:`COLOR`."""
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.model_build import build_model, recorded_model_dims
    from tcip_mcp.tools.model_tools import register_model

    task = "instance_seg" if with_masks else "detection"
    model_source = {"builder": "tests.bespoke_models:build_bright_blob_detector",
                    "builder_kwargs": {"with_masks": with_masks}, "task": task}
    ckpt = tmp_path / "model_best.pt"
    scope = {"subject": "bud", "attributes": [COLOR] if by_band else []}
    data_cfg = {"num_channels": in_chans, "scope": scope,
                **(data or {}), **({"tiling": tiling} if tiling else {})}
    config = {"model_source": model_source, "data": data_cfg}
    model = build_model(config, recorded_model_dims(config))
    torch.save({STATE_DICT_KEY: model.state_dict(), CONFIG_KEY: config}, str(ckpt))
    result = register_model(name="blob", checkpoint_path=str(ckpt), config={},
                            project=tmp_path)
    assert "error" not in result, result
    return str(ckpt), load_registered_checkpoint(str(ckpt), project=tmp_path)


def _frame(bands: int = 3, *, value=255, blobs=BLOBS) -> np.ndarray:
    arr = np.zeros((FRAME, FRAME, bands), dtype=np.uint8)
    for x0, y0, x1, y1 in blobs:
        arr[y0:y1, x0:x1] = value
    return arr


def _pass(checkpoint, **stated):
    """The tiled pass ``checkpoint`` runs at the fixture's tile edge and overlap, merging by NMM
    at 0.5 and keeping every score, with ``stated`` over those."""
    from tcip_mcp.pipelines.execution import prepare_pass

    values = dict(tile=True, tile_size=TILE, overlap=OVERLAP, postprocess="nmm",
                  cross_tile_nms=0.5, conf=0.0)
    values.update(stated)
    return prepare_pass(checkpoint, Stated(**values), device="cpu", tile_batch_size=2)


def _sliced(p, source, *, require_masks: bool = True, **kwargs):
    return p.predictor.predict_sliced(source, execution=p.execution,
                                      tile_batch_size=p.tile_batch_size,
                                      require_masks=require_masks, **kwargs)


def _whole(p, source) -> dict:
    """The untiled record of ``source`` under the pass's own conf and cap."""
    from tcip_mcp.pipelines.execution import untiled_execution

    return p.predictor.predict(source, untiled_execution(
        p.checkpoint, conf=p.execution.conf, max_dets=p.execution.max_dets))


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
    p = _pass(checkpoint)
    npy = tmp_path / "frame.npy"
    np.save(npy, _frame())

    whole = _sliced(p, _png(tmp_path, _frame()))
    with open_raster(str(npy), 3) as reader:
        windowed = _sliced(p, reader, source_label="frame")

    expected = sorted(list(map(float, b)) for b in BLOBS)
    assert whole["tiles"] == windowed["tiles"] == 4
    assert sorted(whole["boxes"]) == expected
    assert sorted(windowed["boxes"]) == expected


def test_a_five_band_slice_reaches_the_model_with_every_band(tmp_path):
    """Every band reaches the model with its own values, never one band repeated or dropped."""
    from tcip_mcp.pipelines.raster_source import open_raster

    band_values = [255, 200, 150, 100, 60]
    _path, checkpoint = _checkpoint(tmp_path, in_chans=5)
    p = _pass(checkpoint)
    npy = tmp_path / "five.npy"
    np.save(npy, _frame(bands=5, value=np.asarray(band_values, dtype=np.uint8)))

    whole = _sliced(p, str(npy))
    with open_raster(str(npy), 5) as reader:
        _sliced(p, reader, source_label="five")

    assert p.predictor.model.seen_channels == [5] * 8
    expected = pytest.approx([v / 255 for v in band_values], abs=1e-6)
    assert any(peaks == expected for peaks in p.predictor.model.seen_band_peaks)
    assert len(whole["boxes"]) == len(BLOBS)


def _mask_extents(record: dict) -> list[tuple]:
    extents = []
    for mask in record["masks"]:
        assert len(mask["segmentation"]) == 1
        xs, ys = mask["segmentation"][0][0::2], mask["segmentation"][0][1::2]
        extents.append((min(xs), min(ys), max(xs), max(ys)))
    return sorted(extents)


def test_instance_masks_merged_across_a_seam_are_one_polygon_per_object_whatever_its_values(
    tmp_path,
):
    """Calls of one footprint wider than any slice, differing in their attribute value: every
    slice's partial masks merge into one polygon spanning the whole object, carrying one value of
    its attribute, since the merge compares the one subject and never its attribute values."""
    _path, checkpoint = _checkpoint(tmp_path, with_masks=True, by_band=True)
    arr = np.zeros((FRAME, FRAME, 3), dtype=np.uint8)
    x0, y0, x1, y1 = WIDE_BLOB
    arr[y0:y1, x0:x1, 0] = 255
    arr[y0:y1, x0:x1, 1] = 255

    result = _sliced(_pass(checkpoint), _png(tmp_path, arr))

    assert result["labels"] == [1]
    assert result["attributes"][0][0] in (0, 1)
    # Contours run through pixel centers, so a blob's polygon ends one pixel inside its box.
    assert _mask_extents(result) == [(x0, y0, x1 - 1, y1 - 1)]


def test_an_untiled_record_carries_the_polygons_the_sliced_record_does(tmp_path):
    _path, checkpoint = _checkpoint(tmp_path, with_masks=True)
    p = _pass(checkpoint)
    path = _png(tmp_path, _frame())

    whole = _whole(p, path)
    sliced = _sliced(p, path)

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

    record = _whole(_pass(checkpoint), path)

    assert [m["segmentation"] for m in record["masks"]] == [expected]


def test_masks_are_cut_at_the_platform_binarize_threshold_the_record_names(tmp_path, monkeypatch):
    import importlib

    _path, checkpoint = _checkpoint(tmp_path, with_masks=True)
    p = _pass(checkpoint)
    path = _png(tmp_path, _frame())
    # The package attribute of this name is a same-named function, so the module comes from
    # sys.modules.
    mask_geometry = importlib.import_module("tcip_mcp.pipelines.measurement.mask_geometry")
    real = mask_geometry.resolve_binarize_threshold

    assert all(m["segmentation"] for m in _whole(p, path)["masks"])
    monkeypatch.setattr(mask_geometry, "resolve_binarize_threshold", lambda *a, **k: real(1.5))
    record = _whole(p, path)
    assert [m["segmentation"] for m in record["masks"]] == [[]] * len(BLOBS)
    assert record["mask_binarize"]["value"] == 1.5


def test_an_untiled_detector_record_carries_no_masks(tmp_path):
    _path, checkpoint = _checkpoint(tmp_path)

    record = _whole(_pass(checkpoint), _png(tmp_path, _frame()))

    assert "masks" not in record and record["count"] == len(BLOBS)


@pytest.mark.parametrize("with_masks", [False, True])
def test_a_windowed_pass_resumed_mid_raster_matches_an_uninterrupted_one(tmp_path, with_masks):
    """The recorded progress carries each slice's shifted predictions whole, so a resumed pass
    merges the seeded predictions, masks included, exactly as the uninterrupted pass does."""
    from tcip_mcp.pipelines.raster_source import open_raster

    _path, checkpoint = _checkpoint(tmp_path, with_masks=with_masks)
    p = _pass(checkpoint)
    npy = tmp_path / "frame.npy"
    np.save(npy, _frame())
    recorded: list[dict] = []

    class PassInterruptedError(Exception):
        pass

    def record_then_stop(_start, _end, batch):
        recorded.append(batch)
        raise PassInterruptedError

    with open_raster(str(npy), 3) as reader:
        uninterrupted = _sliced(p, reader)
        with pytest.raises(PassInterruptedError):
            _sliced(p, reader, progress=record_then_stop)
        prior = {field: [v for b in recorded for v in b[field]]
                 for field in ("slices", "predictions")}
        seen_before = len(p.predictor.model.seen_channels)
        resumed = _sliced(p, reader, prior=prior)

    assert len(prior["slices"]) == 2
    assert len(p.predictor.model.seen_channels) - seen_before == 2
    assert resumed == uninterrupted
    if with_masks:
        assert all(m["segmentation"] for m in resumed["masks"])


def test_training_and_inference_slice_one_frame_on_one_lattice(tmp_path):
    """The tiled training dataset's slices and a sliced inference pass's recorded slices over the
    same frame, each read whole off its own route's record."""
    from tcip_annotation.state import Annotation, BBox

    from tcip_mcp.pipelines.raster_source import open_raster
    from tests._producer_fixtures import dataset_over, label_image

    images = tmp_path / "images" / UNDATED_BUCKET
    images.mkdir(parents=True)
    np.save(images / "frame.npy", _frame())
    label_image(images / "frame.npy", [Annotation(subject="bud", geometry=BBox(*SEAM_BLOB))],
                FRAME, FRAME, keep_empty=True)
    ds = dataset_over("detection", str(images), subject="bud",
                      stated={"num_channels": 3},
                      tiling={"enabled": True, "tile_size": TILE, "overlap": OVERLAP,
                              "sliver_frac": 0.5})  # stated: one box derives no spread

    _path, checkpoint = _checkpoint(tmp_path)
    recorded: list[list[int]] = []
    with open_raster(str(images / "frame.npy"), 3) as reader:
        _sliced(_pass(checkpoint), reader, require_masks=False,
                progress=lambda _s, _e, batch: recorded.extend(batch["slices"]))

    assert [box for _stem, box in ds.tile_entries] == [tuple(s) for s in recorded]


def test_a_postprocess_outside_the_vocabulary_refuses_by_name(tmp_path):
    _path, checkpoint = _checkpoint(tmp_path)

    with pytest.raises(ValueError, match="greedynmm"):
        _pass(checkpoint, postprocess="soft-nms")
    admitted = _pass(checkpoint, postprocess="greedynmm")
    assert len(_sliced(admitted, _png(tmp_path, _frame()))["boxes"]) == len(BLOBS)


def test_the_merge_leaves_the_process_environment_as_it_found_it(tmp_path, monkeypatch):
    """SAHI's torchvision backend selects its device by writing ``CUDA_VISIBLE_DEVICES`` into the
    process, which a training launch's child would inherit; the platform's merge never does."""
    import os

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _path, checkpoint = _checkpoint(tmp_path)

    _sliced(_pass(checkpoint), _png(tmp_path, _frame()), require_masks=False)

    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_a_stated_merge_threshold_never_reaches_the_detectors_own_nms(tmp_path):
    _path, checkpoint = _checkpoint(tmp_path)

    p = _pass(checkpoint, postprocess="nms", cross_tile_nms=0.9)

    assert (p.execution.cross_tile_nms, p.execution.sources["cross_tile_nms"]) == (0.9,
                                                                                   "explicit")
    assert p.predictor.model.nms_thresh == 0.45


def test_each_merge_runs_at_the_threshold_its_own_metric_derives(tmp_path):
    """A box nested in another overlaps it at IoU 0.25 and IoS 1: suppression by IoU derives a
    threshold the pair stays apart under, merging by IoS one it joins under, each merge running at
    its own metric's value and labeled with that metric."""
    from sahi.prediction import ObjectPrediction

    from tcip_mcp.pipelines.derivations import CROSS_TILE_NMS_DERIVATIONS
    from tcip_mcp.pipelines.slicing import cross_tile_merge

    _path, checkpoint = _checkpoint(tmp_path)
    nested = [[[0.0, 0.0, 20.0, 20.0], [5.0, 5.0, 10.0, 10.0]]]
    merged = {}
    for postprocess, metric in (("nms", "IOU"), ("nmm", "IOS")):
        p = _pass(checkpoint, postprocess=postprocess, cross_tile_nms=None)
        p.derive_merge(nested)
        assert p.execution.sources["cross_tile_nms"] == CROSS_TILE_NMS_DERIVATIONS[metric]
        predictions = [ObjectPrediction(bbox=[0, 0, 20, 20], category_id=1, score=0.9),
                       ObjectPrediction(bbox=[5, 5, 15, 15], category_id=1, score=0.8)]
        merged[postprocess] = (p.execution.cross_tile_nms,
                               len(cross_tile_merge(p.execution)(predictions)))

    assert merged["nms"] == (pytest.approx(0.30), 2)
    assert merged["nmm"] == (pytest.approx(0.80), 1)
    assert "provisional" in CROSS_TILE_NMS_DERIVATIONS["IOS"]


def test_the_dry_run_reports_the_record_the_bucket_keeps(tmp_path):
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.tools.inference_tools import run_inference

    ckpt, _checkpoint_record = _checkpoint(tmp_path)
    images = tmp_path / "images" / UNDATED_BUCKET
    images.mkdir(parents=True)
    _png(images, _frame())
    stated = Stated(tile=True, tile_size=TILE, overlap=OVERLAP, conf=0.2,
                    cross_tile_nms=0.4, max_dets=50, postprocess="greedynmm")

    dry = run_inference(tmp_path, ckpt, images_dir=str(images), bucket="sliced",
                        dry_run=True, stated=stated)
    real = run_inference(tmp_path, ckpt, images_dir=str(images), bucket="sliced",
                         stated=stated)

    assert "error" not in dry and "error" not in real, (dry, real)
    assert dry["execution"] == read_bucket(real["dataset_root"], "sliced").execution.record()


N_CALIBRATION_IMAGES = 20


def _blob_capture(project: Path) -> Path:
    """Frames of 20px blobs in three rows of four, each image shifted a pixel further right so no
    two share ground truth, labeled with 32px boxes that overlap their row neighbors by 10px (a
    neighbor tail every metric derives a threshold from), ingested as one capture date of the
    dataset ``ds``; its images directory."""
    from tcip_annotation.state import Annotation, BBox

    from tcip_mcp.tools.ingest_tools import ingest_images
    from tests._producer_fixtures import label_image

    root, raw, date = project / "ds", project / "raw", "2026-01-01"
    raw.mkdir()
    labels = {}
    for i in range(N_CALIBRATION_IMAGES):
        arr = np.zeros((FRAME, FRAME, 3), dtype=np.uint8)
        boxes = []
        for y0 in (20, 80, 140):
            for col in range(4):
                x0 = 40 + col * 22 + i
                arr[y0:y0 + 20, x0:x0 + 20] = 255
                boxes.append(BBox(x0 - 6, y0 - 6, x0 + 26, y0 + 26))
        _png(raw, arr, f"img{i:02d}.png")
        labels[f"img{i:02d}.png"] = [Annotation(subject="bud", geometry=b) for b in boxes]
    ingested = ingest_images(root, source=str(raw), date_from=date)
    assert "error" not in ingested, ingested
    images = root / "images" / date
    for name, boxed in labels.items():
        label_image(images / name, boxed, FRAME, FRAME, keep_empty=True)
    return images


def test_an_assessed_tiled_pass_and_its_bucket_run_one_merge(tmp_path, monkeypatch):
    """A tiled assessment resolves its unstated merge threshold from the calibration side's
    ground truth in the metric its postprocess compares over before any image is predicted; every
    reference slice and every slice of the bucket published under it is merged at that one value,
    and the bucket records the assessment's execution."""
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.pipelines.derivations import CROSS_TILE_NMS_DERIVATIONS
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
    from tcip_mcp.tools.calibration_tools import assess_checkpoint
    from tcip_mcp.tools.data_tools import draw_splits
    from tcip_mcp.tools.inference_tools import run_inference
    from tests import _trait_fixtures as fx

    fx.seed_confirmed_count(tmp_path, measured_subject="bud")
    images = _blob_capture(tmp_path)
    selection = tmp_path / "selection"
    drawn = draw_splits(tmp_path, str(tmp_path / "ds"), output_path=str(selection),
                        subject="bud", seed=2, val_ratio=0.2,
                        calibration_ratio=0.2, holdout_ratio=0.2)
    assert "error" not in drawn, drawn
    ckpt, _record = _checkpoint(tmp_path)
    merged_at: list[float] = []
    real_predict_sliced = GenericPredictor.predict_sliced

    def _capture(self, source, **kwargs):
        merged_at.append(kwargs["execution"].cross_tile_nms)
        return real_predict_sliced(self, source, **kwargs)

    monkeypatch.setattr(GenericPredictor, "predict_sliced", _capture)

    assessment = assess_checkpoint(
        tmp_path, checkpoint_path=ckpt, trait=fx.COUNT_TRAIT, delivery_kind="per_image_count",
        selection_dir=str(selection), device="cpu",
        stated=Stated(tile=True, tile_size=TILE, overlap=OVERLAP, postprocess="nmm"))
    assert "error" not in assessment, assessment
    reference_passes = len(merged_at)
    published = run_inference(tmp_path, ckpt, images_dir=str(images), bucket="blob/2026-01-01",
                              assessment_id=assessment["assessment_id"], device="cpu")
    assert "error" not in published, published

    execution = assessment["execution"]
    assert execution["sources"]["cross_tile_nms"] == CROSS_TILE_NMS_DERIVATIONS["IOS"]
    assert reference_passes > 0 and len(merged_at) > reference_passes
    assert set(merged_at) == {execution["cross_tile_nms"]}
    assert read_bucket(published["dataset_root"], "blob/2026-01-01").execution.record() == (
        execution)


def test_the_bucket_records_the_slice_geometry_the_checkpoint_derived(tmp_path):
    import sahi

    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.tools.inference_tools import run_inference

    ckpt, _checkpoint_record = _checkpoint(tmp_path, tiling={"tile_size": TILE,
                                                             "overlap": OVERLAP})
    images = tmp_path / "images" / UNDATED_BUCKET
    images.mkdir(parents=True)
    _png(images, _frame())

    response = run_inference(tmp_path, ckpt, images_dir=str(images), bucket="sliced",
                             stated=Stated(conf=0.0, postprocess="nmm"))

    assert "error" not in response, response
    execution = read_bucket(response["dataset_root"], "sliced").execution
    assert (execution.tile_size, execution.overlap) == (TILE, OVERLAP)
    assert (execution.postprocess, execution.merge_type, execution.match_metric) == (
        "nmm", "NMM", "IOS")
    assert execution.sahi_version == sahi.__version__
