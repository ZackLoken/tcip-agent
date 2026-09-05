"""``pipelines.calibration.calibrate_operating_point`` must handle a grouped image without crashing:
``label_image_stems``' ``stem_to_image`` can hold a ``BandGroupRef`` (a band-grouped
capture, see ``pipelines.data.band_groups``), and naively ``str()``-ing it produces its
dataclass repr instead of a path any reader could decode. This file exercises a real
``GenericPredictor`` (a tiny real 2-channel detection model, real forward pass) calibrating over
a directory of grouped captures.

The dataset here is two 2-band grouped captures, not a grouped capture mixed with a plain RGB
photo: a real trained-for channel count is one property of the whole dataset a single checkpoint's
``in_chans`` assumes, so a 2-band model has no valid 3-band-photo counterpart in the same
directory anyway: that mismatch belongs to a different scenario (a truly heterogeneous images/
folder), not this crash.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")

TILE = 32


def _write_group(images_dir: Path, stem: str, fill=(111, 222)) -> None:
    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest

    band_a = images_dir / f"{stem}_G.tif"
    band_b = images_dir / f"{stem}_R.tif"
    tifffile.imwrite(str(band_a), np.full((TILE, TILE), fill[0], dtype=np.uint16))
    tifffile.imwrite(str(band_b), np.full((TILE, TILE), fill[1], dtype=np.uint16))
    write_band_group_manifest(images_dir, stem, {"Green": band_a, "Red": band_b})


def _detection_checkpoint(tmp_path: Path) -> str:
    from tcip_mcp.pipelines.model_build import build_model
    from tcip_mcp.tools.model_tools import register_model

    model_source = {
        "builder": "tests.bespoke_models:build_bespoke_detection",
        "builder_kwargs": {
            "num_classes": 1, "in_chans": 2, "min_size": TILE, "max_size": TILE * 2,
            "image_mean": [0.5, 0.5], "image_std": [0.25, 0.25],
        },
        "task": "detection",
    }
    model = build_model({"model_source": model_source})
    ckpt = tmp_path / "model_best.pt"
    torch.save({"model_source": model_source, "model_state_dict": model.state_dict()}, str(ckpt))
    result = register_model(name="band-group-test-model", checkpoint_path=str(ckpt), config={},
                            project_path=str(tmp_path))
    assert "error" not in result, result
    return str(ckpt)


def _grouped_dataset(root: Path) -> tuple[Path, Path]:
    """Two 2-band grouped captures, each with a GT label, a labeled dir every stem of which is a
    ``BandGroupRef``, the shape ``calibrate_operating_point`` hands to ``predict_batch``."""
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    images_dir = root / "images"
    labels_dir = root / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    _write_group(images_dir, "capture_001", fill=(111, 222))
    _write_group(images_dir, "capture_002", fill=(50, 90))

    for stem in ("capture_001", "capture_002"):
        json_io.write_annotations(
            str(labels_dir / f"{stem}.json"),
            [Annotation(subject="bud", geometry=BBox(2, 2, 10, 10))], TILE, TILE, keep_empty=True,
        )
    return images_dir, labels_dir


def test_calibrate_operating_point_over_a_grouped_image_does_not_crash(tmp_path, monkeypatch):
    """A ``BandGroupRef`` (``stem_to_image[stem]``) must decode through the real channel-aware
    stacking, not silently stringify to its dataclass repr, which no reader could open. This
    predictor is 2-channel, so it can only run at all if the grouped captures actually decoded
    that way (a 3-channel predictor could silently "work" on a bad path by
    broadcasting/re-normalizing, masking the bug)."""
    from tcip_mcp.pipelines.calibration import calibrate_operating_point
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor

    from tcip_mcp.model_registry import load_registered_checkpoint

    images_dir, labels_dir = _grouped_dataset(tmp_path)
    ckpt = _detection_checkpoint(tmp_path)
    checkpoint = load_registered_checkpoint(ckpt, project_path=str(tmp_path))
    predictor = GenericPredictor(checkpoint, device="cpu", score_threshold=0.0)

    seen_sources = []
    from tcip_mcp.pipelines import raster_source
    real_open_raster = raster_source.open_raster

    def _spy_open_raster(source, num_channels):
        seen_sources.append(source)
        return real_open_raster(source, num_channels)

    monkeypatch.setattr(raster_source, "open_raster", _spy_open_raster)

    bundle, dataset_hash, n_excluded, _evidence = calibrate_operating_point(
        predictor, "bud_opening", str(labels_dir), str(images_dir),
        tile=False, tile_size=TILE, overlap=0.2, tile_batch_size=8,
        global_nms_iou=0.5, postprocess="nms", cross_tile_nms=None, max_dets=100,
        holdout_ratio=0.0,  # both stems land in calibration -> one predict_batch call sees both
    )

    assert n_excluded == 0
    assert bundle is not None
    assert dataset_hash

    # Both grouped captures really were opened as BandGroupRefs through the channel-aware reading
    # layer, never a stringified stand-in the predictor's own Path(...) would mis-resolve.
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef

    grouped = [s for s in seen_sources if isinstance(s, BandGroupRef)]
    assert {g.stem for g in grouped} == {"capture_001", "capture_002"}


def test_run_inference_images_dir_folds_a_grouped_capture(tmp_path, monkeypatch):
    """run_inference's own images_dir listing fallback (~line 576) must route through
    list_logical_images rather than a bare image_exts scan, or a grouped capture's sibling band
    files each enumerate as their own (spurious) image instead of folding into one. Real forward
    pass, no images_dir mixing (see module docstring)."""
    from tests._verified_checkpoint_fixtures import run_inference_verified

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _write_group(images_dir, "capture_001")
    ckpt = _detection_checkpoint(tmp_path)

    result = run_inference_verified(ckpt, images_dir=str(images_dir), device="cpu", tile=False)

    assert "error" not in result
    assert result["image_count"] == 1  # one grouped capture, never 2 raw sibling band files
    assert result["results"][0]["image"].endswith("capture_001.bandgroup")


def test_calibrate_operating_point_crashes_without_the_fix(tmp_path):
    """Stringifying the BandGroupRef reproduces the crash, against the same real
    predictor/dataset this module's other test proves now works."""
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.data.splits import label_image_stems
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor

    images_dir, labels_dir = _grouped_dataset(tmp_path)
    ckpt = _detection_checkpoint(tmp_path)
    checkpoint = load_registered_checkpoint(ckpt, project_path=str(tmp_path))
    predictor = GenericPredictor(checkpoint, device="cpu", score_threshold=0.0)

    stems, stem_to_image = label_image_stems(str(labels_dir), str(images_dir))
    # The pre-fix call shape: str(stem_to_image[s]) instead of the raw Path|BandGroupRef.
    with pytest.raises(Exception):
        predictor.predict_batch([str(stem_to_image[s]) for s in stems], tile=False)
