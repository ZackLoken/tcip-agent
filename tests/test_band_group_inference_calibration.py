"""A band-grouped capture decodes through the channel-aware reading layer wherever a pass predicts
it: the assessment's evaluation records over grouped samples, ``run_inference`` over a grouped
directory, and a stringified ``BandGroupRef`` refused rather than read as a path. A real
``GenericPredictor`` (a tiny 2-channel detection model, real forward pass) runs each.

The dataset is two 2-band grouped captures, not a grouped capture beside a plain RGB photo: a
checkpoint's channel count is one property of the whole dataset, so a 2-band model has no valid
3-band counterpart in the same directory.
"""

from __future__ import annotations

from tcip_mcp.dataset_layout import UNDATED_BUCKET

from pathlib import Path

import numpy as np
import pytest
import tifffile

torch = pytest.importorskip("torch")

from tcip_mcp.pipelines.model_build import CONFIG_KEY, STATE_DICT_KEY  # noqa: E402
from tests._chain_fixtures import BESPOKE_DETECTION  # noqa: E402
from tests._verified_checkpoint_fixtures import SAMPLE_MAX_DETS  # noqa: E402
pytest.importorskip("torchvision")

TILE = 32


def _write_group(images_dir: Path, stem: str, fill=(111, 222)) -> None:
    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest

    band_a = images_dir / f"{stem}_G.tif"
    band_b = images_dir / f"{stem}_R.tif"
    tifffile.imwrite(str(band_a), np.full((TILE, TILE), fill[0], dtype=np.uint16))
    tifffile.imwrite(str(band_b), np.full((TILE, TILE), fill[1], dtype=np.uint16))
    write_band_group_manifest(images_dir, stem, {"Green": band_a, "Red": band_b})


def _detection_checkpoint(tmp_path: Path) -> str:
    from tcip_mcp.pipelines.model_build import build_model, recorded_model_dims
    from tcip_mcp.tools.model_tools import register_model

    model_source = {
        "builder": BESPOKE_DETECTION,
        "builder_kwargs": {
            "min_size": TILE, "max_size": TILE * 2,
            "image_mean": [0.5, 0.5], "image_std": [0.25, 0.25],
        },
        "task": "detection",
    }
    config = {"model_source": model_source,
              "data": {"num_channels": 2, "scope": {"subject": "bud", "attributes": []}}}
    model = build_model(config, recorded_model_dims(config))
    ckpt = tmp_path / "model_best.pt"
    torch.save({STATE_DICT_KEY: model.state_dict(), CONFIG_KEY: config}, str(ckpt))
    result = register_model(tmp_path, name="band-group-test-model", checkpoint_path=str(ckpt),
                            config={})
    assert "error" not in result, result
    return str(ckpt)


def _grouped_dataset(root: Path) -> Path:
    """Two 2-band grouped captures, each with a ground-truth label document; their images
    directory."""
    from tcip_annotation.state import Annotation, BBox

    from tests._producer_fixtures import label_image

    images_dir = root / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)

    _write_group(images_dir, "capture_001", fill=(111, 222))
    _write_group(images_dir, "capture_002", fill=(50, 90))

    for stem in ("capture_001", "capture_002"):
        label_image(images_dir / f"{stem}.bandgroup",
                    [Annotation(subject="bud", geometry=BBox(2, 2, 10, 10))], TILE, TILE,
                    keep_empty=True)
    return images_dir


def test_the_assessments_records_over_grouped_samples_decode_each_capture(tmp_path, monkeypatch):
    """Each grouped sample is opened as a ``BandGroupRef`` through the channel-aware stacking.
    The predictor is 2-channel, so it runs at all only if the captures decoded that way."""
    from tcip_mcp.assessment import _records, _reference_dataset
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines import raster_source
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.data.selection import source_digests
    from tcip_mcp.pipelines.execution import Stated, prepare
    from tests._producer_fixtures import samples_over

    images_dir = _grouped_dataset(tmp_path)
    checkpoint = load_registered_checkpoint(_detection_checkpoint(tmp_path), project=tmp_path)
    p = prepare(checkpoint,
                Stated(tile=False, conf=0.0, max_dets=SAMPLE_MAX_DETS, postprocess="nms"),
                device="cpu", tile_batch_size=8).runnable()
    samples = samples_over(images_dir, subject="bud")

    seen_sources = []
    real_open_raster = raster_source.open_raster

    def _spy_open_raster(source, num_channels):
        seen_sources.append(source)
        return real_open_raster(source, num_channels)

    monkeypatch.setattr(raster_source, "open_raster", _spy_open_raster)

    records = _records(p, _reference_dataset(p, samples), source_digests(samples), p.execution)

    assert len(records) == 2
    grouped = [s for s in seen_sources if isinstance(s, BandGroupRef)]
    assert {g.stem for g in grouped} == {"capture_001", "capture_002"}


def test_run_inference_images_dir_folds_a_grouped_capture(tmp_path):
    """A grouped capture's sibling band files fold into one logical image rather than each
    enumerating as its own."""
    from tests._verified_checkpoint_fixtures import predicted_over

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    _write_group(images_dir, "capture_001")
    ckpt = _detection_checkpoint(tmp_path)

    _pass, results = predicted_over(tmp_path, ckpt, str(images_dir), device="cpu", tile=False)

    assert len(results) == 1
    assert results[0]["image"].endswith("capture_001.bandgroup")


def test_predict_batch_rejects_stringified_band_group_refs(tmp_path):
    """Stringifying a ``BandGroupRef`` before calling ``predict_batch``, instead of passing the
    raw reference, raises."""
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.execution import Stated, prepare
    from tcip_mcp.pipelines.image_utils import list_logical_images

    images_dir = _grouped_dataset(tmp_path)
    checkpoint = load_registered_checkpoint(_detection_checkpoint(tmp_path), project=tmp_path)
    p = prepare(checkpoint, Stated(tile=False, conf=0.0, max_dets=SAMPLE_MAX_DETS),
                device="cpu").runnable()

    logical = list_logical_images(images_dir)
    with pytest.raises(ValueError):
        p.predict([repr(logical[s]) for s in sorted(logical)])
