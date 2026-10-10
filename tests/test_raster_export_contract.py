"""What a whole-raster bucket records has to be what the pass actually ran at."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("tifffile")

TILE = 32


def test_the_raster_door_records_the_execution_its_prepared_pass_states(tmp_path):
    """The raster door prepares its pass through the one preparation every pass shares, so the
    execution record its bucket keeps is the one that preparation states for the same checkpoint
    and the same stated values."""
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.execution import Stated, prepare
    from tcip_mcp.tools.inference_tools import run_inference
    from tests._verified_checkpoint_fixtures import SAMPLE_CROSS_TILE_NMS
    from tests.test_block_calibration import _build_experiment

    exp = _build_experiment(tmp_path)
    stated = {"cross_tile_nms": SAMPLE_CROSS_TILE_NMS}
    result = run_inference(
        tmp_path, exp["checkpoint_path"], bucket="preds/2026-01-01",
        raster_path=str(exp["raster_path"]),
        stated=Stated(conf=0.0, tile_size=TILE, overlap=0.2, **stated))
    assert "error" not in result, result

    prepared = prepare(
        load_registered_checkpoint(exp["checkpoint_path"], project=tmp_path),
        Stated(tile=True, tile_size=TILE, overlap=0.2, conf=0.0, **stated)).runnable()
    assert result["execution"] == prepared.execution.record()
    bucket = read_bucket(result["dataset_root"], result["bucket"])
    assert bucket.execution.record() == prepared.execution.record()


def test_a_raster_pass_records_the_frame_its_predictions_are_in(tmp_path):
    """A TIFF whose stored axes read as one frame at its own band count and another at the
    model's is identified and predicted through the one reader the pass opens, so the identity
    its bucket records and the prediction document carry one frame."""
    import numpy as np
    import tifffile
    from tcip_annotation import json_io

    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.dataset_layout import prediction_key
    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.tools.inference_tools import run_inference
    from tests._verified_checkpoint_fixtures import SAMPLE_CROSS_TILE_NMS
    from tests.test_block_calibration import _build_experiment

    exp = _build_experiment(tmp_path)
    odd = exp["images_dir"] / "odd.tif"
    tifffile.imwrite(str(odd), np.arange(3 * 64 * 4, dtype=np.uint8).reshape(3, 64, 4))
    result = run_inference(
        tmp_path, exp["checkpoint_path"], bucket="odd/2026-01-01", raster_path=str(odd),
        stated=Stated(conf=0.0, tile_size=TILE, overlap=0.2,
                      cross_tile_nms=SAMPLE_CROSS_TILE_NMS))
    assert "error" not in result, result

    bucket = read_bucket(result["dataset_root"], result["bucket"])
    document = json_io.read_label_document(
        prediction_key(result["dataset_root"], result["bucket"], "odd"))
    identity = bucket.raster_identity
    assert identity is not None
    assert (identity["width"], identity["height"]) == (document.width, document.height)
