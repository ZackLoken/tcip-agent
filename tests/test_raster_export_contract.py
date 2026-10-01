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
    from tcip_mcp.pipelines.execution import Stated, prepare_pass
    from tcip_mcp.tools.inference_tools import run_inference
    from tests.test_block_calibration import _build_experiment

    exp = _build_experiment(tmp_path)
    out = tmp_path / "preds"
    result = run_inference(
        tmp_path, exp["checkpoint_path"], output_dir=str(out),
        raster_path=str(exp["raster_path"]), stated=Stated(conf=0.0, tile_size=TILE, overlap=0.2))
    assert "error" not in result, result

    prepared = prepare_pass(
        load_registered_checkpoint(exp["checkpoint_path"], project=tmp_path),
        Stated(tile=True, tile_size=TILE, overlap=0.2, conf=0.0))
    assert result["execution"] == prepared.execution.record()
    assert read_bucket(out).execution.record() == prepared.execution.record()
