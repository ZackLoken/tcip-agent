"""What a whole-raster export promises in its sidecar has to be what the pass actually ran at.

Two properties of the ``raster_path`` regime that a reader of the delivered bucket depends on:
the full-frame detection cap recorded beside the predictions is the cap the mosaic pass applied,
and a trait export the checkpoint's training run cannot support refuses with the audited door that
can deliver that count named in the refusal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("tifffile")

pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")

TILE = 32


def test_the_whole_mosaic_pass_runs_at_the_cap_its_sidecar_records(tmp_path, monkeypatch):
    """Block calibration's band-scoped density cap is not transferred to the whole-mosaic pass: the
    pass runs uncapped, and the persisted operating point records that same uncapped value, so the
    recorded operating point and the applied one are one fact rather than two."""
    from tests.test_block_calibration import _attest_regions_complete, _build_experiment

    exp = _build_experiment(tmp_path)
    manifest = exp["spatial_manifest"]
    _attest_regions_complete(
        exp["root"], exp["stem"], [manifest["calibration_region"], manifest["test_region"]])

    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor

    real_predict_tiled = GenericPredictor.predict_tiled
    passes: list[dict] = []

    def _capture_predict_tiled(self, source, **kwargs):
        result = real_predict_tiled(self, source, **kwargs)
        passes.append({"max_dets": self.max_dets, "count": result.get("count")})
        return result

    monkeypatch.setattr(GenericPredictor, "predict_tiled", _capture_predict_tiled)

    from tcip_mcp.tools.inference_tools import run_inference

    out_dir = tmp_path / "preds"
    result = run_inference(
        exp["checkpoint_path"], output_dir=str(out_dir), raster_path=str(exp["raster_path"]),
        conf_threshold=0.0, tile_size=TILE, overlap=0.2, trait="bud_opening",
        experiment_id=exp["experiment_id"])

    assert "error" not in result, result
    assert result["conf_source"] == "block_calibration"
    # The band passes block calibration runs come first; the last one is the mosaic pass itself.
    assert len(passes) > 1
    mosaic_pass = passes[-1]

    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

    sidecar = read_operating_point_sidecar(out_dir)
    stamped_cap = sidecar["operating_point"]["max_dets"]["value"]
    assert mosaic_pass["max_dets"] == stamped_cap
    assert stamped_cap is None
    # The band passes did carry a cap, so an uncapped mosaic pass is a real transition, not the
    # same state the predictor was already in.
    assert any(p["max_dets"] is not None for p in passes[:-1])


def test_a_raster_trait_export_with_no_reserved_region_names_the_audited_delivery_door(tmp_path):
    """The refusal is the agent's only in-code pointer to the door that can still deliver a
    calibrated per-plant count for a mosaic, so it names that tool rather than only the missing
    training-time precondition."""
    from tests.test_block_calibration import _build_experiment

    exp = _build_experiment(tmp_path, reserve_frac=0.0, experiment_id="exp_no_reserved_region")

    from tcip_mcp.tools.inference_tools import run_inference

    out_dir = tmp_path / "preds"
    result = run_inference(
        exp["checkpoint_path"], output_dir=str(out_dir), raster_path=str(exp["raster_path"]),
        conf_threshold=0.0, tile_size=TILE, overlap=0.2, trait="bud_opening",
        experiment_id=exp["experiment_id"])

    assert "error" in result
    assert "deliver_orthomosaic_plant_counts" in result["error"]
    assert not Path(out_dir / "operating_point.json").exists()
