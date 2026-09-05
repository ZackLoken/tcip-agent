"""A calibrated run compares the checkpoint's own band count against the imagery it is fed.

The loader will happily coerce a raster whose band count differs from the one the model was
trained at, so the only place a channel-wrong inference becomes visible is the run's own
provenance: the firewall must judge the checkpoint against the probed target, never the target
against itself, and it must leave a legitimate same-channel run untouched.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")

torch = pytest.importorskip("torch")
pytest.importorskip("pycocotools")

PROBED_BANDS = 3  # an ordinary RGB capture, what the target images below actually carry


class _ChannelStub:
    """A predictor carrying the checkpoint's own ``in_chans``, returning one detection per image."""

    def __init__(self, in_chans: int) -> None:
        self.in_chans = in_chans
        self.model = SimpleNamespace(score_thresh=0.5, nms_thresh=0.5, detections_per_img=100)
        self.device = "cpu"
        self.score_threshold = 0.5
        self.train_tile_size = None
        self.train_overlap = None

    def predict_batch(self, paths, **kw):
        return [{"image": p, "width": 120, "height": 90,
                 "boxes": [[10, 10, 30, 30]], "scores": [0.9], "labels": [1], "count": 1}
                for p in paths]


def _rgb_image(tmp_path):
    from PIL import Image

    path = tmp_path / "capture.png"
    Image.new("RGB", (120, 90), color=(40, 80, 120)).save(path)  # a non-square frame
    return str(path)


def _held_out_bundle():
    """A conf resolved from a dense reference that passes its own held-out gate, with the
    resolver arguments behind it."""
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    from tests._dense_op_fixtures import dense_records

    n_images, objects_per_image = 20, 80
    miss, fp = [0] * n_images, [1] * n_images
    inputs = {
        "dataset_hash": "H",
        "calibration_records": dense_records(
            n_images=n_images, objects_per_image=objects_per_image, id_prefix="c",
            miss_pattern=miss, fp_pattern=fp, score=0.9, fp_score=0.05),
        "holdout_records": dense_records(
            n_images=n_images, objects_per_image=objects_per_image, id_prefix="h", shift=5.0,
            miss_pattern=miss, fp_pattern=fp, score=0.9, fp_score=0.05),
        "tiled": False, "staged_conf_floor": 0.01,
    }
    return resolve_operating_point("bud_opening", experiment_id=None, **inputs), inputs


def _run(tmp_path, monkeypatch, in_chans):
    import tcip_mcp.pipelines.calibration as calibration
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    from tests._verified_checkpoint_fixtures import registered_checkpoint, run_inference_verified

    bundle, inputs = _held_out_bundle()
    evidence = {"resolver": "resolve_operating_point", "inputs": inputs,
                "reference_inputs": {"label_dirs": {"calibration": str(tmp_path)}}}
    monkeypatch.setattr(calibration, "calibrate_operating_point",
                        lambda *a, **k: (bundle, "H", 0, evidence))
    monkeypatch.setattr(predictor_mod, "build_predictor",
                        lambda checkpoint, **kw: _ChannelStub(in_chans))
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    ckpt = registered_checkpoint(tmp_path, project_root=tmp_path)
    return run_inference_verified(
        str(ckpt), image_paths=[_rgb_image(tmp_path)], images_dir=str(tmp_path), device="cpu",
        tile=False, trait="bud_opening", calibration_labels_dir=str(tmp_path))


@pytest.mark.parametrize("checkpoint_channels", [5, 1])
def test_a_channel_wrong_run_is_reported_against_the_checkpoints_own_band_count(
    tmp_path, monkeypatch, checkpoint_channels,
):
    """A checkpoint trained at a band count the target imagery does not carry, in either
    direction, surfaces as a named issue and never ships as validated."""
    r = _run(tmp_path, monkeypatch, checkpoint_channels)

    assert "error" not in r, r
    expected = f"in_chans={checkpoint_channels} != probed raster bands={PROBED_BANDS}"
    assert any(expected in issue for issue in r["shippable_issues"]), r["shippable_issues"]
    assert r["validated"] is False


def test_a_matching_band_count_leaves_a_calibrated_run_shippable(tmp_path, monkeypatch):
    """The firewall admits the legitimate case: a checkpoint whose band count equals the probed
    imagery's raises nothing and the held-out calibration still ships."""
    r = _run(tmp_path, monkeypatch, PROBED_BANDS)

    assert "error" not in r, r
    assert r["shippable_issues"] == []
    assert r["validated"] is True
