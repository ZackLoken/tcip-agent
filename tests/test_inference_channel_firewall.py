"""Whether a source can be read at the checkpoint's band count is the predictor's own read.

``load_image`` converts any photographic frame to the model's width itself, so a one-channel
checkpoint over ordinary RGB captures is a legitimate run that must ship; a container with no such
coercion is refused by name before any pixels reach the model. The calibrated inference door
reports what it predicted rather than judging the bands a second time.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("pycocotools")


def _rgb_image(tmp_path: Path) -> str:
    from PIL import Image

    path = tmp_path / "capture.png"
    Image.new("RGB", (120, 90), color=(40, 80, 120)).save(path)  # a non-square frame
    return str(path)


def _five_band_raster(tmp_path: Path) -> str:
    path = tmp_path / "capture.npy"
    np.save(path, np.zeros((90, 120, 5), dtype=np.uint8))
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


def _run(tmp_path, monkeypatch, *, in_chans, image, builder_kwargs=None, **overrides):
    """A calibrated run of a real checkpoint built at ``in_chans`` over one target image."""
    import tcip_mcp.pipelines.calibration as calibration
    from tests._verified_checkpoint_fixtures import registered_checkpoint, run_inference_verified

    bundle, inputs = _held_out_bundle()
    evidence = {"resolver": "resolve_operating_point", "inputs": inputs,
                "reference_inputs": {"label_dirs": {"calibration": str(tmp_path)}}}
    monkeypatch.setattr(calibration, "calibrate_operating_point",
                        lambda *a, **k: (bundle, "H", 0, evidence))
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    ckpt = registered_checkpoint(tmp_path, project_root=tmp_path, model_source={
        "builder": "tests.bespoke_models:build_bespoke_detection",
        "builder_kwargs": {"num_classes": 1, "in_chans": in_chans, "min_size": 64, "max_size": 128,
                           **(builder_kwargs or {})},
        "task": "detection",
    })
    return run_inference_verified(
        str(ckpt), image_paths=[image], images_dir=str(tmp_path), device="cpu",
        trait="bud_opening", calibration_labels_dir=str(tmp_path), **overrides)


def test_a_source_with_no_coercion_is_refused_at_the_models_own_band_count(tmp_path, monkeypatch):
    """A five-band raster under a three-channel checkpoint has no safe coercion, so the read
    refuses by name instead of truncating the bands the model trained on."""
    with pytest.raises(ValueError, match="refusing to silently truncate"):
        _run(tmp_path, monkeypatch, in_chans=3, image=_five_band_raster(tmp_path),
             tile=True, tile_size=64)


def test_a_photographic_source_the_model_reads_leaves_the_run_shippable(tmp_path, monkeypatch):
    """The legitimate case: an RGB capture under a one-channel checkpoint is converted by the
    loader itself, so the run predicts and its held-out calibration still ships."""
    from tcip_mcp.pipelines.derivations import band_normalization_stats

    image = _rgb_image(tmp_path)
    mean, std, _read = band_normalization_stats([image], 1)  # this capture's own single-band stats
    r = _run(tmp_path, monkeypatch, in_chans=1, image=image, tile=False,
             builder_kwargs={"image_mean": mean, "image_std": std})

    assert "error" not in r, r
    assert r["shippable_issues"] == []
    assert r["validated"] is True
