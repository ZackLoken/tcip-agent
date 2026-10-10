"""Whether a source can be read at the checkpoint's band count is the predictor's own read.

``load_image`` converts any photographic frame to the model's width itself, so a one-channel
checkpoint over ordinary RGB captures is a legitimate pass that must run; a container with no such
coercion is refused by name before any pixels reach the model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")


def _rgb_image(images_dir: Path) -> str:
    from tests._producer_fixtures import write_image

    return str(write_image(images_dir / "capture.png", (120, 90), (40, 80, 120)))


def _five_band_raster(images_dir: Path) -> str:
    path = images_dir / "capture.npy"
    np.save(path, np.zeros((90, 120, 5), dtype=np.uint8))
    return str(path)


def _predicted(tmp_path, images_dir: Path, *, in_chans, builder_kwargs=None, **stated):
    """The pass of a real checkpoint built at ``in_chans`` over ``images_dir``, and its results."""
    from tests._verified_checkpoint_fixtures import (
        built_detector, predicted_over, registered_checkpoint,
    )

    ckpt = registered_checkpoint(tmp_path, model_source=built_detector(**(builder_kwargs or {})),
                                 data={"num_channels": in_chans, "scope": {"subject": "bud"}})
    return predicted_over(tmp_path, str(ckpt), str(images_dir), device="cpu", **stated)


def test_a_source_with_no_coercion_is_refused_at_the_models_own_band_count(tmp_path):
    """A five-band raster under a three-channel checkpoint has no safe coercion, so the read
    refuses by name instead of truncating the bands the model trained on."""
    from tests._verified_checkpoint_fixtures import SAMPLE_OVERLAP

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _five_band_raster(images_dir)
    with pytest.raises(ValueError, match="refusing to silently truncate"):
        _predicted(tmp_path, images_dir, in_chans=3, tile=True, tile_size=64,
                   overlap=SAMPLE_OVERLAP)


def test_a_photographic_source_the_model_reads_is_predicted(tmp_path):
    """The legitimate case: an RGB capture under a one-channel checkpoint is converted by the
    loader itself, so the pass predicts it."""
    from tcip_mcp.pipelines.derivations import band_normalization_stats

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    image = _rgb_image(images_dir)
    mean, std, _read = band_normalization_stats([image], 1)  # this capture's own single-band stats
    _pass, results = _predicted(tmp_path, images_dir, in_chans=1, tile=False,
                                builder_kwargs={"image_mean": mean, "image_std": std})

    assert [Path(r["image"]).name for r in results] == ["capture.png"]
