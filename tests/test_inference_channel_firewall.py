"""A pass reads each source at its checkpoint's band count or refuses it by name.

A one-channel checkpoint over an ordinary RGB capture predicts it, and a five-band raster under
a three-channel checkpoint is refused by name.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tcip_mcp.pipelines.execution import MaskCut, Pass

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")


def _rgb_image(images_dir: Path) -> str:
    from tests._producer_fixtures import BRIGHT, painted_frame

    path = images_dir / "capture.png"
    painted_frame(120, 90, (40, 80, 120), [((10, 10, 40, 40), BRIGHT)]).save(
        path
    )
    return str(path)


def _five_band_raster(images_dir: Path) -> str:
    path = images_dir / "capture.npy"
    np.save(path, np.zeros((90, 120, 5), dtype=np.uint8))
    return str(path)


def _predicted(
    tmp_path: Path,
    images_dir: Path,
    *,
    in_chans: int,
    normalization: dict | None = None,
    **stated: bool | int | float | str | MaskCut | None,
) -> tuple[Pass, list[dict]]:
    """Return the pass of a real checkpoint built at ``in_chans`` and its results.

    The pass runs over ``images_dir``, normalizing by ``normalization`` when given.
    """
    from tests._verified_checkpoint_fixtures import (
        built_detector,
        predicted_over,
        registered_checkpoint,
    )

    source = built_detector()
    if normalization is not None:
        source["normalization"] = normalization
    ckpt = registered_checkpoint(
        tmp_path,
        model_source=source,
        data={"num_channels": in_chans, "scope": {"subject": "bud"}},
    )
    return predicted_over(
        tmp_path, str(ckpt), str(images_dir), device="cpu", **stated
    )


def test_a_source_with_no_coercion_is_refused_at_the_models_own_band_count(
    tmp_path: Path,
) -> None:
    """A source with no coercion to the model's band count is refused by name.

    A five-band raster under a three-channel checkpoint refuses, naming the truncation it
    declines.
    """
    from tcip_mcp.dataset_layout import image_root
    from tests._verified_checkpoint_fixtures import SAMPLE_OVERLAP

    images_dir = image_root(tmp_path)
    images_dir.mkdir()
    _five_band_raster(images_dir)
    with pytest.raises(ValueError, match="refusing to silently truncate"):
        _predicted(
            tmp_path,
            images_dir,
            in_chans=3,
            tile=True,
            tile_size=64,
            overlap=SAMPLE_OVERLAP,
        )


def test_a_photographic_source_the_model_reads_is_predicted(
    tmp_path: Path,
) -> None:
    """An RGB capture under a one-channel checkpoint is predicted."""
    from tcip_mcp.dataset_layout import image_root
    from tcip_mcp.pipelines.derivations import band_normalization_stats

    images_dir = image_root(tmp_path)
    images_dir.mkdir()
    image = _rgb_image(images_dir)
    stats = band_normalization_stats(
        [image], 1, max_images=1, seed=0
    )  # this capture's own single-band stats
    assert stats is not None
    _pass, results = _predicted(
        tmp_path,
        images_dir,
        in_chans=1,
        tile=False,
        normalization=stats.model_dump(),
    )

    assert [Path(r["image"]).name for r in results] == ["capture.png"]
