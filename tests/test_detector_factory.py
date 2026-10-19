"""Detector builders: plain ``build_detector`` / ``_build_*`` factories.

Bespoke model code imports them directly (no registry).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from torch import Tensor, nn  # noqa: E402

from tcip_mcp.dataset_layout import UNDATED_BUCKET  # noqa: E402
from tests import bespoke_models  # noqa: E402
from tests._chain_fixtures import (  # noqa: E402
    detector_geometry,
    detector_values,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tcip_mcp.pipelines.components.detectors import BackboneNeckAdapter


@pytest.mark.parametrize("detector", ["faster_rcnn", "fcos", "retinanet"])
def test_build_each_builtin_detector(detector: str) -> None:
    model = bespoke_models.build_bespoke_detection(
        num_classes=1, detector=detector, **detector_values()
    )
    model.eval()
    out = model([torch.rand(3, 64, 64)])
    assert isinstance(out, list) and "boxes" in out[0]
    # train-forward: each detector returns a finite loss dict.
    model.train()
    target = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 40.0, 40.0]]),
            "labels": torch.tensor([1]),
        }
    ]
    loss = model([torch.rand(3, 64, 64)], target)
    assert isinstance(loss, dict) and loss
    assert torch.isfinite(sum(loss.values()))


def test_build_detector_unknown_name_raises() -> None:
    from tcip_mcp.pipelines.components.detectors import build_detector

    with pytest.raises(ValueError, match="Unknown detector 'does_not_exist'"):
        build_detector(
            "does_not_exist", object(), 1, featmap_names=["0"], num_levels=1
        )


class _LinearPatchBackbone(nn.Module):
    """A patch-embedding stem with no convolution in it.

    It unfolds the image into patches and projects them linearly, the shape a transformer-style
    backbone takes. Its band count lives in the fan-in of a linear layer.
    """

    def __init__(
        self, in_chans: int, out_channels: int, patch: int = 8
    ) -> None:
        super().__init__()
        self.patch = patch
        self.out_channels = out_channels
        self.project = nn.Linear(in_chans * patch * patch, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        patches = nn.functional.unfold(
            x, kernel_size=self.patch, stride=self.patch
        )
        batch, _, count = patches.shape
        side = int(count**0.5)
        embedded = self.project(patches.transpose(1, 2))
        return embedded.transpose(1, 2).reshape(
            batch, self.out_channels, side, side
        )


class _PassThroughNeck(nn.Module):
    """A neck that hands its backbone's single feature map on unchanged, declaring its width."""

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.out_channels = out_channels

    def forward(self, features: Tensor) -> Tensor:
        return features


def _unprobable_adapter(
    in_chans: int, out_channels: int = 32
) -> BackboneNeckAdapter:
    """Return a ``BackboneNeckAdapter`` over a linear patch backbone, asserting no probed width."""
    from tcip_mcp.pipelines.components.detectors import (
        BackboneNeckAdapter,
        _probe_in_chans,
    )

    adapter = BackboneNeckAdapter(
        _LinearPatchBackbone(in_chans, out_channels),
        _PassThroughNeck(out_channels),
    )
    assert _probe_in_chans(adapter) is None, (
        "this fixture only means something while unprobable"
    )
    return adapter


def test_a_detector_stating_no_statistics_is_refused_by_its_signature() -> (
    None
):
    """A detector stating no normalization statistics is refused by the builder's signature.

    A build stating no statistics over a five-band backbone refuses naming ``image_mean``.
    """
    from tcip_mcp.pipelines.components.detectors import build_detector

    with pytest.raises(TypeError, match="image_mean"):
        build_detector(
            "faster_rcnn",
            _unprobable_adapter(5),
            1,
            featmap_names=["0"],
            num_levels=1,
            **detector_geometry().handed(),
        )


def test_a_detector_over_that_backbone_builds_from_its_datasets_own_band_statistics(
    tmp_path: Path,
) -> None:
    """A detector over an unprobable backbone builds from its dataset's own band statistics.

    Per-band statistics derived from the run's own admitted sources build the detector and
    reach its transform at the bands they describe, and a caller that states the width builds at
    the three-band statistics it states.
    """
    import numpy as np
    import tifffile

    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.pipelines.components.detectors import build_detector
    from tcip_mcp.pipelines.derivations import band_normalization_stats
    from tests._producer_fixtures import label_image, samples_over

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    for index, stem in enumerate(("a", "b")):
        bands = np.full((32, 32, 5), 20 * (index + 1), dtype=np.uint8)
        bands[..., 4] = np.arange(32, dtype=np.uint8)[None, :] + 150
        tifffile.imwrite(str(images_dir / f"{stem}.tif"), bands)
        label_image(
            images_dir / f"{stem}.tif",
            [Annotation(subject="leaf", geometry=BBox(4, 4, 20, 20))],
            32,
            32,
        )

    samples = samples_over(str(images_dir), subject="leaf")
    derived = band_normalization_stats(
        [s.source for s in samples], 5, max_images=2, seed=0
    )
    assert derived is not None
    mean, std = derived.image_mean, derived.image_std
    assert len(mean) == len(std) == 5
    assert len(getattr(derived.source, "windows", ())) == 2

    model = build_detector(
        "faster_rcnn",
        _unprobable_adapter(5),
        1,
        featmap_names=["0"],
        num_levels=1,
        image_mean=mean,
        image_std=std,
        **detector_geometry().handed(),
    )
    assert list(model.transform.image_mean) == mean
    assert list(model.transform.image_std) == std
    model.eval()
    assert "boxes" in model([torch.rand(5, 64, 64)])[0]

    stated = build_detector(
        "faster_rcnn",
        _unprobable_adapter(3),
        1,
        featmap_names=["0"],
        num_levels=1,
        in_chans=3,
        **detector_values(),
    )
    assert len(stated.transform.image_mean) == 3
