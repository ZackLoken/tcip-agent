"""Detector builders: plain ``build_detector`` / ``_build_*`` factories, imported
directly by bespoke model code (no registry)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from torch import Tensor, nn  # noqa: E402

from tests import bespoke_models  # noqa: E402


@pytest.mark.parametrize("detector", ["faster_rcnn", "fcos", "retinanet"])
def test_build_each_builtin_detector(detector):
    model = bespoke_models.build_bespoke_detection(
        num_classes=1, detector=detector, min_size=64, max_size=128)
    model.eval()
    out = model([torch.rand(3, 64, 64)])
    assert isinstance(out, list) and "boxes" in out[0]
    # train-forward: each detector returns a finite loss dict.
    model.train()
    target = [{"boxes": torch.tensor([[10.0, 10.0, 40.0, 40.0]]), "labels": torch.tensor([1])}]
    loss = model([torch.rand(3, 64, 64)], target)
    assert isinstance(loss, dict) and loss
    assert torch.isfinite(sum(loss.values()))


def test_build_detector_unknown_name_raises():
    from tcip_mcp.pipelines.components.detectors import build_detector
    with pytest.raises(KeyError):
        build_detector("does_not_exist", object(), 1, featmap_names=["0"], num_levels=1)


class _LinearPatchBackbone(nn.Module):
    """A patch-embedding stem with no convolution in it: unfold the image into patches and project
    them linearly, the shape a transformer-style backbone takes. Its band count lives in the fan-in
    of a linear layer, which is why reading the first registered ``Conv2d`` cannot recover it."""

    def __init__(self, in_chans: int, out_channels: int, patch: int = 8) -> None:
        super().__init__()
        self.patch = patch
        self.out_channels = out_channels
        self.project = nn.Linear(in_chans * patch * patch, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        patches = nn.functional.unfold(x, kernel_size=self.patch, stride=self.patch)
        batch, _, count = patches.shape
        side = int(count ** 0.5)
        embedded = self.project(patches.transpose(1, 2))
        return embedded.transpose(1, 2).reshape(batch, self.out_channels, side, side)


class _PassThroughNeck(nn.Module):
    """A neck that hands its backbone's single feature map on unchanged, declaring its width."""

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.out_channels = out_channels

    def forward(self, features: Tensor) -> Tensor:
        return features


def _unprobable_adapter(in_chans: int, out_channels: int = 32):
    """The platform's own adapter over a backbone whose band count no probe can read."""
    from tcip_mcp.pipelines.components.detectors import BackboneNeckAdapter, _probe_in_chans

    adapter = BackboneNeckAdapter(
        _LinearPatchBackbone(in_chans, out_channels), _PassThroughNeck(out_channels))
    assert _probe_in_chans(adapter) is None, "this fixture only means something while unprobable"
    return adapter


def test_a_detector_whose_band_count_cannot_be_read_refuses_rather_than_assuming_rgb():
    """With no in_chans stated and no conv to read one from, torchvision would normalize with
    3-element ImageNet statistics over bands nobody has confirmed are RGB. The build refuses by
    name instead."""
    from tcip_mcp.pipelines.components.detectors import build_detector

    with pytest.raises(ValueError, match="cannot tell how many bands"):
        build_detector("faster_rcnn", _unprobable_adapter(5), 1,
                       featmap_names=["0"], num_levels=1, min_size=64, max_size=128)


def test_a_detector_over_that_backbone_builds_from_its_datasets_own_band_statistics(tmp_path):
    """The rail admits valid work: per-band statistics derived from the run's own admitted sources
    build the same detector and reach its transform at the bands they describe, and a caller that
    states the width builds on torchvision's own three-band default."""
    import numpy as np
    import tifffile
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    from tcip_mcp.pipelines.components.detectors import build_detector
    from tcip_mcp.pipelines.derivations import band_normalization_stats
    from tests._producer_fixtures import samples_over

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    for index, stem in enumerate(("a", "b")):
        bands = np.full((32, 32, 5), 20 * (index + 1), dtype=np.uint8)
        bands[..., 4] = 200
        tifffile.imwrite(str(images_dir / f"{stem}.tif"), bands)
        json_io.write_annotations(
            str(labels_dir / f"{stem}.json"),
            [Annotation(subject="leaf", geometry=BBox(4, 4, 20, 20))], 32, 32, keep_empty=True)

    samples = samples_over(str(images_dir), str(labels_dir), subject="leaf")
    derived = band_normalization_stats([s.source for s in samples], 5)
    assert derived is not None
    mean, std, paths_read = derived
    assert len(mean) == len(std) == 5 and len(paths_read) == 2

    model = build_detector("faster_rcnn", _unprobable_adapter(5), 1, featmap_names=["0"],
                           num_levels=1, min_size=64, max_size=128,
                           image_mean=mean, image_std=std)
    assert list(model.transform.image_mean) == mean
    assert list(model.transform.image_std) == std
    model.eval()
    assert "boxes" in model([torch.rand(5, 64, 64)])[0]

    stated = build_detector("faster_rcnn", _unprobable_adapter(3), 1, featmap_names=["0"],
                            num_levels=1, min_size=64, max_size=128, in_chans=3)
    assert len(stated.transform.image_mean) == 3
