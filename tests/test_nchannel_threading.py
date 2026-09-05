"""N-channel threading: sensor->in_chans, backbones accept in_chans, a train-start channel
guard, and dataset channel metadata. Multi-channel *readers* are separate; here the plumbing
carries channel counts and validates them."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_backbone_accepts_in_chans():
    import timm

    from tcip_mcp.pipelines.components.backbones import BackboneWrapper

    _m = timm.create_model("resnet50", pretrained=False, features_only=True,
                           out_indices=(1, 2, 3, 4), in_chans=4)
    bb = BackboneWrapper(_m, _m.feature_info.channels())
    out = bb(torch.rand(1, 4, 64, 64))
    assert isinstance(out, dict) and len(out) == 4           # 4 feature stages
    assert bb.out_channels == [256, 512, 1024, 2048]


def test_compose_and_forward_4_channel_model():
    from tests import bespoke_models
    model = bespoke_models.build_bespoke_classifier(num_classes=3, in_chans=4)
    out = model(torch.rand(2, 4, 64, 64))
    assert isinstance(out, dict) and len(out) > 0            # forward runs on 4 channels


def test_train_start_channel_guard():
    from tcip_mcp.pipelines.training.generic_trainer import _validate_input_channels
    with pytest.raises(ValueError, match="channels"):
        _validate_input_channels({"model_source": {"builder": "x:y", "in_chans": 4}},
                                 [(torch.rand(2, 3, 16, 16), {})])
    # matching channel counts -> no error
    _validate_input_channels({"model_source": {"builder": "x:y", "in_chans": 3}},
                             [(torch.rand(2, 3, 16, 16), {})])


def _nchan_adapter(in_chans: int):
    """A detector-ready backbone+neck adapter at ``in_chans`` channels."""
    import timm

    from tcip_mcp.pipelines.components.backbones import BackboneWrapper
    from tcip_mcp.pipelines.components.detectors import BackboneNeckAdapter
    from tcip_mcp.pipelines.components.necks import FPN

    m = timm.create_model("resnet18", pretrained=False, features_only=True,
                          out_indices=(1, 2, 3, 4), in_chans=in_chans)
    bb = BackboneWrapper(m, m.feature_info.channels())
    return BackboneNeckAdapter(bb, FPN(bb.out_channels, 64))


def test_n_channel_detector_builds_and_trains():
    """The documented multispectral claim, exercised on the detection path.

    Every prior N-channel test stopped at a classifier, which is why a detector that cannot
    normalize more than 3 bands survived.
    """
    from tcip_mcp.pipelines.components.detectors import build_detector

    adapter = _nchan_adapter(5)
    names = list(adapter(torch.zeros(1, 5, 64, 64)).keys())
    det = build_detector("faster_rcnn", adapter, num_classes=1, featmap_names=names,
                         num_levels=len(names), image_mean=[0.4] * 5, image_std=[0.2] * 5)
    assert det.transform.image_mean == [0.4] * 5

    det.train()
    losses = det(
        [torch.rand(5, 96, 96)],
        [{"boxes": torch.tensor([[8.0, 8.0, 48.0, 48.0]]), "labels": torch.ones(1, dtype=torch.long)}],
    )
    assert losses and all(torch.isfinite(v) for v in losses.values())


def test_n_channel_detector_without_stats_is_refused_by_default():
    """The refusal fires for an agent who never passed in_chans: the agent it exists for."""
    from tcip_mcp.pipelines.components.detectors import build_detector

    adapter = _nchan_adapter(5)
    names = list(adapter(torch.zeros(1, 5, 64, 64)).keys())
    with pytest.raises(ValueError, match="needs per-band image_mean and image_std of length 5"):
        build_detector("faster_rcnn", adapter, num_classes=1, featmap_names=names,
                       num_levels=len(names))


def test_normalization_length_and_channel_count_are_validated():
    from tcip_mcp.pipelines.components.detectors import build_detector

    adapter = _nchan_adapter(5)
    names = list(adapter(torch.zeros(1, 5, 64, 64)).keys())
    common = {"featmap_names": names, "num_levels": len(names)}
    with pytest.raises(ValueError, match="both must be 5"):
        build_detector("faster_rcnn", adapter, num_classes=1, **common,
                       image_mean=[0.4] * 3, image_std=[0.2] * 3)


def test_explicit_in_chans_wins_over_the_registration_order_probe():
    """The probe is a hint. Only the caller knows the band count, so it may not be overruled.

    An adapter that registers its neck before its backbone, or band-projects N bands through a
    1x1 conv into a pretrained 3-channel stem, reports a first-conv width that is not the image's
    band count. Letting that veto the caller blocks correct builds and admits wrong ones.
    """
    import torch.nn as nn

    from tcip_mcp.pipelines.components.backbones import BackboneWrapper
    from tcip_mcp.pipelines.components.detectors import build_detector
    from tcip_mcp.pipelines.components.necks import FPN

    class NeckFirstAdapter(nn.Module):
        def __init__(self, backbone, neck):
            super().__init__()
            self.neck = neck          # registered first: the probe sees this conv
            self.backbone = backbone
            self.out_channels = neck.out_channels if isinstance(neck.out_channels, int) \
                else neck.out_channels[-1]

        def forward(self, x):
            from collections import OrderedDict
            return OrderedDict(sorted(self.neck(self.backbone(x)).items()))

    import timm

    m = timm.create_model("resnet18", pretrained=False, features_only=True,
                          out_indices=(1, 2, 3, 4), in_chans=3)
    bb = BackboneWrapper(m, m.feature_info.channels())
    adapter = NeckFirstAdapter(bb, FPN(bb.out_channels, 64))
    names = list(adapter(torch.zeros(1, 3, 64, 64)).keys())

    # A plain RGB build must succeed even though the probe reports the neck's width.
    det = build_detector("faster_rcnn", adapter, num_classes=1, featmap_names=names,
                         num_levels=len(names), in_chans=3)
    assert det.transform.image_mean == [0.485, 0.456, 0.406]


def test_three_channel_detector_still_uses_torchvision_defaults():
    """Strengthening the N-channel path must not make the ordinary RGB build require stats."""
    from tcip_mcp.pipelines.components.detectors import build_detector

    adapter = _nchan_adapter(3)
    names = list(adapter(torch.zeros(1, 3, 64, 64)).keys())
    det = build_detector("faster_rcnn", adapter, num_classes=1, featmap_names=names,
                         num_levels=len(names))
    assert det.transform.image_mean == [0.485, 0.456, 0.406]


def test_torchvision_constructor_kwargs_forward_but_typos_raise():
    from tcip_mcp.pipelines.components.detectors import build_detector

    adapter = _nchan_adapter(3)
    names = list(adapter(torch.zeros(1, 3, 64, 64)).keys())
    common = {"featmap_names": names, "num_levels": len(names)}
    det = build_detector("faster_rcnn", adapter, num_classes=1, **common, box_score_thresh=0.42)
    assert det.roi_heads.score_thresh == 0.42
    with pytest.raises(TypeError, match="box_score_thresholdd"):
        build_detector("faster_rcnn", adapter, num_classes=1, **common, box_score_thresholdd=0.42)


@pytest.mark.parametrize("dtype,scale", [
    ("uint8", 255), ("uint16", 65535), ("float32_unit", 1), ("float32_large", 4000),
])
def test_band_normalization_stats_matches_the_tensors_the_model_is_fed(tmp_path, dtype, scale):
    """The statistic must be of the tensor the loader yields, not of a scale re-derived here.

    `pil_to_tensor` decides [0,1] scaling from dtype. A derivation that decides from pixel values
    lands in a different unit system silently, and worst on the float rasters multispectral
    imagery actually ships as.
    """
    import numpy as np

    from tcip_mcp.pipelines.derivations import band_normalization_stats
    from tcip_mcp.pipelines.image_utils import load_image, pil_to_tensor

    paths = []
    for i in range(3):
        if dtype.startswith("float"):
            arr = (np.random.rand(8, 8, 5) * scale).astype(np.float32)
        else:
            arr = (np.random.rand(8, 8, 5) * scale).astype(dtype)
        p = tmp_path / f"r{i}.npy"
        np.save(p, arr)
        paths.append(p)

    mean, std, paths_read = band_normalization_stats(paths, 5)
    assert sorted(paths_read) == sorted(str(p) for p in paths)
    actual = torch.stack([pil_to_tensor(load_image(p, 5)) for p in paths])  # [N, C, H, W]
    expected_mean = actual.permute(1, 0, 2, 3).reshape(5, -1).mean(dim=1)
    expected_std = actual.permute(1, 0, 2, 3).reshape(5, -1).std(dim=1, unbiased=False)
    assert torch.allclose(torch.tensor(mean, dtype=torch.float32), expected_mean, atol=1e-4)
    assert torch.allclose(torch.tensor(std, dtype=torch.float32), expected_std, atol=1e-4)


def test_band_normalization_stats_admits_underivable(tmp_path):
    from tcip_mcp.pipelines.derivations import band_normalization_stats

    # Underivable is None, never a stand-in constant.
    assert band_normalization_stats([tmp_path / "missing.npy"], 5) is None


def _strip_tiff_raster(path, *, height=24, width=20, channels=4):
    """A small multi-band strip raster both band-normalization siblings can read: the exact one by
    decoding it whole, the sampled one by reading pixel windows out of it."""
    import numpy as np
    import tifffile

    arr = np.random.default_rng(5).integers(0, 255, size=(height, width, channels), dtype=np.uint8)
    tifffile.imwrite(str(path), arr, rowsperstrip=6)
    return arr


def test_sampled_band_normalization_stats_over_full_coverage_match_the_exact_sibling(tmp_path):
    """Reading every window of the grid is reading every pixel, so the sampled sibling lands on the
    exact sibling's own numbers: the two differ in which pixels they read, never in the arithmetic
    or the [0, 1] unit system they read them into.
    """
    from tcip_mcp.pipelines.derivations import (
        band_normalization_stats,
        band_normalization_stats_sampled,
    )

    path = tmp_path / "mosaic.tif"
    arr = _strip_tiff_raster(path)

    exact = band_normalization_stats([path], arr.shape[-1])
    sampled = band_normalization_stats_sampled(
        [path], arr.shape[-1], seed=1, window_size=8, max_windows_per_image=999)
    assert exact is not None and sampled is not None
    assert sampled.sampling.pixel_fraction == 1.0
    assert sampled.mean == pytest.approx(exact[0], rel=1e-12, abs=1e-12)
    assert sampled.std == pytest.approx(exact[1], rel=1e-12, abs=1e-12)


def test_sampled_band_normalization_stats_records_the_windows_it_read(tmp_path):
    from tcip_mcp.pipelines.derivations import band_normalization_stats_sampled

    path = tmp_path / "mosaic.tif"
    arr = _strip_tiff_raster(path)
    kwargs = {"seed": 4, "window_size": 8, "max_windows_per_image": 3}

    sampled = band_normalization_stats_sampled([path], arr.shape[-1], **kwargs)
    assert sampled is not None
    assert len(sampled.sampling.windows) == 3
    assert {label for label, _ in sampled.sampling.windows} == {str(path)}
    covered = sum(r.width * r.height for _, r in sampled.sampling.windows)
    assert sampled.sampling.pixel_fraction == pytest.approx(
        covered / (arr.shape[0] * arr.shape[1]))
    assert sampled.sampling.seed == 4

    repeat = band_normalization_stats_sampled([path], arr.shape[-1], **kwargs)
    assert repeat == sampled


def test_sampled_band_normalization_stats_admit_underivable(tmp_path):
    from tcip_mcp.pipelines.derivations import band_normalization_stats_sampled

    assert band_normalization_stats_sampled(
        [tmp_path / "missing.npy"], 5, seed=1, window_size=8, max_windows_per_image=2) is None


def test_build_dataset_sets_expected_channels(tmp_path):
    from PIL import Image

    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.pipelines.data.datasets import build_dataset
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    Image.new("RGB", (16, 16)).save(images_dir / "a.png")
    json_io.write_annotations(str(labels_dir / "a.json"),
                              [Annotation(subject="bud", geometry=BBox(6.4, 6.4, 9.6, 9.6))],
                              16, 16, keep_empty=True)

    ds = build_dataset("detection", images_dir=str(images_dir), labels_dir=str(labels_dir),
                       subject="bud", num_channels=4)
    assert ds.expected_channels == 4
