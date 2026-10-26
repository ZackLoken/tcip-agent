"""Tests for N-channel threading through the training and inference plumbing.

Covered: sensor->in_chans, backbones accept in_chans, a train-start channel guard, and dataset
channel metadata. The plumbing carries channel counts and validates them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir
from tests._chain_fixtures import (
    CLASSIFIER_SOURCE,
    detector_geometry,
    training_config,
    weights_statistics,
)
from tests._producer_fixtures import dataset_over  # noqa: E402
from tests._training_values import RATIONALE, drawn_split  # noqa: E402

if TYPE_CHECKING:
    from collections import OrderedDict
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray
    from torch import Tensor
    from torch.utils.data import Dataset

    from tcip_mcp.pipelines.components.detectors import BackboneNeckAdapter
    from tcip_mcp.pipelines.schemas import ImageStatsSampling, Normalization

torch = pytest.importorskip("torch")

pytest.importorskip("torchvision")


def test_backbone_accepts_in_chans() -> None:
    import timm

    from tcip_mcp.pipelines.components.backbones import BackboneWrapper

    _m = timm.create_model(
        "resnet50",
        pretrained=False,
        features_only=True,
        out_indices=(1, 2, 3, 4),
        in_chans=4,
    )
    bb = BackboneWrapper(_m, _m.feature_info.channels())
    out = bb(torch.rand(1, 4, 64, 64))
    assert isinstance(out, dict) and len(out) == 4  # 4 feature stages
    assert bb.out_channels == [256, 512, 1024, 2048]


def test_compose_and_forward_4_channel_model() -> None:
    from tests import bespoke_models
    from tests._chain_fixtures import band_statistics

    model = bespoke_models.build_bespoke_classifier(
        num_classes=3, in_chans=4, **band_statistics(4)
    )
    out = model(torch.rand(2, 4, 64, 64))
    assert isinstance(out, dict) and len(out) > 0  # forward runs on 4 channels


def test_train_start_channel_guard() -> None:
    """A first batch at a width other than the run's recorded one refuses; a matching one passes."""
    from tcip_mcp.pipelines.training.generic_trainer import (
        _validate_input_channels,
    )

    with pytest.raises(ValueError, match="channels"):
        _validate_input_channels(4, [(torch.rand(2, 3, 16, 16), {})])
    # matching channel counts -> no error
    _validate_input_channels(3, [(torch.rand(2, 3, 16, 16), {})])


def _nchan_adapter(in_chans: int) -> BackboneNeckAdapter:
    """Return a ``BackboneNeckAdapter`` over a timm resnet18 at ``in_chans`` and a 64-wide FPN.

    The backbone is ``timm.create_model("resnet18", features_only=True)`` without pretrained
    weights, wrapped in ``BackboneWrapper``.
    """
    import timm

    from tcip_mcp.pipelines.components.backbones import BackboneWrapper
    from tcip_mcp.pipelines.components.detectors import BackboneNeckAdapter
    from tcip_mcp.pipelines.components.necks import FPN

    m = timm.create_model(
        "resnet18",
        pretrained=False,
        features_only=True,
        out_indices=(1, 2, 3, 4),
        in_chans=in_chans,
    )
    bb = BackboneWrapper(m, m.feature_info.channels())
    return BackboneNeckAdapter(bb, FPN(bb.out_channels, 64))


def test_n_channel_detector_builds_and_trains() -> None:
    """A five-band detector builds, normalizes at five bands and returns finite training losses."""
    from tcip_mcp.pipelines.components.detectors import build_detector

    adapter = _nchan_adapter(5)
    names = list(adapter(torch.zeros(1, 5, 64, 64)).keys())
    det = build_detector(
        "faster_rcnn",
        adapter,
        num_classes=1,
        featmap_names=names,
        num_levels=len(names),
        image_mean=[0.4] * 5,
        image_std=[0.2] * 5,
        **detector_geometry().handed(),
    )
    assert det.transform.image_mean == [0.4] * 5

    det.train()
    losses = det(
        [torch.rand(5, 96, 96)],
        [
            {
                "boxes": torch.tensor([[8.0, 8.0, 48.0, 48.0]]),
                "labels": torch.ones(1, dtype=torch.long),
            }
        ],
    )
    assert losses and all(torch.isfinite(v) for v in losses.values())


def test_normalization_length_and_channel_count_are_validated() -> None:
    from tcip_mcp.pipelines.components.detectors import build_detector

    adapter = _nchan_adapter(5)
    names = list(adapter(torch.zeros(1, 5, 64, 64)).keys())
    common = {
        "featmap_names": names,
        "num_levels": len(names),
        **detector_geometry().handed(),
    }
    with pytest.raises(ValueError, match="both must be 5"):
        build_detector(
            "faster_rcnn",
            adapter,
            num_classes=1,
            **common,
            image_mean=[0.4] * 3,
            image_std=[0.2] * 3,
        )


def test_explicit_in_chans_wins_over_the_registration_order_probe() -> None:
    """A stated ``in_chans`` builds the detector where the registration-order probe disagrees.

    An adapter that registers its neck before its backbone reports a first-conv width that is not
    the image's band count; the stated three-band build still succeeds at three-band statistics.
    """
    import torch.nn as nn

    from tcip_mcp.pipelines.components.backbones import BackboneWrapper
    from tcip_mcp.pipelines.components.detectors import build_detector
    from tcip_mcp.pipelines.components.necks import FPN

    class NeckFirstAdapter(nn.Module):
        def __init__(self, backbone: BackboneWrapper, neck: FPN) -> None:
            super().__init__()
            self.neck = neck  # registered first: the probe sees this conv
            self.backbone = backbone
            self.out_channels = (
                neck.out_channels
                if isinstance(neck.out_channels, int)
                else neck.out_channels[-1]
            )

        def forward(self, x: Tensor) -> OrderedDict[str, Tensor]:
            from collections import OrderedDict

            return OrderedDict(sorted(self.neck(self.backbone(x)).items()))

    import timm

    m = timm.create_model(
        "resnet18",
        pretrained=False,
        features_only=True,
        out_indices=(1, 2, 3, 4),
        in_chans=3,
    )
    bb = BackboneWrapper(m, m.feature_info.channels())
    adapter = NeckFirstAdapter(bb, FPN(bb.out_channels, 64))
    names = list(adapter(torch.zeros(1, 3, 64, 64)).keys())

    # A plain RGB build must succeed even though the probe reports the neck's width.
    det = build_detector(
        "faster_rcnn",
        adapter,
        num_classes=1,
        featmap_names=names,
        num_levels=len(names),
        in_chans=3,
        image_mean=[0.4] * 3,
        image_std=[0.2] * 3,
        **detector_geometry().handed(),
    )
    assert det.transform.image_mean == [0.4] * 3


def test_torchvision_constructor_kwargs_forward_but_typos_raise() -> None:
    from tcip_mcp.pipelines.components.detectors import build_detector
    from tests._chain_fixtures import detector_values

    adapter = _nchan_adapter(3)
    names = list(adapter(torch.zeros(1, 3, 64, 64)).keys())
    common = {
        "featmap_names": names,
        "num_levels": len(names),
        **detector_values(),
    }
    det = build_detector(
        "faster_rcnn", adapter, num_classes=1, **common, box_nms_thresh=0.42
    )
    assert det.roi_heads.nms_thresh == 0.42
    with pytest.raises(TypeError, match="box_nms_threshh"):
        build_detector(
            "faster_rcnn",
            adapter,
            num_classes=1,
            **common,
            box_nms_threshh=0.42,
        )


@pytest.mark.parametrize(
    "dtype,scale",
    [
        ("uint8", 255),
        ("uint16", 65535),
        ("float32_unit", 1),
        ("float32_large", 4000),
    ],
)
def test_band_normalization_stats_matches_the_tensors_the_model_is_fed(
    tmp_path: Path, dtype: str, scale: int
) -> None:
    """Band statistics equal the mean and std of the tensors the loader yields.

    `pil_to_tensor` decides [0,1] scaling from dtype; the statistics read every raster and match
    its tensors for uint8, uint16, unit float and large float pixels.
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

    stats = band_normalization_stats(paths, 5, max_images=len(paths), seed=0)
    assert stats is not None
    mean, std = stats.image_mean, stats.image_std
    assert sorted(p for p, _window in _sampling(stats).windows) == sorted(
        str(p) for p in paths
    )
    actual = torch.stack(
        [pil_to_tensor(load_image(p, 5)) for p in paths]
    )  # [N, C, H, W]
    expected_mean = actual.permute(1, 0, 2, 3).reshape(5, -1).mean(dim=1)
    expected_std = (
        actual.permute(1, 0, 2, 3).reshape(5, -1).std(dim=1, unbiased=False)
    )
    assert torch.allclose(
        torch.tensor(mean, dtype=torch.float32), expected_mean, atol=1e-4
    )
    assert torch.allclose(
        torch.tensor(std, dtype=torch.float32), expected_std, atol=1e-4
    )


def _sampling(stats: Normalization) -> ImageStatsSampling:
    """Return the sampling record a band statistics record names as its source."""
    from tcip_mcp.pipelines.schemas import ImageStatsSampling

    assert isinstance(stats.source, ImageStatsSampling), stats.source
    return stats.source


def test_band_normalization_stats_admits_underivable(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.derivations import band_normalization_stats

    # Underivable is None, never a stand-in constant.
    assert (
        band_normalization_stats(
            [tmp_path / "missing.npy"], 5, max_images=1, seed=0
        )
        is None
    )


def test_band_normalization_stats_draw_their_capped_sample_from_the_stated_seed(
    tmp_path: Path,
) -> None:
    """Band statistics read at most the stated number of rasters, drawn from the stated seed.

    One seed reads the same rasters twice and another seed others, the record naming the cap and
    the seed beside what was read; neither sibling takes a cap or a seed by default.
    """
    import inspect

    import numpy as np

    from tcip_mcp.pipelines.derivations import (
        band_normalization_stats,
        band_normalization_stats_sampled,
    )

    paths = []
    for i in range(10):
        p = tmp_path / f"r{i}.npy"
        np.save(p, np.full((4, 4, 5), i, dtype=np.uint8))
        paths.append(p)

    def read(seed: int) -> list[str]:
        stats = band_normalization_stats(paths, 5, max_images=3, seed=seed)
        assert stats is not None
        sampling = _sampling(stats)
        assert (sampling.max_images, sampling.seed) == (3, seed)
        return [p for p, _window in sampling.windows]

    first = read(0)
    assert len(first) == 3
    assert read(0) == first
    assert set(read(1)) != set(first)
    for derivation in (
        band_normalization_stats,
        band_normalization_stats_sampled,
    ):
        parameters = inspect.signature(derivation).parameters
        for name in ("max_images", "seed"):
            assert parameters[name].default is inspect.Parameter.empty
    with pytest.raises(TypeError, match="max_images"):
        band_normalization_stats(paths, 5, seed=0)  # type: ignore[call-arg]  # the omission is the subject


def _strip_tiff_raster(
    path: Path, *, height: int = 24, width: int = 20, channels: int = 4
) -> NDArray[np.uint8]:
    """Write a seeded ``(height, width, channels)`` uint8 array to ``path``; return the array.

    The array is drawn from ``default_rng(5)`` and written by ``tifffile.imwrite`` in six-row
    strips.
    """
    import numpy as np
    import tifffile

    arr = np.random.default_rng(5).integers(
        0, 255, size=(height, width, channels), dtype=np.uint8
    )
    tifffile.imwrite(str(path), arr, rowsperstrip=6)
    return arr


def test_sampled_band_normalization_stats_over_full_coverage_match_the_exact_sibling(
    tmp_path: Path,
) -> None:
    """Sampled band statistics over every window match the exact sibling's numbers.

    The sampled statistics record a pixel fraction of 1 and equal the exact mean and std.
    """
    from tcip_mcp.pipelines.derivations import (
        band_normalization_stats,
        band_normalization_stats_sampled,
    )

    path = tmp_path / "mosaic.tif"
    arr = _strip_tiff_raster(path)

    exact = band_normalization_stats(
        [path], arr.shape[-1], max_images=1, seed=1
    )
    sampled = band_normalization_stats_sampled(
        [path],
        arr.shape[-1],
        seed=1,
        window_size=8,
        max_windows_per_image=999,
        max_images=1,
    )
    assert exact is not None and sampled is not None
    assert _sampling(sampled).pixel_fraction == 1.0
    assert sampled.image_mean == pytest.approx(
        exact.image_mean, rel=1e-12, abs=1e-12
    )
    assert sampled.image_std == pytest.approx(
        exact.image_std, rel=1e-12, abs=1e-12
    )


def test_sampled_band_normalization_stats_records_the_windows_it_read(
    tmp_path: Path,
) -> None:
    from tcip_mcp.pipelines.derivations import band_normalization_stats_sampled

    path = tmp_path / "mosaic.tif"
    arr = _strip_tiff_raster(path)
    kwargs = {
        "seed": 4,
        "window_size": 8,
        "max_windows_per_image": 3,
        "max_images": 1,
    }

    sampled = band_normalization_stats_sampled([path], arr.shape[-1], **kwargs)
    assert sampled is not None
    sampling = _sampling(sampled)
    windows = sampling.windows
    assert len(windows) == 3
    assert {label for label, _ in windows} == {str(path)}
    covered = sum(
        (r.x1 - r.x0) * (r.y1 - r.y0) for _, r in windows if r is not None
    )
    assert sampling.pixel_fraction == pytest.approx(
        covered / (arr.shape[0] * arr.shape[1])
    )
    assert sampling.seed == 4

    repeat = band_normalization_stats_sampled([path], arr.shape[-1], **kwargs)
    assert repeat == sampled


def test_sampled_band_normalization_stats_admit_underivable(
    tmp_path: Path,
) -> None:
    from tcip_mcp.pipelines.derivations import band_normalization_stats_sampled

    assert (
        band_normalization_stats_sampled(
            [tmp_path / "missing.npy"],
            5,
            seed=1,
            window_size=8,
            max_windows_per_image=2,
            max_images=1,
        )
        is None
    )


def test_build_dataset_sets_expected_channels(tmp_path: Path) -> None:
    from tcip_annotation.state import Annotation, BBox
    from tests._producer_fixtures import label_image, write_image

    images_dir = image_dir(tmp_path, UNDATED_BUCKET)
    write_image(images_dir / "a.png", (16, 16))
    label_image(
        images_dir / "a.png",
        [Annotation(subject="bud", geometry=BBox(6.4, 6.4, 9.6, 9.6))],
        16,
        16,
    )

    ds = dataset_over(
        "detection", str(images_dir), subject="bud", stated={"num_channels": 4}
    )
    assert ds.expected_channels == 4


def _classification_data(
    tmp_path: Path, *, num_channels: int | None
) -> tuple[dict, str]:
    """Return a data section over four sources under ``tmp_path``, and one source's path.

    ``write_regression_dataset`` writes the 32 px sources and their label table; the section names
    both, with ``num_channels`` stated when given.
    """
    from tests.tiny_trainer_fixtures import write_regression_dataset

    images_dir, csv_path = write_regression_dataset(
        tmp_path, [0.1, 0.3, 0.5, 0.7], [0, 1, 0, 1], height=32, width=32
    )

    data_cfg = {
        "images_dir": str(images_dir),
        "labels_dir": str(csv_path),
        "split": drawn_split(seed=1),
    }
    if num_channels is not None:
        data_cfg["num_channels"] = num_channels
    return data_cfg, str(images_dir / "img0.png")


def _classification_run(
    tmp_path: Path, *, num_channels: int | None
) -> tuple[dict, Dataset, str]:
    """Bind a run through the producer over :func:`_classification_data`.

    Return the data block it resolved, in the form its record holds it, after its loaders are
    built, with its training dataset and one source.
    """
    from tcip_mcp.pipelines.data.split_construction import auto_train_val
    from tcip_mcp.pipelines.schemas import DataSpec
    from tests._producer_fixtures import staged_layout

    data_cfg, image = _classification_data(tmp_path, num_channels=num_channels)
    train_ds, _val_ds, _partition, resolved = auto_train_val(
        tmp_path,
        "classification",
        DataSpec.model_validate(data_cfg),
        None,
        staged_layout(tmp_path, {}),
    )
    return resolved.record(), train_ds, image


def test_a_checkpoint_reads_images_at_the_width_its_run_recorded(
    tmp_path: Path,
) -> None:
    """A checkpoint's later readers all read images at the width its run recorded.

    A model is built at the width its run resolved: the builder, the contract dims and the
    predictor all take the run's own one channel over three-band sources.
    """
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.execution import Stated, prepare
    from tcip_mcp.pipelines.model_build import (
        recorded_model_dims,
        resolve_contract_dims,
    )
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    data_cfg, image = _classification_data(tmp_path, num_channels=1)
    checkpoint = load_registered_checkpoint(
        registered_checkpoint(
            tmp_path, model_source=CLASSIFIER_SOURCE, data=data_cfg
        ),
        project=tmp_path,
    )
    # The run recorded both: the width it read at and the count its own table carried.
    dims = recorded_model_dims(checkpoint.spec)
    assert dims == {"in_chans": 1, "num_classes": 2}
    assert resolve_contract_dims(
        checkpoint.spec, dims, checkpoint.geometry
    ) == {
        "in_chans": 1,
        "num_classes": 2,
        "img_size": (32, 32),
    }  # the frame its sources share

    p = prepare(
        checkpoint, Stated(tile=False, rationale=RATIONALE), device="cpu"
    ).runnable()

    assert p.predictor.in_chans == 1
    assert p.predict([image])[
        0
    ]  # the three-band source reads at the recorded one channel


def test_a_run_that_states_no_width_records_the_one_its_sources_carry(
    tmp_path: Path,
) -> None:
    """A run stating no width records its sources' three channels in its dataset and model dims."""
    data_cfg, train_ds, _image = _classification_run(
        tmp_path, num_channels=None
    )

    assert data_cfg["num_channels"] == train_ds.expected_channels == 3

    from tcip_mcp.pipelines.model_build import recorded_model_dims
    from tcip_mcp.pipelines.schemas import train_config

    spec = train_config(
        training_config(
            tmp_path, weights_statistics(CLASSIFIER_SOURCE), data_cfg
        )
    )
    assert recorded_model_dims(spec)["in_chans"] == 3


def test_a_source_whose_band_count_cannot_be_read_refuses_rather_than_defaulting(
    tmp_path: Path,
) -> None:
    """A source whose band count cannot be read refuses, naming the band count.

    A run's input channels are derived from its own sources.
    """
    import pytest

    from tcip_annotation.state import Annotation, BBox
    from tests._producer_fixtures import label_image, write_image

    images_dir = image_dir(tmp_path, UNDATED_BUCKET)
    write_image(images_dir / "a.png", (16, 16))
    label_image(
        images_dir / "a.png",
        [Annotation(subject="bud", geometry=BBox(6.4, 6.4, 9.6, 9.6))],
        16,
        16,
    )
    (images_dir / "a.png").write_bytes(b"not an image")

    with pytest.raises(ValueError, match="band count"):
        dataset_over("detection", str(images_dir), subject="bud")
