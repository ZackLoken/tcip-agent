"""Tests of the statistics a model normalizes by, each against the reference that publishes them.

The references are the pretrained weights' own as timm and torchvision publish them, the data's
own as numpy computes them over the frames, and the weights statement a model source makes.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tcip_mcp.pipelines.derivations import (  # noqa: E402
    band_normalization_stats,
    pretrained_normalization,
)
from tcip_mcp.pipelines.schemas import PretrainedWeights  # noqa: E402

TIMM_TAG = "vit_tiny_patch16_224.augreg_in21k"
"""timm weights published under statistics other than ImageNet's (a mean and std of 0.5)."""
TORCHVISION_ENUM = "EfficientNet_V2_L_Weights.IMAGENET1K_V1"
"""torchvision weights published under statistics other than ImageNet's (a mean and std of 0.5),
read from the enum's metadata without fetching them."""


def test_timm_weights_yield_the_created_models_own_pretrained_statistics():
    """The record holds the mean and std of the timm model created at those weights.

    The statistics are those of the created model's ``pretrained_cfg``, also when created
    ``features_only``.
    """
    import timm

    record = pretrained_normalization(
        PretrainedWeights(library="timm", name=TIMM_TAG)
    )

    created = timm.create_model(TIMM_TAG, pretrained=False)
    assert record.image_mean == list(created.pretrained_cfg["mean"])
    assert record.image_std == list(created.pretrained_cfg["std"])
    assert record.source == PretrainedWeights(library="timm", name=TIMM_TAG)
    features = timm.create_model(
        "resnet18.a1_in1k", pretrained=False, features_only=True
    )
    assert pretrained_normalization(
        PretrainedWeights(library="timm", name="resnet18.a1_in1k")
    ).image_mean == list(features.pretrained_cfg["mean"])


def test_torchvision_weights_yield_the_statistics_their_preset_normalizes_by():
    """The record's mean and std are those the torchvision weights' own preset applies to pixels.

    They are read back from what the preset transform makes of a black and a white frame.
    """
    from torchvision.models import get_weight

    record = pretrained_normalization(
        PretrainedWeights(library="torchvision", name=TORCHVISION_ENUM)
    )

    preset = get_weight(TORCHVISION_ENUM).transforms()
    black = preset(torch.zeros(3, 32, 32)).mean(dim=(1, 2))
    white = preset(torch.ones(3, 32, 32)).mean(dim=(1, 2))
    std = 1.0 / (white - black)
    mean = -black * std
    assert record.image_mean == pytest.approx(mean.tolist(), abs=1e-5)
    assert record.image_std == pytest.approx(std.tolist(), abs=1e-5)


def test_the_data_route_yields_the_numpy_mean_and_std_of_the_frames(tmp_path):
    """With no weights, the band statistics are numpy's mean and std of every frame's pixels.

    The frames are read with PIL and scaled to [0, 1] before ``numpy.mean`` and ``numpy.std``.
    """
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(3)
    paths = []
    for i in range(3):
        path = tmp_path / f"frame{i}.png"
        Image.fromarray(
            rng.integers(0, 255, size=(12, 16, 3), dtype=np.uint8)
        ).save(path)
        paths.append(path)

    record = band_normalization_stats(paths, 3, max_images=3, seed=0)

    assert record is not None
    pixels = np.concatenate(
        [
            np.asarray(Image.open(p), dtype=np.float64).reshape(-1, 3) / 255.0
            for p in paths
        ]
    )
    assert record.image_mean == pytest.approx(pixels.mean(axis=0).tolist())
    assert record.image_std == pytest.approx(pixels.std(axis=0).tolist())


def test_a_band_of_one_value_has_no_statistics_to_normalize_by(tmp_path):
    """A frame whose bands each hold one value is refused by the record's type.

    Its std is zero, which no normalization divides by, so the refusal comes where the derivation
    produces the record.
    """
    from PIL import Image
    from pydantic import ValidationError

    path = tmp_path / "flat.png"
    Image.new("RGB", (8, 8), (20, 40, 60)).save(path)

    with pytest.raises(ValidationError, match="image_std"):
        band_normalization_stats([path], 3, max_images=1, seed=0)


def test_every_model_source_records_its_normalization_and_the_weights_it_names(
    tmp_path,
):
    """A model source requires its normalization record and names the weights it loads.

    An omitted record is refused by the field's type. The weights a builder loads are read off its
    source: none for the data's band statistics, the weights themselves for their own statistics.
    """
    from pydantic import ValidationError

    from tcip_mcp.pipelines.schemas import ModelSourceSchema
    from tests._chain_fixtures import frame_statistics, noise_frames

    stated = {"builder": "m:f", "task": "detection"}
    with pytest.raises(ValidationError, match="normalization"):
        ModelSourceSchema.model_validate(stated)
    weights = PretrainedWeights(library="timm", name=TIMM_TAG)
    loaded = ModelSourceSchema.model_validate(
        {**stated, "normalization": pretrained_normalization(weights)}
    )
    unloaded = ModelSourceSchema.model_validate(
        {**stated, "normalization": frame_statistics(noise_frames(), 3)}
    )
    assert loaded.normalization.handed()["pretrained_weights"] == weights
    assert "pretrained_weights" not in unloaded.normalization.handed()
