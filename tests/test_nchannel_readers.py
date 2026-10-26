"""Tests of channel-aware image loading, readers, and ``BaseImageDataset``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from tests._chain_fixtures import CLASSIFIER_SOURCE  # noqa: E402
from tests._producer_fixtures import dataset_over, write_image  # noqa: E402
from tests.tiny_trainer_fixtures import write_regression_dataset  # noqa: E402


def test_pil_to_tensor_grayscale_and_multiband() -> None:
    from tcip_mcp.pipelines.image_utils import pil_to_tensor

    assert pil_to_tensor(np.zeros((8, 8), dtype=np.uint8)).shape == (
        1,
        8,
        8,
    )  # [H,W] -> [1,H,W]
    assert pil_to_tensor(np.ones((8, 8, 5), dtype=np.float32)).shape == (
        5,
        8,
        8,
    )  # 5-band
    assert pil_to_tensor(Image.new("RGB", (8, 8))).shape == (3, 8, 8)


def test_load_image_grayscale_pil_and_npy_multiband(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.image_utils import load_image

    png = write_image(tmp_path / "x.png", (16, 16), (200, 100, 50))
    g = load_image(png, 1)
    assert (
        isinstance(g, Image.Image) and g.mode == "L"
    )  # RGB -> grayscale on request

    npy = tmp_path / "ms.npy"
    np.save(npy, np.zeros((16, 16, 6), dtype=np.float32))
    arr = load_image(npy, 6)
    assert isinstance(arr, np.ndarray) and arr.shape == (16, 16, 6)


def test_build_dataset_grayscale_yields_one_channel(tmp_path: Path) -> None:
    images_dir, labels = write_regression_dataset(
        tmp_path, [0.0], [0], height=16, width=16
    )
    ds = dataset_over(
        "classification",
        str(images_dir),
        str(labels),
        stated={"num_channels": 1},
    )
    img, _ = ds[0]
    assert img.shape[0] == 1


def test_grayscale_classification_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("torchvision")
    from tcip_mcp.pipelines.training.generic_trainer import train
    from tests._chain_fixtures import training_config
    from tests._training_values import drawn_split, evaluation_block
    from tests.tiny_trainer_fixtures import run_context

    images_dir, labels = write_regression_dataset(
        tmp_path / "ds",
        [0.1, 0.3, 0.5, 0.7],
        [0, 1, 0, 1],
        height=32,
        width=32,
    )

    ctx = run_context(
        tmp_path,
        training_config(
            tmp_path,
            CLASSIFIER_SOURCE,
            {
                "images_dir": str(images_dir),
                "labels_dir": str(labels),
                "num_channels": 1,
                "split": drawn_split(val_ratio=0.0),
            },
            evaluation=evaluation_block(selection_metric="loss"),
        ),
    )
    assert ctx.train_loader.dataset[0][0].shape[0] == 1
    run = train(ctx.run, ctx.train_loader)
    assert (
        run.status == "completed"
    )  # 1-channel data + 1-channel model trains end to end
