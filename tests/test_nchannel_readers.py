"""Channel-aware image loading, readers, and the BaseImageDataset refactor."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402


def test_pil_to_tensor_grayscale_and_multiband():
    from tcip_mcp.pipelines.image_utils import pil_to_tensor
    assert pil_to_tensor(np.zeros((8, 8), dtype=np.uint8)).shape == (1, 8, 8)      # [H,W] -> [1,H,W]
    assert pil_to_tensor(np.ones((8, 8, 5), dtype=np.float32)).shape == (5, 8, 8)  # 5-band
    assert pil_to_tensor(Image.new("RGB", (8, 8))).shape == (3, 8, 8)


def test_load_image_grayscale_pil_and_npy_multiband(tmp_path):
    from tcip_mcp.pipelines.image_utils import load_image
    png = tmp_path / "x.png"
    Image.new("RGB", (16, 16), (200, 100, 50)).save(png)
    g = load_image(png, 1)
    assert isinstance(g, Image.Image) and g.mode == "L"        # RGB -> grayscale on request

    npy = tmp_path / "ms.npy"
    np.save(npy, np.zeros((16, 16, 6), dtype=np.float32))
    arr = load_image(npy, 6)
    assert isinstance(arr, np.ndarray) and arr.shape == (16, 16, 6)


def test_build_dataset_grayscale_yields_one_channel(tmp_path):
    from tcip_mcp.pipelines.data.datasets import build_dataset
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (16, 16)).save(images_dir / "a.png")
    (tmp_path / "labels.csv").write_text("stem,label\na,0\n")
    ds = build_dataset("classification", images_dir=str(images_dir),
                       csv_path=str(tmp_path / "labels.csv"), num_classes=2, num_channels=1)
    img, _ = ds[0]
    assert img.shape[0] == 1


def test_grayscale_classification_end_to_end(tmp_path):
    pytest.importorskip("torchvision")
    from torch.utils.data import DataLoader

    from tcip_mcp.pipelines.data.datasets import build_dataset
    from tcip_mcp.pipelines.training.generic_trainer import train
    from tcip_mcp.pipelines.training.collation import task_collate
    from tcip_mcp.pipelines.training.run_registry import create_run

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    rows = ["stem,label"]
    for i in range(4):
        Image.new("RGB", (32, 32), (30 * i, 40, 50)).save(images_dir / f"img{i}.png")
        rows.append(f"img{i},{i % 2}")
    (tmp_path / "labels.csv").write_text("\n".join(rows) + "\n")

    ds = build_dataset("classification", images_dir=str(images_dir),
                       csv_path=str(tmp_path / "labels.csv"), num_classes=2, num_channels=1)
    loader = DataLoader(ds, batch_size=2, collate_fn=task_collate("classification"))
    model_source = {"builder": "tests.bespoke_models:build_bespoke_classifier",
                    "builder_kwargs": {"num_classes": 2, "in_chans": 1},
                    "task": "classification", "in_chans": 1}
    cfg = {"model_source": model_source, "device": "cpu", "stages": [{"freeze_to": -1, "epochs": 1}],
           "mixed_precision": False, "early_stopping": {"enabled": False}}
    run = train(create_run(cfg, str(tmp_path / "out")), loader, task="classification")
    assert run.status == "completed"  # 1-channel data + 1-channel model trains end to end
