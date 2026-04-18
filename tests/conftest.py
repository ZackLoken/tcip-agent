"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Create a minimal YOLO-format dataset for testing."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    labels_dir = tmp_path / "labels" / "detect"
    labels_dir.mkdir(parents=True)
    preds_dir = tmp_path / "predictions" / "detect"
    preds_dir.mkdir(parents=True)

    # Create 3 tiny test images (1x1 white pixel PNG)
    for name in ("img_001", "img_002", "img_003"):
        from PIL import Image

        img = Image.new("RGB", (640, 480), color=(128, 128, 128))
        img.save(images_dir / f"{name}.jpg")

        # Labels: 2 boxes per image
        label_lines = [
            "0 0.5 0.5 0.1 0.1",
            "0 0.3 0.3 0.05 0.05",
        ]
        (labels_dir / f"{name}.txt").write_text("\n".join(label_lines) + "\n")

        # Predictions: 1 correct, 1 wrong location
        pred_lines = [
            "0 0.9 0.5 0.5 0.1 0.1",  # matches first GT box
            "0 0.7 0.8 0.8 0.05 0.05",  # FP
        ]
        (preds_dir / f"{name}.txt").write_text("\n".join(pred_lines) + "\n")

    return tmp_path
