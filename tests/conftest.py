"""Shared test fixtures."""

from __future__ import annotations

import json
import os
import textwrap
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


@pytest.fixture
def registry_path(tmp_path: Path) -> Path:
    """Create a minimal crops.yml for testing."""
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    yml = registry_dir / "crops.yml"
    yml.write_text(textwrap.dedent("""\
        ground_rgb_object_detection:
          image_perspective: ground
          sensor_type: rgb
          isolation_task: bush_isolation
          ml_task: object_detection
          traits:
            - name: catkin_05per_date
              definition: Date at which 5% of catkins are emerged
              format: date
              category: phenology
              crops: [hazelnut]
            - name: catkin_50per_date
              definition: Date at which 50% of catkins are emerged
              format: date
              category: phenology
              crops: [hazelnut]
        non_automatable:
          traits:
            - name: flavor_score
              definition: Subjective flavor rating
              format: ordinal_1_9
              category: quality
              crops: [hazelnut, chestnut]
    """))
    return yml


@pytest.fixture(autouse=True)
def set_registry_env(registry_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the registry loader to the test registry."""
    monkeypatch.setenv("TCIP_REGISTRY_PATH", str(registry_path))
    # Clear cached registry
    import tcip_mcp.tools.registry_tools as rt
    rt._REGISTRY = None
    rt._REGISTRY_PATH = None
