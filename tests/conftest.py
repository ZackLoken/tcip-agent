"""Shared test fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def pytest_collection_modifyitems(config, items):
    """Guardrail: fail loudly when far fewer tests collect than expected.

    Catches the failure mode where a missing dependency makes ~15 files module-level
    ``importorskip`` at collection time, shrinking the suite, while CI still reports
    green. CI sets ``TCIP_MIN_TESTS`` to a floor safely below the real count; a large
    shortfall means a core dep (torch/torchvision/pycocotools/...) is absent.
    """
    floor = os.environ.get("TCIP_MIN_TESTS")
    if floor and len(items) < int(floor):
        raise pytest.UsageError(
            f"Collected only {len(items)} tests (< TCIP_MIN_TESTS={floor}). A core "
            "dependency is likely missing — module-level importorskip silently skipped files."
        )


@pytest.fixture(autouse=True)
def _restore_platform_root_env():
    """Keep the process-global platform-state root hermetic across tests.

    ``set_active_project`` repins ``TCIP_PROJECT_ROOT`` in-process (so a project's audit /
    experiments / registry co-locate under it). Since pytest runs in one process, a test
    that adopts a tmp project would otherwise leak that now-deleted root into later tests.
    Snapshot and restore the var around every test.
    """
    saved = os.environ.get("TCIP_PROJECT_ROOT")
    yield
    if saved is None:
        os.environ.pop("TCIP_PROJECT_ROOT", None)
    else:
        os.environ["TCIP_PROJECT_ROOT"] = saved


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Create a minimal YOLO-format dataset in the canonical layout for testing."""
    date = "2-11-26"
    images_dir = tmp_path / "images" / date
    images_dir.mkdir(parents=True)
    labels_dir = tmp_path / "annotations" / "default" / date / "detect"
    labels_dir.mkdir(parents=True)
    preds_dir = tmp_path / "predictions" / "live" / date / "detect"
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
