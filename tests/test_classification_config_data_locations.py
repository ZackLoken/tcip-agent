"""Launch asks each task for the data locations its own loader reads: a classification config
names its images and its CSV and nothing else, while detection still needs its labels directory.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("torch")


def _wait_terminal(experiment_id: str, seconds: float = 120) -> dict:
    from tcip_mcp.tools.training_tools import monitor_training

    deadline = time.monotonic() + seconds
    status: dict = {}
    while time.monotonic() < deadline:
        status = monitor_training(experiment_id)
        if status.get("status") in ("completed", "failed", "cancelled"):
            return status
        time.sleep(0.5)
    pytest.fail(f"the training subprocess never reached a terminal state: {status}")


def test_a_classification_config_launches_with_images_and_csv_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The batch the loader trains on comes from ``images_dir`` and ``csv_path``
    (``ClassificationDataset``), so a config naming only those launches through the platform's
    own launcher, trains in the worker process and completes."""
    import os

    from tcip_mcp.tools.training_tools import launch_training
    from tests.tiny_trainer_fixtures import write_regression_dataset

    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", lambda *a, **k: {})
    images_dir, csv_path = write_regression_dataset(
        tmp_path / "ds", intensities=[0.1, 0.9] * 4, values=[0, 1] * 4)
    cfg = {
        "model_source": {"builder": "tests.tiny_trainer_fixtures:build_mean_intensity_classifier",
                         "task": "classification", "in_chans": 3},
        "data": {"images_dir": str(images_dir), "csv_path": str(csv_path)},
        "batch_size": 4, "stages": [{"freeze_to": 0, "epochs": 1}],
        "mixed_precision": False, "device": "cpu",
        "checkpoint_every_n_epochs": 0, "early_stopping": {"enabled": False},
    }

    res = launch_training(cfg, str(tmp_path / "out"))

    assert "error" not in res, res
    assert res["pid"] != os.getpid()
    status = _wait_terminal(res["experiment_id"])
    assert status["status"] == "completed", status


def test_a_classification_config_naming_a_missing_csv_is_refused_by_name(tmp_path: Path) -> None:
    from tcip_mcp.tools.training_tools import preflight_config

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    cfg = {
        "model_source": {"builder": "tests.tiny_trainer_fixtures:build_mean_intensity_classifier",
                         "task": "classification", "in_chans": 3},
        "data": {"images_dir": str(images_dir), "csv_path": str(tmp_path / "gone.csv")},
    }

    result = preflight_config(cfg)

    assert result["valid"] is False
    assert any("data.csv_path" in issue for issue in result["issues"]), result["issues"]


def test_a_detection_config_still_needs_its_labels_directory(tmp_path: Path) -> None:
    from tcip_mcp.tools.training_tools import preflight_config

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": {"images_dir": str(images_dir)},
    }

    result = preflight_config(cfg)

    assert result["valid"] is False
    assert any("Missing 'data.labels_dir'" in issue for issue in result["issues"])
