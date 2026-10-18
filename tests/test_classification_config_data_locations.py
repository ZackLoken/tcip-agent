"""Each task's config names the data locations its loader reads.

A classification config naming only its images and CSV launches and completes, one naming a
missing CSV is refused by name, and a detection config naming its images alone passes preflight
with no issue.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from tests._chain_fixtures import (
    BESPOKE_DETECTION,
    BESPOKE_MODELS,
    FRAME_STATISTICS,
    training_config,
)  # noqa: E402
from tests._training_values import drawn_split, evaluation_block  # noqa: E402
from tests._verified_checkpoint_fixtures import run_to_end  # noqa: E402


def test_a_classification_config_launches_with_images_and_csv_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A classification config naming only its images and CSV launches and completes.

    It launches through the platform's own launcher, trains in a process other than this one
    and completes.
    """
    import os

    from tcip_mcp.tools.training_tools import launch_training
    from tests.tiny_trainer_fixtures import (
        MEAN_INTENSITY_CLASSIFIER,
        TINY_TRAINER_FILE,
        write_regression_dataset,
    )

    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard",
        lambda *a, **k: {},
    )
    images_dir, csv_path = write_regression_dataset(
        tmp_path / "ds", intensities=[0.1, 0.9] * 4, values=[0, 1] * 4
    )
    cfg = training_config(
        tmp_path,
        {
            "builder": MEAN_INTENSITY_CLASSIFIER,
            "source_files": [TINY_TRAINER_FILE],
            "task": "classification",
            "normalization": FRAME_STATISTICS,
        },
        {
            "images_dir": str(images_dir),
            "labels_dir": str(csv_path),
            "split": drawn_split(val_ratio=0.15),
        },
        batch_size=4,
        checkpoint_every_n_epochs=0,
        evaluation=evaluation_block(selection_metric="loss"),
    )

    res = launch_training(tmp_path, cfg, actor=None)

    assert "error" not in res, res
    assert res["pid"] != os.getpid()
    status = run_to_end(tmp_path, res["experiment_id"])
    assert status["state"] == "completed", status


def test_a_classification_config_naming_a_missing_csv_is_refused_by_name(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.training_tools import preflight_config
    from tests._chain_fixtures import weights_normalization
    from tests.tiny_trainer_fixtures import (
        MEAN_INTENSITY_CLASSIFIER,
        TINY_TRAINER_FILE,
    )

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    cfg = training_config(
        tmp_path,
        {
            "builder": MEAN_INTENSITY_CLASSIFIER,
            "source_files": [TINY_TRAINER_FILE],
            "task": "classification",
            "normalization": weights_normalization(),
        },
        {
            "images_dir": str(images_dir),
            "labels_dir": str(tmp_path / "gone.csv"),
        },
    )

    result = preflight_config(tmp_path, cfg)

    assert result["valid"] is False
    assert any(
        "data.labels_dir" in issue and "gone.csv" in issue
        for issue in result["issues"]
    ), result["issues"]


def test_a_detection_config_names_its_images_alone(tmp_path: Path) -> None:
    from tcip_mcp.tools.training_tools import preflight_config
    from tests._verified_checkpoint_fixtures import detection_images

    scope = {"subject": "bud"}
    cfg = training_config(
        tmp_path,
        {
            "builder": BESPOKE_DETECTION,
            "builder_kwargs": {},
            "source_files": [BESPOKE_MODELS],
            "task": "detection",
            "normalization": FRAME_STATISTICS,
        },
        {
            **detection_images(tmp_path / "ds", scope),
            "scope": scope,
            "split": drawn_split(val_ratio=0.15),
        },
    )

    result = preflight_config(tmp_path, cfg)

    assert result["issues"] == [], result["issues"]
