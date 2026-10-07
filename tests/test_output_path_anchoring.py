"""Output artifacts anchor to the project they are produced for.

A run's weights and logs, and a sweep's trials, land under the project's own ``.tcip`` tree,
never under the server process's cwd.
"""

from __future__ import annotations

from tcip_mcp.dataset_layout import UNDATED_BUCKET

from pathlib import Path

import pytest


def test_launch_training_defaults_into_the_projects_experiment_store(
    tmp_path: Path, monkeypatch
) -> None:
    """A run's weights and logs land in its own directory under the project's experiments
    directory, never in the launching process's cwd."""
    pytest.importorskip("torchvision")
    monkeypatch.chdir(tmp_path)

    from tcip_mcp.tools import training_tools
    from tests._producer_fixtures import seed_bud_images
    from tests._verified_checkpoint_fixtures import run_to_end

    images_dir = seed_bud_images(tmp_path / "images" / UNDATED_BUCKET, n=2)
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", lambda *a, **k: {})

    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"min_size": 64, "max_size": 128},
                         "task": "detection"},
        "data": {"images_dir": str(images_dir), "scope": {"subject": "bud"},
                 "split": {"seed": 0, "val_ratio": 0.15}},
        "batch_size": 1, "stages": [{"freeze_to": -1, "epochs": 1}],
                     "mixed_precision": False, "device": "cpu",
        # An untrained toy detector scores no objective; its validation loss is what ranks.
        "evaluation": {"selection_metric": "loss"},
    }
    res = training_tools.launch_training(tmp_path, cfg, actor=None)
    assert "error" not in res, res
    run_dir = Path(res["output_dir"])
    assert run_dir == tmp_path / ".tcip" / "experiments" / res["experiment_id"]

    # Wait for the subprocess to finish rather than leaking a child that keeps writing into
    # this test's tmp root after the test moves on.
    assert run_to_end(tmp_path, res["experiment_id"])["state"] == "completed"
    assert (run_dir / "model_best.pt").is_file() or (run_dir / "model_final.pt").is_file()
