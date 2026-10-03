"""Output artifacts anchor to the project they are produced for.

A run's weights and logs, and a sweep's trials, land under the project's own ``.tcip`` tree,
never under the server process's cwd.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest


def test_a_sweeps_directory_lies_under_its_project(tmp_path: Path) -> None:
    from tcip_mcp.tools.training_tools import sweep_dir

    assert sweep_dir("hpo_1", project=tmp_path) == tmp_path / ".tcip" / "hpo" / "hpo_1"


def test_launch_training_defaults_into_the_projects_experiment_store(
    tmp_path: Path, monkeypatch
) -> None:
    """A run's weights and logs land in its own directory under the project's experiments
    directory, never in the launching process's cwd."""
    pytest.importorskip("torchvision")
    monkeypatch.chdir(tmp_path)

    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.tools import training_tools

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    val_images = tmp_path / "val_images"
    val_labels = tmp_path / "val_labels"
    for d in (images_dir, labels_dir, val_images, val_labels):
        d.mkdir()
    for i in range(2):
        Image.new("RGB", (128, 128)).save(images_dir / f"t{i}.png")
        json_io.write_annotations(
            str(labels_dir / f"t{i}.json"),
            [Annotation(subject="bud", geometry=BBox(10, 10, 40, 40))], 128, 128)
    Image.new("RGB", (128, 128)).save(val_images / "v0.png")
    json_io.write_annotations(
        str(val_labels / "v0.json"),
        [Annotation(subject="bud", geometry=BBox(10, 10, 40, 40))], 128, 128)
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", lambda *a, **k: {})

    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"min_size": 64, "max_size": 128},
                         "task": "detection"},
        "data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                 "scope": {"subject": "bud"}},
        "batch_size": 1, "stages": [{"freeze_to": -1, "epochs": 1}],
                     "mixed_precision": False, "device": "cpu",
    }
    res = training_tools.launch_training(tmp_path, cfg, actor=None)
    assert "error" not in res, res
    run_dir = Path(res["output_dir"])
    assert run_dir == tmp_path / ".tcip" / "experiments" / res["experiment_id"]

    # Wait for the subprocess to finish rather than leaking a child that keeps writing into
    # this test's tmp root after the test moves on.
    deadline = time.monotonic() + 90
    final_status = None
    while time.monotonic() < deadline:
        final_status = training_tools.monitor_training(tmp_path, res["experiment_id"]).get("status")
        if final_status in ("completed", "failed", "canceled"):
            break
        time.sleep(0.5)
    else:
        pytest.fail("timed out waiting for the training subprocess to finish")
    assert final_status == "completed"
    assert (run_dir / "model_best.pt").is_file() or (run_dir / "model_final.pt").is_file()
