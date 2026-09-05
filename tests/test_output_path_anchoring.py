"""Output artifacts anchor to the platform state root.

Weights, prediction buckets, delivery CSVs and curated datasets are addressed by caller-supplied
paths; a relative one resolves under the platform state root (the root the adopted project pins),
never the server process's cwd, and an absolute one stays the caller's own explicit choice.
The shared resolver is ``project_paths.resolve_output_path``; every output-writing tool delegates
to it, so the per-tool checks here are representative, not exhaustive.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest


def test_an_absolute_output_path_is_the_callers_own_choice(tmp_path: Path) -> None:
    from tcip_mcp.project_paths import resolve_output_path

    explicit = tmp_path / "elsewhere" / "out.csv"
    assert resolve_output_path(explicit) == explicit
    assert resolve_output_path(str(explicit)) == explicit


def test_a_relative_output_path_resolves_under_the_platform_state_root(tmp_path: Path) -> None:
    from tcip_mcp.project_paths import resolve_output_path

    assert resolve_output_path("exports/counts.csv") == tmp_path / "exports" / "counts.csv"
    assert resolve_output_path(Path("runs") / "exp1") == tmp_path / "runs" / "exp1"


def test_hpo_root_anchors_a_relative_output_dir_to_the_platform_state_root(tmp_path: Path) -> None:
    from tcip_mcp.tools.training_tools import hpo_root

    assert hpo_root("sweeps/hpo_1") == tmp_path / "sweeps" / "hpo_1"
    assert hpo_root("") == tmp_path / ".tcip" / "hpo"
    explicit = tmp_path / "explicit_sweeps"
    assert hpo_root(str(explicit)) == explicit


def test_launch_training_defaults_into_the_platform_state_roots_experiment_store(
    tmp_path: Path, monkeypatch
) -> None:
    """With no output_dir named, a run's weights and logs land in the platform state root's own
    experiment store, beside its experiment record, never in the launching process's cwd."""
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
                         "builder_kwargs": {"num_classes": 1, "min_size": 64, "max_size": 128},
                         "task": "detection"},
        "data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud",
                 "val_images_dir": str(val_images), "val_labels_dir": str(val_labels)},
        "training": {"batch_size": 1, "stages": [{"freeze_to": -1, "epochs": 1}],
                     "mixed_precision": False, "device": "cpu"},
    }
    res = training_tools.launch_training(cfg)
    assert "error" not in res, res
    run_dir = Path(res["output_dir"])
    assert run_dir == tmp_path / ".tcip" / "experiments" / res["run_id"]

    # Wait for the subprocess to finish rather than leaking a child that keeps writing into
    # this test's tmp root after the test moves on.
    deadline = time.monotonic() + 90
    final_status = None
    while time.monotonic() < deadline:
        final_status = training_tools.monitor_training(res["run_id"]).get("status")
        if final_status in ("completed", "failed", "cancelled"):
            break
        time.sleep(0.5)
    else:
        pytest.fail("timed out waiting for the training subprocess to finish")
    assert final_status == "completed"
    assert (run_dir / "model_best.pt").is_file() or (run_dir / "model_final.pt").is_file()
