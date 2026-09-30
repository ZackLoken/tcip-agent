"""A trait's unstated localization, derived through the real training subprocess.

``evaluation.resolve_match_criterion`` derives the kind from the run's own validation GT when the
trait's entry states none, and never writes the entry: a trait changes only through a proposed
revision. This drives that through the process a training run actually executes in,
``subprocess_worker.run``, so the boundary is covered by the entry point itself and not only by
in-process calls.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest


def _seed_dataset(root: Path) -> tuple[Path, Path, Path, Path]:
    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    images_dir, labels_dir = root / "images", root / "labels"
    val_images, val_labels = root / "val_images", root / "val_labels"
    for d in (images_dir, labels_dir, val_images, val_labels):
        d.mkdir(parents=True)
    # 20 px boxes: under a 15 px jitter their achievable IoU is below 0.5, so the derivation
    # lands on center_match, the same geometry the unit test derives from.
    box = BBox(10, 10, 30, 30)
    for i in range(2):
        Image.new("RGB", (128, 128)).save(images_dir / f"t{i}.png")
        json_io.write_annotations(str(labels_dir / f"t{i}.json"),
                                  [Annotation(subject="leaf", geometry=box)], 128, 128)
    Image.new("RGB", (128, 128)).save(val_images / "v0.png")
    json_io.write_annotations(str(val_labels / "v0.json"),
                              [Annotation(subject="leaf", geometry=box)], 128, 128)
    return images_dir, labels_dir, val_images, val_labels


def _wait_terminal(project: Path, run_id: str, seconds: float) -> str:
    from tcip_mcp.tools import training_tools

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        status = training_tools.monitor_training(project, run_id)
        if status.get("status") in ("completed", "failed", "canceled"):
            return str(status.get("status"))
        time.sleep(0.5)
    pytest.fail("timed out waiting for the training subprocess to reach a terminal state")


def test_the_subprocess_derives_an_unstated_kind_and_leaves_the_trait_entry_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """The child derives the unstated kind from the validation GT and completes, and the trait's
    record still holds the one revision the test confirmed."""
    pytest.importorskip("torchvision")
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools import training_tools
    from tcip_mcp.traits import read_trait

    from tests._trait_fixtures import entry, propose_and_confirm

    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", lambda *a, **k: {})
    images_dir, labels_dir, val_images, val_labels = _seed_dataset(tmp_path / "ds")
    proposed = propose_and_confirm(tmp_path, entry("leaf", ("leaf_length",)))

    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"min_size": 64, "max_size": 128},
                         "task": "detection"},
        "data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                 "scope": {"subject": "leaf"}},
        "evaluation": {"trait": "leaf"},
        "batch_size": 1, "stages": [{"freeze_to": -1, "epochs": 1}],
                     "mixed_precision": False, "device": "cpu",
    }
    res = training_tools.launch_training(tmp_path, cfg)
    assert "error" not in res, res
    assert res["pid"] != os.getpid()

    assert _wait_terminal(tmp_path, res["experiment_id"], 180) == "completed"

    assert read_trait("leaf", tmp_path).revisions == (proposed,)
