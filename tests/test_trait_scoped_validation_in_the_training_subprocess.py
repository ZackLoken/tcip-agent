"""A trait-scoped validation through the real training subprocess.

``evaluation.resolve_match_criterion`` resolves the trait's authored localization against the
run's own validation GT and never writes the entry: a trait changes only through a proposed
revision. This drives that through the process a training run actually executes in,
``subprocess_worker.run``, so the boundary is covered by the entry point itself and not only by
in-process calls.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_the_subprocess_resolves_the_authored_criterion_and_leaves_the_trait_entry_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """The child resolves the trait's authored localization over the validation GT and
    completes, and the trait's record still holds the one revision the test confirmed."""
    pytest.importorskip("torchvision")
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools import training_tools
    from tcip_mcp.traits import CENTER_MATCH, read_trait

    from tests._trait_fixtures import entry, propose_and_confirm

    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", lambda *a, **k: {})
    from tcip_annotation.state import Annotation, BBox

    from tests._producer_fixtures import seed_labeled_images

    images_dir = seed_labeled_images(
        tmp_path / "ds" / "images" / "train",
        [Annotation(subject="leaf", geometry=BBox(10, 10, 30, 30))], n=2, width=128, height=128)
    proposed = propose_and_confirm(
        tmp_path, entry("leaf", ("leaf_length",), localization=CENTER_MATCH))

    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"min_size": 64, "max_size": 128},
                         "task": "detection"},
        "data": {"images_dir": str(images_dir), "scope": {"subject": "leaf"},
                 "split": {"seed": 0, "val_ratio": 0.15}},
        "evaluation": {"trait": "leaf"},
        "batch_size": 1, "stages": [{"freeze_to": -1, "epochs": 1}],
                     "mixed_precision": False, "device": "cpu",
    }
    res = training_tools.launch_training(tmp_path, cfg, actor=None)
    assert "error" not in res, res
    assert res["pid"] != os.getpid()

    from tests._verified_checkpoint_fixtures import run_to_end

    assert run_to_end(tmp_path, res["experiment_id"], seconds=180)["state"] == "completed"

    assert read_trait("leaf", tmp_path).revisions == (proposed,)
