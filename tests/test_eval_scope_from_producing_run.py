"""Evaluating by run id reads ground truth through the scope the run trained with.

Labels are stored by subject name in one file per image, so which subject the evaluation reads is
what decides the counts it scores. When the caller names no subject, the producing run's own config
supplies it; nothing else can, and a wrong scope produces a confident metric for the wrong object.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torchvision")


def _two_subject_dataset(root: Path) -> tuple[Path, Path]:
    """Images carrying two subjects with deliberately different counts: 2 buds, 5 leaves."""
    from PIL import Image

    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp import class_registry
    from tcip_mcp.class_registry import ClassRegistry, Subject

    images_dir, labels_dir = root / "images", root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    class_registry.write_registry(
        root / "classes.json",
        ClassRegistry(subjects=(Subject(name="bud"), Subject(name="leaf"))))
    for i in range(3):
        Image.new("RGB", (160, 96), color=(100, 130, 90)).save(images_dir / f"img{i}.png")
        buds = [Annotation(subject="bud", geometry=BBox(5 + 12 * k, 5, 15 + 12 * k, 20))
                   for k in range(2)]
        leaves = [Annotation(subject="leaf", geometry=BBox(6 + 20 * k, 40, 24 + 20 * k, 70))
                  for k in range(5)]
        json_io.write_annotations(str(labels_dir / f"img{i}.json"), buds + leaves, 160, 96)
    return images_dir, labels_dir


def test_run_id_evaluation_scopes_ground_truth_to_the_runs_own_subject(
        tmp_path: Path, monkeypatch) -> None:
    """With no caller-supplied subject, the evaluation dataset reads the subject the run trained
    on, so the ground truth it scores against holds that subject's objects and no others."""
    import tcip_mcp.pipelines.training.eval_runners as runners
    from tcip_mcp.pipelines.training.run_registry import create_run
    from tcip_mcp.tools.training_tools import evaluate_model
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    images_dir, labels_dir = _two_subject_dataset(tmp_path / "ds")
    run = create_run({"data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                               "subject": "leaf"}}, str(tmp_path / "runs"))
    Path(run.output_dir).mkdir(parents=True, exist_ok=True)
    registered_checkpoint(Path(run.output_dir), project_root=tmp_path,
                          filename="model_best.pt")

    captured: dict = {}

    def _fake(ckpt, loader, device, task, output_dir, **kw):
        captured["ds"] = loader.dataset
        return {"tiled": False, "eval_regime": "tile-level"}

    monkeypatch.setattr(runners, "run_test_evaluation", _fake)

    res = evaluate_model(run.run_id, str(images_dir), str(labels_dir), task="detection")
    assert "error" not in res, res

    dataset = captured["ds"]
    assert dataset.subject == "leaf"
    assert len(dataset) == 3
    assert len(dataset[0][1]["boxes"]) == 5  # the leaves, not the two buds on the same image


def test_a_caller_supplied_subject_still_wins_over_the_runs_own(
        tmp_path: Path, monkeypatch) -> None:
    """Reuse never overrides an explicit scope: evaluating the same run against another subject
    stays possible, and reads that subject's ground truth."""
    import tcip_mcp.pipelines.training.eval_runners as runners
    from tcip_mcp.pipelines.training.run_registry import create_run
    from tcip_mcp.tools.training_tools import evaluate_model
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    images_dir, labels_dir = _two_subject_dataset(tmp_path / "ds")
    run = create_run({"data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                               "subject": "leaf"}}, str(tmp_path / "runs"))
    Path(run.output_dir).mkdir(parents=True, exist_ok=True)
    registered_checkpoint(Path(run.output_dir), project_root=tmp_path,
                          filename="model_best.pt")

    captured: dict = {}

    def _fake(ckpt, loader, device, task, output_dir, **kw):
        captured["ds"] = loader.dataset
        return {"tiled": False, "eval_regime": "tile-level"}

    monkeypatch.setattr(runners, "run_test_evaluation", _fake)

    res = evaluate_model(run.run_id, str(images_dir), str(labels_dir), task="detection",
                         subject="bud")
    assert "error" not in res, res
    assert captured["ds"].subject == "bud"
    assert len(captured["ds"][0][1]["boxes"]) == 2
