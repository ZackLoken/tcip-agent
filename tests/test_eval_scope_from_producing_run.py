"""Evaluating by run id reads ground truth through the scope the run's checkpoint records.

The fixture images each hold two buds and five leaves under one label document per image, and
the run is recorded over the leaf scope.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from tcip_mcp.dataset_layout import UNDATED_BUCKET

pytest.importorskip("torchvision")


def _two_subject_dataset(root: Path) -> Path:
    """Write three images carrying two subjects at different counts and return their directory.

    Each image holds 2 buds and 5 leaves, each object painted bright on its frame.
    """
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.subject_registry import Subject, SubjectRegistry
    from tests._producer_fixtures import (
        BRIGHT,
        label_image,
        painted_frame,
        registry_over,
    )

    images_dir = root / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True, exist_ok=True)
    registry_over(
        root,
        SubjectRegistry(subjects=(Subject(name="bud"), Subject(name="leaf"))),
    )
    for i in range(3):
        buds = [
            Annotation(
                subject="bud", geometry=BBox(5 + 12 * k, 5, 15 + 12 * k, 20)
            )
            for k in range(2)
        ]
        leaves = [
            Annotation(
                subject="leaf", geometry=BBox(6 + 20 * k, 40, 24 + 20 * k, 70)
            )
            for k in range(5)
        ]
        extents = [(5 + 12 * k, 5, 15 + 12 * k, 20) for k in range(2)] + [
            (6 + 20 * k, 40, 24 + 20 * k, 70) for k in range(5)
        ]
        painted_frame(
            160, 96, (100, 130, 90), [(box, BRIGHT) for box in extents]
        ).save(images_dir / f"img{i}.png")
        label_image(images_dir / f"img{i}.png", buds + leaves, 160, 96)
    return images_dir


def test_run_id_evaluation_scopes_ground_truth_to_the_runs_own_subject(
    tmp_path: Path, monkeypatch
) -> None:
    """Evaluating by run id reads ground truth in the scope the run's checkpoint records.

    With the evaluation runner replaced by a fake that captures its dataset, the captured
    dataset carries the recorded leaf scope, three samples, and five boxes in its first target.
    """
    import tcip_mcp.pipelines.training.eval_runners as runners
    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.tools.training_tools import evaluate_model
    from tests._producer_fixtures import admit_over
    from tests._training_values import IOU_THRESHOLD
    from tests._verified_checkpoint_fixtures import (
        SAMPLE_DETECTOR_PASS,
        finished_run,
    )

    images_dir = _two_subject_dataset(tmp_path / "ds")
    scope = admit_over(images_dir, subject="leaf").scope
    data = {
        "images_dir": str(images_dir),
        "num_channels": 3,
        "scope": asdict(scope),
    }
    run_dir = finished_run(tmp_path, experiment_id="auto-run-28", data=data)

    captured: dict = {}

    def _fake(pass_, loader, device, **kw):
        captured["ds"] = loader.dataset
        return {"eval_regime": "tile-level"}

    monkeypatch.setattr(runners, "run_test_evaluation", _fake)

    res = evaluate_model(
        tmp_path,
        run_dir.name,
        str(images_dir),
        stated=Stated(**SAMPLE_DETECTOR_PASS),
        iou_threshold=IOU_THRESHOLD,
    )
    assert "error" not in res, res

    dataset = captured["ds"]
    assert dataset.scope == scope
    assert len(dataset) == 3
    assert (
        len(dataset[0][1]["boxes"]) == 5
    )  # the leaves, not the two buds on the same image
