"""Training-run cancellation (cancel_training; graceful stop)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests._chain_fixtures import CLASSIFIER_SOURCE
from tests._training_values import (
    drawn_split,
    evaluation_block,
    schedule,
    stage,
)

if TYPE_CHECKING:
    from pathlib import Path

torch = pytest.importorskip("torch")


def _opened(project: Path, experiment_id: str) -> Path:
    """Open a detector run under ``project`` through the launcher's own producer."""
    from tests._verified_checkpoint_fixtures import (
        detection_config,
        opened_run,
    )

    return opened_run(
        project,
        detection_config(project / "data", project=project),
        experiment_id=experiment_id,
    )


def test_cancel_training_reaches_the_runs_own_poll(tmp_path: Path) -> None:
    """A cancel requested by id reaches the run's own poll; an id naming no run refuses.

    The cancel is the one record the run's own poll reads.
    """
    from tcip_mcp.experiments import observe
    from tcip_mcp.pipelines.training.run_registry import observed_run
    from tcip_mcp.tools.training_tools import cancel_training

    run = observed_run(observe(_opened(tmp_path, "cancel-run-1")))
    assert not run.should_cancel()

    res = cancel_training(tmp_path, run.id, actor=None)
    assert not isinstance(res, dict) and res.run is not None
    assert res.run.experiment_id == run.id
    assert run.should_cancel()
    assert "error" in cancel_training(tmp_path, "missing-run", actor=None)


def test_a_cancel_written_is_answered_and_logged_when_the_run_will_not_read(
    tmp_path: Path,
) -> None:
    """A cancel whose run's record will not read afterward is still recorded, answered and logged.

    The answer's refused read is placed at the record that did not read, and the log holds the
    cancel's one line, stamped ok.
    """
    from tcip_mcp.experiments import CANCEL_FILE, RUN_FILE
    from tests._audit_fixtures import audit_rows
    from tests.test_hpo_cancel import cancel_unread

    run_dir = _opened(tmp_path, "cancel-run-unread")
    (run_dir / RUN_FILE).write_bytes(b"\xff\xfe not json")

    assert cancel_unread(tmp_path, run_dir)["where"] == str(run_dir / RUN_FILE)
    assert (run_dir / CANCEL_FILE).is_file()
    assert [
        row["status"] for row in audit_rows(tmp_path, "cancel_training")
    ] == ["ok"]


@pytest.mark.parametrize("damage", ["empty path", "traversing path"])
def test_a_cancel_is_recorded_at_the_run_record_whose_path_field_will_not_resolve(
    tmp_path: Path, damage: str
) -> None:
    """A run's cancel is recorded, at its run record, when a stored path there will not resolve.

    The record's ``resume_from`` is rewritten on disk empty, or carrying a ``..`` segment.
    """
    from tcip_mcp.experiments import RUN_FILE
    from tests.test_diagnose_project_run_records import DAMAGES
    from tests.test_hpo_cancel import cancel_unread

    run_dir = _opened(tmp_path, "cancel-run-path")
    DAMAGES[damage](run_dir)

    assert cancel_unread(tmp_path, run_dir)["where"] == str(run_dir / RUN_FILE)


def test_a_repeated_cancel_answers_as_the_first_until_its_record_will_not_read(
    tmp_path: Path,
) -> None:
    """A repeated cancel answers as the first did; once the cancel record will not read, it refuses.

    The refusal is an error naming the cancel record.
    """
    from tcip_mcp.experiments import CANCEL_FILE
    from tcip_mcp.tools.training_tools import cancel_training

    run_dir = _opened(tmp_path, "cancel-run-twice")
    first = cancel_training(tmp_path, run_dir.name, actor=None)

    assert cancel_training(tmp_path, run_dir.name, actor=None) == first

    (run_dir / CANCEL_FILE).write_bytes(b"\xff\xfe not json")
    refused = cancel_training(tmp_path, run_dir.name, actor=None)

    assert isinstance(refused, dict), refused
    assert str(run_dir / CANCEL_FILE) in refused["error"]


def test_cancel_before_training_yields_canceled(tmp_path: Path) -> None:
    pytest.importorskip("torchvision")
    from tcip_mcp.experiments import request_cancel
    from tcip_mcp.pipelines.training.generic_trainer import train
    from tests._chain_fixtures import training_config
    from tests.tiny_trainer_fixtures import (
        run_context,
        write_regression_dataset,
    )

    images_dir, labels = write_regression_dataset(
        tmp_path / "ds",
        [0.1, 0.3, 0.5, 0.7],
        [0, 1, 0, 1],
        height=32,
        width=32,
    )

    cfg = training_config(
        tmp_path,
        CLASSIFIER_SOURCE,
        {
            "images_dir": str(images_dir),
            "labels_dir": str(labels),
            "split": drawn_split(val_ratio=0.0),
        },
        stages=[stage(-1)],
        scheduler=schedule("cosine", horizon_epochs=3),
        evaluation=evaluation_block(selection_metric="loss"),
    )
    ctx = run_context(tmp_path, cfg)
    request_cancel(ctx.run_dir)  # request cancellation before any epoch runs
    run = train(ctx.run, ctx.train_loader)

    assert run.status == "canceled"
    assert run.current_epoch == 0  # stopped before training
    assert (
        ctx.run_dir / "model_final.pt"
    ).is_file()  # partial progress still saved
