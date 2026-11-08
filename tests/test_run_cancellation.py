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


def test_cancel_training_reaches_the_runs_own_poll(tmp_path: Path) -> None:
    """A cancel requested by id reaches the run's own poll; an id naming no run refuses.

    The cancel is the one record the run's own poll reads.
    """
    from tcip_mcp.experiments import observe
    from tcip_mcp.pipelines.training.run_registry import observed_run
    from tcip_mcp.tools.training_tools import cancel_training
    from tests._verified_checkpoint_fixtures import (
        detection_config,
        opened_run,
    )

    run_dir = opened_run(
        tmp_path,
        detection_config(tmp_path / "data", project=tmp_path),
        experiment_id="cancel-run-1",
    )
    run = observed_run(observe(run_dir))
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

    The answer names the record that did not read, and the log holds the cancel's one line,
    stamped ok.
    """
    from tcip_mcp.experiments import CANCEL_FILE, RUN_FILE
    from tcip_mcp.tools.training_tools import cancel_training
    from tests._audit_fixtures import audit_rows
    from tests._verified_checkpoint_fixtures import (
        detection_config,
        opened_run,
    )

    run_dir = opened_run(
        tmp_path,
        detection_config(tmp_path / "data", project=tmp_path),
        experiment_id="cancel-run-unread",
    )
    (run_dir / RUN_FILE).write_bytes(b"\xff\xfe not json")

    answer = cancel_training(tmp_path, run_dir.name, actor=None)

    assert not isinstance(answer, dict)
    served = answer.model_dump(mode="json")
    assert (served["kind"], served["experiment_id"]) == (
        "cancel_recorded",
        run_dir.name,
    )
    assert str(run_dir / RUN_FILE) in served["unread"]["refusal"]["message"]
    assert (run_dir / CANCEL_FILE).is_file()
    assert [
        row["status"] for row in audit_rows(tmp_path, "cancel_training")
    ] == ["ok"]


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
