"""The number an epoch reports under ``selection`` is the objective the checkpoint was chosen by.

``metrics.jsonl``, the in-memory history and the HPO ``epoch_callback`` carry one ``selection``
field, the value ``model_best.pt`` was chosen by. These runs keep the training loss far from the
selection objective.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

torch = pytest.importorskip("torch")

from tcip_mcp.pipelines.model_build import METRICS_KEY
from tcip_mcp.pipelines.training.generic_trainer import train
from tests._training_values import evaluation_block
from tests.tiny_trainer_fixtures import (
    TRAIN_INTENSITIES,
    VAL_INTENSITIES,
    classifier_config,
    frames_data,
    opposed_regression_data,
    regressor_config,
    run_context,
    separable_classifier_data,
)


def _config(
    tmp_path: Path,
    evaluation: dict | None = None,
    *,
    validated: bool = True,
) -> dict:
    """Return a regression config over frames under ``tmp_path``.

    The config is over opposed frames (``validated``), or over the training frames alone with no
    validation side, ``evaluation`` stated over its sample evaluation block.
    """
    data = (
        opposed_regression_data(
            tmp_path / "ds", TRAIN_INTENSITIES, VAL_INTENSITIES
        )
        if validated
        else frames_data(
            tmp_path / "ds",
            TRAIN_INTENSITIES,
            [2.0 * c for c in TRAIN_INTENSITIES],
        )
    )
    return regressor_config(
        3,
        data=data,
        project=tmp_path,
        builder_kwargs={"init_weight": 0.0},
        **(
            {"evaluation": evaluation_block(**evaluation)}
            if evaluation is not None
            else {}
        ),
    )


def test_epoch_record_reports_the_value_the_best_checkpoint_was_chosen_by(
    tmp_path: Path,
) -> None:
    """An epoch's recorded ``selection`` is the value the best checkpoint was chosen by.

    The chosen epoch's recorded ``selection``, the copy embedded in ``model_best.pt``, the
    ``metrics.jsonl`` line and the callback payload all carry ``run.best_metric``.
    """
    from tcip_mcp.experiments import METRICS_FILE, partition_rows, read_rows

    callbacks: list[dict] = []
    # The production wiring: the trainer hands each row to the envelope's sink, which logs it
    # to the run's own metrics log and fires the hook a trial prunes on.
    ctx = run_context(
        tmp_path,
        _config(tmp_path),
        epoch_hook=lambda epoch, metrics: callbacks.append(dict(metrics)),
    )
    out_dir = ctx.run_dir
    run = ctx.default_train()

    assert run.status == "completed", run.status_error
    history = run.metrics_history
    assert len(history) == 3

    best = torch.load(out_dir / "model_best.pt", weights_only=False)
    chosen_record = history[best["epoch"] - 1]

    assert chosen_record["selection"] == pytest.approx(
        run.best_metric, abs=1e-6
    )
    assert best[METRICS_KEY]["selection"] == pytest.approx(
        run.best_metric, abs=1e-6
    )
    # A regression run resolves its selection metric to the holdout loss, which this fixture
    # keeps far away from the training loss.
    assert chosen_record["selection"] == pytest.approx(
        chosen_record["val_loss"], abs=1e-6
    )
    assert chosen_record["selection"] != pytest.approx(
        chosen_record["train_loss"], rel=0.2
    )

    persisted = [
        r.model_dump()
        for r in partition_rows(read_rows(out_dir / METRICS_FILE)[0])[0]
    ]
    assert [r["selection"] for r in persisted] == [
        r["selection"] for r in history
    ]
    assert [r["selection"] for r in callbacks] == [
        r["selection"] for r in history
    ]
    assert run.best_metric == pytest.approx(
        min(r["selection"] for r in history), abs=1e-6
    )


def test_the_plateau_schedule_reduces_the_rate_where_torch_does_over_the_selection_values(
    tmp_path: Path,
) -> None:
    """The plateau schedule reduces the rate where torch's ``ReduceLROnPlateau`` does.

    A run selecting on a higher-is-better metric reduces its rate at the epochs
    ``ReduceLROnPlateau`` in ``max`` mode, stepped here over the selection values the run
    recorded, reduces its own.
    """
    from tests._training_values import SCHEDULE_SETTINGS, schedule, stop_rule

    # The holdout fit worsens from the first epoch on, so the stop rule ends the stage at five.
    config = {
        **_config(tmp_path, {"selection_metric": "r_squared"}),
        "scheduler": schedule("plateau"),
        "early_stopping": stop_rule(4),
    }
    ctx = run_context(tmp_path, config)
    run = train(ctx.run, ctx.train_loader, val_loader=ctx.val_loader)

    assert run.status == "completed", run.status_error
    rates = [r["lr"] for r in run.metrics_history]
    optimizer = torch.optim.SGD(
        [torch.zeros(1, requires_grad=True)], lr=rates[0]
    )
    reference = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", **SCHEDULE_SETTINGS["plateau"]
    )
    expected = []
    for row in run.metrics_history:
        expected.append(optimizer.param_groups[0]["lr"])
        reference.step(row["selection"])
    assert len(set(expected)) > 1, (
        "the series never plateaus, so no mode is told apart"
    )
    assert rates == pytest.approx(expected)


def test_epoch_record_follows_a_configured_selection_metric(
    tmp_path: Path,
) -> None:
    """An epoch's recorded ``selection`` follows a configured selection metric.

    An explicit ``evaluation.selection_metric`` drives both the checkpoint objective and the
    reported ``selection``, so the two still name the same number.
    """
    ctx = run_context(tmp_path, _config(tmp_path, {"selection_metric": "mae"}))
    out_dir = ctx.run_dir
    run = train(ctx.run, ctx.train_loader, val_loader=ctx.val_loader)

    assert run.status == "completed", run.status_error
    history = run.metrics_history
    assert len(history) == 3
    for record in history:
        assert record["selection_metric"] == "mae"
        assert record["selection"] == pytest.approx(
            record["val_mae"], abs=1e-6
        )
        assert record["selection"] != pytest.approx(
            record["train_loss"], rel=0.2
        )
    assert run.best_metric == pytest.approx(
        min(r["val_mae"] for r in history), abs=1e-6
    )

    best = torch.load(out_dir / "model_best.pt", weights_only=False)
    assert best[METRICS_KEY]["selection"] == pytest.approx(
        run.best_metric, abs=1e-6
    )


def test_a_run_selecting_on_f1_keeps_its_highest_f1_checkpoint(
    tmp_path: Path,
) -> None:
    """A run selecting on ``f1`` keeps its highest-f1 epoch as ``model_best.pt``.

    Over a run whose val f1 varies, ``model_best.pt`` holds the epoch with the highest val f1,
    and the run's best metric is that value.
    """
    ctx = run_context(
        tmp_path,
        classifier_config(
            5,
            separable_classifier_data(tmp_path / "ds"),
            project=tmp_path,
            evaluation=evaluation_block(selection_metric="f1"),
        ),
    )
    run = train(ctx.run, ctx.train_loader, val_loader=ctx.val_loader)

    assert run.status == "completed", run.status_error
    history = run.metrics_history
    f1_by_epoch = {
        epoch: r["val_f1"] for epoch, r in enumerate(history, start=1)
    }
    best_epoch = max(f1_by_epoch, key=lambda e: f1_by_epoch[e])
    worst_epoch = min(f1_by_epoch, key=lambda e: f1_by_epoch[e])
    assert (
        f1_by_epoch[best_epoch] > f1_by_epoch[worst_epoch]
    )  # the run must actually vary

    best = torch.load(ctx.run_dir / "model_best.pt", weights_only=False)
    assert best["epoch"] == best_epoch
    assert best[METRICS_KEY]["val_f1"] == pytest.approx(
        f1_by_epoch[best_epoch], abs=1e-6
    )
    assert run.best_metric == pytest.approx(f1_by_epoch[best_epoch], abs=1e-6)


def test_a_run_selecting_on_a_metric_its_task_does_not_produce_fails_naming_both(
    tmp_path: Path,
) -> None:
    """A run selecting on a metric its task does not produce fails naming both.

    A regression run selecting on ``f1`` fails naming ``f1``, ``val_f1`` and the keys its
    validation did produce (``val_loss``, ``val_mae``).
    """
    ctx = run_context(tmp_path, _config(tmp_path, {"selection_metric": "f1"}))
    run = train(ctx.run, ctx.train_loader, val_loader=ctx.val_loader)

    assert run.status == "failed"
    assert "'f1'" in run.status_error
    assert "'val_f1'" in run.status_error
    assert "val_loss" in run.status_error and "val_mae" in run.status_error


def test_a_loss_selected_run_with_no_validation_loader_still_completes_and_selects_its_lowest_loss(
    tmp_path: Path,
) -> None:
    """A loss-selected run with no validation loader completes and keeps its lowest loss.

    With no validation loader, every epoch selects on ``loss`` and records no ``val_loss``, and
    the run completes with its lowest selection value as its best.
    """
    ctx = run_context(tmp_path, _config(tmp_path, validated=False))
    assert ctx.val_loader is None
    run = train(ctx.run, ctx.train_loader, val_loader=None)

    assert run.status == "completed", run.status_error
    history = run.metrics_history
    assert len(history) == 3
    for record in history:
        assert record["selection_metric"] == "loss"
        assert "val_loss" not in record
    assert run.best_metric == pytest.approx(
        min(r["selection"] for r in history), abs=1e-6
    )

    best = torch.load(ctx.run_dir / "model_best.pt", weights_only=False)
    assert best[METRICS_KEY]["selection"] == pytest.approx(
        run.best_metric, abs=1e-6
    )
