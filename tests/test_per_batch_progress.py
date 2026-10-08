"""Per-batch progress rows: the trainer emits them at a stated cadence, on the run's one metrics
log, and no reader of that log takes one for an epoch row."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tcip_mcp.experiments import (  # noqa: E402
    EPOCH_KEY, METRICS_FILE, STEP_KEY, best_selection, epoch_rows, observe, partition_rows,
    read_rows, run_summary,
)
from tcip_mcp.pipelines.training.envelope import TrainContext  # noqa: E402
from tcip_mcp.pipelines.training.run_registry import observed_run  # noqa: E402
from tests._verified_checkpoint_fixtures import opened_run  # noqa: E402
from tests.tiny_trainer_fixtures import (  # noqa: E402
    opposed_regression_loaders,
    regressor_config,
    write_regression_dataset,
)


def batch_rows(rows):
    """The per-batch rows among ``rows`` (``experiments.partition_rows``)."""
    return partition_rows(rows)[1]


TRAIN_INTENSITIES = [0.10, 0.25, 0.40, 0.55, 0.70, 0.85]
VAL_INTENSITIES = [0.15, 0.35, 0.60, 0.90]


def _context(tmp_path, hook_calls: list, **extra) -> TrainContext:
    """A context over a regression run opened by the launcher's own producer, three training
    batches an epoch, recording every epoch hook call into ``hook_calls``."""
    images_dir, csv_path = write_regression_dataset(
        tmp_path / "ds", TRAIN_INTENSITIES, [2.0 * c for c in TRAIN_INTENSITIES])
    config = regressor_config(2, data={
        "num_channels": 1, "scope": {}, "images_dir": str(images_dir),
        "labels_dir": str(csv_path), "split": {"seed": 1, "val_ratio": 0.15}}, **extra)
    run_dir = opened_run(tmp_path, config)
    train_loader, val_loader = opposed_regression_loaders(TRAIN_INTENSITIES, VAL_INTENSITIES)
    return TrainContext(run=observed_run(observe(run_dir)), train_loader=train_loader,
                        val_loader=val_loader,
                        epoch_hook=lambda epoch, metrics: hook_calls.append(epoch))


def test_the_trainer_logs_a_batch_row_at_the_stated_cadence_and_no_reader_takes_it_for_an_epoch(
        tmp_path):
    hook_calls: list = []
    ctx = _context(tmp_path, hook_calls, log_every_n_batches=2)
    run = ctx.default_train()
    assert run.status == "completed", run.status_error

    rows = read_rows(ctx.run_dir / METRICS_FILE)[0]
    batches = batch_rows(rows)
    assert [(row[STEP_KEY], row[EPOCH_KEY]) for row in batches] == [(2, 1), (4, 2), (6, 2)]
    assert all(isinstance(row["batch_loss"], float) for row in batches)

    epochs = epoch_rows(rows)
    assert [row[EPOCH_KEY] for row in epochs] == [1, 2]
    assert not any(STEP_KEY in row or "batch_loss" in row for row in epochs)
    assert hook_calls == [1, 2]
    summary = run_summary(observe(ctx.run_dir), rows, None)
    assert summary.current_epoch == 2


@pytest.mark.parametrize("n_batches", [3, 19, 29, 30, 31])
def test_a_config_stating_no_cadence_logs_ten_batch_rows_every_epoch(tmp_path, n_batches):
    """No stated cadence: each of two epochs logs ten rows (every batch of a shorter epoch),
    spaced evenly over the epoch, the gaps between them never differing by more than one batch,
    the last at the epoch's own final batch."""
    from torch.utils.data import DataLoader

    from tcip_mcp.pipelines.training.collation import task_collate
    from tests.tiny_trainer_fixtures import ConstantImageDataset

    ctx = _context(tmp_path, [])
    intensities = [0.1 + 0.6 * i / n_batches for i in range(n_batches)]
    ctx.train_loader = DataLoader(ConstantImageDataset(intensities, [2.0 * c for c in intensities]),
                                  batch_size=1, collate_fn=task_collate("regression"))
    assert ctx.default_train().status == "completed"
    batches = batch_rows(read_rows(ctx.run_dir / METRICS_FILE)[0])
    for epoch in (1, 2):
        steps = [row[STEP_KEY] for row in batches if row[EPOCH_KEY] == epoch]
        assert len(steps) == len(set(steps)) == min(10, n_batches)
        assert all((epoch - 1) * n_batches < step <= epoch * n_batches for step in steps)
        assert steps[-1] == epoch * n_batches
        gaps = [b - a for a, b in zip([(epoch - 1) * n_batches, *steps], steps)]
        assert max(gaps) - min(gaps) <= 1, gaps


def test_the_best_selection_reads_epoch_rows_and_never_a_batch_row_s_selection(tmp_path):
    """A bespoke loop's batch row carrying a ``selection`` better than every epoch's does not
    become the run's best selection."""
    ctx = _context(tmp_path, [])
    ctx.log_metrics(1, {"selection": 0.5})
    ctx.log_batch(3, 1, {"selection": -100.0})
    ctx.log_metrics(2, {"selection": 0.7})

    rows = read_rows(ctx.run_dir / METRICS_FILE)[0]
    assert best_selection(rows, {"higher_is_better": False}) == 0.5
    assert best_selection(rows, {"higher_is_better": True}) == 0.7


def test_a_bespoke_loop_logs_a_batch_row_and_neither_sink_takes_the_other_s_stamp(tmp_path):
    hook_calls: list = []
    ctx = _context(tmp_path, hook_calls)

    ctx.log_batch(5, 1, {"batch_loss": 0.3})
    ctx.log_metrics(1, {"train_loss": 0.4})
    with pytest.raises(ValueError, match="step"):
        ctx.log_batch(6, 1, {"step": 3})
    with pytest.raises(ValueError, match="step"):
        ctx.log_metrics(1, {"step": 3})

    rows = read_rows(ctx.run_dir / METRICS_FILE)[0]
    (batch,) = batch_rows(rows)
    assert (batch[STEP_KEY], batch[EPOCH_KEY], batch["batch_loss"]) == (5, 1, 0.3)
    (epoch,) = epoch_rows(rows)
    assert epoch["train_loss"] == 0.4 and "batch_loss" not in epoch
    assert hook_calls == [1]
