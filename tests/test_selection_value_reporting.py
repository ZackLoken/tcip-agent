"""The number an epoch reports under ``selection`` is the objective the checkpoint was chosen by.

``metrics.jsonl``, the TensorBoard row, the in-memory history and the HPO ``epoch_callback`` all
carry the same ``selection`` field, and a reader takes it for the value that drove
``model_best.pt`` and early stopping. These runs keep the training loss far away from the
selection objective, so a record carrying the training loss under that key is visible.
"""

from __future__ import annotations


import pytest

torch = pytest.importorskip("torch")

from tcip_mcp.pipelines.model_build import METRICS_KEY
from tcip_mcp.pipelines.training.generic_trainer import train
from tests.tiny_trainer_fixtures import (
    classifier_config,
    opposed_regression_loaders,
    regressor_config,
    separable_classifier_loaders,
    trainer_run,
    write_regression_dataset,
)

TRAIN_INTENSITIES = [0.10, 0.25, 0.40, 0.55, 0.70, 0.85]
VAL_INTENSITIES = [0.15, 0.35, 0.60, 0.90]


def _config(evaluation: dict | None = None) -> dict:
    return regressor_config(
        3, builder_kwargs={"init_weight": 0.0},
        data={"num_channels": 1, "scope": {}, "split": {"seed": 1, "val_ratio": 0.15}},
        **({"evaluation": evaluation} if evaluation is not None else {}))


def test_epoch_record_reports_the_value_the_best_checkpoint_was_chosen_by(tmp_path):
    """The chosen epoch's recorded ``selection``, the copy embedded in ``model_best.pt``, the
    ``metrics.jsonl`` line and the callback payload all carry ``run.best_metric``."""
    from tcip_mcp.experiments import METRICS_FILE, observe, partition_rows, read_rows
    from tcip_mcp.pipelines.training.envelope import TrainContext
    from tcip_mcp.pipelines.training.run_registry import observed_run
    from tests._verified_checkpoint_fixtures import opened_run

    train_loader, val_loader = opposed_regression_loaders(TRAIN_INTENSITIES, VAL_INTENSITIES)
    callbacks: list[dict] = []
    images_dir, csv_path = write_regression_dataset(
        tmp_path / "ds", TRAIN_INTENSITIES, [2.0 * c for c in TRAIN_INTENSITIES])
    config = _config()
    config["data"] = {**config["data"], "images_dir": str(images_dir),
                      "labels_dir": str(csv_path)}
    out_dir = opened_run(tmp_path, config)
    run = observed_run(observe(out_dir))
    # The production wiring: the trainer hands each row to the envelope's sink, which logs it
    # to the run's own metrics log and fires the hook a trial prunes on.
    ctx = TrainContext(run=run, train_loader=train_loader, val_loader=val_loader,
                       epoch_hook=lambda epoch, metrics: callbacks.append(dict(metrics)))
    run = ctx.default_train()

    assert run.status == "completed", run.status_error
    history = run.metrics_history
    assert len(history) == 3

    best = torch.load(out_dir / "model_best.pt", weights_only=False)
    chosen_record = history[best["epoch"] - 1]

    assert chosen_record["selection"] == pytest.approx(run.best_metric, abs=1e-6)
    assert best[METRICS_KEY]["selection"] == pytest.approx(run.best_metric, abs=1e-6)
    # A regression run resolves its selection metric to the holdout loss, which this fixture
    # keeps far away from the training loss.
    assert chosen_record["selection"] == pytest.approx(chosen_record["val_loss"], abs=1e-6)
    assert chosen_record["selection"] != pytest.approx(chosen_record["train_loss"], rel=0.2)

    persisted = partition_rows(read_rows(out_dir / METRICS_FILE)[0])[0]
    assert [r["selection"] for r in persisted] == [r["selection"] for r in history]
    assert [r["selection"] for r in callbacks] == [r["selection"] for r in history]
    assert run.best_metric == pytest.approx(min(r["selection"] for r in history), abs=1e-6)


def test_the_plateau_scheduler_steps_on_the_validation_loss_under_its_declared_key(
    tmp_path, monkeypatch,
):
    """The scheduler reads the validation loss by the one declared key, so under another
    validation prefix it still steps on each epoch's validation loss, never the training loss."""
    import tcip_mcp.pipelines.training.generic_trainer as trainer_mod

    monkeypatch.setattr(trainer_mod, "VAL_METRIC_PREFIX", "heldout_")
    monkeypatch.setattr(trainer_mod, "VAL_LOSS_KEY", "heldout_loss")
    stepped: list[float] = []
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau
    real_step = plateau.step

    def step(self, metrics, *args, **kwargs):
        stepped.append(float(metrics))
        return real_step(self, metrics, *args, **kwargs)

    monkeypatch.setattr(plateau, "step", step)
    train_loader, val_loader = opposed_regression_loaders(TRAIN_INTENSITIES, VAL_INTENSITIES)
    config = {**_config(), "scheduler": {"type": "plateau"}}
    run = trainer_run(config, tmp_path / "out", project=tmp_path, has_val_loader=True,
                      id="auto-run-plateau")
    run = train(run, train_loader, val_loader=val_loader)

    assert run.status == "completed", run.status_error
    assert stepped == pytest.approx([r["heldout_loss"] for r in run.metrics_history])


def test_epoch_record_follows_a_configured_selection_metric(tmp_path):
    """An explicit ``evaluation.selection_metric`` drives both the checkpoint objective and the
    reported ``selection``, so the two still name the same number."""
    train_loader, val_loader = opposed_regression_loaders(TRAIN_INTENSITIES, VAL_INTENSITIES)
    out_dir = tmp_path / "out"
    run = trainer_run(_config({"selection_metric": "mae"}), out_dir, project=tmp_path,
                      has_val_loader=True, id="auto-run-64")
    run = train(run, train_loader, val_loader=val_loader)

    assert run.status == "completed", run.status_error
    history = run.metrics_history
    assert len(history) == 3
    for record in history:
        assert record["selection_metric"] == "mae"
        assert record["selection"] == pytest.approx(record["val_mae"], abs=1e-6)
        assert record["selection"] != pytest.approx(record["train_loss"], rel=0.2)
    assert run.best_metric == pytest.approx(min(r["val_mae"] for r in history), abs=1e-6)

    best = torch.load(out_dir / "model_best.pt", weights_only=False)
    assert best[METRICS_KEY]["selection"] == pytest.approx(run.best_metric, abs=1e-6)


def test_a_run_selecting_on_f1_keeps_its_highest_f1_checkpoint(tmp_path):
    """``f1`` is higher-is-better; model_best.pt must hold the epoch with the highest val f1,
    not the lowest."""
    train_loader, val_loader = separable_classifier_loaders()
    config = classifier_config(5, evaluation={"selection_metric": "f1"})
    run = trainer_run(config, tmp_path / "out", project=tmp_path, has_val_loader=True,
                      id="auto-run-65")
    run = train(run, train_loader, val_loader=val_loader)

    assert run.status == "completed", run.status_error
    history = run.metrics_history
    f1_by_epoch = {epoch: r["val_f1"] for epoch, r in enumerate(history, start=1)}
    best_epoch = max(f1_by_epoch, key=lambda e: f1_by_epoch[e])
    worst_epoch = min(f1_by_epoch, key=lambda e: f1_by_epoch[e])
    assert f1_by_epoch[best_epoch] > f1_by_epoch[worst_epoch]  # the run must actually vary

    best = torch.load(tmp_path / "out" / "model_best.pt", weights_only=False)
    assert best["epoch"] == best_epoch
    assert best[METRICS_KEY]["val_f1"] == pytest.approx(f1_by_epoch[best_epoch], abs=1e-6)
    assert run.best_metric == pytest.approx(f1_by_epoch[best_epoch], abs=1e-6)


def test_a_run_selecting_on_a_metric_its_task_does_not_produce_fails_naming_both(tmp_path):
    """``f1`` is declared (a detection/classification metric) but regression's own ``evaluate()``
    never produces it; the run must fail naming the requested metric and the keys validation did
    produce, not silently fall back to the training loss under a name nobody chose."""
    train_loader, val_loader = opposed_regression_loaders(TRAIN_INTENSITIES, VAL_INTENSITIES)
    run = trainer_run(_config({"selection_metric": "f1"}), tmp_path / "out", project=tmp_path,
                      has_val_loader=True, id="auto-run-66")
    run = train(run, train_loader, val_loader=val_loader)

    assert run.status == "failed"
    assert "'f1'" in run.status_error
    assert "'val_f1'" in run.status_error
    assert "val_loss" in run.status_error and "val_mae" in run.status_error


def test_a_loss_selected_run_with_no_validation_loader_still_completes_and_selects_its_lowest_loss(
    tmp_path,
):
    """No validation loader means no metric but the training loss exists; a run selecting on the
    default (loss) metric must still complete and choose the lowest-loss epoch."""
    train_loader, _ = opposed_regression_loaders(TRAIN_INTENSITIES, VAL_INTENSITIES)
    run = trainer_run(_config(), tmp_path / "out", project=tmp_path, has_val_loader=False,
                      id="auto-run-67")
    run = train(run, train_loader, val_loader=None)

    assert run.status == "completed", run.status_error
    history = run.metrics_history
    assert len(history) == 3
    for record in history:
        assert record["selection_metric"] == "loss"
        assert "val_loss" not in record
    assert run.best_metric == pytest.approx(min(r["selection"] for r in history), abs=1e-6)

    best = torch.load(tmp_path / "out" / "model_best.pt", weights_only=False)
    assert best[METRICS_KEY]["selection"] == pytest.approx(run.best_metric, abs=1e-6)
