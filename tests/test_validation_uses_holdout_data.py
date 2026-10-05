"""The per-epoch validation pass scores the held-out loader, never the training loader.

Every ``val_`` metric, the selection value, the best-checkpoint choice and early stopping are
only defensible if they were measured on data the epoch did not just train on. These runs give
the train and val loaders loss landscapes that pull in opposite directions, so a validation pass
that scored the training data would report visibly different numbers and pick a different epoch.
"""

from __future__ import annotations


import pytest

torch = pytest.importorskip("torch")
from tcip_mcp.pipelines.model_build import METRICS_KEY
from tcip_mcp.pipelines.training.evaluation import evaluate
from tcip_mcp.pipelines.training.generic_trainer import train
from tests.tiny_trainer_fixtures import (
    capture_model,
    opposed_regression_loaders,
    trainer_run,
    write_regression_dataset,
)

BUILDER = "tests.tiny_trainer_fixtures:build_mean_intensity_regressor"

# Train values sit at 2x the frame intensity, holdout values at -5x: the two loaders are fit by
# opposite-signed weights, so training progress on one is a regression on the other.
TRAIN_INTENSITIES = [0.10, 0.25, 0.40, 0.55, 0.70, 0.85]
VAL_INTENSITIES = [0.15, 0.35, 0.60, 0.90]


def _config(out_dir, *, epochs: int, early_stopping: dict) -> dict:
    return {
        "model_source": {"builder": BUILDER, "builder_kwargs": {"init_weight": 0.0},
                         "task": "regression"},
        "data": {"num_channels": 1, "scope": {}},
        "device": "cpu",
        "mixed_precision": False,
        "stages": [{"freeze_to": 0, "epochs": epochs}],
        "optimizer": {"name": "adamw", "backbone_lr": 0.05, "head_lr": 0.05, "weight_decay": 0.0},
        "checkpoint_every_n_epochs": 0,
        "early_stopping": early_stopping,
    }


def test_recorded_val_metrics_match_an_evaluation_of_the_holdout_loader(tmp_path, monkeypatch):
    """The ``val_`` metrics an epoch records equal a real evaluation of that epoch's model on the
    holdout loader, and differ from the same evaluation run over the training loader."""
    train_loader, val_loader = opposed_regression_loaders(TRAIN_INTENSITIES, VAL_INTENSITIES)
    models: list = []
    capture_model(monkeypatch, models)

    from tcip_mcp.experiments import METRICS_FILE, observe, read_rows
    from tcip_mcp.pipelines.training.envelope import TrainContext
    from tcip_mcp.pipelines.training.run_registry import observed_run
    from tests._verified_checkpoint_fixtures import opened_run

    images_dir, csv_path = write_regression_dataset(
        tmp_path / "ds", TRAIN_INTENSITIES, [2.0 * c for c in TRAIN_INTENSITIES])
    config = _config(tmp_path, epochs=1, early_stopping={"enabled": False})
    config["data"] = {**config["data"], "images_dir": str(images_dir),
                      "labels_dir": str(csv_path), "split": {"seed": 1, "val_ratio": 0.15}}
    out_dir = opened_run(tmp_path, config)
    run = observed_run(observe(out_dir))
    # The production wiring: the trainer hands each row to the envelope's sink, which logs it
    # to the run's own metrics log.
    ctx = TrainContext(run=run, train_loader=train_loader, val_loader=val_loader)
    run = ctx.default_train()

    assert run.status == "completed", run.error
    assert len(models) == 1
    model = models[0]

    device = torch.device("cpu")
    on_holdout = evaluate(model, val_loader, device, "regression", dims={"in_chans": 1})
    on_training = evaluate(model, train_loader, device, "regression", dims={"in_chans": 1})
    # The two loaders must be distinguishable at all, or nothing below can discriminate.
    assert on_holdout["loss"] > 3.0 * on_training["loss"] > 0.0
    assert on_holdout["mae"] != pytest.approx(on_training["mae"], rel=0.1)

    record = run.metrics_history[-1]
    assert record["val_loss"] == pytest.approx(on_holdout["loss"], abs=1e-6)
    assert record["val_mae"] == pytest.approx(on_holdout["mae"], abs=1e-6)
    assert record["val_rmse"] == pytest.approx(on_holdout["rmse"], abs=1e-6)
    # A regression run selects by val_loss, so the checkpoint objective is the holdout number too.
    assert run.best_metric == pytest.approx(on_holdout["loss"], abs=1e-6)

    persisted = read_rows(out_dir / METRICS_FILE)[0]
    assert len(persisted) == 1
    assert persisted[0]["val_loss"] == pytest.approx(on_holdout["loss"], abs=1e-6)


def test_best_checkpoint_and_early_stopping_follow_the_holdout_loader(tmp_path, monkeypatch):
    """With holdout loss worsening while training loss improves, the run stops early and keeps the
    first epoch's checkpoint: both decisions read the holdout loader, not the training one."""
    train_loader, val_loader = opposed_regression_loaders(TRAIN_INTENSITIES, VAL_INTENSITIES)
    models: list = []
    capture_model(monkeypatch, models)

    out_dir = tmp_path / "out"
    config = _config(out_dir, epochs=4,
                     early_stopping={"enabled": True, "patience": 1, "min_delta": 1e-4})
    run = trainer_run(config, out_dir, project=tmp_path, has_val_loader=True, id="auto-run-76")
    run = train(run, train_loader, val_loader=val_loader)

    assert run.status == "completed", run.error
    history = run.metrics_history
    assert len(history) == 2, history
    # The two directions disagree: training improves epoch over epoch while holdout degrades.
    assert history[1]["train_loss"] < history[0]["train_loss"]
    assert history[1]["val_loss"] > history[0]["val_loss"]

    assert run.best_metric == pytest.approx(history[0]["val_loss"], abs=1e-6)
    best = torch.load(out_dir / "model_best.pt", weights_only=False)
    assert best["epoch"] == 1
    assert best[METRICS_KEY]["val_loss"] == pytest.approx(history[0]["val_loss"], abs=1e-6)
