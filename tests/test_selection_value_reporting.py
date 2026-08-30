"""The number an epoch reports under ``selection`` is the objective the checkpoint was chosen by.

``metrics.jsonl``, the TensorBoard row, the in-memory history and the HPO ``epoch_callback`` all
carry the same ``selection`` field, and a reader takes it for the value that drove
``model_best.pt`` and early stopping. These runs keep the training loss far away from the
selection objective, so a record carrying the training loss under that key is visible.
"""

from __future__ import annotations


import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader

from tcip_mcp.pipelines.training.generic_trainer import train
from tcip_mcp.pipelines.training.collation import task_collate
from tcip_mcp.pipelines.training.run_registry import create_run
from tests.tiny_trainer_fixtures import ConstantImageClassDataset, ConstantImageDataset

BUILDER = "tests.tiny_trainer_fixtures:build_mean_intensity_regressor"
CLASSIFIER_BUILDER = "tests.tiny_trainer_fixtures:build_mean_intensity_classifier"

TRAIN_INTENSITIES = [0.10, 0.25, 0.40, 0.55, 0.70, 0.85]
VAL_INTENSITIES = [0.15, 0.35, 0.60, 0.90]


def _loaders():
    """A training loader fit by weight +2 and a holdout loader fit by weight -5."""
    train_ds = ConstantImageDataset(
        TRAIN_INTENSITIES, [2.0 * c for c in TRAIN_INTENSITIES])
    val_ds = ConstantImageDataset(
        VAL_INTENSITIES, [-5.0 * c for c in VAL_INTENSITIES], height=8, width=12)
    collate = task_collate("regression")
    return (DataLoader(train_ds, batch_size=2, collate_fn=collate),
            DataLoader(val_ds, batch_size=2, collate_fn=collate))


def _config(evaluation: dict | None = None) -> dict:
    config = {
        "model_source": {"builder": BUILDER, "builder_kwargs": {"init_weight": 0.0},
                         "task": "regression", "in_chans": 1},
        "device": "cpu",
        "mixed_precision": False,
        "stages": [{"freeze_to": 0, "epochs": 3}],
        "optimizer": {"name": "adamw", "backbone_lr": 0.05, "head_lr": 0.05, "weight_decay": 0.0},
        "checkpoint_every_n_epochs": 0,
        "early_stopping": {"enabled": False},
    }
    if evaluation is not None:
        config["evaluation"] = evaluation
    return config


def test_epoch_record_reports_the_value_the_best_checkpoint_was_chosen_by(tmp_path):
    """The chosen epoch's recorded ``selection``, the copy embedded in ``model_best.pt``, the
    ``metrics.jsonl`` line and the callback payload all carry ``run.best_metric``."""
    from tcip_mcp.experiments import create_experiment, read_metrics
    from tcip_mcp.pipelines.training.envelope import TrainContext

    train_loader, val_loader = _loaders()
    out_dir = tmp_path / "out"
    callbacks: list[dict] = []
    config = _config()
    create_experiment("exp-selection", config)
    run = create_run(config, str(out_dir))
    # The production wiring: the trainer hands each row to the envelope's sink, which logs it
    # to the experiment's own record and fires the hook a trial prunes on.
    ctx = TrainContext(run=run, train_loader=train_loader, val_loader=val_loader,
                       task="regression", experiment_id="exp-selection",
                       epoch_hook=lambda epoch, metrics: callbacks.append(dict(metrics)))
    run = ctx.default_train()

    assert run.status == "completed", run.error
    history = run.metrics_history
    assert len(history) == 3

    best = torch.load(out_dir / "model_best.pt", weights_only=False)
    chosen = [r for r in history if r["epoch"] == best["epoch"]]
    assert len(chosen) == 1
    chosen_record = chosen[0]

    assert chosen_record["selection"] == pytest.approx(run.best_metric, abs=1e-6)
    assert best["metrics"]["selection"] == pytest.approx(run.best_metric, abs=1e-6)
    # A regression run resolves its selection metric to the holdout loss, which this fixture
    # keeps far away from the training loss.
    assert chosen_record["selection"] == pytest.approx(chosen_record["val_loss"], abs=1e-6)
    assert chosen_record["selection"] != pytest.approx(chosen_record["train_loss"], rel=0.2)

    persisted = read_metrics("exp-selection")
    assert [r["selection"] for r in persisted] == [r["selection"] for r in history]
    assert [r["selection"] for r in callbacks] == [r["selection"] for r in history]
    assert run.best_metric == pytest.approx(min(r["selection"] for r in history), abs=1e-6)


def test_epoch_record_follows_a_configured_selection_metric(tmp_path):
    """An explicit ``evaluation.selection_metric`` drives both the checkpoint objective and the
    reported ``selection``, so the two still name the same number."""
    train_loader, val_loader = _loaders()
    out_dir = tmp_path / "out"
    run = create_run(_config({"selection_metric": "mae"}), str(out_dir))
    run = train(run, train_loader, val_loader=val_loader, task="regression")

    assert run.status == "completed", run.error
    history = run.metrics_history
    assert len(history) == 3
    for record in history:
        assert record["selection_metric"] == "mae"
        assert record["selection"] == pytest.approx(record["val_mae"], abs=1e-6)
        assert record["selection"] != pytest.approx(record["train_loss"], rel=0.2)
    assert run.best_metric == pytest.approx(min(r["val_mae"] for r in history), abs=1e-6)

    best = torch.load(out_dir / "model_best.pt", weights_only=False)
    assert best["metrics"]["selection"] == pytest.approx(run.best_metric, abs=1e-6)


def test_a_run_selecting_on_f1_keeps_its_highest_f1_checkpoint(tmp_path):
    """``f1`` is higher-is-better; model_best.pt must hold the epoch with the highest val f1,
    not the lowest, and early stopping must track improvement in the same direction."""
    train_ds = ConstantImageClassDataset(
        [-2.0, -1.5, -1.0, 1.0, 1.5, 2.0], [0, 0, 0, 1, 1, 1])
    val_ds = ConstantImageClassDataset([-1.8, -0.4, 0.4, 1.8], [0, 0, 1, 1])
    collate = task_collate("classification")
    train_loader = DataLoader(train_ds, batch_size=3, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=4, collate_fn=collate)

    config = {
        "model_source": {"builder": CLASSIFIER_BUILDER, "builder_kwargs": {"init_weight": -1.0},
                         "task": "classification", "in_chans": 1},
        "device": "cpu",
        "mixed_precision": False,
        "stages": [{"freeze_to": 0, "epochs": 5}],
        "optimizer": {"name": "adamw", "backbone_lr": 0.2, "head_lr": 0.2, "weight_decay": 0.0},
        "checkpoint_every_n_epochs": 0,
        "early_stopping": {"enabled": False},
        "evaluation": {"selection_metric": "f1"},
    }
    run = create_run(config, str(tmp_path / "out"))
    run = train(run, train_loader, val_loader=val_loader, task="classification")

    assert run.status == "completed", run.error
    history = run.metrics_history
    f1_by_epoch = {r["epoch"]: r["val_f1"] for r in history}
    best_epoch = max(f1_by_epoch, key=lambda e: f1_by_epoch[e])
    worst_epoch = min(f1_by_epoch, key=lambda e: f1_by_epoch[e])
    assert f1_by_epoch[best_epoch] > f1_by_epoch[worst_epoch]  # the run must actually vary

    best = torch.load(tmp_path / "out" / "model_best.pt", weights_only=False)
    assert best["epoch"] == best_epoch
    assert best["metrics"]["val_f1"] == pytest.approx(f1_by_epoch[best_epoch], abs=1e-6)
    assert run.best_metric == pytest.approx(f1_by_epoch[best_epoch], abs=1e-6)


def test_a_run_selecting_on_a_metric_its_task_does_not_produce_fails_naming_both(tmp_path):
    """``f1`` is declared (a detection/classification metric) but regression's own ``evaluate()``
    never produces it; the run must fail naming the requested metric and the keys validation did
    produce, not silently fall back to the training loss under a name nobody chose."""
    train_loader, val_loader = _loaders()
    run = create_run(_config({"selection_metric": "f1"}), str(tmp_path / "out"))
    run = train(run, train_loader, val_loader=val_loader, task="regression")

    assert run.status == "failed"
    assert "'f1'" in run.error
    assert "'val_f1'" in run.error
    assert "val_loss" in run.error and "val_mae" in run.error


def test_a_loss_selected_run_with_no_validation_loader_still_completes_and_selects_its_lowest_loss(
    tmp_path,
):
    """No validation loader means no metric but the training loss exists; a run selecting on the
    default (loss) metric must still complete and choose the lowest-loss epoch."""
    train_loader, _ = _loaders()
    run = create_run(_config(), str(tmp_path / "out"))
    run = train(run, train_loader, val_loader=None, task="regression")

    assert run.status == "completed", run.error
    history = run.metrics_history
    assert len(history) == 3
    for record in history:
        assert record["selection_metric"] == "loss"
        assert "val_loss" not in record
    assert run.best_metric == pytest.approx(min(r["selection"] for r in history), abs=1e-6)

    best = torch.load(tmp_path / "out" / "model_best.pt", weights_only=False)
    assert best["metrics"]["selection"] == pytest.approx(run.best_metric, abs=1e-6)
