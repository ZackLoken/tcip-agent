"""The number an epoch reports under ``selection`` is the objective the checkpoint was chosen by.

``metrics.jsonl``, the TensorBoard row, the in-memory history and the HPO ``epoch_callback`` all
carry the same ``selection`` field, and a reader takes it for the value that drove
``model_best.pt`` and early stopping. These runs keep the training loss far away from the
selection objective, so a record carrying the training loss under that key is visible.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader

from tcip_mcp.pipelines.training.generic_trainer import create_run, task_collate, train
from tests.tiny_trainer_fixtures import ConstantImageDataset

BUILDER = "tests.tiny_trainer_fixtures:build_mean_intensity_regressor"

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
    train_loader, val_loader = _loaders()
    out_dir = tmp_path / "out"
    callbacks: list[dict] = []
    run = create_run(_config(), str(out_dir))
    run = train(run, train_loader, val_loader=val_loader, task="regression",
                epoch_callback=lambda epoch, metrics: callbacks.append(dict(metrics)))

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

    persisted = [json.loads(line) for line in
                 (out_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
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
