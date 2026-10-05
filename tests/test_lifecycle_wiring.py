"""Training to registry lifecycle wiring: a run's completion is its registration, carrying the
metrics its own checkpoint recorded and never fabricating any."""

from pathlib import Path

from tests._verified_checkpoint_fixtures import opened_run


def _stock_run(root: Path, builder: dict, epochs: int, experiment_id: str) -> Path:
    """A run under ``root`` over a regression dataset of its own on disk, opened by the
    launcher's own writer and run in-process through the envelope's stock trainer over tiny
    in-memory regression loaders. Returns the run directory."""
    from torch.utils.data import DataLoader

    from tcip_mcp.experiments import RUN_FILE, read_record
    from tcip_mcp.pipelines.training.collation import task_collate
    from tcip_mcp.pipelines.training.envelope import TrainContext, run_training_envelope
    from tcip_mcp.pipelines.training.run_registry import TrainRun, trained_config
    from tests.tiny_trainer_fixtures import ConstantImageDataset, write_regression_dataset

    train_ds = ConstantImageDataset([0.1, 0.3, 0.5, 0.7], [0.2, 0.6, 1.0, 1.4])
    val_ds = ConstantImageDataset([0.2, 0.6], [0.4, 1.2])
    collate = task_collate("regression")
    images_dir, csv_path = write_regression_dataset(
        root / f"{experiment_id}-data", [0.1, 0.3, 0.5, 0.7], [0.2, 0.6, 1.0, 1.4])
    config = {
        "model_source": {**builder, "task": "regression"},
        "data": {"images_dir": str(images_dir), "labels_dir": str(csv_path), "num_channels": 1,
                 "split": {"seed": 0, "val_ratio": 0.15}},
        "device": "cpu", "mixed_precision": False,
        "stages": [{"freeze_to": 0, "epochs": epochs}],
        "optimizer": {"name": "adamw", "backbone_lr": 0.05, "head_lr": 0.05, "weight_decay": 0.0},
        "checkpoint_every_n_epochs": 0, "early_stopping": {"enabled": False},
    }
    run_dir = opened_run(root, config, experiment_id=experiment_id)
    record = read_record(run_dir / RUN_FILE)
    run = TrainRun(id=run_dir.name, config=trained_config(record),
                   objective=record["resolved"]["objective"], project=root,
                   output_dir=str(run_dir))
    run_training_envelope(TrainContext(
        run=run, train_loader=DataLoader(train_ds, batch_size=2, collate_fn=collate),
        val_loader=DataLoader(val_ds, batch_size=2, collate_fn=collate)))
    return run_dir


def _entry(root: Path, name: str) -> dict | None:
    """The registry entry ``root``'s listing names ``name``, or ``None``."""
    from tcip_mcp.model_registry import ModelRegistry

    return next((m for m in ModelRegistry(str(root)).list_models() if m["name"] == name), None)


def test_a_stock_trainer_run_registers_with_trainer_source_and_the_best_epochs_metrics(
    tmp_path,
):
    """A completed stock-trainer run (no training_source in config) registers
    metrics_source='trainer', the platform's own measurement, carrying model_best.pt's own
    best-epoch metrics; model_final.pt's metrics is a dict too, the last completed epoch's."""
    import torch

    from tcip_mcp.experiments import observe

    run_dir = _stock_run(tmp_path, {
        "builder": "tests.tiny_trainer_fixtures:build_mean_intensity_regressor",
        "builder_kwargs": {"init_weight": 0.0}}, 2, "exp-trainer-source")

    assert observe(run_dir).state == "completed", observe(run_dir).final
    entry = _entry(tmp_path, "exp-trainer-source")
    assert entry is not None
    assert entry["experiment_id"] == "exp-trainer-source"
    assert entry["metrics_source"] == "trainer"

    best = torch.load(run_dir / "model_best.pt", weights_only=False)
    assert entry["metrics"] == best["metrics"]
    assert entry["metrics"]["epoch"] == best["epoch"]

    final = torch.load(run_dir / "model_final.pt", weights_only=False)
    assert isinstance(final["metrics"], dict)  # one mapping, never a per-epoch list


def test_a_diverged_stock_run_ends_failed_and_registers_nothing(tmp_path):
    """Two full passes with no finite training loss end the run failed with the reason on its
    final status; the verdict leaves no checkpoint on disk and nothing reaches the registry."""
    from tcip_mcp.experiments import observe

    run_dir = _stock_run(
        tmp_path, {"builder": "tests.tiny_trainer_fixtures:build_always_diverged_model"}, 3,
        "exp-diverged")

    final = observe(run_dir).final
    assert final["state"] == "failed"
    assert "2 consecutive full training passes" in final["error"]
    assert final["checkpoint"] is None
    assert not (run_dir / "model_best.pt").exists()
    assert not (run_dir / "model_final.pt").exists()
    assert _entry(tmp_path, "exp-diverged") is None


def test_a_completed_run_with_a_diverged_val_metric_registers_it_as_null(tmp_path):
    """A healthy training loss completes the run even when a validation metric goes
    non-finite; the final checkpoint and the registry entry carry that metric normalized to
    null plus a state companion, exactly as the run's own metrics log does, with
    metrics_source='trainer'."""
    import torch

    from tcip_mcp.experiments import observe

    run_dir = _stock_run(
        tmp_path, {"builder": "tests.tiny_trainer_fixtures:build_nan_eval_regressor"}, 3,
        "exp-nan-val")

    assert observe(run_dir).state == "completed", observe(run_dir).final

    final = torch.load(run_dir / "model_final.pt", weights_only=False)
    assert final["metrics"]["train_loss"] is not None
    assert final["metrics"]["val_mae"] is None
    assert final["metrics"]["val_mae_state"] == "nan"

    entry = _entry(tmp_path, "exp-nan-val")
    assert entry is not None
    assert entry["metrics_source"] == "trainer"
    assert entry["metrics"]["val_mae"] is None
    assert entry["checkpoint_path"].endswith("model_best.pt")
