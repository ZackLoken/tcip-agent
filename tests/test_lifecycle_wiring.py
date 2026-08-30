"""Training↔experiment↔registry lifecycle wiring (registration from an experiment binds the
registry entry to the run through experiment_id, over the digest completion recorded, and never
fabricates metrics)."""

import logging

import tcip_store as ts


def test_register_model_from_experiment_links_lineage_with_no_checkpoint_metrics(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.chdir(tmp_path)  # .tcip/experiments lives under cwd
    from tcip_mcp.experiments import (
        complete_run,
        create_experiment,
        lineage_key,
        log_metrics,
        register_model_from_experiment,
    )
    from tcip_mcp.model_registry import ModelRegistry

    create_experiment("exp1", {"model_source": {"builder": "x:y"}}, data_source="imgs")
    log_metrics("exp1", 1, {"map50": 0.5})
    log_metrics("exp1", 2, {"map50": 0.81})

    # A checkpoint that will not load as a torch payload: the run's own metrics.jsonl (map50
    # 0.81 above) describes a different epoch than this checkpoint, so it is never substituted.
    ckpt = tmp_path / "model_best.pt"
    ckpt.write_bytes(b"weights")
    assert "error" not in complete_run("exp1", str(ckpt))

    with caplog.at_level(logging.WARNING):
        result = register_model_from_experiment("exp1", str(ckpt))
    assert result["registered"] == "exp1"
    assert result["metrics"] == {}
    assert str(ckpt) in caplog.text  # the load failure is logged, naming the file

    # The registry entry's experiment_id binds it to the run, with an honest empty/null pairing.
    m = ModelRegistry(str(tmp_path)).get_model("exp1")
    assert m is not None
    assert m["experiment_id"] == "exp1"
    assert m["metrics"] == {}
    assert m["metrics_source"] is None

    # complete_run's own assert above wrote the lineage pointer and its digest; registration
    # writes nothing there.
    lineage = ts.read(lineage_key("exp1"))
    assert lineage["model_weights"] == str(ckpt)


def test_register_model_from_experiment_unknown_experiment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import register_model_from_experiment
    assert "error" in register_model_from_experiment("does_not_exist", "x.pt")


def test_a_stock_trainer_run_registers_with_trainer_source_and_the_best_epochs_metrics(
    tmp_path, monkeypatch
):
    """A completed default_train run (no training_source in config) registers
    metrics_source='trainer', the platform's own measurement, carrying model_best.pt's own
    best-epoch metrics; model_final.pt's metrics is a dict too, the last completed epoch's."""
    monkeypatch.chdir(tmp_path)
    import torch
    from torch.utils.data import DataLoader

    from tcip_mcp.experiments import create_experiment, update_status
    from tcip_mcp.model_registry import ModelRegistry
    from tcip_mcp.pipelines.training.envelope import TrainContext, run_training_envelope
    from tcip_mcp.pipelines.training.generic_trainer import task_collate
    from tcip_mcp.pipelines.training.run_registry import create_run
    from tests.tiny_trainer_fixtures import ConstantImageDataset

    train_ds = ConstantImageDataset([0.1, 0.3, 0.5, 0.7], [0.2, 0.6, 1.0, 1.4])
    val_ds = ConstantImageDataset([0.2, 0.6], [0.4, 1.2])
    collate = task_collate("regression")
    train_loader = DataLoader(train_ds, batch_size=2, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=2, collate_fn=collate)

    config = {
        "model_source": {"builder": "tests.tiny_trainer_fixtures:build_mean_intensity_regressor",
                         "builder_kwargs": {"init_weight": 0.0}, "task": "regression", "in_chans": 1},
        "device": "cpu", "mixed_precision": False,
        "stages": [{"freeze_to": 0, "epochs": 2}],
        "optimizer": {"name": "adamw", "backbone_lr": 0.05, "head_lr": 0.05, "weight_decay": 0.0},
        "checkpoint_every_n_epochs": 0, "early_stopping": {"enabled": False},
    }
    create_experiment("exp-trainer-source", config, data_source="imgs")
    update_status("exp-trainer-source", "running")
    run = create_run(config, str(tmp_path / "out"))
    ctx = TrainContext(run=run, train_loader=train_loader, val_loader=val_loader,
                       task="regression", experiment_id="exp-trainer-source")
    run_training_envelope(ctx)

    assert run.status == "completed", run.error
    entry = ModelRegistry(str(tmp_path)).get_model("exp-trainer-source")
    assert entry is not None
    assert entry["metrics_source"] == "trainer"

    best = torch.load(tmp_path / "out" / "model_best.pt", weights_only=False)
    assert entry["metrics"] == best["metrics"]
    assert entry["metrics"]["epoch"] == best["epoch"]

    final = torch.load(tmp_path / "out" / "model_final.pt", weights_only=False)
    assert isinstance(final["metrics"], dict)  # not the old per-epoch list


def test_a_diverged_stock_run_registers_with_trainer_source_and_a_null_loss(tmp_path, monkeypatch):
    """A run whose loss never becomes a finite number never improves on the losing-side
    sentinel, so it writes no model_best.pt; the last completed epoch's checkpoint metrics
    (model_final.pt) still normalize the diverged loss to null plus a state companion, exactly
    as the run's own metrics log does, and still register cleanly with metrics_source='trainer'."""
    monkeypatch.chdir(tmp_path)
    import torch
    from torch.utils.data import DataLoader

    from tcip_mcp.experiments import create_experiment, update_status
    from tcip_mcp.model_registry import ModelRegistry
    from tcip_mcp.pipelines.training.envelope import TrainContext, run_training_envelope
    from tcip_mcp.pipelines.training.generic_trainer import task_collate
    from tcip_mcp.pipelines.training.run_registry import create_run
    from tests.tiny_trainer_fixtures import ConstantImageDataset

    train_ds = ConstantImageDataset([0.1, 0.3, 0.5, 0.7], [0.2, 0.6, 1.0, 1.4])
    val_ds = ConstantImageDataset([0.2, 0.6], [0.4, 1.2])
    collate = task_collate("regression")
    train_loader = DataLoader(train_ds, batch_size=2, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=2, collate_fn=collate)

    config = {
        "model_source": {"builder": "tests.tiny_trainer_fixtures:build_always_diverged_model",
                         "task": "regression", "in_chans": 1},
        "device": "cpu", "mixed_precision": False,
        "stages": [{"freeze_to": 0, "epochs": 2}],
        "optimizer": {"name": "adamw", "backbone_lr": 0.05, "head_lr": 0.05, "weight_decay": 0.0},
        "checkpoint_every_n_epochs": 0, "early_stopping": {"enabled": False},
    }
    create_experiment("exp-diverged", config, data_source="imgs")
    update_status("exp-diverged", "running")
    run = create_run(config, str(tmp_path / "out"))
    ctx = TrainContext(run=run, train_loader=train_loader, val_loader=val_loader,
                       task="regression", experiment_id="exp-diverged")
    run_training_envelope(ctx)

    assert run.status == "completed", run.error
    assert not (tmp_path / "out" / "model_best.pt").exists()

    final = torch.load(tmp_path / "out" / "model_final.pt", weights_only=False)
    assert final["metrics"]["train_loss"] is None
    assert final["metrics"]["train_loss_state"] == "nan"
    assert final["metrics"]["selection"] is None
    assert final["metrics"]["val_loss"] is None

    entry = ModelRegistry(str(tmp_path)).get_model("exp-diverged")
    assert entry is not None
    assert entry["metrics_source"] == "trainer"
    assert entry["metrics"]["train_loss"] is None
    assert entry["checkpoint_path"].endswith("model_final.pt")
