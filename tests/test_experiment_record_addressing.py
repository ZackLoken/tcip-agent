"""Where an experiment's members live, and who is allowed to write them.

The record's own module is the one place a member is addressed, so a consumer that resolves a
member on its own can no longer disagree with the module that declares it, and a run's epoch
rows have exactly one writer rather than one per training path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_mcp import experiments


def test_an_experiment_id_that_would_escape_the_store_is_refused(tmp_path):
    """A run id arrives from a URL, so a member key must not accept one that walks out.

    The refusal lives with the key constructor rather than at each caller, since the caller
    that forgets it is the one that hands an untrusted name straight to a path join.
    """
    from tcip_store import BadKey

    for escaping in ("../elsewhere", "sub/dir", r"other\dir", ".."):
        with pytest.raises(BadKey):
            experiments.metrics_key(escaping)


def test_an_ordinary_experiment_id_still_addresses_its_own_members(tmp_path):
    """The refusal above must not cost an ordinary run its record."""
    experiment_id = "exp-014-hazelnut-catkin-det_run_20260114_a1b2"
    experiments.create_experiment(experiment_id, {"model_source": {"builder": "m:f"}})
    experiments.log_metrics(experiment_id, 1, {"loss": 0.5})

    assert experiments.experiment_exists(experiment_id)
    assert [row["epoch"] for row in experiments.read_metrics(experiment_id)] == [1]
    key = experiments.metrics_key(experiment_id)
    assert key.parts == (experiment_id, "metrics")


def test_a_split_manifest_is_read_from_the_store_the_experiment_module_resolves(
    tmp_path, monkeypatch
):
    """One resolver for the experiment store, so a rebound store is not read from two places.

    A consumer that rebuilds ``<platform root>/.tcip/experiments`` of its own reads the
    default location even when the store itself has been moved, and then reports a run whose
    manifest exists as having no recorded training membership at all: a disjointness check
    that silently cannot answer.
    """
    from tcip_mcp.pipelines.operating_point import _train_disjointness
    from tcip_mcp.utils.atomic_io import atomic_write_json

    elsewhere = tmp_path / "relocated_experiments"
    monkeypatch.setattr(experiments, "EXPERIMENTS_DIR", elsewhere)
    experiment_id = "exp-015-chestnut-burr-det"
    experiments.create_experiment(experiment_id, {"model_source": {"builder": "m:f"}})
    manifest = elsewhere / experiment_id / "split.json"
    atomic_write_json(manifest, {"train": ["img_001"], "group_by": "stem"})
    assert manifest.is_file()

    checked = _train_disjointness(experiment_id, {"img_002"}, {"img_003"})
    assert checked["unresolvable"] is False
    assert checked["checked"] is True
    assert checked["leaked_stems"] == []


def test_one_epoch_logs_one_row_when_the_run_writes_where_its_record_lives(tmp_path):
    """A run's output dir is the experiment's own directory by default, so two writers on
    that log wrote every epoch twice, in two different row shapes, into one file.

    The trainer hands its row to the sink instead of writing a log beside its weights, so the
    row count is the epoch count whether or not the two directories coincide.
    """
    pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    from tcip_web.routes._metrics_common import read_metrics_file

    from tcip_mcp.pipelines.training.envelope import TrainContext
    from tcip_mcp.pipelines.training.generic_trainer import create_run, task_collate
    from tests.tiny_trainer_fixtures import ConstantImageDataset

    experiment_id = "exp-016-currant-berry-reg"
    config = {
        "model_source": {
            "builder": "tests.tiny_trainer_fixtures:build_mean_intensity_regressor",
            "builder_kwargs": {"init_weight": 0.0},
            "task": "regression",
            "in_chans": 1,
        },
        "device": "cpu",
        "mixed_precision": False,
        "stages": [{"freeze_to": 0, "epochs": 2}],
        "optimizer": {"name": "adamw", "backbone_lr": 0.05, "head_lr": 0.05, "weight_decay": 0.0},
        "early_stopping": {"enabled": False},
        "checkpoint_every_n_epochs": 0,
    }
    experiments.create_experiment(experiment_id, config)
    experiments.update_status(experiment_id, "running")

    # The default output dir is the experiment's own directory, which is where the two writers
    # used to meet.
    record_dir = experiments.experiments_dir() / experiment_id
    run = create_run(config, str(record_dir))
    dataset = ConstantImageDataset([0.2, 0.8], [1.0, 4.0])
    loader = DataLoader(dataset, batch_size=2, collate_fn=task_collate("regression"))
    ctx = TrainContext(run=run, train_loader=loader, val_loader=None, task="regression",
                       experiment_id=experiment_id)
    ctx.default_train()

    assert run.status == "completed", run.error
    rows = read_metrics_file(record_dir / "metrics.jsonl")["metrics"]
    assert [row["epoch"] for row in rows] == [1, 2]
    assert all(row.get("timestamp") for row in rows)


def test_a_run_with_no_experiment_record_still_logs_beside_its_own_artifacts(tmp_path):
    """An HPO trial has no experiment record, and the Tuning view reads its rows from the
    trial's own directory: routing every row through the experiment log would leave it blank.
    """
    from tcip_web.routes._metrics_common import read_metrics_file

    from tcip_mcp.pipelines.training.envelope import TrainContext
    from tcip_mcp.pipelines.training.generic_trainer import create_run

    trial_dir = tmp_path / "sweep" / "trial_7"
    run = create_run({"model_source": {}}, str(trial_dir))
    ctx = TrainContext(run=run, train_loader=None, val_loader=None, task="regression",
                       experiment_id=None)
    ctx._epoch_sink(1, {"val_loss": 0.25})

    body = read_metrics_file(Path(trial_dir) / "metrics.jsonl")
    assert body["exists"] is True
    assert body["metrics"] == [{"epoch": 1, "val_loss": 0.25}]
