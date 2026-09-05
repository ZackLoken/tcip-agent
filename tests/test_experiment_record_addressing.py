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
    experiment_id = "exp-014-currant-bud-det_run_20260114_a1b2"
    experiments.create_experiment(experiment_id, {"model_source": {"builder": "m:f"}})
    experiments.log_metrics(experiment_id, 1, {"loss": 0.5})

    assert experiments.experiment_exists(experiment_id)
    assert [row["epoch"] for row in experiments.read_metrics(experiment_id)] == [1]
    key = experiments.metrics_key(experiment_id)
    assert key.parts == (experiment_id, "metrics")


def test_an_experiment_payload_that_json_cannot_hold_is_refused_at_the_entry_point(tmp_path):
    """Config, metrics and lineage arrive as the caller's own dicts, so the field that will
    not encode is named here rather than surfacing as a codec failure naming only the file.

    A refusal matters most for a measurement: an unserializable value used to become its
    repr, which reads as a real recorded number forever after.
    """
    experiment_id = "exp-020-json-boundary"
    experiments.create_experiment(experiment_id, {"model_source": {"builder": "m:f"}})

    with pytest.raises(TypeError) as config_refused:
        experiments.create_experiment("exp-021-json-boundary", {"weights": Path("model_best.pt")})
    assert "config.weights" in str(config_refused.value)

    with pytest.raises(TypeError) as overwrite_refused:
        experiments.overwrite_config_if_pristine(experiment_id, {"weights": Path("model_best.pt")})
    assert "config.weights" in str(overwrite_refused.value)

    with pytest.raises(ValueError) as metrics_refused:
        experiments.log_metrics(experiment_id, 1, {"loss": float("nan")})
    assert "metrics.loss" in str(metrics_refused.value)

    with pytest.raises(TypeError) as lineage_refused:
        experiments.update_lineage(experiment_id, predictions=Path("predictions/live"))
    assert "updates.predictions" in str(lineage_refused.value)

    assert not experiments.experiment_exists("exp-021-json-boundary")
    assert experiments.read_metrics(experiment_id) == []


def test_an_ordinary_experiment_payload_is_still_written_by_every_entry_point(tmp_path):
    """The refusal above must not cost an ordinary run its config, its rows or its lineage."""
    experiment_id = "exp-022-json-boundary"

    experiments.create_experiment(experiment_id, {"model_source": {"builder": "m:f"}})
    experiments.overwrite_config_if_pristine(experiment_id, {"model_source": {"builder": "m:g"}})
    experiments.log_metrics(experiment_id, 1, {"loss": 0.5})
    experiments.update_lineage(experiment_id, predictions="predictions/live")

    record = experiments.get_experiment(experiment_id)
    assert record["config"]["model_source"]["builder"] == "m:g"
    assert [row["epoch"] for row in experiments.read_metrics(experiment_id)] == [1]
    assert record["lineage"]["predictions"] == "predictions/live"


def test_a_split_manifest_is_read_from_the_store_the_experiment_module_resolves(
    tmp_path, monkeypatch
):
    """One resolver for the experiment store, so a rebound store is not read from two places.

    A consumer that rebuilds ``<platform root>/.tcip/experiments`` of its own reads the
    default location even when the store itself has been moved, and then reports a run whose
    manifest exists as having no recorded training membership at all: a disjointness check
    that silently cannot answer.
    """
    from tcip_store import store

    from tcip_mcp.pipelines.operating_point import _train_disjointness

    elsewhere = tmp_path / "relocated_experiments"
    monkeypatch.setattr(experiments, "EXPERIMENTS_DIR", elsewhere)
    experiment_id = "exp-015-chestnut-burr-det"
    experiments.create_experiment(experiment_id, {"model_source": {"builder": "m:f"}})
    store.replace(experiments.split_key(experiment_id),
                  {"train": ["img_001"], "group_by": "stem"})

    # A hit here proves _train_disjointness resolved the relocated store, not the default one.
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

    from tcip_store import read_log

    from tcip_mcp.pipelines.training.envelope import TrainContext
    from tcip_mcp.pipelines.training.collation import task_collate
    from tcip_mcp.pipelines.training.run_registry import create_run
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
    rows = read_log(experiments.metrics_key(experiment_id)).records
    assert [row["epoch"] for row in rows] == [1, 2]
    assert all(row.get("timestamp") for row in rows)


def test_a_trial_name_that_would_escape_its_sweep_is_refused(tmp_path):
    """A trial's config and its metrics are addressed by the same name, so both constructors
    refuse one that walks out rather than only the one a route reaches.
    """
    from tcip_store import BadKey

    from tcip_mcp.tools.training_tools import trial_config_key, trial_metrics_key

    sweep = tmp_path / "sweep"
    for escaping in ("../elsewhere", "sub/dir", r"other\dir", ".."):
        with pytest.raises(BadKey):
            trial_config_key(sweep, escaping)
        with pytest.raises(BadKey):
            trial_metrics_key(sweep, escaping)


def test_an_ordinary_trial_name_still_addresses_its_own_documents(tmp_path):
    from tcip_mcp.tools.training_tools import trial_config_key, trial_metrics_key

    sweep = tmp_path / "sweep"
    config = trial_config_key(sweep, "trial_00000")
    metrics = trial_metrics_key(sweep, "trial_00000")

    assert config.parts == ("trial_00000", "resolved_config")
    assert metrics.parts == ("trial_00000", "metrics")
    assert config.root == metrics.root == str(sweep.resolve())


def test_a_run_with_no_experiment_record_still_logs_beside_its_own_artifacts(tmp_path):
    """An HPO trial has no experiment record, and the Tuning view reads its rows from the
    trial's own directory: routing every row through the experiment log would leave it blank.
    """
    from tcip_store import read_log

    from tcip_mcp.pipelines.training.envelope import TrainContext
    from tcip_mcp.pipelines.training.run_registry import create_run
    from tcip_mcp.tools.training_tools import trial_metrics_key

    trial_dir = tmp_path / "sweep" / "trial_7"
    run = create_run({"model_source": {}}, str(trial_dir))
    ctx = TrainContext(run=run, train_loader=None, val_loader=None, task="regression",
                       experiment_id=None)
    ctx._epoch_sink(1, {"val_loss": 0.25})

    page = read_log(trial_metrics_key(trial_dir.parent, trial_dir.name))
    assert page.records == [{"epoch": 1, "val_loss": 0.25}]
