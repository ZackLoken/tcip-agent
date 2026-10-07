"""What a training stage starts from, when it ends, and that a resumed run trains as an
uninterrupted one across a stage boundary.

Every run here trains a one-parameter regressor whose holdout loss worsens with every epoch
(``opposed_regression_loaders``), so each stage's best epoch is its first and never its last: a
stage that started from its predecessor's last epoch instead of its best one is visible.
"""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from tcip_mcp.pipelines.model_build import STATE_DICT_KEY  # noqa: E402
from tcip_mcp.pipelines.schemas import SCHEDULER_TYPES  # noqa: E402
from tcip_mcp.pipelines.training import generic_trainer as gt  # noqa: E402
from tcip_mcp.pipelines.training.generic_trainer import train  # noqa: E402
from tests.tiny_trainer_fixtures import (  # noqa: E402
    capture_model,
    opposed_regression_loaders,
    trainer_run,
)

BUILDER = "tests.tiny_trainer_fixtures:build_mean_intensity_regressor"
TRAIN_INTENSITIES = [0.10, 0.25, 0.40, 0.55, 0.70, 0.85]
VAL_INTENSITIES = [0.15, 0.35, 0.60, 0.90]


def _config(stages: list[dict], **extra) -> dict:
    config = {
        "model_source": {"builder": BUILDER, "task": "regression"},
        "data": {"num_channels": 1, "scope": {}},
        "device": "cpu",
        "mixed_precision": False,
        "seed": 3,
        "stages": stages,
        "optimizer": {"name": "adamw", "backbone_lr": 0.05, "head_lr": 0.05, "weight_decay": 0.0},
        "checkpoint_every_n_epochs": 0,
        "early_stopping": {"enabled": False},
    }
    config.update(extra)
    return config


def _train(tmp_path, config: dict, name: str, *, shuffle_seed=None, builder=None, **kwargs):
    train_loader, val_loader = opposed_regression_loaders(
        TRAIN_INTENSITIES, VAL_INTENSITIES, shuffle_seed=shuffle_seed)
    if builder is not None:
        config = {**config, "model_source": {"builder": builder, "task": "regression"}}
    run = trainer_run(config, tmp_path / name, project=tmp_path, has_val_loader=True, id=name)
    return train(run, train_loader, val_loader=val_loader, **kwargs)


def _tensors(value):
    """Every tensor nested in ``value``, in a fixed order."""
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        return [t for key in sorted(value, key=str) for t in _tensors(value[key])]
    if isinstance(value, (list, tuple)):
        return [t for item in value for t in _tensors(item)]
    return []


def test_a_stage_starts_from_the_weights_and_optimizer_state_of_the_prior_stage_best_epoch(
        tmp_path, monkeypatch):
    models: list = []
    capture_model(monkeypatch, models)
    optimizers: list = []
    real_build_optimizer = gt.build_optimizer

    def recording_build_optimizer(*args, **kwargs):
        optimizer = real_build_optimizer(*args, **kwargs)
        optimizers.append(optimizer)
        return optimizer

    monkeypatch.setattr(gt, "build_optimizer", recording_build_optimizer)
    at_stage_start: list = []
    real_build_scheduler = gt._build_scheduler

    def recording_build_scheduler(optimizer, config, epochs):
        weight = models[0].weight
        at_stage_start.append((weight.detach().clone(),
                               optimizer.state[weight]["exp_avg"].clone()
                               if weight in optimizer.state else None))
        return real_build_scheduler(optimizer, config, epochs)

    monkeypatch.setattr(gt, "_build_scheduler", recording_build_scheduler)
    taken: list = []
    real_epoch_state = gt._epoch_state

    def recording_epoch_state(*args, **kwargs):
        state = real_epoch_state(*args, **kwargs)
        taken.append((state, copy.deepcopy(state)))
        return state

    monkeypatch.setattr(gt, "_epoch_state", recording_epoch_state)
    per_epoch: list = []

    def record_epoch(epoch: int, metrics: dict) -> None:
        weight = models[0].weight
        per_epoch.append((weight.detach().clone(),
                          optimizers[-1].state[weight]["exp_avg"].clone()))

    run = _train(tmp_path, _config([{"freeze_to": 0, "epochs": 3}, {"freeze_to": 0, "epochs": 2}]),
                 "pairing", epoch_callback=record_epoch)

    assert run.status == "completed", run.status_error
    stage0 = [m["val_loss"] for m in run.metrics_history if m["stage"] == 0]
    assert stage0 == sorted(stage0) and len(set(stage0)) == 3
    best_weight, best_momentum = per_epoch[0]
    last_weight, _ = per_epoch[2]
    assert not torch.equal(best_weight, last_weight)
    weight, momentum = at_stage_start[1]
    assert torch.equal(weight, best_weight)
    assert momentum is not None and torch.equal(momentum, best_momentum)
    # Every held epoch state is as it was taken, after the later stage's steps.
    assert taken
    for state, as_taken in taken:
        now, then = _tensors(state), _tensors(as_taken)
        assert len(now) == len(then)
        assert all(torch.equal(a, b) for a, b in zip(now, then))


def test_a_stage_ends_on_its_own_plateau_and_the_next_stage_runs(tmp_path):
    stopping = {"enabled": True, "patience": 1, "min_delta": 1e-4}
    run = _train(tmp_path, _config([{"freeze_to": 0, "epochs": 4}, {"freeze_to": 0, "epochs": 3}],
                                   early_stopping=stopping), "plateau")

    assert run.status == "completed", run.status_error
    assert [m["stage"] for m in run.metrics_history] == [0, 0, 1, 1]


def test_a_config_stating_no_stopping_rule_ends_a_stage_on_the_default_plateau(tmp_path):
    """With no ``early_stopping`` block and a validation loader, a stage whose first epoch is its
    best ends after the default patience of non-improving epochs, before its stated length."""
    from tcip_mcp.pipelines.schemas import EarlyStoppingSpec

    config = _config([{"freeze_to": 0, "epochs": 12}])
    del config["early_stopping"]
    run = _train(tmp_path, config, "default-stop")

    assert run.status == "completed", run.status_error
    assert len(run.metrics_history) == 1 + EarlyStoppingSpec.model_validate({}).patience < 12


def test_a_stage_with_no_selectable_epoch_fails_the_run_naming_it(tmp_path):
    """A model whose predictions are all non-finite gives its prediction-derived selection metric
    no value that ranks, so no epoch of the first stage is selectable."""
    config = _config([{"freeze_to": 0, "epochs": 2}, {"freeze_to": 0, "epochs": 2}],
                     evaluation={"selection_metric": "mae"})
    run = _train(tmp_path, config, "no-best",
                 builder="tests.tiny_trainer_fixtures:build_nan_eval_regressor")

    assert run.status == "failed"
    assert "stage 0 produced no selectable epoch" in run.status_error
    assert [m["stage"] for m in run.metrics_history] == [0, 0]


@pytest.mark.parametrize("scheduler", SCHEDULER_TYPES)
@pytest.mark.parametrize("warmup_epochs", [0, 2])
@pytest.mark.parametrize("resume_epoch", [1, 2, 3])
def test_a_run_resumed_across_a_stage_boundary_trains_as_the_uninterrupted_run(
        tmp_path, resume_epoch, warmup_epochs, scheduler):
    config = _config([{"freeze_to": 0, "epochs": 3}, {"freeze_to": 0, "epochs": 3}],
                     early_stopping={"enabled": True, "patience": 1, "min_delta": 1e-4},
                     checkpoint_every_n_epochs=1, stage_warmup_epochs=warmup_epochs,
                     scheduler={"type": scheduler})
    straight = _train(tmp_path, config, "straight", shuffle_seed=11)
    assert straight.status == "completed", straight.status_error
    assert len(straight.metrics_history) > resume_epoch

    checkpoint = tmp_path / "straight" / f"checkpoint_epoch_{resume_epoch}.pt"
    resumed = _train(tmp_path, config, "resumed", shuffle_seed=11, resume_from=str(checkpoint))
    assert resumed.status == "completed", resumed.status_error

    keys = ("stage", "lr", "train_loss", "val_loss", "selection")
    expected = [{k: m[k] for k in keys} for m in straight.metrics_history[resume_epoch:]]
    assert [{k: m[k] for k in keys} for m in resumed.metrics_history] == expected
    for name in ("model_best", "model_final"):
        a = torch.load(tmp_path / "straight" / f"{name}.pt", weights_only=False)
        b = torch.load(tmp_path / "resumed" / f"{name}.pt", weights_only=False)
        assert torch.equal(a[STATE_DICT_KEY]["weight"], b[STATE_DICT_KEY]["weight"]), name
    assert (torch.load(tmp_path / "resumed" / "model_best.pt", weights_only=False)["epoch"]
            == torch.load(tmp_path / "straight" / "model_best.pt", weights_only=False)["epoch"])


def test_a_resume_whose_optimizer_orders_its_groups_differently_warms_up_as_the_straight_run(
        tmp_path):
    """Two param groups at different rates, a stage boundary with warmup, and a resume part way
    through the warmup into an optimizer whose groups come in the reverse order: each parameter
    warms up from its own handed-off rate, so the resumed run trains as the uninterrupted one."""
    two_rate = "tests.tiny_trainer_fixtures:build_two_rate_regressor"
    config = _config([{"freeze_to": 0, "epochs": 2}, {"freeze_to": 0, "epochs": 4}],
                     optimizer={"name": "adamw", "backbone_lr": 0.01, "head_lr": 0.05,
                                "weight_decay": 0.0},
                     scheduler={"type": "step"}, stage_warmup_epochs=3,
                     checkpoint_every_n_epochs=1)
    config["model_source"] = {"builder": two_rate, "task": "regression"}
    straight = _train(tmp_path, config, "straight")
    assert straight.status == "completed", straight.status_error

    reversed_config = {**config, "model_source": {
        "builder": two_rate, "builder_kwargs": {"reverse_groups": True}, "task": "regression"}}
    checkpoint = tmp_path / "straight" / "checkpoint_epoch_3.pt"
    resumed = _train(tmp_path, reversed_config, "resumed", resume_from=str(checkpoint))
    assert resumed.status == "completed", resumed.status_error

    keys = ("stage", "train_loss", "val_loss")
    assert ([{k: m[k] for k in keys} for m in resumed.metrics_history]
            == [{k: m[k] for k in keys} for m in straight.metrics_history[3:]])
    a = torch.load(tmp_path / "straight" / "model_final.pt", weights_only=False)[STATE_DICT_KEY]
    b = torch.load(tmp_path / "resumed" / "model_final.pt", weights_only=False)[STATE_DICT_KEY]
    assert all(torch.equal(a[name], b[name]) for name in ("weight", "bias"))


@pytest.mark.parametrize(("stated", "read_as"), [
    ({"early_stopping": {"enabled": "false"}}, 4),
    ({"log_every_n_batches": "2", "stages": [{"freeze_to": 0, "epochs": 4,
                                               "gradient_accumulation_steps": None}]}, 4),
])
def test_the_trainer_runs_the_config_its_schema_admits(tmp_path, stated, read_as):
    """A config the schema admits trains as the schema reads it: what validation coerced or
    left as the run's own is what the trainer acts on."""
    from tcip_mcp.pipelines.schemas import checked_train_config

    config = {**_config([{"freeze_to": 0, "epochs": 4}]), **stated}
    assert checked_train_config(config)[1] == []
    steps: list = []
    run = _train(tmp_path, config, "admitted",
                 batch_callback=lambda step, epoch, metrics: steps.append(step))

    assert run.status == "completed", run.status_error
    assert len(run.metrics_history) == read_as
    if "log_every_n_batches" in stated:
        assert steps == [2, 4, 6, 8, 10, 12]


def test_a_worker_run_validates_its_config_once(tmp_path, monkeypatch):
    """A run the worker trains, from its observed record through the default trainer and the
    context's seed, device and model, reads its settings off one validation of its config and
    never validates its model source again."""
    from tcip_mcp.experiments import observe
    from tcip_mcp.pipelines.training.subprocess_worker import prepare_run_context
    from tests._verified_checkpoint_fixtures import opened_run
    from tests.tiny_trainer_fixtures import count_validations, write_regression_dataset

    images_dir, csv_path = write_regression_dataset(
        tmp_path / "ds", TRAIN_INTENSITIES, [2.0 * c for c in TRAIN_INTENSITIES])
    config = _config([{"freeze_to": 0, "epochs": 1}])
    config["data"] = {**config["data"], "images_dir": str(images_dir),
                      "labels_dir": str(csv_path), "split": {"seed": 1, "val_ratio": 0.15}}
    run_dir = opened_run(tmp_path, config)

    validations = count_validations(monkeypatch)
    ctx = prepare_run_context(observe(run_dir))
    assert ctx.default_train().status == "completed", ctx.run.status_error
    ctx.seed, ctx.device, ctx.build_model()
    assert validations == ["TrainConfigSchema"]


def test_a_stage_of_no_epochs_is_refused_and_one_epoch_is_admitted(tmp_path):
    from tcip_mcp.pipelines.schemas import checked_train_config

    refused = _config([{"freeze_to": 0, "epochs": 0}])
    assert any("stages.0.epochs" in issue for issue in checked_train_config(refused)[1])
    with pytest.raises(ValueError, match="stages.0.epochs"):
        trainer_run(refused, tmp_path / "zero", project=tmp_path, has_val_loader=True)

    admitted = _train(tmp_path, _config([{"freeze_to": 0, "epochs": 1}]), "one")
    assert admitted.status == "completed", admitted.status_error


def test_an_unknown_scheduler_is_refused_and_a_named_one_is_admitted(tmp_path):
    with pytest.raises(ValueError, match="scheduler.type"):
        trainer_run(_config([{"freeze_to": 0, "epochs": 1}], scheduler={"type": "cosine_warm"}),
                    tmp_path / "unknown", project=tmp_path, has_val_loader=True)

    admitted = _train(tmp_path, _config([{"freeze_to": 0, "epochs": 1}],
                                        scheduler={"type": "step"}), "step")
    assert admitted.status == "completed", admitted.status_error


def test_a_loader_stating_no_batch_size_is_refused(tmp_path):
    from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

    from tcip_mcp.pipelines.training.collation import task_collate
    from tests.tiny_trainer_fixtures import ConstantImageDataset

    dataset = ConstantImageDataset(TRAIN_INTENSITIES, [2.0 * c for c in TRAIN_INTENSITIES])
    loader = DataLoader(dataset, batch_sampler=BatchSampler(SequentialSampler(dataset), 2, False),
                        collate_fn=task_collate("regression"))
    assert loader.batch_size is None
    run = trainer_run(_config([{"freeze_to": 0, "epochs": 1}]), tmp_path / "sampled",
                      project=tmp_path, has_val_loader=False, id="sampled")
    run = train(run, loader)
    assert run.status == "failed" and "batch_size" in run.status_error
