"""What a training stage starts from, when it ends, and that a resumed run trains as an
uninterrupted one across a stage boundary.

The multi-epoch handoff runs of ``MeanIntensityRegressor`` train over
``opposed_regression_loaders``, whose holdout loss worsens as the training loss improves, so
each of their stages has its best epoch first and not last: a stage that started from its
predecessor's last epoch instead of its best one is visible.
"""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from tcip_mcp.pipelines.model_build import STATE_DICT_KEY  # noqa: E402
from tcip_mcp.pipelines.schemas import SCHEDULER_TYPES  # noqa: E402
from tcip_mcp.pipelines.training import generic_trainer as gt  # noqa: E402
from tcip_mcp.pipelines.training.generic_trainer import train  # noqa: E402
from tests._training_values import schedule  # noqa: E402
from tests.tiny_trainer_fixtures import (  # noqa: E402
    COUNTING_REGRESSOR,
    NAN_EVAL_REGRESSOR,
    TWO_RATE_REGRESSOR,
    capture_model,
    opposed_regression_loaders,
    regressor_config,
    trainer_run,
)

TRAIN_INTENSITIES = [0.10, 0.25, 0.40, 0.55, 0.70, 0.85]
VAL_INTENSITIES = [0.15, 0.35, 0.60, 0.90]


def _config(stages: list[dict], **extra) -> dict:
    return regressor_config(**{"seed": 3, "stages": stages, **extra})


def _train(tmp_path, config: dict, name: str, **kwargs):
    run = trainer_run(config, tmp_path / name, project=tmp_path, has_val_loader=True, id=name)
    train_loader, val_loader = opposed_regression_loaders(run, TRAIN_INTENSITIES, VAL_INTENSITIES)
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
    """The next stage starts from its predecessor's best epoch whole: the weight, its optimizer
    momentum and the model's registered buffer as they stood then, not as the last epoch left
    them; and every held epoch state carries the optimizer's tensor-valued group setting beside
    them, each as it was taken."""
    from tcip_mcp.pipelines.training.optimizer_factory import GROUPS_KEY

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
                               if weight in optimizer.state else None,
                               models[0].forwards.clone()))
        return real_build_scheduler(optimizer, config, epochs)

    monkeypatch.setattr(gt, "_build_scheduler", recording_build_scheduler)
    taken: list = []
    real_epoch_state = gt._epoch_state

    def recording_epoch_state(*args, **kwargs):
        state = real_epoch_state(*args, **kwargs)
        taken.append((state, copy.deepcopy(state),
                      [group["rates"].clone() for group in optimizers[-1].param_groups]))
        return state

    monkeypatch.setattr(gt, "_epoch_state", recording_epoch_state)
    per_epoch: list = []

    def record_epoch(epoch: int, metrics: dict) -> None:
        weight = models[0].weight
        per_epoch.append((weight.detach().clone(),
                          optimizers[-1].state[weight]["exp_avg"].clone(),
                          models[0].forwards.clone()))

    run = _train(tmp_path, _config([{"freeze_to": 0, "epochs": 3}, {"freeze_to": 0, "epochs": 2}],
                                   builder=COUNTING_REGRESSOR),
                 "pairing", epoch_callback=record_epoch)

    assert run.status == "completed", run.status_error
    stage0 = [m["val_loss"] for m in run.metrics_history if m["stage"] == 0]
    assert stage0 == sorted(stage0) and len(set(stage0)) == 3
    best_weight, best_momentum, best_forwards = per_epoch[0]
    last_weight, _, last_forwards = per_epoch[2]
    assert not torch.equal(best_weight, last_weight)
    assert not torch.equal(best_forwards, last_forwards)
    weight, momentum, forwards = at_stage_start[1]
    assert torch.equal(weight, best_weight)
    assert momentum is not None and torch.equal(momentum, best_momentum)
    assert torch.equal(forwards, best_forwards)
    # Every held epoch state is as it was taken, after the later stage's steps, and holds the
    # optimizer's tensor setting as the optimizer held it then.
    assert taken
    for state, as_taken, rates in taken:
        now, then = _tensors(state), _tensors(as_taken)
        assert len(now) == len(then)
        assert all(torch.equal(a, b) for a, b in zip(now, then))
        held = [group["settings"]["rates"] for group in state[GROUPS_KEY]]
        assert len(held) == len(rates) and all(map(torch.equal, held, rates))


def test_a_resume_puts_back_the_buffer_and_the_tensor_setting_its_capture_holds(tmp_path):
    """What a capture of a model and its optimizer holds, a resume's restore puts back into a
    freshly built pair: the model's registered buffer, and a tensor-valued group setting moved
    in place since the pair was built, the way a scheduler fills a tensor setting. The capture
    stays as it was taken once the restored pair moves on."""
    from tcip_mcp.pipelines.schemas import train_config
    from tcip_mcp.pipelines.training.optimizer_factory import (
        GROUPS_KEY, build_optimizer, capture_training_state, restore_training_state,
    )
    from tests._chain_fixtures import built_model

    config = _config([{"freeze_to": 0, "epochs": 1}], builder=COUNTING_REGRESSOR)
    regime = train_config(config).trainer_reads().regime
    assert regime is not None
    stated = regime.optimizer

    def built_pair():
        model = built_model(config)
        return model, build_optimizer(stated, model, backbone_lr=stated.backbone_lr,
                                      head_lr=stated.head_lr)

    trained, optimizer = built_pair()
    trained.forwards.fill_(7)
    optimizer.param_groups[0]["rates"].mul_(0.5)
    captured = capture_training_state(trained, optimizer)

    fresh, fresh_optimizer = built_pair()
    restore_training_state(fresh, fresh_optimizer, captured, group_settings=True)

    assert float(fresh.forwards) == 7.0
    restored = fresh_optimizer.param_groups[0]["rates"]
    assert torch.equal(restored, optimizer.param_groups[0]["rates"])
    restored.mul_(3.0)
    fresh.forwards.add_(1)
    held = optimizer.param_groups[0]["rates"].clone()
    assert torch.equal(captured[GROUPS_KEY][0]["settings"]["rates"], held)
    assert float(captured[STATE_DICT_KEY]["forwards"]) == 7.0
    optimizer.param_groups[0]["rates"].mul_(10.0)
    assert torch.equal(captured[GROUPS_KEY][0]["settings"]["rates"], held)


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
                     evaluation={"selection_metric": "mae"}, builder=NAN_EVAL_REGRESSOR)
    run = _train(tmp_path, config, "no-best")

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
                     scheduler=schedule(scheduler))
    straight = _train(tmp_path, config, "straight")
    assert straight.status == "completed", straight.status_error
    assert len(straight.metrics_history) > resume_epoch

    checkpoint = tmp_path / "straight" / f"checkpoint_epoch_{resume_epoch}.pt"
    resumed = _train(tmp_path, config, "resumed", resume_from=str(checkpoint))
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
    def two_rate_config(**builder_kwargs) -> dict:
        return _config([{"freeze_to": 0, "epochs": 2}, {"freeze_to": 0, "epochs": 4}],
                       optimizer={"name": "adamw", "backbone_lr": 0.01, "head_lr": 0.05,
                                  "weight_decay": 0.0},
                       scheduler=schedule("step"), stage_warmup_epochs=3,
                       checkpoint_every_n_epochs=1, builder=TWO_RATE_REGRESSOR,
                       builder_kwargs=builder_kwargs or None)

    straight = _train(tmp_path, two_rate_config(), "straight")
    assert straight.status == "completed", straight.status_error

    reversed_config = two_rate_config(reverse_groups=True)
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
    with pytest.raises(ValueError, match="scheduler.*cosine_warm"):
        trainer_run(_config([{"freeze_to": 0, "epochs": 1}], scheduler={"type": "cosine_warm"}),
                    tmp_path / "unknown", project=tmp_path, has_val_loader=True)

    admitted = _train(tmp_path, _config([{"freeze_to": 0, "epochs": 1}],
                                        scheduler=schedule("step")), "step")
    assert admitted.status == "completed", admitted.status_error


def test_a_onecycle_schedule_trains_at_the_stated_sgd_momentum():
    """OneCycle cycles the rate alone: the momentum the config states is the one each step
    takes, never a momentum cycle the schedule brings of its own."""
    from torch import nn

    from tcip_mcp.pipelines.schemas import train_config
    from tests._training_values import sgd_optimizer

    spec = train_config(_config([{"freeze_to": 0, "epochs": 4}], optimizer=sgd_optimizer(),
                                scheduler=schedule("onecycle")))
    assert spec.optimizer is not None and spec.scheduler is not None
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=spec.optimizer.head_lr,
                                momentum=spec.optimizer.momentum)
    scheduler = gt._build_scheduler(optimizer, spec.scheduler, 4)
    model(torch.ones(3, 2)).sum().backward()
    optimizer.step()
    scheduler.step()

    assert [g["momentum"] for g in optimizer.param_groups] == [spec.optimizer.momentum]
