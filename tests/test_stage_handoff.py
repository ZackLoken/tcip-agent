"""What a training stage starts from, when it and the run end, and that a resumed run trains as
an uninterrupted one across a stage boundary.

The multi-epoch handoff runs of ``MeanIntensityRegressor`` train over opposed frames
(``opposed_regression_data``), whose holdout loss worsens as the training loss improves, so
each of their stages has its best epoch first and not last: a stage that started from its
predecessor's last epoch instead of its best one is visible, and no stage improves on the one
before it. ``_consistent`` frames fit both sides by one weight, so every epoch improves.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tcip_mcp.experiments import experiment_dir  # noqa: E402
from tcip_mcp.pipelines.model_build import (  # noqa: E402
    METRICS_KEY,
    STATE_DICT_KEY,
)
from tcip_mcp.pipelines.schemas import SCHEDULER_TYPES  # noqa: E402
from tcip_mcp.pipelines.training import generic_trainer as gt  # noqa: E402
from tcip_mcp.pipelines.training.generic_trainer import train  # noqa: E402
from tests._training_values import (  # noqa: E402
    adamw_optimizer,
    evaluation_block,
    schedule,
    stage,
    stop_rule,
)
from tests.tiny_trainer_fixtures import (  # noqa: E402
    COUNTING_REGRESSOR,
    NAN_EVAL_REGRESSOR,
    TRAIN_INTENSITIES,
    TWO_RATE_REGRESSOR,
    VAL_INTENSITIES,
    capture_model,
    drawn_data,
    frames_data,
    opposed_regression_data,
    regressor_config,
    run_context,
)


def _config(stages: list[dict], data: dict | None = None, **extra) -> dict:
    """A regression config at seed 3 over ``stages``; ``data`` is filled in by :func:`_train`,
    and a config never trained states its one band alone.
    """
    return regressor_config(
        data=data or {"num_channels": 1},
        **{"seed": 3, "stages": stages, **extra},
    )


def _opposed(root: Path) -> dict:
    return opposed_regression_data(root, TRAIN_INTENSITIES, VAL_INTENSITIES)


def _consistent(root: Path) -> dict:
    """Training and holdout frames both fit by weight +2, so the holdout loss improves as the
    training loss does.
    """
    return drawn_data(
        root,
        (TRAIN_INTENSITIES, [2.0 * c for c in TRAIN_INTENSITIES]),
        (VAL_INTENSITIES, [2.0 * c for c in VAL_INTENSITIES]),
    )


def _train(tmp_path, config: dict, name: str, frames=_opposed, **kwargs):
    """The run ``name`` under ``tmp_path`` over ``config`` with ``frames``' data block, opened by
    the launcher's own producer and trained over its own loaders with ``kwargs``.
    """
    ctx = run_context(
        tmp_path,
        {**config, "data": frames(tmp_path / frames.__name__)},
        experiment_id=name,
    )
    return train(
        ctx.run, ctx.train_loader, val_loader=ctx.val_loader, **kwargs
    )


def _tensors(value):
    """Every tensor nested in ``value``, in a fixed order."""
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        return [
            t for key in sorted(value, key=str) for t in _tensors(value[key])
        ]
    if isinstance(value, (list, tuple)):
        return [t for item in value for t in _tensors(item)]
    return []


def test_a_stage_starts_from_the_weights_and_optimizer_state_of_the_prior_stage_best_epoch(
    tmp_path, monkeypatch
):
    """The next stage starts from its predecessor's best epoch whole: the weight, its optimizer
    momentum and the model's registered buffer as they stood then, not as the last epoch left
    them; and every held epoch state carries the optimizer's tensor-valued group setting beside
    them, each as it was taken.
    """
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

    def recording_build_scheduler(optimizer, config):
        weight = models[0].weight
        at_stage_start.append(
            (
                weight.detach().clone(),
                optimizer.state[weight]["exp_avg"].clone()
                if weight in optimizer.state
                else None,
                models[0].forwards.clone(),
            )
        )
        return real_build_scheduler(optimizer, config)

    monkeypatch.setattr(gt, "_build_scheduler", recording_build_scheduler)
    taken: list = []
    real_epoch_state = gt._epoch_state

    def recording_epoch_state(*args, **kwargs):
        state = real_epoch_state(*args, **kwargs)
        taken.append(
            (
                state,
                copy.deepcopy(state),
                [
                    group["rates"].clone()
                    for group in optimizers[-1].param_groups
                ],
            )
        )
        return state

    monkeypatch.setattr(gt, "_epoch_state", recording_epoch_state)
    per_epoch: list = []

    def record_epoch(epoch: int, metrics: dict) -> None:
        weight = models[0].weight
        per_epoch.append(
            (
                weight.detach().clone(),
                optimizers[-1].state[weight]["exp_avg"].clone(),
                models[0].forwards.clone(),
            )
        )

    run = _train(
        tmp_path,
        _config(
            [stage(0), stage(0)],
            horizon=3,
            builder=COUNTING_REGRESSOR,
        ),
        "pairing",
        epoch_callback=record_epoch,
    )

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


def test_a_resume_puts_back_the_buffer_and_the_tensor_setting_its_capture_holds(
    tmp_path,
):
    """What a capture of a model and its optimizer holds, a resume's restore puts back into a
    freshly built pair: the model's registered buffer, and a tensor-valued group setting moved
    in place since the pair was built, the way a scheduler fills a tensor setting. The capture
    stays as it was taken once the restored pair moves on.
    """
    from tcip_mcp.pipelines.training.optimizer_factory import (
        GROUPS_KEY,
        build_optimizer,
        capture_training_state,
        restore_training_state,
    )

    ctx = run_context(
        tmp_path,
        _config(
            [stage(0)],
            _opposed(tmp_path / "ds"),
            builder=COUNTING_REGRESSOR,
        ),
    )
    regime = ctx.run.spec.trainer_reads().regime
    assert regime is not None
    stated = regime.optimizer

    def built_pair():
        model = ctx.build_model()
        return model, build_optimizer(
            stated,
            model,
            backbone_lr=stated.backbone_lr,
            head_lr=stated.head_lr,
        )

    trained, optimizer = built_pair()
    trained.forwards.fill_(7)
    optimizer.param_groups[0]["rates"].mul_(0.5)
    captured = capture_training_state(trained, optimizer)

    fresh, fresh_optimizer = built_pair()
    restore_training_state(
        fresh, fresh_optimizer, captured, group_settings=True
    )

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
    """A stage whose first epoch is its best ends one epoch later, before its schedule's
    four-epoch horizon, and the next stage starts.
    """
    run = _train(
        tmp_path,
        _config(
            [stage(0), stage(0)],
            horizon=4,
            early_stopping=stop_rule(1, 1e-4),
        ),
        "plateau",
    )

    assert run.status == "completed", run.status_error
    assert [m["stage"] for m in run.metrics_history] == [0, 0, 1, 1]


@pytest.mark.parametrize(
    ("frames", "min_delta", "stages_run"),
    [
        (_consistent, 0.0, [0, 0, 1, 1, 2, 2]),
        (_opposed, 0.0, [0, 0, 1, 1]),
        (_consistent, 10.0, [0, 0, 1, 1]),
    ],
    ids=[
        "every-stage-improves",
        "a-stage-worsens",
        "a-stage-improves-by-less-than-min-delta",
    ],
)
def test_the_run_ends_with_the_first_stage_that_does_not_improve_on_the_one_before(
    tmp_path, frames, min_delta, stages_run
):
    """A run runs every stage while each stage's best improves by ``min_delta`` on its
    predecessor's best; the first that does not ends the run completed, the stages after it
    never running, and the run's last epoch row names the stage that ended it.
    """
    run = _train(
        tmp_path,
        _config(
            [stage(0)] * 3,
            horizon=2,
            early_stopping=stop_rule(min_delta=min_delta),
        ),
        "run-stop",
        frames=frames,
    )

    assert run.status == "completed", run.status_error
    assert [m["stage"] for m in run.metrics_history] == stages_run
    assert (
        experiment_dir("run-stop", project=tmp_path) / "model_best.pt"
    ).is_file()


def test_a_stage_under_a_schedule_with_no_horizon_runs_until_its_plateau(
    tmp_path,
):
    """A stage carries no epoch count: under a step schedule holding its rate, a model whose
    holdout loss keeps improving trains past every count a sample config states and ends on its
    plateau.
    """
    config = _config(
        [stage(0)],
        scheduler=schedule("step", gamma=1.0),
        optimizer=adamw_optimizer(backbone_lr=2e-3, head_lr=2e-3),
        early_stopping=stop_rule(3, 1e-6),
    )
    run = _train(tmp_path, config, "unbounded", frames=_consistent)

    assert run.status == "completed", run.status_error
    history = run.metrics_history
    assert len(history) > 200
    # The fourth epoch from the end is the last to improve; the three after it do not beat it.
    last_improved = history[-4]["selection"]
    assert all(m["selection"] >= last_improved - 1e-6 for m in history[-3:])


@pytest.mark.parametrize("kind", ["cosine", "onecycle"])
def test_a_horizon_bound_schedule_is_built_over_its_horizon_and_ends_the_stage_there(
    tmp_path, monkeypatch, kind
):
    """The block's ``horizon_epochs`` is what the scheduler is built over and where a stage
    whose selection keeps improving ends; a plateau before it ends the stage first.
    """
    built: list = []
    real_build_scheduler = gt._build_scheduler

    def recording_build_scheduler(optimizer, config):
        built.append(real_build_scheduler(optimizer, config))
        return built[-1]

    monkeypatch.setattr(gt, "_build_scheduler", recording_build_scheduler)
    reached = _train(
        tmp_path,
        _config([stage(0)], scheduler=schedule(kind)),
        "horizon",
        frames=_consistent,
    )
    assert reached.status == "completed", reached.status_error
    horizon = schedule(kind)["horizon_epochs"]
    assert len(reached.metrics_history) == horizon
    (scheduler,) = built
    assert (
        scheduler.T_max if kind == "cosine" else scheduler.total_steps
    ) == horizon

    plateaued = _train(
        tmp_path,
        _config(
            [stage(0)],
            scheduler=schedule(kind),
            early_stopping=stop_rule(1),
        ),
        "plateau-first",
    )
    assert plateaued.status == "completed", plateaued.status_error
    assert len(plateaued.metrics_history) == 2 < horizon


def test_a_run_with_no_validation_loader_ends_its_stage_on_the_training_loss_plateau(
    tmp_path,
):
    """Selecting on ``loss`` with no validation loader, the training loss is the selection value
    the stop rule reads: a loss that cannot improve ends the stage one epoch after its first.
    """
    # Zero-intensity frames: the prediction is zero whatever the weight, so the loss is fixed.
    config = _config(
        [stage(0)],
        frames_data(tmp_path / "flat", [0.0] * 6, [1.0] * 6),
        scheduler=schedule("step"),
        early_stopping=stop_rule(1),
    )
    ctx = run_context(tmp_path, config)
    assert ctx.val_loader is None
    run = train(ctx.run, ctx.train_loader, val_loader=None)

    assert run.status == "completed", run.status_error
    assert [m["selection"] for m in run.metrics_history] == [1.0, 1.0]


def test_a_stage_with_no_selectable_epoch_fails_the_run_naming_it(tmp_path):
    """A model whose predictions are all non-finite gives its prediction-derived selection metric
    no value that ranks, so no epoch of the first stage is selectable.
    """
    config = _config(
        [stage(0), stage(0)],
        horizon=2,
        evaluation=evaluation_block(selection_metric="mae"),
        builder=NAN_EVAL_REGRESSOR,
    )
    run = _train(tmp_path, config, "no-best")

    assert run.status == "failed"
    assert "stage 0 produced no selectable epoch" in run.status_error
    assert [m["stage"] for m in run.metrics_history] == [0, 0]


@pytest.mark.parametrize("scheduler", SCHEDULER_TYPES)
@pytest.mark.parametrize("warmup_epochs", [0, 2])
@pytest.mark.parametrize("resume_epoch", [1, 2, 3])
def test_a_run_resumed_across_a_stage_boundary_trains_as_the_uninterrupted_run(
    tmp_path, resume_epoch, warmup_epochs, scheduler
):
    config = _config(
        [stage(0), stage(0)],
        early_stopping=stop_rule(1, 1e-4),
        checkpoint_every_n_epochs=1,
        stage_warmup_epochs=warmup_epochs,
        scheduler=schedule(scheduler),
    )
    straight = _train(tmp_path, config, "straight")
    assert straight.status == "completed", straight.status_error
    assert len(straight.metrics_history) > resume_epoch

    straight_dir, resumed_dir = (
        experiment_dir(name, project=tmp_path)
        for name in ("straight", "resumed")
    )
    checkpoint = straight_dir / f"checkpoint_epoch_{resume_epoch}.pt"
    resumed = _train(tmp_path, config, "resumed", resume_from=str(checkpoint))
    assert resumed.status == "completed", resumed.status_error

    keys = ("stage", "lr", "train_loss", "val_loss", "selection")
    expected = [
        {k: m[k] for k in keys}
        for m in straight.metrics_history[resume_epoch:]
    ]
    assert [
        {k: m[k] for k in keys} for m in resumed.metrics_history
    ] == expected
    for name in ("model_best", "model_final"):
        a = torch.load(straight_dir / f"{name}.pt", weights_only=False)
        b = torch.load(resumed_dir / f"{name}.pt", weights_only=False)
        assert torch.equal(
            a[STATE_DICT_KEY]["weight"], b[STATE_DICT_KEY]["weight"]
        ), name
    assert (
        torch.load(resumed_dir / "model_best.pt", weights_only=False)["epoch"]
        == torch.load(straight_dir / "model_best.pt", weights_only=False)[
            "epoch"
        ]
    )


def test_a_resume_whose_optimizer_orders_its_groups_differently_warms_up_as_the_straight_run(
    tmp_path,
):
    """Two param groups at different rates, a stage boundary with warmup, and a resume part way
    through the warmup into an optimizer whose groups come in the reverse order: each parameter
    warms up from its own handed-off rate, so the resumed run trains as the uninterrupted one.
    Each stage plateaus one epoch after its first, so the second ends inside its warmup.
    """

    def two_rate_config(**builder_kwargs) -> dict:
        return _config(
            [stage(0), stage(0)],
            optimizer=adamw_optimizer(backbone_lr=0.01, head_lr=0.05),
            scheduler=schedule("step"),
            stage_warmup_epochs=3,
            early_stopping=stop_rule(1),
            checkpoint_every_n_epochs=1,
            builder=TWO_RATE_REGRESSOR,
            builder_kwargs=builder_kwargs or None,
        )

    straight = _train(tmp_path, two_rate_config(), "straight")
    assert straight.status == "completed", straight.status_error

    reversed_config = two_rate_config(reverse_groups=True)
    straight_dir, resumed_dir = (
        experiment_dir(name, project=tmp_path)
        for name in ("straight", "resumed")
    )
    checkpoint = straight_dir / "checkpoint_epoch_3.pt"
    resumed = _train(
        tmp_path, reversed_config, "resumed", resume_from=str(checkpoint)
    )
    assert resumed.status == "completed", resumed.status_error

    keys = ("stage", "train_loss", "val_loss")
    assert [{k: m[k] for k in keys} for m in resumed.metrics_history] == [
        {k: m[k] for k in keys} for m in straight.metrics_history[3:]
    ]
    a = torch.load(straight_dir / "model_final.pt", weights_only=False)[
        STATE_DICT_KEY
    ]
    b = torch.load(resumed_dir / "model_final.pt", weights_only=False)[
        STATE_DICT_KEY
    ]
    assert all(torch.equal(a[name], b[name]) for name in ("weight", "bias"))


def _canceled_at(tmp_path, call: int):
    """A run over six training frames in batches of two (the second epoch is calls 4, 5 and 6)
    whose ``call``-th training forward requests its cancellation, saving every epoch: the run
    and its directory.
    """
    name = f"canceled-at-{call}"
    run_dir = experiment_dir(name, project=tmp_path)
    config = _config(
        [stage(0)],
        horizon=5,
        batch_size=2,
        checkpoint_every_n_epochs=1,
        builder_kwargs={
            "cancel_at_call": call,
            "cancel_output_dir": str(run_dir),
        },
    )

    def unsplit(root: Path) -> dict:
        return frames_data(
            root, TRAIN_INTENSITIES, [2.0 * c for c in TRAIN_INTENSITIES]
        )

    return _train(tmp_path, config, name, frames=unsplit), run_dir


def test_a_cancel_inside_an_epoch_leaves_that_epoch_no_row_best_or_checkpoint(
    tmp_path,
):
    """A cancel requested partway through the second epoch's batches ends the run canceled
    before that epoch's end: it appends no row, never becomes the best, and writes no
    checkpoint, while the first epoch's stand. The final weights moved past the first epoch's
    row, so ``model_final`` carries no metrics.
    """
    run, run_dir = _canceled_at(tmp_path, 5)

    assert run.status == "canceled", run.status_error
    assert run.current_epoch == 2
    assert len(run.metrics_history) == 1
    assert (run_dir / "checkpoint_epoch_1.pt").is_file()
    assert not (run_dir / "checkpoint_epoch_2.pt").exists()
    best = torch.load(run_dir / "model_best.pt", weights_only=False)
    assert best["epoch"] == 1
    final = torch.load(run_dir / "model_final.pt", weights_only=False)
    assert METRICS_KEY not in final


def test_a_cancel_seen_at_an_epoch_top_saves_the_final_weights_with_their_row(
    tmp_path,
):
    """A cancel requested during the second epoch's last batch is seen at the third epoch's top:
    the second epoch completes, and ``model_final`` carries its row.
    """
    run, run_dir = _canceled_at(tmp_path, 6)

    assert run.status == "canceled", run.status_error
    assert len(run.metrics_history) == 2
    final = torch.load(run_dir / "model_final.pt", weights_only=False)
    assert final[METRICS_KEY]["train_loss"] == pytest.approx(
        run.metrics_history[-1]["train_loss"]
    )


def test_a_cancel_seen_as_a_stage_starts_saves_its_best_epoch_with_that_epoch_row(
    tmp_path, monkeypatch
):
    """A cancel requested once the second stage has started from the first stage's best epoch,
    whose best is its first epoch and not its last, is seen at that stage's first poll:
    ``model_final`` holds the best epoch's weights and carries the best epoch's row.
    """
    from tcip_mcp.experiments import request_cancel

    name = "canceled-at-a-stage-start"
    run_dir = experiment_dir(name, project=tmp_path)
    real_build_scheduler = gt._build_scheduler
    built: list = []

    def cancel_at_the_second_stage(optimizer, config):
        built.append(config)
        if len(built) == 2:
            request_cancel(run_dir)
        return real_build_scheduler(optimizer, config)

    monkeypatch.setattr(gt, "_build_scheduler", cancel_at_the_second_stage)
    run = _train(
        tmp_path,
        _config([stage(0), stage(0)], horizon=3),
        name,
    )

    assert run.status == "canceled", run.status_error
    stage0 = [m["val_loss"] for m in run.metrics_history]
    assert len(stage0) == 3 and stage0 == sorted(stage0)
    final = torch.load(run_dir / "model_final.pt", weights_only=False)
    best = torch.load(run_dir / "model_best.pt", weights_only=False)
    assert best["epoch"] == 1
    assert final[METRICS_KEY] == best[METRICS_KEY]
    assert torch.equal(
        final[STATE_DICT_KEY]["weight"], best[STATE_DICT_KEY]["weight"]
    )


def test_a_resumed_run_canceled_at_its_first_poll_saves_the_checkpoint_row(
    tmp_path,
):
    """A run resumed from a periodic checkpoint and canceled before its first epoch saves the
    checkpoint's weights under the checkpoint's own row.
    """
    from tcip_mcp.experiments import request_cancel

    config = _config(
        [stage(0)], horizon=3, checkpoint_every_n_epochs=1
    )
    straight = _train(tmp_path, config, "straight")
    assert straight.status == "completed", straight.status_error
    checkpoint = (
        experiment_dir("straight", project=tmp_path) / "checkpoint_epoch_1.pt"
    )

    ctx = run_context(
        tmp_path,
        {**config, "data": _opposed(tmp_path / "resumed-frames")},
        experiment_id="resumed",
    )
    request_cancel(ctx.run_dir)
    resumed = train(
        ctx.run,
        ctx.train_loader,
        val_loader=ctx.val_loader,
        resume_from=str(checkpoint),
    )

    assert resumed.status == "canceled", resumed.status_error
    assert resumed.metrics_history == []
    final = torch.load(ctx.run_dir / "model_final.pt", weights_only=False)
    held = torch.load(checkpoint, weights_only=False)
    assert final[METRICS_KEY] == held[METRICS_KEY]


@pytest.mark.parametrize("warmup_epochs", [0, 2])
def test_the_scheduler_steps_every_epoch_outside_a_stage_warmup(
    tmp_path, warmup_epochs
):
    """Under a schedule halving the rate each step, each epoch row's rate is the one it trained
    at: a stage with no warmup halves it every epoch, and a stage warming up for two epochs
    ramps to its target, holds it through the first epoch after the warmup, then halves.
    """
    config = _config(
        [stage(0), stage(0)],
        scheduler=schedule("step", step_size=1, gamma=0.5),
        early_stopping=stop_rule(3, 1e9),
        stage_warmup_epochs=warmup_epochs,
    )
    run = _train(tmp_path, config, "warmup", frames=_consistent)
    assert run.status == "completed", run.status_error

    rates = [
        [m["lr"] for m in run.metrics_history if m["stage"] == index]
        for index in (0, 1)
    ]
    target = rates[0][0]
    halving = [target * 0.5**i for i in range(4)]
    assert rates[0] == pytest.approx(halving)
    if warmup_epochs:
        assert rates[1][0] < target
        assert rates[1][1:] == pytest.approx([target, target, target * 0.5])
    else:
        assert rates[1] == pytest.approx(halving)


@pytest.mark.parametrize(
    ("stated", "read_as"),
    [
        (
            {
                "early_stopping": {
                    **stop_rule(),
                    "patience": "2",
                    "min_delta": "0",
                }
            },
            3,
        ),
        (
            {
                "log_every_n_batches": "2",
                "stages": [stage(0, gradient_accumulation_steps=None)],
            },
            4,
        ),
    ],
)
def test_the_trainer_runs_the_config_its_schema_admits(
    tmp_path, stated, read_as
):
    """A config the schema admits trains as the schema reads it: what validation coerced or
    left as the run's own is what the trainer acts on, here under a four-epoch horizon.
    """
    from tcip_mcp.pipelines.schemas import checked_train_config

    config = {**_config([stage(0)], horizon=4), **stated}
    assert checked_train_config(config)[1] == []
    steps: list = []
    run = _train(
        tmp_path,
        config,
        "admitted",
        batch_callback=lambda step, epoch, metrics: steps.append(step),
    )

    assert run.status == "completed", run.status_error
    assert len(run.metrics_history) == read_as
    if "log_every_n_batches" in stated:
        assert steps == [2, 4, 6, 8, 10, 12]


def test_a_worker_run_validates_its_config_once(tmp_path, monkeypatch):
    """A run the worker trains, from its observed record through the default trainer and the
    context's seed, device and model, reads its settings off one validation of its config and
    never validates its model source again.
    """
    from tcip_mcp.experiments import observe
    from tcip_mcp.pipelines.training.subprocess_worker import (
        prepare_run_context,
    )
    from tests._verified_checkpoint_fixtures import opened_run
    from tests.tiny_trainer_fixtures import count_validations

    run_dir = opened_run(
        tmp_path, _config([stage(0)], _opposed(tmp_path / "ds"))
    )

    validations = count_validations(monkeypatch)
    ctx = prepare_run_context(observe(run_dir))
    assert ctx.default_train().status == "completed", ctx.run.status_error
    ctx.seed, ctx.device, ctx.build_model()
    assert validations == ["TrainConfigSchema"]


def test_a_stage_stating_an_epoch_count_is_refused_and_one_stating_none_is_admitted(
    tmp_path,
):
    from tcip_mcp.pipelines.schemas import checked_train_config

    refused = _config([stage(0, epochs=3)])
    assert any(
        "stages.0.epochs" in issue
        for issue in checked_train_config(refused)[1]
    )
    with pytest.raises(ValueError, match="stages.0.epochs"):
        _train(tmp_path, refused, "counted")

    admitted = _train(tmp_path, _config([stage(0)]), "one")
    assert admitted.status == "completed", admitted.status_error


def test_an_unknown_scheduler_is_refused_and_a_named_one_is_admitted(tmp_path):
    with pytest.raises(ValueError, match="scheduler.*cosine_warm"):
        _train(
            tmp_path,
            _config([stage(0)], scheduler={"type": "cosine_warm"}),
            "unknown",
        )

    admitted = _train(
        tmp_path,
        _config(
            [stage(0)],
            scheduler=schedule("step"),
            early_stopping=stop_rule(1),
        ),
        "step",
    )
    assert admitted.status == "completed", admitted.status_error


def test_a_onecycle_schedule_trains_at_the_stated_sgd_momentum():
    """OneCycle cycles the rate alone: the momentum the config states is the one each step
    takes, never a momentum cycle the schedule brings of its own.
    """
    from torch import nn

    from tcip_mcp.pipelines.schemas import train_config
    from tests._training_values import sgd_optimizer

    spec = train_config(
        _config(
            [stage(0)],
            optimizer=sgd_optimizer(),
            scheduler=schedule("onecycle"),
        )
    )
    assert spec.optimizer is not None and spec.scheduler is not None
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=spec.optimizer.head_lr,
        momentum=spec.optimizer.momentum,
    )
    scheduler = gt._build_scheduler(optimizer, spec.scheduler)
    model(torch.ones(3, 2)).sum().backward()
    optimizer.step()
    scheduler.step()

    assert [g["momentum"] for g in optimizer.param_groups] == [
        spec.optimizer.momentum
    ]


def _built(scheduler_block: dict):
    """The torch scheduler a run whose config states ``scheduler_block`` builds over an SGD
    optimizer at its stated momentum.
    """
    from torch import nn

    from tcip_mcp.pipelines.schemas import train_config
    from tests._training_values import sgd_optimizer

    spec = train_config(
        _config(
            [stage(0)], optimizer=sgd_optimizer(), scheduler=scheduler_block
        )
    )
    assert spec.optimizer is not None and spec.scheduler is not None
    optimizer = torch.optim.SGD(
        nn.Linear(2, 1).parameters(),
        lr=spec.optimizer.head_lr,
        momentum=spec.optimizer.momentum,
    )
    return gt._build_scheduler(optimizer, spec.scheduler), optimizer


def test_a_plateau_schedule_builds_at_every_setting_it_states():
    """Each setting a plateau block states is the one the torch scheduler holds."""
    scheduler, _ = _built(
        schedule(
            "plateau",
            factor=0.4,
            patience=5,
            threshold=1e-3,
            threshold_mode="abs",
            cooldown=2,
            min_lr=1e-5,
            eps=1e-7,
        )
    )

    assert (
        scheduler.factor,
        scheduler.patience,
        scheduler.threshold,
        scheduler.threshold_mode,
        scheduler.cooldown,
        scheduler.min_lrs,
        scheduler.eps,
    ) == (0.4, 5, 1e-3, "abs", 2, [1e-5], 1e-7)


def test_a_onecycle_schedule_builds_at_every_setting_it_states():
    """Each setting a one-cycle block states is the one the torch scheduler holds: its horizon,
    its phases and where the first ends, its annealing, its momentum cycle and both rate
    divisors.
    """
    scheduler, optimizer = _built(
        schedule(
            "onecycle",
            max_lr=2e-2,
            horizon_epochs=10,
            pct_start=0.4,
            anneal_strategy="linear",
            cycle_momentum=True,
            base_momentum=0.8,
            max_momentum=0.9,
            div_factor=10.0,
            final_div_factor=100.0,
            three_phase=True,
        )
    )
    (group,) = optimizer.param_groups

    assert scheduler.total_steps == 10
    assert len(scheduler._schedule_phases) == 3
    assert scheduler._schedule_phases[0]["end_step"] == pytest.approx(3.0)
    assert scheduler._anneal_func_type == "linear"
    assert scheduler.cycle_momentum is True
    assert (group["base_momentum"], group["max_momentum"]) == (0.8, 0.9)
    assert group["initial_lr"] == pytest.approx(2e-3)
    assert group["min_lr"] == pytest.approx(2e-5)
