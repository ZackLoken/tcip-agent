"""A learning-rate range test: its body, its reading of the curve, and the gate its range sets.

A range test is a run of its own (``experiments.RangeTest``): it ramps one stage's rates up over
its batches, logs each step's loss and head rate as a per-batch row, and writes its reading once as
the run's ``lr_range.json`` (:class:`LrRangeResult`). A config naming the run through
``optimizer.lr_range_run`` is held to the range the reading derives (:func:`range_issues`).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from pathlib import Path

    from tcip_mcp.experiments import RangeTest
    from tcip_mcp.pipelines.schemas import TrainConfigSchema
    from tcip_mcp.pipelines.training.envelope import TrainContext

HEAD_LR_AXIS = "optimizer.head_lr"
"""The sweep axis a range test's range is written into."""
BACKBONE_LR_AXIS = "optimizer.backbone_lr"
"""The sweep axis a trial's backbone rate is derived in, unless the point samples it."""

LOW_DIVISOR = 10.0
"""Provisional: the lowest rate a range reaches below its high end, as a divisor of it (Smith
2018, arXiv 1803.09820, section 5: a minimum rate a tenth of the maximum), until a validation on
this platform's data replaces it."""


class RangeStop(BaseModel):
    """Where a range test stopped short of its last step, and why.

    ``step`` is the one-based optimizer step it stopped at, which logged no row.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int
    reason: str


class LrRangeResult(BaseModel):
    """A range test's reading of its curve, its run's ``lr_range.json``.

    It holds where the test stopped (``None`` for a ramp run to its last step), the trainable
    parameter count and the ``freeze_to`` of the stage it probed, the ``backbone_lr / head_lr``
    ratio its groups were ramped under, the head rate of minimum smoothed loss and the head rate
    of steepest descent (the least slope of the smoothed loss over the log rate).

    The range is derived from those, never stored beside them, by a rule that is provisional
    until a validation on this platform's data replaces it: :attr:`high` is the rate of minimum
    smoothed loss (Smith 2018, arXiv 1803.09820, section 4.1: the rate at that extremum is the
    largest usable as the maximum bound) and :attr:`low` the lesser of ``high /``
    :data:`LOW_DIVISOR` and the rate of steepest descent.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    stop: RangeStop | None
    trainable_params: int
    freeze_to: int
    ratio: float
    min_loss_rate: float
    steepest_rate: float

    @property
    def high(self) -> float:
        """The range's upper end, which a stated rate stays strictly below."""
        return self.min_loss_rate

    @property
    def low(self) -> float:
        """The range's lower end."""
        return min(self.high / LOW_DIVISOR, self.steepest_rate)


def read_curve(
    rates: list[float], smoothed: list[float], stop: RangeStop | None
) -> tuple[float, float]:
    """Return the rate of minimum smoothed loss and the rate of steepest descent of a curve.

    ``rates`` and ``smoothed`` are a range test's logged rows in order, each finite, and ``stop``
    where it stopped. The steepest descent is the least of ``torch.gradient`` of the smoothed
    losses with the log rates as its coordinates. Refuses (``ValueError``) fewer than two rows, a
    minimum at the first row (no descent in the probe), and, with no ``stop``, a minimum at the
    last row (the ramp ended still descending, so no turning point was found).
    """
    import torch

    if len(smoothed) < 2:
        raise ValueError(
            f"the range test logged {len(smoothed)} finite smoothed row(s) before it stopped; "
            "a reading needs two. Lower ramp_low, or raise batches."
        )
    losses = torch.tensor(smoothed, dtype=torch.float64)
    lowest = int(torch.argmin(losses))
    if lowest == 0:
        raise ValueError(
            f"the smoothed loss is least at the first row, rate {rates[0]:.3g}: the loss "
            "never fell in the probe window. Lower ramp_low."
        )
    if stop is None and lowest == len(smoothed) - 1:
        raise ValueError(
            f"the smoothed loss is least at the last row, rate {rates[-1]:.3g}, and the ramp "
            "ended without diverging: the loss was still falling, so no turning point was "
            "found. Raise ramp_high."
        )
    (slope,) = torch.gradient(
        losses, spacing=[torch.log(torch.tensor(rates, dtype=torch.float64))]
    )
    return rates[lowest], rates[int(torch.argmin(slope))]


def run_range_test(ctx: TrainContext, test: RangeTest) -> None:
    """Run the range test ``test`` over ``ctx``'s run, then write its reading.

    The model is built (``ctx.build_model``) under the run's seed, the probed stage's freeze
    applied (``generic_trainer.apply_stage_freeze``) and its trainable count read
    (``generic_trainer.trained_parameters``, refusing a stage that trains nothing). The
    optimizer is the run's block built at the rates the default trainer trains that stage at
    (``generic_trainer.stage_rates``), every group then scaled so the head group starts at
    ``ramp_low`` and ramped by ``torch.optim.lr_scheduler.ExponentialLR`` one factor per step.
    Each step accumulates the stage's batches (``generic_trainer.stage_accumulation``) from the
    train loader, cycled, through the trainer's own forward (``generic_trainer.batch_loss``),
    and logs ``batch_loss``, ``smoothed_loss`` and ``lr`` (the head group's rate) through
    ``ctx.log_batch``. The test stops at the first step whose loss is not finite or whose
    smoothed loss exceeds ``divergence`` times the least before it, logging nothing for that
    step. The reading (:func:`read_curve`, whose refusals propagate) is written as the run's
    :class:`LrRangeResult` and the run completed; a cancel ends it ``canceled`` with no reading.
    """
    import itertools

    import torch

    from tcip_mcp.experiments import (
        BATCH_LOSS_KEY,
        LR_RANGE_FILE,
        require_open,
        write_record,
    )
    from tcip_mcp.pipelines.training.generic_trainer import (
        apply_stage_freeze,
        batch_loss,
        stage_accumulation,
        stage_rates,
        trained_parameters,
    )
    from tcip_mcp.pipelines.training.optimizer_factory import (
        build_optimizer,
        head_group,
    )

    run = ctx.run
    spec = run.spec
    regime = spec.default_trainer_regime()
    stage = regime.stages[test.stage]
    ctx.set_seed()
    device = ctx.device
    model = ctx.build_model().to(device)
    apply_stage_freeze(model, stage.freeze_to)
    trainable = trained_parameters(
        model, f"stage {test.stage} (freeze_to={stage.freeze_to})"
    )
    backbone_lr, head_lr = stage_rates(
        spec, regime, stage, run.reads.batch_size
    )
    optimizer = build_optimizer(
        regime.optimizer, model, backbone_lr=backbone_lr, head_lr=head_lr
    )
    for group in optimizer.param_groups:
        group["lr"] *= test.ramp_low / head_lr
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=(test.ramp_high / test.ramp_low) ** (1 / (test.batches - 1)),
    )
    use_amp = spec.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type) if use_amp else None
    if not len(ctx.train_loader):
        raise ValueError("this run's train loader yields no batch")
    batches = (
        (epoch, batch)
        for epoch in itertools.count(1)
        for batch in ctx.train_loader
    )
    window = stage_accumulation(spec, stage)

    rates: list[float] = []
    smoothed: list[float] = []
    least = math.inf
    stop = None
    model.train()
    for step in range(1, test.batches + 1):
        if run.should_cancel():
            run.status = "canceled"
            return
        optimizer.zero_grad()
        loss_value = 0.0
        for _ in range(window):
            epoch, batch = next(batches)
            loss = (
                batch_loss(model, batch, ctx.task, device, use_amp=use_amp)
                / window
            )
            (loss if scaler is None else scaler.scale(loss)).backward()
            loss_value += loss.item()
        rate = head_group(optimizer)["lr"]
        if not math.isfinite(loss_value):
            stop = RangeStop(
                step=step,
                reason=f"the loss at rate {rate:.3g} is not finite",
            )
            break
        value = (
            test.smoothing * loss_value + (1 - test.smoothing) * smoothed[-1]
            if smoothed
            else loss_value
        )
        least = min(least, value)
        if value > test.divergence * least:
            stop = RangeStop(
                step=step,
                reason=f"the smoothed loss {value:.4g} at rate {rate:.3g} exceeds "
                f"divergence {test.divergence} times the least, {least:.4g}",
            )
            break
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        ctx.log_batch(
            step,
            epoch,
            {BATCH_LOSS_KEY: loss_value, "smoothed_loss": value, "lr": rate},
        )
        rates.append(rate)
        smoothed.append(value)
        scheduler.step()

    min_loss_rate, steepest_rate = read_curve(rates, smoothed, stop)
    require_open(ctx.run_dir)
    write_record(
        ctx.run_dir / LR_RANGE_FILE,
        LrRangeResult(
            stop=stop,
            trainable_params=trainable,
            freeze_to=stage.freeze_to,
            ratio=backbone_lr / head_lr,
            min_loss_rate=min_loss_rate,
            steepest_rate=steepest_rate,
        ),
    )
    run.status = "completed"


def read_lr_range(path: Path) -> LrRangeResult:
    """Return the :class:`LrRangeResult` at ``path``, a run's ``lr_range.json``.

    A record its model refuses raises ``tcip_store.ReadRefusedError`` naming the file.
    """
    from tcip_mcp.experiments import read_run_record
    from tcip_store import validated

    return validated(LrRangeResult, read_run_record(path), str(path))


def range_result(run_id: str, project: Path) -> LrRangeResult:
    """Return the reading of the range test ``run_id`` names in ``project``.

    Refuses (``ValueError``) an id naming no run, a run that is no range test, and a range test
    that ended with no reading, naming its state and error.
    """
    from tcip_mcp.experiments import (
        LR_RANGE_FILE,
        RangeTest,
        find_observation,
    )

    observation = find_observation(run_id, project=project)
    if observation is None:
        raise ValueError(
            f"optimizer.lr_range_run {run_id!r} names no run of this project."
        )
    if not isinstance(observation.record.launched_as, RangeTest):
        raise ValueError(
            f"optimizer.lr_range_run {run_id!r} names a run launched as "
            f"{observation.record.launched_as.kind!r}, not a range test (derive_lr_range)."
        )
    written = observation.directory / LR_RANGE_FILE
    if observation.state != "completed" or not written.is_file():
        raise ValueError(
            f"range test {run_id!r} derived no range: it is {observation.state}"
            + (
                f" ({observation.status_error})"
                if observation.status_error
                else ""
            )
            + "."
        )
    return read_lr_range(written)


def named_range_run(config: dict) -> str | None:
    """Return the range test a config's ``optimizer.lr_range_run`` names, before validation.

    ``None`` for a config naming none, or stating no ``optimizer`` mapping.
    """
    optimizer = config.get("optimizer")
    return (
        optimizer.get("lr_range_run") if isinstance(optimizer, dict) else None
    )


def with_ranged_backbone(config: dict, point: dict, project: Path) -> dict:
    """Return ``config`` with its ``backbone_lr`` derived from a sampled ``head_lr``.

    When ``point`` samples :data:`HEAD_LR_AXIS` and no backbone rate of its own, and ``config``
    names a range test (:func:`named_range_run`), the backbone rate is the point's head rate
    times the ratio the test ramped under, on a copy; otherwise ``config`` comes back as it is.
    Refuses (``ValueError``) as :func:`range_result` does.
    """
    run_id = named_range_run(config)
    if (
        run_id is None
        or HEAD_LR_AXIS not in point
        or BACKBONE_LR_AXIS in point
    ):
        return config
    ratio = range_result(run_id, project).ratio
    return {
        **config,
        "optimizer": {
            **config["optimizer"],
            "backbone_lr": point[HEAD_LR_AXIS] * ratio,
        },
    }


def range_issues(spec: TrainConfigSchema, project: Path) -> list[str]:
    """Return why ``spec`` falls outside the range its ``optimizer.lr_range_run`` derived.

    Nothing is checked for a config naming no range test. Otherwise the issues are the test's
    refusal (:func:`range_result`), or each of: a ``head_lr`` outside ``[low, high)``, a stage
    whose ``freeze_to`` differs from the one probed, and a ``backbone_lr / head_lr`` ratio unlike
    the one ramped under.
    """
    optimizer = spec.optimizer
    if optimizer is None or optimizer.lr_range_run is None:
        return []
    run_id = optimizer.lr_range_run
    try:
        result = range_result(run_id, project)
    except ValueError as exc:
        return [str(exc)]
    issues = []
    if not result.low <= optimizer.head_lr < result.high:
        issues.append(
            f"optimizer.head_lr {optimizer.head_lr:g} lies outside [{result.low:g}, "
            f"{result.high:g}), the range range test {run_id!r} derived; state a rate inside "
            "it, below its high end."
        )
    issues.extend(
        f"stages[{index}] freezes to {stage.freeze_to}, and range test {run_id!r} probed "
        f"freeze_to={result.freeze_to}: its range governs that freeze state alone."
        for index, stage in enumerate(spec.stages or [])
        if stage.freeze_to != result.freeze_to
    )
    ratio = optimizer.backbone_lr / optimizer.head_lr
    # isclose's default absorbs the round-off of a rate times the ratio, divided back.
    if not math.isclose(ratio, result.ratio):
        issues.append(
            f"optimizer.backbone_lr / head_lr is {ratio:g}, and range test {run_id!r} ramped "
            f"its groups at {result.ratio:g}; state backbone_lr at head_lr times {result.ratio:g}."
        )
    return issues
