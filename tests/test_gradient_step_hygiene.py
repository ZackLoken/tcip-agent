"""An optimizer step carries the mean gradient of the batches in its own window.

These runs use a model whose per-batch gradient is fixed by the batch's data, and compare the
gradient each optimizer step receives against the mean of the independently computed gradients
of the batches it covers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from tcip_mcp.pipelines.model_build import SourceLayout
    from tcip_mcp.pipelines.schemas import (
        ModelSourceSchema,
        OptimizerSpec,
        TrainGeometry,
    )
    from tcip_mcp.pipelines.training.run_registry import TrainRun
    from tcip_mcp.subject_registry import Attribute

pytest.importorskip("torch")
import torch  # noqa: E402

from tcip_mcp.pipelines.training import generic_trainer as gt
from tcip_mcp.pipelines.training.generic_trainer import train
from tests._training_values import adamw_optimizer
from tests.tiny_trainer_fixtures import (
    build_data_scaled_gradient_model,
    capture_model,
    frames_data,
    regressor_config,
    run_context,
)

BUILDER = "tiny_trainer_fixtures:build_data_scaled_gradient_model"

# Deliberately spread over three orders of magnitude: a step that carried a neighbor's gradient
# too would land nowhere near the batch's own.
SKEWED_VALUES = [1.0, 10.0, 100.0]
SKEWED_INTENSITIES = [0.2, 0.5, 0.9]


def _trained(
    tmp_path: Path,
    values: list[float] = SKEWED_VALUES,
    intensities: list[float] = SKEWED_INTENSITIES,
    *,
    accumulation: int = 1,
) -> TrainRun:
    """Train a run over frames at ``intensities`` valued ``values`` and return it.

    The run has no validation side, one frame a batch at ``accumulation`` batches a step,
    opened by the launcher's own producer and trained over its own loader.
    """
    config = regressor_config(
        1,
        data=frames_data(tmp_path / "ds", intensities, values),
        project=tmp_path,
        builder=BUILDER,
        gradient_accumulation_steps=accumulation,
        batch_size=1,
        seed=0,
        optimizer=adamw_optimizer(backbone_lr=0.01, head_lr=0.01),
    )
    ctx = run_context(tmp_path, config)
    return train(ctx.run, ctx.train_loader)


def _batch_gradient(
    images: torch.Tensor, targets: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Return one batch's own gradient from a fresh instance of the model the run builds."""
    model = build_data_scaled_gradient_model()
    model.train()
    sum(model(images, targets).values()).backward()
    # every parameter must carry a gradient here; skipping one would shrink the comparison
    for p in model.parameters():
        assert p.grad is not None
    return torch.cat([p.grad.detach().reshape(-1) for p in model.parameters()])


def _record_steps(monkeypatch: pytest.MonkeyPatch, steps: list) -> None:
    """Record each optimizer step's gradient and the batches fed since the previous step.

    A step pre-hook appends to ``steps`` the gradient the parameters carry and the training
    batches a forward pre-hook saw the run's model fed since the previous step.
    """
    fed: list = []
    real_build_model, real_build_optimizer = (
        gt.build_from_model_source,
        gt.build_optimizer,
    )

    def build_model(
        source: ModelSourceSchema,
        plan: SourceLayout,
        dims: Mapping[str, int | tuple[Attribute, ...]],
        geometry: TrainGeometry,
    ) -> torch.nn.Module:
        model = real_build_model(source, plan, dims, geometry)

        def record(
            module: torch.nn.Module,
            args: tuple[torch.Tensor, dict[str, torch.Tensor]],
        ) -> None:
            if module.training:
                images, targets = args
                fed.append(
                    (
                        images.detach().clone(),
                        {k: v.detach().clone() for k, v in targets.items()},
                    )
                )

        model.register_forward_pre_hook(record)
        return model

    def build_optimizer(
        *args: OptimizerSpec | torch.nn.Module, **kwargs: float
    ) -> torch.optim.Optimizer:
        optimizer = real_build_optimizer(*args, **kwargs)

        def hook(opt: torch.optim.Optimizer, *_: object) -> None:
            steps.append(
                (
                    torch.cat(
                        [
                            p.grad.detach().reshape(-1)
                            for group in opt.param_groups
                            for p in group["params"]
                            if p.grad is not None
                        ]
                    ).clone(),
                    list(fed),
                )
            )
            fed.clear()

        optimizer.register_step_pre_hook(hook)
        return optimizer

    monkeypatch.setattr(gt, "build_from_model_source", build_model)
    monkeypatch.setattr(gt, "build_optimizer", build_optimizer)


def _window_sizes(steps: list) -> list[int]:
    """Assert each step's gradient is its batches' mean, and return each step's batch count.

    The assertions hold each step's gradient to the mean of the independent gradients of the
    batches fed since the previous step, and require no two batches to share a gradient.
    """
    batch_grads = [
        _batch_gradient(*batch) for _, batches in steps for batch in batches
    ]
    assert len({round(float(g.sum()), 6) for g in batch_grads}) == len(
        batch_grads
    )
    for gradient, batches in steps:
        assert batches
        own = sum(_batch_gradient(*batch) for batch in batches) / len(batches)
        assert gradient.tolist() == pytest.approx(own.tolist(), rel=1e-6)
    return [len(batches) for _, batches in steps]


def test_each_optimizer_step_sees_only_its_own_batch_gradient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At accumulation 1, each batch makes one step carrying that batch's gradient alone."""
    steps: list = []
    _record_steps(monkeypatch, steps)
    run = _trained(tmp_path)

    assert run.status == "completed", run.status_error
    assert _window_sizes(steps) == [1, 1, 1]


def test_no_accumulated_gradient_survives_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No accumulated gradient survives the end of the run.

    After the run ends, each trainable parameter's gradient is absent or zero.
    """
    models: list = []
    capture_model(monkeypatch, models)

    run = _trained(tmp_path)

    assert run.status == "completed", run.status_error
    assert len(models) == 1
    trainable = [p for p in models[0].parameters() if p.requires_grad]
    assert trainable
    for param in trainable:
        assert param.grad is None or float(param.grad.abs().max()) == 0.0


def test_gradient_accumulation_combines_only_its_own_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gradient accumulation combines its own window's batches and no others.

    With two batches per step, each of the two steps carries the mean of its own two batches'
    gradients.
    """
    steps: list = []
    _record_steps(monkeypatch, steps)
    run = _trained(
        tmp_path,
        [1.0, 10.0, 100.0, 1000.0],
        [0.2, 0.5, 0.9, 0.35],
        accumulation=2,
    )

    assert run.status == "completed", run.status_error
    assert _window_sizes(steps) == [2, 2]


def test_an_epoch_last_shorter_window_averages_over_the_batches_it_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An epoch's shorter last window averages over the batches it holds.

    Three batches at accumulation two: the second window holds one batch, and its step carries
    that batch's own gradient, not half of it.
    """
    steps: list = []
    _record_steps(monkeypatch, steps)
    run = _trained(tmp_path, accumulation=2)

    assert run.status == "completed", run.status_error
    assert _window_sizes(steps) == [2, 1]
    assert run.metrics_history[0]["optimizer_steps"] == 2
