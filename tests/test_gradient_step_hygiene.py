"""An optimizer step descends on its own window's gradient, never on a running sum.

The trained model is the phenotype's measuring instrument, so the loop must feed each step the
gradient of the batches that step covers and nothing else. These runs use a model whose per-batch
gradient is fixed by the batch's data, so the gradient an optimizer step receives can be compared
against that batch's own gradient computed independently.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from tcip_mcp.pipelines.training import generic_trainer as gt
from tcip_mcp.pipelines.training.generic_trainer import run_loaders, train
from tests.tiny_trainer_fixtures import (
    ConstantImageDataset,
    build_data_scaled_gradient_model,
    capture_model,
    regressor_config,
    trainer_run,
)

BUILDER = "tests.tiny_trainer_fixtures:build_data_scaled_gradient_model"

# Deliberately spread over three orders of magnitude: a step that carried a neighbor's gradient
# too would land nowhere near the batch's own.
SKEWED_VALUES = [1.0, 10.0, 100.0]
SKEWED_INTENSITIES = [0.2, 0.5, 0.9]


def _loader(run, values, intensities):
    """``run``'s training loader over frames at ``intensities`` valued ``values``, built by the
    platform's own loader builder (``generic_trainer.run_loaders``): a loader built twice for
    one run serves its first epoch in one order, seeded from the run's config."""
    dataset = ConstantImageDataset(intensities, values, height=7, width=11)
    return run_loaders(run, dataset, None)[0]


def _config(*, accumulation: int = 1) -> dict:
    return regressor_config(
        1, builder=BUILDER, gradient_accumulation_steps=accumulation, batch_size=1, seed=0,
        optimizer={"name": "adamw", "backbone_lr": 0.01, "head_lr": 0.01, "weight_decay": 0.0})


def _batch_gradient(images, targets) -> torch.Tensor:
    """One batch's own gradient, from a fresh instance of the model the run builds."""
    model = build_data_scaled_gradient_model()
    model.train()
    sum(model(images, targets).values()).backward()
    # every parameter must carry a gradient here; skipping one would shrink the comparison
    for p in model.parameters():
        assert p.grad is not None
    return torch.cat([p.grad.detach().reshape(-1) for p in model.parameters()])


def _record_steps(monkeypatch, steps: list) -> None:
    """Append to ``steps``, at every optimizer step, the gradient the parameters carry and the
    training batches the run's model was fed since the previous step."""
    fed: list = []
    real_build_model, real_build_optimizer = gt.build_from_model_source, gt.build_optimizer

    def build_model(source, dims):
        model = real_build_model(source, dims)

        def record(module, args):
            if module.training:
                images, targets = args
                fed.append((images.detach().clone(),
                            {k: v.detach().clone() for k, v in targets.items()}))

        model.register_forward_pre_hook(record)
        return model

    def build_optimizer(*args, **kwargs):
        optimizer = real_build_optimizer(*args, **kwargs)

        def hook(opt, *_):
            steps.append((torch.cat([
                p.grad.detach().reshape(-1) for group in opt.param_groups
                for p in group["params"] if p.grad is not None]).clone(), list(fed)))
            fed.clear()

        optimizer.register_step_pre_hook(hook)
        return optimizer

    monkeypatch.setattr(gt, "build_from_model_source", build_model)
    monkeypatch.setattr(gt, "build_optimizer", build_optimizer)


def _window_sizes(steps: list) -> list[int]:
    """How many batches each of ``steps`` covers, after asserting that each step's gradient is
    the mean of the independent gradients of exactly the batches fed since the step before, and
    that no two batches share a gradient, so a step carrying another window's cannot pass."""
    batch_grads = [_batch_gradient(*batch) for _, batches in steps for batch in batches]
    assert len({round(float(g.sum()), 6) for g in batch_grads}) == len(batch_grads)
    for gradient, batches in steps:
        assert batches
        own = sum(_batch_gradient(*batch) for batch in batches) / len(batches)
        assert gradient.tolist() == pytest.approx(own.tolist(), rel=1e-6)
    return [len(batches) for _, batches in steps]


def test_each_optimizer_step_sees_only_its_own_batch_gradient(tmp_path, monkeypatch):
    """One step per batch at accumulation 1, each carrying that batch's gradient alone."""
    run = trainer_run(
        _config(), tmp_path / "out", project=tmp_path, has_val_loader=False, id="auto-run-34"
    )
    steps: list = []
    _record_steps(monkeypatch, steps)
    run = train(run, _loader(run, SKEWED_VALUES, SKEWED_INTENSITIES))

    assert run.status == "completed", run.status_error
    assert _window_sizes(steps) == [1, 1, 1]


def test_no_accumulated_gradient_survives_the_run(tmp_path, monkeypatch):
    """The parameters carry no leftover gradient once the run ends, so nothing from the epoch's
    batches is still sitting there to be descended on again."""
    models: list = []
    capture_model(monkeypatch, models)

    run = trainer_run(
        _config(), tmp_path / "out", project=tmp_path, has_val_loader=False, id="auto-run-35"
    )
    run = train(run, _loader(run, SKEWED_VALUES, SKEWED_INTENSITIES))

    assert run.status == "completed", run.status_error
    assert len(models) == 1
    trainable = [p for p in models[0].parameters() if p.requires_grad]
    assert trainable
    for param in trainable:
        assert param.grad is None or float(param.grad.abs().max()) == 0.0


def test_gradient_accumulation_combines_only_its_own_window(tmp_path, monkeypatch):
    """Accumulation still holds: with two batches per step, a step carries the mean of exactly
    those two batches' gradients, and the window does not carry over into the next step."""
    values, intensities = [1.0, 10.0, 100.0, 1000.0], [0.2, 0.5, 0.9, 0.35]
    run = trainer_run(_config(accumulation=2), tmp_path / "out", project=tmp_path,
                      has_val_loader=False, id="auto-run-36")
    steps: list = []
    _record_steps(monkeypatch, steps)
    run = train(run, _loader(run, values, intensities))

    assert run.status == "completed", run.status_error
    assert _window_sizes(steps) == [2, 2]


def test_an_epoch_last_shorter_window_averages_over_the_batches_it_holds(tmp_path, monkeypatch):
    """Three batches at accumulation two: the second window holds one batch, and its step carries
    that batch's own gradient, not half of it."""
    run = trainer_run(_config(accumulation=2), tmp_path / "out", project=tmp_path,
                      has_val_loader=False, id="auto-run-37")
    steps: list = []
    _record_steps(monkeypatch, steps)
    run = train(run, _loader(run, SKEWED_VALUES, SKEWED_INTENSITIES))

    assert run.status == "completed", run.status_error
    assert _window_sizes(steps) == [2, 1]
    assert run.metrics_history[0]["optimizer_steps"] == 2
