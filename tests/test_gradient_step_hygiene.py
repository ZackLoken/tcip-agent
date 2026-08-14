"""An optimizer step descends on its own window's gradient, never on a running sum.

The trained model is the phenotype's measuring instrument, so the loop must feed each step the
gradient of the batches that step covers and nothing else. These runs use a model whose per-batch
gradient is fixed by the batch's data, so the gradient an optimizer step receives can be compared
against that batch's own gradient computed independently.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader

from tcip_mcp.pipelines.training import generic_trainer as gt
from tcip_mcp.pipelines.training.generic_trainer import create_run, task_collate, train
from tests.tiny_trainer_fixtures import (
    ConstantImageDataset,
    build_data_scaled_gradient_model,
)

BUILDER = "tests.tiny_trainer_fixtures:build_data_scaled_gradient_model"

# Deliberately spread over three orders of magnitude: a step that carried a neighbour's gradient
# too would land nowhere near the batch's own.
SKEWED_VALUES = [1.0, 10.0, 100.0]
SKEWED_INTENSITIES = [0.2, 0.5, 0.9]


def _loader(values, intensities, batch_size: int = 1) -> DataLoader:
    dataset = ConstantImageDataset(intensities, values, height=7, width=11)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=task_collate("regression"))


def _config(*, accumulation: int = 1) -> dict:
    return {
        "model_source": {"builder": BUILDER, "task": "regression", "in_chans": 1},
        "device": "cpu",
        "mixed_precision": False,
        "stages": [{"freeze_to": 0, "epochs": 1}],
        "optimizer": {"name": "adamw", "backbone_lr": 0.01, "head_lr": 0.01, "weight_decay": 0.0},
        "gradient_accumulation_steps": accumulation,
        "checkpoint_every_n_epochs": 0,
        "early_stopping": {"enabled": False},
    }


def _per_batch_gradients(loader: DataLoader) -> list[torch.Tensor]:
    """Each batch's own gradient, from a fresh instance of the same model the run builds."""
    model = build_data_scaled_gradient_model()
    model.train()
    grads = []
    for images, targets in loader:
        model.zero_grad(set_to_none=True)
        losses = model(images, targets)
        sum(losses.values()).backward()
        grads.append(torch.cat([p.grad.detach().reshape(-1) for p in model.parameters()]))
    return grads


def _record_step_gradients(monkeypatch, sink: list) -> None:
    """Snapshot the gradients present on the parameters at every optimizer step."""
    real_build_optimizer = gt.build_optimizer

    def build(*args, **kwargs):
        optimizer = real_build_optimizer(*args, **kwargs)

        def hook(opt, *_):
            sink.append(torch.cat([
                p.grad.detach().reshape(-1) for group in opt.param_groups
                for p in group["params"] if p.grad is not None]).clone())

        optimizer.register_step_pre_hook(hook)
        return optimizer

    monkeypatch.setattr(gt, "build_optimizer", build)


def _capture_model(monkeypatch, sink: list) -> None:
    real_build_model = gt.build_model

    def build(config):
        model = real_build_model(config)
        sink.append(model)
        return model

    monkeypatch.setattr(gt, "build_model", build)


def test_each_optimizer_step_sees_only_its_own_batch_gradient(tmp_path, monkeypatch):
    """One step per batch at accumulation 1, each carrying that batch's gradient alone."""
    loader = _loader(SKEWED_VALUES, SKEWED_INTENSITIES)
    expected = _per_batch_gradients(loader)
    assert len(expected) == 3
    # Distinct per batch, so a step reading the wrong window cannot coincide with the right one.
    assert len({round(float(g.sum()), 6) for g in expected}) == 3

    steps: list = []
    _record_step_gradients(monkeypatch, steps)
    run = create_run(_config(), str(tmp_path / "out"))
    run = train(run, loader, task="regression")

    assert run.status == "completed", run.error
    assert len(steps) == len(expected)
    for seen, own in zip(steps, expected):
        assert seen.tolist() == pytest.approx(own.tolist(), rel=1e-6)


def test_no_accumulated_gradient_survives_the_run(tmp_path, monkeypatch):
    """The parameters carry no leftover gradient once the run ends, so nothing from the epoch's
    batches is still sitting there to be descended on again."""
    loader = _loader(SKEWED_VALUES, SKEWED_INTENSITIES)
    models: list = []
    _capture_model(monkeypatch, models)

    run = create_run(_config(), str(tmp_path / "out"))
    run = train(run, loader, task="regression")

    assert run.status == "completed", run.error
    assert len(models) == 1
    trainable = [p for p in models[0].parameters() if p.requires_grad]
    assert trainable
    for param in trainable:
        assert param.grad is None or float(param.grad.abs().max()) == 0.0


def test_gradient_accumulation_combines_only_its_own_window(tmp_path, monkeypatch):
    """Accumulation still holds: with two batches per step, a step carries the mean of exactly
    those two batches' gradients, and the window does not carry over into the next step."""
    values = [1.0, 10.0, 100.0, 1000.0]
    loader = _loader(values, [0.2, 0.5, 0.9, 0.35])
    expected = _per_batch_gradients(loader)
    assert len(expected) == 4

    steps: list = []
    _record_step_gradients(monkeypatch, steps)
    run = create_run(_config(accumulation=2), str(tmp_path / "out"))
    run = train(run, loader, task="regression")

    assert run.status == "completed", run.error
    assert len(steps) == 2
    windows = [(expected[0] + expected[1]) / 2, (expected[2] + expected[3]) / 2]
    for seen, window in zip(steps, windows):
        assert seen.tolist() == pytest.approx(window.tolist(), rel=1e-6)
