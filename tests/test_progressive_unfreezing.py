"""Progressive-unfreezing fidelity.

Covers the three optimizer_factory helpers (LR scaling, name-keyed optimizer
snapshot/restore) and their integration in generic_trainer.train(): the
non-decreasing-unfreeze guard, inter-stage LR warmup, effective-batch LR
scaling, and a two-stage handoff smoke test. Tiny synthetic data, CPU,
``pretrained=False`` so it stays under the CI per-test timeout.
"""

from __future__ import annotations

from tcip_mcp.dataset_layout import UNDATED_BUCKET

import csv
import math
from functools import partial
from pathlib import Path

import pytest

from tests._chain_fixtures import CLASSIFIER_SOURCE

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from torch import nn  # noqa: E402

from tcip_mcp.pipelines.training.generic_trainer import run_loaders, train  # noqa: E402
from tests.tiny_trainer_fixtures import trainer_run  # noqa: E402
from tcip_mcp.pipelines.training.optimizer_factory import (  # noqa: E402
    GROUPS_KEY, OPTIMIZER_STATE_KEY, capture_training_state, captured_sources, compute_lr_scale,
    restore_training_state,
)
from tcip_mcp.pipelines.training.generic_trainer import _warmup_starts  # noqa: E402
from tcip_mcp.pipelines.model_build import STATE_DICT_KEY  # noqa: E402
from tests._image_fixtures import write_noise_image  # noqa: E402
from tests._producer_fixtures import dataset_over  # noqa: E402

IMG = 64
BASE_BB_LR = 1e-3


# --------------------------------------------------------------------------
# Unit tests: pure helpers
# --------------------------------------------------------------------------

def test_compute_lr_scale():
    assert compute_lr_scale(128, 64, 0.5) == pytest.approx(2 ** 0.5)
    assert compute_lr_scale(1024, 64, 0.5) == pytest.approx(4.0)
    assert compute_lr_scale(64, 64, 0.5) == 1.0


def test_capture_restore_roundtrip():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    for p in model[0].parameters():  # freeze first Linear
        p.requires_grad = False

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    x = torch.randn(3, 4)
    opt.zero_grad()
    model(x).sum().backward()
    opt.step()  # creates exp_avg / exp_avg_sq for the trainable params

    state = capture_training_state(model, opt)
    assert set(state[OPTIMIZER_STATE_KEY]) == {"1.weight", "1.bias"}
    assert [(g["members"], g["settings"]["lr"]) for g in state[GROUPS_KEY]] == [
        (["1.weight", "1.bias"], 1e-3)]
    captured_weight = model[1].weight.detach().clone()

    # Move on, unfreeze everything, and restore into a fresh optimizer over the larger set, its
    # newly unfrozen parameters in a group of their own ahead of the captured ones.
    with torch.no_grad():
        model[1].weight.add_(1.0)
    for p in model.parameters():
        p.requires_grad = True
    new_opt = torch.optim.AdamW([{"params": model[0].parameters(), "lr": 5e-4},
                                 {"params": model[1].parameters()}], lr=2e-3)
    restore_training_state(model, new_opt, state, group_settings=False)

    assert torch.equal(model[1].weight, captured_weight)
    assert torch.allclose(
        new_opt.state[model[1].weight]["exp_avg"],
        state[OPTIMIZER_STATE_KEY]["1.weight"]["exp_avg"],
    )
    # Newly unfrozen params carry no optimizer state yet, and the groups keep their own rates.
    assert model[0].weight not in new_opt.state
    assert [g["lr"] for g in new_opt.param_groups] == [5e-4, 2e-3]
    # A group's source is found through the captured membership, whatever its index now; the
    # group of parameters the captured optimizer never held has none, and warms up from zero.
    assert captured_sources(model, new_opt, state[GROUPS_KEY]) == [{None}, {0}]
    assert _warmup_starts(model, new_opt, state[GROUPS_KEY], [5e-4, 2e-3]) == [0.0, 1e-3]
    mixed = torch.optim.AdamW(model.parameters(), lr=2e-3)
    with pytest.raises(ValueError, match="param group 0"):
        _warmup_starts(model, mixed, state[GROUPS_KEY], [2e-3])


def test_warmup_reads_only_the_rates_of_the_groups_it_merges():
    """Two captured groups at one rate that differ in an unrelated setting merge into one new
    group, which warms up from that rate."""
    model = nn.Linear(2, 1)
    opt = torch.optim.AdamW([{"params": [model.weight], "lr": 1e-2, "label": "weight"},
                             {"params": [model.bias], "lr": 1e-2, "label": "bias"}])
    model(torch.randn(3, 2)).sum().backward()
    opt.step()
    state = capture_training_state(model, opt)

    merged = torch.optim.AdamW(model.parameters(), lr=2e-2)
    restore_training_state(model, merged, state, group_settings=False)
    assert _warmup_starts(model, merged, state[GROUPS_KEY], [2e-2]) == [1e-2]


def test_a_resume_refuses_a_group_merging_two_captured_groups_even_at_equal_settings():
    """A resumed group whose members came from two captured groups has no one setting on
    record, even where the two captured groups' settings are equal; a group per captured group
    resumes."""
    model = nn.Linear(2, 1)
    opt = torch.optim.AdamW([{"params": [model.weight], "lr": 1e-2},
                             {"params": [model.bias], "lr": 1e-2}])
    model(torch.randn(3, 2)).sum().backward()
    opt.step()
    state = capture_training_state(model, opt)
    assert state[GROUPS_KEY][0]["settings"] == state[GROUPS_KEY][1]["settings"]

    with pytest.raises(ValueError, match="param group 0"):
        restore_training_state(model, torch.optim.AdamW(model.parameters(), lr=1e-2), state,
                               group_settings=True)
    split = torch.optim.AdamW([{"params": [model.bias]}, {"params": [model.weight]}], lr=5e-3)
    restore_training_state(model, split, state, group_settings=True)
    assert [g["lr"] for g in split.param_groups] == [1e-2, 1e-2]


class _CustomGroupRegressor(nn.Linear):
    """A bespoke model whose ``get_param_groups`` puts a tensor-valued setting on its group."""

    def __init__(self):
        super().__init__(2, 1)

    def get_param_groups(self, backbone_lr, head_lr):
        return [{"params": self.parameters(), "lr": head_lr,
                 "custom": torch.tensor([1.0, 2.0])}]


def test_a_tensor_valued_bespoke_group_setting_survives_a_resume():
    """A setting a bespoke ``get_param_groups`` puts on its group, changed while training, comes
    back whole on a resume restore into an optimizer ``build_optimizer`` builds afresh, a
    multi-element tensor included."""
    from tcip_mcp.pipelines.schemas import OptimizerSpec
    from tcip_mcp.pipelines.training.optimizer_factory import build_optimizer
    from tests._training_values import adamw_optimizer

    spec = OptimizerSpec.model_validate(adamw_optimizer())
    model = _CustomGroupRegressor()
    opt = build_optimizer(spec, model, backbone_lr=spec.backbone_lr, head_lr=spec.head_lr)
    model(torch.randn(3, 2)).sum().backward()
    opt.step()
    opt.param_groups[0]["custom"] = torch.tensor([3.0, 4.0])
    state = capture_training_state(model, opt)

    resumed = build_optimizer(spec, model, backbone_lr=spec.backbone_lr, head_lr=spec.head_lr)
    assert torch.equal(resumed.param_groups[0]["custom"], torch.tensor([1.0, 2.0]))
    restore_training_state(model, resumed, state, group_settings=True)
    assert torch.equal(resumed.param_groups[0]["custom"], torch.tensor([3.0, 4.0]))


def test_a_restore_puts_every_tensor_on_its_parameter_dtype_and_the_next_step_runs():
    """A one-element parameter's momentum is a one-element tensor: it lands on the parameter's
    dtype like every other state tensor, and the restored optimizer steps."""
    model = nn.Linear(1, 1, bias=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(3, 1)).sum().backward()
    opt.step()
    state = capture_training_state(model, opt)

    model = model.double()
    new_opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    restore_training_state(model, new_opt, {**state, STATE_DICT_KEY: model.state_dict()},
                           group_settings=False)
    assert new_opt.state[model.weight]["exp_avg"].dtype == torch.float64
    model(torch.randn(3, 1, dtype=torch.float64)).sum().backward()
    new_opt.step()


def test_a_resume_restore_puts_back_every_scheduled_group_setting():
    """The settings a scheduler writes into a group (OneCycle's betas beside its rate) come back
    on a resume restore, and a handoff restore leaves the built ones."""
    model = nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-2, total_steps=10)
    for _ in range(3):
        model(torch.randn(3, 2)).sum().backward()
        opt.step()
        sched.step()
    state = capture_training_state(model, opt)
    reached = {k: v for k, v in opt.param_groups[0].items() if k != "params"}

    resumed = torch.optim.AdamW(model.parameters(), lr=1e-3)
    torch.optim.lr_scheduler.OneCycleLR(resumed, max_lr=1e-2, total_steps=10)
    restore_training_state(model, resumed, state, group_settings=True)
    assert {k: v for k, v in resumed.param_groups[0].items() if k != "params"} == reached

    handed = torch.optim.AdamW(model.parameters(), lr=5e-4)
    restore_training_state(model, handed, state, group_settings=False)
    assert handed.param_groups[0]["lr"] == 5e-4 and handed.param_groups[0]["betas"] == (0.9,
                                                                                        0.999)


def test_a_restore_into_an_optimizer_missing_a_trained_parameter_is_refused():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(3, 4)).sum().backward()
    opt.step()
    state = capture_training_state(model, opt)

    with pytest.raises(ValueError, match="1.weight"):
        restore_training_state(model, torch.optim.AdamW(model[0].parameters(), lr=1e-3), state,
                               group_settings=False)
    restore_training_state(model, torch.optim.AdamW(model.parameters(), lr=1e-3), state,
                           group_settings=False)


def test_a_restore_dropping_a_parameter_a_stateless_optimizer_held_is_refused():
    """Membership, not optimizer state, says what the captured optimizer held: a momentum-free
    SGD keeps no per-parameter state, and a restore that drops one of its parameters still
    refuses."""
    model = nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.0)
    model(torch.randn(3, 2)).sum().backward()
    opt.step()
    state = capture_training_state(model, opt)
    assert state[OPTIMIZER_STATE_KEY] == {}

    for group_settings in (False, True):
        with pytest.raises(ValueError, match="weight"):
            restore_training_state(model, torch.optim.SGD([model.bias], lr=1e-2), state,
                                   group_settings=group_settings)
    restore_training_state(model, torch.optim.SGD(model.parameters(), lr=1e-2), state,
                           group_settings=True)


def test_freeze_to_is_per_stage_for_a_wrapped_bespoke_backbone():
    """A wrapped agent-written backbone, its stages one level down inside a ModuleList, freezes
    per stage, not all-or-nothing."""
    import torch.nn as nn

    from tcip_mcp.pipelines.components.backbones import BackboneWrapper

    class Staged(nn.Module):
        """Sole named child is a ModuleList, the shape the descent exists for."""

        def __init__(self) -> None:
            super().__init__()
            chans = [16, 32, 64, 128]
            self.stages = nn.ModuleList([
                nn.Conv2d(c_in, c_out, 3, stride=2, padding=1)
                for c_in, c_out in zip([3, *chans[:-1]], chans)
            ])

        def forward(self, x):
            out = {}
            for i, stage in enumerate(self.stages):
                x = stage(x)
                out[f"s{i}"] = x
            return out

    bb = BackboneWrapper(Staged(), [16, 32, 64, 128])
    counts = []
    for stage in range(bb.num_stages + 1):  # 0 .. 4
        bb.freeze_to(stage)
        counts.append(sum(p.numel() for p in bb.model.parameters() if p.requires_grad))
    assert counts[0] > 0  # freeze_to=0 leaves everything trainable
    assert counts[-1] == 0  # freeze_to=num_stages freezes everything
    # Strictly decreasing: each extra stage frozen removes trainable params.
    assert all(b < a for a, b in zip(counts, counts[1:])), counts


# --------------------------------------------------------------------------
# Integration helpers
# --------------------------------------------------------------------------

_save_png = partial(write_noise_image, size=IMG)


def _classification_loader(tmp_path: Path, run, n: int = 6):
    """``run``'s training loader over ``n`` two-label frames under ``tmp_path``, built by the
    platform's own loader builder (``generic_trainer.run_loaders``) at the run's batch size."""
    images_dir = tmp_path / "images" / UNDATED_BUCKET
    rows = []
    for i in range(n):
        _save_png(images_dir / f"img{i}.png", bright=(i % 2 == 0))
        rows.append((f"img{i}", i % 2))
    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("stem", "label"))
        w.writerows(rows)
    ds = dataset_over("classification", str(images_dir), str(csv_path))
    return run_loaders(run, ds, None)[0]


def _model_source() -> dict:
    # resnet18 (not tv_resnet50): these integration tests exercise backbone-agnostic
    # LR-schedule / freeze / warmup logic. The smaller backbone routes through the identical
    # BackboneWrapper.freeze_to path and cuts per-test model-construction cost. The tv_* freeze
    # branch stays covered by test_freeze_to_is_per_stage_for_tv_backbones (kept on resnet50).
    return dict(CLASSIFIER_SOURCE)


def _cfg(stages, **extra) -> dict:
    from tests._chain_fixtures import training_config

    # The sizes _classification_loader's RGB, two-label table resolves.
    return training_config(
        _model_source(), {"num_channels": 3, "num_classes": 2, "scope": {}},
        **{"stages": stages,
           "optimizer": {"name": "adamw", "backbone_lr": BASE_BB_LR, "head_lr": 1e-3,
                         "weight_decay": 0},
           **extra})


# --------------------------------------------------------------------------
# Integration tests: train()
# --------------------------------------------------------------------------

def test_monotonic_unfreeze_guard_fails(tmp_path: Path):
    # Stage 0 fully unfreezes; stage 1 re-freezes the backbone -> guard must fire.
    cfg = _cfg([{"freeze_to": 0, "epochs": 1}, {"freeze_to": -1, "epochs": 1}])
    run = trainer_run(cfg, tmp_path / "out", project=tmp_path, has_val_loader=False,
                      id="auto-run-46")
    run = train(run, _classification_loader(tmp_path, run), val_loader=None)
    assert run.status == "failed"
    assert "Non-decreasing unfreeze" in run.status_error


def test_warmup_lr_ramps_at_stage_boundary(tmp_path: Path):
    cfg = _cfg(
        [{"freeze_to": -1, "epochs": 1}, {"freeze_to": 0, "epochs": 2}],
        stage_warmup_epochs=2,
    )
    run = trainer_run(cfg, tmp_path / "out", project=tmp_path, has_val_loader=False,
                      id="auto-run-47")
    run = train(run, _classification_loader(tmp_path, run), val_loader=None)
    assert run.status == "completed", run.status_error

    stage1 = [m for m in run.metrics_history if m["stage"] == 1]
    assert len(stage1) == 2
    assert 0 < stage1[0]["lr"] < stage1[1]["lr"]
    assert stage1[1]["lr"] == pytest.approx(BASE_BB_LR)
    for m in run.metrics_history:
        assert "target_eff_batch" in m and "trainable_params" in m


def test_lr_scaling_is_relative_to_the_first_stage_effective_batch(tmp_path: Path):
    """Batch size 2 targets an effective batch of 2 in the first stage and 8 in the second, so
    the multiplier is (8/2)^0.5 == 2.0. Six samples make three batches, so the second stage's one
    window holds three of the four batches it targets: one optimizer step against three."""
    stages = [{"freeze_to": -1, "epochs": 1},
              {"freeze_to": 0, "epochs": 1, "gradient_accumulation_steps": 4}]
    cfg = _cfg(stages, lr_scaling={"scale_power": 0.5}, batch_size=2)
    run = trainer_run(cfg, tmp_path / "out", project=tmp_path, has_val_loader=False,
                      id="auto-run-48")
    run = train(run, _classification_loader(tmp_path, run))
    assert run.status == "completed", run.status_error
    assert [m["target_eff_batch"] for m in run.metrics_history] == [2, 8]
    assert [m["optimizer_steps"] for m in run.metrics_history] == [3, 1]
    assert run.metrics_history[0]["lr"] == pytest.approx(BASE_BB_LR)
    assert run.metrics_history[1]["lr"] == pytest.approx(BASE_BB_LR * 2.0)


def test_two_stage_handoff_smoke(tmp_path: Path):
    cfg = _cfg([{"freeze_to": -1, "epochs": 1}, {"freeze_to": 0, "epochs": 1}])
    run = trainer_run(cfg, tmp_path / "out", project=tmp_path, has_val_loader=False,
                      id="auto-run-50")
    run = train(run, _classification_loader(tmp_path, run), val_loader=None)
    assert run.status == "completed", run.status_error
    assert all(math.isfinite(m["train_loss"]) for m in run.metrics_history)
    tps = [m["trainable_params"] for m in run.metrics_history]
    assert all(b >= a for a, b in zip(tps, tps[1:]))  # non-decreasing across stages
    assert (tmp_path / "out" / "model_best.pt").is_file()
