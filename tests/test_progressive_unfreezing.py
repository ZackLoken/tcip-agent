"""Progressive-unfreezing fidelity.

Covers the three optimizer_factory helpers (LR scaling, name-keyed optimizer
snapshot/restore) and their integration in generic_trainer.train(): the
non-decreasing-unfreeze guard, inter-stage LR warmup, effective-batch LR
scaling, and a two-stage handoff smoke test, over tiny synthetic data on CPU with
``pretrained=False``.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from tests._chain_fixtures import CLASSIFIER_SOURCE

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from torch import nn  # noqa: E402

from tcip_mcp.pipelines.model_build import STATE_DICT_KEY  # noqa: E402
from tcip_mcp.pipelines.training.generic_trainer import (
    _warmup_starts,  # noqa: E402
    train,  # noqa: E402
)
from tcip_mcp.pipelines.training.optimizer_factory import (  # noqa: E402
    GROUPS_KEY,
    OPTIMIZER_STATE_KEY,
    capture_training_state,
    captured_sources,
    compute_lr_scale,
    restore_training_state,
)
from tests._training_values import (  # noqa: E402
    RATIONALE,
    adamw_optimizer,
    drawn_split,
    evaluation_block,
    stage,
)

IMG = 64
BASE_BB_LR = 1e-3


# --------------------------------------------------------------------------
# Unit tests: pure helpers
# --------------------------------------------------------------------------


def test_compute_lr_scale():
    assert compute_lr_scale(128, 64, 0.5) == pytest.approx(2**0.5)
    assert compute_lr_scale(1024, 64, 0.5) == pytest.approx(4.0)
    assert compute_lr_scale(64, 64, 0.5) == 1.0


def test_capture_restore_roundtrip():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    for p in model[0].parameters():  # freeze first Linear
        p.requires_grad = False

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    x = torch.randn(3, 4)
    opt.zero_grad()
    model(x).sum().backward()
    opt.step()  # creates exp_avg / exp_avg_sq for the trainable params

    state = capture_training_state(model, opt)
    assert set(state[OPTIMIZER_STATE_KEY]) == {"1.weight", "1.bias"}
    assert [
        (g["members"], g["settings"]["lr"]) for g in state[GROUPS_KEY]
    ] == [(["1.weight", "1.bias"], 1e-3)]
    captured_weight = model[1].weight.detach().clone()

    # Move on, unfreeze everything, and restore into a fresh optimizer over the larger set, its
    # newly unfrozen parameters in a group of their own ahead of the captured ones.
    with torch.no_grad():
        model[1].weight.add_(1.0)
    for p in model.parameters():
        p.requires_grad = True
    new_opt = torch.optim.AdamW(
        [
            {"params": model[0].parameters(), "lr": 5e-4},
            {"params": model[1].parameters()},
        ],
        lr=2e-3,
    )
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
    assert _warmup_starts(model, new_opt, state[GROUPS_KEY], [5e-4, 2e-3]) == [
        0.0,
        1e-3,
    ]
    mixed = torch.optim.AdamW(model.parameters(), lr=2e-3)
    with pytest.raises(ValueError, match="param group 0"):
        _warmup_starts(model, mixed, state[GROUPS_KEY], [2e-3])


def test_warmup_reads_only_the_rates_of_the_groups_it_merges():
    """A merged group warms up from its captured groups' shared rate, not the new optimizer's.

    Two captured groups at one rate that differ in an unrelated setting merge into one new group,
    which warms up from that rate.
    """
    model = nn.Linear(2, 1)
    opt = torch.optim.AdamW(
        [
            {"params": [model.weight], "lr": 1e-2, "label": "weight"},
            {"params": [model.bias], "lr": 1e-2, "label": "bias"},
        ]
    )
    model(torch.randn(3, 2)).sum().backward()
    opt.step()
    state = capture_training_state(model, opt)

    merged = torch.optim.AdamW(model.parameters(), lr=2e-2)
    restore_training_state(model, merged, state, group_settings=False)
    assert _warmup_starts(model, merged, state[GROUPS_KEY], [2e-2]) == [1e-2]


def test_a_resume_refuses_a_group_merging_two_captured_groups_even_at_equal_settings():
    """A resume refuses a group merging two captured groups, even at equal settings.

    A resumed group whose members came from two captured groups has no one setting on record, even
    where the two captured groups' settings are equal; a group per captured group resumes.
    """
    model = nn.Linear(2, 1)
    opt = torch.optim.AdamW(
        [
            {"params": [model.weight], "lr": 1e-2},
            {"params": [model.bias], "lr": 1e-2},
        ]
    )
    model(torch.randn(3, 2)).sum().backward()
    opt.step()
    state = capture_training_state(model, opt)
    assert state[GROUPS_KEY][0]["settings"] == state[GROUPS_KEY][1]["settings"]

    with pytest.raises(ValueError, match="param group 0"):
        restore_training_state(
            model,
            torch.optim.AdamW(model.parameters(), lr=1e-2),
            state,
            group_settings=True,
        )
    split = torch.optim.AdamW(
        [{"params": [model.bias]}, {"params": [model.weight]}], lr=5e-3
    )
    restore_training_state(model, split, state, group_settings=True)
    assert [g["lr"] for g in split.param_groups] == [1e-2, 1e-2]


class _CustomGroupRegressor(nn.Linear):
    """A bespoke model whose ``get_param_groups`` puts a tensor-valued setting on its group."""

    def __init__(self):
        super().__init__(2, 1)

    def get_param_groups(self, backbone_lr, head_lr):
        return [
            {
                "params": self.parameters(),
                "lr": head_lr,
                "custom": torch.tensor([1.0, 2.0]),
            }
        ]


def test_a_tensor_valued_bespoke_group_setting_survives_a_resume():
    """A tensor-valued bespoke group setting survives a resume.

    A setting a bespoke ``get_param_groups`` puts on its group, changed while training, comes back
    whole on a resume restore into an optimizer ``build_optimizer`` builds afresh, a multi-element
    tensor included.
    """
    from tcip_mcp.pipelines.schemas import OptimizerSpec
    from tcip_mcp.pipelines.training.optimizer_factory import build_optimizer

    spec = OptimizerSpec.model_validate(adamw_optimizer())
    model = _CustomGroupRegressor()
    opt = build_optimizer(
        spec, model, backbone_lr=spec.backbone_lr, head_lr=spec.head_lr
    )
    model(torch.randn(3, 2)).sum().backward()
    opt.step()
    opt.param_groups[0]["custom"] = torch.tensor([3.0, 4.0])
    state = capture_training_state(model, opt)

    resumed = build_optimizer(
        spec, model, backbone_lr=spec.backbone_lr, head_lr=spec.head_lr
    )
    assert torch.equal(
        resumed.param_groups[0]["custom"], torch.tensor([1.0, 2.0])
    )
    restore_training_state(model, resumed, state, group_settings=True)
    assert torch.equal(
        resumed.param_groups[0]["custom"], torch.tensor([3.0, 4.0])
    )


@pytest.mark.parametrize(
    "attributes",
    [(), ("grade",)],
    ids=["plain", "with-attribute-heads"],
)
def test_an_optimizer_trains_a_built_detectors_backbone_at_backbone_lr_and_the_rest_at_head_lr(
    attributes,
):
    """The optimizer gives a built detector's backbone ``backbone_lr`` and the rest ``head_lr``.

    For a detector the platform's ``build_detector`` builds, its subject's attribute heads beside
    it or not, the optimizer assigns the parameters of its ``backbone`` the stated
    ``backbone_lr`` and every other ``head_lr``; a model exposing neither groups nor a backbone
    refuses differing rates by name and builds one group at equal ones.
    """
    from tcip_mcp import subject_registry
    from tcip_mcp.pipelines.schemas import OptimizerSpec
    from tcip_mcp.pipelines.training.optimizer_factory import build_optimizer
    from tests._chain_fixtures import detector_values
    from tests.bespoke_models import build_bespoke_detection

    spec = OptimizerSpec.model_validate(
        adamw_optimizer(backbone_lr=1e-4, head_lr=1e-3)
    )
    model = build_bespoke_detection(
        **detector_values(),
        attributes=tuple(
            subject_registry.Attribute(name, "ordinal", ("low", "high"))
            for name in attributes
        ),
    ).detector
    opt = build_optimizer(
        spec, model, backbone_lr=spec.backbone_lr, head_lr=spec.head_lr
    )
    rate = {id(p): g["lr"] for g in opt.param_groups for p in g["params"]}
    backbone = {id(p) for p in model.backbone.parameters()}
    assert backbone and len(backbone) < len(rate)
    assert {rate[i] for i in backbone} == {1e-4}
    assert {r for i, r in rate.items() if i not in backbone} == {1e-3}

    bare = nn.Linear(2, 1)
    with pytest.raises(
        ValueError, match="backbone_lr .* and head_lr .* differ"
    ):
        build_optimizer(spec, bare, backbone_lr=1e-4, head_lr=1e-3)
    one_rate = build_optimizer(spec, bare, backbone_lr=1e-3, head_lr=1e-3)
    assert [g["lr"] for g in one_rate.param_groups] == [1e-3]


class _NoStageCount(nn.Module):
    """A model freezing through ``freeze_backbone`` over a backbone declaring no stage count."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)

    def freeze_backbone(self, to_stage: int) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False


def test_a_stage_freeze_a_model_cannot_apply_refuses_by_name():
    """A stage freeze a model cannot apply refuses, naming what is missing.

    A nonzero ``freeze_to`` over a model with no ``freeze_backbone``, and a negative one over a
    backbone declaring no ``num_stages``, refuse; a built classifier's backbone freezes whole
    under a negative one while its head stays trainable.
    """
    from tcip_mcp.pipelines.training.generic_trainer import apply_stage_freeze
    from tests.bespoke_models import build_bespoke_classifier

    with pytest.raises(ValueError, match="freeze_backbone"):
        apply_stage_freeze(nn.Linear(2, 1), 1)
    with pytest.raises(ValueError, match="num_stages"):
        apply_stage_freeze(_NoStageCount(), -1)
    assert apply_stage_freeze(nn.Linear(2, 1), 0) == 3

    from tests._chain_fixtures import band_statistics

    model = build_bespoke_classifier(num_classes=2, **band_statistics())
    apply_stage_freeze(model, -1)
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.heads.parameters())


def test_a_restore_puts_every_tensor_on_its_parameter_dtype_and_the_next_step_runs():
    """A restore puts a one-element parameter's momentum on its dtype, and the next step runs.

    The momentum restored into a float64 model is float64, and the restored optimizer steps.
    """
    model = nn.Linear(1, 1, bias=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(3, 1)).sum().backward()
    opt.step()
    state = capture_training_state(model, opt)

    model = model.double()
    new_opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    restore_training_state(
        model,
        new_opt,
        {**state, STATE_DICT_KEY: model.state_dict()},
        group_settings=False,
    )
    assert new_opt.state[model.weight]["exp_avg"].dtype == torch.float64
    model(torch.randn(3, 1, dtype=torch.float64)).sum().backward()
    new_opt.step()


def test_a_resume_restore_puts_back_every_scheduled_group_setting():
    """A resume restore puts back every scheduled group setting.

    The settings a scheduler writes into a group (OneCycle's betas beside its rate) come back on a
    resume restore, and a handoff restore leaves the built ones.
    """
    model = nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1e-2, total_steps=10
    )
    for _ in range(3):
        model(torch.randn(3, 2)).sum().backward()
        opt.step()
        sched.step()
    state = capture_training_state(model, opt)
    reached = {k: v for k, v in opt.param_groups[0].items() if k != "params"}

    resumed = torch.optim.AdamW(model.parameters(), lr=1e-3)
    torch.optim.lr_scheduler.OneCycleLR(resumed, max_lr=1e-2, total_steps=10)
    restore_training_state(model, resumed, state, group_settings=True)
    assert {
        k: v for k, v in resumed.param_groups[0].items() if k != "params"
    } == reached

    handed = torch.optim.AdamW(model.parameters(), lr=5e-4)
    restore_training_state(model, handed, state, group_settings=False)
    assert handed.param_groups[0]["lr"] == 5e-4 and handed.param_groups[0][
        "betas"
    ] == (0.9, 0.999)


def test_a_restore_into_an_optimizer_missing_a_trained_parameter_is_refused():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(3, 4)).sum().backward()
    opt.step()
    state = capture_training_state(model, opt)

    with pytest.raises(ValueError, match="1.weight"):
        restore_training_state(
            model,
            torch.optim.AdamW(model[0].parameters(), lr=1e-3),
            state,
            group_settings=False,
        )
    restore_training_state(
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        state,
        group_settings=False,
    )


def test_a_restore_dropping_a_parameter_a_stateless_optimizer_held_is_refused():
    """A restore dropping a parameter a stateless optimizer held is refused.

    A momentum-free SGD keeps no per-parameter state, and a restore that drops one of its
    parameters refuses naming it.
    """
    model = nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.0)
    model(torch.randn(3, 2)).sum().backward()
    opt.step()
    state = capture_training_state(model, opt)
    assert state[OPTIMIZER_STATE_KEY] == {}

    for group_settings in (False, True):
        with pytest.raises(ValueError, match="weight"):
            restore_training_state(
                model,
                torch.optim.SGD([model.bias], lr=1e-2),
                state,
                group_settings=group_settings,
            )
    restore_training_state(
        model,
        torch.optim.SGD(model.parameters(), lr=1e-2),
        state,
        group_settings=True,
    )


def test_freeze_to_is_per_stage_for_a_wrapped_bespoke_backbone():
    """``freeze_to`` freezes a wrapped bespoke backbone per stage, not all-or-nothing.

    The agent-written backbone keeps its stages one level down inside a ModuleList.
    """
    import torch.nn as nn

    from tcip_mcp.pipelines.components.backbones import BackboneWrapper

    class Staged(nn.Module):
        """A backbone whose sole named child is a ModuleList of stages."""

        def __init__(self) -> None:
            super().__init__()
            chans = [16, 32, 64, 128]
            self.stages = nn.ModuleList(
                [
                    nn.Conv2d(c_in, c_out, 3, stride=2, padding=1)
                    for c_in, c_out in zip([3, *chans[:-1]], chans)
                ]
            )

        def forward(self, x):
            out = {}
            for i, layer in enumerate(self.stages):
                x = layer(x)
                out[f"s{i}"] = x
            return out

    bb = BackboneWrapper(Staged(), [16, 32, 64, 128])
    counts = []
    for to_stage in range(bb.num_stages + 1):  # 0 .. 4
        bb.freeze_to(to_stage)
        counts.append(
            sum(p.numel() for p in bb.model.parameters() if p.requires_grad)
        )
    assert counts[0] > 0  # freeze_to=0 leaves everything trainable
    assert counts[-1] == 0  # freeze_to=num_stages freezes everything
    # Strictly decreasing: each extra stage frozen removes trainable params.
    assert all(b < a for a, b in zip(counts, counts[1:])), counts


# --------------------------------------------------------------------------
# Integration helpers
# --------------------------------------------------------------------------


def _trained(tmp_path: Path, stages, n: int = 6, **extra):
    """Train a resnet18 classification run over ``stages``; return the run and its directory.

    ``extra`` are the run's other config keys; it trains on ``n`` frames under ``tmp_path``,
    labeled alternately 0 and 1, with no validation side, opened by the launcher's own producer
    and trained over its own loader at its batch size.
    """
    from tests._chain_fixtures import training_config
    from tests.tiny_trainer_fixtures import (
        run_context,
        write_regression_dataset,
    )

    images_dir, csv_path = write_regression_dataset(
        tmp_path / "ds",
        [0.8 if i % 2 == 0 else 0.2 for i in range(n)],
        [i % 2 for i in range(n)],
        height=IMG,
        width=IMG,
    )
    config = training_config(
        tmp_path,
        dict(CLASSIFIER_SOURCE),
        {
            "images_dir": str(images_dir),
            "labels_dir": str(csv_path),
            "split": drawn_split(val_ratio=0.0),
        },
        **{
            "stages": stages,
            "optimizer": adamw_optimizer(
                backbone_lr=BASE_BB_LR, head_lr=1e-3, weight_decay=0
            ),
            "evaluation": evaluation_block(selection_metric="loss"),
            **extra,
        },
    )
    ctx = run_context(tmp_path, config)
    return train(ctx.run, ctx.train_loader, val_loader=None), ctx.run_dir


# --------------------------------------------------------------------------
# Integration tests: train()
# --------------------------------------------------------------------------


def test_monotonic_unfreeze_guard_fails(tmp_path: Path):
    # Stage 0 fully unfreezes; stage 1 re-freezes the backbone -> guard must fire.
    run, _ = _trained(tmp_path, [stage(0), stage(-1)])
    assert run.status == "failed"
    assert "Non-decreasing unfreeze" in run.status_error


def test_warmup_lr_ramps_at_stage_boundary(tmp_path: Path):
    """The learning rate ramps up over warmup at a stage boundary.

    The second stage's three epochs start below the base backbone rate, rise, and reach it at the
    second epoch; every epoch records its target effective batch and trainable parameters.
    """
    run, _ = _trained(tmp_path, [stage(-1), stage(0)], stage_warmup_epochs=2)
    assert run.status == "completed", run.status_error

    stage1 = [m for m in run.metrics_history if m["stage"] == 1]
    assert len(stage1) == 3
    assert 0 < stage1[0]["lr"] < stage1[1]["lr"]
    assert stage1[1]["lr"] == pytest.approx(BASE_BB_LR)
    for m in run.metrics_history:
        assert "target_eff_batch" in m and "trainable_params" in m


def test_lr_scaling_is_relative_to_the_first_stage_effective_batch(
    tmp_path: Path,
):
    """Learning-rate scaling is relative to the first stage's effective batch.

    Batch size 2 targets an effective batch of 2 in the first stage and 8 in the second, so the
    multiplier is (8/2)^0.5 == 2.0. Six samples make three batches, so the second stage's one
    window holds three of the four batches it targets: one optimizer step against three.
    """
    stages = [stage(-1), stage(0, gradient_accumulation_steps=4)]
    run, _ = _trained(
        tmp_path,
        stages,
        lr_scaling={"scale_power": 0.5, "rationale": RATIONALE},
        batch_size=2,
    )
    assert run.status == "completed", run.status_error
    assert [m["target_eff_batch"] for m in run.metrics_history] == [2, 8]
    assert [m["optimizer_steps"] for m in run.metrics_history] == [3, 1]
    assert run.metrics_history[0]["lr"] == pytest.approx(BASE_BB_LR)
    assert run.metrics_history[1]["lr"] == pytest.approx(BASE_BB_LR * 2.0)


def test_two_stage_handoff_smoke(tmp_path: Path):
    run, run_dir = _trained(tmp_path, [stage(-1), stage(0)])
    assert run.status == "completed", run.status_error
    assert all(math.isfinite(m["train_loss"]) for m in run.metrics_history)
    tps = [m["trainable_params"] for m in run.metrics_history]
    assert all(
        b >= a for a, b in zip(tps, tps[1:])
    )  # non-decreasing across stages
    assert (run_dir / "model_best.pt").is_file()
