"""A model's parameter groups are named, one of them the head group whose rate a run reports."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from tests._training_values import adamw_optimizer  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path


class _OwnGroups(nn.Module):
    """A model whose ``get_param_groups`` returns the groups it was built with, rates filled in.

    Each group is ``(name, rate)``: ``name`` ``None`` states no name, ``rate`` ``"backbone"`` or
    ``"head"`` places that rate and ``None`` states no ``lr``.
    """

    def __init__(self, groups: list[tuple[str | None, str | None]]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(2, 2) for _ in groups)
        self.groups = groups

    def get_param_groups(
        self, backbone_lr: float, head_lr: float
    ) -> list[dict]:
        rates = {"backbone": backbone_lr, "head": head_lr}
        built = []
        for layer, (name, rate) in zip(self.layers, self.groups):
            group: dict = {"params": list(layer.parameters())}
            if name is not None:
                group["name"] = name
            if rate is not None:
                group["lr"] = rates[rate]
            built.append(group)
        return built


@pytest.mark.parametrize(
    "groups",
    [
        [("backbone", "backbone"), ("neck", "head")],
        [("backbone", "backbone"), ("head", "head"), ("head", "head")],
        [("backbone", "backbone"), (None, "head")],
        [("backbone", "backbone"), ("head", None)],
    ],
    ids=["no-head", "two-heads", "unnamed", "no-lr"],
)
def test_a_models_own_groups_without_one_named_head_with_rates_are_refused(
    groups: list[tuple[str | None, str | None]],
) -> None:
    """The factory refuses a model's own groups naming no head, two, a nameless group or no rate.

    Each refusal names the head group, so a group torch would train at its own default rate never
    reaches an optimizer.
    """
    from tcip_mcp.pipelines.schemas import OptimizerSpec
    from tcip_mcp.pipelines.training.optimizer_factory import build_optimizer

    spec = OptimizerSpec.model_validate(adamw_optimizer())
    with pytest.raises(ValueError, match="'head'"):
        build_optimizer(
            spec,
            _OwnGroups(groups),
            backbone_lr=spec.backbone_lr,
            head_lr=spec.head_lr,
        )


def test_a_models_own_named_groups_train_at_their_own_rates() -> None:
    """A model's own groups named uniquely, one of them the head, each with a rate, are admitted.

    The head group trains at ``head_lr`` and the others at the rates the model placed.
    """
    from tcip_mcp.pipelines.schemas import OptimizerSpec
    from tcip_mcp.pipelines.training.optimizer_factory import (
        build_optimizer,
        head_group,
    )

    spec = OptimizerSpec.model_validate(adamw_optimizer())
    optimizer = build_optimizer(
        spec,
        _OwnGroups(
            [("backbone", "backbone"), ("neck", "head"), ("head", "head")]
        ),
        backbone_lr=spec.backbone_lr,
        head_lr=spec.head_lr,
    )
    assert head_group(optimizer)["lr"] == spec.head_lr
    assert [g["lr"] for g in optimizer.param_groups] == [
        spec.backbone_lr,
        spec.head_lr,
        spec.head_lr,
    ]


def test_the_epoch_rows_rate_is_the_head_groups_for_a_model_with_a_backbone(
    tmp_path: Path,
) -> None:
    """A run of a model holding a ``backbone`` module reports the head group's rate as ``lr``.

    The model's offset trains in its backbone at 1e-4 and its weight at 1e-2; the one epoch row
    the default trainer writes reports 1e-2.
    """
    from tcip_mcp.pipelines.training.generic_trainer import train
    from tests.tiny_trainer_fixtures import (
        BACKBONE_REGRESSOR,
        TRAIN_INTENSITIES,
        frames_data,
        regressor_config,
        run_context,
    )

    config = regressor_config(
        1,
        data=frames_data(
            tmp_path / "ds",
            TRAIN_INTENSITIES,
            [2.0 * c for c in TRAIN_INTENSITIES],
        ),
        project=tmp_path,
        builder=BACKBONE_REGRESSOR,
        optimizer=adamw_optimizer(backbone_lr=1e-4, head_lr=1e-2),
    )
    ctx = run_context(tmp_path, config)
    run = train(ctx.run, ctx.train_loader, val_loader=None)

    assert run.status == "completed", run.status_error
    assert [row["lr"] for row in run.metrics_history] == [1e-2]
