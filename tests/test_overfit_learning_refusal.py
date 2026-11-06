"""``overfit_check`` refuses a model whose loss never moves, and admits one that learns.

A flat loss curve, from a disconnected learnable path or a model with nothing to optimize, fails
the check. The check steps through the config's own optimizer block, and a config stating none
fails it naming the block.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("torch")
import torch  # noqa: E402

from tcip_mcp.pipelines.model_contract import overfit_check  # noqa: E402
from tests._training_values import overfit_optimizer  # noqa: E402

# What the checks below synthesize their batch at, the shape a run resolves for itself.
CLS_DIMS = {"in_chans": 3, "num_classes": 2, "img_size": (64, 64)}


class _DisconnectedParameter(torch.nn.Module):
    """A learnable parameter the loss does not actually depend on.

    Every gradient is zero, so the optimizer moves nothing and each step returns the identical
    loss.
    """

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(
        self, images: torch.Tensor, targets: dict | None = None
    ) -> dict[str, torch.Tensor]:
        if self.training and targets is not None:
            return {"loss": self.scale * 0.0 + 5.0}
        return {"logits": torch.zeros(2, 1)}


class _LinearProbe(torch.nn.Module):
    """A linear classifier over the image's channel means."""

    def __init__(self, num_classes: int = 2, in_chans: int = 3) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(in_chans, num_classes)

    def forward(
        self, images: torch.Tensor, targets: dict | None = None
    ) -> dict[str, torch.Tensor]:
        logits = self.fc(images.mean(dim=(2, 3)))
        if self.training and targets is not None:
            return {
                "cls_loss": torch.nn.functional.cross_entropy(
                    logits, targets["labels"]
                )
            }
        return {"logits": logits}


def test_a_model_whose_loss_never_moves_is_refused() -> None:
    """``overfit_check`` fails a model whose loss is identical at its last and first steps."""
    report = overfit_check(
        _DisconnectedParameter(),
        "classification",
        optimizer=overfit_optimizer(),
        steps=12,
        dims=CLS_DIMS,
        seed=0,
        deterministic=False,
        device=torch.device("cpu"),
    )
    assert report.passed is False
    assert report.issue is not None
    assert "did not decrease" in report.issue, report.issue
    assert len(report.losses) == 12
    assert report.final == report.initial


def test_a_model_that_really_learns_passes() -> None:
    """A model with a real learnable path drives the loss down and passes the overfit check."""
    report = overfit_check(
        _LinearProbe(),
        "classification",
        optimizer=overfit_optimizer(),
        steps=25,
        dims=CLS_DIMS,
        seed=0,
        deterministic=False,
        device=torch.device("cpu"),
    )
    assert report.passed, report.issue
    assert report.final is not None and report.initial is not None
    assert report.final < report.initial


class _AllFrozen(torch.nn.Module):
    """A stage-0 classifier with every parameter frozen, so nothing to optimize."""

    def __init__(self, num_classes: int = 2, in_chans: int = 3) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(in_chans, num_classes)
        for p in self.fc.parameters():
            p.requires_grad = False

    def forward(
        self, images: torch.Tensor, targets: dict | None = None
    ) -> dict[str, torch.Tensor]:
        logits = self.fc(images.mean(dim=(2, 3)))
        if self.training and targets is not None:
            return {
                "cls_loss": torch.nn.functional.cross_entropy(
                    logits, targets["labels"]
                )
            }
        return {"logits": logits}


def test_an_all_frozen_model_reports_rather_than_raises() -> None:
    """An all-frozen model yields a failed overfit report saying it trains no parameter.

    The report records no losses and no initial or final loss.
    """
    report = overfit_check(
        _AllFrozen(),
        "classification",
        optimizer=overfit_optimizer(),
        steps=5,
        dims=CLS_DIMS,
        seed=0,
        deterministic=False,
        device=torch.device("cpu"),
    )
    assert report.issue is not None and "trains no parameter" in report.issue
    assert report.passed is False
    assert report.losses == []
    assert report.initial is None and report.final is None


def _regressor_config(project: Path, **overrides: object) -> dict:
    """Return a backbone regressor's config over frames written under ``project``.

    The frames are the sample training intensities valued twice their intensity, with no
    validation side; ``overrides`` replace their keys.
    """
    from tests.tiny_trainer_fixtures import (
        BACKBONE_REGRESSOR,
        TRAIN_INTENSITIES,
        frames_data,
        regressor_config,
    )

    return regressor_config(
        1,
        data=frames_data(
            project / "ds",
            TRAIN_INTENSITIES,
            [2.0 * c for c in TRAIN_INTENSITIES],
        ),
        project=project,
        builder=BACKBONE_REGRESSOR,
        **overrides,
    )


def test_the_preflight_overfit_check_steps_at_the_configs_head_rate(
    tmp_path: Path,
) -> None:
    """Two preflights whose configs differ only in ``head_lr`` report different overfit losses.

    The check steps the model through the config's own optimizer block, so the head rate it
    states moves the weight by a different amount; both checks pass.
    """
    from tcip_mcp.tools.training_tools import preflight_config
    from tests._training_values import adamw_optimizer

    def losses(head_lr: float) -> list[float]:
        config = _regressor_config(
            tmp_path,
            optimizer=adamw_optimizer(backbone_lr=1e-3, head_lr=head_lr),
        )
        report = preflight_config(tmp_path, config, smoke=True, overfit=True)
        assert report["valid"], report["issues"]
        check = report["overfit_check"]
        assert check["issue"] is None, check
        return report["overfit_check"]["losses"]

    slow, fast = losses(1e-3), losses(1e-1)
    assert slow[0] == fast[0]
    assert slow[1:] != fast[1:]


def test_the_preflight_overfit_check_of_a_config_with_no_optimizer_names_it(
    tmp_path: Path,
) -> None:
    """A preflight's overfit check of a config stating no ``optimizer`` block fails naming it.

    The config trains through its own ``training_source``, which reads no block, so the preflight
    stands; its overfit check takes no step and its issue names the block.
    """
    from tcip_mcp.tools.training_tools import preflight_config
    from tests._chain_fixtures import BESPOKE_MODELS, SAVE_BUILT_WEIGHTS

    config = _regressor_config(tmp_path, training_source=SAVE_BUILT_WEIGHTS)
    del config["optimizer"]
    config["model_source"]["source_files"].append(BESPOKE_MODELS)

    report = preflight_config(tmp_path, config, smoke=True, overfit=True)
    assert report["valid"], report["issues"]
    check = report["overfit_check"]
    assert "optimizer" in check["issue"], check["issue"]
    assert check["losses"] == []
