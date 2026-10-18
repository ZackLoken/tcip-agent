"""``overfit_check`` refuses a model whose loss never moves, and admits one that learns.

A flat loss curve, from a disconnected learnable path or a model with nothing to optimize, fails
the check.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from tcip_mcp.pipelines.model_contract import overfit_check  # noqa: E402

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

    def forward(self, images, targets=None):
        if self.training and targets is not None:
            return {"loss": self.scale * 0.0 + 5.0}
        return {"logits": torch.zeros(2, 1)}


class _LinearProbe(torch.nn.Module):
    """A linear classifier over the image's channel means."""

    def __init__(self, num_classes: int = 2, in_chans: int = 3) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(in_chans, num_classes)

    def forward(self, images, targets=None):
        logits = self.fc(images.mean(dim=(2, 3)))
        if self.training and targets is not None:
            return {
                "cls_loss": torch.nn.functional.cross_entropy(
                    logits, targets["labels"]
                )
            }
        return {"logits": logits}


def test_a_model_whose_loss_never_moves_is_refused():
    """``overfit_check`` fails a model whose loss is identical at its last and first steps."""
    report = overfit_check(
        _DisconnectedParameter(),
        "classification",
        steps=12,
        dims=CLS_DIMS,
        seed=0,
        deterministic=False,
    )
    assert report["passed"] is False
    assert report["issue"] is not None
    assert "did not decrease" in report["issue"], report["issue"]
    assert len(report["losses"]) == 12
    assert report["final"] == report["initial"]


def test_a_model_that_really_learns_passes():
    """A model with a real learnable path drives the loss down and passes the overfit check."""
    report = overfit_check(
        _LinearProbe(),
        "classification",
        steps=25,
        dims=CLS_DIMS,
        seed=0,
        deterministic=False,
    )
    assert report["passed"], report["issue"]
    assert report["issue"] is None
    assert report["final"] < report["initial"]


class _AllFrozen(torch.nn.Module):
    """A stage-0 classifier with every parameter frozen, so nothing to optimize."""

    def __init__(self, num_classes: int = 2, in_chans: int = 3) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(in_chans, num_classes)
        for p in self.fc.parameters():
            p.requires_grad = False

    def forward(self, images, targets=None):
        logits = self.fc(images.mean(dim=(2, 3)))
        if self.training and targets is not None:
            return {
                "cls_loss": torch.nn.functional.cross_entropy(
                    logits, targets["labels"]
                )
            }
        return {"logits": logits}


def test_an_all_frozen_model_reports_rather_than_raises():
    """An all-frozen model yields a failed overfit report naming the empty parameter list.

    The report records no losses and no initial or final loss.
    """
    report = overfit_check(
        _AllFrozen(),
        "classification",
        steps=5,
        dims=CLS_DIMS,
        seed=0,
        deterministic=False,
    )
    assert report["passed"] is False
    assert "empty parameter list" in report["issue"], report["issue"]
    assert report["losses"] == []
    assert report["initial"] is None and report["final"] is None
