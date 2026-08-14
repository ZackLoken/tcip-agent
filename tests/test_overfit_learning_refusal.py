"""``overfit_check`` refuses a model whose loss never moves, and admits one that learns.

The check is the cheap proof a from-scratch model actually optimizes, so a flat loss curve (a
frozen or disconnected learnable path, an optimizer stepping nothing) is the failure it exists to
report. An unchanged loss is a refusal, never a pass.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tcip_mcp.pipelines.model_contract import overfit_check  # noqa: E402


class _DisconnectedParameter(torch.nn.Module):
    """A learnable parameter the loss does not actually depend on.

    Every gradient is zero, so the optimizer moves nothing and each step returns the identical
    loss: the shape of a model that reports a learnable path it cannot train through.
    """

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, images, targets=None):
        if self.training and targets is not None:
            return {"loss": self.scale * 0.0 + 5.0}
        return {"logits": torch.zeros(2, 1)}


class _LinearProbe(torch.nn.Module):
    """A small but genuinely learnable classifier over channel means."""

    def __init__(self, num_classes: int = 2, in_chans: int = 3) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(in_chans, num_classes)

    def forward(self, images, targets=None):
        logits = self.fc(images.mean(dim=(2, 3)))
        if self.training and targets is not None:
            return {"cls_loss": torch.nn.functional.cross_entropy(logits, targets["labels"])}
        return {"logits": logits}


def test_a_model_whose_loss_never_moves_is_refused():
    """A loss identical at the last step and the first one is no evidence of optimization."""
    report = overfit_check(_DisconnectedParameter(), "classification", steps=12, num_classes=2,
                           seed=0)
    assert report["passed"] is False
    assert report["issue"] is not None
    assert "did not decrease" in report["issue"], report["issue"]
    assert len(report["losses"]) == 12
    assert report["final"] == report["initial"]


def test_a_model_that_really_learns_passes():
    """The rail admits valid work: a real learnable path drives the loss down and passes."""
    report = overfit_check(_LinearProbe(), "classification", steps=25, num_classes=2, seed=0)
    assert report["passed"], report["issue"]
    assert report["issue"] is None
    assert report["final"] < report["initial"]
