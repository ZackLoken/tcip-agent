"""The objective the contract smokes and optimizes is the sum of a returned loss dict's terms.

``check_model_contract`` reports the sum of three terms and fails a non-finite second term, and
``overfit_check`` refuses a run whose summed loss rises while its first term falls.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from tcip_mcp.pipelines.model_contract import (
    check_model_contract,
    overfit_check,
)  # noqa: E402

# What the smokes below synthesize their batch at, the shape a run resolves for itself.
CLS_DIMS = {"in_chans": 3, "num_classes": 2, "img_size": (64, 64)}


class _ThreeTermLoss(torch.nn.Module):
    """Three loss terms, one learnable scale times 1, 10 and 100, the scale starting at 1."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(
        self, images: torch.Tensor, targets: dict | None = None
    ) -> dict:
        if self.training and targets is not None:
            return {
                "first": self.scale * 1.0,
                "second": self.scale * 10.0,
                "third": self.scale * 100.0,
            }
        return {"logits": torch.zeros(2, 1)}


class _NonFiniteSecondTerm(torch.nn.Module):
    """A first loss term, a learnable scale starting at 1, beside a second that is NaN."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(
        self, images: torch.Tensor, targets: dict | None = None
    ) -> dict:
        if self.training and targets is not None:
            return {
                "first": self.scale * 1.0,
                "second": torch.tensor(float("nan")),
            }
        return {"logits": torch.zeros(2, 1)}


class _UnoptimizedCompanionTerm(torch.nn.Module):
    """A ``primary`` term carrying a gradient beside a ``companion`` term carrying none.

    In training, ``primary`` is the squared weight, starting at 2, and ``companion`` is
    ``2 * (4 - primary)`` over a detached copy of it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))

    def forward(
        self, images: torch.Tensor, targets: dict | None = None
    ) -> dict:
        if self.training and targets is not None:
            primary = self.weight**2
            return {
                "primary": primary,
                "companion": 2.0 * (4.0 - primary.detach()),
            }
        return {"logits": torch.zeros(2, 1)}


class _TwoOptimizableTerms(torch.nn.Module):
    """Two loss terms, ``(a - 1) ** 2`` and ``0.5 * (b + 1) ** 2``, from a at 3 and b at -4."""

    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(3.0))
        self.b = torch.nn.Parameter(torch.tensor(-4.0))

    def forward(
        self, images: torch.Tensor, targets: dict | None = None
    ) -> dict:
        if self.training and targets is not None:
            return {
                "a_term": (self.a - 1.0) ** 2,
                "b_term": 0.5 * (self.b + 1.0) ** 2,
            }
        return {"logits": torch.zeros(2, 1)}


def test_train_loss_is_the_sum_of_every_returned_loss_term() -> None:
    """The contract's reported train loss is the sum of every loss term the model returns.

    The three terms, 1, 10 and 100 at scale 1, report a train loss of 111.
    """
    report = check_model_contract(
        _ThreeTermLoss(), "classification", dims=CLS_DIMS
    )
    assert report["ok"], report["issues"]
    assert report["train_loss"] == pytest.approx(111.0)


def test_a_non_finite_later_loss_term_fails_the_smoke() -> None:
    """A non-finite second loss term fails the smoke despite a finite first term."""
    report = check_model_contract(
        _NonFiniteSecondTerm(), "classification", dims=CLS_DIMS
    )
    assert report["ok"] is False
    assert any("not finite" in issue for issue in report["issues"]), report[
        "issues"
    ]


def test_overfit_check_judges_the_whole_objective_not_the_first_term() -> None:
    """A run where the first term improves while the objective as a whole worsens is refused."""
    model = _UnoptimizedCompanionTerm()
    report = overfit_check(
        model,
        "classification",
        steps=15,
        dims=CLS_DIMS,
        seed=0,
        deterministic=False,
    )
    assert report["passed"] is False
    assert "did not decrease" in report["issue"], report["issue"]
    assert report["final"] > report["initial"]
    assert abs(float(model.weight.detach())) < 2.0


def test_a_multi_term_model_that_optimizes_every_term_passes() -> None:
    """A two-term model whose summed loss falls over 25 steps passes the overfit check."""
    report = overfit_check(
        _TwoOptimizableTerms(),
        "classification",
        steps=25,
        dims=CLS_DIMS,
        seed=0,
        deterministic=False,
    )
    assert report["passed"], report["issue"]
    assert report["final"] < report["initial"]
    assert len(report["losses"]) == 25
