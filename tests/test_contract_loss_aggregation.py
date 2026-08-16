"""The objective the contract smokes and optimizes is every term of a returned loss dict.

A detector's train-mode forward returns several loss terms at once (classifier, box regression,
objectness, rpn). ``check_model_contract`` and ``overfit_check`` prove the model against their
sum, the same objective the trainer descends, so a later term that is broken, non-finite or left
unoptimized cannot pass behind a healthy first term.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from tcip_mcp.pipelines.model_contract import check_model_contract, overfit_check  # noqa: E402


class _ThreeTermLoss(torch.nn.Module):
    """Three loss terms of deliberately unequal magnitude, all on one learnable scale."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, images, targets=None):
        if self.training and targets is not None:
            return {"first": self.scale * 1.0,
                    "second": self.scale * 10.0,
                    "third": self.scale * 100.0}
        return {"logits": torch.zeros(2, 1)}


class _NonFiniteSecondTerm(torch.nn.Module):
    """A healthy first loss term beside a second one that has gone non-finite."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, images, targets=None):
        if self.training and targets is not None:
            return {"first": self.scale * 1.0, "second": torch.tensor(float("nan"))}
        return {"logits": torch.zeros(2, 1)}


class _UnoptimizedCompanionTerm(torch.nn.Module):
    """A first term the optimizer can drive down beside a companion term outside its reach.

    The companion grows twice as fast as the first term falls, so the objective as a whole gets
    worse over the run while the first term alone looks like healthy progress.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, images, targets=None):
        if self.training and targets is not None:
            primary = self.weight ** 2
            return {"primary": primary, "companion": 2.0 * (4.0 - primary.detach())}
        return {"logits": torch.zeros(2, 1)}


class _TwoOptimizableTerms(torch.nn.Module):
    """Two loss terms with different optima, both reachable by the optimizer."""

    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(3.0))
        self.b = torch.nn.Parameter(torch.tensor(-4.0))

    def forward(self, images, targets=None):
        if self.training and targets is not None:
            return {"a_term": (self.a - 1.0) ** 2, "b_term": 0.5 * (self.b + 1.0) ** 2}
        return {"logits": torch.zeros(2, 1)}


def test_train_loss_is_the_sum_of_every_returned_loss_term():
    """The reported train loss accounts for all three terms (1 + 10 + 100 at scale 1), not the
    first, the largest, or the last of them."""
    report = check_model_contract(_ThreeTermLoss(), "classification", num_classes=2)
    assert report["ok"], report["issues"]
    assert report["train_loss"] == pytest.approx(111.0)


def test_a_non_finite_later_loss_term_fails_the_smoke():
    """A finite first term never covers for a non-finite one behind it."""
    report = check_model_contract(_NonFiniteSecondTerm(), "classification", num_classes=2)
    assert report["ok"] is False
    assert any("not finite" in issue for issue in report["issues"]), report["issues"]


def test_overfit_check_judges_the_whole_objective_not_the_first_term():
    """A run where the first term improves while the objective as a whole worsens is refused."""
    model = _UnoptimizedCompanionTerm()
    report = overfit_check(model, "classification", steps=15, num_classes=2, seed=0)
    assert report["passed"] is False
    assert "did not decrease" in report["issue"], report["issue"]
    assert report["final"] > report["initial"]
    assert abs(float(model.weight.detach())) < 2.0


def test_a_multi_term_model_that_optimizes_every_term_passes():
    """The rail admits valid work: several terms are fine as long as their sum falls."""
    report = overfit_check(_TwoOptimizableTerms(), "classification", steps=25, num_classes=2,
                           seed=0)
    assert report["passed"], report["issue"]
    assert report["final"] < report["initial"]
    assert len(report["losses"]) == 25
