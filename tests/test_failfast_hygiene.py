"""Fail-fast hygiene: IoU over a degenerate box, an untaken panel event's reason, multi-head AL."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from tcip_mcp.workspace import BoundProject


def test_iou_matrix_valid_and_degenerate() -> None:
    from tcip_annotation.matching import iou_matrix

    a = [[0, 0, 10, 10]]
    assert iou_matrix(a, a)[0, 0] == pytest.approx(1.0)
    # a degenerate (zero-area) box must not crash or divide by zero -> 0.0
    assert iou_matrix([[0, 0, 0, 0]], [[0, 0, 0, 0]])[0, 0] == 0.0


def test_push_panel_event_reports_an_undelivered_event_as_not_ok(
    bound: BoundProject,
) -> None:
    from tcip_mcp.tools.gui_tools import push_panel_event
    from tcip_mcp.web_client import Banner

    res = push_panel_event(bound, "annotate", Banner(text="load the labels"))
    assert isinstance(res.reason, str) and "pytest" in res.reason


def test_uncertainty_scorer_averages_over_heads(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    import torch

    from tcip_mcp.pipelines.active_learning.scorer import (
        UncertaintyScorer,
        _entropy,
    )
    from tests._producer_fixtures import write_image

    img = write_image(tmp_path / "a.png", (16, 16))

    from tests.scorer_models import predictor_for

    predictor = predictor_for(tmp_path, "build_two_head", "classification")
    scored = UncertaintyScorer(task="classification").score([img], predictor)
    expected = (
        _entropy(torch.tensor([[2.0, 0.0]]))
        + _entropy(torch.tensor([[0.0, 0.0]]))
    ) / 2
    assert scored[0][1] == pytest.approx(
        expected
    )  # averaged across both heads, not first-only
