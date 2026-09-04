"""Fail-fast hygiene: narrowed polygon_iou, delivered flag, multi-head AL."""

import pytest


def test_polygon_iou_valid_and_degenerate():
    pytest.importorskip("shapely")
    from shapely.geometry import Polygon as SP

    from tcip_annotation.matching import polygon_iou
    a = SP([(0, 0), (10, 0), (10, 10), (0, 10)])
    assert polygon_iou(a, a.area, a, a.area) == pytest.approx(1.0)
    # a degenerate (zero-area) geometry must not crash -> 0.0
    bad = SP([(0, 0), (0, 0), (0, 0)])
    assert polygon_iou(a, a.area, bad, 0.0) == 0.0


def test_push_panel_event_reports_delivered_flag(tmp_path):
    from tests.test_canvas_liveview import _mint_binding

    from tcip_mcp.tools.gui_tools import push_panel_event

    # A matching binding, so the call reaches the HTTP push this asserts on rather than being
    # refused by the binding rail before it.
    _mint_binding(tmp_path)
    res = push_panel_event("review", "load_matches", {"x": 1}, project_root=str(tmp_path))
    # The delivery outcome is now an explicit bool: "backend down" can't read as success.
    assert "delivered" in res and isinstance(res["delivered"], bool)


def test_uncertainty_scorer_averages_over_heads(tmp_path):
    pytest.importorskip("torch")
    import torch
    from PIL import Image

    from tcip_mcp.pipelines.active_learning.scorer import UncertaintyScorer, _entropy

    img = tmp_path / "a.png"
    Image.new("RGB", (16, 16)).save(img)

    class MultiHead(torch.nn.Module):
        def forward(self, x):
            return {"head0_logits": torch.tensor([[2.0, 0.0]]),
                    "head1_logits": torch.tensor([[0.0, 0.0]])}

    scored = UncertaintyScorer(task="classification").score(
        [str(img)], MultiHead(), torch.device("cpu"))
    expected = (_entropy(torch.tensor([[2.0, 0.0]])) + _entropy(torch.tensor([[0.0, 0.0]]))) / 2
    assert scored[0][1] == pytest.approx(expected)  # averaged across both heads, not first-only
