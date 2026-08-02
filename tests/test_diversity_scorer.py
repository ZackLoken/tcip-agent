"""DiversityScorer: no silent random-noise, no silent no-op."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402


def test_diversity_scorer_no_backbone_raises():
    from tcip_mcp.pipelines.active_learning.scorer import DiversityScorer
    with pytest.raises(RuntimeError, match="backbone"):
        DiversityScorer().score(["whatever.png"], torch.nn.Linear(3, 3), torch.device("cpu"))


def test_diversity_scorer_no_labeled_warns_and_returns_uniform(tmp_path, caplog):
    from tcip_mcp.pipelines.active_learning.scorer import DiversityScorer

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Conv2d(3, 8, 3, padding=1)

    img = tmp_path / "a.png"
    Image.new("RGB", (16, 16), (100, 100, 100)).save(img)
    with caplog.at_level("WARNING"):
        scored = DiversityScorer().score([str(img)], M(), torch.device("cpu"))
    assert scored == [(str(img), 1.0)]
    assert any("no labeled embeddings" in r.message.lower() for r in caplog.records)
