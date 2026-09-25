"""DiversityScorer: no silent random-noise, no silent no-op."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from tests.scorer_models import predictor_for  # noqa: E402


def test_diversity_scorer_no_backbone_raises(tmp_path):
    from tcip_mcp.pipelines.active_learning.scorer import DiversityScorer
    predictor = predictor_for(tmp_path, "build_linear", "classification")
    with pytest.raises(RuntimeError, match="backbone"):
        DiversityScorer().score(["whatever.png"], predictor)


def test_diversity_scorer_no_labeled_warns_and_returns_uniform(tmp_path, caplog):
    from tcip_mcp.pipelines.active_learning.scorer import DiversityScorer

    img = tmp_path / "a.png"
    Image.new("RGB", (16, 16), (100, 100, 100)).save(img)
    predictor = predictor_for(tmp_path, "build_conv_backbone", "classification")
    with caplog.at_level("WARNING"):
        scored = DiversityScorer().score([str(img)], predictor)
    assert scored == [(str(img), 1.0)]
    assert any("no labeled embeddings" in r.message.lower() for r in caplog.records)
