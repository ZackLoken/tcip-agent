"""DiversityScorer refuses a model with no backbone and warns when it has no labeled embeddings.

With no labeled embeddings, it scores the one image it is given 1.0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("torch")
from tests._producer_fixtures import write_image  # noqa: E402
from tests.scorer_models import predictor_for  # noqa: E402


def test_diversity_scorer_no_backbone_raises(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.active_learning.scorer import DiversityScorer

    predictor = predictor_for(tmp_path, "build_linear", "classification")
    with pytest.raises(RuntimeError, match="backbone"):
        DiversityScorer().score([tmp_path / "whatever.png"], predictor)


def test_diversity_scorer_no_labeled_warns_and_returns_uniform(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from tcip_mcp.pipelines.active_learning.scorer import DiversityScorer

    img = write_image(tmp_path / "a.png", (16, 16), (100, 100, 100))
    predictor = predictor_for(
        tmp_path, "build_conv_backbone", "classification"
    )
    with caplog.at_level("WARNING"):
        scored = DiversityScorer().score([img], predictor)
    assert scored == [(img, 1.0)]
    assert any(
        "no labeled embeddings" in r.message.lower() for r in caplog.records
    )
