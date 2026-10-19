"""The uncertainty scorer gives an empty-detection frame 0.0 and a low-confidence one more."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from tcip_mcp.pipelines.active_learning.scorer import UncertaintyScorer  # noqa: E402
from tests.scorer_models import predictor_for  # noqa: E402


def test_empty_frame_ranks_low_not_max(tmp_path: Path) -> None:
    for name in ("a", "b"):
        Image.new("RGB", (64, 64), (100, 100, 100)).save(
            tmp_path / f"{name}.png"
        )
    paths = [tmp_path / "a.png", tmp_path / "b.png"]

    predictor = predictor_for(tmp_path, "build_empty_then_low", "detection")
    ranked = UncertaintyScorer(task="detection").score(paths, predictor)
    by_path = dict(ranked)

    assert (
        by_path[paths[0]] == 0.0
    )  # the empty frame: no ambiguous decision -> does not flood
    assert (
        by_path[paths[1]] > 0.0
    )  # the low-confidence detection is genuinely uncertain
    assert (
        ranked[0][0] == paths[1]
    )  # the detection frame outranks the empty one


def test_a_detection_output_with_no_scores_refuses_rather_than_ranking_as_certain(
    tmp_path: Path,
) -> None:
    Image.new("RGB", (64, 64), (100, 100, 100)).save(tmp_path / "a.png")

    predictor = predictor_for(tmp_path, "build_scoreless", "detection")
    with pytest.raises(KeyError, match="scores"):
        UncertaintyScorer(task="detection").score(
            [tmp_path / "a.png"], predictor
        )
