"""The diversity scorer ranks candidates by their distance to the labeled images.

Its distance is scipy's cosine distance to the nearest labeled embedding; it refuses a model with
no backbone and a ranking with no labeled image to measure against. The combined ranking adds
the min-max scaled uncertainty and diversity terms at their weights.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.inference.generic_predictor import (
        GenericPredictor,
    )

pytest.importorskip("torch")
from tests._producer_fixtures import write_image  # noqa: E402
from tests.scorer_models import predictor_for  # noqa: E402


def test_diversity_scorer_no_backbone_raises(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.active_learning.scorer import DiversityScorer

    predictor = predictor_for(tmp_path, "build_linear", "classification")
    image = tmp_path / "whatever.png"
    with pytest.raises(RuntimeError, match="backbone"):
        DiversityScorer().score([image], [image], predictor)


def test_diversity_with_no_labeled_image_refuses_naming_the_finished_images(
    tmp_path: Path,
) -> None:
    """A diversity ranking handed no labeled image refuses, naming what makes one and another."""
    from tcip_mcp.pipelines.active_learning import ScorerSpec
    from tcip_mcp.pipelines.active_learning.scorer import resolve_scorer

    img = write_image(tmp_path / "a.png", (16, 16), (100, 100, 100))
    predictor = predictor_for(
        tmp_path, "build_conv_backbone", "classification"
    )
    for name in ("diversity", "combined"):
        block = ScorerSpec.model_validate(
            {"name": name}
            if name == "diversity"
            else {
                "name": name,
                "uncertainty_weight": 1.0,
                "diversity_weight": 1.0,
            }
        )
        with pytest.raises(ValueError, match="no image finished") as refused:
            resolve_scorer(block, "classification").score([img], [], predictor)
        assert "'uncertainty'" in str(refused.value)


def test_combined_adds_min_max_scaled_terms_a_constant_term_scaling_to_zeros(
    tmp_path: Path,
) -> None:
    """The combined ranking's known answer: a spread term scales to [0, 1], a constant one to 0."""
    from tcip_mcp.pipelines.active_learning.scorer import (
        BaseScorer,
        CombinedScorer,
    )
    from tcip_mcp.pipelines.image_utils import stem_of

    images: list[Path | BandGroupRef] = [
        tmp_path / f"{stem}.png" for stem in ("a", "b", "c")
    ]

    class _Table(BaseScorer):
        """Score each image from a fixed table, highest first."""

        def __init__(self, table: dict[str, float]) -> None:
            self.table = table

        def score(
            self,
            image_paths: list[Path | BandGroupRef],
            labeled: list[Path | BandGroupRef],
            predictor: GenericPredictor,
        ) -> list[tuple[Path | BandGroupRef, float]]:
            return sorted(
                ((p, self.table[stem_of(p)]) for p in image_paths),
                key=lambda pair: pair[1],
                reverse=True,
            )

    scorer = CombinedScorer(
        task="classification", uncertainty_weight=1.0, diversity_weight=2.0
    )
    scorer.unc = _Table({"a": 0.3, "b": 0.3, "c": 0.3})
    scorer.div = _Table({"a": 0.2, "b": 0.6, "c": 0.4})

    ranked = scorer.score(images, [], cast("GenericPredictor", None))

    assert [stem_of(p) for p, _ in ranked] == ["b", "c", "a"]
    assert [s for _, s in ranked] == pytest.approx([2.0, 1.0, 0.0])


def test_diversity_ranks_by_scipys_cosine_distance_to_the_nearest_labeled_image(
    tmp_path: Path,
) -> None:
    """The registry's diversity ranking agrees with ``cdist(..., "cosine")`` over the embeddings.

    The backbone hands its input on, so each image's embedding is its model input's band mean;
    the candidate colored like the labeled image ranks last.
    """
    import numpy as np
    from scipy.spatial.distance import cdist

    from tcip_mcp.pipelines.active_learning import ScorerSpec
    from tcip_mcp.pipelines.active_learning.scorer import resolve_scorer

    predictor = predictor_for(
        tmp_path, "build_pass_through_backbone", "classification"
    )
    labeled = [
        write_image(tmp_path / "labeled" / "red.png", (16, 16), (200, 30, 30)),
        write_image(tmp_path / "labeled" / "dark.png", (16, 16), (20, 20, 20)),
    ]
    candidates = [
        write_image(tmp_path / name, (16, 16), color)
        for name, color in (
            ("reddish.png", (190, 40, 40)),
            ("blue.png", (30, 30, 200)),
            ("green.png", (30, 200, 30)),
        )
    ]

    ranked = resolve_scorer(
        ScorerSpec(name="diversity"), "classification"
    ).score(candidates, labeled, predictor)

    def band_means(images: list[Path]) -> np.ndarray:
        return np.stack(
            [
                predictor.model_input(image)[0].mean(dim=(1, 2)).numpy()
                for image in images
            ]
        )

    nearest = cdist(band_means(candidates), band_means(labeled), "cosine").min(
        axis=1
    )
    expected = sorted(
        zip(candidates, nearest, strict=True), key=lambda pair: -pair[1]
    )
    assert [p for p, _ in ranked] == [p for p, _ in expected]
    assert [s for _, s in ranked] == pytest.approx(
        [float(s) for _, s in expected]
    )
    assert ranked[-1][0] == candidates[0]
