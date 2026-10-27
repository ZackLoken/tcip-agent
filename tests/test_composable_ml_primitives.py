"""Tests for composable ML primitives.

Covers: dataset construction, the optimizer factory, a registered-checkpoint load and the
active-learning scorers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir, image_root

if TYPE_CHECKING:
    from pathlib import Path

    from torch.optim import Optimizer

    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor

torch = pytest.importorskip("torch")
import torch.nn as nn

from tests._producer_fixtures import dataset_over, run_over  # noqa: E402
from tests._training_values import adamw_optimizer, sgd_optimizer  # noqa: E402

# ====================================================================
# Data Layer & Datasets
# ====================================================================


class TestDatasets:
    def test_build_dataset_classification(self, tmp_path: Path) -> None:
        """A classification dataset over a four-row table holds four samples of two classes."""
        from tests.tiny_trainer_fixtures import write_regression_dataset

        images_dir, csv_path = write_regression_dataset(
            tmp_path, [0.1, 0.3, 0.5, 0.7], [0, 1, 0, 1], height=32, width=32
        )

        ds, data = run_over("classification", str(images_dir), str(csv_path))
        assert data["num_classes"] == 2
        assert ds.num_samples == 4
        assert ds.task_type == "classification"

    def test_build_dataset_detection(self, tmp_path: Path) -> None:
        """A detection dataset over one labeled image holds one sample."""
        from tcip_annotation.state import Annotation, BBox
        from tests._producer_fixtures import label_image

        imgs = image_dir(tmp_path, UNDATED_BUCKET)
        imgs.mkdir(parents=True)
        # Create one image + label
        img = torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8)
        from torchvision.utils import save_image

        save_image(img.float() / 255.0, str(imgs / "test.png"))
        label_image(
            imgs / "test.png",
            [
                Annotation(
                    subject="bud", geometry=BBox(25.6, 22.4, 38.4, 41.6)
                ),
                Annotation(
                    subject="bud", geometry=BBox(16.0, 16.0, 22.4, 22.4)
                ),
            ],
            64,
            64,
            keep_empty=True,
        )

        ds = dataset_over("detection", str(imgs), subject="bud")
        assert ds.task_type == "detection"
        assert len(ds) == 1


# ====================================================================
# Optimizer Factory & Trainer Config
# ====================================================================


def _built(block: dict, model: nn.Module) -> Optimizer:
    """Build the optimizer ``block`` states over ``model`` at its own rates and return it.

    The block is validated through ``schemas.OptimizerSpec``.
    """
    from tcip_mcp.pipelines.schemas import OptimizerSpec
    from tcip_mcp.pipelines.training.optimizer_factory import build_optimizer

    spec = OptimizerSpec.model_validate(block)
    return build_optimizer(
        spec, model, backbone_lr=spec.backbone_lr, head_lr=spec.head_lr
    )


class TestOptimizerFactory:
    def test_build_adamw(self) -> None:
        assert isinstance(
            _built(adamw_optimizer(backbone_lr=1e-3), nn.Linear(10, 5)),
            torch.optim.AdamW,
        )

    def test_build_sgd_at_its_stated_momentum(self) -> None:
        opt = _built(sgd_optimizer(backbone_lr=1e-2), nn.Linear(10, 5))
        assert isinstance(opt, torch.optim.SGD)
        assert opt.param_groups[0]["momentum"] == sgd_optimizer()["momentum"]

    def test_a_builder_refuses_a_setting_it_does_not_read(self) -> None:
        from tcip_mcp.pipelines.training.optimizer_factory import (
            _OPTIMIZER_BUILDERS,
        )

        with pytest.raises(TypeError, match="momentum"):
            _OPTIMIZER_BUILDERS["adamw"](
                nn.Linear(10, 5).parameters(),
                lr=1e-3,
                weight_decay=0.0,
                momentum=0.8,
            )

    def test_optimizer_builders_available(self) -> None:
        from tcip_mcp.pipelines.training.optimizer_factory import (
            _OPTIMIZER_BUILDERS,
        )

        assert "adamw" in _OPTIMIZER_BUILDERS
        assert "sgd" in _OPTIMIZER_BUILDERS

    def test_lamb_raises_when_torch_optimizer_missing(self) -> None:
        """Building Lamb without torch_optimizer installed raises ImportError naming it."""
        from importlib.util import find_spec

        if find_spec("torch_optimizer") is not None:
            pytest.skip("torch_optimizer is importable in this environment")
        with pytest.raises(ImportError, match="torch_optimizer"):
            _built(
                adamw_optimizer(name="lamb", backbone_lr=1e-3),
                nn.Linear(10, 5),
            )


# ====================================================================
# Generic Predictor
# ====================================================================


# GenericPredictor no longer opens a path itself; a missing file is load_registered_checkpoint's
# own contract now, the one load every predictor construction goes through.
def test_load_registered_checkpoint_raises_filenotfounderror_on_a_missing_file(
    tmp_path: Path,
) -> None:
    from tcip_mcp.model_registry import load_registered_checkpoint

    with pytest.raises(FileNotFoundError):
        load_registered_checkpoint(
            str(tmp_path / "nonexistent.pt"), project=tmp_path
        )


# ====================================================================
# Active Learning
# ====================================================================


class TestActiveLearningScoringLogic:
    def test_uncertainty_scorer_init(self) -> None:
        from tcip_mcp.pipelines.active_learning.scorer import UncertaintyScorer

        scorer = UncertaintyScorer(task="classification")
        assert scorer.task == "classification"

    def test_uncertainty_scorer_detection_init(self) -> None:
        from tcip_mcp.pipelines.active_learning.scorer import UncertaintyScorer

        scorer = UncertaintyScorer(task="detection")
        assert scorer.task == "detection"

    def test_combined_scorer_normalization(self) -> None:
        from tcip_mcp.pipelines.active_learning.scorer import CombinedScorer

        scorer = CombinedScorer(
            task="classification", uncertainty_weight=0.5, diversity_weight=0.5
        )
        assert scorer.uw == 0.5

    def test_combined_scorer_ranks_a_grouped_capture_beside_a_plain_image(
        self, tmp_path: Path
    ) -> None:
        """The combined ranking joins its two scorers' answers over a band group and a raster."""
        import numpy as np
        import tifffile

        from tcip_mcp.pipelines.active_learning.scorer import (
            BaseScorer,
            CombinedScorer,
        )
        from tcip_mcp.pipelines.image_utils import list_logical_images, stem_of
        from tests._producer_fixtures import write_band_group

        images = image_root(tmp_path)
        images.mkdir()
        bands = {name: images / f"cap_{name}.tif" for name in ("G", "R")}
        for path in [*bands.values(), images / "plain.tif"]:
            tifffile.imwrite(str(path), np.zeros((8, 8), dtype=np.uint16))
        write_band_group(images, "cap", bands, central_wavelength_nm=None)
        candidates = list(list_logical_images(images).values())

        class _Fixed(BaseScorer):
            """Score each candidate by its stem from a fixed table."""

            def score(
                self,
                image_paths: list[Path | BandGroupRef],
                predictor: GenericPredictor | None,
            ) -> list[tuple[Path | BandGroupRef, float]]:
                """Return each candidate at its stem's table score, descending."""
                by_stem = {"cap": 1.0, "plain": 0.0}
                return sorted(
                    ((p, by_stem[stem_of(p)]) for p in image_paths),
                    key=lambda pair: pair[1],
                    reverse=True,
                )

        scorer = CombinedScorer(
            task="classification", uncertainty_weight=1.0, diversity_weight=1.0
        )
        scorer.unc = _Fixed()
        scorer.div = _Fixed()
        ranked = scorer.score(candidates, predictor=None)
        assert [(stem_of(p), s) for p, s in ranked] == [
            ("cap", 2.0),
            ("plain", 0.0),
        ]

    def test_review_queue(self) -> None:
        from tcip_mcp.pipelines.active_learning.selector import review_queue

        predictions = [
            {"image": "a.png", "scores": [0.95]},
            {"image": "b.png", "scores": [0.5]},
            {"image": "c.png", "scores": [0.2]},
        ]
        queue = review_queue(predictions, low=0.3, high=0.8)
        assert len(queue) == 1  # only b.png
