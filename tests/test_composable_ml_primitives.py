"""Tests for composable ML primitives.

Covers: datasets, optimizer factory, generic trainer basics,
predictor, active learning scorers, and tool imports.
"""

from __future__ import annotations

import pytest

from tcip_mcp.dataset_layout import UNDATED_BUCKET

torch = pytest.importorskip("torch")
import torch.nn as nn

from tests._producer_fixtures import dataset_over, run_over  # noqa: E402
from tests._training_values import adamw_optimizer, sgd_optimizer  # noqa: E402

# ====================================================================
# Data Layer & Datasets
# ====================================================================


class TestDatasets:
    def test_build_dataset_classification(self, tmp_path):
        """Classification dataset from its ground-truth table, one row per image."""
        from tests.tiny_trainer_fixtures import write_regression_dataset

        images_dir, csv_path = write_regression_dataset(
            tmp_path, [0.1, 0.3, 0.5, 0.7], [0, 1, 0, 1], height=32, width=32
        )

        ds, data = run_over("classification", str(images_dir), str(csv_path))
        assert data["num_classes"] == 2
        assert ds.num_samples == 4
        assert ds.task_type == "classification"

    def test_build_dataset_detection(self, tmp_path):
        """Detection dataset from per-image label documents."""
        from tcip_annotation.state import Annotation, BBox
        from tests._producer_fixtures import label_image

        imgs = tmp_path / "images" / UNDATED_BUCKET
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


def _built(block: dict, model):
    """The optimizer ``block`` (validated through ``schemas.OptimizerSpec``) builds over
    ``model`` at its own rates.
    """
    from tcip_mcp.pipelines.schemas import OptimizerSpec
    from tcip_mcp.pipelines.training.optimizer_factory import build_optimizer

    spec = OptimizerSpec.model_validate(block)
    return build_optimizer(
        spec, model, backbone_lr=spec.backbone_lr, head_lr=spec.head_lr
    )


class TestOptimizerFactory:
    def test_build_adamw(self):
        assert isinstance(
            _built(adamw_optimizer(), nn.Linear(10, 5)), torch.optim.AdamW
        )

    def test_build_sgd_at_its_stated_momentum(self):
        opt = _built(sgd_optimizer(), nn.Linear(10, 5))
        assert isinstance(opt, torch.optim.SGD)
        assert opt.param_groups[0]["momentum"] == sgd_optimizer()["momentum"]

    def test_a_builder_refuses_a_setting_it_does_not_read(self):
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

    def test_optimizer_builders_available(self):
        from tcip_mcp.pipelines.training.optimizer_factory import (
            _OPTIMIZER_BUILDERS,
        )

        assert "adamw" in _OPTIMIZER_BUILDERS
        assert "sgd" in _OPTIMIZER_BUILDERS

    def test_lamb_raises_when_torch_optimizer_missing(self):
        """Lamb must never silently fall back to AdamW when torch_optimizer isn't installed."""
        from importlib.util import find_spec

        if find_spec("torch_optimizer") is not None:
            pytest.skip("torch_optimizer is importable in this environment")
        with pytest.raises(ImportError, match="torch_optimizer"):
            _built({**adamw_optimizer(), "name": "lamb"}, nn.Linear(10, 5))


# ====================================================================
# Generic Predictor
# ====================================================================


# GenericPredictor no longer opens a path itself; a missing file is load_registered_checkpoint's
# own contract now, the one load every predictor construction goes through.
def test_load_registered_checkpoint_raises_filenotfounderror_on_a_missing_file(
    tmp_path,
):
    from tcip_mcp.model_registry import load_registered_checkpoint

    with pytest.raises(FileNotFoundError):
        load_registered_checkpoint(
            str(tmp_path / "nonexistent.pt"), project=tmp_path
        )


# ====================================================================
# Active Learning
# ====================================================================


class TestActiveLearningScoringLogic:
    def test_uncertainty_scorer_init(self):
        from tcip_mcp.pipelines.active_learning.scorer import UncertaintyScorer

        scorer = UncertaintyScorer(task="classification")
        assert scorer.task == "classification"

    def test_uncertainty_scorer_detection_init(self):
        from tcip_mcp.pipelines.active_learning.scorer import UncertaintyScorer

        scorer = UncertaintyScorer(task="detection")
        assert scorer.task == "detection"

    def test_combined_scorer_normalization(self):
        from tcip_mcp.pipelines.active_learning.scorer import CombinedScorer

        scorer = CombinedScorer(
            task="classification", uncertainty_weight=0.5, diversity_weight=0.5
        )
        assert scorer.uw == 0.5

    def test_review_queue(self):
        from tcip_mcp.pipelines.active_learning.selector import review_queue

        predictions = [
            {"image": "a.png", "scores": [0.95]},
            {"image": "b.png", "scores": [0.5]},
            {"image": "c.png", "scores": [0.2]},
        ]
        queue = review_queue(predictions, low=0.3, high=0.8)
        assert len(queue) == 1  # only b.png
