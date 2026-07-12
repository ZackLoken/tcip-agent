"""Tests for composable ML primitives — Phases 5-12.

Covers: datasets, samplers, optimizer factory, generic trainer basics,
orchestrator validation, predictor, active learning scorers,
PointNet++ backbone, temporal heads, and pipeline tools.
"""

from __future__ import annotations

import pytest
torch = pytest.importorskip("torch")
import torch.nn as nn


# ====================================================================
# Phase 5: Data Layer — Datasets & Samplers
# ====================================================================

class TestDatasets:
    def test_build_dataset_classification(self, tmp_path):
        """Classification dataset from folder structure."""
        from tcip_mcp.pipelines.data.datasets import build_dataset
        # Create minimal folder-based classification data
        for cls in ("a", "b"):
            d = tmp_path / cls
            d.mkdir()
            # Create 2 tiny PNG files per class
            for i in range(2):
                img = torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)
                from torchvision.utils import save_image
                save_image(img.float() / 255.0, str(d / f"{i}.png"))

        ds = build_dataset("classification", images_dir=str(tmp_path))
        assert ds.num_classes == 2
        assert ds.num_samples == 4
        assert ds.task_type == "classification"

    def test_build_dataset_detection(self, tmp_path):
        """Detection dataset from YOLO labels."""
        from tcip_mcp.pipelines.data.datasets import build_dataset
        imgs = tmp_path / "images"
        lbls = tmp_path / "labels"
        imgs.mkdir()
        lbls.mkdir()
        # Create one image + label
        img = torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8)
        from torchvision.utils import save_image
        save_image(img.float() / 255.0, str(imgs / "test.png"))
        (lbls / "test.txt").write_text("0 0.5 0.5 0.2 0.3\n1 0.3 0.3 0.1 0.1\n")

        ds = build_dataset("detection", images_dir=str(imgs), labels_dir=str(lbls))
        assert ds.task_type == "detection"
        assert len(ds) == 1


class TestSamplers:
    def test_build_sampler_random(self):
        from tcip_mcp.pipelines.data.samplers import build_sampler
        sampler = build_sampler("random", None)
        assert sampler is None

    def test_weighted_random_sampler(self):
        from tcip_mcp.pipelines.data.samplers import build_sampler

        class FakeDataset:
            task_type = "classification"
            num_classes = 3
            num_samples = 6
            class_distribution = {0: 3, 1: 2, 2: 1}
            def __len__(self):
                return self.num_samples
            def __getitem__(self, idx):
                labels = [0, 0, 0, 1, 1, 2]
                return torch.zeros(3, 32, 32), {"label": labels[idx]}

        ds = FakeDataset()
        sampler = build_sampler("weighted_random", ds)
        assert sampler is not None


# ====================================================================
# Phase 6: Optimizer Factory & Trainer Config
# ====================================================================

class TestOptimizerFactory:
    def test_build_adamw(self):
        from tcip_mcp.pipelines.training.optimizer_factory import build_optimizer
        model = nn.Linear(10, 5)
        opt = build_optimizer("adamw", model, head_lr=1e-3)
        assert isinstance(opt, torch.optim.AdamW)

    def test_build_sgd(self):
        from tcip_mcp.pipelines.training.optimizer_factory import build_optimizer
        model = nn.Linear(10, 5)
        opt = build_optimizer("sgd", model, head_lr=1e-2)
        assert isinstance(opt, torch.optim.SGD)

    def test_registry_listing(self):
        from tcip_mcp.pipelines.registry import OPTIMIZERS
        import tcip_mcp.pipelines.training.optimizer_factory  # noqa: F401
        entries = OPTIMIZERS.list()
        names = [e["name"] for e in entries]
        assert "adamw" in names
        assert "sgd" in names


class TestTrainConfig:
    def test_dataclass_defaults(self):
        from tcip_mcp.pipelines.training.generic_trainer import TrainConfig
        cfg = TrainConfig(model_spec={"backbone": "resnet50"}, dataset={"task": "classification"})
        assert cfg.mixed_precision is True
        assert cfg.batch_size == 4
        assert len(cfg.stages) >= 2


# ====================================================================
# Phase 7: Pipeline Orchestrator Validation
# ====================================================================

class TestOrchestrator:
    def test_validate_simple_pipeline(self):
        from tcip_mcp.pipelines.orchestrator import validate_pipeline
        spec = {
            "phases": [
                {
                    "name": "train",
                    "type": "training",
                    "model_spec": {"backbone": "resnet50", "neck": "fpn", "heads": [{"name": "classification", "task": "classification", "num_classes": 5}]},
                    "dataset": {"task": "classification"},
                },
                {
                    "name": "infer",
                    "type": "inference",
                    "checkpoint": "$train.checkpoint",
                    "images_dir": "/some/dir",
                },
            ]
        }
        issues = validate_pipeline(spec)
        assert isinstance(issues, list)

    def test_empty_pipeline_fails(self):
        from tcip_mcp.pipelines.orchestrator import validate_pipeline
        spec = {"phases": []}
        issues = validate_pipeline(spec)
        assert any("phase" in i.lower() or "empty" in i.lower() for i in issues)


# ====================================================================
# Phase 8: Generic Predictor
# ====================================================================

class TestGenericPredictor:
    def test_predictor_missing_checkpoint(self):
        from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
        with pytest.raises(FileNotFoundError):
            GenericPredictor("/nonexistent/model.pt")


# ====================================================================
# Phase 9: Active Learning
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
        scorer = CombinedScorer(task="classification", uncertainty_weight=0.5, diversity_weight=0.5)
        assert scorer.uw == 0.5

    def test_auto_accept(self):
        from tcip_mcp.pipelines.active_learning.selector import auto_accept
        predictions = [
            {"image": "a.png", "scores": [0.95, 0.9]},
            {"image": "b.png", "scores": [0.3]},
            {"image": "c.png", "scores": [0.85, 0.82]},
        ]
        accepted = auto_accept(predictions, threshold=0.8)
        assert len(accepted) == 2

    def test_review_queue(self):
        from tcip_mcp.pipelines.active_learning.selector import review_queue
        predictions = [
            {"image": "a.png", "scores": [0.95]},
            {"image": "b.png", "scores": [0.5]},
            {"image": "c.png", "scores": [0.2]},
        ]
        queue = review_queue(predictions, low=0.3, high=0.8)
        assert len(queue) == 1  # only b.png


# ====================================================================
# Phase 10: 3D Point Cloud
# ====================================================================

class TestPointCloud:
    def test_pointnet_backbone_forward(self):
        from tcip_mcp.pipelines.components.backbones_3d import PointNetPPBackbone
        model = PointNetPPBackbone(in_channels=0)
        # [B, N, 3] input — xyz only, no extra features
        pts = torch.randn(2, 256, 3)
        out = model(pts)
        assert isinstance(out, dict)
        assert "sa3" in out
        assert out["sa3"].shape[0] == 2

    def test_pointnet_not_registered(self):
        # Phase 0.3 task-honesty: PointNet++/3D is deferred and intentionally NOT
        # registered (no point-cloud dataset/task/inference path exists yet). The
        # backbone class still works in isolation (see test_pointnet_backbone_forward).
        from tcip_mcp.pipelines.registry import BACKBONES
        import tcip_mcp.pipelines.components.backbones_3d  # noqa: F401
        assert "pointnet++" not in BACKBONES


# ====================================================================
# Phase 11: Temporal Modeling
# ====================================================================

class TestTemporal:
    def test_lstm_head_forward(self):
        from tcip_mcp.pipelines.components.temporal import TemporalLSTMHead
        head = TemporalLSTMHead(in_channels=128, num_milestones=4)
        x = torch.randn(2, 6, 128)  # [B, T, C]
        out = head(x)
        assert out.shape == (2, 4)

    def test_transformer_head_forward(self):
        from tcip_mcp.pipelines.components.temporal import TemporalTransformerHead
        head = TemporalTransformerHead(in_channels=128, num_milestones=4)
        x = torch.randn(2, 6, 128)
        out = head(x)
        assert out.shape == (2, 4)

    def test_temporal_heads_registered(self):
        from tcip_mcp.pipelines.registry import HEADS
        import tcip_mcp.pipelines.components.temporal  # noqa: F401
        assert "temporal_lstm" in HEADS
        assert "temporal_transformer" in HEADS


# ====================================================================
# Phase 12: Pipeline & Active Learning Tools (import-level only)
# ====================================================================

class TestToolImports:
    def test_pipeline_tools_importable(self):
        import tcip_mcp.tools.pipeline_tools  # noqa: F401

    def test_active_learning_tools_importable(self):
        import tcip_mcp.tools.active_learning_tools  # noqa: F401

    def test_model_tools_no_old_builder_import(self):
        """model_tools should not import from pipelines.models.builder."""
        import inspect
        import tcip_mcp.tools.model_tools as mt
        src = inspect.getsource(mt)
        assert "pipelines.models.builder" not in src

    def test_inference_tools_uses_build_predictor(self):
        """inference_tools should build its predictor through the model-kind factory."""
        import inspect
        import tcip_mcp.tools.inference_tools as it
        src = inspect.getsource(it)
        assert "build_predictor" in src
