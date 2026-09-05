"""Tests for composable ML primitives.

Covers: datasets, samplers, optimizer factory, generic trainer basics,
predictor, active learning scorers, and tool imports.
"""

from __future__ import annotations

import pytest
torch = pytest.importorskip("torch")
import torch.nn as nn


# ====================================================================
# Data Layer, Datasets & Samplers
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
        """Detection dataset from per-image JSON labels."""
        from tcip_annotation import json_io
        from tcip_annotation.state import Annotation, BBox
        from tcip_mcp.pipelines.data.datasets import build_dataset
        imgs = tmp_path / "images"
        lbls = tmp_path / "labels"
        imgs.mkdir()
        lbls.mkdir()
        # Create one image + label
        img = torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8)
        from torchvision.utils import save_image
        save_image(img.float() / 255.0, str(imgs / "test.png"))
        json_io.write_annotations(str(lbls / "test.json"),
                                  [Annotation(subject="bud", geometry=BBox(25.6, 22.4, 38.4, 41.6)),
                                   Annotation(subject="bud", geometry=BBox(16.0, 16.0, 22.4, 22.4))],
                                  64, 64, keep_empty=True)

        ds = build_dataset("detection", images_dir=str(imgs), labels_dir=str(lbls), subject="bud")
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
# Optimizer Factory & Trainer Config
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

    def test_optimizer_builders_available(self):
        from tcip_mcp.pipelines.training.optimizer_factory import _OPTIMIZER_BUILDERS
        assert "adamw" in _OPTIMIZER_BUILDERS
        assert "sgd" in _OPTIMIZER_BUILDERS

    def test_lamb_raises_when_torch_optimizer_missing(self):
        """lamb must never silently fall back to AdamW when torch_optimizer isn't installed."""
        from importlib.util import find_spec
        from tcip_mcp.pipelines.training.optimizer_factory import build_optimizer

        if find_spec("torch_optimizer") is not None:
            pytest.skip("torch_optimizer is importable in this environment")
        model = nn.Linear(10, 5)
        with pytest.raises(ImportError, match="torch_optimizer"):
            build_optimizer("lamb", model, head_lr=1e-3)


class TestTrainConfig:
    def test_dataclass_defaults(self):
        from tcip_mcp.pipelines.training.generic_trainer import TrainConfig
        cfg = TrainConfig(model_source={"builder": "x:y"}, dataset={"task": "classification"})
        assert cfg.batch_size == 4
        assert cfg.num_workers == 2

    def test_dead_schedule_fields_removed(self):
        # stages/optimizer/early_stopping/scheduler and other never-read-post-construction knobs
        # (including mixed_precision/seed/deterministic, unread duplicates of values already
        # threaded via run.config) are gone: train() reads run.config directly, never a
        # TrainConfig instance, except for .sampler/.batch_size/.num_workers.
        import dataclasses
        from tcip_mcp.pipelines.training.generic_trainer import TrainConfig
        field_names = {f.name for f in dataclasses.fields(TrainConfig)}
        assert not field_names & {
            "stages", "optimizer", "early_stopping", "scheduler",
            "evaluation", "stage_warmup_epochs", "lr_scaling",
            "enforce_monotonic_unfreeze", "checkpoint_every_n_epochs",
            "gradient_accumulation_steps", "mixed_precision", "seed", "deterministic",
        }


# ====================================================================
# Generic Predictor
# ====================================================================

# GenericPredictor no longer opens a path itself; a missing file is load_registered_checkpoint's
# own contract now, the one load every predictor construction goes through.
def test_load_registered_checkpoint_raises_filenotfounderror_on_a_missing_file(tmp_path):
    from tcip_mcp.model_registry import load_registered_checkpoint
    with pytest.raises(FileNotFoundError):
        load_registered_checkpoint(str(tmp_path / "nonexistent.pt"), project_path=str(tmp_path))


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
# Temporal Modeling (removed)
# ====================================================================
# components/temporal.py was deleted: it declared task_type = "temporal", which
# _DATASET_MAP does not route, so a model built on it could not be trained,
# evaluated, or measured. Its milestone convention was also a trait semantic
# frozen as a constructor default. Sequence modeling remains available; the
# agent writes it in its own model_source builder, where the milestone
# definition comes from the trait's TraitSpec.


def test_temporal_component_module_is_gone():
    """The unroutable task_type must not come back as importable scaffolding."""
    import importlib
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("tcip_mcp.pipelines.components.temporal")


# ====================================================================
# Pipeline & Active Learning Tools (import-level only)
# ====================================================================

class TestToolImports:
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
