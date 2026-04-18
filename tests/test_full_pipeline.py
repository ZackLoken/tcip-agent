"""End-to-end integration test: compose → train → infer → export.

Proves the full composable ML pipeline works as a connected system.
Uses synthetic data for classification and real sample data for detection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader
torchvision = pytest.importorskip("torchvision")
from torchvision.utils import save_image

# Trigger component registration (side-effect imports)
import tcip_mcp.pipelines.components.backbones  # noqa: F401
import tcip_mcp.pipelines.components.necks  # noqa: F401
import tcip_mcp.pipelines.components.heads  # noqa: F401
import tcip_mcp.pipelines.components.losses  # noqa: F401


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_classification_data(tmp_path):
    """Create a minimal 2-class image classification dataset."""
    images_dir = tmp_path / "cls_images"
    for cls_name, cls_idx in [("healthy", 0), ("diseased", 1)]:
        cls_dir = images_dir / cls_name
        cls_dir.mkdir(parents=True)
        for i in range(6):
            if cls_idx == 0:
                img = torch.rand(3, 64, 64) * 0.3  # dark-ish
            else:
                img = torch.rand(3, 64, 64) * 0.3 + 0.7  # bright-ish
            save_image(img, str(cls_dir / f"{i:03d}.png"))
    return str(images_dir)


@pytest.fixture()
def output_dir(tmp_path):
    return str(tmp_path / "run_output")


# ---------------------------------------------------------------------------
# Test: full classification pipeline
# ---------------------------------------------------------------------------

class TestFullClassificationPipeline:
    """End-to-end: recommend → compose → train → checkpoint → predict → CSV."""

    def test_recommend_compose_train_infer_export(self, tiny_classification_data, output_dir):
        # --- Step 1: Agent recommends a model spec ---
        from tcip_mcp.pipelines.composer import recommend_model_spec, compose_model

        spec = recommend_model_spec(
            task="classification",
            dataset_size=12,  # tiny dataset → small backbone
            num_classes=2,
        )
        assert "backbone" in spec
        assert "heads" in spec
        assert spec["heads"][0]["name"] == "classification"

        # --- Step 2: Compose model and verify it runs ---
        model = compose_model(spec)
        dummy = torch.randn(2, 3, 64, 64)
        model.eval()
        with torch.no_grad():
            out = model(dummy)
        assert isinstance(out, dict)
        # Should have head0 predictions
        pred_keys = [k for k in out if k.startswith("head0")]
        assert len(pred_keys) > 0

        # --- Step 3: Build dataset + dataloader ---
        from tcip_mcp.pipelines.data.datasets import build_dataset
        from tcip_mcp.pipelines.training.generic_trainer import task_collate

        dataset = build_dataset("classification", images_dir=tiny_classification_data)
        assert dataset.num_classes == 2
        assert dataset.num_samples == 12

        loader = DataLoader(
            dataset, batch_size=4, shuffle=True,
            collate_fn=task_collate("classification"),
        )

        # Verify one batch works
        batch_images, batch_targets = next(iter(loader))
        assert batch_images.shape[0] == 4
        assert "labels" in batch_targets

        # --- Step 4: Create run and train 2 epochs ---
        from tcip_mcp.pipelines.training.generic_trainer import create_run, train

        config = {
            "model_spec": spec,
            "device": "cpu",
            "stages": [{"freeze_to": -1, "epochs": 2}],
            "mixed_precision": False,
            "optimizer": {"name": "adamw", "backbone_lr": 1e-4, "head_lr": 1e-3, "weight_decay": 0},
            "scheduler": {"type": "cosine"},
            "early_stopping": {"enabled": False},
            "gradient_accumulation_steps": 1,
            "checkpoint_every_n_epochs": 1,
        }
        run = create_run(config, output_dir)

        completed_run = train(run, loader, val_loader=None, task="classification")

        assert completed_run.status == "completed"
        assert completed_run.current_epoch == 2
        assert len(completed_run.metrics_history) == 2

        # Verify output files exist
        out = Path(output_dir)
        assert (out / "model_best.pt").is_file()
        assert (out / "model_final.pt").is_file()
        assert (out / "metrics.jsonl").is_file()

        # Verify metrics JSONL is valid
        with open(out / "metrics.jsonl") as f:
            lines = f.readlines()
        assert len(lines) == 2
        m = json.loads(lines[0])
        assert "epoch" in m
        assert "train_loss" in m

        # Verify checkpoint has required keys
        ckpt = torch.load(out / "model_best.pt", map_location="cpu", weights_only=False)
        assert "model_state_dict" in ckpt
        assert "model_spec" in ckpt

        # --- Step 5: Load checkpoint and run inference ---
        from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor

        predictor = GenericPredictor(
            str(out / "model_best.pt"), device="cpu", score_threshold=0.1,
        )

        # Pick some test images
        test_images = sorted(Path(tiny_classification_data).rglob("*.png"))[:4]
        results = predictor.predict_batch([str(p) for p in test_images])

        assert len(results) == 4
        for r in results:
            assert "image" in r

        # --- Step 6: Export CSV ---
        from tcip_mcp.pipelines.postprocessing.export import export_detection_csv

        csv_path = str(out / "results.csv")
        # Adapt classification results to detection CSV format
        csv_results = []
        for r in results:
            preds = r.get("head0_labels", [])
            confs = r.get("head0_confidences", [])
            csv_results.append({
                "image": r["image"],
                "count": len(preds) if isinstance(preds, list) else 1,
                "scores": confs if isinstance(confs, list) else [confs] if confs else [],
            })
        export_detection_csv(csv_results, csv_path)

        assert Path(csv_path).is_file()
        content = Path(csv_path).read_text()
        assert "image" in content
        assert "detection_count" in content
        lines = content.strip().splitlines()
        assert len(lines) == 5  # header + 4 images


class TestFullPipelineWithValidation:
    """Train with train/val split and verify early stopping wiring."""

    def test_train_val_split(self, tiny_classification_data, output_dir):
        from tcip_mcp.pipelines.composer import recommend_model_spec
        from tcip_mcp.pipelines.data.datasets import build_dataset
        from tcip_mcp.pipelines.training.generic_trainer import (
            create_run, train, task_collate,
        )

        spec = recommend_model_spec(task="classification", dataset_size=12, num_classes=2)

        dataset = build_dataset("classification", images_dir=tiny_classification_data)

        # Split into train (8) / val (4)
        train_ds, val_ds = torch.utils.data.random_split(dataset, [8, 4])

        collate = task_collate("classification")
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate)

        config = {
            "model_spec": spec,
            "device": "cpu",
            "stages": [{"freeze_to": -1, "epochs": 2}],
            "mixed_precision": False,
            "optimizer": {"name": "sgd", "backbone_lr": 1e-3, "head_lr": 1e-2, "weight_decay": 0},
            "scheduler": {"type": "cosine"},
            "early_stopping": {"enabled": True, "patience": 10, "min_delta": 1e-4},
            "gradient_accumulation_steps": 1,
            "checkpoint_every_n_epochs": 0,
        }
        run = create_run(config, output_dir)
        completed = train(run, train_loader, val_loader=val_loader, task="classification")

        assert completed.status == "completed"
        assert completed.current_epoch == 2
        # Metrics should have val_loss when val_loader provided
        assert "val_loss" in completed.metrics_history[-1]


class TestModelSpecValidation:
    """Verify the agent's spec validation catches errors."""

    def test_invalid_backbone_rejected(self):
        from tcip_mcp.pipelines.composer import validate_model_spec
        issues = validate_model_spec({
            "backbone": {"name": "nonexistent_net"},
            "heads": [{"name": "classification", "num_classes": 5}],
        })
        assert any("backbone" in i.lower() or "unknown" in i.lower() for i in issues)

    def test_missing_heads_rejected(self):
        from tcip_mcp.pipelines.composer import validate_model_spec
        issues = validate_model_spec({
            "backbone": {"name": "resnet50"},
        })
        assert any("head" in i.lower() for i in issues)

    def test_valid_spec_accepted(self):
        from tcip_mcp.pipelines.composer import validate_model_spec
        issues = validate_model_spec({
            "backbone": {"name": "resnet50"},
            "neck": {"name": "gap"},
            "heads": [{"name": "classification", "num_classes": 3}],
        })
        assert issues == []


# ---------------------------------------------------------------------------
# Test: detection pipeline with real catkin data
# ---------------------------------------------------------------------------

SAMPLE_IMAGES = Path(__file__).resolve().parent.parent / "data" / "images"
SAMPLE_LABELS = Path(__file__).resolve().parent.parent / "data" / "labels" / "detect"


@pytest.fixture()
def detection_output_dir(tmp_path):
    return str(tmp_path / "det_output")


@pytest.mark.skipif(
    not SAMPLE_IMAGES.exists() or not SAMPLE_LABELS.exists(),
    reason="Sample detection data not found",
)
class TestDetectionPipelineRealData:
    """End-to-end: compose → train → infer → export CSV using real catkin images."""

    def test_compose_train_infer_export(self, detection_output_dir):
        from tcip_mcp.pipelines.composer import (
            compose_model, DetectionModel, recommend_model_spec,
        )
        from tcip_mcp.pipelines.data.datasets import build_dataset
        from tcip_mcp.pipelines.training.generic_trainer import (
            create_run, train, task_collate,
        )
        from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
        from tcip_mcp.pipelines.postprocessing.export import export_detection_csv

        # --- Step 1: Recommend and compose detection model ---
        spec = recommend_model_spec(
            task="detection", dataset_size=18, num_classes=1,
        )
        assert spec["heads"][0]["name"] == "anchor_detection"
        # Override sizes for speed (real images are 5712×4284)
        spec["backbone"]["pretrained"] = False
        spec["heads"][0]["min_size"] = 320
        spec["heads"][0]["max_size"] = 512

        model = compose_model(spec)
        assert isinstance(model, DetectionModel)

        # --- Step 2: Build dataset from real YOLO labels ---
        dataset = build_dataset(
            "detection",
            images_dir=str(SAMPLE_IMAGES),
            labels_dir=str(SAMPLE_LABELS),
            num_classes=1,
        )
        assert dataset.num_samples == 18
        assert dataset.num_classes == 1

        # Verify one sample loads correctly
        img, target = dataset[0]
        assert img.ndim == 3 and img.shape[0] == 3  # [C, H, W]
        assert target["boxes"].ndim == 2 and target["boxes"].shape[1] == 4
        assert target["labels"].ndim == 1
        assert len(target["labels"]) > 0  # has annotations

        # Use first 4 images for fast training
        subset = torch.utils.data.Subset(dataset, list(range(4)))
        loader = DataLoader(
            subset, batch_size=2, shuffle=True,
            collate_fn=task_collate("detection"),
        )

        # --- Step 3: Train 1 epoch ---
        config = {
            "model_spec": spec,
            "device": "cpu",
            "stages": [{"freeze_to": 0, "epochs": 1}],
            "mixed_precision": False,
            "optimizer": {"name": "sgd", "backbone_lr": 1e-3, "head_lr": 1e-2, "weight_decay": 0},
            "scheduler": {"type": "cosine"},
            "early_stopping": {"enabled": False},
            "gradient_accumulation_steps": 1,
            "checkpoint_every_n_epochs": 1,
        }
        run = create_run(config, detection_output_dir)
        completed = train(run, loader, val_loader=None, task="detection")

        assert completed.status == "completed"
        assert completed.current_epoch == 1

        out = Path(detection_output_dir)
        assert (out / "model_best.pt").is_file()
        assert (out / "metrics.jsonl").is_file()

        # Verify checkpoint format
        ckpt = torch.load(out / "model_best.pt", map_location="cpu", weights_only=False)
        assert "model_state_dict" in ckpt
        assert "model_spec" in ckpt

        # --- Step 4: Inference on real images ---
        predictor = GenericPredictor(
            str(out / "model_best.pt"), device="cpu", score_threshold=0.01,
        )

        test_images = sorted(SAMPLE_IMAGES.glob("*.JPG"))[:3]
        results = predictor.predict_batch([str(p) for p in test_images])

        assert len(results) == 3
        for r in results:
            assert "image" in r
            assert "boxes" in r
            assert "scores" in r
            assert "count" in r

        # --- Step 5: Export detection CSV ---
        csv_path = str(out / "catkin_detections.csv")
        export_detection_csv(results, csv_path)

        assert Path(csv_path).is_file()
        content = Path(csv_path).read_text()
        assert "image" in content
        assert "detection_count" in content
        lines = content.strip().splitlines()
        assert len(lines) == 4  # header + 3 images
