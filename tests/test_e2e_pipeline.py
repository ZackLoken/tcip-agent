"""End-to-end integration test: agent pipeline through MCP tools.

Verifies the full workflow using the MCP tool layer:
  init_project → load_dataset → validate_data_quality →
  load_annotations → save_annotations → evaluate_detections →
  evaluate_dataset → split_dataset → export_project

Each step asserts filesystem state to prove persistence works.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from tcip_mcp.tools.project_tools import (
    init_project,
    get_project_status,
    export_project,
)
from tcip_mcp.tools.data_tools import (
    load_dataset,
    validate_data_quality,
    split_dataset,
)
from tcip_mcp.tools.annotation_tools import (
    load_annotations,
    save_annotations,
    evaluate_detections,
    evaluate_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def project_dir(tmp_path: Path) -> Path:
    """Fully populated TCIP project with images, labels, and predictions."""
    root = tmp_path / "my_project"
    date = "2-11-26"
    images = root / "images" / date
    labels_det = root / "annotations" / "default" / date / "detect"
    preds_det = root / "predictions" / "live" / date / "detect"
    for d in (images, labels_det, preds_det):
        d.mkdir(parents=True)

    from tcip_annotation import json_io
    from tcip_annotation.state import PredBBox

    # 5 synthetic images (640x480 grey) with GT labels and predictions
    for i in range(5):
        name = f"img_{i:03d}"
        img = Image.new("RGB", (640, 480), color=(100 + i * 20, 100, 100))
        img.save(images / f"{name}.jpg")

        # GT: 2 boxes per image. load_annotations / validate_data_quality read GT through
        # format_io, which understands YOLO .txt (not the json_io per-image schema).
        gt_lines = [
            "0 0.5 0.5 0.10 0.10",
            "0 0.3 0.3 0.05 0.05",
        ]
        (labels_det / f"{name}.txt").write_text("\n".join(gt_lines) + "\n")

        # Predictions: 1 matching (TP) + 1 false positive (FP), per-image JSON with a native
        # score. The file keeps a .txt name because find_prediction resolves it with fmt='yolo';
        # json_io parses the JSON content.
        json_io.write_detect(
            str(preds_det / f"{name}.txt"),
            [PredBBox(288, 216, 352, 264, 0, confidence=0.92),
             PredBBox(499.2, 374.4, 524.8, 393.6, 0, confidence=0.60)],
            640, 480,
        )

    return root


# ---------------------------------------------------------------------------
# Phase 13: E2E Pipeline Test
# ---------------------------------------------------------------------------

class TestE2EPipeline:
    """Walk the full pipeline using MCP tools, asserting state at each step."""

    def test_full_pipeline(self, project_dir: Path, tmp_path: Path):
        root = str(project_dir)

        # ── Step 1: Init project ─────────────────────────────────────
        init_project(root)
        assert (project_dir / ".tcip").is_dir()
        assert (project_dir / ".tcip" / "config.toml").is_file()

        status = get_project_status(root)
        assert status["initialized"] is True

        # ── Step 3: Load dataset ─────────────────────────────────────
        ds = load_dataset(root)
        assert ds["image_count"] == 5
        assert ds["labels_detect_count"] == 5
        assert ds["predictions_detect_count"] == 5
        assert ds["paired_images"] == 5
        assert ds["unlabelled_images"] == 0

        # ── Step 4: Validate data quality ────────────────────────────
        quality = validate_data_quality(root)
        assert quality["total_images"] == 5
        assert quality["is_valid"] is True
        assert 0 in quality["class_ids"]

        # ── Step 5: Load annotations for one image ───────────────────
        img_path = str(project_dir / "images" / "2-11-26" / "img_000.jpg")
        ann = load_annotations(img_path)
        assert "error" not in ann
        assert ann["detect_labels"]["count"] >= 2
        assert ann["detect_predictions"]["count"] >= 2

        # ── Step 6: Modify annotations — add a box and save ─────────
        new_boxes = [
            {"x1": 200, "y1": 200, "x2": 264, "y2": 248, "class_id": 0},
            {"x1": 128, "y1": 112, "x2": 160, "y2": 136, "class_id": 0},
            {"x1": 400, "y1": 300, "x2": 440, "y2": 340, "class_id": 1},
        ]
        save_result = save_annotations(img_path, boxes=new_boxes)
        assert save_result["count"] == 1
        assert len(save_result["written"]) == 1

        # Verify file was updated (canonical per-image JSON)
        from tcip_annotation import json_io

        label_path = project_dir / "annotations" / "default" / "2-11-26" / "detect" / "img_000.json"
        boxes, _ = json_io.read_detect(str(label_path))
        assert len(boxes) == 3  # we wrote 3 boxes

        # ── Step 7: Evaluate single image detections ─────────────────
        eval_result = evaluate_detections(img_path, iou_threshold=0.5, conf_threshold=0.25)
        assert "error" not in eval_result
        # Should have precision, recall, f1 keys
        assert "precision" in eval_result
        assert "recall" in eval_result
        assert "f1" in eval_result
        assert isinstance(eval_result["precision"], float)

        # ── Step 8: Detailed per-detection breakdown (evaluate_detections detail=True) ─
        match_result = evaluate_detections(img_path, iou_threshold=0.5, conf_threshold=0.25, detail=True)
        assert "error" not in match_result
        assert "detections" in match_result
        assert "img_w" in match_result
        assert "img_h" in match_result

        # ── Step 9: Evaluate full dataset ────────────────────────────
        dataset_eval = evaluate_dataset(root, iou_threshold=0.5, conf_threshold=0.25)
        assert "error" not in dataset_eval
        assert dataset_eval["image_count"] == 5
        assert "precision" in dataset_eval
        assert "recall" in dataset_eval
        assert "f1" in dataset_eval

        # ── Step 10: Split dataset ───────────────────────────────────
        split_dir = tmp_path / "splits"
        split_result = split_dataset(root, output_path=str(split_dir))
        assert split_result["total"] == 5
        assert (split_dir / "train.json").is_file()
        assert (split_dir / "val.json").is_file()
        assert sum(split_result["splits"].values()) == 5

        # Verify split JSON content
        train_data = json.loads((split_dir / "train.json").read_text())
        assert isinstance(train_data, list)
        assert len(train_data) > 0

        # ── Step 11: Export project as ZIP ───────────────────────────
        zip_path = str(tmp_path / "export.zip")
        export_result = export_project(root, zip_path)
        assert "error" not in export_result
        assert Path(zip_path).is_file()
        assert Path(zip_path).stat().st_size > 0


class TestE2EPipelineEdgeCases:
    """Edge-case scenarios for the pipeline."""

    def test_empty_project(self, tmp_path: Path):
        """Pipeline tools handle an uninitialised project gracefully."""
        status = get_project_status(str(tmp_path))
        assert status["initialized"] is False

    def test_dataset_with_missing_labels(self, tmp_path: Path):
        """load_dataset reports unlabelled images correctly."""
        images = tmp_path / "images"
        images.mkdir()
        labels = tmp_path / "annotations" / "default" / "detect"
        labels.mkdir(parents=True)

        # 3 images, only 1 label
        for i in range(3):
            img = Image.new("RGB", (64, 64))
            img.save(images / f"img_{i:03d}.jpg")
        (labels / "img_000.txt").write_text("0 0.5 0.5 0.1 0.1\n")

        ds = load_dataset(str(tmp_path))
        assert ds["image_count"] == 3
        assert ds["labels_detect_count"] == 1
        assert ds["unlabelled_images"] == 2

    def test_evaluate_no_predictions(self, tmp_path: Path):
        """evaluate_detections handles images with no predictions."""
        images = tmp_path / "images"
        labels = tmp_path / "annotations" / "default" / "detect"
        for d in (images, labels):
            d.mkdir(parents=True)

        img = Image.new("RGB", (640, 480))
        img_path = images / "test.jpg"
        img.save(img_path)
        (labels / "test.txt").write_text("0 0.5 0.5 0.1 0.1\n")

        # No predictions directory — evaluate should handle gracefully
        result = evaluate_detections(str(img_path))
        # Either returns an error dict or metrics with 0 TP
        assert isinstance(result, dict)

    def test_save_annotations_creates_directories(self, tmp_path: Path):
        """save_annotations creates label directories if missing."""
        images = tmp_path / "images"
        images.mkdir()
        img = Image.new("RGB", (640, 480))
        img_path = images / "new_img.jpg"
        img.save(img_path)

        # No labels dir exists yet
        result = save_annotations(str(img_path), boxes=[
            {"x1": 10, "y1": 10, "x2": 50, "y2": 50, "class_id": 0},
        ])
        assert result["count"] == 1
        assert (tmp_path / "annotations" / "default" / "detect" / "new_img.json").is_file()
