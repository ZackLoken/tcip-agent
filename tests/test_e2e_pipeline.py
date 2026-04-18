"""End-to-end integration test: agent pipeline through MCP tools.

Verifies the full workflow using the MCP tool layer:
  init_project → create_session → load_dataset → validate_data_quality →
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
    create_session,
    append_session_event,
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
    run_matching,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def project_dir(tmp_path: Path) -> Path:
    """Fully populated TCIP project with images, labels, and predictions."""
    root = tmp_path / "my_project"
    images = root / "images"
    labels_det = root / "labels" / "detect"
    preds_det = root / "predictions" / "detect"
    for d in (images, labels_det, preds_det):
        d.mkdir(parents=True)

    # 5 synthetic images (640x480 grey) with GT labels and predictions
    for i in range(5):
        name = f"img_{i:03d}"
        img = Image.new("RGB", (640, 480), color=(100 + i * 20, 100, 100))
        img.save(images / f"{name}.jpg")

        # GT: 2 boxes per image (YOLO normalised: class cx cy w h)
        gt_lines = [
            "0 0.5 0.5 0.10 0.10",
            "0 0.3 0.3 0.05 0.05",
        ]
        (labels_det / f"{name}.txt").write_text("\n".join(gt_lines) + "\n")

        # Predictions: 1 matching, 1 false positive (class conf cx cy w h)
        pred_lines = [
            "0 0.92 0.5 0.5 0.10 0.10",   # TP
            "0 0.60 0.8 0.8 0.04 0.04",    # FP
        ]
        (preds_det / f"{name}.txt").write_text("\n".join(pred_lines) + "\n")

    return root


# ---------------------------------------------------------------------------
# Phase 13: E2E Pipeline Test
# ---------------------------------------------------------------------------

class TestE2EPipeline:
    """Walk the full pipeline using MCP tools, asserting state at each step."""

    def test_full_pipeline(self, project_dir: Path, tmp_path: Path):
        root = str(project_dir)

        # ── Step 1: Init project ─────────────────────────────────────
        result = init_project(root)
        assert (project_dir / ".tcip").is_dir()
        assert (project_dir / ".tcip" / "sessions").is_dir()
        assert (project_dir / ".tcip" / "config.toml").is_file()

        status = get_project_status(root)
        assert status["initialized"] is True

        # ── Step 2: Create session ───────────────────────────────────
        session = create_session(root, description="E2E test run")
        sid = session["session_id"]
        assert sid
        session_file = project_dir / ".tcip" / "sessions" / f"{sid}.jsonl"
        assert session_file.is_file()

        # ── Step 3: Load dataset ─────────────────────────────────────
        ds = load_dataset(root)
        assert ds["image_count"] == 5
        assert ds["labels_detect_count"] == 5
        assert ds["predictions_detect_count"] == 5
        assert ds["paired_images"] == 5
        assert ds["unlabelled_images"] == 0

        append_session_event(root, sid, "tool_call", {
            "tool": "load_dataset", "image_count": ds["image_count"],
        })

        # ── Step 4: Validate data quality ────────────────────────────
        quality = validate_data_quality(root)
        assert quality["total_images"] == 5
        assert quality["is_valid"] is True
        assert 0 in quality["class_ids"]

        append_session_event(root, sid, "tool_call", {
            "tool": "validate_data_quality", "valid": quality["is_valid"],
        })

        # ── Step 5: Load annotations for one image ───────────────────
        img_path = str(project_dir / "images" / "img_000.jpg")
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

        # Verify file was updated
        label_path = project_dir / "labels" / "detect" / "img_000.txt"
        lines = label_path.read_text().strip().splitlines()
        assert len(lines) == 3  # we wrote 3 boxes

        append_session_event(root, sid, "tool_call", {
            "tool": "save_annotations", "image": "img_000.jpg",
        })

        # ── Step 7: Evaluate single image detections ─────────────────
        eval_result = evaluate_detections(img_path, iou_threshold=0.5, conf_threshold=0.25)
        assert "error" not in eval_result
        # Should have precision, recall, f1 keys
        assert "precision" in eval_result
        assert "recall" in eval_result
        assert "f1" in eval_result
        assert isinstance(eval_result["precision"], float)

        # ── Step 8: Run detailed matching ────────────────────────────
        match_result = run_matching(img_path, iou_threshold=0.5, conf_threshold=0.25)
        assert "error" not in match_result
        assert "tp_count" in match_result
        assert "fp_count" in match_result
        assert "fn_count" in match_result

        # ── Step 9: Evaluate full dataset ────────────────────────────
        dataset_eval = evaluate_dataset(root, iou_threshold=0.5, conf_threshold=0.25)
        assert "error" not in dataset_eval
        assert dataset_eval["image_count"] == 5
        assert "precision" in dataset_eval
        assert "recall" in dataset_eval
        assert "f1" in dataset_eval

        append_session_event(root, sid, "tool_call", {
            "tool": "evaluate_dataset",
            "f1": dataset_eval["f1"],
        })

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

        append_session_event(root, sid, "tool_call", {
            "tool": "split_dataset", "splits": split_result["splits"],
        })

        # ── Step 11: Export project as ZIP ───────────────────────────
        zip_path = str(tmp_path / "export.zip")
        export_result = export_project(root, zip_path)
        assert "error" not in export_result
        assert Path(zip_path).is_file()
        assert Path(zip_path).stat().st_size > 0

        append_session_event(root, sid, "result", {
            "pipeline": "complete", "export": zip_path,
        })

        # ── Verify session log has all events ────────────────────────
        events = session_file.read_text().strip().splitlines()
        # session_start + 5 tool_call + 1 result = 7 total
        assert len(events) == 7

        # Verify each event line is valid JSON with required fields
        for line in events:
            evt = json.loads(line)
            assert "type" in evt
            assert "timestamp" in evt


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
        labels = tmp_path / "labels" / "detect"
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
        labels = tmp_path / "labels" / "detect"
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
        assert (tmp_path / "labels" / "detect" / "new_img.txt").is_file()
