"""End-to-end integration test: agent pipeline through MCP tools and the demoted library calls.

Verifies the full workflow:
  initialize_project → scan_dataset → the doctor's check_data_quality →
  read_annotations → save_annotations → score_predictions (image) →
  score_predictions (dataset) → draw_splits → archive_project

Each step asserts filesystem state to prove persistence works.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from scripts import doctor
from tcip_mcp.tools.project_tools import (
    initialize_project,
    inspect_project,
    archive_project,
)
from tcip_mcp.tools.data_tools import (
    scan_dataset,
    draw_splits,
)
from tcip_mcp.tools.annotation_tools import (
    read_annotations,
    save_annotations,
    score_predictions,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def project_dir(tmp_path: Path) -> Path:
    """Fully populated TCIP project with images, labels, predictions, and a nested registry."""
    root = tmp_path / "my_project"
    date = "2-11-26"
    images = root / "images" / date
    labels_dir = root / "annotations" / date
    preds_dir = root / "predictions" / "live" / date
    for d in (images, labels_dir, preds_dir):
        d.mkdir(parents=True)

    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp import class_registry
    from tcip_mcp.class_registry import ClassRegistry, Subject

    class_registry.write_registry(
        root / "classes.json",
        ClassRegistry(subjects=(Subject(name="bud", description="a currant bud"),)))

    # 5 synthetic images (640x480 grey) with GT labels and predictions
    for i in range(5):
        name = f"img_{i:03d}"
        img = Image.new("RGB", (640, 480), color=(100 + i * 20, 100, 100))
        img.save(images / f"{name}.jpg")

        # GT: 2 boxes per image, name-based per-image JSON (pixel xyxy).
        json_io.write_annotations(
            str(labels_dir / f"{name}.json"),
            [Annotation(subject="bud", geometry=BBox(288, 216, 352, 264)),
             Annotation(subject="bud", geometry=BBox(176, 132, 208, 156))],
            640, 480,
        )
        # Predictions: 1 matching (TP) + 1 false positive (FP), the confidence in each score.
        json_io.write_annotations(
            str(preds_dir / f"{name}.json"),
            [Annotation(subject="bud", geometry=BBox(288, 216, 352, 264), score=0.92),
             Annotation(subject="bud", geometry=BBox(499.2, 374.4, 524.8, 393.6), score=0.60)],
            640, 480,
        )

    return root


# ---------------------------------------------------------------------------
# E2E Pipeline Test
# ---------------------------------------------------------------------------

class TestE2EPipeline:
    """Walk the full pipeline using MCP tools, asserting state at each step."""

    def test_full_pipeline(self, project_dir: Path, tmp_path: Path):
        root = str(project_dir)

        # ── Step 1: Init project ─────────────────────────────────────
        initialize_project(root, site="north orchard")
        assert (project_dir / ".tcip").is_dir()
        assert (project_dir / ".tcip" / "artifacts").is_dir()

        status = inspect_project(root)
        assert status["initialized"] is True

        # ── Step 3: Load dataset ─────────────────────────────────────
        ds = scan_dataset(root)
        assert ds["image_count"] == 5
        assert ds["labels_count"] == 5
        assert ds["predictions_count"] == 5
        assert ds["paired_images"] == 5
        assert ds["unlabelled_images"] == 0

        # ── Step 4: Validate data quality ────────────────────────────
        findings: list[tuple[str, str]] = []
        doctor.check_data_quality(Path(root), findings)
        assert not [f for f in findings if f[0] == "error"]

        # ── Step 5: Load annotations for one image ───────────────────
        img_path = str(project_dir / "images" / "2-11-26" / "img_000.jpg")
        ann = read_annotations(img_path)
        assert "error" not in ann
        assert ann["labels"]["count"] >= 2
        assert ann["predictions"]["count"] >= 2

        # ── Step 6: Modify annotations, add boxes and save ─────────
        new_anns = [
            {"subject": "bud", "bbox": [200, 200, 264, 248]},
            {"subject": "bud", "bbox": [128, 112, 160, 136]},
            {"subject": "bud", "bbox": [400, 300, 440, 340]},
        ]
        save_result = save_annotations(img_path, annotations=new_anns)
        assert save_result["count"] == 3  # 3 annotations written
        assert len(save_result["written"]) == 1

        # Verify file was updated (name-based per-image JSON)
        from tcip_annotation import json_io

        label_path = project_dir / "annotations" / "2-11-26" / "img_000.json"
        anns = json_io.read_annotations(str(label_path))
        assert len(anns) == 3  # we wrote 3 boxes

        # ── Step 7: Evaluate single image detections ─────────────────
        eval_result = score_predictions(img_path, iou_threshold=0.5, conf_threshold=0.25)
        assert "error" not in eval_result
        # Should have precision, recall, f1 keys
        assert "precision" in eval_result
        assert "recall" in eval_result
        assert "f1" in eval_result
        assert isinstance(eval_result["precision"], float)

        # ── Step 8: Detailed per-detection breakdown (score_predictions detail=True) ─
        match_result = score_predictions(img_path, iou_threshold=0.5, conf_threshold=0.25, detail=True)
        assert "error" not in match_result
        assert "detections" in match_result
        assert "img_w" in match_result
        assert "img_h" in match_result

        # ── Step 9: Evaluate full dataset ────────────────────────────
        dataset_eval = score_predictions(root, iou_threshold=0.5, conf_threshold=0.25)
        assert "error" not in dataset_eval
        assert dataset_eval["image_count"] == 5
        assert "precision" in dataset_eval
        assert "recall" in dataset_eval
        assert "f1" in dataset_eval

        # ── Step 10: Split dataset ───────────────────────────────────
        import tcip_store as ts
        from tcip_mcp.tools.data_tools import split_manifest_key

        split_dir = tmp_path / "splits"
        split_result = draw_splits(root, output_path=str(split_dir), materialize=True,
                                   subject="bud", train_ratio=0.5, val_ratio=0.25,
                                   calibration_ratio=0.25)
        assert split_result["total_stems"] == 5
        manifest = ts.read(split_manifest_key(split_dir))
        assert manifest["splits"]["train"]
        assert manifest["splits"]["val"]
        assert sum(split_result["splits"].values()) == 5

        # Verify split membership content
        train_data = manifest["splits"]["train"]
        assert isinstance(train_data, list)
        assert len(train_data) > 0

        # ── Step 11: Export project as ZIP ───────────────────────────
        from tcip_store.file_backend import database_file

        if database_file(root).is_file():
            # A database-backed project's state is not in the files a bundle carries: the
            # platform's own export step lands it there first, same as an operator would run.
            from tcip_store.export import export_root

            export_root(root, report=lambda line: None)
        zip_path = str(tmp_path / "export.zip")
        export_result = archive_project(root, zip_path)
        assert "error" not in export_result
        assert Path(zip_path).is_file()
        assert Path(zip_path).stat().st_size > 0


class TestE2EPipelineEdgeCases:
    """Edge-case scenarios for the pipeline."""

    def test_empty_project(self, tmp_path: Path):
        """Pipeline tools handle an uninitialised project gracefully."""
        status = inspect_project(str(tmp_path))
        assert status["initialized"] is False

    def test_dataset_with_missing_labels(self, tmp_path: Path):
        """scan_dataset reports unlabelled images correctly."""
        images = tmp_path / "images"
        images.mkdir()
        labels = tmp_path / "annotations"
        labels.mkdir(parents=True)

        # 3 images, only 1 label
        from tcip_annotation import json_io
        from tcip_annotation.state import Annotation, BBox
        for i in range(3):
            img = Image.new("RGB", (64, 64))
            img.save(images / f"img_{i:03d}.jpg")
        json_io.write_annotations(str(labels / "img_000.json"),
                                  [Annotation(subject="bud", geometry=BBox(28, 28, 36, 36))], 64, 64)

        ds = scan_dataset(str(tmp_path))
        assert ds["image_count"] == 3
        assert ds["labels_count"] == 1
        assert ds["unlabelled_images"] == 2

    def test_evaluate_no_predictions(self, tmp_path: Path):
        """score_predictions handles images with no predictions."""
        images = tmp_path / "images"
        labels = tmp_path / "annotations"
        for d in (images, labels):
            d.mkdir(parents=True)

        from tcip_annotation import json_io
        from tcip_annotation.state import Annotation, BBox
        img = Image.new("RGB", (640, 480))
        img_path = images / "test.jpg"
        img.save(img_path)
        json_io.write_annotations(str(labels / "test.json"),
                                  [Annotation(subject="bud", geometry=BBox(288, 216, 352, 264))], 640, 480)

        # No predictions directory: evaluate should handle gracefully
        result = score_predictions(str(img_path))
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
        result = save_annotations(str(img_path), annotations=[
            {"subject": "bud", "bbox": [10, 10, 50, 50]},
        ])
        assert result["count"] == 1
        assert (tmp_path / "annotations" / "new_img.json").is_file()
