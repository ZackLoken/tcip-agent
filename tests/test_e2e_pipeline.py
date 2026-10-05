"""End-to-end integration test: agent pipeline through MCP tools and the demoted library calls.

Verifies the full workflow:
  initialize_project → scan_dataset → the doctor's check_data_quality →
  read_annotations → save_annotations → score_predictions (image) →
  score_predictions (dataset) → draw_splits → archive_project

Each step asserts filesystem state to prove persistence works.
"""

from __future__ import annotations

from tcip_mcp.dataset_layout import UNDATED_BUCKET

from pathlib import Path

import pytest
from PIL import Image

from tcip_mcp.cli import doctor
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
    images.mkdir(parents=True)

    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.subject_registry import SubjectRegistry, Subject
    from tests._producer_fixtures import label_image, registry_over

    registry_over(root, SubjectRegistry(subjects=(Subject(name="bud", description="a currant bud"),)))

    # 5 synthetic images (640x480 gray) with GT labels and predictions
    results = []
    for i in range(5):
        name = f"img_{i:03d}"
        img = Image.new("RGB", (640, 480), color=(100 + i * 20, 100, 100))
        img.save(images / f"{name}.jpg")

        # GT: 2 boxes per image, name-based per-image JSON (pixel xyxy).
        label_image(images / f"{name}.jpg",
                    [Annotation(subject="bud", geometry=BBox(288, 216, 352, 264)),
                     Annotation(subject="bud", geometry=BBox(176, 132, 208, 156))], 640, 480)
        # Predictions: 1 matching (TP) + 1 false positive (FP), the confidence in each score.
        results.append({"image": str(images / f"{name}.jpg"), "width": 640, "height": 480,
                        "boxes": [[288, 216, 352, 264], [499.2, 374.4, 524.8, 393.6]],
                        "scores": [0.92, 0.60], "labels": [1, 1]})
    pytest.importorskip("torch")
    from tests._chain_fixtures import published

    published(root, BUCKET, results, scope={"subject": "bud"})
    return root


BUCKET = "live/2-11-26"


# ---------------------------------------------------------------------------
# E2E Pipeline Test
# ---------------------------------------------------------------------------

class TestE2EPipeline:
    """Walk the full pipeline using MCP tools, asserting state at each step."""

    def test_full_pipeline(self, project_dir: Path, tmp_path: Path):
        root = str(project_dir)

        # ── Step 1: Init project ─────────────────────────────────────
        initialize_project(root, "North orchard", site="north orchard")
        assert (project_dir / ".tcip").is_dir()
        assert (project_dir / ".tcip" / "artifacts").is_dir()

        status = inspect_project(project_dir)
        assert status["record_problem"] is None
        assert status["display_name"] == "North orchard"

        # ── Step 3: Load dataset ─────────────────────────────────────
        ds = scan_dataset(root)
        assert ds["image_count"] == 5
        assert ds["labels_count"] == 5
        assert ds["predictions_count"] == 5
        assert ds["paired_images"] == 5
        assert ds["unlabeled_images"] == 0

        # ── Step 4: Validate data quality ────────────────────────────
        findings: list[tuple[str, str]] = []
        doctor.check_data_quality(Path(root), findings, census=doctor._census(Path(root), findings))
        assert not [f for f in findings if f[0] == "error"]

        # ── Step 5: Load annotations for one image ───────────────────
        img_path = str(project_dir / "images" / "2-11-26" / "img_000.jpg")
        ann = read_annotations(img_path, BUCKET)
        assert "error" not in ann
        assert ann["labels"]["count"] >= 2
        assert ann["predictions"]["count"] >= 2

        # ── Step 6: Modify annotations, add boxes and save ─────────
        new_anns = [
            {"subject": "bud", "bbox": [200, 200, 264, 248]},
            {"subject": "bud", "bbox": [128, 112, 160, 136]},
            {"subject": "bud", "bbox": [400, 300, 440, 340]},
        ]
        save_result = save_annotations(project_dir, project_dir.parent, img_path,
                                       annotations=new_anns)
        assert save_result["count"] == 3  # 3 annotations written
        assert len(save_result["written"]) == 1

        from tcip_annotation import json_io

        from tests._producer_fixtures import image_label_key

        doc = json_io.read_label_document(image_label_key(img_path))
        assert len(doc.annotations) == 3  # we wrote 3 boxes

        # ── Step 7: Evaluate single image detections ─────────────────
        eval_result = score_predictions(img_path, BUCKET,iou_threshold=0.5, conf_threshold=0.25)
        assert "error" not in eval_result
        # Should have precision, recall, f1 keys
        assert "precision" in eval_result
        assert "recall" in eval_result
        assert "f1" in eval_result
        assert isinstance(eval_result["precision"], float)

        # ── Step 8: Detailed per-detection breakdown (score_predictions detail=True) ─
        match_result = score_predictions(img_path, BUCKET,iou_threshold=0.5, conf_threshold=0.25,
                                         detail=True)
        assert "error" not in match_result
        assert "detections" in match_result
        assert "img_w" in match_result
        assert "img_h" in match_result

        # ── Step 9: Evaluate full dataset ────────────────────────────
        dataset_eval = score_predictions(str(project_dir / "images" / "2-11-26"), BUCKET,
                                         iou_threshold=0.5, conf_threshold=0.25)
        assert "error" not in dataset_eval
        assert dataset_eval["image_count"] == 5
        assert "precision" in dataset_eval
        assert "recall" in dataset_eval
        assert "f1" in dataset_eval

        # ── Step 10: Split dataset ───────────────────────────────────
        from tcip_mcp.pipelines.data.selection import read_selection

        split_dir = tmp_path / "splits"
        split_result = draw_splits(project_dir, root, output_path=str(split_dir),
                                   subject="bud", seed=1, val_ratio=0.25,
                                   calibration_ratio=0.125, holdout_ratio=0.125)
        assert split_result["total_stems"] == 5
        drawn = read_selection(split_dir, project=project_dir)
        assert drawn.on("train")
        assert drawn.on("val")
        assert sum(split_result["splits"].values()) == 5

        # Every sample names its own image and label document; nothing was copied.
        import tcip_store

        for sample in drawn.samples:
            assert Path(sample.source).is_file()
            assert tcip_store.exists(sample.ground_truth)

        # ── Step 11: Export project as ZIP ───────────────────────────
        zip_path = str(tmp_path / "export.zip")
        export_result = archive_project(project_dir, zip_path)
        assert "error" not in export_result
        assert Path(zip_path).is_file()
        assert Path(zip_path).stat().st_size > 0


class TestE2EPipelineEdgeCases:
    """Edge-case scenarios for the pipeline."""

    def test_empty_project(self, tmp_path: Path):
        """Pipeline tools handle an uninitialized project gracefully."""
        status = inspect_project(tmp_path)
        assert status["id"] is None and status["record_problem"]

    def test_dataset_with_missing_labels(self, tmp_path: Path):
        """scan_dataset reports unlabeled images correctly."""
        images = tmp_path / "images" / UNDATED_BUCKET
        images.mkdir(parents=True)

        # 3 images, only 1 label
        from tcip_annotation.state import Annotation, BBox

        from tests._producer_fixtures import label_image

        for i in range(3):
            img = Image.new("RGB", (64, 64))
            img.save(images / f"img_{i:03d}.jpg")
        label_image(images / "img_000.jpg",
                    [Annotation(subject="bud", geometry=BBox(28, 28, 36, 36))], 64, 64)

        ds = scan_dataset(str(tmp_path))
        assert ds["image_count"] == 3
        assert ds["labels_count"] == 1
        assert ds["unlabeled_images"] == 2

    def test_evaluate_no_predictions(self, tmp_path: Path):
        """score_predictions handles images with no predictions."""
        images = tmp_path / "images" / UNDATED_BUCKET
        images.mkdir(parents=True)

        from tcip_annotation.state import Annotation, BBox

        from tests._producer_fixtures import label_image

        img = Image.new("RGB", (640, 480))
        img_path = images / "test.jpg"
        img.save(img_path)
        label_image(img_path, [Annotation(subject="bud", geometry=BBox(288, 216, 352, 264))],
                    640, 480)

        # A bucket name never published: evaluate should handle gracefully
        result = score_predictions(str(img_path), "none/2026-01-01")
        # Either returns an error dict or metrics with 0 TP
        assert isinstance(result, dict)

    def test_save_annotations_writes_a_first_label_document(self, tmp_path: Path):
        """save_annotations writes an image's first label document."""
        import tcip_store

        from tests._producer_fixtures import image_label_key

        images = tmp_path / "images" / UNDATED_BUCKET
        images.mkdir(parents=True)
        img = Image.new("RGB", (640, 480))
        img_path = images / "new_img.jpg"
        img.save(img_path)

        result = save_annotations(tmp_path, tmp_path.parent, str(img_path), annotations=[
            {"subject": "bud", "bbox": [10, 10, 50, 50]},
        ])
        assert result["count"] == 1
        assert tcip_store.exists(image_label_key(img_path))
