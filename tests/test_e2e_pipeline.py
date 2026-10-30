"""End-to-end integration test: the agent pipeline through MCP tools and library calls.

The workflow covered:
  initialize_project → scan_dataset → the doctor's check_data_quality →
  read_annotations → save_annotations → score_predictions (image) →
  score_predictions (dataset) → draw_splits → archive_project
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir
from tcip_mcp.tools import doctor_tools as doctor
from tcip_mcp.tools.annotation_tools import (
    read_annotations,
    save_annotations,
    score_predictions,
)
from tcip_mcp.tools.data_tools import (
    draw_splits,
    scan_dataset,
)
from tcip_mcp.tools.project_tools import (
    ZipBundle,
    archive_project,
    initialize_project,
    inspect_project,
)
from tcip_mcp.workspace import BoundProject, workspace_from_environment
from tests._producer_fixtures import WrittenDataset
from tests._web_fixtures import INVARIANCES

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def project_dataset(tmp_path: Path) -> WrittenDataset:
    """Write a project of five 640x480 images, two bud labels each, and a published bucket.

    The registry declares ``bud``; each image's predictions, published as :data:`BUCKET`, are
    one box on its first label's box at 0.92 and one box away from both labels at 0.60.
    """
    root = tmp_path / "my_project"
    date = "2-11-26"
    images = image_dir(root, date)
    images.mkdir(parents=True)

    from tcip_annotation.state import Annotation, BBox
    from tests._producer_fixtures import (
        label_image,
        registry_of,
        registry_over,
        write_image,
    )

    registry_over(root, registry_of("bud"))

    # 5 synthetic images (640x480 gray) with GT labels and predictions
    written = [
        write_image(
            images / f"img_{i:03d}.jpg", (640, 480), (100 + i * 20, 100, 100)
        )
        for i in range(5)
    ]
    results = []
    for image in written:
        # GT: 2 boxes per image, name-based per-image JSON (pixel xyxy).
        label_image(
            image,
            [
                Annotation(subject="bud", geometry=BBox(288, 216, 352, 264)),
                Annotation(subject="bud", geometry=BBox(176, 132, 208, 156)),
            ],
            640,
            480,
        )
        # Predictions: 1 matching (TP) + 1 false positive (FP), the confidence in each score.
        results.append(
            {
                "image": str(image),
                "width": 640,
                "height": 480,
                "boxes": [[288, 216, 352, 264], [499.2, 374.4, 524.8, 393.6]],
                "scores": [0.92, 0.60],
                "labels": [1, 1],
                "cap": 3,
            }
        )
    pytest.importorskip("torch")
    from tests._chain_fixtures import published

    published(root, BUCKET, results, scope={"subject": "bud"})
    return WrittenDataset(root, images, written)


BUCKET = "live/2-11-26"


# ---------------------------------------------------------------------------
# E2E Pipeline Test
# ---------------------------------------------------------------------------


class TestE2EPipeline:
    """The full pipeline walked through MCP tools, its state asserted at each step."""

    def test_full_pipeline(
        self, project_dataset: WrittenDataset, tmp_path: Path
    ) -> None:
        project_dir = project_dataset.root
        root = str(project_dir)

        # ── Step 1: Init project ─────────────────────────────────────
        created = initialize_project(
            root,
            "North orchard",
            site="north orchard",
            invariances=INVARIANCES,
            actor=None,
        )
        assert (project_dir / ".tcip").is_dir()
        assert (project_dir / ".tcip" / "artifacts").is_dir()

        status = inspect_project(project_dir)
        assert status["record_problem"] is None
        assert status["record"]["display_name"] == "North orchard"

        # ── Step 3: Load dataset ─────────────────────────────────────
        ds = scan_dataset(project_dir, root)
        assert ds["image_count"] == 5
        assert ds["labels_count"] == 5
        assert ds["predictions_count"] == 5
        assert ds["paired_images"] == 5
        assert ds["unlabeled_images"] == 0

        # ── Step 4: Validate data quality ────────────────────────────
        findings: list[tuple[str, str]] = []
        census = doctor._census(Path(root), findings)
        assert census is not None, findings
        doctor.check_data_quality(Path(root), findings, census=census)
        assert not [f for f in findings if f[0] == "error"]

        # ── Step 5: Load annotations for one image ───────────────────
        img_path = str(project_dataset.images[0])
        ann = read_annotations(project_dir, img_path, BUCKET)
        assert "error" not in ann
        assert ann["labels"]["count"] >= 2
        assert ann["predictions"]["count"] >= 2

        # ── Step 6: Modify annotations, add boxes and save ─────────
        new_anns = [
            {"subject": "bud", "bbox": [200, 200, 264, 248]},
            {"subject": "bud", "bbox": [128, 112, 160, 136]},
            {"subject": "bud", "bbox": [400, 300, 440, 340]},
        ]
        save_result = save_annotations(
            BoundProject(
                project_dir, created["record"]["id"], workspace_from_environment()
            ),
            img_path,
            annotations=new_anns,
            actor=None,
        )
        assert save_result["count"] == 3  # 3 annotations written
        assert len(save_result["written"]) == 1

        from tcip_annotation import json_io
        from tests._producer_fixtures import image_label_key

        doc = json_io.read_label_document(image_label_key(img_path))
        assert len(doc.annotations) == 3  # we wrote 3 boxes

        # ── Step 7: Evaluate single image detections ─────────────────
        eval_result = score_predictions(
            project_dir,
            img_path,
            BUCKET,
            iou_threshold=0.5,
            conf_threshold=0.25,
        )
        assert "error" not in eval_result
        # Should have precision, recall, f1 keys
        assert "precision" in eval_result
        assert "recall" in eval_result
        assert "f1" in eval_result
        assert isinstance(eval_result["precision"], float)

        # ── Step 8: Detailed per-detection breakdown (score_predictions detail=True) ─
        match_result = score_predictions(
            project_dir,
            img_path,
            BUCKET,
            iou_threshold=0.5,
            conf_threshold=0.25,
            detail=True,
        )
        assert "error" not in match_result
        assert "detections" in match_result
        assert "img_w" in match_result
        assert "img_h" in match_result

        # ── Step 9: Evaluate full dataset ────────────────────────────
        dataset_eval = score_predictions(
            project_dir,
            str(project_dataset.images_dir),
            BUCKET,
            iou_threshold=0.5,
            conf_threshold=0.25,
        )
        assert "error" not in dataset_eval
        assert dataset_eval["image_count"] == 5
        assert "precision" in dataset_eval
        assert "recall" in dataset_eval
        assert "f1" in dataset_eval

        # ── Step 10: Split dataset ───────────────────────────────────
        from tcip_mcp.pipelines.data.selection import read_selection
        from tests._training_values import drawn_split

        split_dir = tmp_path / "splits"
        split_result = draw_splits(
            project_dir,
            root,
            output_path=str(split_dir),
            subject="bud",
            **drawn_split(
                seed=1,
                val_ratio=0.25,
                calibration_ratio=0.125,
                holdout_ratio=0.125,
                grouping="tile_prefix",
            ),
            actor=None,
        )
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
        export_result = archive_project(
            project_dir, ZipBundle(output_path=zip_path)
        )
        assert "error" not in export_result
        assert Path(zip_path).is_file()
        assert Path(zip_path).stat().st_size > 0


class TestE2EPipelineEdgeCases:
    """Edge-case scenarios for the pipeline."""

    def test_empty_project(self, tmp_path: Path) -> None:
        """An uninitialized project inspects as no record, its record problem named."""
        status = inspect_project(tmp_path)
        assert status["record"] is None and status["record_problem"]

    def test_dataset_with_missing_labels(self, tmp_path: Path) -> None:
        """scan_dataset counts three images, one label and two unlabeled images."""
        images = image_dir(tmp_path, UNDATED_BUCKET)

        # 3 images, only 1 label
        from tcip_annotation.state import Annotation, BBox
        from tests._producer_fixtures import label_image, write_image

        for i in range(3):
            write_image(images / f"img_{i:03d}.jpg", (64, 64))
        label_image(
            images / "img_000.jpg",
            [Annotation(subject="bud", geometry=BBox(28, 28, 36, 36))],
            64,
            64,
        )

        ds = scan_dataset(tmp_path, str(tmp_path))
        assert ds["image_count"] == 3
        assert ds["labels_count"] == 1
        assert ds["unlabeled_images"] == 2

    def test_evaluate_no_predictions(self, tmp_path: Path) -> None:
        """score_predictions answers a dict for an image under an unpublished bucket name."""
        from tcip_annotation.state import Annotation, BBox
        from tests._producer_fixtures import label_image, write_image

        img_path = write_image(
            image_dir(tmp_path, UNDATED_BUCKET) / "test.jpg", (640, 480)
        )
        label_image(
            img_path,
            [Annotation(subject="bud", geometry=BBox(288, 216, 352, 264))],
            640,
            480,
        )

        # A bucket name never published: evaluate should handle gracefully
        result = score_predictions(
            tmp_path,
            str(img_path),
            "none/2026-01-01",
            iou_threshold=0.5,
            conf_threshold=0.25,
        )
        # Either returns an error dict or metrics with 0 TP
        assert isinstance(result, dict)

    def test_save_annotations_writes_a_first_label_document(
        self, tmp_path: Path, bound: BoundProject
    ) -> None:
        """save_annotations writes an image's first label document."""
        import tcip_store
        from tests._producer_fixtures import image_label_key, write_image

        img_path = write_image(
            image_dir(tmp_path, UNDATED_BUCKET) / "new_img.jpg", (640, 480)
        )

        result = save_annotations(
            bound,
            str(img_path),
            annotations=[
                {"subject": "bud", "bbox": [10, 10, 50, 50]},
            ],
            actor=None,
        )
        assert result["count"] == 1
        assert tcip_store.exists(image_label_key(img_path))
