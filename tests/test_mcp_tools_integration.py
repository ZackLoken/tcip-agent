"""Integration tests for MCP tool functions.

Tests tool functions directly (not through MCP server protocol) to verify
end-to-end behavior with actual file I/O and annotation parsing on the canonical
per-image JSON labels.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def json_dataset(tmp_path: Path) -> Path:
    """A minimal dataset in the canonical name-based per-image JSON layout."""
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    labels_dir = tmp_path / "annotations"
    labels_dir.mkdir(parents=True)
    preds_dir = tmp_path / "predictions" / "live"
    preds_dir.mkdir(parents=True)

    for name in ("img_001", "img_002", "img_003"):
        Image.new("RGB", (640, 480), color=(128, 128, 128)).save(images_dir / f"{name}.jpg")
        json_io.write_annotations(
            str(labels_dir / f"{name}.json"),
            [Annotation(subject="bud", geometry=BBox(288, 216, 352, 264)),
             Annotation(subject="bud", geometry=BBox(176, 132, 208, 156))], 640, 480)
        json_io.write_annotations(
            str(preds_dir / f"{name}.json"),
            [Annotation(subject="bud", geometry=BBox(288, 216, 352, 264), score=0.9),
             Annotation(subject="bud", geometry=BBox(496, 372, 528, 396), score=0.7)], 640, 480)
    return tmp_path


# ── Annotation tool integration tests ───────────────────────────────────────


class TestReadAnnotations:
    def test_load_missing_image(self):
        from tcip_mcp.tools.annotation_tools import read_annotations

        result = read_annotations("/nonexistent/image.jpg")
        assert "error" in result

    def test_polygon_is_reported_as_rings_including_every_contour(self, tmp_path):
        """The tool response represents a polygon as ``rings``, all of them.

        A stored annotation can be occlusion-split (one instance, several contours), so reporting a
        single flat point list would silently hide part of the object from the agent reading it.
        """
        from tcip_annotation import json_io
        from tcip_annotation.state import Annotation, Polygon
        from tcip_mcp.tools.annotation_tools import read_annotations

        images_dir, labels_dir = tmp_path / "images", tmp_path / "annotations"
        images_dir.mkdir()
        labels_dir.mkdir()
        Image.new("RGB", (100, 100)).save(images_dir / "a.jpg")
        json_io.write_annotations(
            str(labels_dir / "a.json"),
            [Annotation(subject="bud", geometry=Polygon([
                [(10.0, 10.0), (30.0, 10.0), (30.0, 30.0)],
                [(60.0, 10.0), (80.0, 10.0), (80.0, 30.0)],
            ]))], 100, 100)

        result = read_annotations(str(images_dir / "a.jpg"))
        (ann,) = result["labels"]["annotations"]
        assert "points" not in ann
        assert ann["rings"] == [[[10.0, 10.0], [30.0, 10.0], [30.0, 30.0]],
                                [[60.0, 10.0], [80.0, 10.0], [80.0, 30.0]]]

    def test_forcing_fmt_coco_over_a_per_image_document_is_an_error_not_a_silent_zero(self, tmp_path):
        """A caller-supplied fmt is a claim the document must satisfy: forcing 'coco' over the
        canonical per-image schema must not answer an empty, error-free read."""
        from tcip_annotation import json_io
        from tcip_annotation.state import Annotation, BBox
        from tcip_mcp.tools.annotation_tools import read_annotations

        images_dir, labels_dir = tmp_path / "images", tmp_path / "annotations"
        images_dir.mkdir()
        labels_dir.mkdir()
        Image.new("RGB", (100, 100)).save(images_dir / "a.jpg")
        json_io.write_annotations(
            str(labels_dir / "a.json"),
            [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], 100, 100)

        result = read_annotations(str(images_dir / "a.jpg"), fmt="coco")
        assert "error" in result
        assert "labels" not in result

    def test_forcing_fmt_coco_over_a_labelme_shaped_document_is_an_error_not_a_silent_zero(
        self, tmp_path,
    ):
        """A document carrying neither the per-image nor the COCO shape's markers satisfies no
        stated fmt: it must be refused, not read as a store with zero annotations."""
        from tcip_mcp.tools.annotation_tools import read_annotations

        images_dir, labels_dir = tmp_path / "images", tmp_path / "annotations"
        images_dir.mkdir()
        labels_dir.mkdir()
        Image.new("RGB", (100, 100)).save(images_dir / "a.jpg")
        (labels_dir / "a.json").write_text(
            '{"shapes": [{"label": "bud", "points": [[1, 1], [8, 8]]}]}'
        )

        result = read_annotations(str(images_dir / "a.jpg"), fmt="coco")
        assert "error" in result
        assert "labels" not in result

    def test_forcing_fmt_coco_over_an_old_objects_schema_document_is_an_error_not_a_silent_zero(
        self, tmp_path,
    ):
        """The old 'objects'-keyed schema carries neither current shape's markers either, and
        must be refused the same way, never read in place as an empty COCO store."""
        from tcip_mcp.tools.annotation_tools import read_annotations

        images_dir, labels_dir = tmp_path / "images", tmp_path / "annotations"
        images_dir.mkdir()
        labels_dir.mkdir()
        Image.new("RGB", (100, 100)).save(images_dir / "a.jpg")
        (labels_dir / "a.json").write_text(
            '{"image": "a", "objects": [{"category_id": 0, "bbox": [1, 1, 9, 9]}]}'
        )

        result = read_annotations(str(images_dir / "a.jpg"), fmt="coco")
        assert "error" in result
        assert "labels" not in result


# ── Evaluate predictions integration test ───────────────────────────────────


class TestEvaluatePredictions:
    """Test score_predictions with actual file I/O."""

    def test_evaluate_single_image(self, json_dataset: Path):
        from tcip_mcp.tools.annotation_tools import score_predictions

        img = str(json_dataset / "images" / "img_001.jpg")
        result = score_predictions(img, iou_threshold=0.5, conf_threshold=0.25)
        assert "error" not in result
        assert result["tp"] >= 0
        assert result["fp"] >= 0
        assert result["fn"] >= 0
        assert 0.0 <= result["precision"] <= 1.0
        assert 0.0 <= result["recall"] <= 1.0

    def test_evaluate_folder(self, json_dataset: Path):
        from tcip_mcp.tools.annotation_tools import score_predictions

        result = score_predictions(str(json_dataset), iou_threshold=0.5)
        assert result["image_count"] == 3
        assert "precision" in result
        assert "recall" in result


# ── Detail-mode matching integration test ───────────────────────────────────


class TestEvaluatePredictionsDetail:
    """Test score_predictions(detail=True) per-detection breakdown with actual file I/O."""

    def test_detail_breakdown(self, json_dataset: Path):
        from tcip_mcp.tools.annotation_tools import score_predictions

        img = str(json_dataset / "images" / "img_001.jpg")
        result = score_predictions(img, iou_threshold=0.5, detail=True)
        assert "error" not in result
        assert "detections" in result


# ── score_predictions(dataset) enumeration: cumulative over the layout, one bucket level ────


def _write_empty_label(path: Path, w: int, h: int) -> None:
    from tcip_annotation import json_io

    path.parent.mkdir(parents=True, exist_ok=True)
    json_io.write_annotations(str(path), [], w, h, keep_empty=True)


class TestEvaluateFolderEnumeration:
    def test_a_band_grouped_capture_scores_as_one_logical_image(self, tmp_path: Path):
        import numpy as np
        import tifffile

        from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest
        from tcip_mcp.tools.annotation_tools import score_predictions

        images_dir = tmp_path / "images" / "2024-01-01"
        images_dir.mkdir(parents=True)
        band_a, band_b = images_dir / "cap_G.tif", images_dir / "cap_R.tif"
        tifffile.imwrite(str(band_a), np.full((8, 8), 111, dtype=np.uint16))
        tifffile.imwrite(str(band_b), np.full((8, 8), 222, dtype=np.uint16))
        write_band_group_manifest(images_dir, "cap", {"Green": band_a, "Red": band_b})
        _write_empty_label(tmp_path / "annotations" / "2024-01-01" / "cap.json", 8, 8)

        result = score_predictions(str(tmp_path), iou_threshold=0.5)
        assert result["image_count"] == 1
        assert [row["image"] for row in result["per_image"]] == ["cap.bandgroup"]

    def test_a_loose_image_beside_a_dated_bucket_still_scores(self, tmp_path: Path):
        from tcip_mcp.tools.annotation_tools import score_predictions

        images_dir = tmp_path / "images"
        (images_dir / "2024-01-01").mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(images_dir / "2024-01-01" / "bucketed.jpg")
        Image.new("RGB", (8, 8)).save(images_dir / "loose.jpg")
        _write_empty_label(tmp_path / "annotations" / "2024-01-01" / "bucketed.json", 8, 8)
        _write_empty_label(tmp_path / "annotations" / "loose.json", 8, 8)

        result = score_predictions(str(tmp_path), iou_threshold=0.5)
        assert result["image_count"] == 2
        assert {row["image"] for row in result["per_image"]} == {"bucketed.jpg", "loose.jpg"}

    def test_an_npz_capture_scores(self, tmp_path: Path):
        import numpy as np

        from tcip_mcp.tools.annotation_tools import score_predictions

        images_dir = tmp_path / "images" / "2024-01-01"
        images_dir.mkdir(parents=True)
        np.savez(str(images_dir / "cap.npz"), bands=np.zeros((8, 8, 3), dtype=np.uint16))
        _write_empty_label(tmp_path / "annotations" / "2024-01-01" / "cap.json", 8, 8)

        result = score_predictions(str(tmp_path), iou_threshold=0.5)
        assert result["image_count"] == 1
        assert [row["image"] for row in result["per_image"]] == ["cap.npz"]

    def test_a_folder_nested_inside_a_bucket_is_not_scored(self, tmp_path: Path):
        """``images/<bucket>/`` is the layout; a folder inside a bucket is not itself one, so it is
        not descended into."""
        from tcip_mcp.tools.annotation_tools import score_predictions

        nested = tmp_path / "images" / "2024-01-01" / "nested"
        nested.mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(nested / "inner.jpg")

        result = score_predictions(str(tmp_path), iou_threshold=0.5)
        assert result["image_count"] == 0


# ── Augmentation tests ──────────────────────────────────────────────────────


class TestAugmentations:
    """Test data augmentation pipeline."""

    def test_build_empty_augmentation(self):
        from tcip_mcp.pipelines.data.augmentations import build_augmentation

        transforms = build_augmentation({})
        # Should just have ToTensor
        assert len(transforms.transforms) == 1

    def test_build_full_augmentation(self):
        from tcip_mcp.pipelines.data.augmentations import build_augmentation

        config = {
            "horizontal_flip": 0.5,
            "vertical_flip": 0.3,
            "color_jitter": {"brightness": 0.3, "contrast": 0.3},
            "gaussian_blur": 0.1,
        }
        transforms = build_augmentation(config)
        # 4 augmentations + ToTensor
        assert len(transforms.transforms) == 5

    def test_augmentation_preserves_detection_target(self):
        import torch
        from tcip_mcp.pipelines.data.augmentations import build_augmentation

        config = {"horizontal_flip": 1.0}  # always flip
        transforms = build_augmentation(config)

        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        target = {
            "boxes": torch.tensor([[10, 20, 30, 40]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
        }
        out_img, out_target = transforms(img, target)
        assert isinstance(out_img, torch.Tensor)
        assert out_target["boxes"].shape == (1, 4)
        # Flipped: x1 should become 100 - 30 = 70, x2 = 100 - 10 = 90
        assert out_target["boxes"][0, 0].item() == pytest.approx(70.0)
        assert out_target["boxes"][0, 2].item() == pytest.approx(90.0)

    def test_augmentation_classification(self):
        import torch
        from tcip_mcp.pipelines.data.augmentations import build_augmentation

        config = {"color_jitter": {"brightness": 0.1}}
        transforms = build_augmentation(config)

        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        target = {"labels": 3}
        out_img, out_target = transforms(img, target)
        assert isinstance(out_img, torch.Tensor)
        assert out_target["labels"] == 3


class TestReadAnnotationsUnknownFormat:
    def test_unrecognized_store_returns_error_not_raise(self, tmp_path):
        """detect_format refuses an unknown store; read_annotations must surface that as an error
        dict, matching its own convention and the docs, not propagate an uncaught ValueError."""
        from PIL import Image

        from tcip_mcp.tools.annotation_tools import read_annotations

        det = tmp_path / "annotations"
        det.mkdir(parents=True)
        (tmp_path / "images").mkdir()
        Image.new("RGB", (32, 32)).save(tmp_path / "images" / "a.jpg")
        (det / "a.json").write_text('{"regions": []}')  # a schema we do not recognize

        result = read_annotations(str(tmp_path / "images" / "a.jpg"))
        assert "error" in result
        assert "Cannot determine the annotation format" in result["error"]
