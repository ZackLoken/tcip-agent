"""Integration tests for MCP tool functions.

Tests tool functions directly (not through MCP server protocol) to verify end-to-end behavior
over real images and the canonical per-image label documents.
"""

from __future__ import annotations

from tcip_mcp.dataset_layout import UNDATED_BUCKET

from pathlib import Path

import pytest
from PIL import Image

from tests.conftest import DATA_DIR_BUCKET


DATE = "2-11-26"
"""The capture date conftest's ``data_dir`` lays its images and its published bucket under."""


def _empty_bucket(project: Path, *images: Path) -> str:
    """A bucket ``m/2024-01-01`` that predicted nothing on each of ``images`` (8 by 8 px), under
    their dataset root; its name."""
    pytest.importorskip("torch")
    from tests._chain_fixtures import published

    return published(project, "m/2024-01-01", [
        {"image": str(image), "width": 8, "height": 8, "boxes": [], "scores": [], "labels": [],
         "cap": 1} for image in images],
        scope={"subject": "bud"}).name


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
        from tcip_annotation.state import Annotation, Polygon
        from tcip_mcp.tools.annotation_tools import read_annotations
        from tests._producer_fixtures import label_image

        images_dir = tmp_path / "images" / UNDATED_BUCKET
        images_dir.mkdir(parents=True)
        Image.new("RGB", (100, 100)).save(images_dir / "a.jpg")
        label_image(images_dir / "a.jpg", [Annotation(subject="bud", geometry=Polygon([
            [(10.0, 10.0), (30.0, 10.0), (30.0, 30.0)],
            [(60.0, 10.0), (80.0, 10.0), (80.0, 30.0)],
        ]))], 100, 100)

        result = read_annotations(str(images_dir / "a.jpg"))
        (ann,) = result["labels"]["annotations"]
        assert "points" not in ann
        assert ann["rings"] == [[[10.0, 10.0], [30.0, 10.0], [30.0, 30.0]],
                                [[60.0, 10.0], [80.0, 10.0], [80.0, 30.0]]]


# ── Evaluate predictions integration test ───────────────────────────────────


class TestEvaluatePredictions:
    """Test score_predictions with actual file I/O."""

    def test_evaluate_single_image(self, data_dir: Path):
        from tcip_mcp.tools.annotation_tools import score_predictions

        img = str(data_dir / "images" / DATE / "img_001.jpg")
        result = score_predictions(img, DATA_DIR_BUCKET, iou_threshold=0.5, conf_threshold=0.25)
        assert "error" not in result
        assert result["tp"] >= 0
        assert result["fp"] >= 0
        assert result["fn"] >= 0
        assert 0.0 <= result["precision"] <= 1.0
        assert 0.0 <= result["recall"] <= 1.0

    def test_evaluate_folder(self, data_dir: Path):
        from tcip_mcp.tools.annotation_tools import score_predictions

        result = score_predictions(str(data_dir / "images" / DATE), DATA_DIR_BUCKET,
                                   iou_threshold=0.5)
        assert result["image_count"] == 3
        assert "precision" in result
        assert "recall" in result


# ── Detail-mode matching integration test ───────────────────────────────────


class TestEvaluatePredictionsDetail:
    """Test score_predictions(detail=True) per-detection breakdown with actual file I/O."""

    def test_detail_breakdown(self, data_dir: Path):
        from tcip_mcp.tools.annotation_tools import score_predictions

        img = str(data_dir / "images" / DATE / "img_001.jpg")
        result = score_predictions(img, DATA_DIR_BUCKET, iou_threshold=0.5, detail=True)
        assert "error" not in result
        assert "detections" in result


# ── score_predictions(images directory) enumeration ────


def _write_empty_label(image: Path, w: int, h: int) -> None:
    from tests._producer_fixtures import label_image

    label_image(image, [], w, h, keep_empty=True)


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
        _write_empty_label(images_dir / "cap.bandgroup", 8, 8)

        bucket = _empty_bucket(tmp_path, images_dir / "cap.bandgroup")
        result = score_predictions(str(images_dir), bucket, iou_threshold=0.5)
        assert result["image_count"] == 1
        assert [row["image"] for row in result["per_image"]] == ["cap.bandgroup"]

    def test_an_npz_capture_scores(self, tmp_path: Path):
        import numpy as np

        from tcip_mcp.tools.annotation_tools import score_predictions

        images_dir = tmp_path / "images" / "2024-01-01"
        images_dir.mkdir(parents=True)
        np.savez(str(images_dir / "cap.npz"), bands=np.zeros((8, 8, 3), dtype=np.uint16))
        _write_empty_label(images_dir / "cap.npz", 8, 8)

        bucket = _empty_bucket(tmp_path, images_dir / "cap.npz")
        result = score_predictions(str(images_dir), bucket, iou_threshold=0.5)
        assert result["image_count"] == 1
        assert [row["image"] for row in result["per_image"]] == ["cap.npz"]

    def test_a_folder_nested_inside_the_images_directory_is_not_scored(self, tmp_path: Path):
        from tcip_mcp.tools.annotation_tools import score_predictions

        images_dir = tmp_path / "images" / "2024-01-01"
        nested = images_dir / "nested"
        nested.mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(nested / "inner.jpg")
        elsewhere = tmp_path / "images" / "2024-01-02" / "other.jpg"
        elsewhere.parent.mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(elsewhere)

        result = score_predictions(str(images_dir), _empty_bucket(tmp_path, elsewhere),
                                   iou_threshold=0.5)
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
            "horizontal_flip": {"p": 0.5},
            "vertical_flip": {"p": 0.3},
            "color_jitter": {"brightness": 0.3, "contrast": 0.3, "saturation": 0.0},
            "gaussian_blur": {"p": 0.1, "radius": 2.0},
        }
        transforms = build_augmentation(config)
        # 4 augmentations + ToTensor
        assert len(transforms.transforms) == 5

    def test_augmentation_preserves_detection_target(self):
        import torch
        from tcip_mcp.pipelines.data.augmentations import build_augmentation

        config = {"horizontal_flip": {"p": 1.0}}  # always flip
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

        config = {"color_jitter": {"brightness": 0.1, "contrast": 0.0, "saturation": 0.0}}
        transforms = build_augmentation(config)

        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        target = {"labels": 3}
        out_img, out_target = transforms(img, target)
        assert isinstance(out_img, torch.Tensor)
        assert out_target["labels"] == 3


class TestReadAnnotationsUnknownFormat:
    def test_unrecognized_store_returns_error_not_raise(self, tmp_path):
        """The per-image reader refuses a record of a shape it does not read; read_annotations
        surfaces that as an error dict, matching its own convention and the docs, not an
        uncaught error."""
        import tcip_store
        from PIL import Image

        from tcip_mcp.tools.annotation_tools import read_annotations
        from tests._producer_fixtures import image_label_key

        (tmp_path / "images" / UNDATED_BUCKET).mkdir(parents=True)
        Image.new("RGB", (32, 32)).save(tmp_path / "images" / UNDATED_BUCKET / "a.jpg")
        # A record of a schema this platform does not read, stored past the writer.
        tcip_store.replace(
            image_label_key(tmp_path / "images" / UNDATED_BUCKET / "a.jpg"), {"regions": []})

        result = read_annotations(str(tmp_path / "images" / UNDATED_BUCKET / "a.jpg"))
        assert "error" in result
        assert "not the list a label document holds" in result["error"]
