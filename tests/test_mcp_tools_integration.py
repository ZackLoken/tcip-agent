"""Integration tests for MCP tool functions.

Tests tool functions directly (not through MCP server protocol) to verify end-to-end behavior
over real images and the canonical per-image label documents.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from PIL import Image

from tcip_mcp.dataset_layout import UNDATED_BUCKET
from tests._training_values import augmentation
from tests.conftest import DATA_DIR_BUCKET

if TYPE_CHECKING:
    from tcip_mcp.pipelines.data.augmentations import Compose

DATE = "2-11-26"
"""The capture date conftest's ``data_dir`` lays its images and its published bucket under."""


def _empty_bucket(project: Path, *images: Path) -> str:
    """Publish a bucket ``m/2024-01-01`` that predicted nothing and return its name.

    The bucket holds an empty prediction for each of ``images`` (8 by 8 px), under their dataset
    root.
    """
    pytest.importorskip("torch")
    from tests._chain_fixtures import published

    return published(
        project,
        "m/2024-01-01",
        [
            {
                "image": str(image),
                "width": 8,
                "height": 8,
                "boxes": [],
                "scores": [],
                "labels": [],
                "cap": 1,
            }
            for image in images
        ],
        scope={"subject": "bud"},
    ).name


# ── Annotation tool integration tests ───────────────────────────────────────


class TestReadAnnotations:
    def test_load_missing_image(self, tmp_path: Path) -> None:
        from tcip_mcp.tools.annotation_tools import read_annotations

        result = read_annotations(tmp_path, "/nonexistent/image.jpg")
        assert "error" in result

    def test_polygon_is_reported_as_rings_including_every_contour(
        self, tmp_path: Path
    ) -> None:
        """The tool response represents a two-contour polygon as ``rings``, both of them."""
        from tcip_annotation.state import Annotation, Polygon
        from tcip_mcp.tools.annotation_tools import read_annotations
        from tests._producer_fixtures import label_image

        images_dir = tmp_path / "images" / UNDATED_BUCKET
        images_dir.mkdir(parents=True)
        Image.new("RGB", (100, 100)).save(images_dir / "a.jpg")
        label_image(
            images_dir / "a.jpg",
            [
                Annotation(
                    subject="bud",
                    geometry=Polygon(
                        [
                            [(10.0, 10.0), (30.0, 10.0), (30.0, 30.0)],
                            [(60.0, 10.0), (80.0, 10.0), (80.0, 30.0)],
                        ]
                    ),
                )
            ],
            100,
            100,
        )

        result = read_annotations(tmp_path, str(images_dir / "a.jpg"))
        (ann,) = result["labels"]["annotations"]
        assert "points" not in ann
        assert ann["rings"] == [
            [[10.0, 10.0], [30.0, 10.0], [30.0, 30.0]],
            [[60.0, 10.0], [80.0, 10.0], [80.0, 30.0]],
        ]


# ── Evaluate predictions integration test ───────────────────────────────────


class TestEvaluatePredictions:
    """Tests of score_predictions over files on disk."""

    def test_evaluate_single_image(self, data_dir: Path) -> None:
        from tcip_mcp.tools.annotation_tools import score_predictions

        img = str(data_dir / "images" / DATE / "img_001.jpg")
        result = score_predictions(
            data_dir,
            img,
            DATA_DIR_BUCKET,
            iou_threshold=0.5,
            conf_threshold=0.25,
        )
        assert "error" not in result
        assert result["tp"] >= 0
        assert result["fp"] >= 0
        assert result["fn"] >= 0
        assert 0.0 <= result["precision"] <= 1.0
        assert 0.0 <= result["recall"] <= 1.0

    def test_evaluate_folder(self, data_dir: Path) -> None:
        from tcip_mcp.tools.annotation_tools import score_predictions

        result = score_predictions(
            data_dir,
            str(data_dir / "images" / DATE),
            DATA_DIR_BUCKET,
            iou_threshold=0.5,
            conf_threshold=0.25,
        )
        assert result["image_count"] == 3
        assert "precision" in result
        assert "recall" in result


# ── Detail-mode matching integration test ───────────────────────────────────


class TestEvaluatePredictionsDetail:
    """Tests of score_predictions(detail=True)'s per-detection breakdown over files on disk."""

    def test_detail_breakdown(self, data_dir: Path) -> None:
        from tcip_mcp.tools.annotation_tools import score_predictions

        img = str(data_dir / "images" / DATE / "img_001.jpg")
        result = score_predictions(
            data_dir,
            img,
            DATA_DIR_BUCKET,
            iou_threshold=0.5,
            conf_threshold=0.25,
            detail=True,
        )
        assert "error" not in result
        assert "detections" in result


# ── score_predictions(images directory) enumeration ────


def _write_empty_label(image: Path, w: int, h: int) -> None:
    from tests._producer_fixtures import label_image

    label_image(image, [], w, h, keep_empty=True)


class TestEvaluateFolderEnumeration:
    def test_a_band_grouped_capture_scores_as_one_logical_image(
        self, tmp_path: Path
    ) -> None:
        import numpy as np
        import tifffile

        from tcip_mcp.pipelines.data.band_groups import (
            write_band_group_manifest,
        )
        from tcip_mcp.tools.annotation_tools import score_predictions

        images_dir = tmp_path / "images" / "2024-01-01"
        images_dir.mkdir(parents=True)
        band_a, band_b = images_dir / "cap_G.tif", images_dir / "cap_R.tif"
        tifffile.imwrite(str(band_a), np.full((8, 8), 111, dtype=np.uint16))
        tifffile.imwrite(str(band_b), np.full((8, 8), 222, dtype=np.uint16))
        write_band_group_manifest(
            images_dir,
            "cap",
            {"Green": band_a, "Red": band_b},
            central_wavelength_nm=None,
            identity_tolerance={},
        )
        _write_empty_label(images_dir / "cap.bandgroup", 8, 8)

        bucket = _empty_bucket(tmp_path, images_dir / "cap.bandgroup")
        result = score_predictions(
            tmp_path,
            str(images_dir),
            bucket,
            iou_threshold=0.5,
            conf_threshold=0.25,
        )
        assert result["image_count"] == 1
        assert [row["image"] for row in result["per_image"]] == [
            "cap.bandgroup"
        ]

    def test_an_npz_capture_scores(self, tmp_path: Path) -> None:
        import numpy as np

        from tcip_mcp.tools.annotation_tools import score_predictions

        images_dir = tmp_path / "images" / "2024-01-01"
        images_dir.mkdir(parents=True)
        np.savez(
            str(images_dir / "cap.npz"),
            bands=np.zeros((8, 8, 3), dtype=np.uint16),
        )
        _write_empty_label(images_dir / "cap.npz", 8, 8)

        bucket = _empty_bucket(tmp_path, images_dir / "cap.npz")
        result = score_predictions(
            tmp_path,
            str(images_dir),
            bucket,
            iou_threshold=0.5,
            conf_threshold=0.25,
        )
        assert result["image_count"] == 1
        assert [row["image"] for row in result["per_image"]] == ["cap.npz"]

    def test_a_folder_nested_inside_the_images_directory_is_not_scored(
        self, tmp_path: Path
    ) -> None:
        from tcip_mcp.tools.annotation_tools import score_predictions

        images_dir = tmp_path / "images" / "2024-01-01"
        nested = images_dir / "nested"
        nested.mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(nested / "inner.jpg")
        elsewhere = tmp_path / "images" / "2024-01-02" / "other.jpg"
        elsewhere.parent.mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(elsewhere)

        result = score_predictions(
            tmp_path,
            str(images_dir),
            _empty_bucket(tmp_path, elsewhere),
            iou_threshold=0.5,
            conf_threshold=0.25,
        )
        assert result["image_count"] == 0


# ── Augmentation tests ──────────────────────────────────────────────────────


def _chain(**transforms: object) -> Compose:
    """Validate an ``augmentation`` block of ``transforms`` and return the chain it builds."""
    from tcip_mcp.pipelines.data.augmentations import build_augmentation
    from tcip_mcp.pipelines.schemas import AugmentationSpec

    return build_augmentation(
        AugmentationSpec.model_validate(augmentation(**transforms))
    )


class TestAugmentations:
    """Tests of the data augmentation pipeline."""

    def test_build_empty_augmentation(self) -> None:
        transforms = _chain()
        assert len(transforms.transforms) == 0

    def test_build_full_augmentation(self) -> None:
        transforms = _chain(
            horizontal_flip={"p": 0.5},
            vertical_flip={"p": 0.3},
            color_jitter={
                "brightness": 0.3,
                "contrast": 0.3,
                "saturation": 0.0,
            },
            gaussian_blur={"p": 0.1, "radius": 2.0},
        )
        assert len(transforms.transforms) == 4

    def test_augmentation_preserves_detection_target(self) -> None:
        import torch

        transforms = _chain(horizontal_flip={"p": 1.0})  # always flip

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

    def test_augmentation_classification(self) -> None:
        import torch

        transforms = _chain(
            color_jitter={
                "brightness": 0.1,
                "contrast": 0.0,
                "saturation": 0.0,
            }
        )

        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        target = {"labels": 3}
        out_img, out_target = transforms(img, target)
        assert isinstance(out_img, torch.Tensor)
        assert out_target["labels"] == 3


class TestReadAnnotationsUnknownFormat:
    def test_unrecognized_store_returns_error_not_raise(
        self, tmp_path: Path
    ) -> None:
        """``read_annotations`` answers an unreadable record's refusal as an error dict.

        The per-image reader refuses a record of a shape it does not read; read_annotations
        surfaces that as an error dict naming the problem, not an uncaught error.
        """
        from PIL import Image

        import tcip_store
        from tcip_mcp.tools.annotation_tools import read_annotations
        from tests._producer_fixtures import image_label_key

        (tmp_path / "images" / UNDATED_BUCKET).mkdir(parents=True)
        Image.new("RGB", (32, 32)).save(
            tmp_path / "images" / UNDATED_BUCKET / "a.jpg"
        )
        # A record of a schema this platform does not read, stored past the writer.
        tcip_store.replace(
            image_label_key(tmp_path / "images" / UNDATED_BUCKET / "a.jpg"),
            {"regions": []},
        )

        result = read_annotations(
            tmp_path, str(tmp_path / "images" / UNDATED_BUCKET / "a.jpg")
        )
        assert "error" in result
        assert "not the list a label document holds" in result["error"]
