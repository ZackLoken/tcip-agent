"""Integration tests for MCP tool functions.

Tests tool functions directly (not through MCP server protocol) to verify
end-to-end behavior with actual file I/O, annotation parsing, and format
detection across YOLO, VOC, and LabelMe formats.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def yolo_dataset(tmp_path: Path) -> Path:
    """Create a minimal YOLO-format dataset."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    labels_dir = tmp_path / "labels" / "detect"
    labels_dir.mkdir(parents=True)
    preds_dir = tmp_path / "predictions" / "detect"
    preds_dir.mkdir(parents=True)

    for name in ("img_001", "img_002", "img_003"):
        img = Image.new("RGB", (640, 480), color=(128, 128, 128))
        img.save(images_dir / f"{name}.jpg")
        (labels_dir / f"{name}.txt").write_text("0 0.5 0.5 0.1 0.1\n0 0.3 0.3 0.05 0.05\n")
        (preds_dir / f"{name}.txt").write_text("0 0.9 0.5 0.5 0.1 0.1\n0 0.7 0.8 0.8 0.05 0.05\n")

    return tmp_path


@pytest.fixture
def voc_dataset(tmp_path: Path) -> Path:
    """Create a minimal PASCAL VOC-format dataset."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    labels_dir = tmp_path / "labels" / "detect"
    labels_dir.mkdir(parents=True)

    for name in ("img_001", "img_002"):
        img = Image.new("RGB", (640, 480), color=(128, 128, 128))
        img.save(images_dir / f"{name}.jpg")

        xml_content = f"""<annotation>
  <filename>{name}.jpg</filename>
  <size><width>640</width><height>480</height><depth>3</depth></size>
  <object>
    <name>catkin</name>
    <bndbox><xmin>100</xmin><ymin>100</ymin><xmax>200</xmax><ymax>200</ymax></bndbox>
  </object>
  <object>
    <name>bud</name>
    <bndbox><xmin>300</xmin><ymin>300</ymin><xmax>400</xmax><ymax>400</ymax></bndbox>
  </object>
</annotation>"""
        (labels_dir / f"{name}.xml").write_text(xml_content)

    return tmp_path


@pytest.fixture
def labelme_dataset(tmp_path: Path) -> Path:
    """Create a minimal LabelMe-format dataset."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    labels_dir = tmp_path / "labels" / "detect"
    labels_dir.mkdir(parents=True)

    for name in ("img_001", "img_002"):
        img = Image.new("RGB", (640, 480), color=(128, 128, 128))
        img.save(images_dir / f"{name}.jpg")

        labelme_data = {
            "version": "5.0.0",
            "flags": {},
            "shapes": [
                {
                    "label": "catkin",
                    "points": [[100, 100], [200, 200]],
                    "shape_type": "rectangle",
                    "flags": {},
                },
                {
                    "label": "bud",
                    "points": [[300, 300], [400, 400]],
                    "shape_type": "rectangle",
                    "flags": {},
                },
            ],
            "imagePath": f"{name}.jpg",
            "imageHeight": 480,
            "imageWidth": 640,
            "imageData": None,
        }
        (labels_dir / f"{name}.json").write_text(json.dumps(labelme_data))

    return tmp_path


# ── Data tool integration tests ─────────────────────────────────────────────


class TestLoadDatasetMultiFormat:
    """Test load_dataset across annotation formats."""

    def test_load_yolo_dataset(self, yolo_dataset: Path):
        from tcip_mcp.tools.data_tools import load_dataset

        result = load_dataset(str(yolo_dataset))
        assert result["image_count"] == 3
        assert result["labels_detect_count"] == 3
        assert result["format"] == "yolo"
        assert result["paired_images"] == 3

    def test_load_voc_dataset(self, voc_dataset: Path):
        from tcip_mcp.tools.data_tools import load_dataset

        result = load_dataset(str(voc_dataset))
        assert result["image_count"] == 2
        assert result["labels_detect_count"] == 2
        assert result["format"] == "voc"

    def test_load_labelme_dataset(self, labelme_dataset: Path):
        from tcip_mcp.tools.data_tools import load_dataset

        result = load_dataset(str(labelme_dataset))
        assert result["image_count"] == 2
        assert result["labels_detect_count"] == 2
        assert result["format"] == "labelme"


class TestValidateDataQualityMultiFormat:
    """Test validate_data_quality across formats."""

    def test_validate_yolo(self, yolo_dataset: Path):
        from tcip_mcp.tools.data_tools import validate_data_quality

        result = validate_data_quality(str(yolo_dataset))
        assert result["is_valid"] is True
        assert result["format"] == "yolo"
        assert 0 in result["class_ids"]

    def test_validate_voc(self, voc_dataset: Path):
        from tcip_mcp.tools.data_tools import validate_data_quality

        result = validate_data_quality(str(voc_dataset))
        assert result["is_valid"] is True
        assert result["format"] == "voc"
        assert len(result["class_ids"]) == 2  # catkin=0, bud=1 (alphabetical)

    def test_validate_labelme(self, labelme_dataset: Path):
        from tcip_mcp.tools.data_tools import validate_data_quality

        result = validate_data_quality(str(labelme_dataset))
        assert result["is_valid"] is True
        assert result["format"] == "labelme"
        assert len(result["class_ids"]) == 2


# ── Annotation tool integration tests ───────────────────────────────────────


class TestLoadAnnotationsMultiFormat:
    """Test load_annotations across formats."""

    def test_load_yolo_annotations(self, yolo_dataset: Path):
        from tcip_mcp.tools.annotation_tools import load_annotations

        img = str(yolo_dataset / "images" / "img_001.jpg")
        result = load_annotations(img)
        assert "detect_labels" in result
        assert result["detect_labels"]["count"] == 2
        assert result["detect_labels"]["format"] == "yolo"

    def test_load_voc_annotations(self, voc_dataset: Path):
        from tcip_mcp.tools.annotation_tools import load_annotations

        img = str(voc_dataset / "images" / "img_001.jpg")
        result = load_annotations(img)
        assert "detect_labels" in result
        assert result["detect_labels"]["count"] == 2
        assert result["detect_labels"]["format"] == "voc"

    def test_load_labelme_annotations(self, labelme_dataset: Path):
        from tcip_mcp.tools.annotation_tools import load_annotations

        img = str(labelme_dataset / "images" / "img_001.jpg")
        result = load_annotations(img)
        assert "detect_labels" in result
        assert result["detect_labels"]["count"] == 2
        assert result["detect_labels"]["format"] == "labelme"

    def test_load_missing_image(self):
        from tcip_mcp.tools.annotation_tools import load_annotations

        result = load_annotations("/nonexistent/image.jpg")
        assert "error" in result


class TestSaveAnnotationsMultiFormat:
    """Test save_annotations across formats."""

    def test_save_yolo(self, yolo_dataset: Path):
        from tcip_mcp.tools.annotation_tools import save_annotations

        img = str(yolo_dataset / "images" / "img_001.jpg")
        boxes = [{"x1": 100, "y1": 100, "x2": 200, "y2": 200, "class_id": 0}]
        result = save_annotations(img, boxes=boxes, fmt="yolo")
        assert result["count"] == 1
        assert result["format"] == "yolo"
        assert result["written"][0].endswith(".txt")

    def test_save_voc(self, yolo_dataset: Path):
        from tcip_mcp.tools.annotation_tools import save_annotations

        img = str(yolo_dataset / "images" / "img_001.jpg")
        boxes = [{"x1": 100, "y1": 100, "x2": 200, "y2": 200, "class_id": 0}]
        result = save_annotations(img, boxes=boxes, fmt="voc")
        assert result["count"] == 1
        assert result["format"] == "voc"
        assert result["written"][0].endswith(".xml")
        # Verify the XML is valid
        import xml.etree.ElementTree as ET
        tree = ET.parse(result["written"][0])
        assert tree.getroot().tag == "annotation"

    def test_save_labelme(self, yolo_dataset: Path):
        from tcip_mcp.tools.annotation_tools import save_annotations

        img = str(yolo_dataset / "images" / "img_001.jpg")
        boxes = [{"x1": 100, "y1": 100, "x2": 200, "y2": 200, "class_id": 0}]
        result = save_annotations(img, boxes=boxes, fmt="labelme")
        assert result["count"] == 1
        assert result["format"] == "labelme"
        assert result["written"][0].endswith(".json")
        # Verify the JSON is valid LabelMe
        data = json.loads(Path(result["written"][0]).read_text())
        assert "shapes" in data
        assert len(data["shapes"]) == 1

    def test_save_roundtrip_voc(self, yolo_dataset: Path):
        """Save as VOC, then load back and verify."""
        from tcip_mcp.tools.annotation_tools import save_annotations, load_annotations

        img_path = str(yolo_dataset / "images" / "img_001.jpg")
        boxes = [
            {"x1": 100, "y1": 100, "x2": 200, "y2": 200, "class_id": 0},
            {"x1": 300, "y1": 300, "x2": 400, "y2": 400, "class_id": 1},
        ]
        save_result = save_annotations(img_path, boxes=boxes, fmt="voc")
        assert save_result["count"] == 1

        load_result = load_annotations(img_path, fmt="voc")
        assert load_result["detect_labels"]["count"] == 2


# ── Evaluate detections integration test ────────────────────────────────────


class TestEvaluateDetections:
    """Test evaluate_detections with actual file I/O."""

    def test_evaluate_single_image(self, yolo_dataset: Path):
        from tcip_mcp.tools.annotation_tools import evaluate_detections

        img = str(yolo_dataset / "images" / "img_001.jpg")
        result = evaluate_detections(img, iou_threshold=0.5, conf_threshold=0.25)
        assert "error" not in result
        assert result["tp"] >= 0
        assert result["fp"] >= 0
        assert result["fn"] >= 0
        assert 0.0 <= result["precision"] <= 1.0
        assert 0.0 <= result["recall"] <= 1.0

    def test_evaluate_dataset(self, yolo_dataset: Path):
        from tcip_mcp.tools.annotation_tools import evaluate_dataset

        result = evaluate_dataset(str(yolo_dataset), iou_threshold=0.5)
        assert result["image_count"] == 3
        assert "precision" in result
        assert "recall" in result


# ── Run matching integration test ───────────────────────────────────────────


class TestRunMatching:
    """Test run_matching with actual file I/O."""

    def test_run_matching(self, yolo_dataset: Path):
        from tcip_mcp.tools.annotation_tools import run_matching

        img = str(yolo_dataset / "images" / "img_001.jpg")
        result = run_matching(img, iou_threshold=0.5)
        assert "error" not in result
        assert "detections" in result


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
