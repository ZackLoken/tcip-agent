"""Tests for the vision rendering engine and MCP tools."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def viz_dataset(tmp_path: Path) -> Path:
    """Create a dataset with images, labels, and predictions."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    labels_dir = tmp_path / "labels" / "detect"
    labels_dir.mkdir(parents=True)
    preds_dir = tmp_path / "predictions" / "detect"
    preds_dir.mkdir(parents=True)

    for name in ("img_001", "img_002", "img_003", "img_004"):
        img = Image.new("RGB", (640, 480), color=(100, 120, 80))
        img.save(images_dir / f"{name}.jpg")
        # YOLO: class_id cx cy w h
        (labels_dir / f"{name}.txt").write_text(
            "0 0.5 0.5 0.1 0.1\n1 0.3 0.3 0.05 0.05\n"
        )
        # Predictions: class_id conf cx cy w h
        (preds_dir / f"{name}.txt").write_text(
            "0 0.95 0.5 0.5 0.1 0.1\n1 0.6 0.8 0.8 0.05 0.05\n"
        )

    return tmp_path


# ── Rendering engine tests ──────────────────────────────────────────────────


class TestRenderDetections:
    def test_basic_render(self, viz_dataset: Path):
        from tcip_annotation.viz import render_detections

        img_path = str(viz_dataset / "images" / "img_001.jpg")
        boxes = [
            {"x1": 100, "y1": 100, "x2": 200, "y2": 200, "class_id": 0},
            {"x1": 300, "y1": 300, "x2": 400, "y2": 400, "class_id": 1},
        ]
        out = str(viz_dataset / "test_render.png")
        result = render_detections(img_path, boxes, output_path=out)
        assert Path(result).is_file()
        rendered = Image.open(result)
        assert rendered.size[0] > 0

    def test_with_class_names(self, viz_dataset: Path):
        from tcip_annotation.viz import render_detections

        img_path = str(viz_dataset / "images" / "img_001.jpg")
        boxes = [{"x1": 100, "y1": 100, "x2": 200, "y2": 200, "class_id": 0}]
        out = str(viz_dataset / "test_names.png")
        result = render_detections(
            img_path, boxes,
            class_names={0: "catkin", 1: "nut"},
            output_path=out,
        )
        assert Path(result).is_file()

    def test_with_confidence(self, viz_dataset: Path):
        from tcip_annotation.viz import render_detections

        img_path = str(viz_dataset / "images" / "img_001.jpg")
        boxes = [{"x1": 100, "y1": 100, "x2": 200, "y2": 200, "class_id": 0, "confidence": 0.95}]
        out = str(viz_dataset / "test_conf.png")
        result = render_detections(img_path, boxes, output_path=out)
        assert Path(result).is_file()


class TestRenderSegmentations:
    def test_basic_render(self, viz_dataset: Path):
        from tcip_annotation.viz import render_segmentations

        img_path = str(viz_dataset / "images" / "img_001.jpg")
        polys = [
            {"points": [(100, 100), (200, 100), (200, 200), (100, 200)], "class_id": 0},
        ]
        out = str(viz_dataset / "test_seg.png")
        result = render_segmentations(img_path, polys, output_path=out)
        assert Path(result).is_file()


class TestRenderComparison:
    def test_basic_comparison(self, viz_dataset: Path):
        from tcip_annotation.viz import render_comparison

        img_path = str(viz_dataset / "images" / "img_001.jpg")
        gt = [{"x1": 100, "y1": 100, "x2": 200, "y2": 200, "class_id": 0}]
        pred = [{"x1": 110, "y1": 110, "x2": 210, "y2": 210, "class_id": 0, "confidence": 0.9}]
        out = str(viz_dataset / "test_comp.png")
        result = render_comparison(img_path, gt, pred, output_path=out)
        assert Path(result).is_file()


class TestRenderGrid:
    def test_grid(self, viz_dataset: Path):
        from tcip_annotation.viz import render_grid

        paths = [str(viz_dataset / "images" / f"img_{i:03d}.jpg") for i in range(1, 5)]
        out = str(viz_dataset / "test_grid.png")
        result = render_grid(paths, titles=["a", "b", "c", "d"], output_path=out)
        assert Path(result).is_file()
        grid = Image.open(result)
        assert grid.size[0] == 4 * 256  # 4 cols * 256 cell_size

    def test_empty_grid(self, viz_dataset: Path):
        from tcip_annotation.viz import render_grid

        out = str(viz_dataset / "test_empty.png")
        result = render_grid([], output_path=out)
        assert Path(result).is_file()


class TestYoloToPixel:
    def test_conversion(self):
        from tcip_annotation.viz import yolo_to_pixel

        x1, y1, x2, y2 = yolo_to_pixel(0.5, 0.5, 0.2, 0.2, 640, 480)
        assert abs(x1 - 256.0) < 0.01
        assert abs(y1 - 192.0) < 0.01
        assert abs(x2 - 384.0) < 0.01
        assert abs(y2 - 288.0) < 0.01


# ── Vision MCP tool tests ──────────────────────────────────────────────────


class TestVisualizeAnnotations:
    def test_detect(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import visualize_annotations

        img = str(viz_dataset / "images" / "img_001.jpg")
        result = visualize_annotations(img, task="detect")
        assert "error" not in result
        assert Path(result["image_path"]).is_file()
        assert result["annotation_count"] == 2

    def test_with_class_names(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import visualize_annotations

        img = str(viz_dataset / "images" / "img_001.jpg")
        result = visualize_annotations(img, task="detect", class_names="catkin,nut")
        assert "error" not in result
        assert "catkin" in result["summary"] or "nut" in result["summary"]

    def test_missing_image(self):
        from tcip_mcp.tools.vision_tools import visualize_annotations

        result = visualize_annotations("/nonexistent/image.jpg")
        assert "error" in result

    def test_no_labels(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import visualize_annotations

        # Create an image with no labels
        img = Image.new("RGB", (100, 100))
        no_label = viz_dataset / "images" / "no_label.jpg"
        img.save(no_label)
        result = visualize_annotations(str(no_label))
        assert "error" in result


class TestVisualizePredictions:
    def test_detect(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import visualize_predictions

        img = str(viz_dataset / "images" / "img_001.jpg")
        result = visualize_predictions(img, task="detect")
        assert "error" not in result
        assert Path(result["image_path"]).is_file()
        assert result["prediction_count"] == 2

    def test_missing_predictions(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import visualize_predictions

        img = Image.new("RGB", (100, 100))
        no_pred = viz_dataset / "images" / "no_pred.jpg"
        img.save(no_pred)
        result = visualize_predictions(str(no_pred))
        assert "error" in result


class TestVisualizeComparison:
    def test_basic(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import visualize_comparison

        img = str(viz_dataset / "images" / "img_001.jpg")
        result = visualize_comparison(img)
        assert "error" not in result
        assert Path(result["image_path"]).is_file()
        assert result["gt_count"] == 2
        assert result["pred_count"] == 2


class TestVisualizeDatasetSample:
    def test_sample(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import visualize_dataset_sample

        result = visualize_dataset_sample(str(viz_dataset), n=4)
        assert "error" not in result
        assert Path(result["image_path"]).is_file()
        assert result["sample_count"] == 4
        assert result["total_images"] == 4

    def test_no_images(self, tmp_path: Path):
        from tcip_mcp.tools.vision_tools import visualize_dataset_sample

        (tmp_path / "images").mkdir()
        result = visualize_dataset_sample(str(tmp_path), n=4)
        assert "error" in result


class TestVisualizeWorstPredictions:
    def test_basic(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import visualize_worst_predictions

        result = visualize_worst_predictions(
            predictions_dir=str(viz_dataset / "predictions" / "detect"),
            labels_dir=str(viz_dataset / "labels" / "detect"),
            images_dir=str(viz_dataset / "images"),
            top_k=3,
        )
        assert "error" not in result
        # Should have rendered some cases
        assert len(result.get("case_images", [])) > 0


# === New tests for SAM repositioning ===


class TestRenderCandidates:
    def test_basic_render(self, viz_dataset: Path):
        from tcip_annotation.viz import render_candidates

        img_path = str(viz_dataset / "images" / "img_001.jpg")
        candidates = [
            {
                "candidate_id": 0,
                "bbox": [100.0, 100.0, 200.0, 200.0],
                "area": 10000,
                "stability_score": 0.95,
                "predicted_iou": 0.90,
                "polygon": [(100, 100), (200, 100), (200, 200), (100, 200)],
            },
            {
                "candidate_id": 1,
                "bbox": [300.0, 300.0, 400.0, 400.0],
                "area": 5000,
                "stability_score": 0.88,
                "predicted_iou": 0.85,
                "polygon": [(300, 300), (400, 300), (400, 400), (300, 400)],
            },
        ]
        out = render_candidates(img_path, candidates)
        assert Path(out).is_file()

    def test_empty_candidates(self, viz_dataset: Path):
        from tcip_annotation.viz import render_candidates

        img_path = str(viz_dataset / "images" / "img_001.jpg")
        out = render_candidates(img_path, [])
        assert Path(out).is_file()


class TestRenderGridOverlay:
    def test_basic(self, viz_dataset: Path):
        from tcip_annotation.viz import render_grid_overlay

        img_path = str(viz_dataset / "images" / "img_001.jpg")
        out = render_grid_overlay(img_path)
        assert Path(out).is_file()

    def test_custom_grid(self, viz_dataset: Path):
        from tcip_annotation.viz import render_grid_overlay

        img_path = str(viz_dataset / "images" / "img_001.jpg")
        out = render_grid_overlay(img_path, cols=4, rows=3)
        assert Path(out).is_file()


class TestGridToPixel:
    def test_basic_conversion(self):
        from tcip_annotation.sam_wrapper import grid_to_pixel

        # A1 should be top-left cell center
        x, y = grid_to_pixel("A1", 640, 480, cols=8, rows=6)
        assert x == pytest.approx(640 / 8 / 2)  # center of first col
        assert y == pytest.approx(480 / 6 / 2)  # center of first row

    def test_bottom_right(self):
        from tcip_annotation.sam_wrapper import grid_to_pixel

        x, y = grid_to_pixel("H6", 640, 480, cols=8, rows=6)
        assert x == pytest.approx(640 - 640 / 8 / 2)
        assert y == pytest.approx(480 - 480 / 6 / 2)

    def test_case_insensitive(self):
        from tcip_annotation.sam_wrapper import grid_to_pixel

        x1, y1 = grid_to_pixel("b3", 640, 480)
        x2, y2 = grid_to_pixel("B3", 640, 480)
        assert x1 == x2
        assert y1 == y2

    def test_invalid_column(self):
        from tcip_annotation.sam_wrapper import grid_to_pixel

        with pytest.raises(ValueError, match="Column"):
            grid_to_pixel("Z1", 640, 480, cols=8, rows=6)

    def test_invalid_row(self):
        from tcip_annotation.sam_wrapper import grid_to_pixel

        with pytest.raises(ValueError, match="Row"):
            grid_to_pixel("A9", 640, 480, cols=8, rows=6)


class TestVisualizeGridOverlayTool:
    def test_basic(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import visualize_grid_overlay

        result = visualize_grid_overlay(
            image_path=str(viz_dataset / "images" / "img_001.jpg"),
        )
        assert "error" not in result
        assert Path(result["image_path"]).is_file()
        assert result["cols"] == 8
        assert result["rows"] == 6

    def test_missing_image(self):
        from tcip_mcp.tools.vision_tools import visualize_grid_overlay

        result = visualize_grid_overlay(image_path="/nonexistent.jpg")
        assert "error" in result


class TestSamAutoLabelTool:
    """Test sam_auto_label tool (mocked SAM)."""

    def test_missing_image(self):
        from tcip_mcp.tools.vision_tools import sam_auto_label

        result = sam_auto_label(image_path="/nonexistent.jpg")
        assert "error" in result


class TestAcceptCandidatesTool:
    def test_no_prior_candidates(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import accept_candidates

        result = accept_candidates(
            image_path=str(viz_dataset / "images" / "img_003.jpg"),
            assignments=[{"candidate_id": 0, "class_id": 1}],
        )
        assert "error" in result
        assert "Run sam_auto_label first" in result["error"]

    def test_with_cached_candidates(self, viz_dataset: Path):
        import json
        from tcip_mcp.tools.vision_tools import accept_candidates

        # Simulate cached candidates from sam_auto_label
        state_dir = Path(".tcip") / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        candidates = [
            {
                "candidate_id": 0,
                "bbox": [100.0, 100.0, 200.0, 200.0],
                "area": 10000,
                "stability_score": 0.95,
                "predicted_iou": 0.90,
                "polygon": [[100, 100], [200, 100], [200, 200], [100, 200]],
            },
            {
                "candidate_id": 1,
                "bbox": [300.0, 300.0, 400.0, 400.0],
                "area": 5000,
                "stability_score": 0.88,
                "predicted_iou": 0.85,
                "polygon": [[300, 300], [400, 300], [400, 400], [300, 400]],
            },
        ]
        (state_dir / "candidates_img_001.json").write_text(
            json.dumps(candidates), encoding="utf-8"
        )

        result = accept_candidates(
            image_path=str(viz_dataset / "images" / "img_001.jpg"),
            assignments=[
                {"candidate_id": 0, "class_id": 0},
                {"candidate_id": 1, "class_id": 1},
            ],
        )
        assert "error" not in result
        assert result["detection_count"] == 2
        assert result["segmentation_count"] == 2
        assert Path(result["image_path"]).is_file()

        # Verify labels were written
        det_file = viz_dataset / "labels" / "detect" / "img_001.txt"
        assert det_file.is_file()
        seg_file = viz_dataset / "labels" / "segment" / "img_001.txt"
        assert seg_file.is_file()
