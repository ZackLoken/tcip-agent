"""Tests for the vision rendering engine and MCP tools."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image


@pytest.fixture
def viz_dataset(tmp_path: Path) -> Path:
    """Create a dataset with images, labels, and predictions (name-based layout)."""
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    labels_dir = tmp_path / "annotations"
    labels_dir.mkdir(parents=True)
    preds_dir = tmp_path / "predictions" / "live"
    preds_dir.mkdir(parents=True)

    for name in ("img_001", "img_002", "img_003", "img_004"):
        img = Image.new("RGB", (640, 480), color=(100, 120, 80))
        img.save(images_dir / f"{name}.jpg")
        # Per-image JSON GT: two pixel-space boxes under two distinct subjects.
        json_io.write_annotations(
            labels_dir / f"{name}.json",
            [Annotation(subject="catkin", geometry=BBox(288, 216, 352, 264)),
             Annotation(subject="nut", geometry=BBox(176, 132, 208, 156))],
            640, 480,
        )
        # Per-image JSON predictions carry a score.
        json_io.write_annotations(
            preds_dir / f"{name}.json",
            [Annotation(subject="catkin", geometry=BBox(288, 216, 352, 264), score=0.95),
             Annotation(subject="nut", geometry=BBox(496, 372, 528, 396), score=0.6)],
            640, 480,
        )

    return tmp_path


# â”€â”€ Rendering engine tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def test_default_output_resolves_under_project_root(tmp_path, monkeypatch):
    # The default viz dir must resolve under TCIP_PROJECT_ROOT (the active project), not the
    # process CWD: the agent's CWD is often the repo, which fragmented renders away from the
    # project. The returned path is absolute so callers know which root it used.
    from tcip_annotation.viz import _default_output

    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    out = Path(_default_output("detections"))
    assert out.is_absolute()
    assert out.parent == (tmp_path / ".tcip" / "artifacts" / "viz").resolve()


def test_default_output_falls_back_to_cwd_when_unset(tmp_path, monkeypatch):
    from tcip_annotation.viz import _default_output

    monkeypatch.delenv("TCIP_PROJECT_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    out = Path(_default_output("detections"))
    assert out.parent == (tmp_path / ".tcip" / "artifacts" / "viz").resolve()


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
            {"rings": [[(100, 100), (200, 100), (200, 200), (100, 200)]], "class_id": 0},
        ]
        out = str(viz_dataset / "test_seg.png")
        result = render_segmentations(img_path, polys, output_path=out)
        assert Path(result).is_file()

    def test_renders_every_ring_of_an_occlusion_split_instance(self, viz_dataset: Path):
        """An instance's rings all get drawn, and it is labelled once, not once per contour."""
        from tcip_annotation.viz import render_segmentations

        img_path = str(viz_dataset / "images" / "img_001.jpg")
        polys = [{"rings": [[(50, 50), (120, 50), (120, 150), (50, 150)],
                            [(300, 60), (380, 60), (380, 140), (300, 140)]],
                  "class_id": 0}]
        one_ring = [{"rings": [polys[0]["rings"][0]], "class_id": 0}]

        both = str(viz_dataset / "seg_two_rings.png")
        first_only = str(viz_dataset / "seg_one_ring.png")
        render_segmentations(img_path, polys, output_path=both)
        render_segmentations(img_path, one_ring, output_path=first_only)

        # The second ring really is painted: the two renders differ.
        assert Path(both).read_bytes() != Path(first_only).read_bytes()

    def test_polygon_without_rings_key_is_skipped_not_crashed(self, viz_dataset: Path):
        """The renderer's rail admits an entry with no drawable ring rather than raising."""
        from tcip_annotation.viz import render_segmentations

        img_path = str(viz_dataset / "images" / "img_001.jpg")
        out = str(viz_dataset / "seg_empty.png")
        assert Path(render_segmentations(img_path, [{"class_id": 0}], output_path=out)).is_file()


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


# â”€â”€ Vision MCP tool tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestVisualizeAnnotations:
    def test_detect(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import visualize

        img = str(viz_dataset / "images" / "img_001.jpg")
        result = visualize("annotations", img, task="detect")
        assert "error" not in result
        assert Path(result["image_path"]).is_file()
        assert result["annotation_count"] == 2

    def test_with_class_names(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import visualize

        img = str(viz_dataset / "images" / "img_001.jpg")
        result = visualize("annotations", img, task="detect", class_names="catkin,nut")
        assert "error" not in result
        assert "catkin" in result["summary"] or "nut" in result["summary"]

    def test_missing_image(self):
        from tcip_mcp.tools.vision_tools import visualize

        result = visualize("annotations", "/nonexistent/image.jpg")
        assert "error" in result

    def test_no_labels(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import visualize

        # Create an image with no labels
        img = Image.new("RGB", (100, 100))
        no_label = viz_dataset / "images" / "no_label.jpg"
        img.save(no_label)
        result = visualize("annotations", str(no_label))
        assert "error" in result

    def test_unknown_source(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import visualize

        img = str(viz_dataset / "images" / "img_001.jpg")
        result = visualize("bogus", img)
        assert "error" in result


class TestVisualizePredictions:
    def test_detect(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import visualize

        img = str(viz_dataset / "images" / "img_001.jpg")
        result = visualize("predictions", img, task="detect")
        assert "error" not in result
        assert Path(result["image_path"]).is_file()
        assert result["prediction_count"] == 2

    def test_missing_predictions(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import visualize

        img = Image.new("RGB", (100, 100))
        no_pred = viz_dataset / "images" / "no_pred.jpg"
        img.save(no_pred)
        result = visualize("predictions", str(no_pred))
        assert "error" in result


class TestVisualizeComparison:
    def test_basic(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import visualize

        img = str(viz_dataset / "images" / "img_001.jpg")
        result = visualize("comparison", img)
        assert "error" not in result
        assert Path(result["image_path"]).is_file()
        assert result["gt_count"] == 2
        assert result["pred_count"] == 2


class TestVisualizeDatasetSample:
    def test_sample(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import visualize

        result = visualize("dataset", str(viz_dataset), n=4)
        assert "error" not in result
        assert Path(result["image_path"]).is_file()
        assert result["sample_count"] == 4
        assert result["total_images"] == 4

    def test_no_images(self, tmp_path: Path):
        from tcip_mcp.tools.vision_tools import visualize

        (tmp_path / "images").mkdir()
        result = visualize("dataset", str(tmp_path), n=4)
        assert "error" in result


class TestVisualizeWorstPredictions:
    def test_basic(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import render_failure_cases

        result = render_failure_cases(
            predictions_dir=str(viz_dataset / "predictions" / "live"),
            labels_dir=str(viz_dataset / "annotations"),
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
                "rings": [[(100, 100), (200, 100), (200, 200), (100, 200)]],
            },
            {
                "candidate_id": 1,
                "bbox": [300.0, 300.0, 400.0, 400.0],
                "area": 5000,
                "stability_score": 0.88,
                "predicted_iou": 0.85,
                "rings": [[(300, 300), (400, 300), (400, 400), (300, 400)]],
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

    def test_grid_wider_than_alphabet(self, viz_dataset: Path):
        from tcip_annotation.viz import render_grid_overlay

        img_path = str(viz_dataset / "images" / "img_001.jpg")
        out = render_grid_overlay(img_path, cols=33, rows=4)
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


class TestColumnLabels:
    """Spreadsheet-style column labels shared by the grid renderer and cell parser."""

    def test_round_trip_boundaries(self):
        from tcip_annotation.sam_wrapper import column_index, column_label

        expected = {0: "A", 25: "Z", 26: "AA", 27: "AB", 31: "AF", 32: "AG", 51: "AZ", 52: "BA"}
        for idx, label in expected.items():
            assert column_label(idx) == label
            assert column_index(label) == idx

    def test_multi_letter_cell_parses(self):
        from tcip_annotation.sam_wrapper import grid_to_pixel

        x, y = grid_to_pixel("AA1", 2700, 480, cols=27, rows=6)
        assert x == pytest.approx(26.5 * 100)
        assert y == pytest.approx(40)

    def test_high_columns_do_not_alias(self):
        # chr() past 'Z' labeled column 32 'a', which case folding silently parsed as column 0
        from tcip_annotation.sam_wrapper import column_label, grid_to_pixel

        cols = 40
        labels = [column_label(c) for c in range(cols)]
        assert len(set(labels)) == cols
        for c, label in enumerate(labels):
            x, _ = grid_to_pixel(f"{label}1", cols * 100, 480, cols=cols, rows=6)
            assert x == pytest.approx((c + 0.5) * 100)

    def test_out_of_range_hint_uses_column_labels(self):
        from tcip_annotation.sam_wrapper import grid_to_pixel

        with pytest.raises(ValueError, match="Use A-AD"):
            grid_to_pixel("BA1", 640, 480, cols=30, rows=6)


class TestSamPredictorCache:
    """Predictor/image caching in sam_wrapper (fake SAM2, no checkpoint)."""

    def test_model_swap_invalidates_image_cache(self, monkeypatch, tmp_path: Path):
        # sam_wrapper loads images via cv2, which ships only with the optional ``sam`` extra
        # (not installed in CI): skip there, like the other SAM-stack tests.
        pytest.importorskip("cv2")
        import sys
        import types

        import numpy as np

        from tcip_annotation import sam_wrapper

        set_image_calls: list[object] = []

        class FakePredictor:
            def __init__(self, model):
                self.model = model

            def set_image(self, img):
                set_image_calls.append(self)

            def predict(self, **kwargs):
                masks = np.zeros((1, 16, 16), dtype=bool)
                masks[0, 4:12, 4:12] = True
                return masks, np.array([0.9]), None

        build_sam_mod = types.ModuleType("sam2.build_sam")
        build_sam_mod.build_sam2 = lambda config, ckpt, device: object()
        predictor_mod = types.ModuleType("sam2.sam2_image_predictor")
        predictor_mod.SAM2ImagePredictor = FakePredictor
        monkeypatch.setitem(sys.modules, "sam2", types.ModuleType("sam2"))
        monkeypatch.setitem(sys.modules, "sam2.build_sam", build_sam_mod)
        monkeypatch.setitem(sys.modules, "sam2.sam2_image_predictor", predictor_mod)

        # Fake home with the expected checkpoint files present.
        ckpt_dir = tmp_path / ".cache" / "tcip" / "sam2"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "sam2.1_hiera_tiny.pt").write_bytes(b"fake")
        (ckpt_dir / "sam2.1_hiera_large.pt").write_bytes(b"fake")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Start from a clean singleton state (monkeypatch restores afterwards).
        monkeypatch.setattr(sam_wrapper, "_predictor", None)
        monkeypatch.setattr(sam_wrapper, "_current_model_type", None)
        monkeypatch.setattr(sam_wrapper, "_current_image_path", None)

        img_path = str(tmp_path / "cache_test.jpg")
        Image.new("RGB", (16, 16), color=(120, 120, 120)).save(img_path)

        rings = sam_wrapper.predict_from_point(img_path, 8, 8, model_type="hiera_t")
        assert len(rings) == 1 and len(rings[0]) >= 3
        assert len(set_image_calls) == 1

        # Same image + same model: embedding cache hit, no new set_image.
        sam_wrapper.predict_from_point(img_path, 8, 8, model_type="hiera_t")
        assert len(set_image_calls) == 1

        # Same image + different model: new predictor must recompute the embedding.
        sam_wrapper.predict_from_point(img_path, 8, 8, model_type="hiera_l")
        assert len(set_image_calls) == 2
        assert set_image_calls[1] is not set_image_calls[0]


class TestVisualizeGridOverlayTool:
    def test_basic(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import overlay_reference_grid

        result = overlay_reference_grid(
            image_path=str(viz_dataset / "images" / "img_001.jpg"),
        )
        assert "error" not in result
        assert Path(result["image_path"]).is_file()
        assert result["cols"] == 8
        assert result["rows"] == 6

    def test_missing_image(self):
        from tcip_mcp.tools.vision_tools import overlay_reference_grid

        result = overlay_reference_grid(image_path="/nonexistent.jpg")
        assert "error" in result

    def test_wide_grid_summary_labels(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import overlay_reference_grid

        result = overlay_reference_grid(
            image_path=str(viz_dataset / "images" / "img_001.jpg"), cols=33, rows=4,
        )
        assert "error" not in result
        assert "'AG4'" in result["summary"]


class TestProposeAnnotationsTool:
    """Test propose_annotations tool (mocked engine)."""

    def test_missing_image(self):
        from tcip_mcp.tools.vision_tools import propose_annotations

        result = propose_annotations(image_path="/nonexistent.jpg")
        assert "error" in result

    def test_unknown_engine(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import propose_annotations

        result = propose_annotations(
            image_path=str(viz_dataset / "images" / "img_001.jpg"),
            engine="does_not_exist",
        )
        assert "error" in result
        assert "does_not_exist" in result["error"]


class TestAcceptProposalsTool:
    def test_no_prior_proposals(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import accept_proposals

        result = accept_proposals(
            image_path=str(viz_dataset / "images" / "img_003.jpg"),
            assignments=[{"candidate_id": 0, "subject": "catkin"}],
        )
        assert "error" in result
        assert "Run propose_annotations first" in result["error"]

    def test_with_cached_proposals(self, viz_dataset: Path):
        import json
        from tcip_mcp.tools.vision_tools import accept_proposals

        # Simulate cached proposals from propose_annotations (neutral schema + engine envelope).
        from tcip_mcp.project_paths import resolve_state

        state_dir = resolve_state(Path(".tcip") / "state")
        state_dir.mkdir(parents=True, exist_ok=True)
        candidates = [
            {
                "candidate_id": 0,
                "bbox": [100.0, 100.0, 200.0, 200.0],
                "area": 10000,
                "score": 0.90,
                "engine": "sam",
                "engine_meta": {"stability_score": 0.95, "predicted_iou": 0.90},
                "rings": [[[100, 100], [200, 100], [200, 200], [100, 200]]],
            },
            {
                "candidate_id": 1,
                "bbox": [300.0, 300.0, 400.0, 400.0],
                "area": 5000,
                "score": 0.85,
                "engine": "sam",
                "engine_meta": {"stability_score": 0.88, "predicted_iou": 0.85},
                "rings": [[[300, 300], [400, 300], [400, 400], [300, 400]]],
            },
        ]
        (state_dir / "proposals_img_001.json").write_text(
            json.dumps({"engine": "sam", "candidates": candidates}), encoding="utf-8"
        )

        result = accept_proposals(
            image_path=str(viz_dataset / "images" / "img_001.jpg"),
            assignments=[
                {"candidate_id": 0, "subject": "catkin"},
                {"candidate_id": 1, "subject": "nut"},
            ],
        )
        assert "error" not in result
        assert result["proposal_count"] == 2
        assert Path(result["image_path"]).is_file()

        # Masks are staged as SAM predictions (predictions/<engine>, engine="sam"), not GT: one
        # unified per-image file holding both accepted objects by subject name.
        from tcip_annotation import json_io

        pred_file = viz_dataset / "predictions" / "sam" / "img_001.json"
        assert pred_file.is_file()
        anns = json_io.read_annotations(pred_file)
        assert len(anns) == 2
        assert {a.subject for a in anns} == {"catkin", "nut"}

        # Each staged object is SAM output: created_by="sam" and a numeric score.
        objs = json.loads(pred_file.read_text(encoding="utf-8"))["annotations"]
        assert objs and all(o["created_by"] == "sam" for o in objs)
        assert all(isinstance(o["score"], float) for o in objs)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Layer 2: Integration tests â€” full pipeline with mocked SAM candidates
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


MOCK_CANDIDATES = [
    {
        "candidate_id": 0,
        "bbox": [50.0, 40.0, 200.0, 180.0],
        "area": 21000,
        "score": 0.93,
        "engine": "sam",
        "engine_meta": {"stability_score": 0.96, "predicted_iou": 0.93},
        "rings": [[
            (50, 40), (200, 40), (200, 180), (50, 180),
        ]],
    },
    {
        "candidate_id": 1,
        "bbox": [300.0, 250.0, 450.0, 400.0],
        "area": 15000,
        "score": 0.88,
        "engine": "sam",
        "engine_meta": {"stability_score": 0.91, "predicted_iou": 0.88},
        "rings": [[
            (300, 250), (450, 250), (450, 400), (300, 400),
        ]],
    },
    {
        "candidate_id": 2,
        "bbox": [500.0, 100.0, 600.0, 200.0],
        "area": 8000,
        "score": 0.80,
        "engine": "sam",
        "engine_meta": {"stability_score": 0.85, "predicted_iou": 0.80},
        "rings": [[
            (500, 100), (600, 100), (600, 200), (500, 200),
        ]],
    },
]


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Layer 3: SAM integration tests (skipped when SAM is not installed)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


def _sam_available() -> bool:
    """Whether SAM2 and its runtime deps are importable (the wrapper uses ``sam2``, not SAM1)."""
    try:
        import cv2  # noqa: F401
        import sam2  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


#: The SAM2 model the guarded tests exercise: the smallest variant, for speed.
SAM_TEST_MODEL = "hiera_t"


def _sam_checkpoint_available() -> bool:
    """Whether the SAM2 checkpoint the guarded tests load (``SAM_TEST_MODEL``) exists, resolved
    through the wrapper's own ``checkpoint_path`` so the guard can't drift from what the code reads."""
    from tcip_annotation.sam_wrapper import checkpoint_path

    return checkpoint_path(SAM_TEST_MODEL).is_file()


requires_sam = pytest.mark.skipif(
    not _sam_available() or not _sam_checkpoint_available(),
    reason="SAM2 (sam2 package + hiera_t checkpoint) not available",
)


@requires_sam
class TestSamAutoMask:
    """Integration tests for auto_mask with real SAM model."""

    @pytest.fixture
    def real_image(self, tmp_path: Path) -> str:
        """Create a synthetic image with distinct regions for SAM."""
        img = Image.new("RGB", (512, 512), color=(200, 200, 200))
        # Draw some colored rectangles to give SAM something to segment
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 200, 200], fill=(255, 0, 0))
        draw.rectangle([300, 100, 480, 300], fill=(0, 0, 255))
        draw.ellipse([100, 300, 250, 450], fill=(0, 200, 0))
        path = str(tmp_path / "test_sam.jpg")
        img.save(path)
        return path

    def test_auto_mask_returns_candidates(self, real_image: str):
        from tcip_annotation.sam_wrapper import auto_mask

        candidates = auto_mask(
            real_image,
            model_type=SAM_TEST_MODEL,
            points_per_side=16,  # smaller for speed
            min_mask_region_area=500,
        )
        assert isinstance(candidates, list)
        assert len(candidates) > 0

        for c in candidates:
            assert "candidate_id" in c
            assert "bbox" in c
            assert len(c["bbox"]) == 4
            assert "rings" in c
            assert c["rings"] and all(len(r) >= 3 for r in c["rings"])
            assert "area" in c
            assert c["area"] > 0
            assert 0.0 <= c["stability_score"] <= 1.0
            assert c["predicted_iou"] >= 0.0  # can slightly exceed 1.0 due to model FP

    def test_auto_mask_sorted_by_area(self, real_image: str):
        from tcip_annotation.sam_wrapper import auto_mask

        candidates = auto_mask(real_image, model_type=SAM_TEST_MODEL, points_per_side=16)
        areas = [c["area"] for c in candidates]
        assert areas == sorted(areas, reverse=True)

    def test_auto_mask_polygon_within_image(self, real_image: str):
        """All polygon vertices, in every ring, should be within image bounds."""
        from tcip_annotation.sam_wrapper import auto_mask

        candidates = auto_mask(real_image, model_type=SAM_TEST_MODEL, points_per_side=16)
        for c in candidates:
            for x, y in (pt for ring in c["rings"] for pt in ring):
                assert 0 <= x <= 512, f"x={x} out of bounds"
                assert 0 <= y <= 512, f"y={y} out of bounds"


@requires_sam
class TestSamPredictFromGrid:
    """Integration tests for segment_prompt with grid_cells (real SAM)."""

    @pytest.fixture
    def real_dataset(self, tmp_path: Path) -> Path:
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        img = Image.new("RGB", (640, 480), color=(200, 200, 200))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        # Object at roughly B2 area (col=1, row=1 â†’ center ~120,120)
        draw.rectangle([80, 80, 160, 160], fill=(255, 0, 0))
        img.save(images_dir / "grid_test.jpg")
        return tmp_path

    def test_grid_cell_produces_polygon(self, real_dataset: Path):
        from tcip_mcp.tools.annotation_tools import segment_prompt

        img_path = str(real_dataset / "images" / "grid_test.jpg")
        result = segment_prompt(
            image_path=img_path,
            grid_cells=["B2"],
            cols=8, rows=6,
            engine_params={"model_type": SAM_TEST_MODEL},
        )
        assert "error" not in result
        assert result["rings"] and all(len(r) >= 3 for r in result["rings"])
        assert result["vertex_count"] >= 3


@requires_sam
class TestFullSamPipeline:
    """End-to-end: auto_mask â†’ accept â†’ verify, with real SAM."""

    @pytest.fixture
    def sam_dataset(self, tmp_path: Path) -> Path:
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        img = Image.new("RGB", (512, 512), color=(220, 220, 220))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([100, 100, 300, 300], fill=(255, 0, 0))
        draw.rectangle([350, 350, 480, 480], fill=(0, 0, 255))
        img.save(images_dir / "e2e.jpg")
        return tmp_path

    def test_auto_label_then_accept(self, sam_dataset: Path):
        from tcip_mcp.tools.vision_tools import accept_proposals, propose_annotations

        img_path = str(sam_dataset / "images" / "e2e.jpg")

        # Step 1: auto-label
        auto_result = propose_annotations(
            image_path=img_path,
            engine_params={
                "model_type": SAM_TEST_MODEL,
                "points_per_side": 16,
                "min_mask_region_area": 500,
            },
        )
        assert "error" not in auto_result
        assert auto_result["candidate_count"] > 0
        assert Path(auto_result["image_path"]).is_file()

        # Step 2: accept first two candidates
        cands = auto_result["candidates"][:2]
        accept_result = accept_proposals(
            image_path=img_path,
            assignments=[
                {"candidate_id": c["id"], "subject": subj}
                for c, subj in zip(cands, ("catkin", "nut"))
            ],
        )
        assert "error" not in accept_result
        assert accept_result["proposal_count"] == 2
        assert Path(accept_result["image_path"]).is_file()


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Layer 2: Integration tests â€” full pipeline with mocked SAM candidates
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


MOCK_CANDIDATES = [
    {
        "candidate_id": 0,
        "bbox": [50.0, 40.0, 200.0, 180.0],
        "area": 21000,
        "score": 0.93,
        "engine": "sam",
        "engine_meta": {"stability_score": 0.96, "predicted_iou": 0.93},
        "rings": [[
            (50, 40), (200, 40), (200, 180), (50, 180),
        ]],
    },
    {
        "candidate_id": 1,
        "bbox": [300.0, 250.0, 450.0, 400.0],
        "area": 15000,
        "score": 0.88,
        "engine": "sam",
        "engine_meta": {"stability_score": 0.91, "predicted_iou": 0.88},
        "rings": [[
            (300, 250), (450, 250), (450, 400), (300, 400),
        ]],
    },
    {
        "candidate_id": 2,
        "bbox": [500.0, 100.0, 600.0, 200.0],
        "area": 8000,
        "score": 0.80,
        "engine": "sam",
        "engine_meta": {"stability_score": 0.85, "predicted_iou": 0.80},
        "rings": [[
            (500, 100), (600, 100), (600, 200), (500, 200),
        ]],
    },
]


class TestCandidateCacheRoundTrip:
    """Verify candidates survive JSON serialize â†’ deserialize."""

    def test_polygon_geometry_preserved(self, tmp_path: Path):
        state_file = tmp_path / "candidates_test.json"
        state_file.write_text(json.dumps(MOCK_CANDIDATES, default=str), encoding="utf-8")

        loaded = json.loads(state_file.read_text(encoding="utf-8"))
        assert len(loaded) == 3

        for orig, restored in zip(MOCK_CANDIDATES, loaded):
            assert orig["candidate_id"] == restored["candidate_id"]
            assert orig["bbox"] == restored["bbox"]
            assert orig["area"] == restored["area"]
            assert abs(orig["score"] - restored["score"]) < 1e-6
            assert abs(orig["engine_meta"]["stability_score"]
                       - restored["engine_meta"]["stability_score"]) < 1e-6
            # Every ring, every vertex â€” JSON turns tuples into lists
            assert len(orig["rings"]) == len(restored["rings"])
            for o_ring, r_ring in zip(orig["rings"], restored["rings"]):
                for (ox, oy), rp in zip(o_ring, r_ring):
                    rx, ry = rp if isinstance(rp, (list, tuple)) else (rp["x"], rp["y"])
                    assert abs(ox - rx) < 1e-6
                    assert abs(oy - ry) < 1e-6

    def test_empty_candidates_roundtrip(self, tmp_path: Path):
        state_file = tmp_path / "candidates_empty.json"
        state_file.write_text(json.dumps([]), encoding="utf-8")
        loaded = json.loads(state_file.read_text(encoding="utf-8"))
        assert loaded == []


class TestFullPipelineIntegration:
    """End-to-end flow: mock candidates â†’ accept â†’ verify output files."""

    @pytest.fixture
    def pipeline_dataset(self, tmp_path: Path) -> Path:
        """Fresh dataset without pre-existing labels."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        img = Image.new("RGB", (640, 480), color=(80, 120, 60))
        img.save(images_dir / "sample.jpg")
        return tmp_path

    def _cache_candidates(self, stem: str, candidates: list[dict]) -> None:
        from tcip_mcp.project_paths import resolve_state

        state_dir = resolve_state(Path(".tcip") / "state")
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / f"proposals_{stem}.json").write_text(
            json.dumps({"engine": "sam", "candidates": candidates}, default=str), encoding="utf-8",
        )

    def test_accept_writes_json_detect(self, pipeline_dataset: Path):
        """SAM proposals are staged as predictions: pixel geometry, subject names, and score preserved."""
        from tcip_annotation import bbox_of, json_io
        from tcip_mcp.tools.vision_tools import accept_proposals

        img_path = str(pipeline_dataset / "images" / "sample.jpg")
        self._cache_candidates("sample", MOCK_CANDIDATES)

        result = accept_proposals(
            image_path=img_path,
            assignments=[
                {"candidate_id": 0, "subject": "catkin"},
                {"candidate_id": 1, "subject": "nut"},
            ],
        )
        assert "error" not in result
        assert result["proposal_count"] == 2

        pred_file = pipeline_dataset / "predictions" / "sam" / "sample.json"
        assert pred_file.is_file()
        anns = json_io.read_annotations(pred_file)
        assert len(anns) == 2
        assert {a.subject for a in anns} == {"catkin", "nut"}
        # Pixel coords within the 640x480 image; staged as SAM predictions (created_by="sam").
        for a in anns:
            b = bbox_of(a.geometry)
            assert 0.0 <= b.x1 < b.x2 <= 640.0
            assert 0.0 <= b.y1 < b.y2 <= 480.0
            assert a.created_by == "sam"
        objs = json.loads(pred_file.read_text(encoding="utf-8"))["annotations"]
        assert all(isinstance(o["score"], float) for o in objs)

    def test_accept_writes_json_segment(self, pipeline_dataset: Path):
        """SAM proposals are staged as prediction polygons: pixel vertices, subject, score."""
        from tcip_annotation import json_io
        from tcip_mcp.tools.vision_tools import accept_proposals

        img_path = str(pipeline_dataset / "images" / "sample.jpg")
        self._cache_candidates("sample", MOCK_CANDIDATES)

        result = accept_proposals(
            image_path=img_path,
            assignments=[{"candidate_id": 0, "subject": "bud"}],
        )
        assert "error" not in result
        assert result["proposal_count"] == 1

        pred_file = pipeline_dataset / "predictions" / "sam" / "sample.json"
        assert pred_file.is_file()
        anns = json_io.read_annotations(pred_file)
        assert len(anns) == 1
        assert {a.subject for a in anns} == {"bud"}
        rings = anns[0].geometry.rings
        assert rings and all(len(r) >= 3 for r in rings)
        for x, y in (pt for ring in rings for pt in ring):
            assert 0.0 <= x <= 640.0
            assert 0.0 <= y <= 480.0
        objs = json.loads(pred_file.read_text(encoding="utf-8"))["annotations"]
        assert objs and all(o["created_by"] == "sam" for o in objs)
        assert all(isinstance(o["score"], float) for o in objs)

    def test_detect_and_segment_consistent(self, pipeline_dataset: Path):
        """Box and mask views of the staged predictions cover the same objects (one unified file)."""
        from tcip_annotation import bbox_of, json_io
        from tcip_mcp.tools.vision_tools import accept_proposals

        img_path = str(pipeline_dataset / "images" / "sample.jpg")
        self._cache_candidates("sample", MOCK_CANDIDATES)

        result = accept_proposals(
            image_path=img_path,
            assignments=[
                {"candidate_id": 0, "subject": "catkin"},
                {"candidate_id": 2, "subject": "nut"},
            ],
        )
        assert result["proposal_count"] == 2

        pred_file = pipeline_dataset / "predictions" / "sam" / "sample.json"
        anns = json_io.read_annotations(pred_file)
        # Each object is one polygon with a derivable box under the same subject: the box and mask
        # views can never diverge because they are the same annotations.
        subjects_poly = sorted(a.subject for a in anns if a.geometry is not None)
        subjects_box = sorted(a.subject for a in anns if bbox_of(a.geometry) is not None)
        assert subjects_poly == subjects_box == ["catkin", "nut"]

    def test_partial_accept_skips_rejected(self, pipeline_dataset: Path):
        """Only accepted candidates appear in output; rejected are omitted."""
        from tcip_mcp.tools.vision_tools import accept_proposals

        img_path = str(pipeline_dataset / "images" / "sample.jpg")
        self._cache_candidates("sample", MOCK_CANDIDATES)

        # Accept only candidate 1 out of 3
        result = accept_proposals(
            image_path=img_path,
            assignments=[{"candidate_id": 1, "subject": "catkin"}],
        )
        assert result["proposal_count"] == 1

    def test_invalid_candidate_id_silently_skipped(self, pipeline_dataset: Path):
        """Assignments with non-existent candidate_id are ignored."""
        from tcip_mcp.tools.vision_tools import accept_proposals

        img_path = str(pipeline_dataset / "images" / "sample.jpg")
        self._cache_candidates("sample", MOCK_CANDIDATES)

        result = accept_proposals(
            image_path=img_path,
            assignments=[
                {"candidate_id": 999, "subject": "catkin"},  # non-existent
                {"candidate_id": 0, "subject": "nut"},        # valid
            ],
        )
        assert result["proposal_count"] == 1

    def test_render_then_accept_pipeline(self, pipeline_dataset: Path):
        """Full render â†’ accept â†’ verify pipeline (sans SAM)."""
        from tcip_annotation.viz import render_candidates, render_grid_overlay
        from tcip_mcp.tools.vision_tools import accept_proposals

        img_path = str(pipeline_dataset / "images" / "sample.jpg")

        # Step 1: Render candidates
        candidate_render = render_candidates(img_path, MOCK_CANDIDATES)
        assert Path(candidate_render).is_file()

        # Step 2: Render grid overlay (for correction reference)
        grid_render = render_grid_overlay(img_path)
        assert Path(grid_render).is_file()

        # Step 3: Cache candidates (simulating propose_annotations state save)
        self._cache_candidates("sample", MOCK_CANDIDATES)

        # Step 4: Accept with class assignments
        result = accept_proposals(
            image_path=img_path,
            assignments=[
                {"candidate_id": 0, "subject": "catkin"},
                {"candidate_id": 1, "subject": "nut"},
                {"candidate_id": 2, "subject": "catkin"},
            ],
        )
        assert "error" not in result
        assert result["proposal_count"] == 3

        # Step 5: QA render was produced
        assert Path(result["image_path"]).is_file()


class TestGridCellToSamPrompt:
    """Test grid cell -> point prompt conversion in segment_prompt."""

    def test_grid_cells_converted_to_points(self, viz_dataset: Path):
        """Grid cells should convert to points before hitting SAM."""
        from tcip_mcp.tools.annotation_tools import segment_prompt

        img_path = str(viz_dataset / "images" / "img_001.jpg")

        # Mock at the source module where the lazy import resolves
        with patch(
            "tcip_annotation.sam_wrapper.predict_from_points"
        ) as mock_predict:
            mock_predict.return_value = [
                [(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)],
            ]

            result = segment_prompt(
                image_path=img_path,
                grid_cells=["A1", "D3"],
                cols=8, rows=6,
            )

        # Should have called predict_from_points (2 points â†’ multi-point)
        assert mock_predict.called
        call_args = mock_predict.call_args
        pts = call_args[0][1]  # second positional arg: points
        lbls = call_args[0][2]  # third: labels
        assert len(pts) == 2
        assert all(lbl == 1 for lbl in lbls)

        # Verify result carries the mask's rings
        assert result["ring_count"] == 1
        assert result["vertex_count"] == 4

    def test_single_grid_cell_uses_single_point(self, viz_dataset: Path):
        """Single grid cell should use predict_from_point (not predict_from_points)."""
        from tcip_mcp.tools.annotation_tools import segment_prompt

        img_path = str(viz_dataset / "images" / "img_001.jpg")

        with patch(
            "tcip_annotation.sam_wrapper.predict_from_point"
        ) as mock_predict:
            mock_predict.return_value = [
                [(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)],
            ]

            segment_prompt(
                image_path=img_path,
                grid_cells=["C4"],
                cols=8, rows=6,
            )

        assert mock_predict.called
        # Verify the coordinates are reasonable for C4 on 640x480
        call_args = mock_predict.call_args
        x = call_args[0][1]  # positional: image_path, x, y
        y = call_args[0][2]
        # C = column 2 (0-indexed), cell center x = (2+0.5)*640/8 = 200
        assert abs(x - 200.0) < 1.0
        # 4 = row 3 (0-indexed), cell center y = (3+0.5)*480/6 = 280
        assert abs(y - 280.0) < 1.0

    def test_invalid_grid_cell_returns_error(self, viz_dataset: Path):
        """Invalid grid cell like 'Z9' should return error, not crash."""
        from tcip_mcp.tools.annotation_tools import segment_prompt

        img_path = str(viz_dataset / "images" / "img_001.jpg")
        result = segment_prompt(image_path=img_path, grid_cells=["Z9"], cols=8, rows=6)
        assert "error" in result
        assert "Invalid grid cell" in result["error"]

    def test_grid_cells_without_cols_and_rows_is_refused(self, viz_dataset: Path):
        """A cell name means nothing without its grid: resolving 'B3' against an assumed 8x6 when the
        breeder rendered another grid picks the wrong pixel silently, so the call is refused."""
        from tcip_mcp.tools.annotation_tools import segment_prompt

        img_path = str(viz_dataset / "images" / "img_001.jpg")
        for kwargs in ({}, {"cols": 10}, {"rows": 8}):
            result = segment_prompt(image_path=img_path, grid_cells=["B3"], **kwargs)
            assert "error" in result, kwargs
            assert "cols and rows" in result["error"]

    def test_grid_cells_resolve_against_the_callers_own_grid(self, viz_dataset: Path):
        """The cells are resolved with the caller's cols/rows, the grid overlay_reference_grid
        actually rendered, not the wrapper's own 8x6 signature default."""
        from tcip_mcp.tools.annotation_tools import segment_prompt

        img_path = str(viz_dataset / "images" / "img_001.jpg")
        with patch("tcip_annotation.sam_wrapper.predict_from_point") as mock_predict:
            mock_predict.return_value = [
                [(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)],
            ]
            segment_prompt(image_path=img_path, grid_cells=["C4"], cols=10, rows=8)

        x, y = mock_predict.call_args[0][1], mock_predict.call_args[0][2]
        # C4 on a 10x8 grid over 640x480: x = (2+0.5)*640/10 = 160, y = (3+0.5)*480/8 = 210;
        # the 8x6 reading of the same cell would be (200, 280).
        assert abs(x - 160.0) < 1.0
        assert abs(y - 210.0) < 1.0

    def test_no_prompts_returns_error(self, viz_dataset: Path):
        """Calling with no prompts at all should error."""
        from tcip_mcp.tools.annotation_tools import segment_prompt

        img_path = str(viz_dataset / "images" / "img_001.jpg")
        result = segment_prompt(image_path=img_path)
        assert "error" in result


class TestSamPredictionStaging:
    """accept_proposals stages engine masks as predictions (predictions/sam), not ground truth."""

    @pytest.fixture
    def format_dataset(self, tmp_path: Path) -> Path:
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        img = Image.new("RGB", (640, 480), color=(100, 100, 100))
        img.save(images_dir / "fmt_test.jpg")
        return tmp_path

    def _cache(self, stem: str) -> None:
        from tcip_mcp.project_paths import resolve_state

        state_dir = resolve_state(Path(".tcip") / "state")
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / f"proposals_{stem}.json").write_text(
            json.dumps({"engine": "sam", "candidates": MOCK_CANDIDATES}, default=str), encoding="utf-8",
        )

    def test_json_detect_and_segment_written(self, format_dataset: Path):
        from tcip_annotation import bbox_of, json_io
        from tcip_mcp.tools.vision_tools import accept_proposals

        self._cache("fmt_test")
        result = accept_proposals(
            image_path=str(format_dataset / "images" / "fmt_test.jpg"),
            assignments=[{"candidate_id": 0, "subject": "catkin"}],
        )
        assert "error" not in result
        assert "format" not in result  # fmt param dropped in the JSON cutover
        assert result["proposal_count"] == 1

        pred = format_dataset / "predictions" / "sam" / "fmt_test.json"
        assert pred.is_file()
        anns = json_io.read_annotations(pred)
        assert len(anns) == 1 and {a.subject for a in anns} == {"catkin"}
        # The staged object carries a polygon (mask) with a derivable box: both views of one object.
        assert anns[0].geometry is not None
        assert bbox_of(anns[0].geometry) is not None

    def test_prediction_carries_sam_score(self, format_dataset: Path):
        """Staged SAM output is a prediction: each object has created_by="sam" and a ``score``."""
        self._cache("fmt_test")
        from tcip_mcp.tools.vision_tools import accept_proposals

        accept_proposals(
            image_path=str(format_dataset / "images" / "fmt_test.jpg"),
            assignments=[{"candidate_id": 0, "subject": "catkin"}],
        )
        pred = format_dataset / "predictions" / "sam" / "fmt_test.json"
        data = json.loads(pred.read_text(encoding="utf-8"))
        assert data["annotations"]
        assert all(o["created_by"] == "sam" for o in data["annotations"])
        assert all(isinstance(o["score"], float) for o in data["annotations"])

