"""Tests for the vision rendering engine and MCP tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from tcip_mcp.dataset_layout import UNDATED_BUCKET


def _display(image_path: str) -> tuple[np.ndarray, tuple[int, int]]:
    """A file's pixels and native size, the pair the pixel-in renderers take.

    Stands in for whatever the calling tool would have read (``vision_tools._read_for_display``),
    so a renderer test exercises the renderer and nothing else.
    """
    with Image.open(image_path) as im:
        rgb = im.convert("RGB")
        return np.asarray(rgb), rgb.size


@pytest.fixture
def viz_dataset(tmp_path: Path) -> Path:
    """Create a dataset with images and their label documents (name-based layout)."""
    from tcip_annotation.state import Annotation, BBox

    from tests._producer_fixtures import label_image

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)

    for name in ("img_001", "img_002", "img_003", "img_004"):
        img = Image.new("RGB", (640, 480), color=(100, 120, 80))
        img.save(images_dir / f"{name}.jpg")
        # Two pixel-space boxes under two distinct subjects.
        label_image(images_dir / f"{name}.jpg",
                    [Annotation(subject="bud", geometry=BBox(288, 216, 352, 264)),
                     Annotation(subject="nut", geometry=BBox(176, 132, 208, 156))],
                    640, 480)

    return tmp_path


PUBLISHED = "published"
"""The name :func:`viz_bucket` publishes the dataset's predictions under."""


@pytest.fixture
def viz_bucket(viz_dataset: Path) -> Path:
    """``viz_dataset`` with each image's two predictions published as the bucket
    :data:`PUBLISHED` (``_chain_fixtures.published``)."""
    pytest.importorskip("torch")
    from tests._chain_fixtures import published

    published(viz_dataset, PUBLISHED, [
        {"image": str(viz_dataset / "images" / UNDATED_BUCKET / f"{name}.jpg"), "width": 640, "height": 480,
         "boxes": [[288, 216, 352, 264], [496, 372, 528, 396]], "scores": [0.95, 0.6],
         "labels": [1, 1]} for name in ("img_001", "img_002", "img_003", "img_004")],
        scope={"subject": "bud"})
    return viz_dataset


def _damage(key, stored: bytes = b"{not json") -> None:
    """The record ``key`` names replaced in the store by ``stored``."""
    from tests._record_damage_fixtures import damage_record

    damage_record(key, stored)


def _damage_record(project: Path) -> None:
    """Rewrite the published bucket's record as bytes that do not decode."""
    from tcip_mcp.dataset_layout import bucket_key

    _damage(bucket_key(project, PUBLISHED))


def _label_key(image: Path):
    from tests._producer_fixtures import image_label_key

    return image_label_key(image)


def _prediction_key(project: Path, image: Path):
    from tcip_mcp.dataset_layout import prediction_key

    return prediction_key(project, PUBLISHED, image.stem)


# ── Rendering engine tests ──────────────────────────────────────────────────


class TestRenderDetections:
    def test_basic_render(self, viz_dataset: Path):
        from tcip_annotation.viz import render_detections

        pixels, native = _display(str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"))
        boxes = [
            {"x1": 100, "y1": 100, "x2": 200, "y2": 200, "class_id": 0},
            {"x1": 300, "y1": 300, "x2": 400, "y2": 400, "class_id": 1},
        ]
        out = str(viz_dataset / "test_render.png")
        result = render_detections(pixels, boxes, native_size=native, output_path=out)
        assert Path(result).is_file()
        rendered = Image.open(result)
        assert rendered.size[0] > 0

    def test_with_class_names(self, viz_dataset: Path):
        from tcip_annotation.viz import render_detections

        pixels, native = _display(str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"))
        boxes = [{"x1": 100, "y1": 100, "x2": 200, "y2": 200, "class_id": 0}]
        out = str(viz_dataset / "test_names.png")
        result = render_detections(
            pixels, boxes,
            native_size=native,
            class_names={0: "bud", 1: "nut"},
            output_path=out,
        )
        assert Path(result).is_file()

    def test_with_score(self, viz_dataset: Path):
        from tcip_annotation.viz import render_detections

        pixels, native = _display(str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"))
        boxes = [{"x1": 100, "y1": 100, "x2": 200, "y2": 200, "class_id": 0, "score": 0.95}]
        out = str(viz_dataset / "test_conf.png")
        result = render_detections(pixels, boxes, native_size=native, output_path=out)
        assert Path(result).is_file()

    def test_a_shape_without_its_class_and_a_candidate_without_its_score_refuse_by_name(
            self, tmp_path: Path):
        """A renderer's inputs are required: a box stating no ``class_id`` and a candidate
        stating no ``score`` refuse naming the key rather than drawing as class 0 or score 0."""
        from tcip_annotation.viz import render_candidates, render_detections

        frame = Image.new("RGB", (40, 40))
        with pytest.raises(KeyError, match="class_id"):
            render_detections(frame, [{"x1": 1, "y1": 1, "x2": 9, "y2": 9}], native_size=(40, 40),
                              output_path=str(tmp_path / "box.png"))
        candidate = {"candidate_id": 0, "rings": [[(1, 1), (9, 1), (9, 9)]], "bbox": [1, 1, 8, 8],
                     "area": 32}
        with pytest.raises(KeyError, match="score"):
            render_candidates(frame, [candidate], native_size=(40, 40),
                              output_path=str(tmp_path / "candidate.png"))
        assert Path(render_candidates(frame, [{**candidate, "score": 0.5}], native_size=(40, 40),
                                      output_path=str(tmp_path / "admitted.png"))).is_file()

    def test_a_pil_frame_renders_the_same_as_its_own_pixels(self, viz_dataset: Path):
        """Both input forms are drawn on, and the caller's own frame is never mutated."""
        from tcip_annotation.viz import render_detections

        pixels, native = _display(str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"))
        boxes = [{"x1": 100, "y1": 100, "x2": 200, "y2": 200, "class_id": 0}]
        frame = Image.fromarray(pixels, mode="RGB")
        from_array = str(viz_dataset / "from_array.png")
        from_pil = str(viz_dataset / "from_pil.png")
        render_detections(pixels, boxes, native_size=native, output_path=from_array)
        render_detections(frame, boxes, native_size=native, output_path=from_pil)

        assert Path(from_array).read_bytes() == Path(from_pil).read_bytes()
        assert np.array_equal(np.asarray(frame), pixels)  # the caller's frame is untouched

    def test_pixels_that_are_not_uint8_rgb_are_refused(self, viz_dataset: Path):
        from tcip_annotation.viz import render_detections

        with pytest.raises(ValueError, match="uint8"):
            render_detections(np.zeros((8, 8, 5), dtype=np.uint16), [], native_size=(8, 8),
                              output_path=str(viz_dataset / "refused.png"))

    def test_a_native_coordinate_lands_at_its_scaled_position(self, tmp_path: Path):
        """Annotations are authored in the raster's own frame, so a renderer handed reduced
        pixels scales each coordinate by the served size over ``native_size``."""
        from tcip_annotation.viz import render_detections

        native = (400, 200)
        served = Image.new("RGB", (200, 100), (80, 80, 80))
        out = str(tmp_path / "scaled.png")
        render_detections(served, [{"x1": 100, "y1": 50, "x2": 300, "y2": 150, "class_id": 0}],
                          native_size=native, output_path=out)
        px = Image.open(out).convert("RGB")

        def red_at(xy):
            r, g, _b = px.getpixel(xy)
            return r - g

        assert red_at((50, 50)) > 100     # the box edge at half of its native x
        assert red_at((100, 50)) < 20     # nothing where the unscaled coordinate would have put it


class TestRenderSegmentations:
    def test_basic_render(self, viz_dataset: Path):
        from tcip_annotation.viz import render_segmentations

        pixels, native = _display(str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"))
        polys = [
            {"rings": [[(100, 100), (200, 100), (200, 200), (100, 200)]], "class_id": 0},
        ]
        out = str(viz_dataset / "test_seg.png")
        result = render_segmentations(pixels, polys, native_size=native, output_path=out)
        assert Path(result).is_file()

    def test_renders_every_ring_of_an_occlusion_split_instance(self, viz_dataset: Path):
        """An instance's rings all get drawn, and it is labeled once, not once per contour."""
        from tcip_annotation.viz import render_segmentations

        pixels, native = _display(str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"))
        polys = [{"rings": [[(50, 50), (120, 50), (120, 150), (50, 150)],
                            [(300, 60), (380, 60), (380, 140), (300, 140)]],
                  "class_id": 0}]
        one_ring = [{"rings": [polys[0]["rings"][0]], "class_id": 0}]

        both = str(viz_dataset / "seg_two_rings.png")
        first_only = str(viz_dataset / "seg_one_ring.png")
        render_segmentations(pixels, polys, native_size=native, output_path=both)
        render_segmentations(pixels, one_ring, native_size=native, output_path=first_only)

        # The second ring really is painted: the two renders differ.
        assert Path(both).read_bytes() != Path(first_only).read_bytes()

    def test_a_polygon_stating_no_rings_refuses_and_one_with_none_drawn_admits(
            self, viz_dataset: Path):
        """An entry naming no ``rings`` refuses by name; one whose ring list is empty renders."""
        from tcip_annotation.viz import render_segmentations

        pixels, native = _display(str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"))
        out = str(viz_dataset / "seg_empty.png")
        with pytest.raises(KeyError, match="rings"):
            render_segmentations(pixels, [{"class_id": 0}], native_size=native, output_path=out)
        assert Path(render_segmentations(pixels, [{"class_id": 0, "rings": []}],
                                         native_size=native, output_path=out)).is_file()


class TestRenderComparison:
    def test_basic_comparison(self, viz_dataset: Path):
        from tcip_annotation.viz import render_comparison

        pixels, native = _display(str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"))
        gt = [{"x1": 100, "y1": 100, "x2": 200, "y2": 200, "class_id": 0}]
        pred = [{"x1": 110, "y1": 110, "x2": 210, "y2": 210, "class_id": 0, "score": 0.9}]
        out = str(viz_dataset / "test_comp.png")
        result = render_comparison(pixels, gt, pred, native_size=native, output_path=out)
        assert Path(result).is_file()

    def test_a_tp_entry_draws_a_line_between_the_boxes_it_indexes(self, viz_dataset: Path):
        """``matches`` carries the one matcher's ``(gt index, prediction index)`` pairs; the line
        is resolved from ``gt_boxes``/``pred_boxes`` by those indices, not from a box pair embedded
        in the match itself."""
        from tcip_annotation.viz import render_comparison

        pixels, native = _display(str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"))
        gt = [{"x1": 100, "y1": 100, "x2": 200, "y2": 200, "class_id": 0}]
        pred = [{"x1": 110, "y1": 110, "x2": 210, "y2": 210, "class_id": 0, "score": 0.9}]
        tp = [(0, 0)]

        with_line = str(viz_dataset / "test_comp_with_line.png")
        render_comparison(pixels, gt, pred, native_size=native, matches=tp, output_path=with_line)
        without_line = str(viz_dataset / "test_comp_without_line.png")
        render_comparison(pixels, gt, pred, native_size=native, output_path=without_line)

        yellow = (255, 255, 0)
        midpoint = (155, 155)  # halfway between the gt and pred centers, (150,150) and (160,160)
        with Image.open(with_line) as im:
            assert im.convert("RGB").getpixel(midpoint) == yellow
        with Image.open(without_line) as im:
            assert im.convert("RGB").getpixel(midpoint) != yellow


class TestRenderGrid:
    def test_grid(self, viz_dataset: Path):
        from tcip_annotation.viz import render_grid

        paths = [str(viz_dataset / "images" / UNDATED_BUCKET / f"img_{i:03d}.jpg") for i in range(1, 5)]
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


# ── Vision MCP tool tests ──────────────────────────────────────────────────


class TestVisualizeAnnotations:
    def test_detect(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import visualize

        img = str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg")
        result = visualize(viz_dataset, "annotations", img, task="detect")
        assert "error" not in result
        assert Path(result["image_path"]).is_file()
        assert result["count"] == 2

    def test_with_class_names(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import visualize

        img = str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg")
        result = visualize(viz_dataset, "annotations", img, task="detect", class_names="bud,nut")
        assert "error" not in result
        assert "bud" in result["summary"] or "nut" in result["summary"]

    def test_missing_image(self, tmp_path: Path):
        from tcip_mcp.tools.vision_tools import visualize

        result = visualize(tmp_path, "annotations", "/nonexistent/image.jpg")
        assert "error" in result

    def test_no_labels(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import visualize

        # Create an image with no labels
        img = Image.new("RGB", (100, 100))
        no_label = viz_dataset / "images" / UNDATED_BUCKET / "no_label.jpg"
        img.save(no_label)
        result = visualize(viz_dataset, "annotations", str(no_label))
        assert "error" in result

    def test_unknown_source(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import visualize

        img = str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg")
        result = visualize(viz_dataset, "bogus", img)
        assert "error" in result

    def test_an_unreadable_label_returns_an_error_naming_the_document(self, viz_dataset: Path):
        """An undecodable record is refused by the shared reader itself (UnreadableLabelDocument),
        naming the document, never answered as an image with no labels."""
        from tcip_mcp.tools.vision_tools import visualize

        img = viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"
        _damage(_label_key(img))

        result = visualize(viz_dataset, "annotations", str(img))
        assert "error" in result
        assert "img_001" in result["error"]


class TestVisualizePredictions:
    def test_detect(self, viz_bucket: Path):
        from tcip_mcp.tools.vision_tools import visualize

        img = str(viz_bucket / "images" / UNDATED_BUCKET / "img_001.jpg")
        result = visualize(viz_bucket, "predictions", img, task="detect", bucket=PUBLISHED)
        assert "error" not in result, result
        assert Path(result["image_path"]).is_file()
        assert result["count"] == 2

    def test_missing_predictions(self, viz_bucket: Path):
        from tcip_mcp.tools.vision_tools import visualize

        img = Image.new("RGB", (100, 100))
        no_pred = viz_bucket / "images" / UNDATED_BUCKET / "no_pred.jpg"
        img.save(no_pred)
        result = visualize(viz_bucket, "predictions", str(no_pred), bucket=PUBLISHED)
        assert "error" in result

    def test_no_bucket_named_refuses_naming_the_parameter(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import visualize

        result = visualize(viz_dataset, "predictions", str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"))
        assert "requires bucket" in result["error"]

    def test_an_unreadable_prediction_returns_an_error_naming_the_document(self, viz_bucket: Path):
        from tcip_mcp.tools.vision_tools import visualize

        img = viz_bucket / "images" / UNDATED_BUCKET / "img_001.jpg"
        _damage(_prediction_key(viz_bucket, img))

        result = visualize(viz_bucket, "predictions", str(img), bucket=PUBLISHED)
        assert "img_001" in result["error"]

    def test_an_undecodable_bucket_record_refuses(self, viz_bucket: Path):
        from tcip_mcp.tools.vision_tools import visualize

        _damage_record(viz_bucket)

        result = visualize(viz_bucket, "predictions", str(viz_bucket / "images" / UNDATED_BUCKET / "img_001.jpg"),
                           bucket=PUBLISHED)
        assert "error" in result


def test_scoring_and_the_comparison_render_state_one_count_for_one_image(tmp_path: Path):
    """With no trait, the single-image scoring's TP/FP/FN, its per-detection tags and the
    comparison render's counts are one matching's: a detection centered in a thin crowd strip,
    which it overlaps by little of its own area, is ignored by all three."""
    pytest.importorskip("torch")
    from tcip_annotation.state import Annotation, BBox

    from tcip_mcp.tools.annotation_tools import score_predictions
    from tcip_mcp.tools.vision_tools import visualize
    from tests._chain_fixtures import published
    from tests._producer_fixtures import label_image

    (tmp_path / "images" / UNDATED_BUCKET).mkdir(parents=True)
    image = tmp_path / "images" / UNDATED_BUCKET / "img_001.jpg"
    Image.new("RGB", (640, 480), color=(100, 120, 80)).save(image)
    label_image(image, [
        Annotation(subject="bud", geometry=BBox(288, 216, 352, 264)),
        Annotation(subject="bud", geometry=BBox(0, 395, 640, 405), iscrowd=True)], 640, 480)
    published(tmp_path, PUBLISHED, [
        {"image": str(image), "width": 640, "height": 480,
         "boxes": [[288, 216, 352, 264], [100, 350, 200, 450]], "scores": [0.95, 0.9],
         "labels": [1, 1]}], scope={"subject": "bud"})

    scored = score_predictions(str(image), PUBLISHED, detail=True)
    rendered = visualize(tmp_path, "comparison", str(image), bucket=PUBLISHED)

    assert "error" not in scored and "error" not in rendered, (scored, rendered)
    tags = [d["tag"] for d in scored["detections"]]
    counted = {"tp": tags.count("tp"), "fp": tags.count("fp"), "fn": tags.count("fn")}
    assert {k: scored[k] for k in counted} == counted == {"tp": 1, "fp": 0, "fn": 0}
    assert {k: rendered[k] for k in counted} == counted


class TestVisualizeComparison:
    def test_basic(self, viz_bucket: Path):
        from tcip_mcp.tools.vision_tools import visualize

        img = str(viz_bucket / "images" / UNDATED_BUCKET / "img_001.jpg")
        result = visualize(viz_bucket, "comparison", img, bucket=PUBLISHED)
        assert "error" not in result, result
        assert Path(result["image_path"]).is_file()
        assert result["gt_count"] == 2
        assert result["pred_count"] == 2

    def test_an_image_the_bucket_did_not_predict_draws_its_ground_truth_all_missed(
            self, viz_bucket: Path):
        from tcip_annotation.state import Annotation, BBox

        from tcip_mcp.tools.vision_tools import visualize
        from tests._producer_fixtures import label_image

        img = viz_bucket / "images" / UNDATED_BUCKET / "img_005.jpg"
        Image.new("RGB", (640, 480), color=(100, 120, 80)).save(img)
        label_image(img, [Annotation(subject="bud", geometry=BBox(288, 216, 352, 264)),
                          Annotation(subject="leaf", geometry=BBox(176, 132, 208, 156))], 640, 480)

        result = visualize(viz_bucket, "comparison", str(img), bucket=PUBLISHED)

        assert "error" not in result, result
        assert (result["gt_count"], result["pred_count"]) == (2, 0)
        assert (result["tp"], result["fp"], result["fn"]) == (0, 0, 2)

    def test_an_unreadable_gt_returns_an_error_naming_the_document(self, viz_bucket: Path):
        """Same as the annotations source: the shared reader's own message, naming the
        document."""
        from tcip_mcp.tools.vision_tools import visualize

        img = viz_bucket / "images" / UNDATED_BUCKET / "img_001.jpg"
        _damage(_label_key(img))

        result = visualize(viz_bucket, "comparison", str(img), bucket=PUBLISHED)
        assert "img_001" in result["error"]

    def test_an_unreadable_prediction_returns_an_error_naming_the_document(self, viz_bucket: Path):
        from tcip_mcp.tools.vision_tools import visualize

        img = viz_bucket / "images" / UNDATED_BUCKET / "img_001.jpg"
        _damage(_prediction_key(viz_bucket, img))

        result = visualize(viz_bucket, "comparison", str(img), bucket=PUBLISHED)
        assert "img_001" in result["error"]

    def test_an_undecodable_bucket_record_refuses(self, viz_bucket: Path):
        from tcip_mcp.tools.vision_tools import visualize

        _damage_record(viz_bucket)

        result = visualize(viz_bucket, "comparison", str(viz_bucket / "images" / UNDATED_BUCKET / "img_001.jpg"),
                           bucket=PUBLISHED)
        assert "error" in result


class TestDisplayRead:
    """The one decode the visualization tools render through."""

    def test_a_large_source_is_read_down_to_the_artifact_bound(self, tmp_path: Path):
        """The artifact bound is the read's target, not a resize after a whole decode; the native
        frame the annotations live in is reported unchanged."""
        from tcip_mcp.pipelines.display_bounds import VIZ_ARTIFACT_MAX_EDGE
        from tcip_mcp.tools.vision_tools import _display_for_path

        images = tmp_path / "images" / UNDATED_BUCKET
        images.mkdir(parents=True)
        path = images / "big.jpg"
        Image.new("RGB", (VIZ_ARTIFACT_MAX_EDGE * 2, VIZ_ARTIFACT_MAX_EDGE)).save(path)

        read = _display_for_path(str(path))
        assert read.pixels.shape[:2] == (VIZ_ARTIFACT_MAX_EDGE // 2, VIZ_ARTIFACT_MAX_EDGE)
        assert read.native_size == (VIZ_ARTIFACT_MAX_EDGE * 2, VIZ_ARTIFACT_MAX_EDGE)
        assert read.scale == 0.5

    def test_a_source_within_the_bound_is_read_at_native_resolution(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import _display_for_path

        read = _display_for_path(str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"))
        assert read.pixels.shape[:2] == (480, 640)
        assert read.scale == 1.0

    def test_a_three_band_raster_that_is_not_8_bit_is_stretched_to_be_visible(self,
                                                                             tmp_path: Path):
        """Band count alone doesn't make a raster displayable: a 16-bit capture's values occupy a
        sliver of their dtype's range, and passing them through unstretched renders it black."""
        import tifffile

        from tcip_mcp.tools.vision_tools import _display_for_path

        images = tmp_path / "images" / UNDATED_BUCKET
        images.mkdir(parents=True)
        path = images / "capture.tif"
        arr = np.stack([np.linspace(100, 400, 12, dtype=np.uint16)] * 10)
        tifffile.imwrite(str(path), np.stack([arr, arr + 50, arr + 90], axis=-1))

        pixels = _display_for_path(str(path)).pixels
        assert pixels.dtype == np.uint8
        assert pixels.max() == 255

    def test_a_region_read_reports_the_rect_it_served(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import _display_for_path

        read = _display_for_path(str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"),
                                 region=(100.0, 50.0, 200.0, 150.0))
        assert (read.rect.x0, read.rect.y0, read.rect.x1, read.rect.y1) == (100, 50, 300, 200)
        assert read.pixels.shape[:2] == (150, 200)

    def test_a_region_hanging_off_the_edge_is_clamped_into_the_raster(self, viz_dataset: Path):
        """A human can pan past the image, so a viewport is clamped rather than refused."""
        from tcip_mcp.tools.vision_tools import _display_for_path

        read = _display_for_path(str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"),
                                 region=(500.0, 400.0, 400.0, 400.0))
        assert (read.rect.x1, read.rect.y1) == (640, 480)
        assert read.pixels.shape[:2] == (80, 140)


class TestVisualizeDatasetSample:
    def test_sample(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import visualize

        result = visualize(viz_dataset, "dataset", str(viz_dataset), n=4)
        assert "error" not in result
        assert Path(result["image_path"]).is_file()
        assert result["count"] == 4
        assert result["total_images"] == 4

    def test_only_the_images_of_a_capture_are_sampled(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import visualize

        Image.new("RGB", (64, 48)).save(viz_dataset / "images" / "loose.jpg")
        (viz_dataset / "images" / UNDATED_BUCKET / "nested").mkdir()
        Image.new("RGB", (64, 48)).save(viz_dataset / "images" / UNDATED_BUCKET / "nested" / "deep.jpg")

        result = visualize(viz_dataset, "dataset", str(viz_dataset), n=16)

        assert "error" not in result, result
        assert result["total_images"] == 4

    def test_no_images(self, tmp_path: Path):
        from tcip_mcp.tools.vision_tools import visualize

        (tmp_path / "images" / UNDATED_BUCKET).mkdir(parents=True)
        result = visualize(tmp_path, "dataset", str(tmp_path), n=4)
        assert "error" in result

    def test_a_corrupt_label_returns_an_error_not_an_unlabeled_render(self, viz_dataset: Path):
        """A present, unreadable label document must surface as an error, not be silently
        rendered as though the image carried no labels."""
        from tcip_mcp.tools.vision_tools import visualize

        from tcip_annotation.state import Annotation, BBox

        from tests._producer_fixtures import label_image

        bad = viz_dataset / "images" / UNDATED_BUCKET / "img_bad.jpg"
        Image.new("RGB", (640, 480), color=(100, 120, 80)).save(bad)
        label_image(bad, [Annotation(subject="bud", geometry=BBox(1, 1, 5, 5))], 640, 480)
        _damage(_label_key(bad))

        # n covers every image (5, with img_bad): sampling is otherwise random, and the corrupt
        # image must be reached deterministically for this assertion.
        result = visualize(viz_dataset, "dataset", str(viz_dataset), n=5)
        assert "error" in result
        assert "img_bad" in result["error"]

    def test_an_unlabeled_multiband_sample_is_a_rendered_cell(self, tmp_path: Path, monkeypatch):
        """Every grid cell is a rendered artifact, labels or not: the grid tiles renders, and a
        raw source path in that list is one the tiler has to decode itself."""
        import tifffile

        from tcip_mcp.tools import vision_tools

        images = tmp_path / "images" / UNDATED_BUCKET
        images.mkdir(parents=True)
        rng = np.random.default_rng(5)
        src = images / "capture.tif"
        tifffile.imwrite(str(src), rng.integers(0, 4096, size=(24, 20, 6)).astype(np.uint16))

        tiled: list[list[str]] = []
        real_grid = vision_tools.render_grid

        def spy(image_paths, **kwargs):
            tiled.append(list(image_paths))
            return real_grid(image_paths, **kwargs)

        monkeypatch.setattr(vision_tools, "render_grid", spy)
        result = vision_tools.visualize(tmp_path, "dataset", str(tmp_path), n=1)

        assert "error" not in result
        viz_dir = (tmp_path / ".tcip" / "artifacts" / "viz").resolve()
        assert tiled and tiled[0]
        for cell in tiled[0]:
            assert Path(cell).parent == viz_dir     # a render, not the source or a temp preview
            assert Path(cell).is_file()


class TestVisualizeWorstPredictions:
    def test_basic(self, viz_bucket: Path):
        from tcip_mcp.tools.vision_tools import render_failure_cases

        result = render_failure_cases(viz_bucket, str(viz_bucket), PUBLISHED, top_k=3)
        assert "error" not in result
        # Should have rendered some cases
        assert len(result.get("case_images", [])) > 0


# ── Candidate and grid renders ──────────────────────────────────────────────


class TestRenderCandidates:
    def test_basic_render(self, viz_dataset: Path):
        from tcip_annotation.viz import render_candidates

        pixels, native = _display(str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"))
        candidates = [
            {
                "candidate_id": 0,
                "bbox": [100.0, 100.0, 200.0, 200.0],
                "area": 10000,
                "score": 0.95,
                "rings": [[(100, 100), (200, 100), (200, 200), (100, 200)]],
            },
            {
                "candidate_id": 1,
                "bbox": [300.0, 300.0, 400.0, 400.0],
                "area": 5000,
                "score": 0.88,
                "rings": [[(300, 300), (400, 300), (400, 400), (300, 400)]],
            },
        ]
        out = render_candidates(pixels, candidates, native_size=native,
                                output_path=str(viz_dataset / "candidates.png"))
        assert Path(out).is_file()

    def test_empty_candidates(self, viz_dataset: Path):
        from tcip_annotation.viz import render_candidates

        pixels, native = _display(str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"))
        out = render_candidates(pixels, [], native_size=native,
                                output_path=str(viz_dataset / "no_candidates.png"))
        assert Path(out).is_file()


def _uniform_cells(width: int, height: int, tile: int) -> list:
    """The clamped reference cells for a frame, the list every cells-in consumer takes."""
    from tcip_mcp.pipelines.reference_grid import reference_cells

    return reference_cells(width, height, tile, clamp=True)


class TestRenderGridOverlay:
    def test_basic(self, viz_dataset: Path):
        from tcip_annotation.viz import render_grid_overlay

        pixels, native = _display(str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"))
        out = render_grid_overlay(pixels, _uniform_cells(*native, 80), native_size=native,
                                  output_path=str(viz_dataset / "grid.png"))
        assert Path(out).is_file()

    def test_cell_dicts_accepted(self, viz_dataset: Path):
        """The renderer takes the plain dicts a JSON route serves as well as cell objects."""
        from tcip_annotation.viz import render_grid_overlay

        pixels, native = _display(str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"))
        cells = [{"name": c.name, "x0": c.x0, "y0": c.y0, "x1": c.x1, "y1": c.y1}
                 for c in _uniform_cells(*native, 160)]
        out = render_grid_overlay(pixels, cells, native_size=native,
                                  output_path=str(viz_dataset / "grid_cells.png"))
        assert Path(out).is_file()

    def test_grid_wider_than_alphabet(self, viz_dataset: Path):
        from tcip_annotation.viz import render_grid_overlay

        pixels, _native = _display(str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"))
        out = render_grid_overlay(pixels, _uniform_cells(3300, 400, 100),
                                  native_size=(3300, 400),
                                  output_path=str(viz_dataset / "grid_wide.png"))
        assert Path(out).is_file()

    def test_empty_cells_refused(self, viz_dataset: Path):
        from tcip_annotation.viz import render_grid_overlay

        pixels, native = _display(str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"))
        with pytest.raises(ValueError, match="empty"):
            render_grid_overlay(pixels, [], native_size=native,
                                output_path=str(viz_dataset / "grid_empty.png"))

    def test_lines_land_on_supplied_cell_boundaries(self, tmp_path: Path):
        """The renderer draws the caller's own cell rects: on a clamped 100x80 grid at
        tile 64 the interior boundaries sit at 64, not at the uniform division (50/40) a
        renderer that split the frame evenly by cell count would draw."""
        from tcip_annotation.viz import render_grid_overlay

        pixels = np.zeros((80, 100, 3), dtype=np.uint8)
        out = render_grid_overlay(pixels, _uniform_cells(100, 80, 64),
                                  native_size=(100, 80),
                                  output_path=str(tmp_path / "grid.png"))
        arr = np.asarray(Image.open(out))
        yellow = (arr[:, :, 0] > 200) & (arr[:, :, 1] > 200) & (arr[:, :, 2] < 80)
        assert yellow[:, 64].sum() >= 60, "no vertical line on the supplied boundary x=64"
        assert yellow[64, :].sum() >= 75, "no horizontal line on the supplied boundary y=64"
        # Away from the true boundaries and the A1 label, the uniform-division positions
        # (x=50, y=40) hold no line.
        assert yellow[25:60, 50].sum() == 0, "a line at x=50 is the uniform division"
        assert yellow[40, 25:60].sum() == 0, "a line at y=40 is the uniform division"
        # The outer boundaries render inside the frame: a far edge scaling to the frame
        # size must pin to the last pixel row/column, not clip away.
        assert yellow[:, 0].sum() >= 60, "no left outer boundary at x=0"
        assert yellow[0, :].sum() >= 75, "no top outer boundary at y=0"
        assert yellow[:, 99].sum() >= 60, "no right outer boundary at x=99"
        assert yellow[79, :].sum() >= 75, "no bottom outer boundary at y=79"


class TestGridToRect:
    def test_a_name_resolves_to_its_cells_rect(self):
        from tcip_annotation.grid import grid_to_rect

        assert grid_to_rect("A1", _uniform_cells(640, 480, 80)) == (0.0, 0.0, 80.0, 80.0)
        assert grid_to_rect("H6", _uniform_cells(640, 480, 80)) == (560.0, 400.0, 640.0, 480.0)

    def test_case_insensitive(self):
        from tcip_annotation.grid import grid_to_rect

        cells = _uniform_cells(640, 480, 80)
        assert grid_to_rect(" b3 ", cells) == grid_to_rect("B3", cells)

    def test_a_clamped_edge_cell_is_its_own_clipped_rect(self):
        from tcip_annotation.grid import grid_to_rect

        assert grid_to_rect("B2", _uniform_cells(100, 80, 64)) == (64.0, 64.0, 100.0, 80.0)

    def test_cell_dicts_accepted(self):
        """The lookup takes the plain dicts a JSON route serves as well as cell objects."""
        from tcip_annotation.grid import grid_to_rect

        cells = [{"name": c.name, "x0": c.x0, "y0": c.y0, "x1": c.x1, "y1": c.y1}
                 for c in _uniform_cells(640, 480, 80)]
        assert grid_to_rect("C2", cells) == (160.0, 80.0, 240.0, 160.0)

    def test_unknown_column_names_the_valid_range(self):
        from tcip_annotation.grid import grid_to_rect

        with pytest.raises(ValueError, match="Use A1 through H6"):
            grid_to_rect("Z1", _uniform_cells(640, 480, 80))

    def test_unknown_row_names_the_valid_range(self):
        from tcip_annotation.grid import grid_to_rect

        with pytest.raises(ValueError, match="Use A1 through H6"):
            grid_to_rect("A9", _uniform_cells(640, 480, 80))

    def test_malformed_reference_is_invalid(self):
        from tcip_annotation.grid import grid_to_rect

        with pytest.raises(ValueError, match="Invalid cell reference"):
            grid_to_rect("3B", _uniform_cells(640, 480, 80))


class TestColumnLabels:
    """Spreadsheet-style column labels shared by the grid geometry and cell lookup."""

    def test_round_trip_boundaries(self):
        from tcip_annotation.grid import column_index, column_label

        expected = {0: "A", 25: "Z", 26: "AA", 27: "AB", 31: "AF", 32: "AG", 51: "AZ", 52: "BA"}
        for idx, label in expected.items():
            assert column_label(idx) == label
            assert column_index(label) == idx

    def test_multi_letter_cell_parses(self):
        from tcip_annotation.grid import grid_to_rect

        assert grid_to_rect("AA1", _uniform_cells(2700, 480, 100))[:2] == (2600.0, 0.0)

    def test_high_columns_do_not_alias(self):
        # chr() past 'Z' labeled column 32 'a', which case folding silently parsed as column 0
        from tcip_annotation.grid import column_label, grid_to_rect

        cols = 40
        labels = [column_label(c) for c in range(cols)]
        assert len(set(labels)) == cols
        cells = _uniform_cells(cols * 100, 480, 100)
        for c, label in enumerate(labels):
            assert grid_to_rect(f"{label}1", cells)[0] == c * 100

    def test_out_of_range_hint_uses_column_labels(self):
        from tcip_annotation.grid import grid_to_rect

        with pytest.raises(ValueError, match="Use A1 through AD6"):
            grid_to_rect("BA1", _uniform_cells(3000, 600, 100))


class TestVisualizeGridOverlayTool:
    def test_derived_default_echoes_geometry(self, viz_dataset: Path):
        """With no tile_size the tool derives the pointing grain and still echoes the full
        geometry, so the caller can hand it straight to propose_annotations."""
        from tcip_mcp.pipelines.reference_grid import derive_pointing_tile_size
        from tcip_mcp.tools.vision_tools import overlay_reference_grid

        result = overlay_reference_grid(
            viz_dataset, image_path=str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"),
        )
        assert "error" not in result
        assert Path(result["image_path"]).is_file()
        assert result["width"] == 640
        assert result["height"] == 480
        assert result["tile_size"] == derive_pointing_tile_size(640, 480)
        assert result["overlap"] == 0.0
        assert result["cols"] >= 1 and result["rows"] >= 1

    def test_explicit_tile_size_echoes_back(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import overlay_reference_grid

        result = overlay_reference_grid(
            viz_dataset, image_path=str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"), tile_size=80,
        )
        assert "error" not in result
        assert result["tile_size"] == 80
        assert result["cols"] == 8
        assert result["rows"] == 6

    def test_missing_image(self, tmp_path: Path):
        from tcip_mcp.tools.vision_tools import overlay_reference_grid

        result = overlay_reference_grid(tmp_path, image_path="/nonexistent.jpg")
        assert "error" in result

    def test_invalid_tile_size_is_an_error(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import overlay_reference_grid

        result = overlay_reference_grid(
            viz_dataset, image_path=str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"), tile_size=0,
        )
        assert "error" in result
        assert "tile_size" in result["error"]

    def test_wide_grid_summary_labels(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import overlay_reference_grid

        result = overlay_reference_grid(
            viz_dataset, image_path=str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"), tile_size=20,
        )
        assert "error" not in result
        assert result["cols"] == 32
        assert "'AF24'" in result["summary"]


class TestProposeAnnotationsTool:
    """Test propose_annotations tool (mocked engine)."""

    def test_missing_image(self, tmp_path: Path):
        from tcip_mcp.tools.proposal_tools import propose_annotations

        result = propose_annotations(tmp_path, image_path="/nonexistent.jpg", engine="stub")
        assert "error" in result

    def test_unknown_engine(self, viz_dataset: Path):
        from tcip_mcp.tools.proposal_tools import propose_annotations

        result = propose_annotations(
            viz_dataset, image_path=str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"),
            engine="does_not_exist",
        )
        assert "error" in result
        assert "does_not_exist" in result["error"]


class TestAcceptProposalsTool:
    def test_no_prior_proposals(self, viz_dataset: Path):
        from tcip_mcp.tools.proposal_tools import stage_proposals

        result = stage_proposals(
            viz_dataset, image_path=str(viz_dataset / "images" / UNDATED_BUCKET / "img_003.jpg"),
            assignments=[{"candidate_id": 0, "subject": "bud"}],
        )
        assert "error" in result
        assert "Run propose_annotations first" in result["error"]

    def test_with_cached_proposals(self, viz_dataset: Path, monkeypatch: pytest.MonkeyPatch):
        from tcip_mcp.pipelines import proposal
        from tcip_mcp.tools.proposal_tools import stage_proposals, propose_annotations

        # The candidates propose_annotations stages, in the neutral engine schema.
        candidates = [
            {
                "candidate_id": 0,
                "bbox": [100.0, 100.0, 200.0, 200.0],
                "area": 10000,
                "score": 0.90,
                "engine": "stub",
                "engine_meta": {"stability_score": 0.95, "predicted_iou": 0.90},
                "rings": [[[100, 100], [200, 100], [200, 200], [100, 200]]],
            },
            {
                "candidate_id": 1,
                "bbox": [300.0, 300.0, 400.0, 400.0],
                "area": 5000,
                "score": 0.85,
                "engine": "stub",
                "engine_meta": {"stability_score": 0.88, "predicted_iou": 0.85},
                "rings": [[[300, 300], [400, 300], [400, 400], [300, 400]]],
            },
        ]

        class StubProposer:
            def propose(self, image_path: str, **params: object) -> list[dict]:
                return candidates

        monkeypatch.setattr(proposal, "resolve_proposer", lambda engine: StubProposer())

        propose_result = propose_annotations(
            viz_dataset, image_path=str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"), engine="stub")
        assert "error" not in propose_result, propose_result
        assert propose_result["staged"] is True

        result = stage_proposals(
            viz_dataset, image_path=str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"),
            assignments=[
                {"candidate_id": 0, "subject": "bud"},
                {"candidate_id": 1, "subject": "nut"},
            ],
        )
        assert "error" not in result
        assert result["proposal_count"] == 2
        assert Path(result["image_path"]).is_file()

        # Masks are staged as the engine's predictions, not GT: one per-image document holding
        # both accepted objects by subject name.
        anns, objs = _staged(viz_dataset, viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg", result)
        assert len(anns) == 2
        assert {a.subject for a in anns} == {"bud", "nut"}

        # Each staged object is the engine's output: created_by="stub" and a numeric score.
        assert objs and all(o["created_by"] == "stub" for o in objs)
        assert all(isinstance(o["score"], float) for o in objs)


# ── Integration over the full pipeline with a stub engine's candidates ─────


MOCK_CANDIDATES: list[dict[str, Any]] = [
    {
        "candidate_id": 0,
        "bbox": [50.0, 40.0, 200.0, 180.0],
        "area": 21000,
        "score": 0.93,
        "engine": "stub",
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
        "engine": "stub",
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
        "engine": "stub",
        "engine_meta": {"stability_score": 0.85, "predicted_iou": 0.80},
        "rings": [[
            (500, 100), (600, 100), (600, 200), (500, 200),
        ]],
    },
]


def _staged(project: Path, image: Path, staged: dict) -> tuple[list, list[dict]]:
    """The document ``staged``'s bucket holds for ``image``: its annotations as the reader
    decodes them and its records as the store holds them."""
    import tcip_store
    from tcip_annotation.json_io import read_label_document

    from tcip_mcp.buckets import read_bucket

    key = read_bucket(project, staged["bucket"]).document_key(image.stem)
    assert key is not None, staged
    return read_label_document(key).annotations, tcip_store.read(key)["annotations"]


class TestCandidateCacheRoundTrip:
    """Verify candidates survive JSON serialize → deserialize."""

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
            # Every ring, every vertex: JSON turns tuples into lists
            assert len(orig["rings"]) == len(restored["rings"])
            for o_ring, r_ring in zip(orig["rings"], restored["rings"], strict=True):
                for (ox, oy), rp in zip(o_ring, r_ring, strict=True):
                    rx, ry = rp if isinstance(rp, (list, tuple)) else (rp["x"], rp["y"])
                    assert abs(ox - rx) < 1e-6
                    assert abs(oy - ry) < 1e-6

    def test_empty_candidates_roundtrip(self, tmp_path: Path):
        state_file = tmp_path / "candidates_empty.json"
        state_file.write_text(json.dumps([]), encoding="utf-8")
        loaded = json.loads(state_file.read_text(encoding="utf-8"))
        assert loaded == []


class TestFullPipelineIntegration:
    """End-to-end flow: mock candidates → accept → verify output files."""

    @pytest.fixture
    def pipeline_dataset(self, tmp_path: Path) -> Path:
        """Fresh dataset without pre-existing labels."""
        images_dir = tmp_path / "images" / UNDATED_BUCKET
        images_dir.mkdir(parents=True)
        img = Image.new("RGB", (640, 480), color=(80, 120, 60))
        img.save(images_dir / "sample.jpg")
        return tmp_path

    def _propose(
        self, monkeypatch: pytest.MonkeyPatch, project: Path, image_path: str,
        candidates: list[dict],
    ) -> None:
        """Stage ``candidates`` for ``project`` through the real writer: a stub engine that hands
        them back verbatim, driven by an actual ``propose_annotations`` call."""
        from tcip_mcp.pipelines import proposal
        from tcip_mcp.tools.proposal_tools import propose_annotations

        class StubProposer:
            def propose(self, image_path: str, **params: object) -> list[dict]:
                return candidates

        monkeypatch.setattr(proposal, "resolve_proposer", lambda engine: StubProposer())
        result = propose_annotations(project, image_path=image_path, engine="stub")
        assert "error" not in result, result

    def test_accept_writes_json_detect(
        self, pipeline_dataset: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Engine proposals are staged as predictions: pixel geometry, subject names, and score
        preserved."""
        from tcip_annotation import bbox_of
        from tcip_mcp.tools.proposal_tools import stage_proposals

        img_path = str(pipeline_dataset / "images" / UNDATED_BUCKET / "sample.jpg")
        self._propose(monkeypatch, pipeline_dataset, img_path, MOCK_CANDIDATES)

        result = stage_proposals(
            pipeline_dataset, image_path=img_path,
            assignments=[
                {"candidate_id": 0, "subject": "bud"},
                {"candidate_id": 1, "subject": "nut"},
            ],
        )
        assert "error" not in result
        assert result["proposal_count"] == 2

        anns, objs = _staged(pipeline_dataset, Path(img_path), result)
        assert len(anns) == 2
        assert {a.subject for a in anns} == {"bud", "nut"}
        # Pixel coords within the 640x480 image; staged as the engine's (created_by="stub").
        for a in anns:
            b = bbox_of(a.geometry)
            assert 0.0 <= b.x1 < b.x2 <= 640.0
            assert 0.0 <= b.y1 < b.y2 <= 480.0
            assert a.created_by == "stub"
        assert all(isinstance(o["score"], float) for o in objs)

    def test_accept_writes_json_segment(
        self, pipeline_dataset: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Engine proposals are staged as prediction polygons: pixel vertices, subject, score."""
        from tcip_mcp.tools.proposal_tools import stage_proposals

        img_path = str(pipeline_dataset / "images" / UNDATED_BUCKET / "sample.jpg")
        self._propose(monkeypatch, pipeline_dataset, img_path, MOCK_CANDIDATES)

        result = stage_proposals(
            pipeline_dataset, image_path=img_path,
            assignments=[{"candidate_id": 0, "subject": "bud"}],
        )
        assert "error" not in result
        assert result["proposal_count"] == 1

        anns, objs = _staged(pipeline_dataset, Path(img_path), result)
        assert len(anns) == 1
        assert {a.subject for a in anns} == {"bud"}
        rings = anns[0].geometry.rings
        assert rings and all(len(r) >= 3 for r in rings)
        for x, y in (pt for ring in rings for pt in ring):
            assert 0.0 <= x <= 640.0
            assert 0.0 <= y <= 480.0
        assert objs and all(o["created_by"] == "stub" for o in objs)
        assert all(isinstance(o["score"], float) for o in objs)

    def test_detect_and_segment_consistent(
        self, pipeline_dataset: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Box and mask views of the staged predictions cover the same objects (one document)."""
        from tcip_annotation import bbox_of
        from tcip_mcp.tools.proposal_tools import stage_proposals

        img_path = str(pipeline_dataset / "images" / UNDATED_BUCKET / "sample.jpg")
        self._propose(monkeypatch, pipeline_dataset, img_path, MOCK_CANDIDATES)

        result = stage_proposals(
            pipeline_dataset, image_path=img_path,
            assignments=[
                {"candidate_id": 0, "subject": "bud"},
                {"candidate_id": 2, "subject": "nut"},
            ],
        )
        assert result["proposal_count"] == 2

        anns, _objs = _staged(pipeline_dataset, Path(img_path), result)
        # Each object is one polygon with a derivable box under the same subject: the box and mask
        # views can never diverge because they are the same annotations.
        subjects_poly = sorted(a.subject for a in anns if a.geometry is not None)
        subjects_box = sorted(a.subject for a in anns if bbox_of(a.geometry) is not None)
        assert subjects_poly == subjects_box == ["bud", "nut"]

    def test_partial_accept_skips_rejected(
        self, pipeline_dataset: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Only accepted candidates appear in output; rejected are omitted."""
        from tcip_mcp.tools.proposal_tools import stage_proposals

        img_path = str(pipeline_dataset / "images" / UNDATED_BUCKET / "sample.jpg")
        self._propose(monkeypatch, pipeline_dataset, img_path, MOCK_CANDIDATES)

        # Accept only candidate 1 out of 3
        result = stage_proposals(
            pipeline_dataset, image_path=img_path,
            assignments=[{"candidate_id": 1, "subject": "bud"}],
        )
        assert result["proposal_count"] == 1

    def test_invalid_candidate_id_silently_skipped(
        self, pipeline_dataset: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Assignments with non-existent candidate_id are ignored."""
        from tcip_mcp.tools.proposal_tools import stage_proposals

        img_path = str(pipeline_dataset / "images" / UNDATED_BUCKET / "sample.jpg")
        self._propose(monkeypatch, pipeline_dataset, img_path, MOCK_CANDIDATES)

        result = stage_proposals(
            pipeline_dataset, image_path=img_path,
            assignments=[
                {"candidate_id": 999, "subject": "bud"},  # non-existent
                {"candidate_id": 0, "subject": "nut"},        # valid
            ],
        )
        assert result["proposal_count"] == 1

    def test_render_then_accept_pipeline(
        self, pipeline_dataset: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Full render → accept → verify pipeline."""
        from tcip_annotation.viz import render_candidates, render_grid_overlay
        from tcip_mcp.tools.proposal_tools import stage_proposals

        img_path = str(pipeline_dataset / "images" / UNDATED_BUCKET / "sample.jpg")
        pixels, native = _display(img_path)

        # Step 1: Render candidates
        candidate_render = render_candidates(
            pixels, MOCK_CANDIDATES, native_size=native,
            output_path=str(pipeline_dataset / "candidates.png"))
        assert Path(candidate_render).is_file()

        # Step 2: Render grid overlay (for correction reference)
        grid_render = render_grid_overlay(pixels, _uniform_cells(*native, 80),
                                          native_size=native,
                                          output_path=str(pipeline_dataset / "grid.png"))
        assert Path(grid_render).is_file()

        # Step 3: propose_annotations stages the candidates for real
        self._propose(monkeypatch, pipeline_dataset, img_path, MOCK_CANDIDATES)

        # Step 4: Accept with subject assignments
        result = stage_proposals(
            pipeline_dataset, image_path=img_path,
            assignments=[
                {"candidate_id": 0, "subject": "bud"},
                {"candidate_id": 1, "subject": "nut"},
                {"candidate_id": 2, "subject": "bud"},
            ],
        )
        assert "error" not in result
        assert result["proposal_count"] == 3

        # Step 5: QA render was produced
        assert Path(result["image_path"]).is_file()


class TestEnginePredictionStaging:
    """stage_proposals stages engine masks as predictions in a bucket of the engine's, not ground
    truth."""

    @pytest.fixture
    def format_dataset(self, tmp_path: Path) -> Path:
        images_dir = tmp_path / "images" / UNDATED_BUCKET
        images_dir.mkdir(parents=True)
        img = Image.new("RGB", (640, 480), color=(100, 100, 100))
        img.save(images_dir / "fmt_test.jpg")
        return tmp_path

    def _propose(self, monkeypatch: pytest.MonkeyPatch, project: Path, image_path: str) -> None:
        """Stage MOCK_CANDIDATES for ``project`` through the real writer: a stub engine that
        hands them back verbatim, driven by an actual ``propose_annotations`` call."""
        from tcip_mcp.pipelines import proposal
        from tcip_mcp.tools.proposal_tools import propose_annotations

        class StubProposer:
            def propose(self, image_path: str, **params: object) -> list[dict]:
                return MOCK_CANDIDATES

        monkeypatch.setattr(proposal, "resolve_proposer", lambda engine: StubProposer())
        result = propose_annotations(project, image_path=image_path, engine="stub")
        assert "error" not in result, result

    def test_json_detect_and_segment_written(
        self, format_dataset: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        from tcip_annotation import bbox_of
        from tcip_mcp.tools.proposal_tools import stage_proposals

        img_path = str(format_dataset / "images" / UNDATED_BUCKET / "fmt_test.jpg")
        self._propose(monkeypatch, format_dataset, img_path)
        result = stage_proposals(
            format_dataset, image_path=img_path,
            assignments=[{"candidate_id": 0, "subject": "bud"}],
        )
        assert "error" not in result
        assert "format" not in result
        assert result["proposal_count"] == 1

        anns, _objs = _staged(format_dataset, Path(img_path), result)
        assert len(anns) == 1 and {a.subject for a in anns} == {"bud"}
        # The staged object carries a polygon (mask) with a derivable box: both views of one object.
        assert anns[0].geometry is not None
        assert bbox_of(anns[0].geometry) is not None

    def test_prediction_carries_the_engine_score(
        self, format_dataset: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Staged engine output is a prediction: each object has created_by="stub" and a
        ``score``."""
        from tcip_mcp.tools.proposal_tools import stage_proposals

        img_path = str(format_dataset / "images" / UNDATED_BUCKET / "fmt_test.jpg")
        self._propose(monkeypatch, format_dataset, img_path)
        staged = stage_proposals(
            format_dataset, image_path=img_path,
            assignments=[{"candidate_id": 0, "subject": "bud"}],
        )
        _anns, objs = _staged(format_dataset, Path(img_path), staged)
        assert objs
        assert all(o["created_by"] == "stub" for o in objs)
        assert all(isinstance(o["score"], float) for o in objs)

