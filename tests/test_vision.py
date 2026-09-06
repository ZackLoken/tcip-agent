"""Tests for the vision rendering engine and MCP tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image


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
            [Annotation(subject="bud", geometry=BBox(288, 216, 352, 264)),
             Annotation(subject="nut", geometry=BBox(176, 132, 208, 156))],
            640, 480,
        )
        # Per-image JSON predictions carry a score.
        json_io.write_annotations(
            preds_dir / f"{name}.json",
            [Annotation(subject="bud", geometry=BBox(288, 216, 352, 264), score=0.95),
             Annotation(subject="nut", geometry=BBox(496, 372, 528, 396), score=0.6)],
            640, 480,
        )

    return tmp_path


def _seed_sidecar(pred_dir: Path, sidecar: dict) -> None:
    """A bucket's own stamp, written straight through the store: the rail refuses a fresh
    write_sidecar call missing the (subject, attribute) pair."""
    import tcip_store
    from tcip_mcp.pipelines.resolution import sidecar_key

    tcip_store.replace(sidecar_key(pred_dir, "operating_point"), sidecar, expect=tcip_store.Version.ABSENT)


def _damage_sidecar(pred_dir: Path) -> None:
    """Corrupt a bucket's already-seeded stamp in place, wherever the bound backend keeps it."""
    import os

    from tcip_mcp.pipelines.resolution import sidecar_key
    from tcip_store.binding import BACKEND_ENV, DEFAULT_BACKEND, FILE_BACKEND
    from tcip_store.store import _backend

    key = sidecar_key(pred_dir, "operating_point")
    if (os.environ.get(BACKEND_ENV) or DEFAULT_BACKEND) == FILE_BACKEND:
        _backend().path_for(key).write_bytes(b"{not json")
        return
    import sqlite3

    from tcip_store.sqlite_backend import database_path, encode_parts

    conn = sqlite3.connect(str(database_path(str(key.root))), isolation_level=None)
    try:
        conn.execute("update records set value = ? where store = ? and parts = ?",
                    (b"{not json", key.store, encode_parts(key.parts)))
    finally:
        conn.close()


# â”€â”€ Rendering engine tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def test_default_output_resolves_under_the_platform_state_root(tmp_path, monkeypatch):
    # Must resolve under TCIP_STATE_ROOT, not the process CWD (often the repo, which
    # fragmented renders away from the project); the returned path is absolute.
    from tcip_annotation.viz import _default_output

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    out = Path(_default_output("detections"))
    assert out.is_absolute()
    assert out.parent == (tmp_path / ".tcip" / "artifacts" / "viz").resolve()


def test_default_output_falls_back_to_cwd_when_unset(tmp_path, monkeypatch):
    from tcip_annotation.viz import _default_output

    monkeypatch.delenv("TCIP_STATE_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    out = Path(_default_output("detections"))
    assert out.parent == (tmp_path / ".tcip" / "artifacts" / "viz").resolve()


class TestRenderDetections:
    def test_basic_render(self, viz_dataset: Path):
        from tcip_annotation.viz import render_detections

        pixels, native = _display(str(viz_dataset / "images" / "img_001.jpg"))
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

        pixels, native = _display(str(viz_dataset / "images" / "img_001.jpg"))
        boxes = [{"x1": 100, "y1": 100, "x2": 200, "y2": 200, "class_id": 0}]
        out = str(viz_dataset / "test_names.png")
        result = render_detections(
            pixels, boxes,
            native_size=native,
            class_names={0: "bud", 1: "nut"},
            output_path=out,
        )
        assert Path(result).is_file()

    def test_with_confidence(self, viz_dataset: Path):
        from tcip_annotation.viz import render_detections

        pixels, native = _display(str(viz_dataset / "images" / "img_001.jpg"))
        boxes = [{"x1": 100, "y1": 100, "x2": 200, "y2": 200, "class_id": 0, "confidence": 0.95}]
        out = str(viz_dataset / "test_conf.png")
        result = render_detections(pixels, boxes, native_size=native, output_path=out)
        assert Path(result).is_file()

    def test_a_pil_frame_renders_the_same_as_its_own_pixels(self, viz_dataset: Path):
        """Both input forms are drawn on, and the caller's own frame is never mutated."""
        from tcip_annotation.viz import render_detections

        pixels, native = _display(str(viz_dataset / "images" / "img_001.jpg"))
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
            render_detections(np.zeros((8, 8, 5), dtype=np.uint16), [], native_size=(8, 8))

    def test_a_native_coordinate_lands_at_its_scaled_position(self, tmp_path: Path):
        """Annotations are authored in the raster's own frame, so a renderer handed reduced pixels
        places them by ``native_size``: the scaling the internal decode used to do."""
        from tcip_annotation.viz import render_detections

        native = (400, 200)
        served = Image.new("RGB", (200, 100), (80, 80, 80))
        out = str(tmp_path / "scaled.png")
        render_detections(served, [{"x1": 100, "y1": 50, "x2": 300, "y2": 150, "class_id": 0}],
                          native_size=native, output_path=out, conf_key=None)
        px = Image.open(out).convert("RGB")

        def red_at(xy):
            r, g, _b = px.getpixel(xy)
            return r - g

        assert red_at((50, 50)) > 100     # the box edge at half of its native x
        assert red_at((100, 50)) < 20     # nothing where the unscaled coordinate would have put it


class TestRenderSegmentations:
    def test_basic_render(self, viz_dataset: Path):
        from tcip_annotation.viz import render_segmentations

        pixels, native = _display(str(viz_dataset / "images" / "img_001.jpg"))
        polys = [
            {"rings": [[(100, 100), (200, 100), (200, 200), (100, 200)]], "class_id": 0},
        ]
        out = str(viz_dataset / "test_seg.png")
        result = render_segmentations(pixels, polys, native_size=native, output_path=out)
        assert Path(result).is_file()

    def test_renders_every_ring_of_an_occlusion_split_instance(self, viz_dataset: Path):
        """An instance's rings all get drawn, and it is labelled once, not once per contour."""
        from tcip_annotation.viz import render_segmentations

        pixels, native = _display(str(viz_dataset / "images" / "img_001.jpg"))
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

    def test_polygon_without_rings_key_is_skipped_not_crashed(self, viz_dataset: Path):
        """The renderer's rail admits an entry with no drawable ring rather than raising."""
        from tcip_annotation.viz import render_segmentations

        pixels, native = _display(str(viz_dataset / "images" / "img_001.jpg"))
        out = str(viz_dataset / "seg_empty.png")
        assert Path(render_segmentations(pixels, [{"class_id": 0}], native_size=native,
                                         output_path=out)).is_file()


class TestRenderComparison:
    def test_basic_comparison(self, viz_dataset: Path):
        from tcip_annotation.viz import render_comparison

        pixels, native = _display(str(viz_dataset / "images" / "img_001.jpg"))
        gt = [{"x1": 100, "y1": 100, "x2": 200, "y2": 200, "class_id": 0}]
        pred = [{"x1": 110, "y1": 110, "x2": 210, "y2": 210, "class_id": 0, "confidence": 0.9}]
        out = str(viz_dataset / "test_comp.png")
        result = render_comparison(pixels, gt, pred, native_size=native, output_path=out)
        assert Path(result).is_file()

    def test_a_tp_entry_draws_a_line_between_the_boxes_it_indexes(self, viz_dataset: Path):
        """``matches`` carries ``compute_matches``'s own index-shaped ``tp`` entries; the line is
        resolved from ``gt_boxes``/``pred_boxes`` by ``gt_idx``/``pred_idx``, not from a box pair
        embedded in the match entry itself."""
        from tcip_annotation.viz import render_comparison

        pixels, native = _display(str(viz_dataset / "images" / "img_001.jpg"))
        gt = [{"x1": 100, "y1": 100, "x2": 200, "y2": 200, "class_id": 0}]
        pred = [{"x1": 110, "y1": 110, "x2": 210, "y2": 210, "class_id": 0, "confidence": 0.9}]
        tp = [{"gt_idx": 0, "pred_idx": 0, "iou": 0.7, "class_name": "0", "conf": 0.9}]

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
        result = visualize("annotations", img, task="detect", class_names="bud,nut")
        assert "error" not in result
        assert "bud" in result["summary"] or "nut" in result["summary"]

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

    def test_an_unreadable_label_returns_an_error_naming_the_file(self, viz_dataset: Path):
        """Undecodable text is refused by the shared parser itself (UnreadableLabelDocument),
        not by detect_format falling through to its own generic 'cannot determine' message."""
        from tcip_mcp.tools.vision_tools import visualize

        img = str(viz_dataset / "images" / "img_001.jpg")
        label = viz_dataset / "annotations" / "img_001.json"
        label.write_text("not json {][", encoding="utf-8")

        result = visualize("annotations", img)
        assert "error" in result
        assert str(label) in result["error"]
        assert "does not decode as JSON" in result["error"]


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

    def test_an_unreadable_prediction_returns_an_error_naming_the_file(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import visualize

        img = str(viz_dataset / "images" / "img_001.jpg")
        pred = viz_dataset / "predictions" / "live" / "img_001.json"
        pred.write_text("not json {][", encoding="utf-8")

        result = visualize("predictions", img)
        assert "error" in result
        assert str(pred) in result["error"]

    def test_a_neither_key_stamp_refuses_by_name(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import visualize

        img = str(viz_dataset / "images" / "img_001.jpg")
        _seed_sidecar(viz_dataset / "predictions" / "live", {"id_map": {"bud": 0, "nut": 1}})

        result = visualize("predictions", img)
        assert "error" in result
        assert "repair-classified-predictions" in result["error"]

    def test_an_undecodable_stamp_refuses_by_name(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import visualize

        img = str(viz_dataset / "images" / "img_001.jpg")
        preds_dir = viz_dataset / "predictions" / "live"
        _seed_sidecar(preds_dir, {"id_map": {"bud": 0}})
        _damage_sidecar(preds_dir)

        result = visualize("predictions", img)
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

    def test_an_unreadable_gt_returns_an_error_naming_the_file(self, viz_dataset: Path):
        """Same distinction as the annotations source: the shared parser's own message, not
        detect_format's generic fallback."""
        from tcip_mcp.tools.vision_tools import visualize

        img = str(viz_dataset / "images" / "img_001.jpg")
        label = viz_dataset / "annotations" / "img_001.json"
        label.write_text("not json {][", encoding="utf-8")

        result = visualize("comparison", img)
        assert "error" in result
        assert str(label) in result["error"]
        assert "does not decode as JSON" in result["error"]

    def test_an_unreadable_prediction_returns_an_error_naming_the_file(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import visualize

        img = str(viz_dataset / "images" / "img_001.jpg")
        pred = viz_dataset / "predictions" / "live" / "img_001.json"
        pred.write_text("not json {][", encoding="utf-8")

        result = visualize("comparison", img)
        assert "error" in result
        assert str(pred) in result["error"]

    def test_a_neither_key_stamp_refuses_by_name(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import visualize

        img = str(viz_dataset / "images" / "img_001.jpg")
        _seed_sidecar(viz_dataset / "predictions" / "live", {"id_map": {"bud": 0, "nut": 1}})

        result = visualize("comparison", img)
        assert "error" in result
        assert "repair-classified-predictions" in result["error"]

    def test_an_undecodable_stamp_refuses_by_name(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import visualize

        img = str(viz_dataset / "images" / "img_001.jpg")
        preds_dir = viz_dataset / "predictions" / "live"
        _seed_sidecar(preds_dir, {"id_map": {"bud": 0}})
        _damage_sidecar(preds_dir)

        result = visualize("comparison", img)
        assert "error" in result


class TestDisplayRead:
    """The one decode the visualization tools render through."""

    def test_a_large_source_is_read_down_to_the_artifact_bound(self, tmp_path: Path):
        """The artifact bound is the read's target, not a resize after a whole decode; the native
        frame the annotations live in is reported unchanged."""
        from tcip_mcp.pipelines.display_bounds import VIZ_ARTIFACT_MAX_EDGE
        from tcip_mcp.tools.vision_tools import _display_for_path

        images = tmp_path / "images"
        images.mkdir()
        path = images / "big.jpg"
        Image.new("RGB", (VIZ_ARTIFACT_MAX_EDGE * 2, VIZ_ARTIFACT_MAX_EDGE)).save(path)

        read = _display_for_path(str(path))
        assert read.pixels.shape[:2] == (VIZ_ARTIFACT_MAX_EDGE // 2, VIZ_ARTIFACT_MAX_EDGE)
        assert read.native_size == (VIZ_ARTIFACT_MAX_EDGE * 2, VIZ_ARTIFACT_MAX_EDGE)
        assert read.scale == 0.5

    def test_a_source_within_the_bound_is_read_at_native_resolution(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import _display_for_path

        read = _display_for_path(str(viz_dataset / "images" / "img_001.jpg"))
        assert read.pixels.shape[:2] == (480, 640)
        assert read.scale == 1.0

    def test_a_three_band_raster_that_is_not_8_bit_is_stretched_to_be_visible(self,
                                                                             tmp_path: Path):
        """Band count alone doesn't make a raster displayable: a 16-bit capture's values occupy a
        sliver of their dtype's range, and passing them through unstretched renders it black."""
        import tifffile

        from tcip_mcp.tools.vision_tools import _display_for_path

        images = tmp_path / "images"
        images.mkdir()
        path = images / "capture.tif"
        arr = np.stack([np.linspace(100, 400, 12, dtype=np.uint16)] * 10)
        tifffile.imwrite(str(path), np.stack([arr, arr + 50, arr + 90], axis=-1))

        pixels = _display_for_path(str(path)).pixels
        assert pixels.dtype == np.uint8
        assert pixels.max() == 255

    def test_a_region_read_reports_the_rect_it_served(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import _display_for_path

        read = _display_for_path(str(viz_dataset / "images" / "img_001.jpg"),
                                 region=(100.0, 50.0, 200.0, 150.0))
        assert (read.rect.x0, read.rect.y0, read.rect.x1, read.rect.y1) == (100, 50, 300, 200)
        assert read.pixels.shape[:2] == (150, 200)

    def test_a_region_hanging_off_the_edge_is_clamped_into_the_raster(self, viz_dataset: Path):
        """A human can pan past the image, so a viewport is clamped rather than refused."""
        from tcip_mcp.tools.vision_tools import _display_for_path

        read = _display_for_path(str(viz_dataset / "images" / "img_001.jpg"),
                                 region=(500.0, 400.0, 400.0, 400.0))
        assert (read.rect.x1, read.rect.y1) == (640, 480)
        assert read.pixels.shape[:2] == (80, 140)


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

    def test_a_corrupt_label_returns_an_error_not_an_unlabeled_render(self, viz_dataset: Path):
        """A present, unreadable label document must surface as an error, not be silently
        rendered as though the image carried no labels."""
        from tcip_mcp.tools.vision_tools import visualize

        img = Image.new("RGB", (640, 480), color=(100, 120, 80))
        img.save(viz_dataset / "images" / "img_bad.jpg")
        (viz_dataset / "annotations" / "img_bad.json").write_text("not json {][", encoding="utf-8")

        # n covers every image (5, with img_bad): sampling is otherwise random, and the corrupt
        # image must be reached deterministically for this assertion.
        result = visualize("dataset", str(viz_dataset), n=5)
        assert "error" in result
        assert "img_bad.json" in result["error"]

    def test_an_unlabeled_multiband_sample_is_a_rendered_cell(self, tmp_path: Path, monkeypatch):
        """Every grid cell is a rendered artifact, labels or not: the grid tiles renders, and a
        raw source path in that list is one the tiler has to decode itself."""
        import tifffile

        from tcip_mcp.tools import vision_tools

        monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
        images = tmp_path / "images"
        images.mkdir()
        rng = np.random.default_rng(5)
        src = images / "capture.tif"
        tifffile.imwrite(str(src), rng.integers(0, 4096, size=(24, 20, 6)).astype(np.uint16))

        tiled: list[list[str]] = []
        real_grid = vision_tools.render_grid

        def spy(image_paths, **kwargs):
            tiled.append(list(image_paths))
            return real_grid(image_paths, **kwargs)

        monkeypatch.setattr(vision_tools, "render_grid", spy)
        result = vision_tools.visualize("dataset", str(tmp_path), n=1)

        assert "error" not in result
        viz_dir = (tmp_path / ".tcip" / "artifacts" / "viz").resolve()
        assert tiled and tiled[0]
        for cell in tiled[0]:
            assert Path(cell).parent == viz_dir     # a render, not the source or a temp preview
            assert Path(cell).is_file()


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

        pixels, native = _display(str(viz_dataset / "images" / "img_001.jpg"))
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
        out = render_candidates(pixels, candidates, native_size=native)
        assert Path(out).is_file()

    def test_empty_candidates(self, viz_dataset: Path):
        from tcip_annotation.viz import render_candidates

        pixels, native = _display(str(viz_dataset / "images" / "img_001.jpg"))
        out = render_candidates(pixels, [], native_size=native)
        assert Path(out).is_file()


def _uniform_cells(width: int, height: int, tile: int) -> list:
    """The clamped reference cells for a frame, the list every cells-in consumer takes."""
    from tcip_mcp.pipelines.reference_grid import reference_cells

    return reference_cells(width, height, tile, clamp=True)


class TestRenderGridOverlay:
    def test_basic(self, viz_dataset: Path):
        from tcip_annotation.viz import render_grid_overlay

        pixels, native = _display(str(viz_dataset / "images" / "img_001.jpg"))
        out = render_grid_overlay(pixels, _uniform_cells(*native, 80), native_size=native)
        assert Path(out).is_file()

    def test_cell_dicts_accepted(self, viz_dataset: Path):
        """The renderer takes the same plain dicts the coverage route serves."""
        from tcip_annotation.viz import render_grid_overlay

        pixels, native = _display(str(viz_dataset / "images" / "img_001.jpg"))
        cells = [{"name": c.name, "x0": c.x0, "y0": c.y0, "x1": c.x1, "y1": c.y1}
                 for c in _uniform_cells(*native, 160)]
        out = render_grid_overlay(pixels, cells, native_size=native)
        assert Path(out).is_file()

    def test_grid_wider_than_alphabet(self, viz_dataset: Path):
        from tcip_annotation.viz import render_grid_overlay

        pixels, _native = _display(str(viz_dataset / "images" / "img_001.jpg"))
        out = render_grid_overlay(pixels, _uniform_cells(3300, 400, 100),
                                  native_size=(3300, 400))
        assert Path(out).is_file()

    def test_empty_cells_refused(self, viz_dataset: Path):
        from tcip_annotation.viz import render_grid_overlay

        pixels, native = _display(str(viz_dataset / "images" / "img_001.jpg"))
        with pytest.raises(ValueError, match="empty"):
            render_grid_overlay(pixels, [], native_size=native)

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


class TestGridToPixel:
    def test_basic_conversion(self):
        from tcip_annotation.sam_wrapper import grid_to_pixel

        # A1 is the top-left cell's center
        x, y = grid_to_pixel("A1", _uniform_cells(640, 480, 80))
        assert x == pytest.approx(40)
        assert y == pytest.approx(40)

    def test_bottom_right(self):
        from tcip_annotation.sam_wrapper import grid_to_pixel

        x, y = grid_to_pixel("H6", _uniform_cells(640, 480, 80))
        assert x == pytest.approx(600)
        assert y == pytest.approx(440)

    def test_case_insensitive(self):
        from tcip_annotation.sam_wrapper import grid_to_pixel

        cells = _uniform_cells(640, 480, 80)
        assert grid_to_pixel(" b3 ", cells) == grid_to_pixel("B3", cells)

    def test_clamped_edge_cell_center_is_its_own(self):
        """A truncated edge cell's center is its clipped rect's, not a uniform cell's."""
        from tcip_annotation.sam_wrapper import grid_to_pixel

        x, y = grid_to_pixel("B2", _uniform_cells(100, 80, 64))
        assert x == pytest.approx((64 + 100) / 2)
        assert y == pytest.approx((64 + 80) / 2)

    def test_cell_dicts_accepted(self):
        """The lookup takes the same plain dicts the coverage route serves."""
        from tcip_annotation.sam_wrapper import grid_to_pixel

        cells = [{"name": c.name, "x0": c.x0, "y0": c.y0, "x1": c.x1, "y1": c.y1}
                 for c in _uniform_cells(640, 480, 80)]
        assert grid_to_pixel("C2", cells) == (200.0, 120.0)

    def test_unknown_column_names_the_valid_range(self):
        from tcip_annotation.sam_wrapper import grid_to_pixel

        with pytest.raises(ValueError, match="Use A1 through H6"):
            grid_to_pixel("Z1", _uniform_cells(640, 480, 80))

    def test_unknown_row_names_the_valid_range(self):
        from tcip_annotation.sam_wrapper import grid_to_pixel

        with pytest.raises(ValueError, match="Use A1 through H6"):
            grid_to_pixel("A9", _uniform_cells(640, 480, 80))

    def test_malformed_reference_is_invalid(self):
        from tcip_annotation.sam_wrapper import grid_to_pixel

        with pytest.raises(ValueError, match="Invalid cell reference"):
            grid_to_pixel("3B", _uniform_cells(640, 480, 80))


class TestColumnLabels:
    """Spreadsheet-style column labels shared by the grid geometry and cell lookup."""

    def test_round_trip_boundaries(self):
        from tcip_annotation.sam_wrapper import column_index, column_label

        expected = {0: "A", 25: "Z", 26: "AA", 27: "AB", 31: "AF", 32: "AG", 51: "AZ", 52: "BA"}
        for idx, label in expected.items():
            assert column_label(idx) == label
            assert column_index(label) == idx

    def test_multi_letter_cell_parses(self):
        from tcip_annotation.sam_wrapper import grid_to_pixel

        x, y = grid_to_pixel("AA1", _uniform_cells(2700, 480, 100))
        assert x == pytest.approx(26.5 * 100)
        assert y == pytest.approx(50)

    def test_high_columns_do_not_alias(self):
        # chr() past 'Z' labeled column 32 'a', which case folding silently parsed as column 0
        from tcip_annotation.sam_wrapper import column_label, grid_to_pixel

        cols = 40
        labels = [column_label(c) for c in range(cols)]
        assert len(set(labels)) == cols
        cells = _uniform_cells(cols * 100, 480, 100)
        for c, label in enumerate(labels):
            x, _ = grid_to_pixel(f"{label}1", cells)
            assert x == pytest.approx((c + 0.5) * 100)

    def test_out_of_range_hint_uses_column_labels(self):
        from tcip_annotation.sam_wrapper import grid_to_pixel

        with pytest.raises(ValueError, match="Use A1 through AD6"):
            grid_to_pixel("BA1", _uniform_cells(3000, 600, 100))


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
    def test_derived_default_echoes_geometry(self, viz_dataset: Path):
        """With no tile_size the tool derives the pointing grain and still echoes the full
        geometry, so the caller can hand it straight to segment_prompt."""
        from tcip_mcp.pipelines.reference_grid import derive_pointing_tile_size
        from tcip_mcp.tools.vision_tools import overlay_reference_grid

        result = overlay_reference_grid(
            image_path=str(viz_dataset / "images" / "img_001.jpg"),
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
            image_path=str(viz_dataset / "images" / "img_001.jpg"), tile_size=80,
        )
        assert "error" not in result
        assert result["tile_size"] == 80
        assert result["cols"] == 8
        assert result["rows"] == 6

    def test_missing_image(self):
        from tcip_mcp.tools.vision_tools import overlay_reference_grid

        result = overlay_reference_grid(image_path="/nonexistent.jpg")
        assert "error" in result

    def test_invalid_tile_size_is_an_error(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import overlay_reference_grid

        result = overlay_reference_grid(
            image_path=str(viz_dataset / "images" / "img_001.jpg"), tile_size=0,
        )
        assert "error" in result
        assert "tile_size" in result["error"]

    def test_wide_grid_summary_labels(self, viz_dataset: Path):
        from tcip_mcp.tools.vision_tools import overlay_reference_grid

        result = overlay_reference_grid(
            image_path=str(viz_dataset / "images" / "img_001.jpg"), tile_size=20,
        )
        assert "error" not in result
        assert result["cols"] == 32
        assert "'AF24'" in result["summary"]


class TestProposeAnnotationsTool:
    """Test propose_annotations tool (mocked engine)."""

    def test_missing_image(self):
        from tcip_mcp.tools.proposal_tools import propose_annotations

        result = propose_annotations(image_path="/nonexistent.jpg")
        assert "error" in result

    def test_unknown_engine(self, viz_dataset: Path):
        from tcip_mcp.tools.proposal_tools import propose_annotations

        result = propose_annotations(
            image_path=str(viz_dataset / "images" / "img_001.jpg"),
            engine="does_not_exist",
        )
        assert "error" in result
        assert "does_not_exist" in result["error"]


class TestAcceptProposalsTool:
    def test_no_prior_proposals(self, viz_dataset: Path):
        from tcip_mcp.tools.proposal_tools import stage_proposals

        result = stage_proposals(
            image_path=str(viz_dataset / "images" / "img_003.jpg"),
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

        class StubProposer:
            def propose(self, image_path: str, **params: object) -> list[dict]:
                return candidates

        monkeypatch.setattr(proposal, "resolve_proposer", lambda engine: StubProposer())

        propose_result = propose_annotations(
            image_path=str(viz_dataset / "images" / "img_001.jpg"), engine="sam")
        assert "error" not in propose_result, propose_result
        assert propose_result["staged"] is True

        result = stage_proposals(
            image_path=str(viz_dataset / "images" / "img_001.jpg"),
            assignments=[
                {"candidate_id": 0, "subject": "bud"},
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
        assert {a.subject for a in anns} == {"bud", "nut"}

        # Each staged object is SAM output: created_by="sam" and a numeric score.
        objs = json.loads(pred_file.read_text(encoding="utf-8"))["annotations"]
        assert objs and all(o["created_by"] == "sam" for o in objs)
        assert all(isinstance(o["score"], float) for o in objs)


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
        from tcip_mcp.tools.proposal_tools import segment_prompt

        img_path = str(real_dataset / "images" / "grid_test.jpg")
        result = segment_prompt(
            image_path=img_path,
            grid_cells=["B2"],
            tile_size=80,
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
        from tcip_mcp.tools.proposal_tools import stage_proposals, propose_annotations

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
        accept_result = stage_proposals(
            image_path=img_path,
            assignments=[
                {"candidate_id": c["id"], "subject": subj}
                for c, subj in zip(cands, ("bud", "nut"))
            ],
        )
        assert "error" not in accept_result
        assert accept_result["proposal_count"] == 2
        assert Path(accept_result["image_path"]).is_file()


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Layer 2: Integration tests â€” full pipeline with mocked SAM candidates
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


MOCK_CANDIDATES: list[dict[str, Any]] = [
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
    """End-to-end flow: mock candidates â†’ accept â†’ verify output files."""

    @pytest.fixture
    def pipeline_dataset(self, tmp_path: Path) -> Path:
        """Fresh dataset without pre-existing labels."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        img = Image.new("RGB", (640, 480), color=(80, 120, 60))
        img.save(images_dir / "sample.jpg")
        return tmp_path

    def _propose(
        self, monkeypatch: pytest.MonkeyPatch, image_path: str, candidates: list[dict],
    ) -> None:
        """Stage ``candidates`` through the real writer: a stub engine that hands them back
        verbatim, driven by an actual ``propose_annotations`` call."""
        from tcip_mcp.pipelines import proposal
        from tcip_mcp.tools.proposal_tools import propose_annotations

        class StubProposer:
            def propose(self, image_path: str, **params: object) -> list[dict]:
                return candidates

        monkeypatch.setattr(proposal, "resolve_proposer", lambda engine: StubProposer())
        result = propose_annotations(image_path=image_path, engine="sam")
        assert "error" not in result, result

    def test_accept_writes_json_detect(
        self, pipeline_dataset: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """SAM proposals are staged as predictions: pixel geometry, subject names, and score preserved."""
        from tcip_annotation import bbox_of, json_io
        from tcip_mcp.tools.proposal_tools import stage_proposals

        img_path = str(pipeline_dataset / "images" / "sample.jpg")
        self._propose(monkeypatch, img_path, MOCK_CANDIDATES)

        result = stage_proposals(
            image_path=img_path,
            assignments=[
                {"candidate_id": 0, "subject": "bud"},
                {"candidate_id": 1, "subject": "nut"},
            ],
        )
        assert "error" not in result
        assert result["proposal_count"] == 2

        pred_file = pipeline_dataset / "predictions" / "sam" / "sample.json"
        assert pred_file.is_file()
        anns = json_io.read_annotations(pred_file)
        assert len(anns) == 2
        assert {a.subject for a in anns} == {"bud", "nut"}
        # Pixel coords within the 640x480 image; staged as SAM predictions (created_by="sam").
        for a in anns:
            b = bbox_of(a.geometry)
            assert 0.0 <= b.x1 < b.x2 <= 640.0
            assert 0.0 <= b.y1 < b.y2 <= 480.0
            assert a.created_by == "sam"
        objs = json.loads(pred_file.read_text(encoding="utf-8"))["annotations"]
        assert all(isinstance(o["score"], float) for o in objs)

    def test_accept_writes_json_segment(
        self, pipeline_dataset: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """SAM proposals are staged as prediction polygons: pixel vertices, subject, score."""
        from tcip_annotation import json_io
        from tcip_mcp.tools.proposal_tools import stage_proposals

        img_path = str(pipeline_dataset / "images" / "sample.jpg")
        self._propose(monkeypatch, img_path, MOCK_CANDIDATES)

        result = stage_proposals(
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

    def test_detect_and_segment_consistent(
        self, pipeline_dataset: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Box and mask views of the staged predictions cover the same objects (one unified file)."""
        from tcip_annotation import bbox_of, json_io
        from tcip_mcp.tools.proposal_tools import stage_proposals

        img_path = str(pipeline_dataset / "images" / "sample.jpg")
        self._propose(monkeypatch, img_path, MOCK_CANDIDATES)

        result = stage_proposals(
            image_path=img_path,
            assignments=[
                {"candidate_id": 0, "subject": "bud"},
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
        assert subjects_poly == subjects_box == ["bud", "nut"]

    def test_partial_accept_skips_rejected(
        self, pipeline_dataset: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Only accepted candidates appear in output; rejected are omitted."""
        from tcip_mcp.tools.proposal_tools import stage_proposals

        img_path = str(pipeline_dataset / "images" / "sample.jpg")
        self._propose(monkeypatch, img_path, MOCK_CANDIDATES)

        # Accept only candidate 1 out of 3
        result = stage_proposals(
            image_path=img_path,
            assignments=[{"candidate_id": 1, "subject": "bud"}],
        )
        assert result["proposal_count"] == 1

    def test_invalid_candidate_id_silently_skipped(
        self, pipeline_dataset: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Assignments with non-existent candidate_id are ignored."""
        from tcip_mcp.tools.proposal_tools import stage_proposals

        img_path = str(pipeline_dataset / "images" / "sample.jpg")
        self._propose(monkeypatch, img_path, MOCK_CANDIDATES)

        result = stage_proposals(
            image_path=img_path,
            assignments=[
                {"candidate_id": 999, "subject": "bud"},  # non-existent
                {"candidate_id": 0, "subject": "nut"},        # valid
            ],
        )
        assert result["proposal_count"] == 1

    def test_render_then_accept_pipeline(
        self, pipeline_dataset: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Full render â†’ accept â†’ verify pipeline (sans SAM)."""
        from tcip_annotation.viz import render_candidates, render_grid_overlay
        from tcip_mcp.tools.proposal_tools import stage_proposals

        img_path = str(pipeline_dataset / "images" / "sample.jpg")
        pixels, native = _display(img_path)

        # Step 1: Render candidates
        candidate_render = render_candidates(pixels, MOCK_CANDIDATES, native_size=native)
        assert Path(candidate_render).is_file()

        # Step 2: Render grid overlay (for correction reference)
        grid_render = render_grid_overlay(pixels, _uniform_cells(*native, 80),
                                          native_size=native)
        assert Path(grid_render).is_file()

        # Step 3: propose_annotations stages the candidates for real
        self._propose(monkeypatch, img_path, MOCK_CANDIDATES)

        # Step 4: Accept with subject assignments
        result = stage_proposals(
            image_path=img_path,
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


class TestGridCellToSamPrompt:
    """Test grid cell -> point prompt conversion in segment_prompt."""

    def test_grid_cells_converted_to_points(self, viz_dataset: Path):
        """Grid cells should convert to points before hitting SAM."""
        from tcip_mcp.tools.proposal_tools import segment_prompt

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
                tile_size=80,
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
        from tcip_mcp.tools.proposal_tools import segment_prompt

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
                tile_size=80,
            )

        assert mock_predict.called
        # Verify the coordinates are reasonable for C4 on 640x480 at tile 80
        call_args = mock_predict.call_args
        x = call_args[0][1]  # positional: image_path, x, y
        y = call_args[0][2]
        # C = column 2 (0-indexed), cell center x = (2 + 0.5) * 80 = 200
        assert abs(x - 200.0) < 1.0
        # 4 = row 3 (0-indexed), cell center y = (3 + 0.5) * 80 = 280
        assert abs(y - 280.0) < 1.0

    def test_invalid_grid_cell_returns_error(self, viz_dataset: Path):
        """Invalid grid cell like 'Z9' should return error, not crash."""
        from tcip_mcp.tools.proposal_tools import segment_prompt

        img_path = str(viz_dataset / "images" / "img_001.jpg")
        result = segment_prompt(image_path=img_path, grid_cells=["Z9"], tile_size=80)
        assert "error" in result
        assert "Invalid grid cell" in result["error"]

    def test_grid_cells_without_tile_size_is_refused(self, viz_dataset: Path):
        """A cell name means nothing without its grid: resolving 'B3' against an assumed
        grid when the overlay rendered another picks the wrong pixel silently, so the
        call is refused."""
        from tcip_mcp.tools.proposal_tools import segment_prompt

        img_path = str(viz_dataset / "images" / "img_001.jpg")
        result = segment_prompt(image_path=img_path, grid_cells=["B3"])
        assert "error" in result
        assert "tile_size" in result["error"]

    def test_grid_cells_resolve_against_the_callers_own_grid(self, viz_dataset: Path):
        """The cells are resolved with the caller's tile_size, the grid
        overlay_reference_grid actually rendered, never a derived default."""
        from tcip_mcp.tools.proposal_tools import segment_prompt

        img_path = str(viz_dataset / "images" / "img_001.jpg")
        with patch("tcip_annotation.sam_wrapper.predict_from_point") as mock_predict:
            mock_predict.return_value = [
                [(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)],
            ]
            segment_prompt(image_path=img_path, grid_cells=["C4"], tile_size=64)

        x, y = mock_predict.call_args[0][1], mock_predict.call_args[0][2]
        # C4 at tile 64 over 640x480: x = (2 + 0.5) * 64 = 160, y = (3 + 0.5) * 64 = 224;
        # the derived-default grid's reading of the same cell lands elsewhere.
        assert abs(x - 160.0) < 1.0
        assert abs(y - 224.0) < 1.0

    def test_echoed_geometry_round_trips_from_the_overlay(self, viz_dataset: Path):
        """A legitimate call with the overlay's own echoed geometry succeeds end to end,
        and the resolved center is the rendered grid's own cell center."""
        from tcip_mcp.tools.proposal_tools import segment_prompt
        from tcip_mcp.tools.vision_tools import overlay_reference_grid

        img_path = str(viz_dataset / "images" / "img_001.jpg")
        overlay = overlay_reference_grid(image_path=img_path)
        assert "error" not in overlay
        with patch("tcip_annotation.sam_wrapper.predict_from_point") as mock_predict:
            mock_predict.return_value = [
                [(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)],
            ]
            result = segment_prompt(
                image_path=img_path, grid_cells=["B2"],
                tile_size=overlay["tile_size"], overlap=overlay["overlap"],
            )
        assert "error" not in result
        ts = overlay["tile_size"]
        assert mock_predict.call_args[0][1] == pytest.approx(1.5 * ts)
        assert mock_predict.call_args[0][2] == pytest.approx(1.5 * ts)

    def test_no_prompts_returns_error(self, viz_dataset: Path):
        """Calling with no prompts at all should error."""
        from tcip_mcp.tools.proposal_tools import segment_prompt

        img_path = str(viz_dataset / "images" / "img_001.jpg")
        result = segment_prompt(image_path=img_path)
        assert "error" in result


class TestSamPredictionStaging:
    """stage_proposals stages engine masks as predictions (predictions/sam), not ground truth."""

    @pytest.fixture
    def format_dataset(self, tmp_path: Path) -> Path:
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        img = Image.new("RGB", (640, 480), color=(100, 100, 100))
        img.save(images_dir / "fmt_test.jpg")
        return tmp_path

    def _propose(self, monkeypatch: pytest.MonkeyPatch, image_path: str) -> None:
        """Stage MOCK_CANDIDATES through the real writer: a stub engine that hands them back
        verbatim, driven by an actual ``propose_annotations`` call."""
        from tcip_mcp.pipelines import proposal
        from tcip_mcp.tools.proposal_tools import propose_annotations

        class StubProposer:
            def propose(self, image_path: str, **params: object) -> list[dict]:
                return MOCK_CANDIDATES

        monkeypatch.setattr(proposal, "resolve_proposer", lambda engine: StubProposer())
        result = propose_annotations(image_path=image_path, engine="sam")
        assert "error" not in result, result

    def test_json_detect_and_segment_written(
        self, format_dataset: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        from tcip_annotation import bbox_of, json_io
        from tcip_mcp.tools.proposal_tools import stage_proposals

        img_path = str(format_dataset / "images" / "fmt_test.jpg")
        self._propose(monkeypatch, img_path)
        result = stage_proposals(
            image_path=img_path,
            assignments=[{"candidate_id": 0, "subject": "bud"}],
        )
        assert "error" not in result
        assert "format" not in result  # fmt param dropped in the JSON cutover
        assert result["proposal_count"] == 1

        pred = format_dataset / "predictions" / "sam" / "fmt_test.json"
        assert pred.is_file()
        anns = json_io.read_annotations(pred)
        assert len(anns) == 1 and {a.subject for a in anns} == {"bud"}
        # The staged object carries a polygon (mask) with a derivable box: both views of one object.
        assert anns[0].geometry is not None
        assert bbox_of(anns[0].geometry) is not None

    def test_prediction_carries_sam_score(
        self, format_dataset: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Staged SAM output is a prediction: each object has created_by="sam" and a ``score``."""
        from tcip_mcp.tools.proposal_tools import stage_proposals

        img_path = str(format_dataset / "images" / "fmt_test.jpg")
        self._propose(monkeypatch, img_path)
        stage_proposals(
            image_path=img_path,
            assignments=[{"candidate_id": 0, "subject": "bud"}],
        )
        pred = format_dataset / "predictions" / "sam" / "fmt_test.json"
        data = json.loads(pred.read_text(encoding="utf-8"))
        assert data["annotations"]
        assert all(o["created_by"] == "sam" for o in data["annotations"])
        assert all(isinstance(o["score"], float) for o in data["annotations"])

