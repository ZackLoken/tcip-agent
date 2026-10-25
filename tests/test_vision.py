"""Tests for the vision rendering engine and MCP tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from PIL import Image

from tcip_annotation.matching import REVIEW_CONF_FLOOR
from tcip_mcp.dataset_layout import UNDATED_BUCKET
from tcip_mcp.pipelines.raster_source import SourceHeader
from tests._producer_fixtures import kept
from tests._training_values import IOU_THRESHOLD, TRIAGE_WEIGHTS

if TYPE_CHECKING:
    from typing import Required, TypedDict, Unpack

    from tcip_store import Key

    class _RenderGridKeywords(TypedDict, total=False):
        """The keywords ``render_grid`` takes beside its image paths."""

        titles: list[str] | None
        cols: int
        cell_size: int
        output_path: Required[str]


def _display(image_path: str) -> tuple[np.ndarray, tuple[int, int]]:
    """Return a file's RGB pixels and native size."""
    with Image.open(image_path) as im:
        rgb = im.convert("RGB")
        return np.asarray(rgb), rgb.size


@pytest.fixture
def viz_dataset(tmp_path: Path) -> Path:
    """Create a dataset of four labeled 640x480 JPEGs under ``images/<undated bucket>``.

    Each of ``img_001`` to ``img_004`` holds one ``bud`` box and one ``nut`` box; the return is
    the dataset root.
    """
    from tcip_annotation.state import Annotation, BBox
    from tests._producer_fixtures import label_image

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)

    for name in ("img_001", "img_002", "img_003", "img_004"):
        img = Image.new("RGB", (640, 480), color=(100, 120, 80))
        img.save(images_dir / f"{name}.jpg")
        # Two pixel-space boxes under two distinct subjects.
        label_image(
            images_dir / f"{name}.jpg",
            [
                Annotation(subject="bud", geometry=BBox(288, 216, 352, 264)),
                Annotation(subject="nut", geometry=BBox(176, 132, 208, 156)),
            ],
            640,
            480,
        )

    return tmp_path


PUBLISHED = "published"
"""The name :func:`viz_bucket` publishes the dataset's predictions under."""


@pytest.fixture
def viz_bucket(viz_dataset: Path) -> Path:
    """Return ``viz_dataset`` with each image's two predictions published as :data:`PUBLISHED`.

    The bucket is published through ``_chain_fixtures.published``.
    """
    pytest.importorskip("torch")
    from tests._chain_fixtures import published

    published(
        viz_dataset,
        PUBLISHED,
        [
            {
                "image": str(
                    viz_dataset / "images" / UNDATED_BUCKET / f"{name}.jpg"
                ),
                "width": 640,
                "height": 480,
                "boxes": [[288, 216, 352, 264], [496, 372, 528, 396]],
                "scores": [0.95, 0.6],
                "labels": [1, 1],
                "cap": 3,
            }
            for name in ("img_001", "img_002", "img_003", "img_004")
        ],
        scope={"subject": "bud"},
    )
    return viz_dataset


def _damage(key: Key, stored: bytes = b"{not json") -> None:
    """Replace the record ``key`` names in the store by ``stored``."""
    from tests._record_damage_fixtures import damage_record

    damage_record(key, stored)


def _damage_record(project: Path) -> None:
    """Rewrite the published bucket's record as bytes that do not decode."""
    from tcip_mcp.dataset_layout import bucket_key

    _damage(bucket_key(project, PUBLISHED))


def _label_key(image: Path) -> Key:
    from tests._producer_fixtures import image_label_key

    return image_label_key(image)


def _prediction_key(project: Path, image: Path) -> Key:
    from tcip_mcp.dataset_layout import prediction_key

    return prediction_key(project, PUBLISHED, image.stem)


# ── Rendering engine tests ──────────────────────────────────────────────────


class TestRenderDetections:
    def test_basic_render(self, viz_dataset: Path) -> None:
        from tcip_annotation.viz import render_detections

        pixels, native = _display(
            str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg")
        )
        boxes = [
            {"x1": 100, "y1": 100, "x2": 200, "y2": 200, "class_id": 0},
            {"x1": 300, "y1": 300, "x2": 400, "y2": 400, "class_id": 1},
        ]
        out = str(viz_dataset / "test_render.png")
        result = render_detections(
            pixels, boxes, native_size=native, output_path=out
        )
        assert Path(result).is_file()
        rendered = Image.open(result)
        assert rendered.size[0] > 0

    def test_with_class_names(self, viz_dataset: Path) -> None:
        from tcip_annotation.viz import render_detections

        pixels, native = _display(
            str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg")
        )
        boxes = [{"x1": 100, "y1": 100, "x2": 200, "y2": 200, "class_id": 0}]
        out = str(viz_dataset / "test_names.png")
        result = render_detections(
            pixels,
            boxes,
            native_size=native,
            class_names={0: "bud", 1: "nut"},
            output_path=out,
        )
        assert Path(result).is_file()

    def test_with_score(self, viz_dataset: Path) -> None:
        from tcip_annotation.viz import render_detections

        pixels, native = _display(
            str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg")
        )
        boxes = [
            {
                "x1": 100,
                "y1": 100,
                "x2": 200,
                "y2": 200,
                "class_id": 0,
                "score": 0.95,
            }
        ]
        out = str(viz_dataset / "test_conf.png")
        result = render_detections(
            pixels, boxes, native_size=native, output_path=out
        )
        assert Path(result).is_file()

    def test_a_shape_without_its_class_and_a_candidate_without_its_score_refuse_by_name(
        self, tmp_path: Path
    ) -> None:
        """A shape without its class and a candidate without its score refuse by name.

        A box stating no ``class_id`` and a candidate stating no ``score`` raise a KeyError
        naming the key, and the candidate with its score renders.
        """
        from tcip_annotation.viz import render_candidates, render_detections

        frame = Image.new("RGB", (40, 40))
        with pytest.raises(KeyError, match="class_id"):
            render_detections(
                frame,
                [{"x1": 1, "y1": 1, "x2": 9, "y2": 9}],
                native_size=(40, 40),
                output_path=str(tmp_path / "box.png"),
            )
        candidate = {
            "candidate_id": 0,
            "rings": [[(1, 1), (9, 1), (9, 9)]],
            "bbox": [1, 1, 8, 8],
            "area": 32,
        }
        with pytest.raises(KeyError, match="score"):
            render_candidates(
                frame,
                [candidate],
                native_size=(40, 40),
                output_path=str(tmp_path / "candidate.png"),
            )
        assert Path(
            render_candidates(
                frame,
                [{**candidate, "score": 0.5}],
                native_size=(40, 40),
                output_path=str(tmp_path / "admitted.png"),
            )
        ).is_file()

    def test_a_pil_frame_renders_the_same_as_its_own_pixels(
        self, viz_dataset: Path
    ) -> None:
        """A PIL frame renders byte-identically to its own pixels and is left untouched."""
        from tcip_annotation.viz import render_detections

        pixels, native = _display(
            str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg")
        )
        boxes = [{"x1": 100, "y1": 100, "x2": 200, "y2": 200, "class_id": 0}]
        frame = Image.fromarray(pixels, mode="RGB")
        from_array = str(viz_dataset / "from_array.png")
        from_pil = str(viz_dataset / "from_pil.png")
        render_detections(
            pixels, boxes, native_size=native, output_path=from_array
        )
        render_detections(
            frame, boxes, native_size=native, output_path=from_pil
        )

        assert Path(from_array).read_bytes() == Path(from_pil).read_bytes()
        assert np.array_equal(
            np.asarray(frame), pixels
        )  # the caller's frame is untouched

    def test_pixels_that_are_not_uint8_rgb_are_refused(
        self, viz_dataset: Path
    ) -> None:
        from tcip_annotation.viz import render_detections

        with pytest.raises(ValueError, match="uint8"):
            render_detections(
                np.zeros((8, 8, 5), dtype=np.uint16),
                [],
                native_size=(8, 8),
                output_path=str(viz_dataset / "refused.png"),
            )

    def test_a_native_coordinate_lands_at_its_scaled_position(
        self, tmp_path: Path
    ) -> None:
        """A native coordinate lands at its scaled position on reduced pixels.

        A box edge at native x 100 of a 400x200 frame draws at x 50 on 200x100 pixels, and
        nothing draws at x 100.
        """
        from tcip_annotation.viz import render_detections

        native = (400, 200)
        served = Image.new("RGB", (200, 100), (80, 80, 80))
        out = str(tmp_path / "scaled.png")
        render_detections(
            served,
            [{"x1": 100, "y1": 50, "x2": 300, "y2": 150, "class_id": 0}],
            native_size=native,
            output_path=out,
        )
        px = Image.open(out).convert("RGB")

        def red_at(xy: tuple[int, int]) -> int:
            r, g, _b = px.getpixel(xy)
            return r - g

        assert red_at((50, 50)) > 100  # the box edge at half of its native x
        assert (
            red_at((100, 50)) < 20
        )  # nothing where the unscaled coordinate would have put it


class TestRenderSegmentations:
    def test_basic_render(self, viz_dataset: Path) -> None:
        from tcip_annotation.viz import render_segmentations

        pixels, native = _display(
            str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg")
        )
        polys = [
            {
                "rings": [[(100, 100), (200, 100), (200, 200), (100, 200)]],
                "class_id": 0,
            },
        ]
        out = str(viz_dataset / "test_seg.png")
        result = render_segmentations(
            pixels, polys, native_size=native, output_path=out
        )
        assert Path(result).is_file()

    def test_renders_every_ring_of_an_occlusion_split_instance(
        self, viz_dataset: Path
    ) -> None:
        """An instance's second ring is drawn: its render differs from its first ring's alone."""
        from tcip_annotation.viz import render_segmentations

        pixels, native = _display(
            str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg")
        )
        polys = [
            {
                "rings": [
                    [(50, 50), (120, 50), (120, 150), (50, 150)],
                    [(300, 60), (380, 60), (380, 140), (300, 140)],
                ],
                "class_id": 0,
            }
        ]
        one_ring = [{"rings": [polys[0]["rings"][0]], "class_id": 0}]

        both = str(viz_dataset / "seg_two_rings.png")
        first_only = str(viz_dataset / "seg_one_ring.png")
        render_segmentations(
            pixels, polys, native_size=native, output_path=both
        )
        render_segmentations(
            pixels, one_ring, native_size=native, output_path=first_only
        )

        # The second ring really is painted: the two renders differ.
        assert Path(both).read_bytes() != Path(first_only).read_bytes()

    def test_a_polygon_stating_no_rings_refuses_and_one_with_none_drawn_admits(
        self, viz_dataset: Path
    ) -> None:
        """An entry naming no ``rings`` refuses by name; one whose ring list is empty renders."""
        from tcip_annotation.viz import render_segmentations

        pixels, native = _display(
            str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg")
        )
        out = str(viz_dataset / "seg_empty.png")
        with pytest.raises(KeyError, match="rings"):
            render_segmentations(
                pixels, [{"class_id": 0}], native_size=native, output_path=out
            )
        assert Path(
            render_segmentations(
                pixels,
                [{"class_id": 0, "rings": []}],
                native_size=native,
                output_path=out,
            )
        ).is_file()


class TestRenderComparison:
    def test_basic_comparison(self, viz_dataset: Path) -> None:
        from tcip_annotation.viz import render_comparison

        pixels, native = _display(
            str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg")
        )
        gt = [{"x1": 100, "y1": 100, "x2": 200, "y2": 200, "class_id": 0}]
        pred = [
            {
                "x1": 110,
                "y1": 110,
                "x2": 210,
                "y2": 210,
                "class_id": 0,
                "score": 0.9,
            }
        ]
        out = str(viz_dataset / "test_comp.png")
        result = render_comparison(
            pixels, gt, pred, native_size=native, output_path=out
        )
        assert Path(result).is_file()

    def test_a_tp_entry_draws_a_line_between_the_boxes_it_indexes(
        self, viz_dataset: Path
    ) -> None:
        """A true-positive entry draws a line between the boxes it indexes.

        ``matches`` carries the one matcher's ``(gt index, prediction index)`` pairs; the line
        is resolved from ``gt_boxes``/``pred_boxes`` by those indices, not from a box pair embedded
        in the match itself.
        """
        from tcip_annotation.viz import render_comparison

        pixels, native = _display(
            str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg")
        )
        gt = [{"x1": 100, "y1": 100, "x2": 200, "y2": 200, "class_id": 0}]
        pred = [
            {
                "x1": 110,
                "y1": 110,
                "x2": 210,
                "y2": 210,
                "class_id": 0,
                "score": 0.9,
            }
        ]
        tp = [(0, 0)]

        with_line = str(viz_dataset / "test_comp_with_line.png")
        render_comparison(
            pixels,
            gt,
            pred,
            native_size=native,
            matches=tp,
            output_path=with_line,
        )
        without_line = str(viz_dataset / "test_comp_without_line.png")
        render_comparison(
            pixels, gt, pred, native_size=native, output_path=without_line
        )

        yellow = (255, 255, 0)
        midpoint = (
            155,
            155,
        )  # halfway between the gt and pred centers, (150,150) and (160,160)
        with Image.open(with_line) as im:
            assert im.convert("RGB").getpixel(midpoint) == yellow
        with Image.open(without_line) as im:
            assert im.convert("RGB").getpixel(midpoint) != yellow


class TestRenderGrid:
    def test_grid(self, viz_dataset: Path) -> None:
        from tcip_annotation.viz import render_grid

        paths = [
            str(viz_dataset / "images" / UNDATED_BUCKET / f"img_{i:03d}.jpg")
            for i in range(1, 5)
        ]
        out = str(viz_dataset / "test_grid.png")
        result = render_grid(
            paths, titles=["a", "b", "c", "d"], output_path=out
        )
        assert Path(result).is_file()
        grid = Image.open(result)
        assert grid.size[0] == 4 * 256  # 4 cols * 256 cell_size

    def test_empty_grid(self, viz_dataset: Path) -> None:
        from tcip_annotation.viz import render_grid

        out = str(viz_dataset / "test_empty.png")
        result = render_grid([], output_path=out)
        assert Path(result).is_file()


# ── Vision MCP tool tests ──────────────────────────────────────────────────


def _render(project: Path, **stated: object) -> dict:
    """Return ``visualize``'s answer for the render ``stated``, admitted as its type admits one."""
    from pydantic import TypeAdapter

    from tcip_mcp.tools.vision_tools import VizRender, visualize

    return visualize(project, TypeAdapter(VizRender).validate_python(stated))


class TestVisualizeAnnotations:
    def test_detect(self, viz_dataset: Path) -> None:
        img = str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg")
        result = _render(
            viz_dataset, source="annotations", path=img, task="detect"
        )
        assert "error" not in result
        assert Path(result["image_path"]).is_file()
        assert result["count"] == 2

    def test_missing_image(self, tmp_path: Path) -> None:
        result = _render(
            tmp_path, source="annotations", path="/nonexistent/image.jpg"
        )
        assert "error" in result

    def test_no_labels(self, viz_dataset: Path) -> None:
        # Create an image with no labels
        img = Image.new("RGB", (100, 100))
        no_label = viz_dataset / "images" / UNDATED_BUCKET / "no_label.jpg"
        img.save(no_label)
        result = _render(viz_dataset, source="annotations", path=str(no_label))
        assert "error" in result

    def test_an_unreadable_label_returns_an_error_naming_the_document(
        self, viz_dataset: Path
    ) -> None:
        """An annotations render over an unreadable label returns an error naming the document."""
        img = viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"
        _damage(_label_key(img))

        result = _render(viz_dataset, source="annotations", path=str(img))
        assert "error" in result
        assert "img_001" in result["error"]


class TestVisualizePredictions:
    def test_detect(self, viz_bucket: Path) -> None:
        img = str(viz_bucket / "images" / UNDATED_BUCKET / "img_001.jpg")
        result = _render(
            viz_bucket,
            source="predictions",
            path=img,
            task="detect",
            bucket=PUBLISHED,
        )
        assert "error" not in result, result
        assert Path(result["image_path"]).is_file()
        assert result["count"] == 2

    def test_missing_predictions(self, viz_bucket: Path) -> None:
        img = Image.new("RGB", (100, 100))
        no_pred = viz_bucket / "images" / UNDATED_BUCKET / "no_pred.jpg"
        img.save(no_pred)
        result = _render(
            viz_bucket,
            source="predictions",
            path=str(no_pred),
            bucket=PUBLISHED,
        )
        assert "error" in result

    def test_a_predictions_render_names_its_bucket_by_its_type(
        self, viz_dataset: Path
    ) -> None:
        """A predictions render stating no bucket is refused by its type, naming the field."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="bucket"):
            _render(
                viz_dataset,
                source="predictions",
                path=str(
                    viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"
                ),
            )

    def test_an_unreadable_prediction_returns_an_error_naming_the_document(
        self, viz_bucket: Path
    ) -> None:
        img = viz_bucket / "images" / UNDATED_BUCKET / "img_001.jpg"
        _damage(_prediction_key(viz_bucket, img))

        result = _render(
            viz_bucket, source="predictions", path=str(img), bucket=PUBLISHED
        )
        assert "img_001" in result["error"]

    def test_an_undecodable_bucket_record_refuses(
        self, viz_bucket: Path
    ) -> None:
        _damage_record(viz_bucket)

        result = _render(
            viz_bucket,
            source="predictions",
            path=str(viz_bucket / "images" / UNDATED_BUCKET / "img_001.jpg"),
            bucket=PUBLISHED,
        )
        assert "error" in result


def test_scoring_and_the_comparison_render_state_one_count_for_one_image(
    tmp_path: Path,
) -> None:
    """The scoring and the comparison render state one count for one image.

    With no trait, the single-image scoring's TP/FP/FN, its per-detection tags and the
    comparison render's counts agree at one TP, no FP and no FN: a detection centered in a thin
    crowd strip, which it overlaps by little of its own area, is ignored by all three.
    """
    pytest.importorskip("torch")
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.tools.annotation_tools import score_predictions
    from tests._chain_fixtures import published
    from tests._producer_fixtures import label_image

    (tmp_path / "images" / UNDATED_BUCKET).mkdir(parents=True)
    image = tmp_path / "images" / UNDATED_BUCKET / "img_001.jpg"
    Image.new("RGB", (640, 480), color=(100, 120, 80)).save(image)
    label_image(
        image,
        [
            Annotation(subject="bud", geometry=BBox(288, 216, 352, 264)),
            Annotation(
                subject="bud", geometry=BBox(0, 395, 640, 405), iscrowd=True
            ),
        ],
        640,
        480,
    )
    published(
        tmp_path,
        PUBLISHED,
        [
            {
                "image": str(image),
                "width": 640,
                "height": 480,
                "boxes": [[288, 216, 352, 264], [100, 350, 200, 450]],
                "scores": [0.95, 0.9],
                "labels": [1, 1],
                "cap": 3,
            }
        ],
        scope={"subject": "bud"},
    )

    scored = score_predictions(
        tmp_path,
        str(image),
        PUBLISHED,
        IOU_THRESHOLD,
        REVIEW_CONF_FLOOR,
        detail=True,
    )
    rendered = _render(
        tmp_path,
        source="comparison",
        path=str(image),
        iou_threshold=IOU_THRESHOLD,
        bucket=PUBLISHED,
    )

    assert "error" not in scored and "error" not in rendered, (
        scored,
        rendered,
    )
    tags = [d["tag"] for d in scored["detections"]]
    counted = {
        "tp": tags.count("tp"),
        "fp": tags.count("fp"),
        "fn": tags.count("fn"),
    }
    assert (
        {k: scored[k] for k in counted}
        == counted
        == {"tp": 1, "fp": 0, "fn": 0}
    )
    assert {k: rendered[k] for k in counted} == counted


class TestVisualizeComparison:
    def test_basic(self, viz_bucket: Path) -> None:
        img = str(viz_bucket / "images" / UNDATED_BUCKET / "img_001.jpg")
        result = _render(
            viz_bucket,
            source="comparison",
            path=img,
            iou_threshold=IOU_THRESHOLD,
            bucket=PUBLISHED,
        )
        assert "error" not in result, result
        assert Path(result["image_path"]).is_file()
        assert result["gt_count"] == 2
        assert result["pred_count"] == 2

    def test_the_scoring_doors_refuse_an_unstated_iou_threshold_by_name(
        self, viz_bucket: Path
    ) -> None:
        """The scoring doors refuse an unstated IoU threshold by name.

        A comparison render's type and ``score_predictions`` refuse naming ``iou_threshold``
        when none is stated, and ``score_predictions`` scores when one is.
        """
        from pydantic import ValidationError

        from tcip_mcp.tools.annotation_tools import score_predictions

        img = str(viz_bucket / "images" / UNDATED_BUCKET / "img_001.jpg")
        with pytest.raises(ValidationError, match="iou_threshold"):
            _render(
                viz_bucket, source="comparison", path=img, bucket=PUBLISHED
            )
        with pytest.raises(TypeError, match="iou_threshold"):
            score_predictions(  # type: ignore[call-arg]
                viz_bucket, img, PUBLISHED, conf_threshold=REVIEW_CONF_FLOOR
            )

        scored = score_predictions(
            viz_bucket, img, PUBLISHED, IOU_THRESHOLD, REVIEW_CONF_FLOOR
        )
        assert "error" not in scored, scored

    def test_an_image_the_bucket_did_not_predict_draws_its_ground_truth_all_missed(
        self, viz_bucket: Path
    ) -> None:
        from tcip_annotation.state import Annotation, BBox
        from tests._producer_fixtures import label_image

        img = viz_bucket / "images" / UNDATED_BUCKET / "img_005.jpg"
        Image.new("RGB", (640, 480), color=(100, 120, 80)).save(img)
        label_image(
            img,
            [
                Annotation(subject="bud", geometry=BBox(288, 216, 352, 264)),
                Annotation(subject="leaf", geometry=BBox(176, 132, 208, 156)),
            ],
            640,
            480,
        )

        result = _render(
            viz_bucket,
            source="comparison",
            path=str(img),
            iou_threshold=IOU_THRESHOLD,
            bucket=PUBLISHED,
        )

        assert "error" not in result, result
        assert (result["gt_count"], result["pred_count"]) == (2, 0)
        assert (result["tp"], result["fp"], result["fn"]) == (0, 0, 2)

    def test_an_unreadable_gt_returns_an_error_naming_the_document(
        self, viz_bucket: Path
    ) -> None:
        """A comparison render over an unreadable GT label returns an error naming the document."""
        img = viz_bucket / "images" / UNDATED_BUCKET / "img_001.jpg"
        _damage(_label_key(img))

        result = _render(
            viz_bucket,
            source="comparison",
            path=str(img),
            iou_threshold=IOU_THRESHOLD,
            bucket=PUBLISHED,
        )
        assert "img_001" in result["error"]

    def test_an_unreadable_prediction_returns_an_error_naming_the_document(
        self, viz_bucket: Path
    ) -> None:
        img = viz_bucket / "images" / UNDATED_BUCKET / "img_001.jpg"
        _damage(_prediction_key(viz_bucket, img))

        result = _render(
            viz_bucket,
            source="comparison",
            path=str(img),
            iou_threshold=IOU_THRESHOLD,
            bucket=PUBLISHED,
        )
        assert "img_001" in result["error"]

    def test_an_undecodable_bucket_record_refuses(
        self, viz_bucket: Path
    ) -> None:
        _damage_record(viz_bucket)

        result = _render(
            viz_bucket,
            source="comparison",
            path=str(viz_bucket / "images" / UNDATED_BUCKET / "img_001.jpg"),
            iou_threshold=IOU_THRESHOLD,
            bucket=PUBLISHED,
        )
        assert "error" in result


class TestDisplayRead:
    """Tests of the display read the visualization tools render through."""

    def test_a_large_source_is_read_down_to_the_artifact_bound(
        self, tmp_path: Path
    ) -> None:
        """A large source is read down to the artifact bound.

        A source twice the bound wide is served with its long edge at the bound, and the native
        size and the read's rect are reported as the source's own.
        """
        from tcip_mcp.pipelines.display_bounds import VIZ_ARTIFACT_MAX_EDGE
        from tcip_mcp.tools.vision_tools import _display_of

        images = tmp_path / "images" / UNDATED_BUCKET
        images.mkdir(parents=True)
        path = images / "big.jpg"
        Image.new(
            "RGB", (VIZ_ARTIFACT_MAX_EDGE * 2, VIZ_ARTIFACT_MAX_EDGE)
        ).save(path)

        read = _display_of(SourceHeader(path))
        assert read.pixels.shape[:2] == (
            VIZ_ARTIFACT_MAX_EDGE // 2,
            VIZ_ARTIFACT_MAX_EDGE,
        )
        assert read.native_size == (
            VIZ_ARTIFACT_MAX_EDGE * 2,
            VIZ_ARTIFACT_MAX_EDGE,
        )
        assert (read.rect.width, read.rect.height) == read.native_size

    def test_a_source_within_the_bound_is_read_at_native_resolution(
        self, viz_dataset: Path
    ) -> None:
        from tcip_mcp.tools.vision_tools import _display_of

        read = _display_of(
            SourceHeader(
                viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"
            )
        )
        assert read.pixels.shape[:2] == (480, 640)
        assert (read.rect.width, read.rect.height) == (640, 480)

    @pytest.mark.parametrize(
        "shape,native_point,served_point",
        [
            ((200_000, 1), (0, 100_000), (0, 800)),
            ((1, 200_000), (100_000, 0), (800, 0)),
        ],
        ids=["tall", "wide"],
    )
    def test_a_skinny_raster_places_annotations_by_each_axis_own_scale(
        self,
        tmp_path: Path,
        shape: tuple[int, int],
        native_point: tuple[int, int],
        served_point: tuple[int, int],
    ) -> None:
        """A skinny raster places annotations by each axis's own scale.

        A one-pixel-wide raster read down to the artifact edge is reduced along its long axis
        only, and a point at its native middle is drawn at its served middle.
        """
        import numpy as np
        import tifffile

        from tcip_annotation.viz import CanvasPoint, render_canvas_state
        from tcip_mcp.tools.vision_tools import _display_of

        path = tmp_path / "skinny.tif"
        tifffile.imwrite(str(path), np.full(shape, 100, dtype=np.uint8))
        read = _display_of(SourceHeader(path), max_edge=1600)
        assert read.native_size == (shape[1], shape[0])
        assert read.pixels.shape[:2] == tuple(
            1600 if n > 1 else 1 for n in shape
        )

        out = render_canvas_state(
            read.pixels,
            [
                CanvasPoint.model_validate(
                    {
                        "kind": "point",
                        "points": [list(native_point)],
                        "color": "#FF0000",
                        "tag": "gt",
                    }
                )
            ],
            region=(read.rect.x0, read.rect.y0, read.rect.x1, read.rect.y1),
            output_path=str(tmp_path / "skinny.png"),
        )
        rendered = Image.open(out).convert("RGB")
        red, green, _ = rendered.getpixel(served_point)
        assert red - green > 40

    def test_a_three_band_raster_that_is_not_8_bit_is_stretched_to_be_visible(
        self, tmp_path: Path
    ) -> None:
        """A three-band raster that is not 8-bit is stretched to be visible.

        A 16-bit capture with values from 100 to 490 is served as uint8 pixels reaching 255.
        """
        import tifffile

        from tcip_mcp.tools.vision_tools import _display_of

        images = tmp_path / "images" / UNDATED_BUCKET
        images.mkdir(parents=True)
        path = images / "capture.tif"
        arr = np.stack([np.linspace(100, 400, 12, dtype=np.uint16)] * 10)
        tifffile.imwrite(
            str(path), np.stack([arr, arr + 50, arr + 90], axis=-1)
        )

        pixels = _display_of(SourceHeader(path)).pixels
        assert pixels.dtype == np.uint8
        assert pixels.max() == 255

    def test_a_region_read_reports_the_rect_it_served(
        self, viz_dataset: Path
    ) -> None:
        from tcip_mcp.tools.vision_tools import _display_of

        read = _display_of(
            SourceHeader(
                viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"
            ),
            region=(100.0, 50.0, 200.0, 150.0),
        )
        assert (read.rect.x0, read.rect.y0, read.rect.x1, read.rect.y1) == (
            100,
            50,
            300,
            200,
        )
        assert read.pixels.shape[:2] == (150, 200)

    def test_a_region_hanging_off_the_edge_is_clamped_into_the_raster(
        self, viz_dataset: Path
    ) -> None:
        """A viewport hanging off the raster's edge is clamped to the raster.

        A 400x400 region at (500, 400) on the 640x480 frame reads the 140x80 part inside it.
        """
        from tcip_mcp.tools.vision_tools import _display_of

        read = _display_of(
            SourceHeader(
                viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"
            ),
            region=(500.0, 400.0, 400.0, 400.0),
        )
        assert (read.rect.x1, read.rect.y1) == (640, 480)
        assert read.pixels.shape[:2] == (80, 140)


class TestVisualizeDatasetSample:
    def test_sample(self, viz_dataset: Path) -> None:
        result = _render(
            viz_dataset, source="dataset", path=str(viz_dataset), n=4
        )
        assert "error" not in result
        assert Path(result["image_path"]).is_file()
        assert result["count"] == 4
        assert result["total_images"] == 4

    def test_only_the_images_of_a_capture_are_sampled(
        self, viz_dataset: Path
    ) -> None:
        Image.new("RGB", (64, 48)).save(viz_dataset / "images" / "loose.jpg")
        (viz_dataset / "images" / UNDATED_BUCKET / "nested").mkdir()
        Image.new("RGB", (64, 48)).save(
            viz_dataset / "images" / UNDATED_BUCKET / "nested" / "deep.jpg"
        )

        result = _render(
            viz_dataset, source="dataset", path=str(viz_dataset), n=16
        )

        assert "error" not in result, result
        assert result["total_images"] == 4

    def test_no_images(self, tmp_path: Path) -> None:
        (tmp_path / "images" / UNDATED_BUCKET).mkdir(parents=True)
        result = _render(tmp_path, source="dataset", path=str(tmp_path), n=4)
        assert "error" in result

    def test_a_corrupt_label_returns_an_error_not_an_unlabeled_render(
        self, viz_dataset: Path
    ) -> None:
        """A dataset render over a corrupt label returns an error naming the image."""
        from tcip_annotation.state import Annotation, BBox
        from tests._producer_fixtures import label_image

        bad = viz_dataset / "images" / UNDATED_BUCKET / "img_bad.jpg"
        Image.new("RGB", (640, 480), color=(100, 120, 80)).save(bad)
        label_image(
            bad,
            [Annotation(subject="bud", geometry=BBox(1, 1, 5, 5))],
            640,
            480,
        )
        _damage(_label_key(bad))

        # n covers every image (5, with img_bad): sampling is otherwise random, and the corrupt
        # image must be reached deterministically for this assertion.
        result = _render(
            viz_dataset, source="dataset", path=str(viz_dataset), n=5
        )
        assert "error" in result
        assert "img_bad" in result["error"]

    def test_an_unlabeled_multiband_sample_is_a_rendered_cell(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unlabeled multiband sample is a rendered grid cell.

        Each cell the grid tiles is a rendered file in the project's viz directory, labels or
        not.
        """
        import tifffile

        from tcip_mcp.tools import vision_tools

        images = tmp_path / "images" / UNDATED_BUCKET
        images.mkdir(parents=True)
        rng = np.random.default_rng(5)
        src = images / "capture.tif"
        tifffile.imwrite(
            str(src), rng.integers(0, 4096, size=(24, 20, 6)).astype(np.uint16)
        )

        tiled: list[list[str]] = []
        real_grid = vision_tools.render_grid

        def spy(
            image_paths: list[str], **kwargs: Unpack[_RenderGridKeywords]
        ) -> str:
            tiled.append(list(image_paths))
            return real_grid(image_paths, **kwargs)

        monkeypatch.setattr(vision_tools, "render_grid", spy)
        result = _render(tmp_path, source="dataset", path=str(tmp_path), n=1)

        assert "error" not in result
        viz_dir = (tmp_path / ".tcip" / "artifacts" / "viz").resolve()
        assert tiled and tiled[0]
        for cell in tiled[0]:
            assert (
                Path(cell).parent == viz_dir
            )  # a render, not the source or a temp preview
            assert Path(cell).is_file()


class TestVisualizeWorstPredictions:
    def test_basic(self, viz_bucket: Path) -> None:
        from tcip_mcp.tools.vision_tools import render_failure_cases

        result = render_failure_cases(
            viz_bucket, str(viz_bucket), PUBLISHED, top_k=3, **TRIAGE_WEIGHTS
        )
        assert "error" not in result
        # Should have rendered some cases
        assert len(result.get("case_images", [])) > 0

    def test_the_ranking_takes_its_size_and_weights_with_no_default(
        self,
    ) -> None:
        import inspect

        from tcip_mcp.tools.vision_tools import (
            get_worst_predictions,
            render_failure_cases,
        )

        for door in (get_worst_predictions, render_failure_cases):
            parameters = inspect.signature(door).parameters
            for name in ("top_k", *TRIAGE_WEIGHTS):
                assert parameters[name].default is inspect.Parameter.empty, (
                    door,
                    name,
                )


# ── Candidate and grid renders ──────────────────────────────────────────────


class TestRenderCandidates:
    def test_basic_render(self, viz_dataset: Path) -> None:
        from tcip_annotation.viz import render_candidates

        pixels, native = _display(
            str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg")
        )
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
        out = render_candidates(
            pixels,
            candidates,
            native_size=native,
            output_path=str(viz_dataset / "candidates.png"),
        )
        assert Path(out).is_file()

    def test_empty_candidates(self, viz_dataset: Path) -> None:
        from tcip_annotation.viz import render_candidates

        pixels, native = _display(
            str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg")
        )
        out = render_candidates(
            pixels,
            [],
            native_size=native,
            output_path=str(viz_dataset / "no_candidates.png"),
        )
        assert Path(out).is_file()


def _uniform_cells(width: int, height: int, tile: int) -> list:
    """Return the clamped reference cells for a ``width`` by ``height`` frame at ``tile``."""
    from tcip_mcp.pipelines.reference_grid import reference_cells

    return reference_cells(width, height, tile, 0.0, clamp=True)


class TestRenderGridOverlay:
    def test_basic(self, viz_dataset: Path) -> None:
        from tcip_annotation.viz import render_grid_overlay

        pixels, native = _display(
            str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg")
        )
        out = render_grid_overlay(
            pixels,
            _uniform_cells(*native, 80),
            native_size=native,
            output_path=str(viz_dataset / "grid.png"),
        )
        assert Path(out).is_file()

    def test_grid_wider_than_alphabet(self, viz_dataset: Path) -> None:
        from tcip_annotation.viz import render_grid_overlay

        pixels, _native = _display(
            str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg")
        )
        out = render_grid_overlay(
            pixels,
            _uniform_cells(3300, 400, 100),
            native_size=(3300, 400),
            output_path=str(viz_dataset / "grid_wide.png"),
        )
        assert Path(out).is_file()

    def test_empty_cells_refused(self, viz_dataset: Path) -> None:
        from tcip_annotation.viz import render_grid_overlay

        pixels, native = _display(
            str(viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg")
        )
        with pytest.raises(ValueError, match="empty"):
            render_grid_overlay(
                pixels,
                [],
                native_size=native,
                output_path=str(viz_dataset / "grid_empty.png"),
            )

    def test_lines_land_on_supplied_cell_boundaries(
        self, tmp_path: Path
    ) -> None:
        """The grid overlay draws its lines on the supplied cell boundaries.

        The renderer draws the supplied cell rects: on a clamped 100x80 grid at tile 64 the
        interior boundaries sit at 64, no line sits at x=50 or y=40, and the outer boundaries
        render on the frame's edge pixels.
        """
        from tcip_annotation.viz import render_grid_overlay

        pixels = np.zeros((80, 100, 3), dtype=np.uint8)
        out = render_grid_overlay(
            pixels,
            _uniform_cells(100, 80, 64),
            native_size=(100, 80),
            output_path=str(tmp_path / "grid.png"),
        )
        arr = np.asarray(Image.open(out))
        yellow = (
            (arr[:, :, 0] > 200) & (arr[:, :, 1] > 200) & (arr[:, :, 2] < 80)
        )
        assert yellow[:, 64].sum() >= 60, (
            "no vertical line on the supplied boundary x=64"
        )
        assert yellow[64, :].sum() >= 75, (
            "no horizontal line on the supplied boundary y=64"
        )
        # Away from the true boundaries and the A1 label, the uniform-division positions
        # (x=50, y=40) hold no line.
        assert yellow[25:60, 50].sum() == 0, (
            "a line at x=50 is the uniform division"
        )
        assert yellow[40, 25:60].sum() == 0, (
            "a line at y=40 is the uniform division"
        )
        # The outer boundaries render inside the frame: a far edge scaling to the frame
        # size must pin to the last pixel row/column, not clip away.
        assert yellow[:, 0].sum() >= 60, "no left outer boundary at x=0"
        assert yellow[0, :].sum() >= 75, "no top outer boundary at y=0"
        assert yellow[:, 99].sum() >= 60, "no right outer boundary at x=99"
        assert yellow[79, :].sum() >= 75, "no bottom outer boundary at y=79"


class TestGridToRect:
    def test_a_name_resolves_to_its_cells_rect(self) -> None:
        from tcip_annotation.grid import grid_to_rect

        assert grid_to_rect("A1", _uniform_cells(640, 480, 80)) == (
            0.0,
            0.0,
            80.0,
            80.0,
        )
        assert grid_to_rect("H6", _uniform_cells(640, 480, 80)) == (
            560.0,
            400.0,
            640.0,
            480.0,
        )

    def test_case_insensitive(self) -> None:
        from tcip_annotation.grid import grid_to_rect

        cells = _uniform_cells(640, 480, 80)
        assert grid_to_rect(" b3 ", cells) == grid_to_rect("B3", cells)

    def test_a_clamped_edge_cell_is_its_own_clipped_rect(self) -> None:
        from tcip_annotation.grid import grid_to_rect

        assert grid_to_rect("B2", _uniform_cells(100, 80, 64)) == (
            64.0,
            64.0,
            100.0,
            80.0,
        )

    def test_unknown_column_names_the_valid_range(self) -> None:
        from tcip_annotation.grid import grid_to_rect

        with pytest.raises(ValueError, match="Use A1 through H6"):
            grid_to_rect("Z1", _uniform_cells(640, 480, 80))

    def test_unknown_row_names_the_valid_range(self) -> None:
        from tcip_annotation.grid import grid_to_rect

        with pytest.raises(ValueError, match="Use A1 through H6"):
            grid_to_rect("A9", _uniform_cells(640, 480, 80))

    def test_malformed_reference_is_invalid(self) -> None:
        from tcip_annotation.grid import grid_to_rect

        with pytest.raises(ValueError, match="Invalid cell reference"):
            grid_to_rect("3B", _uniform_cells(640, 480, 80))


class TestColumnLabels:
    """Spreadsheet-style column labels shared by the grid geometry and cell lookup."""

    def test_round_trip_boundaries(self) -> None:
        from tcip_annotation.grid import column_index, column_label

        expected = {
            0: "A",
            25: "Z",
            26: "AA",
            27: "AB",
            31: "AF",
            32: "AG",
            51: "AZ",
            52: "BA",
        }
        for idx, label in expected.items():
            assert column_label(idx) == label
            assert column_index(label) == idx

    def test_multi_letter_cell_parses(self) -> None:
        from tcip_annotation.grid import grid_to_rect

        assert grid_to_rect("AA1", _uniform_cells(2700, 480, 100))[:2] == (
            2600.0,
            0.0,
        )

    def test_high_columns_do_not_alias(self) -> None:
        # chr() past 'Z' labeled column 32 'a', which case folding silently parsed as column 0
        from tcip_annotation.grid import column_label, grid_to_rect

        cols = 40
        labels = [column_label(c) for c in range(cols)]
        assert len(set(labels)) == cols
        cells = _uniform_cells(cols * 100, 480, 100)
        for c, label in enumerate(labels):
            assert grid_to_rect(f"{label}1", cells)[0] == c * 100

    def test_out_of_range_hint_uses_column_labels(self) -> None:
        from tcip_annotation.grid import grid_to_rect

        with pytest.raises(ValueError, match="Use A1 through AD6"):
            grid_to_rect("BA1", _uniform_cells(3000, 600, 100))


class TestVisualizeGridOverlayTool:
    def test_derived_default_echoes_geometry(self, viz_dataset: Path) -> None:
        """With no tile_size the grid overlay derives the pointing grain and echoes the geometry.

        The answer carries the width, height, derived tile size, overlap, columns and rows.
        """
        from tcip_mcp.pipelines.reference_grid import derive_pointing_tile_size
        from tcip_mcp.tools.vision_tools import overlay_reference_grid

        result = overlay_reference_grid(
            viz_dataset,
            image_path=str(
                viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"
            ),
            overlap=0.0,
        )
        assert "error" not in result
        assert Path(result["image_path"]).is_file()
        assert result["width"] == 640
        assert result["height"] == 480
        assert result["tile_size"] == derive_pointing_tile_size(640, 480)
        assert result["overlap"] == 0.0
        assert result["cols"] >= 1 and result["rows"] >= 1

    def test_explicit_tile_size_echoes_back(self, viz_dataset: Path) -> None:
        from tcip_mcp.tools.vision_tools import overlay_reference_grid

        result = overlay_reference_grid(
            viz_dataset,
            image_path=str(
                viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"
            ),
            tile_size=80,
            overlap=0.0,
        )
        assert "error" not in result
        assert result["tile_size"] == 80
        assert result["cols"] == 8
        assert result["rows"] == 6

    def test_missing_image(self, tmp_path: Path) -> None:
        from tcip_mcp.tools.vision_tools import overlay_reference_grid

        result = overlay_reference_grid(
            tmp_path, image_path="/nonexistent.jpg", overlap=0.0
        )
        assert "error" in result

    def test_invalid_tile_size_is_an_error(self, viz_dataset: Path) -> None:
        from tcip_mcp.tools.vision_tools import overlay_reference_grid

        result = overlay_reference_grid(
            viz_dataset,
            image_path=str(
                viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"
            ),
            tile_size=0,
            overlap=0.0,
        )
        assert "error" in result
        assert "tile_size" in result["error"]

    def test_wide_grid_summary_labels(self, viz_dataset: Path) -> None:
        from tcip_mcp.tools.vision_tools import overlay_reference_grid

        result = overlay_reference_grid(
            viz_dataset,
            image_path=str(
                viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"
            ),
            tile_size=20,
            overlap=0.0,
        )
        assert "error" not in result
        assert result["cols"] == 32
        assert "'AF24'" in result["summary"]


class TestProposeAnnotationsTool:
    """Tests of the propose_annotations tool over a mocked engine."""

    def test_missing_image(self, tmp_path: Path) -> None:
        from tcip_mcp.tools.proposal_tools import propose_annotations

        result = propose_annotations(
            tmp_path, image_path="/nonexistent.jpg", engine="stub", actor=None
        )
        assert "error" in result

    def test_unknown_engine(self, viz_dataset: Path) -> None:
        from tcip_mcp.tools.proposal_tools import propose_annotations

        result = propose_annotations(
            viz_dataset,
            image_path=str(
                viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"
            ),
            engine="does_not_exist",
            actor=None,
        )
        assert "error" in result
        assert "does_not_exist" in result["error"]


class TestAcceptProposalsTool:
    def test_no_prior_proposals(self, viz_dataset: Path) -> None:
        from tcip_mcp.tools.proposal_tools import stage_proposals

        result = stage_proposals(
            viz_dataset,
            image_path=str(
                viz_dataset / "images" / UNDATED_BUCKET / "img_003.jpg"
            ),
            actor=None,
            assignments=kept({"staging_id": "staging_0_000000"}, (0, "bud")),
        )
        assert "error" in result
        assert "staging_0_000000" in result["error"]
        assert "Run propose_annotations" in result["error"]

    def test_with_cached_proposals(
        self, viz_dataset: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tcip_mcp.pipelines import proposal
        from tcip_mcp.tools.proposal_tools import (
            propose_annotations,
            stage_proposals,
        )

        # The candidates propose_annotations stages, in the neutral engine schema.
        candidates = [
            {
                "candidate_id": 0,
                "bbox": [100.0, 100.0, 200.0, 200.0],
                "area": 10000,
                "score": 0.90,
                "engine_meta": {
                    "stability_score": 0.95,
                    "predicted_iou": 0.90,
                },
                "rings": [[[100, 100], [200, 100], [200, 200], [100, 200]]],
            },
            {
                "candidate_id": 1,
                "bbox": [300.0, 300.0, 400.0, 400.0],
                "area": 5000,
                "score": 0.85,
                "engine_meta": {
                    "stability_score": 0.88,
                    "predicted_iou": 0.85,
                },
                "rings": [[[300, 300], [400, 300], [400, 400], [300, 400]]],
            },
        ]

        class StubProposer:
            def propose(
                self, image_path: str, **params: object
            ) -> proposal.Proposal:
                return proposal.Proposal(candidates, None)

        monkeypatch.setattr(
            proposal, "resolve_proposer", lambda engine: StubProposer()
        )

        propose_result = propose_annotations(
            viz_dataset,
            image_path=str(
                viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"
            ),
            engine="stub",
            actor=None,
        )
        assert "error" not in propose_result, propose_result
        assert propose_result["staged"] is True

        result = stage_proposals(
            viz_dataset,
            image_path=str(
                viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg"
            ),
            actor=None,
            assignments=kept(propose_result, (0, "bud"), (1, "nut")),
        )
        assert "error" not in result
        assert result["proposal_count"] == 2
        assert Path(result["image_path"]).is_file()

        # Masks are staged as the engine's predictions, not GT: one per-image document holding
        # both accepted objects by subject name.
        anns, objs = _staged(
            viz_dataset,
            viz_dataset / "images" / UNDATED_BUCKET / "img_001.jpg",
            result,
        )
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
        "engine_meta": {"stability_score": 0.96, "predicted_iou": 0.93},
        "rings": [
            [
                (50, 40),
                (200, 40),
                (200, 180),
                (50, 180),
            ]
        ],
    },
    {
        "candidate_id": 1,
        "bbox": [300.0, 250.0, 450.0, 400.0],
        "area": 15000,
        "score": 0.88,
        "engine_meta": {"stability_score": 0.91, "predicted_iou": 0.88},
        "rings": [
            [
                (300, 250),
                (450, 250),
                (450, 400),
                (300, 400),
            ]
        ],
    },
    {
        "candidate_id": 2,
        "bbox": [500.0, 100.0, 600.0, 200.0],
        "area": 8000,
        "score": 0.80,
        "engine_meta": {"stability_score": 0.85, "predicted_iou": 0.80},
        "rings": [
            [
                (500, 100),
                (600, 100),
                (600, 200),
                (500, 200),
            ]
        ],
    },
]


def _staged(
    project: Path, image: Path, staged: dict
) -> tuple[list, list[dict]]:
    """Return the document ``staged``'s bucket holds for ``image``, decoded and as stored.

    The pair is its annotations as the reader decodes them and its records as the store holds
    them.
    """
    from pydantic import JsonValue, TypeAdapter

    import tcip_store
    from tcip_annotation.json_io import read_label_document
    from tcip_mcp.buckets import read_bucket

    key = read_bucket(project, staged["bucket"]).document_key(image.stem)
    assert key is not None, staged
    document = tcip_store.read(key)
    assert isinstance(document, dict)
    return read_label_document(key).annotations, TypeAdapter(
        list[dict[str, JsonValue]]
    ).validate_python(document["annotations"])


class TestCandidateCacheRoundTrip:
    """Tests that candidates survive a JSON serialize and deserialize round trip."""

    def test_polygon_geometry_preserved(self, tmp_path: Path) -> None:
        state_file = tmp_path / "candidates_test.json"
        state_file.write_text(
            json.dumps(MOCK_CANDIDATES, default=str), encoding="utf-8"
        )

        loaded = json.loads(state_file.read_text(encoding="utf-8"))
        assert len(loaded) == 3

        for orig, restored in zip(MOCK_CANDIDATES, loaded):
            assert orig["candidate_id"] == restored["candidate_id"]
            assert orig["bbox"] == restored["bbox"]
            assert orig["area"] == restored["area"]
            assert abs(orig["score"] - restored["score"]) < 1e-6
            assert (
                abs(
                    orig["engine_meta"]["stability_score"]
                    - restored["engine_meta"]["stability_score"]
                )
                < 1e-6
            )
            # Every ring, every vertex: JSON turns tuples into lists
            assert len(orig["rings"]) == len(restored["rings"])
            for o_ring, r_ring in zip(
                orig["rings"], restored["rings"], strict=True
            ):
                for (ox, oy), rp in zip(o_ring, r_ring, strict=True):
                    rx, ry = (
                        rp
                        if isinstance(rp, (list, tuple))
                        else (rp["x"], rp["y"])
                    )
                    assert abs(ox - rx) < 1e-6
                    assert abs(oy - ry) < 1e-6

    def test_empty_candidates_roundtrip(self, tmp_path: Path) -> None:
        state_file = tmp_path / "candidates_empty.json"
        state_file.write_text(json.dumps([]), encoding="utf-8")
        loaded = json.loads(state_file.read_text(encoding="utf-8"))
        assert loaded == []


class TestFullPipelineIntegration:
    """Tests of staging mocked engine candidates as predictions and checking what was staged."""

    @pytest.fixture
    def pipeline_dataset(self, tmp_path: Path) -> Path:
        """Create a fresh dataset of one image with no labels; return its root."""
        images_dir = tmp_path / "images" / UNDATED_BUCKET
        images_dir.mkdir(parents=True)
        img = Image.new("RGB", (640, 480), color=(80, 120, 60))
        img.save(images_dir / "sample.jpg")
        return tmp_path

    def _propose(
        self,
        monkeypatch: pytest.MonkeyPatch,
        project: Path,
        image_path: str,
        candidates: list[dict],
    ) -> dict:
        """Stage ``candidates`` for ``project`` through the real writer and return its answer.

        A stub engine hands them back verbatim, driven by an actual ``propose_annotations`` call.
        """
        from tcip_mcp.pipelines import proposal
        from tcip_mcp.tools.proposal_tools import propose_annotations

        class StubProposer:
            def propose(
                self, image_path: str, **params: object
            ) -> proposal.Proposal:
                return proposal.Proposal(candidates, None)

        monkeypatch.setattr(
            proposal, "resolve_proposer", lambda engine: StubProposer()
        )
        result = propose_annotations(
            project, image_path=image_path, engine="stub", actor=None
        )
        assert "error" not in result, result
        return result

    def test_accept_writes_json_detect(
        self,
        pipeline_dataset: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Engine proposals are staged as predictions.

        The two staged objects carry the assigned subjects, boxes inside the 640x480 frame,
        ``created_by`` ``stub`` and a float score.
        """
        from tcip_annotation import bbox_of
        from tcip_mcp.tools.proposal_tools import stage_proposals

        img_path = str(
            pipeline_dataset / "images" / UNDATED_BUCKET / "sample.jpg"
        )
        proposed = self._propose(
            monkeypatch, pipeline_dataset, img_path, MOCK_CANDIDATES
        )

        result = stage_proposals(
            pipeline_dataset,
            image_path=img_path,
            actor=None,
            assignments=kept(proposed, (0, "bud"), (1, "nut")),
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
        self,
        pipeline_dataset: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Engine proposals are staged as prediction polygons: pixel vertices, subject, score."""
        from tcip_mcp.tools.proposal_tools import stage_proposals

        img_path = str(
            pipeline_dataset / "images" / UNDATED_BUCKET / "sample.jpg"
        )
        proposed = self._propose(
            monkeypatch, pipeline_dataset, img_path, MOCK_CANDIDATES
        )

        result = stage_proposals(
            pipeline_dataset,
            image_path=img_path,
            actor=None,
            assignments=kept(proposed, (0, "bud")),
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
        self,
        pipeline_dataset: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Box and mask views of the staged predictions cover the same objects (one document)."""
        from tcip_annotation import bbox_of
        from tcip_mcp.tools.proposal_tools import stage_proposals

        img_path = str(
            pipeline_dataset / "images" / UNDATED_BUCKET / "sample.jpg"
        )
        proposed = self._propose(
            monkeypatch, pipeline_dataset, img_path, MOCK_CANDIDATES
        )

        result = stage_proposals(
            pipeline_dataset,
            image_path=img_path,
            actor=None,
            assignments=kept(proposed, (0, "bud"), (2, "nut")),
        )
        assert result["proposal_count"] == 2

        anns, _objs = _staged(pipeline_dataset, Path(img_path), result)
        # Each object is one polygon with a derivable box under the same subject: the box and mask
        # views can never diverge because they are the same annotations.
        subjects_poly = sorted(
            a.subject for a in anns if a.geometry is not None
        )
        subjects_box = sorted(
            a.subject for a in anns if bbox_of(a.geometry) is not None
        )
        assert subjects_poly == subjects_box == ["bud", "nut"]

    def test_partial_accept_skips_rejected(
        self,
        pipeline_dataset: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Staging one assignment out of three candidates stages one proposal."""
        from tcip_mcp.tools.proposal_tools import stage_proposals

        img_path = str(
            pipeline_dataset / "images" / UNDATED_BUCKET / "sample.jpg"
        )
        proposed = self._propose(
            monkeypatch, pipeline_dataset, img_path, MOCK_CANDIDATES
        )

        # Accept only candidate 1 out of 3
        result = stage_proposals(
            pipeline_dataset,
            image_path=img_path,
            actor=None,
            assignments=kept(proposed, (1, "bud")),
        )
        assert result["proposal_count"] == 1

    def test_invalid_candidate_id_silently_skipped(
        self,
        pipeline_dataset: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Assignments with non-existent candidate_id are ignored."""
        from tcip_mcp.tools.proposal_tools import stage_proposals

        img_path = str(
            pipeline_dataset / "images" / UNDATED_BUCKET / "sample.jpg"
        )
        proposed = self._propose(
            monkeypatch, pipeline_dataset, img_path, MOCK_CANDIDATES
        )

        # Candidate 999 does not exist; candidate 0 does.
        result = stage_proposals(
            pipeline_dataset,
            image_path=img_path,
            actor=None,
            assignments=kept(proposed, (999, "bud"), (0, "nut")),
        )
        assert result["proposal_count"] == 1

    def test_render_then_accept_pipeline(
        self,
        pipeline_dataset: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Candidate and grid renders are written, and three assignments stage with a QA render."""
        from tcip_annotation.viz import render_candidates, render_grid_overlay
        from tcip_mcp.tools.proposal_tools import stage_proposals

        img_path = str(
            pipeline_dataset / "images" / UNDATED_BUCKET / "sample.jpg"
        )
        pixels, native = _display(img_path)

        # Step 1: Render candidates
        candidate_render = render_candidates(
            pixels,
            MOCK_CANDIDATES,
            native_size=native,
            output_path=str(pipeline_dataset / "candidates.png"),
        )
        assert Path(candidate_render).is_file()

        # Step 2: Render grid overlay (for correction reference)
        grid_render = render_grid_overlay(
            pixels,
            _uniform_cells(*native, 80),
            native_size=native,
            output_path=str(pipeline_dataset / "grid.png"),
        )
        assert Path(grid_render).is_file()

        # Step 3: propose_annotations stages the candidates for real
        proposed = self._propose(
            monkeypatch, pipeline_dataset, img_path, MOCK_CANDIDATES
        )

        # Step 4: Accept with subject assignments
        result = stage_proposals(
            pipeline_dataset,
            image_path=img_path,
            actor=None,
            assignments=kept(proposed, (0, "bud"), (1, "nut"), (2, "bud")),
        )
        assert "error" not in result
        assert result["proposal_count"] == 3

        # Step 5: QA render was produced
        assert Path(result["image_path"]).is_file()


class TestEnginePredictionStaging:
    """stage_proposals stages engine masks as predictions in a bucket of the engine's."""

    @pytest.fixture
    def format_dataset(self, tmp_path: Path) -> Path:
        images_dir = tmp_path / "images" / UNDATED_BUCKET
        images_dir.mkdir(parents=True)
        img = Image.new("RGB", (640, 480), color=(100, 100, 100))
        img.save(images_dir / "fmt_test.jpg")
        return tmp_path

    def _propose(
        self, monkeypatch: pytest.MonkeyPatch, project: Path, image_path: str
    ) -> dict:
        """Stage MOCK_CANDIDATES for ``project`` through the real writer and return its answer.

        A stub engine hands them back verbatim, driven by an actual ``propose_annotations`` call.
        """
        from tcip_mcp.pipelines import proposal
        from tcip_mcp.tools.proposal_tools import propose_annotations

        class StubProposer:
            def propose(
                self, image_path: str, **params: object
            ) -> proposal.Proposal:
                return proposal.Proposal(MOCK_CANDIDATES, None)

        monkeypatch.setattr(
            proposal, "resolve_proposer", lambda engine: StubProposer()
        )
        result = propose_annotations(
            project, image_path=image_path, engine="stub", actor=None
        )
        assert "error" not in result, result
        return result

    def test_json_detect_and_segment_written(
        self,
        format_dataset: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from tcip_annotation import bbox_of
        from tcip_mcp.tools.proposal_tools import stage_proposals

        img_path = str(
            format_dataset / "images" / UNDATED_BUCKET / "fmt_test.jpg"
        )
        proposed = self._propose(monkeypatch, format_dataset, img_path)
        result = stage_proposals(
            format_dataset,
            image_path=img_path,
            actor=None,
            assignments=kept(proposed, (0, "bud")),
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
        self,
        format_dataset: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Staged engine output carries ``created_by="stub"`` and a float ``score``."""
        from tcip_mcp.tools.proposal_tools import stage_proposals

        img_path = str(
            format_dataset / "images" / UNDATED_BUCKET / "fmt_test.jpg"
        )
        proposed = self._propose(monkeypatch, format_dataset, img_path)
        staged = stage_proposals(
            format_dataset,
            image_path=img_path,
            actor=None,
            assignments=kept(proposed, (0, "bud")),
        )
        _anns, objs = _staged(format_dataset, Path(img_path), staged)
        assert objs
        assert all(o["created_by"] == "stub" for o in objs)
        assert all(isinstance(o["score"], float) for o in objs)
