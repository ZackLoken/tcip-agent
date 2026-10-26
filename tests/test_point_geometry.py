"""``state.Point`` is an annotation geometry with a location and no box or area.

A Point is a placed prompt (SAM-style) or a keypoint/landmark, and ``bbox_of`` refuses one.
Covered:

  * training targets, IoU matching and delivery-grade paths skip a Point and handle the boxes and
    polygons beside it; and
  * serializers and write paths represent a Point with the ``"point": [x, y]`` key, on read and
    write.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

import tcip_store
from tcip_annotation import json_io
from tcip_annotation.matching import pair_proposals
from tcip_annotation.state import Annotation, BBox, Point, Polygon, bbox_of
from tcip_mcp.dataset_layout import (
    UNDATED_BUCKET,
    image_dir,
    image_root,
    label_key,
)
from tests._producer_fixtures import (
    blank_image,
    image_label_key,
    label_image,
    saved_annotations,
)
from tests._training_values import TRIAGE_WEIGHTS

if TYPE_CHECKING:
    from tcip_mcp.pipelines.data.selection import ClassScope
    from tcip_mcp.workspace import BoundProject

BOX = BBox(10.0, 10.0, 30.0, 30.0)
RING = [(50.0, 50.0), (70.0, 50.0), (70.0, 70.0)]


# ── the geometry itself ──────────────────────────────────────────────────────


def test_bbox_of_still_reads_a_box_and_a_polygon() -> None:
    """``bbox_of`` answers a box as itself and a polygon as its bounding box."""
    assert bbox_of(BOX) is BOX
    b = bbox_of(Polygon([RING]))
    assert (b.x1, b.y1, b.x2, b.y2) == (50.0, 50.0, 70.0, 70.0)


# ── the record's round trip (json_io) ────────────────────────────────────────


def test_point_round_trips_through_the_per_image_document(
    tmp_path: Path,
) -> None:
    key = label_key(tmp_path, UNDATED_BUCKET, "IMG_0001")
    json_io.write_label_document(
        key, [Annotation(subject="bud", geometry=Point(12.5, 34.25))], 100, 80
    )

    stored = tcip_store.read(key)
    assert isinstance(stored, dict) and isinstance(stored["annotations"], list)
    (rec,) = stored["annotations"]
    assert isinstance(rec, dict)
    assert rec["point"] == [12.5, 34.25]
    assert "bbox" not in rec  # no fabricated box travels with a point

    (back,) = json_io.read_label_document(key).annotations
    assert isinstance(back.geometry, Point)
    assert (back.geometry.x, back.geometry.y) == (12.5, 34.25)


def test_a_point_alongside_a_box_and_a_polygon_all_survive_one_document(
    tmp_path: Path,
) -> None:
    key = label_key(tmp_path, UNDATED_BUCKET, "IMG_0002")
    json_io.write_label_document(
        key,
        [
            Annotation(subject="bud", geometry=BOX),
            Annotation(subject="bud", geometry=Polygon([RING])),
            Annotation(subject="bud", geometry=Point(1.0, 2.0)),
            Annotation(subject="bud"),  # image-level label, no geometry
        ],
        100,
        80,
    )
    kinds = [
        type(a.geometry) for a in json_io.read_label_document(key).annotations
    ]
    assert kinds == [BBox, Polygon, Point, type(None)]


# ── target membership (the one shared decision) ──────────────────────────────


def _scope(root: Path, *values: str) -> ClassScope:
    """Return the registry scope of ``bud`` over a dataset at ``root``.

    The dataset declares ``opening`` over ``values`` on it, or no attribute for no values.
    """
    from tcip_mcp import subject_registry as cr
    from tcip_mcp.pipelines.data.label_queries import registry_scope
    from tests._producer_fixtures import registry_over

    attributes = (
        (cr.Attribute("opening", "categorical", values),) if values else ()
    )
    registry_over(
        root,
        cr.SubjectRegistry(
            subjects=(
                cr.Subject(name="bud", description="", attributes=attributes),
            )
        ),
    )
    return registry_scope(image_root(root), "bud")


def test_attribute_ids_still_give_a_box_its_row() -> None:
    a = Annotation(subject="bud", geometry=BOX)
    assert json_io.attribute_ids(a, "bud", ()) == []


# ── the loader's own per-image target read ───────────────────────────────────


def test_json_det_targets_yields_no_box_for_a_point(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.data.label_queries import json_det_targets

    annotations = [
        Annotation(subject="bud", geometry=Point(20.0, 20.0)),
        Annotation(subject="bud", geometry=BOX),
    ]
    key = label_key(tmp_path, UNDATED_BUCKET, "IMG_0001")
    json_io.write_label_document(key, annotations, 100, 80)
    stored = json_io.read_label_document(key).annotations
    target = json_det_targets(stored, _scope(tmp_path / "plain"))
    assert target["boxes"] == [[10.0, 10.0, 30.0, 30.0]]
    assert target["labels"] == [1]
    # An attribute scope must not turn the point into a decode failure either: it is simply not a
    # target, and the box, never assessed, is one row marked unassessed.
    target = json_det_targets(stored, _scope(tmp_path / "attributed", "open"))
    assert target["boxes"] == [[10.0, 10.0, 30.0, 30.0]]
    assert target["attributes"].tolist() == [[json_io.UNASSESSED]]


def test_a_point_only_document_carries_the_subject_and_the_detection_loader_refuses_it(
    tmp_path: Path,
) -> None:
    """A point-only document carries the subject, and the detection loader refuses it.

    Both documents read as partial for the subject. The detection loader refuses the point-only
    one, saying its subject lies only in geometries a detection loader does not read, and the
    loader admits the box-bearing document alone as its one sample.
    """
    from tests._producer_fixtures import dataset_over, write_image

    images = image_dir(tmp_path, UNDATED_BUCKET)
    for stem in ("IMG_0001", "IMG_0002"):
        write_image(images / f"{stem}.png")
    label_image(
        images / "IMG_0001.png",
        [Annotation(subject="bud", geometry=Point(20.0, 20.0))],
        100,
        80,
    )
    label_image(
        images / "IMG_0002.png",
        [Annotation(subject="bud", geometry=BOX)],
        100,
        80,
    )

    for stem in ("IMG_0001", "IMG_0002"):
        doc = json_io.read_label_document(
            image_label_key(images / f"{stem}.png")
        )
        assert doc.state("bud") == "partial"

    with pytest.raises(
        ValueError, match="only in geometries a detection loader does not read"
    ):
        dataset_over("detection", images, subject="bud")

    # Admits valid work: the document carrying a box still trains.
    ds = dataset_over("detection", images, subject="bud", members=["IMG_0002"])
    assert [Path(s).stem for s in ds.stems] == ["IMG_0002"]


# ── pairing ──────────────────────────────────────────────────────────────────

IOU = {"kind": "iou", "iou_threshold": 0.5}


def test_the_pairing_ignores_a_point_on_either_side() -> None:
    gt = [Annotation(subject="bud", geometry=Point(20.0, 20.0))]
    preds = [Annotation(subject="bud", geometry=Point(20.0, 20.0), score=0.9)]
    # A point makes no spatial claim to pair: nothing pairs, nothing crashes on bbox_of.
    assert pair_proposals(gt, preds, IOU).pairs == []


def test_the_pairing_still_pairs_the_boxes_around_a_point() -> None:
    gt = [
        Annotation(subject="bud", geometry=Point(1.0, 1.0)),
        Annotation(subject="bud", geometry=BOX),
    ]
    preds = [
        Annotation(subject="bud", geometry=Point(1.0, 1.0), score=0.9),
        Annotation(subject="bud", geometry=BOX, score=0.9),
    ]
    # The indices address the caller's own lists, so they must still point at the boxes.
    assert pair_proposals(gt, preds, IOU).pairs == [(1, 1)]


# ── COCO scoring records ─────────────────────────────────────────────────────


def test_records_from_annotation_omits_a_point_and_its_category() -> None:
    from tcip_mcp.pipelines.training.evaluation import records_from_annotation

    rec = records_from_annotation(
        [
            Annotation(subject="prompt", geometry=Point(5.0, 5.0)),
            Annotation(subject="bud", geometry=BOX),
        ],
        [Annotation(subject="bud", geometry=BOX, score=0.8)],
        width=100,
        height=80,
        cap=None,
    )
    assert len(rec["gt"]) == 1 and len(rec["dt"]) == 1
    # 'prompt' minted no category, so the box keeps id 1 rather than being pushed to 2 by a subject
    # that contributes no record at all.
    assert rec["gt"][0]["category_id"] == 1


# ── triage heuristics / phenology counts ─────────────────────────────────────


def test_worst_predictions_does_not_count_a_point_as_a_detection(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.tools.vision_tools import get_worst_predictions
    from tests._chain_fixtures import published

    image = blank_image(tmp_path, "IMG_0001.jpg")
    label_image(image, [Annotation(subject="bud", geometry=BOX)], 100, 80)
    bucket = published(
        tmp_path,
        "pred/2026-01-01",
        [
            {
                "image": str(image),
                "width": 100,
                "height": 80,
                "boxes": [[BOX.x1, BOX.y1, BOX.x2, BOX.y2]],
                "scores": [1.0],
                "labels": [1],
                "cap": 2,
            }
        ],
        scope={"subject": "bud"},
    )
    # A point beside the published box, as an edit in place would leave it: no head emits one.
    json_io.write_label_document(
        bucket.document_key(image.stem),
        [
            Annotation(subject="bud", geometry=BOX, score=1.0),
            Annotation(subject="bud", geometry=Point(60.0, 60.0), score=1.0),
        ],
        100,
        80,
    )

    res = get_worst_predictions(
        read_bucket(bucket.root, bucket.name), 8, **TRIAGE_WEIGHTS
    )
    # 1 GT box vs 1 predicted box: no shortfall, no surplus, full confidence -> a zero error score.
    # Counting the point as a surplus prediction would score this perfect frame as wrong.
    assert res["worst_images"][0]["error_score"] == 0.0


def test_phenology_detection_counts_exclude_a_point(tmp_path: Path) -> None:
    from tcip_mcp.dataset_layout import prediction_key
    from tcip_mcp.pipelines.postprocessing.phenology import count_by_class
    from tcip_mcp.traits import PositiveState

    path = prediction_key(tmp_path, "m/2026-01-01", "IMG_0001")
    json_io.write_label_document(
        path,
        [
            Annotation(
                subject="bud",
                geometry=BOX,
                score=0.9,
                attributes={"opening": "open"},
            ),
            Annotation(
                subject="bud",
                geometry=Point(60.0, 60.0),
                score=0.9,
                attributes={"opening": "open"},
            ),
        ],
        100,
        80,
    )
    total, positive, unclassified = count_by_class(
        path,
        PositiveState(attribute="opening", value="open"),
        scope=_scope(tmp_path, "open", "closed"),
    )
    assert (total, positive, unclassified) == (1, 1, 0)


# ── renderers ────────────────────────────────────────────────────────────────


def test_box_renderer_skips_a_point_and_discloses_the_skip() -> None:
    from tcip_mcp.tools.vision_tools import _boxable, _n_points, _point_note

    anns = [
        Annotation(subject="bud", geometry=Point(1.0, 1.0)),
        Annotation(subject="bud", geometry=BOX),
        Annotation(subject="bud"),
    ]
    assert _boxable(anns) == [anns[1]]
    assert _n_points(anns) == 1
    assert "not drawn" in _point_note(_n_points(anns))
    assert _point_note(0) == ""


def test_visualize_annotations_renders_the_box_and_reports_the_point(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.vision_tools import _viz_annotations

    img = blank_image(tmp_path / "ds", "IMG_0001.JPG")
    label_image(
        img,
        [
            Annotation(subject="bud", geometry=BOX),
            Annotation(subject="bud", geometry=Point(60.0, 60.0)),
        ],
        100,
        80,
    )

    res = _viz_annotations(tmp_path, str(img), task="detect")
    assert "error" not in res  # the point did not crash the box renderer
    assert res["count"] == 1
    assert res["points_not_rendered"] == 1
    assert "point annotation(s) not drawn" in res["summary"]


# ── dict serializers (the read side of every agent/GUI surface) ──────────────


def test_the_client_projection_emits_the_point_key() -> None:
    from tcip_annotation.json_io import client_annotation

    d = client_annotation(
        Annotation(subject="bud", geometry=Point(12.0, 34.0))
    )
    assert d["point"] == [12.0, 34.0]


def test_mcp_read_annotations_tool_returns_a_point(tmp_path: Path) -> None:
    from tcip_mcp.tools.annotation_tools import (
        read_annotations as read_annotations_tool,
    )

    img = blank_image(tmp_path / "ds", "IMG_0001.JPG")
    label_image(
        img, [Annotation(subject="bud", geometry=Point(12.0, 34.0))], 100, 80
    )
    res = read_annotations_tool(tmp_path, str(img))
    (ann,) = res["labels"]["annotations"]
    assert ann["point"] == [12.0, 34.0]


def test_subject_task_names_a_point_only_frame(tmp_path: Path) -> None:
    """A point-only frame is annotated (a non-None task) but is neither 'detect' nor 'segment'."""
    from tcip_mcp.tools.gui_tools import _subject_task

    assert (
        _subject_task(
            [Annotation(subject="bud", geometry=Point(1.0, 1.0))], "bud"
        )
        == "point"
    )
    assert (
        _subject_task([Annotation(subject="bud", geometry=BOX)], "bud")
        == "detect"
    )
    assert (
        _subject_task(
            [Annotation(subject="bud", geometry=Polygon([RING]))], "bud"
        )
        == "segment"
    )
    assert _subject_task([Annotation(subject="bud")], "bud") is None


# ── the agent's own write door ────────────────────────────────────────────────


def test_save_annotations_tool_writes_an_incoming_point(
    tmp_path: Path, bound: BoundProject
) -> None:
    from tcip_mcp.tools.annotation_tools import save_annotations

    img = blank_image(tmp_path)
    res = save_annotations(
        bound,
        str(img),
        annotations=[{"subject": "bud", "point": [12.0, 34.0]}],
        actor=None,
    )
    assert "error" not in res
    (stored,) = saved_annotations(img)
    assert isinstance(stored.geometry, Point)
    assert (stored.geometry.x, stored.geometry.y) == (12.0, 34.0)


def test_save_annotations_tool_keeps_points_and_point_distinct(
    tmp_path: Path, bound: BoundProject
) -> None:
    """``save_annotations`` keeps ``points`` and ``point`` distinct.

    ``points`` is a polygon contour and ``point`` is one location: the two save as a polygon and a
    point.
    """
    from tcip_mcp.tools.annotation_tools import save_annotations

    img = blank_image(tmp_path)
    save_annotations(
        bound,
        str(img),
        annotations=[
            {"subject": "bud", "points": [[50, 50], [70, 50], [70, 70]]},
            {"subject": "bud", "point": [12.0, 34.0]},
        ],
        actor=None,
    )
    kinds = [type(a.geometry) for a in saved_annotations(img)]
    assert kinds == [Polygon, Point]


# ── web routes (the human's canvas) ──────────────────────────────────────────


def test_annotate_route_round_trips_a_point(
    client: TestClient, opened_project: Path
) -> None:
    img = blank_image(opened_project)

    resp = client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(img),
            "annotations": [{"subject": "bud", "point": [12.0, 34.0]}],
            "user": "breeder",
        },
    )
    assert resp.status_code == 200
    (stored,) = saved_annotations(img)
    assert isinstance(stored.geometry, Point)

    body = client.get(
        "/api/annotate/labels", params={"image_path": str(img)}
    ).json()
    (ann,) = body["annotations"]
    assert ann["point"] == [
        12.0,
        34.0,
    ]  # read back as itself, not as a geometry-less label


def test_annotate_route_round_trips_mixed_point_and_box_geometry(
    client: TestClient, opened_project: Path
) -> None:
    img = blank_image(opened_project)
    resp = client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(img),
            "annotations": [
                {"subject": "bud", "point": [12.0, 34.0]},
                {"subject": "bud", "bbox": [10.0, 10.0, 30.0, 30.0]},
            ],
            "user": "breeder",
        },
    )
    assert resp.status_code == 200
    kinds = [type(a.geometry) for a in saved_annotations(img)]
    assert kinds == [Point, BBox]


def test_the_proposals_route_pairs_no_proposal_with_a_point(
    client: TestClient, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    from tests._chain_fixtures import published
    from tests._web_fixtures import open_new_project

    tmp_path = open_new_project(tmp_path / "proj").root
    img = blank_image(tmp_path)
    label_image(
        img, [Annotation(subject="bud", geometry=Point(20.0, 20.0))], 100, 80
    )
    bucket = published(
        tmp_path,
        "baseline/2026-01-01",
        [
            {
                "image": str(img),
                "width": 100,
                "height": 80,
                "boxes": [[BOX.x1, BOX.y1, BOX.x2, BOX.y2]],
                "scores": [0.9],
                "labels": [1],
                "cap": 2,
            }
        ],
        scope={"subject": "bud"},
    )

    resp = client.get(
        "/api/annotate/proposals",
        params={"image_path": str(img), "bucket": bucket.name},
    )
    assert resp.status_code == 200, resp.text

    # The point makes no spatial claim: the box proposal pairs with nothing.
    (proposal,) = resp.json()["proposals"]
    assert proposal["paired"] is None and proposal["bbox"] == [
        BOX.x1,
        BOX.y1,
        BOX.x2,
        BOX.y2,
    ]
