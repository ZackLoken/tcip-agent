"""An inverted or zero-extent box is refused wherever a writer builds one: the save doors, the
per-image writer and the prediction encoder."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import tcip_store as ts
from tcip_annotation.json_io import label_document, read_label_document, write_label_document
from tcip_annotation.state import Annotation, BBox, Polygon
from tcip_mcp.dataset_layout import UNDATED_BUCKET, label_key
from tcip_mcp.pipelines.data.label_queries import registry_scope
from tests._producer_fixtures import image_label_key, write_image


# ── the shared constructor ──────────────────────────────────────────────────


def test_the_encoding_keeps_only_what_has_extent_on_the_stored_grid(tmp_path):
    """A box that rounds to no width, and a mask whose rings have none, are no detection: the
    encoding keeps the one detection with extent, and the count it reports is the document's."""
    import numpy as np

    from tcip_annotation.mask_contours import mask_to_polygon_rings
    from tcip_mcp.pipelines.postprocessing.export import encode_predictions

    solid = np.zeros((40, 40), dtype=np.uint8)
    solid[5:30, 5:30] = 1
    blob = {"segmentation": [[c for point in ring for c in point]
                             for ring in mask_to_polygon_rings(solid)]}
    # A merged sliced polygon wholly past the right edge clips to one column.
    past_the_edge = {"segmentation": [[105.0, 5.0, 130.0, 5.0, 130.0, 30.0, 105.0, 30.0]]}

    def result() -> dict:
        return {"image": "a.jpg", "width": 40, "height": 40, "labels": [1, 1, 1],
                "scores": [0.9, 0.8, 0.7],
                "boxes": [[5.0, 5.0, 30.0, 30.0], [5.0, 5.0, 30.0, 30.0],
                          [10.0, 5.0, 10.004, 30.0]],
                "masks": [past_the_edge, blob], "count": 3}

    written = result()
    data, _dropped = encode_predictions(written, "model:fixture",
                                        scope=registry_scope(tmp_path, "bur"))
    (kept,) = label_document(data).annotations
    assert written["count"] == 1
    assert kept.score == written["scores"][0] == 0.8


def test_check_box_extent_refuses_an_inverted_box():
    from tcip_annotation.json_io import check_box_extent

    with pytest.raises(ValueError, match="leaf"):
        check_box_extent(BBox(10, 10, 5, 5), where="subject 'leaf'")


def test_check_box_extent_refuses_a_zero_extent_box():
    from tcip_annotation.json_io import check_box_extent

    with pytest.raises(ValueError):
        check_box_extent(BBox(10, 10, 10, 20), where="subject 'leaf'")


def test_check_box_extent_admits_an_ordered_box():
    from tcip_annotation.json_io import check_box_extent

    check_box_extent(BBox(5, 5, 10, 20), where="subject 'leaf'")  # must not raise


def test_a_corner_box_is_refused_and_admitted_through_the_save_conversion():
    from tcip_annotation.json_io import annotation_from_payload

    with pytest.raises(ValueError, match="positive extent"):
        annotation_from_payload({"subject": "leaf", "bbox": [10, 10, 5, 5]})
    box = annotation_from_payload({"subject": "leaf", "bbox": [5, 5, 10, 20]}).geometry
    assert (box.x1, box.y1, box.x2, box.y2) == (5, 5, 10, 20)


# ── the persistence boundary: the one writer covers every writer at once ──


def test_the_writer_refuses_an_inverted_box(tmp_path):
    key = label_key(tmp_path, UNDATED_BUCKET, "img")
    with pytest.raises(ValueError):
        write_label_document(key, [Annotation(subject="leaf", geometry=BBox(10, 10, 5, 5))],
                             200, 150)
    assert ts.read(key, default=None) is None


def test_the_writer_admits_an_ordered_box(tmp_path):
    # admits valid work: every existing writer builds an ordered box.
    key = label_key(tmp_path, UNDATED_BUCKET, "img")
    write_label_document(key, [Annotation(subject="leaf", geometry=BBox(5, 5, 10, 20))], 200, 150)
    assert ts.read(key)["annotations"][0]["bbox"] == [5, 5, 5, 15]


def test_the_writer_refuses_a_collinear_polygon(tmp_path):
    """A polygon whose points all sit on one line has a derived bbox with no real extent either:
    the same boundary check catches it, not only a bare BBox."""
    key = label_key(tmp_path, UNDATED_BUCKET, "img")
    with pytest.raises(ValueError):
        write_label_document(
            key,
            [Annotation(subject="leaf", geometry=Polygon(rings=[[(5, 10), (8, 10), (12, 10)]]))],
            200, 150)
    assert ts.read(key, default=None) is None


def test_the_writer_admits_a_real_polygon(tmp_path):
    # admits valid work: a polygon with real area is unaffected by the collinear-polygon refusal.
    key = label_key(tmp_path, UNDATED_BUCKET, "img")
    write_label_document(
        key, [Annotation(subject="leaf", geometry=Polygon(rings=[[(5, 10), (15, 10), (15, 20)]]))],
        200, 150)
    assert "bbox" not in ts.read(key)["annotations"][0]
    assert read_label_document(key).annotations[0].geometry == Polygon(
        rings=[[(5, 10), (15, 10), (15, 20)]])


# ── the MCP save door ────────────────────────────────────────────────────────


def test_save_annotations_refuses_an_inverted_box(tmp_path):
    from tcip_mcp.tools.annotation_tools import save_annotations

    img = tmp_path / "images" / UNDATED_BUCKET / "img_001.jpg"
    write_image(img, (200, 150))

    result = save_annotations(tmp_path, tmp_path.parent, str(img),
                              annotations=[{"subject": "leaf", "bbox": [10, 10, 5, 5]}])

    assert "error" in result
    assert ts.read(image_label_key(img), default=None) is None


def test_save_annotations_admits_an_ordered_box(tmp_path):
    from tcip_mcp.tools.annotation_tools import save_annotations

    img = tmp_path / "images" / UNDATED_BUCKET / "img_001.jpg"
    write_image(img, (200, 150))

    result = save_annotations(tmp_path, tmp_path.parent, str(img),
                              annotations=[{"subject": "leaf", "bbox": [5, 5, 10, 20]}])

    assert "error" not in result
    assert len(read_label_document(image_label_key(img)).annotations) == 1


# ── the annotate route's save door ──────────────────────────────────────────


def test_annotate_save_refuses_an_inverted_box(client: TestClient, tmp_path: Path) -> None:
    img = tmp_path / "images" / UNDATED_BUCKET / "img_001.jpg"
    write_image(img, (200, 150))

    resp = client.post(
        "/api/annotate/labels",
        json={"image_path": str(img), "annotations": [{"subject": "leaf", "bbox": [10, 10, 5, 5]}],
              "user": "breeder"},
    )

    assert resp.status_code == 400
    assert ts.read(image_label_key(img), default=None) is None


def test_annotate_save_admits_an_ordered_box(client: TestClient, tmp_path: Path) -> None:
    img = tmp_path / "images" / UNDATED_BUCKET / "img_001.jpg"
    write_image(img, (200, 150))

    resp = client.post(
        "/api/annotate/labels",
        json={"image_path": str(img), "annotations": [{"subject": "leaf", "bbox": [5, 5, 10, 20]}],
              "user": "breeder"},
    )

    assert resp.status_code == 200
    assert len(read_label_document(image_label_key(img)).annotations) == 1


# ── the save door's corrected box and accepted proposal ─────────────────────


def _seed_review_dataset(tmp_path: Path, *, pred_box=(10, 10, 20, 20), gt_box=None):
    """The label document of one image and the bucket ``m`` proposing ``pred_box`` on it; the
    label document's key."""
    from tcip_mcp.subject_registry import SubjectRegistry, Subject
    from tests._producer_fixtures import label_image, registry_over
    from tests._web_fixtures import open_new_project

    dataset_root = open_new_project(tmp_path)
    img = dataset_root / "images" / UNDATED_BUCKET / "img_001.jpg"
    write_image(img, (200, 150))
    registry_over(dataset_root, SubjectRegistry(subjects=(Subject(name="leaf"),)))
    gt_annotations = (
        [Annotation(subject="leaf", geometry=BBox(*gt_box))] if gt_box is not None else []
    )
    label_image(img, gt_annotations, 200, 150, keep_empty=True)
    pytest.importorskip("torch")
    from tests._chain_fixtures import published

    x1, y1, x2, y2 = pred_box
    ordered = x2 > x1 and y2 > y1
    bucket = published(dataset_root, "m", [
        {"image": str(img), "width": 200, "height": 150,
         "boxes": [list(pred_box) if ordered else [10, 10, 20, 20]], "scores": [0.9],
         "labels": [1]}], scope={"subject": "leaf"})
    if not ordered:
        # A degenerate box can reach a document only by an edit in place after publication.
        ts.replace(bucket.document_key(img.stem), {
            "width": 200, "height": 150,
            "annotations": [{"subject": "leaf", "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": 0.9, "created_by": "m"}],
        })
    return image_label_key(img)


def _save(dataset_root: Path, annotations: list, **gestures) -> dict:
    return {"image_path": str(dataset_root / "images" / UNDATED_BUCKET / "img_001.jpg"),
            "annotations": annotations, "user": "breeder", **gestures}


def test_the_save_door_refuses_an_inverted_box(client: TestClient, tmp_path: Path) -> None:
    gt = _seed_review_dataset(tmp_path, gt_box=(1, 1, 3, 3))

    resp = client.post("/api/annotate/labels", json=_save(
        tmp_path, [{"subject": "leaf", "bbox": [10, 10, 5, 5]}]))

    assert resp.status_code == 400
    assert ts.read(gt)["annotations"][0]["bbox"] == [1, 1, 2, 2]


@pytest.mark.parametrize("shape", [
    {"points": [[10.0, 10.0]]},                                  # one vertex: no polygon
    {"bbox": [5, 5, 10, 20], "points": [[10.0, 10.0]]},          # beside a valid box
    {"bbox": ["10", "10", "50", "50"]},                          # coordinates as strings
    {"points": [["10", "10"], ["50", "10"], ["50", "50"]]},
], ids=["one_vertex_contour", "one_vertex_contour_beside_a_box", "string_box", "string_ring"])
def test_a_corrected_geometry_is_checked_as_any_saved_shape_is(
    client: TestClient, tmp_path: Path, shape,
) -> None:
    """A geometry correction is a save like any other, so every value the corrected shape carries
    is checked, whichever geometry it resolves to."""
    gt = _seed_review_dataset(tmp_path, gt_box=(1, 1, 3, 3))

    resp = client.post("/api/annotate/labels", json=_save(tmp_path, [{"subject": "leaf", **shape}]))

    assert resp.status_code == 400, resp.text
    assert ts.read(gt)["annotations"][0]["bbox"] == [1, 1, 2, 2]


def test_the_save_door_admits_an_ordered_corrected_box(client: TestClient, tmp_path: Path) -> None:
    gt = _seed_review_dataset(tmp_path, gt_box=(1, 1, 3, 3))

    resp = client.post("/api/annotate/labels", json=_save(
        tmp_path, [{"subject": "leaf", "bbox": [5, 5, 10, 20]}]))

    assert resp.status_code == 200
    assert ts.read(gt)["annotations"][0]["bbox"] == [5, 5, 5, 15]


def test_the_save_door_refuses_accepting_a_degenerate_proposal(
    client: TestClient, tmp_path: Path
) -> None:
    # A degenerate proposal reaching the document bypasses the publication's own drop (an
    # edit in place): accepting it still refuses.
    gt = _seed_review_dataset(tmp_path, pred_box=(10, 10, 10, 20))

    resp = client.post("/api/annotate/labels", json=_save(tmp_path, [], bucket="m", accept=[0]))

    assert resp.status_code == 400, resp.text
    assert ts.read(gt)["annotations"] == []


def test_the_save_door_admits_accepting_an_ordered_proposal(
    client: TestClient, tmp_path: Path
) -> None:
    gt = _seed_review_dataset(tmp_path)

    resp = client.post("/api/annotate/labels", json=_save(tmp_path, [], bucket="m", accept=[0]))

    assert resp.status_code == 200, resp.text
    assert ts.read(gt)["annotations"][0]["bbox"] == [10, 10, 10, 10]


# ── prediction writers drop a degenerate box and report it, rather than fail ─


LEAF = registry_scope(Path(__file__).parent, "leaf")
"""The class space a dataset with no registry reads ``leaf`` under: the subject, no attributes."""


def test_encode_predictions_drops_a_degenerate_box_and_reports_the_count():
    from tcip_mcp.pipelines.postprocessing.export import encode_predictions

    result = {
        "image": "preds.jpg", "width": 200, "height": 150,
        "boxes": [[10, 10, 20, 20], [30, 30, 30, 40]],  # the second collapses to zero width
        "scores": [0.9, 0.8],
        "labels": [1, 1],
    }

    data, dropped = encode_predictions(result, "model:fixture", scope=LEAF)

    assert dropped == 1
    assert len(data["annotations"]) == 1


def test_encode_predictions_drops_a_box_that_rounds_to_zero_extent():
    """A box with real pre-round extent that collapses to nothing at the document's stored
    2-decimal quantum is dropped here, the same as an already-zero-extent box: the encoder must
    never be handed a box it would refuse and fail the whole run over."""
    from tcip_mcp.pipelines.postprocessing.export import encode_predictions

    result = {
        "image": "preds.jpg", "width": 200, "height": 150,
        "boxes": [[10, 10, 20, 20], [30, 30, 30.003, 30.003]],
        "scores": [0.9, 0.8],
        "labels": [1, 1],
    }

    data, dropped = encode_predictions(result, "model:fixture", scope=LEAF)

    assert dropped == 1
    assert len(data["annotations"]) == 1


def test_stage_proposals_drops_a_degenerate_box_and_reports_the_count(tmp_path):
    from tcip_mcp.tools.proposal_tools import stage_proposals

    images_dir = tmp_path / "images" / "2026-01-01"
    image = images_dir / "img_001.jpg"
    write_image(image, (200, 150))

    result = stage_proposals(
        tmp_path, str(image), model_name="sam",
        boxes=[
            {"subject": "leaf", "conf": 0.9, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
            {"subject": "leaf", "conf": 0.8, "cx": 0.5, "cy": 0.5, "w": 0.0, "h": 0.2},
        ],
    )

    assert "error" not in result
    assert result["dropped_boxes"] == 1
    assert result["n_detect"] == 1


@pytest.mark.parametrize("shape", ["box", "polygon"])
def test_stage_proposals_refuses_a_shape_stating_no_confidence(tmp_path, shape):
    """A staged shape is a prediction, so it states the confidence its producer reported; one
    stating none is refused by name, never staged at a confidence nobody reported."""
    from tcip_mcp.tools.proposal_tools import stage_proposals

    image = tmp_path / "images" / "2026-01-01" / "img_001.jpg"
    write_image(image, (200, 150))
    unscored = ({"boxes": [{"subject": "leaf", "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}]}
                if shape == "box" else
                {"polygons": [{"subject": "leaf", "points": [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]]}]})

    result = stage_proposals(tmp_path, str(image), model_name="sam", **unscored)

    assert "conf" in result.get("error", ""), result


@pytest.mark.parametrize("shape", ["box", "polygon"])
def test_stage_proposals_refuses_a_shape_stating_no_subject_by_its_index(tmp_path, shape):
    """Every staged shape states its own subject; one stating none is refused naming the shape's
    index and the missing field, never staged under an empty subject."""
    from tcip_mcp.tools.proposal_tools import stage_proposals

    image = tmp_path / "images" / "2026-01-01" / "img_001.jpg"
    write_image(image, (200, 150))
    unstated = ({"boxes": [{"conf": 0.9, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}]}
                if shape == "box" else
                {"polygons": [{"conf": 0.9, "points": [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]]}]})

    result = stage_proposals(tmp_path, str(image), model_name="sam", **unstated)

    assert result.get("error", "").startswith(f"{shape} 0: "), result
    assert "subject" in result["error"] and "Field required" in result["error"]
