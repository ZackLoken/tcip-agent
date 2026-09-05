"""An inverted or zero-extent box is refused wherever a writer builds one, and a save naming no
label document is refused rather than silently skipped: both touch the same route module and the
same per-image reader/writer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tcip_annotation.json_io import write_annotations
from tcip_annotation.state import Annotation, BBox, Polygon


def _write_image(path: Path, size=(200, 150)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(100, 100, 100)).save(path)


@pytest.fixture
def client() -> TestClient:
    from tcip_web.app import app

    return TestClient(app, base_url="http://127.0.0.1")


# ── the shared constructor ──────────────────────────────────────────────────


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


def test_bbox_from_corners_refuses_and_admits():
    from tcip_annotation.json_io import bbox_from_corners

    with pytest.raises(ValueError):
        bbox_from_corners(10, 10, 5, 5, where="subject 'leaf'")
    box = bbox_from_corners(5, 5, 10, 20, where="subject 'leaf'")
    assert (box.x1, box.y1, box.x2, box.y2) == (5, 5, 10, 20)


# ── the persistence boundary: write_annotations covers every writer at once ──


def test_write_annotations_refuses_an_inverted_box(tmp_path):
    with pytest.raises(ValueError):
        write_annotations(
            str(tmp_path / "img.json"),
            [Annotation(subject="leaf", geometry=BBox(10, 10, 5, 5))],
            200, 150,
        )
    assert not (tmp_path / "img.json").exists()


def test_write_annotations_admits_an_ordered_box(tmp_path):
    # admits valid work: every existing writer builds an ordered box.
    path = tmp_path / "img.json"
    write_annotations(
        str(path), [Annotation(subject="leaf", geometry=BBox(5, 5, 10, 20))], 200, 150,
    )
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["annotations"][0]["bbox"] == [5, 5, 5, 15]


def test_write_annotations_refuses_a_collinear_polygon(tmp_path):
    """A polygon whose points all sit on one line has a derived bbox with no real extent either:
    the same boundary check catches it, not only a bare BBox."""
    with pytest.raises(ValueError):
        write_annotations(
            str(tmp_path / "img.json"),
            [Annotation(subject="leaf", geometry=Polygon(rings=[[(5, 10), (8, 10), (12, 10)]]))],
            200, 150,
        )
    assert not (tmp_path / "img.json").exists()


def test_write_annotations_admits_a_real_polygon(tmp_path):
    # admits valid work: a polygon with real area is unaffected by the collinear-polygon refusal.
    path = tmp_path / "img.json"
    write_annotations(
        str(path),
        [Annotation(subject="leaf", geometry=Polygon(rings=[[(5, 10), (15, 10), (15, 20)]]))],
        200, 150,
    )
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["annotations"][0]["bbox"] == [5, 10, 10, 10]


# ── the MCP save door ────────────────────────────────────────────────────────


def test_save_annotations_refuses_an_inverted_box(tmp_path):
    from tcip_mcp.tools.annotation_tools import save_annotations

    img = tmp_path / "images" / "img_001.jpg"
    _write_image(img)
    out_path = tmp_path / "annotations" / "img_001.json"

    result = save_annotations(
        str(img), annotations=[{"subject": "leaf", "bbox": [10, 10, 5, 5]}],
        path=str(out_path),
    )

    assert "error" in result
    assert not out_path.exists()


def test_save_annotations_admits_an_ordered_box(tmp_path):
    from tcip_mcp.tools.annotation_tools import save_annotations

    img = tmp_path / "images" / "img_001.jpg"
    _write_image(img)
    out_path = tmp_path / "annotations" / "img_001.json"

    result = save_annotations(
        str(img), annotations=[{"subject": "leaf", "bbox": [5, 5, 10, 20]}],
        path=str(out_path),
    )

    assert "error" not in result
    assert out_path.is_file()


# ── the annotate route's save door ──────────────────────────────────────────


def test_annotate_save_refuses_an_inverted_box(client: TestClient, tmp_path: Path) -> None:
    img = tmp_path / "images" / "img_001.jpg"
    _write_image(img)
    label_path = tmp_path / "annotations" / "img_001.json"

    resp = client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(img), "label_path": str(label_path),
            "annotations": [{"subject": "leaf", "bbox": [10, 10, 5, 5]}],
        },
    )

    assert resp.status_code == 400
    assert not label_path.exists()


def test_annotate_save_admits_an_ordered_box(client: TestClient, tmp_path: Path) -> None:
    img = tmp_path / "images" / "img_001.jpg"
    _write_image(img)
    label_path = tmp_path / "annotations" / "img_001.json"

    resp = client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(img), "label_path": str(label_path),
            "annotations": [{"subject": "leaf", "bbox": [5, 5, 10, 20]}],
        },
    )

    assert resp.status_code == 200
    assert label_path.is_file()


def test_annotate_save_refuses_an_empty_label_path(client: TestClient, tmp_path: Path) -> None:
    img = tmp_path / "images" / "img_001.jpg"
    _write_image(img)

    resp = client.post(
        "/api/annotate/labels",
        json={"image_path": str(img), "label_path": "", "annotations": []},
    )

    assert resp.status_code == 422


def test_annotate_save_admits_every_selected_dataset_save(
    client: TestClient, tmp_path: Path
) -> None:
    # admits valid work: every Python caller passes a real label path.
    img = tmp_path / "images" / "img_001.jpg"
    _write_image(img)
    label_path = tmp_path / "annotations" / "img_001.json"

    resp = client.post(
        "/api/annotate/labels",
        json={"image_path": str(img), "label_path": str(label_path), "annotations": []},
    )

    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── the review action's edited box and accept branch ────────────────────────


def _seed_review_dataset(tmp_path: Path, *, pred_box=(10, 10, 20, 20), gt_box=None) -> tuple[Path, Path]:
    from tcip_mcp.class_registry import ClassRegistry, Subject, write_registry

    dataset_root = tmp_path
    img = dataset_root / "images" / "img_001.jpg"
    _write_image(img)
    write_registry(dataset_root / "classes.json", ClassRegistry(subjects=(Subject(name="leaf"),)))
    gt_path = dataset_root / "annotations" / "img_001.json"
    gt_annotations = (
        [Annotation(subject="leaf", geometry=BBox(*gt_box))] if gt_box is not None else []
    )
    write_annotations(str(gt_path), gt_annotations, 200, 150, keep_empty=True)
    pred_path = dataset_root / "predictions" / "m" / "img_001.json"
    x1, y1, x2, y2 = pred_box
    if x2 > x1 and y2 > y1:
        write_annotations(
            str(pred_path),
            [Annotation(subject="leaf", geometry=BBox(*pred_box), score=0.9, created_by="m")],
            200, 150, keep_empty=True,
        )
    else:
        # A degenerate box can no longer reach this file through write_annotations (the
        # persistence boundary refuses it); placed directly, as a foreign/hand-edited file would.
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_path.write_text(json.dumps({
            "image": "img_001", "width": 200, "height": 150,
            "annotations": [{"subject": "leaf", "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": 0.9, "created_by": "m"}],
        }), encoding="utf-8")
    return gt_path, pred_path


def _action_payload(dataset_root: Path, gt_path: Path, pred_path: Path, **overrides) -> dict:
    payload = {
        "dataset_root": str(dataset_root),
        "image_name": "img_001.jpg",
        "image_path": str(dataset_root / "images" / "img_001.jpg"),
        "gt_path": str(gt_path),
        "pred_path": str(pred_path),
        "det_type": "fp",
        "class_name": "leaf",
        "pred_idx": 0,
        "bbox": [10, 10, 20, 20],
        "action": "accepted",
    }
    payload.update(overrides)
    return payload


def test_review_action_refuses_an_inverted_edited_box(client: TestClient, tmp_path: Path) -> None:
    gt_path, pred_path = _seed_review_dataset(tmp_path, gt_box=(1, 1, 3, 3))

    resp = client.post(
        "/api/review/action",
        json=_action_payload(
            tmp_path, gt_path, pred_path, det_type="fn", gt_idx=0,
            action="edited", edited_box=[10, 10, 5, 5],
        ),
    )

    assert resp.status_code == 400
    assert json.loads(gt_path.read_text())["annotations"][0]["bbox"] == [1, 1, 2, 2]


def test_review_action_admits_an_ordered_edited_box(client: TestClient, tmp_path: Path) -> None:
    gt_path, pred_path = _seed_review_dataset(tmp_path, gt_box=(1, 1, 3, 3))

    resp = client.post(
        "/api/review/action",
        json=_action_payload(
            tmp_path, gt_path, pred_path, det_type="fn", gt_idx=0,
            action="edited", edited_box=[5, 5, 10, 20],
        ),
    )

    assert resp.status_code == 200
    assert json.loads(gt_path.read_text())["annotations"][0]["bbox"] == [5, 5, 5, 15]


def test_review_action_refuses_accepting_a_degenerate_prediction(
    client: TestClient, tmp_path: Path
) -> None:
    # A degenerate prediction reaching the store bypasses the staging door's own drop (a file
    # placed directly, as a foreign bucket might hold): the accept branch still refuses it.
    gt_path, pred_path = _seed_review_dataset(tmp_path, pred_box=(10, 10, 10, 20))

    resp = client.post("/api/review/action", json=_action_payload(tmp_path, gt_path, pred_path))

    assert resp.status_code == 400
    assert json.loads(gt_path.read_text())["annotations"] == []


def test_review_action_admits_accepting_an_ordered_prediction(
    client: TestClient, tmp_path: Path
) -> None:
    gt_path, pred_path = _seed_review_dataset(tmp_path)

    resp = client.post("/api/review/action", json=_action_payload(tmp_path, gt_path, pred_path))

    assert resp.status_code == 200
    assert json.loads(gt_path.read_text())["annotations"][0]["bbox"] == [10, 10, 10, 10]


# ── prediction writers drop a degenerate box and report it, rather than fail ─


def test_write_predictions_json_drops_a_degenerate_box_and_reports_the_count(tmp_path):
    from tcip_mcp.pipelines.postprocessing.export import write_predictions_json

    out = tmp_path / "preds.json"
    result = {
        "width": 200, "height": 150,
        "boxes": [[10, 10, 20, 20], [30, 30, 30, 40]],  # the second collapses to zero width
        "scores": [0.9, 0.8],
        "labels": [1, 1],
    }

    dropped = write_predictions_json(out, result, subject="leaf", attribute=None, id_map={"leaf": 0})

    assert dropped == 1
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert len(saved["annotations"]) == 1


def test_write_predictions_json_drops_a_box_that_rounds_to_zero_extent(tmp_path):
    """A box with real pre-round extent that collapses to nothing at the document's stored
    2-decimal quantum is dropped here, the same as an already-zero-extent box: the writer must
    never be handed a box it would refuse and fail the whole run over."""
    from tcip_annotation.json_io import read_annotations
    from tcip_mcp.pipelines.postprocessing.export import write_predictions_json

    out = tmp_path / "preds.json"
    result = {
        "width": 200, "height": 150,
        "boxes": [[10, 10, 20, 20], [30, 30, 30.003, 30.003]],
        "scores": [0.9, 0.8],
        "labels": [1, 1],
    }

    dropped = write_predictions_json(out, result, subject="leaf", attribute=None, id_map={"leaf": 0})

    assert dropped == 1
    assert len(read_annotations(out)) == 1


def test_write_predictions_json_refuses_a_reserved_stem(tmp_path):
    """An image stem reserved for a bucket's own provenance stamp must never reach a per-image
    prediction write: the stamp write into that same bucket would otherwise destroy or refuse
    over it, naming the operator at a file that was never the actual cause."""
    from tcip_mcp.pipelines.postprocessing.export import write_predictions_json

    out = tmp_path / "operating_point.json"
    result = {"width": 100, "height": 100, "boxes": [[1, 1, 5, 5]], "scores": [0.9], "labels": [1]}

    with pytest.raises(ValueError, match="operating_point"):
        write_predictions_json(out, result, subject="leaf", attribute=None, id_map={"leaf": 0})
    assert not out.exists()


def test_write_predictions_json_still_writes_an_ordinary_stem(tmp_path):
    from tcip_annotation.json_io import read_annotations
    from tcip_mcp.pipelines.postprocessing.export import write_predictions_json

    out = tmp_path / "IMG_0001.json"
    result = {"width": 100, "height": 100, "boxes": [[1, 1, 5, 5]], "scores": [0.9], "labels": [1]}

    write_predictions_json(out, result, subject="leaf", attribute=None, id_map={"leaf": 0})
    assert len(read_annotations(out)) == 1


def test_stage_proposals_drops_a_degenerate_box_and_reports_the_count(tmp_path):
    from tcip_mcp.tools.proposal_tools import stage_proposals

    images_dir = tmp_path / "images" / "2026-01-01"
    image = images_dir / "img_001.jpg"
    _write_image(image)

    result = stage_proposals(
        str(image), model_name="sam",
        boxes=[
            {"subject": "leaf", "conf": 0.9, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
            {"subject": "leaf", "conf": 0.8, "cx": 0.5, "cy": 0.5, "w": 0.0, "h": 0.2},
        ],
    )

    assert "error" not in result
    assert result["dropped_nonpositive_boxes"] == 1
    assert result["n_detect"] == 1
