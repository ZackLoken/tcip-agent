"""The Review routes under a classified bucket's own recorded scope: /matches and /action judge
the confirmed value of an object a person already placed, rather than the object's presence.

Built over a real staged bucket (``stage_prediction_shapes``) and a hand-authored ground-truth
file in the classified shape (the object class in ``subject``, the confirmed value under
``attributes[attribute]``, the same shape a real accept would have written); the bucket's own
stamp is seeded through the platform's own writers, ``operating_point_stamp`` and
``write_sidecar``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tcip_annotation.json_io import write_annotations
from tcip_annotation.state import Annotation, BBox
from tcip_mcp.dataset_layout import prediction_dir
from tcip_mcp.pipelines.resolution import operating_point_stamp, write_sidecar
from tcip_mcp.prediction_buckets import stage_prediction_shapes
from tcip_web.app import app

DATE = "2026-03-05"
STEM = "IMG_0100"
IMG_W, IMG_H = 200, 150
BOX = (20.0, 20.0, 60.0, 60.0)
SUBJECT = "leaf"
ATTRIBUTE = "condition"
ID_MAP = {"healthy": 0, "diseased": 1}


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _image(dataset_root: Path) -> Path:
    img_dir = dataset_root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    path = img_dir / f"{STEM}.jpg"
    Image.new("RGB", (IMG_W, IMG_H), color=(90, 100, 80)).save(path)
    return path


def _stage_classified_prediction(dataset_root: Path, *, value: str) -> dict:
    """One classified prediction record: the object class in subject, the decoded value under
    attributes[attribute], the shape write_predictions_json now writes."""
    return stage_prediction_shapes(
        str(dataset_root), "classifier", DATE, STEM,
        annotations=[Annotation(subject=SUBJECT, geometry=BBox(*BOX), score=0.9,
                               attributes={ATTRIBUTE: value})],
        img_w=IMG_W, img_h=IMG_H,
    )


def _stamp_classified_bucket(bucket: Path) -> None:
    stamp = operating_point_stamp(
        {"conf": {"value": 0.25}}, validated=False, validated_by=None,
        tile_size_validated=None, shippable_issues=[], id_map=ID_MAP,
        subject=SUBJECT, attribute=ATTRIBUTE, trait=ATTRIBUTE, dataset_hash="H",
        checkpoint="m", checkpoint_sha256="sha-classifier", experiment_id=None,
        images_dir=None, raster_path=None, produced_at="2026-03-05T00:00:00+00:00",
    )
    write_sidecar(bucket, stamp)


def _bare_bucket(dataset_root: Path, *, value: str) -> Path:
    """A bucket with no stamp at all, holding the same shape."""
    d = dataset_root / "predictions" / "bare" / DATE
    write_annotations(
        str(d / f"{STEM}.json"),
        [Annotation(subject=SUBJECT, geometry=BBox(*BOX), score=0.9,
                   attributes={ATTRIBUTE: value})],
        IMG_W, IMG_H,
    )
    return d


def _write_gt(dataset_root: Path, *, value: str) -> Path:
    gt_dir = dataset_root / "annotations" / DATE
    write_annotations(
        str(gt_dir / f"{STEM}.json"),
        [Annotation(subject=SUBJECT, geometry=BBox(*BOX), attributes={ATTRIBUTE: value},
                   created_by="user:breeder", created_at="2026-03-05T00:00:00+00:00")],
        IMG_W, IMG_H,
    )
    return gt_dir / f"{STEM}.json"


def test_matches_resolves_the_classified_scope_with_no_request_side_statement(
    client: TestClient, tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "data"
    img = _image(dataset_root)
    staged = _stage_classified_prediction(dataset_root, value="healthy")
    bucket = Path(prediction_dir(dataset_root, "classifier", DATE))
    _stamp_classified_bucket(bucket)
    gt = _write_gt(dataset_root, value="diseased")

    resp = client.post("/api/review/matches", json={
        "dataset_root": str(dataset_root), "image_name": f"{STEM}.jpg", "image_path": str(img),
        "gt_path": str(gt), "pred_path": staged["path"],
        "iou_threshold": 0.3, "conf_threshold": 0.1,
    })

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["subject"] == SUBJECT and body["attribute"] == ATTRIBUTE


def test_a_disagreeing_stated_pair_refuses(client: TestClient, tmp_path: Path) -> None:
    dataset_root = tmp_path / "data"
    img = _image(dataset_root)
    staged = _stage_classified_prediction(dataset_root, value="healthy")
    bucket = Path(prediction_dir(dataset_root, "classifier", DATE))
    _stamp_classified_bucket(bucket)
    gt = _write_gt(dataset_root, value="diseased")

    resp = client.post("/api/review/matches", json={
        "dataset_root": str(dataset_root), "image_name": f"{STEM}.jpg", "image_path": str(img),
        "gt_path": str(gt), "pred_path": staged["path"],
        "subject": SUBJECT, "attribute": "a-different-attribute",
    })

    assert resp.status_code == 400
    assert "this bucket's stamp records scope" in resp.json()["detail"]


def test_a_disagreeing_stated_subject_alone_refuses(client: TestClient, tmp_path: Path) -> None:
    """A stated subject that disagrees with the bucket's own recorded scope refuses even when no
    attribute is stated alongside it, rather than being silently replaced by the bucket's own."""
    dataset_root = tmp_path / "data"
    img = _image(dataset_root)
    staged = _stage_classified_prediction(dataset_root, value="healthy")
    bucket = Path(prediction_dir(dataset_root, "classifier", DATE))
    _stamp_classified_bucket(bucket)
    gt = _write_gt(dataset_root, value="diseased")

    resp = client.post("/api/review/matches", json={
        "dataset_root": str(dataset_root), "image_name": f"{STEM}.jpg", "image_path": str(img),
        "gt_path": str(gt), "pred_path": staged["path"],
        "subject": "a-different-subject",
    })

    assert resp.status_code == 400
    assert "this bucket's stamp records scope" in resp.json()["detail"]


def test_a_bare_directory_under_a_stated_attribute_refuses(
    client: TestClient, tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "data"
    img = _image(dataset_root)
    bare = _bare_bucket(dataset_root, value="healthy")
    gt = _write_gt(dataset_root, value="diseased")

    resp = client.post("/api/review/matches", json={
        "dataset_root": str(dataset_root), "image_name": f"{STEM}.jpg", "image_path": str(img),
        "gt_path": str(gt), "pred_path": str(bare / f"{STEM}.json"),
        "subject": SUBJECT, "attribute": ATTRIBUTE,
    })

    assert resp.status_code == 400
    assert "carries no stamp and no scope" in resp.json()["detail"]


def test_a_same_geometry_value_disagreement_pairs_as_an_fp_and_fn(
    client: TestClient, tmp_path: Path,
) -> None:
    """One ground-truth record confirmed 'diseased', one prediction of 'healthy' over the same
    box: the two never match by value, but the geometry-only second pass pairs them, each
    carrying the other's index."""
    dataset_root = tmp_path / "data"
    img = _image(dataset_root)
    staged = _stage_classified_prediction(dataset_root, value="healthy")
    bucket = Path(prediction_dir(dataset_root, "classifier", DATE))
    _stamp_classified_bucket(bucket)
    gt = _write_gt(dataset_root, value="diseased")

    resp = client.post("/api/review/matches", json={
        "dataset_root": str(dataset_root), "image_name": f"{STEM}.jpg", "image_path": str(img),
        "gt_path": str(gt), "pred_path": staged["path"],
        "iou_threshold": 0.3, "conf_threshold": 0.1,
    })

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["n_tp"] == 0 and body["n_fp"] == 1 and body["n_fn"] == 1
    fp = next(d for d in body["detections"] if d["det_type"] == "fp")
    fn = next(d for d in body["detections"] if d["det_type"] == "fn")
    assert fp["class_name"] == "healthy" and fn["class_name"] == "diseased"
    assert fp["gt_idx"] is not None and fp["gt_idx"] == fn["gt_idx"]
    assert fn["pred_idx"] is not None and fn["pred_idx"] == fp["pred_idx"]


def test_accept_on_the_paired_fp_replaces_the_value_and_keeps_geometry_and_authorship(
    client: TestClient, tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "data"
    img = _image(dataset_root)
    staged = _stage_classified_prediction(dataset_root, value="healthy")
    bucket = Path(prediction_dir(dataset_root, "classifier", DATE))
    _stamp_classified_bucket(bucket)
    gt = _write_gt(dataset_root, value="diseased")

    matches = client.post("/api/review/matches", json={
        "dataset_root": str(dataset_root), "image_name": f"{STEM}.jpg", "image_path": str(img),
        "gt_path": str(gt), "pred_path": staged["path"],
        "iou_threshold": 0.3, "conf_threshold": 0.1,
    }).json()
    fp = next(d for d in matches["detections"] if d["det_type"] == "fp")

    resp = client.post("/api/review/action", json={
        "dataset_root": str(dataset_root), "image_name": f"{STEM}.jpg", "image_path": str(img),
        "gt_path": str(gt), "pred_path": staged["path"],
        "det_type": "fp", "class_name": "healthy", "conf": fp["conf"],
        "iou": None, "gt_idx": fp["gt_idx"], "pred_idx": fp["pred_idx"],
        "bbox": list(BOX), "action": "accepted", "user": "breeder",
        "iou_threshold": 0.3, "conf_threshold": 0.1,
    })

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["matches"]["n_tp"] == 1
    assert body["matches"]["n_fp"] == 0 and body["matches"]["n_fn"] == 0

    from tcip_annotation.json_io import read_annotations

    written = read_annotations(str(gt))
    assert len(written) == 1
    assert written[0].subject == SUBJECT
    assert written[0].attributes == {ATTRIBUTE: "healthy"}
    assert written[0].geometry.x1 == BOX[0] and written[0].geometry.y1 == BOX[1]
    assert written[0].created_by == "user:breeder"  # the person's own authorship kept
    assert written[0].accepted_by == "user:breeder"


def test_action_refusal_leaves_the_label_file_unchanged(client: TestClient, tmp_path: Path) -> None:
    """The staged prediction is a pre-conform record, its ``subject`` the value rather than this
    bucket's object class. The pre-mutation checks only test the accepted value against the
    vocabulary, so an in-vocabulary ``class_name`` admits it and the mutation appends a GT record;
    the post-mutation recompute then holds every prediction record positively to the object class
    and refuses this one. The GT file on disk is exactly what it was before the call, never a
    write behind the 400."""
    dataset_root = tmp_path / "data"
    img = _image(dataset_root)
    staged = stage_prediction_shapes(
        str(dataset_root), "classifier", DATE, STEM,
        annotations=[Annotation(subject="healthy", geometry=BBox(*BOX), score=0.9)],
        img_w=IMG_W, img_h=IMG_H,
    )
    bucket = Path(prediction_dir(dataset_root, "classifier", DATE))
    _stamp_classified_bucket(bucket)
    gt = _write_gt(dataset_root, value="diseased")
    before = gt.read_bytes()

    resp = client.post("/api/review/action", json={
        "dataset_root": str(dataset_root), "image_name": f"{STEM}.jpg", "image_path": str(img),
        "gt_path": str(gt), "pred_path": staged["path"],
        "det_type": "fp", "class_name": "healthy", "conf": 0.9,
        "iou": None, "gt_idx": None, "pred_idx": 0,
        "bbox": list(BOX), "action": "accepted", "user": "breeder",
        "iou_threshold": 0.3, "conf_threshold": 0.1,
    })

    assert resp.status_code == 400
    assert gt.read_bytes() == before


def test_accept_paired_to_a_foreign_subject_record_refuses(
    client: TestClient, tmp_path: Path,
) -> None:
    """A paired accept's ``gt_idx`` must name a record of this bucket's own object class: a client
    that names a foreign-subject record's index is refused by name rather than confirming a value
    onto an object this review was never scoped to."""
    dataset_root = tmp_path / "data"
    img = _image(dataset_root)
    staged = _stage_classified_prediction(dataset_root, value="healthy")
    bucket = Path(prediction_dir(dataset_root, "classifier", DATE))
    _stamp_classified_bucket(bucket)
    gt_dir = dataset_root / "annotations" / DATE
    write_annotations(
        str(gt_dir / f"{STEM}.json"),
        [Annotation(subject="bush", geometry=BBox(*BOX), created_by="user:breeder",
                   created_at="2026-03-05T00:00:00+00:00")],
        IMG_W, IMG_H,
    )
    gt = gt_dir / f"{STEM}.json"
    before = gt.read_bytes()

    resp = client.post("/api/review/action", json={
        "dataset_root": str(dataset_root), "image_name": f"{STEM}.jpg", "image_path": str(img),
        "gt_path": str(gt), "pred_path": staged["path"],
        "det_type": "fp", "class_name": "healthy", "conf": 0.9,
        "iou": None, "gt_idx": 0, "pred_idx": 0,
        "bbox": list(BOX), "action": "accepted", "user": "breeder",
        "iou_threshold": 0.3, "conf_threshold": 0.1,
    })

    assert resp.status_code == 400
    assert "bush" in resp.json()["detail"]
    assert gt.read_bytes() == before


def test_reject_on_a_true_positive_under_a_classified_scope_refuses(
    client: TestClient, tmp_path: Path,
) -> None:
    """A confirmed value the model also predicted (a tp) rejected would remove the object, a
    detector-scope act a classified review never adjudicated."""
    dataset_root = tmp_path / "data"
    img = _image(dataset_root)
    staged = _stage_classified_prediction(dataset_root, value="healthy")
    bucket = Path(prediction_dir(dataset_root, "classifier", DATE))
    _stamp_classified_bucket(bucket)
    gt = _write_gt(dataset_root, value="healthy")

    matches = client.post("/api/review/matches", json={
        "dataset_root": str(dataset_root), "image_name": f"{STEM}.jpg", "image_path": str(img),
        "gt_path": str(gt), "pred_path": staged["path"],
        "iou_threshold": 0.3, "conf_threshold": 0.1,
    }).json()
    assert matches["n_tp"] == 1
    tp = next(d for d in matches["detections"] if d["det_type"] == "tp")

    resp = client.post("/api/review/action", json={
        "dataset_root": str(dataset_root), "image_name": f"{STEM}.jpg", "image_path": str(img),
        "gt_path": str(gt), "pred_path": staged["path"],
        "det_type": "tp", "class_name": "healthy", "conf": tp["conf"],
        "iou": tp["iou"], "gt_idx": tp["gt_idx"], "pred_idx": tp["pred_idx"],
        "bbox": list(BOX), "action": "rejected", "user": "breeder",
        "iou_threshold": 0.3, "conf_threshold": 0.1,
    })

    assert resp.status_code == 400
    assert "detector-scope act" in resp.json()["detail"]
