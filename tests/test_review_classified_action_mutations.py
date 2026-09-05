"""``/action`` under a classified bucket's own scope: the mutation branches that judge the
*value* of an object a person already placed, rather than the object's presence.

Built over a real staged bucket (``stage_prediction_shapes``) and a hand-authored ground-truth
file in the classified shape (the object class in ``subject``, the confirmed value under
``attributes[attribute]``), the bucket's own stamp seeded through the platform's own writers,
``operating_point_stamp`` and ``write_sidecar``, the same construction
``test_review_classified_scope.py`` uses.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tcip_annotation.json_io import read_annotations, write_annotations
from tcip_annotation.state import Annotation, BBox
from tcip_mcp.dataset_layout import prediction_dir
from tcip_mcp.pipelines.resolution import operating_point_stamp, write_sidecar
from tcip_mcp.prediction_buckets import stage_prediction_shapes
from tcip_web.app import app

DATE = "2026-04-01"
STEM = "IMG_0200"
IMG_W, IMG_H = 200, 150
BOX = (20.0, 20.0, 60.0, 60.0)
SUBJECT = "bud"
ATTRIBUTE = "opening"
ID_MAP = {"open": 0, "closed": 1}


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _image(dataset_root: Path) -> Path:
    img_dir = dataset_root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    path = img_dir / f"{STEM}.jpg"
    Image.new("RGB", (IMG_W, IMG_H), color=(90, 100, 80)).save(path)
    return path


def _stage(dataset_root: Path, *, value: str) -> dict:
    return stage_prediction_shapes(
        str(dataset_root), "classifier", DATE, STEM,
        annotations=[Annotation(subject=SUBJECT, geometry=BBox(*BOX), score=0.9,
                               attributes={ATTRIBUTE: value})],
        img_w=IMG_W, img_h=IMG_H,
    )


def _stamp_bucket(bucket: Path) -> None:
    stamp = operating_point_stamp(
        {"conf": {"value": 0.25}}, validated=False, validated_by=None,
        tile_size_validated=None, shippable_issues=[], id_map=ID_MAP,
        subject=SUBJECT, attribute=ATTRIBUTE, trait=ATTRIBUTE, dataset_hash="H",
        checkpoint="m", checkpoint_sha256="sha-classifier", experiment_id=None,
        images_dir=None, raster_path=None, produced_at="2026-04-01T00:00:00+00:00",
    )
    write_sidecar(bucket, stamp)


def _write_gt(dataset_root: Path, *, value: str) -> Path:
    gt_dir = dataset_root / "annotations" / DATE
    write_annotations(
        str(gt_dir / f"{STEM}.json"),
        [Annotation(subject=SUBJECT, geometry=BBox(*BOX), attributes={ATTRIBUTE: value},
                   created_by="user:breeder", created_at="2026-04-01T00:00:00+00:00")],
        IMG_W, IMG_H,
    )
    return gt_dir / f"{STEM}.json"


def _setup(tmp_path: Path, *, pred_value: str, gt_value: str) -> dict:
    """A staged, stamped classified bucket and a ground-truth file over the same box, one
    prediction confirming ``pred_value``, the person's own record confirming ``gt_value``."""
    dataset_root = tmp_path / "data"
    img = _image(dataset_root)
    staged = _stage(dataset_root, value=pred_value)
    bucket = Path(prediction_dir(dataset_root, "classifier", DATE))
    _stamp_bucket(bucket)
    gt = _write_gt(dataset_root, value=gt_value)
    return {"dataset_root": dataset_root, "img": img, "staged": staged, "bucket": bucket, "gt": gt}


def _matches(client: TestClient, s: dict) -> dict:
    resp = client.post("/api/review/matches", json={
        "dataset_root": str(s["dataset_root"]), "image_name": f"{STEM}.jpg",
        "image_path": str(s["img"]), "gt_path": str(s["gt"]), "pred_path": s["staged"]["path"],
        "iou_threshold": 0.3, "conf_threshold": 0.1,
    })
    assert resp.status_code == 200, resp.text
    return resp.json()


def _action(client: TestClient, s: dict, **payload):
    body = {
        "dataset_root": str(s["dataset_root"]), "image_name": f"{STEM}.jpg",
        "image_path": str(s["img"]), "gt_path": str(s["gt"]), "pred_path": s["staged"]["path"],
        "conf": None, "iou": None, "gt_idx": None, "pred_idx": None,
        "bbox": list(BOX), "user": "breeder",
        "iou_threshold": 0.3, "conf_threshold": 0.1,
    }
    body.update(payload)
    return client.post("/api/review/action", json=body)


def _shard(dataset_root: Path, image_name: str) -> dict:
    import tcip_store
    from tcip_annotation.review_engine import REVIEW_VERDICTS_STORE
    from tcip_mcp.prediction_buckets import review_state_dir_of

    state_dir = str(review_state_dir_of(dataset_root))
    found = [k for k in tcip_store.keys(REVIEW_VERDICTS_STORE, state_dir) if k.parts[1] == image_name]
    assert len(found) == 1, found
    return tcip_store.read(found[0])["state"]


def test_reject_on_the_paired_fp_leaves_the_record_and_the_fn_in_place(
    client: TestClient, tmp_path: Path,
) -> None:
    s = _setup(tmp_path, pred_value="open", gt_value="closed")
    matches = _matches(client, s)
    fp = next(d for d in matches["detections"] if d["det_type"] == "fp")
    before = s["gt"].read_bytes()

    resp = _action(
        client, s, det_type="fp", class_name="open", action="rejected",
        gt_idx=fp["gt_idx"], pred_idx=fp["pred_idx"], conf=fp["conf"],
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["matches"]["n_fp"] == 1 and resp.json()["matches"]["n_fn"] == 1
    assert s["gt"].read_bytes() == before


def test_accept_on_the_fn_keeps_it(client: TestClient, tmp_path: Path) -> None:
    s = _setup(tmp_path, pred_value="open", gt_value="closed")
    matches = _matches(client, s)
    fn = next(d for d in matches["detections"] if d["det_type"] == "fn")
    before = s["gt"].read_bytes()

    resp = _action(
        client, s, det_type="fn", class_name="closed", action="accepted",
        gt_idx=fn["gt_idx"], pred_idx=fn["pred_idx"],
    )

    assert resp.status_code == 200, resp.text
    assert s["gt"].read_bytes() == before


def test_edit_on_the_paired_fp_replaces_geometry_and_value_with_the_reviewer_as_author(
    client: TestClient, tmp_path: Path,
) -> None:
    s = _setup(tmp_path, pred_value="open", gt_value="closed")
    matches = _matches(client, s)
    fp = next(d for d in matches["detections"] if d["det_type"] == "fp")
    new_box = (25.0, 25.0, 65.0, 65.0)

    resp = _action(
        client, s, det_type="fp", class_name="open", action="edited",
        gt_idx=fp["gt_idx"], pred_idx=fp["pred_idx"], edited_box=list(new_box),
    )

    assert resp.status_code == 200, resp.text
    written = read_annotations(str(s["gt"]))
    assert len(written) == 1
    assert written[0].subject == SUBJECT
    assert written[0].attributes == {ATTRIBUTE: "open"}
    assert (written[0].geometry.x1, written[0].geometry.y1) == (new_box[0], new_box[1])
    assert written[0].created_by == "user:breeder"
    assert written[0].accepted_by is None


def test_reject_on_a_false_negative_under_a_classified_scope_refuses(
    client: TestClient, tmp_path: Path,
) -> None:
    s = _setup(tmp_path, pred_value="open", gt_value="closed")
    matches = _matches(client, s)
    fn = next(d for d in matches["detections"] if d["det_type"] == "fn")
    before = s["gt"].read_bytes()

    resp = _action(
        client, s, det_type="fn", class_name="closed", action="rejected",
        gt_idx=fn["gt_idx"], pred_idx=fn["pred_idx"],
    )

    assert resp.status_code == 400
    assert "detector-scope act" in resp.json()["detail"]
    assert s["gt"].read_bytes() == before


def test_an_out_of_range_edit_refuses(client: TestClient, tmp_path: Path) -> None:
    s = _setup(tmp_path, pred_value="open", gt_value="open")
    matches = _matches(client, s)
    assert matches["n_tp"] == 1
    tp = next(d for d in matches["detections"] if d["det_type"] == "tp")
    before = s["gt"].read_bytes()

    resp = _action(
        client, s, det_type="tp", class_name="open", action="edited",
        gt_idx=99, pred_idx=tp["pred_idx"], edited_box=list(BOX),
    )

    assert resp.status_code == 400
    assert "out of range" in resp.json()["detail"]
    assert s["gt"].read_bytes() == before


def test_a_reviewer_drawn_new_shape_under_a_classified_scope_refuses(
    client: TestClient, tmp_path: Path,
) -> None:
    s = _setup(tmp_path, pred_value="open", gt_value="closed")
    _matches(client, s)
    before = s["gt"].read_bytes()

    resp = _action(
        client, s, det_type="fp", class_name="open", action="edited",
        gt_idx=None, pred_idx=None, edited_box=list(BOX),
    )

    assert resp.status_code == 400
    assert "reviewer-drawn new shape" in resp.json()["detail"]
    assert s["gt"].read_bytes() == before


def test_accept_on_the_paired_fp_turns_the_image_completed_on_the_post_mutation_recompute(
    client: TestClient, tmp_path: Path,
) -> None:
    """The one detection this image carries is resolved by the accept: the post-mutation
    recompute (over the resolved classified scope) reads the image as fully reviewed."""
    s = _setup(tmp_path, pred_value="open", gt_value="closed")
    matches = _matches(client, s)
    fp = next(d for d in matches["detections"] if d["det_type"] == "fp")

    resp = _action(
        client, s, det_type="fp", class_name="open", action="accepted",
        gt_idx=fp["gt_idx"], pred_idx=fp["pred_idx"], conf=fp["conf"],
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["image_status"] == "completed"


def test_a_swept_attestation_stamps_missed_object_attested_and_a_paired_fp_verdict_does_not(
    client: TestClient, tmp_path: Path,
) -> None:
    s = _setup(tmp_path, pred_value="open", gt_value="closed")
    matches = _matches(client, s)
    fp = next(d for d in matches["detections"] if d["det_type"] == "fp")

    swept = _action(
        client, s, det_type="fp", class_name="open", action="swept",
        gt_idx=None, pred_idx=None,
    )
    assert swept.status_code == 200, swept.text
    state = _shard(s["dataset_root"], f"{STEM}.jpg")
    swept_entry = next(d for d in state["detections"] if d["action"] == "swept")
    assert swept_entry["missed_object_attested"] is True

    rejected = _action(
        client, s, det_type="fp", class_name="open", action="rejected",
        gt_idx=fp["gt_idx"], pred_idx=fp["pred_idx"],
    )
    assert rejected.status_code == 200, rejected.text
    state = _shard(s["dataset_root"], f"{STEM}.jpg")
    paired_entry = next(d for d in state["detections"] if d["action"] == "rejected")
    assert paired_entry["missed_object_attested"] is False
