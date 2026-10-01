"""The rule-admitted accept: a Review confirm of a prediction the bucket's own assessment admits,
verified before anything is written, refused by name otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_annotation.json_io import read_annotations, read_annotations_versioned, write_annotations
from tcip_annotation.state import Annotation
from tcip_web.app import app

from tests._web_fixtures import open_new_project

pytest.importorskip("torch")


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _reviewed(tmp_path: Path, bucket: Path, images_dir: Path) -> dict:
    """The first image of ``bucket`` holding a prediction, reviewed against a ground-truth file
    that holds nothing yet, so its first prediction reads as an fp; ``tmp_path`` opened in the web
    backend."""
    stem = next(p.stem for p in sorted(bucket.glob("*.json"))
                if p.name != "bucket.json" and read_annotations(p))
    open_new_project(tmp_path)
    return {"dataset_root": tmp_path / "ds", "bucket": bucket, "images_dir": images_dir,
            "stem": stem, "pred": read_annotations(bucket / f"{stem}.json")[0],
            "gt_path": tmp_path / "ds" / "annotations" / "review" / f"{stem}.json"}


def _assessed(tmp_path: Path) -> dict:
    """A bucket published under a passing assessment (``_chain_fixtures.run_the_chain``)."""
    from tests._chain_fixtures import run_the_chain

    chain = run_the_chain(tmp_path, experiment_id="exp-admit")
    return {**_reviewed(tmp_path, chain.bucket, chain.images_dir),
            "assessment_id": chain.assessment["assessment_id"]}


def _accept_payload(built: dict, *, det_type: str = "fp", gt_idx: int | None = None,
                    pred_idx: int | None = 0, rule_admitted: bool = True) -> dict:
    pred = built["pred"]
    box = pred.geometry
    return {
        "dataset_root": str(built["dataset_root"]), "image_name": f"{built['stem']}.png",
        "image_path": str(built["images_dir"] / f"{built['stem']}.png"),
        "gt_path": str(built["gt_path"]),
        "pred_path": str(built["bucket"] / f"{built['stem']}.json"),
        "det_type": det_type, "class_name": pred.subject, "conf": pred.score,
        "iou": 1.0 if det_type == "tp" else None, "gt_idx": gt_idx, "pred_idx": pred_idx,
        "bbox": [box.x1, box.y1, box.x2, box.y2], "action": "accepted",
        "rule_admitted": rule_admitted,
    }


def test_generation_conf_answers_the_rule_the_assessment_admits(
    client: TestClient, tmp_path: Path,
) -> None:
    from tcip_mcp.buckets import read_bucket

    built = _assessed(tmp_path)
    conf = read_bucket(built["bucket"]).execution.conf

    body = client.get("/api/review/generation_conf",
                      params={"pred_dir": str(built["bucket"])}).json()

    assert body["admission_reason"] == ""
    assert body["admission_conf"] == pytest.approx(conf)
    assert body["generation_conf"] == pytest.approx(conf)
    assert built["pred"].score >= conf


def test_a_verified_claim_writes_the_assessment_beside_the_person(
    client: TestClient, tmp_path: Path,
) -> None:
    """An accept on the fp with rule_admitted writes accepted_by_rule naming the assessment; a
    second accept on the now-tp answers reviewed and leaves the document's version unchanged."""
    import tcip_store
    from tcip_mcp.audit import audit_log_key

    built = _assessed(tmp_path)

    resp = client.post("/api/review/action", json=_accept_payload(built))

    assert resp.status_code == 200, resp.text
    fresh = resp.json()["matches"]
    assert fresh["gt"][0]["accepted_by_rule"] == built["assessment_id"]
    record = json.loads(built["gt_path"].read_text(encoding="utf-8"))["annotations"][0]
    assert record.get("accepted_by_rule") == built["assessment_id"]
    assert record.get("accepted_by", "").startswith("user:")
    assert "score" not in record
    last = tcip_store.read_log(audit_log_key(built["dataset_root"])).records[-1]
    assert last["arguments"]["rule_admitted"] is True
    assert last["arguments"]["accepted_by_rule"] == built["assessment_id"]

    _, version_before = read_annotations_versioned(built["gt_path"])
    again = client.post("/api/review/action", json=_accept_payload(built, det_type="tp", gt_idx=0))
    assert again.status_code == 200, again.text
    _, version_after = read_annotations_versioned(built["gt_path"])
    assert version_after == version_before


def test_an_ordinary_accept_on_the_same_bucket_writes_no_rule(
    client: TestClient, tmp_path: Path,
) -> None:
    built = _assessed(tmp_path)

    resp = client.post("/api/review/action", json=_accept_payload(built, rule_admitted=False))

    assert resp.status_code == 200, resp.text
    record = json.loads(built["gt_path"].read_text(encoding="utf-8"))["annotations"][0]
    assert record.get("accepted_by_rule") is None


def test_a_bucket_published_under_no_assessment_admits_no_claim(
    client: TestClient, tmp_path: Path,
) -> None:
    from tests._chain_fixtures import DATE, unassessed_bucket

    bucket = unassessed_bucket(tmp_path, experiment_id="exp-unassessed")
    built = _reviewed(tmp_path, bucket, tmp_path / "ds" / "images" / DATE)

    conf = client.get("/api/review/generation_conf", params={"pred_dir": str(bucket)}).json()
    resp = client.post("/api/review/action", json=_accept_payload(built))

    assert conf["admission_conf"] is None
    assert "no assessment answers for" in conf["admission_reason"]
    assert resp.status_code == 400, resp.text
    assert "no assessment answers for" in resp.text
    assert not built["gt_path"].exists()


def test_a_prediction_below_the_rules_conf_refuses(client: TestClient, tmp_path: Path) -> None:
    """A document rewritten after publication to hold a prediction under the published conf is not
    what the rule admits."""
    built = _assessed(tmp_path)
    pred_path = built["bucket"] / f"{built['stem']}.json"
    write_annotations(pred_path, [Annotation(subject=built["pred"].subject,
                                             geometry=built["pred"].geometry, score=0.01)],
                      64, 64)
    built["pred"] = read_annotations(pred_path)[0]

    resp = client.post("/api/review/action", json=_accept_payload(built))

    assert resp.status_code == 400, resp.text
    assert "below the rule" in resp.text


def test_a_scoreless_prediction_refuses_naming_the_key(client: TestClient, tmp_path: Path) -> None:
    """A prediction record with no score is refused where it is read: no reader stands in a
    confidence the model never reported."""
    built = _assessed(tmp_path)
    pred_path = built["bucket"] / f"{built['stem']}.json"
    write_annotations(pred_path, [Annotation(subject=built["pred"].subject,
                                             geometry=built["pred"].geometry)], 64, 64)

    payload = {**_accept_payload(built), "conf": None}
    resp = client.post("/api/review/action", json=payload)

    assert resp.status_code == 400, resp.text
    assert "'score'" in resp.text


def test_a_classified_bucket_admits_an_ordinary_accept_and_refuses_a_claim(
    client: TestClient, tmp_path: Path,
) -> None:
    """The assessment's rule admits detections, not values."""
    from PIL import Image

    from tests._chain_fixtures import published

    id_map = {"open": 0, "closed": 1}
    bucket = tmp_path / "ds" / "predictions" / "m" / "2026-03-04"
    published(tmp_path, bucket, [{"image": "img.png", "width": 64, "height": 64,
                                  "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9],
                                  "labels": [1]}],
              scope={"subject": "bud", "attribute": "state", "id_map": id_map})
    images_dir = tmp_path / "ds" / "images" / "2026-03-04"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64)).save(images_dir / "img.png")
    built = _reviewed(tmp_path, bucket, images_dir)
    payload = {**_accept_payload(built, rule_admitted=False), "class_name": "open"}

    ordinary = client.post("/api/review/action", json=payload)
    claimed = client.post("/api/review/action", json={**payload, "rule_admitted": True})

    assert ordinary.status_code == 200, ordinary.text
    assert claimed.status_code == 400, claimed.text
    assert "classified" in claimed.text


def test_a_stale_detection_refuses_409_rather_than_writing_twice(
    client: TestClient, tmp_path: Path,
) -> None:
    """A ground-truth record written under the fp's box after the tab's matches were taken makes
    the submitted fp a tp in a fresh recompute."""
    built = _assessed(tmp_path)
    built["gt_path"].parent.mkdir(parents=True, exist_ok=True)
    write_annotations(built["gt_path"], [Annotation(subject=built["pred"].subject,
                                                    geometry=built["pred"].geometry,
                                                    created_by="user:someone_else")], 64, 64)

    resp = client.post("/api/review/action", json=_accept_payload(built))

    assert resp.status_code == 409, resp.text


@pytest.mark.parametrize("change, named", [
    ({"action": "rejected"}, "rejected"),
    ({"det_type": "fn"}, "fn"),
    ({"pred_idx": 5}, "pred_idx"),
], ids=["wrong-action", "wrong-det-type", "out-of-range-pred-idx"])
def test_a_claim_on_the_wrong_act_refuses_by_name(
    client: TestClient, tmp_path: Path, change: dict, named: str,
) -> None:
    built = _assessed(tmp_path)

    resp = client.post("/api/review/action", json={**_accept_payload(built), **change})

    assert resp.status_code == 400, resp.text
    assert named in resp.text


def test_the_rail_admits_the_directory_a_rule_admitted_accept_produced(
    client: TestClient, tmp_path: Path,
) -> None:
    """The reference rail fed the directory the accept actually produced: a signed, rule-admitted
    record is admitted."""
    from tcip_annotation.json_io import require_reference_ground_truth

    built = _assessed(tmp_path)
    resp = client.post("/api/review/action", json=_accept_payload(built))
    assert resp.status_code == 200, resp.text

    require_reference_ground_truth([built["gt_path"]])
