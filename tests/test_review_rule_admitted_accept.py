"""The rule-admitted accept: a Review confirm of a prediction the bucket's own validated count
operating point pre-admits, verified against the binding before anything is written, refused by
name otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_annotation.json_io import read_annotations, read_annotations_versioned, write_annotations
from tcip_annotation.state import Annotation, BBox
from tcip_web.app import app

from tests._clear_prediction_bucket_fixtures import build_published_bucket, earn_validated_stamp

pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")

_EARNED_TRAIT = "bud_opening"
_STEM = "img"
_BOX = (10.0, 10.0, 30.0, 30.0)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _earned_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, experiment_id: str,
    scores: tuple[float, ...] = (0.9,),
) -> dict:
    """A published, sealed bucket: one image, one prediction at ``scores[0]``, a validated stamp
    earned over its content (so every refusal case's bucket content predates the seal)."""
    dataset_root = tmp_path / "data"
    built = build_published_bucket(
        tmp_path, monkeypatch, experiment_id=experiment_id, dataset_root=dataset_root,
        stems=(_STEM,), boxes=(_BOX,), scores=scores,
    )
    bucket = Path(built["bucket"])
    stamp = earn_validated_stamp(bucket, dataset_root, trait=_EARNED_TRAIT)
    pred = read_annotations(bucket / f"{_STEM}.json")[0]
    return {
        "dataset_root": dataset_root, "bucket": bucket, "images_dir": Path(built["images_dir"]),
        "stamp": stamp, "pred": pred,
    }


def _accept_payload(built: dict, *, det_type: str, gt_idx: int | None, pred_idx: int | None,
                    rule_admitted: bool, conf: float | None = None) -> dict:
    pred = built["pred"]
    img_path = built["images_dir"] / f"{_STEM}.png"
    gt_path = built["dataset_root"] / "annotations" / "2026-03-04" / f"{_STEM}.json"
    return {
        "dataset_root": str(built["dataset_root"]), "image_name": f"{_STEM}.png",
        "image_path": str(img_path), "gt_path": str(gt_path),
        "pred_path": str(built["bucket"] / f"{_STEM}.json"),
        "det_type": det_type, "class_name": pred.subject,
        "conf": conf if conf is not None else pred.score, "iou": 1.0 if det_type == "tp" else None,
        "gt_idx": gt_idx, "pred_idx": pred_idx, "bbox": list(_BOX),
        "action": "accepted", "rule_admitted": rule_admitted,
    }


def test_generation_conf_answers_the_earned_rule(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coverage: the rule an earned, bound stamp answers through /generation_conf, read back
    against the score the accept below relies on being at or above it."""
    built = _earned_bucket(tmp_path, monkeypatch, experiment_id="exp-admit-conf")
    stamp = built["stamp"]
    validated_by = stamp["validated_by"]
    rule_conf = stamp["operating_point"]["conf"]["value"]
    assert built["pred"].score is not None and built["pred"].score >= rule_conf

    conf_resp = client.get(
        "/api/review/generation_conf", params={"pred_dir": str(built["bucket"])})
    assert conf_resp.status_code == 200, conf_resp.text
    body = conf_resp.json()
    assert body["admission_reason"] == ""
    admission = body["admission_rule"]
    assert admission == {
        "conf": pytest.approx(rule_conf), "experiment_id": validated_by["experiment_id"],
        "record_digest": validated_by["record_digest"],
    }


def test_a_verified_claim_writes_the_marker_beside_the_person(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The producer chain: a published, sealed bucket; an accept on the fp with rule_admitted
    true writes accepted_by_rule naming the stamp's own validated_by; a second accept on the
    now-tp answers reviewed and leaves the raw document's version unchanged."""
    built = _earned_bucket(tmp_path, monkeypatch, experiment_id="exp-admit-1")
    stamp = built["stamp"]
    validated_by = stamp["validated_by"]

    resp = client.post("/api/review/action", json=_accept_payload(
        built, det_type="fp", gt_idx=None, pred_idx=0, rule_admitted=True))
    assert resp.status_code == 200, resp.text
    fresh = resp.json()["matches"]
    assert fresh["detections"][0]["reviewed"] is True
    expected_identity_early = f"{validated_by['experiment_id']}:{validated_by['record_digest']}"
    assert fresh["gt"][0]["accepted_by_rule"] == expected_identity_early  # review._ann_dict

    gt_path = Path(_accept_payload(built, det_type="fp", gt_idx=None, pred_idx=0,
                                   rule_admitted=True)["gt_path"])
    raw = json.loads(gt_path.read_text(encoding="utf-8"))
    record = raw["annotations"][0]
    expected_identity = f"{validated_by['experiment_id']}:{validated_by['record_digest']}"
    assert record.get("accepted_by_rule") == expected_identity  # a guard: the identity, not a bare flag
    assert record.get("accepted_by", "").startswith("user:")
    assert "score" not in record

    import tcip_store
    from tcip_mcp.audit import audit_log_key

    last = tcip_store.read_log(audit_log_key(built["dataset_root"])).records[-1]
    assert last["arguments"]["rule_admitted"] is True
    assert last["arguments"]["accepted_by_rule"] == expected_identity

    _, version_before = read_annotations_versioned(gt_path)
    resp2 = client.post("/api/review/action", json=_accept_payload(
        built, det_type="tp", gt_idx=0, pred_idx=0, rule_admitted=True))
    assert resp2.status_code == 200, resp2.text
    assert resp2.json()["matches"]["detections"][0]["reviewed"] is True
    _, version_after = read_annotations_versioned(gt_path)
    assert version_after == version_before


def test_ordinary_accept_on_the_same_bucket_writes_none(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admits valid work: an ordinary accept (rule_admitted omitted) on a bucket that carries a
    rule writes accepted_by_rule as None, unaffected by the rule's existence."""
    built = _earned_bucket(tmp_path, monkeypatch, experiment_id="exp-admit-ordinary")
    resp = client.post("/api/review/action", json=_accept_payload(
        built, det_type="fp", gt_idx=None, pred_idx=0, rule_admitted=False))
    assert resp.status_code == 200, resp.text
    gt_path = Path(_accept_payload(built, det_type="fp", gt_idx=None, pred_idx=0,
                                   rule_admitted=False)["gt_path"])
    raw = json.loads(gt_path.read_text(encoding="utf-8"))
    assert raw["annotations"][0].get("accepted_by_rule") is None


def test_refusal_unvalidated_stamp(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A guard: an accept against an unvalidated stamp refuses (400) even when rule_admitted is
    set."""
    dataset_root = tmp_path / "data"
    built = build_published_bucket(
        tmp_path, monkeypatch, experiment_id="exp-unvalidated", dataset_root=dataset_root,
        stems=(_STEM,), boxes=(_BOX,), scores=(0.9,),
    )
    built = {"dataset_root": dataset_root, "bucket": Path(built["bucket"]),
             "images_dir": Path(built["images_dir"]),
             "pred": read_annotations(Path(built["bucket"]) / f"{_STEM}.json")[0]}
    resp = client.post("/api/review/action", json=_accept_payload(
        built, det_type="fp", gt_idx=None, pred_idx=0, rule_admitted=True))
    assert resp.status_code == 400, resp.text
    assert "validated" in resp.text or "claim" in resp.text


def _row_gone_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, experiment_id: str,
) -> tuple[dict, dict]:
    """An earned bucket whose validated_by row's own experiment has since been deleted from the
    store: the raw-write case verify_stamp_binding's own docstring accepts as a bound claim's
    experiment going missing after the fact."""
    from tcip_mcp.experiments import config_key
    import tcip_store

    built = _earned_bucket(tmp_path, monkeypatch, experiment_id=experiment_id)
    validated_by = built["stamp"]["validated_by"]
    tcip_store.delete(config_key(validated_by["experiment_id"]))
    return built, validated_by


def test_ordinary_accept_still_succeeds_when_the_pointed_experiment_is_gone(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admits valid work: an ordinary accept (no claim), a real fp accept rather than a no-op,
    answers 200 rather than a refusal on a bucket whose validated_by row's own experiment
    record is gone; the write it makes is the ordinary accept's, covered by its own tests."""
    built, _validated_by = _row_gone_bucket(tmp_path, monkeypatch, experiment_id="exp-admit-gone")
    resp = client.post("/api/review/action", json=_accept_payload(
        built, det_type="fp", gt_idx=None, pred_idx=0, rule_admitted=False))
    assert resp.status_code == 200, resp.text


def test_generation_conf_names_the_gone_experiment(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coverage: /generation_conf answers a null rule naming the experiment its pointer names."""
    built, validated_by = _row_gone_bucket(
        tmp_path, monkeypatch, experiment_id="exp-admit-gone-conf")
    conf_resp = client.get(
        "/api/review/generation_conf", params={"pred_dir": str(built["bucket"])})
    assert conf_resp.json()["admission_rule"] is None
    assert validated_by["experiment_id"] in conf_resp.json()["admission_reason"]


def test_refusal_pointer_names_a_row_no_experiment_holds(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    built, _validated_by = _row_gone_bucket(
        tmp_path, monkeypatch, experiment_id="exp-admit-gone-refuse")
    resp = client.post("/api/review/action", json=_accept_payload(
        built, det_type="fp", gt_idx=None, pred_idx=0, rule_admitted=True))
    assert resp.status_code == 400, resp.text


def test_refusal_changed_bucket_content(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = _earned_bucket(tmp_path, monkeypatch, experiment_id="exp-admit-changed")
    write_annotations(
        built["bucket"] / f"{_STEM}.json",
        [Annotation(subject=built["pred"].subject, geometry=BBox(*_BOX), score=0.95)],
        100, 100, keep_empty=True,
    )
    resp = client.post("/api/review/action", json=_accept_payload(
        built, det_type="fp", gt_idx=None, pred_idx=0, rule_admitted=True))
    assert resp.status_code == 400, resp.text
    assert "hash" in resp.text or "content" in resp.text or "added, replaced" in resp.text


def test_refusal_below_conf_prediction(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = _earned_bucket(tmp_path, monkeypatch, experiment_id="exp-admit-lowconf", scores=(0.05,))
    resp = client.post("/api/review/action", json=_accept_payload(
        built, det_type="fp", gt_idx=None, pred_idx=0, rule_admitted=True))
    assert resp.status_code == 400, resp.text
    assert "below the rule" in resp.text


def test_refusal_scoreless_prediction(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prediction document holding a record with no score is refused where it is read, naming
    the key: no reader stands in a confidence the model never reported."""
    from tests._clear_prediction_bucket_fixtures import write_image

    dataset_root = tmp_path / "data"
    built = build_published_bucket(
        tmp_path, monkeypatch, experiment_id="exp-admit-scoreless", dataset_root=dataset_root,
        stems=(_STEM,), boxes=(_BOX,), scores=(0.9,),
    )
    bucket = Path(built["bucket"])
    subject = read_annotations(bucket / f"{_STEM}.json")[0].subject
    images_dir = Path(built["images_dir"])
    write_image(images_dir / "scoreless.png")
    write_annotations(
        bucket / "scoreless.json", [Annotation(subject=subject, geometry=BBox(*_BOX))], 100, 100,
    )
    earn_validated_stamp(bucket, dataset_root, trait=_EARNED_TRAIT)
    scoreless_pred = read_annotations(bucket / "scoreless.json")[0]
    assert scoreless_pred.score is None

    payload = {
        "dataset_root": str(dataset_root), "image_name": "scoreless.png",
        "image_path": str(images_dir / "scoreless.png"),
        "gt_path": str(dataset_root / "annotations" / "2026-03-04" / "scoreless.json"),
        "pred_path": str(bucket / "scoreless.json"),
        "det_type": "fp", "class_name": subject, "conf": None, "iou": None,
        "gt_idx": None, "pred_idx": 0, "bbox": list(_BOX), "action": "accepted",
        "rule_admitted": True,
    }
    resp = client.post("/api/review/action", json=payload)
    assert resp.status_code == 400, resp.text
    assert "'score'" in resp.text


def test_refusal_classified_scope(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The precondition: an ordinary accept the tab makes today succeeds on a classified bucket;
    a claim refuses by name, since the count operating point admits detections, not values."""
    import tcip_store
    from tcip_mcp.pipelines.resolution import sidecar_key

    dataset_root = tmp_path / "data"
    bucket = dataset_root / "predictions" / "m" / "2026-03-04"
    bucket.mkdir(parents=True)
    images_dir = tmp_path / "images"
    from tests._clear_prediction_bucket_fixtures import write_image

    write_image(images_dir / f"{_STEM}.png")
    write_annotations(
        bucket / f"{_STEM}.json",
        [Annotation(subject="bud", geometry=BBox(*_BOX), score=0.9,
                    attributes={"state": "open"})],
        100, 100,
    )
    tcip_store.replace(sidecar_key(bucket, "operating_point"), {
        "checkpoint_sha256": "sha-classified", "experiment_id": None, "validated": True,
        "validated_by": {"experiment_id": "exp-classified", "record_digest": "0" * 16},
        "id_map": {"open": 0, "closed": 1}, "subject": "bud", "attribute": "state",
        "operating_point": {"tiled": {"value": False}, "conf": {"value": 0.1}},
    }, expect=tcip_store.Version.ABSENT)

    payload = {
        "dataset_root": str(dataset_root), "image_name": f"{_STEM}.png",
        "image_path": str(images_dir / f"{_STEM}.png"),
        "gt_path": str(dataset_root / "annotations" / "2026-03-04" / f"{_STEM}.json"),
        "pred_path": str(bucket / f"{_STEM}.json"),
        "det_type": "fp", "class_name": "open", "conf": 0.9, "iou": None,
        "gt_idx": None, "pred_idx": 0, "bbox": list(_BOX), "action": "accepted",
        "subject": "bud", "attribute": "state",
    }
    ordinary = client.post("/api/review/action", json=payload)
    assert ordinary.status_code == 200, ordinary.text

    claimed = client.post("/api/review/action", json={**payload, "rule_admitted": True})
    assert claimed.status_code == 400, claimed.text
    assert "classified" in claimed.text


def test_refusal_stale_detection(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ground-truth record written under the fp's box after the tab's matches were taken makes
    the submitted fp a tp in a fresh recompute; the claim refuses with 409, not a duplicate write."""
    built = _earned_bucket(tmp_path, monkeypatch, experiment_id="exp-admit-stale")
    gt_path = built["dataset_root"] / "annotations" / "2026-03-04" / f"{_STEM}.json"
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    write_annotations(
        gt_path, [Annotation(subject=built["pred"].subject, geometry=BBox(*_BOX),
                             created_by="user:someone_else")],
        100, 100,
    )
    resp = client.post("/api/review/action", json=_accept_payload(
        built, det_type="fp", gt_idx=None, pred_idx=0, rule_admitted=True))
    assert resp.status_code == 409, resp.text


def test_refusal_wrong_action(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A guard: rule_admitted on a reject action refuses (400) naming "rejected", rather than the
    no-op a plain reject on an fp always is."""
    built = _earned_bucket(tmp_path, monkeypatch, experiment_id="exp-admit-wrong-action")
    payload = _accept_payload(built, det_type="fp", gt_idx=None, pred_idx=0, rule_admitted=True)
    payload["action"] = "rejected"
    resp = client.post("/api/review/action", json=payload)
    assert resp.status_code == 400, resp.text
    assert "rejected" in resp.text


def test_refusal_wrong_det_type(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A guard: rule_admitted on an fn accept refuses (400) naming "fn", rather than the no-op an
    accept on an fn otherwise is."""
    built = _earned_bucket(tmp_path, monkeypatch, experiment_id="exp-admit-wrong-dettype")
    payload = _accept_payload(built, det_type="fn", gt_idx=None, pred_idx=0, rule_admitted=True)
    resp = client.post("/api/review/action", json=payload)
    assert resp.status_code == 400, resp.text
    assert "fn" in resp.text


def test_refusal_out_of_range_pred_idx(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A guard: rule_admitted with an out-of-range pred_idx refuses (400) naming "pred_idx",
    rather than the silent no-op it otherwise is."""
    built = _earned_bucket(tmp_path, monkeypatch, experiment_id="exp-admit-oor-pred-idx")
    payload = _accept_payload(built, det_type="fp", gt_idx=None, pred_idx=5, rule_admitted=True)
    resp = client.post("/api/review/action", json=payload)
    assert resp.status_code == 400, resp.text
    assert "pred_idx" in resp.text


def test_parsed_marker_resolves_through_find_validation(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coverage: parse_validation_reference has no production caller today; this proves the
    marker a rule-admitted accept writes parses back to the experiment and digest find_validation
    resolves its own row from."""
    from tcip_mcp.experiments import find_validation
    from tcip_mcp.pipelines.resolution import parse_validation_reference

    built = _earned_bucket(tmp_path, monkeypatch, experiment_id="exp-admit-parse")
    resp = client.post("/api/review/action", json=_accept_payload(
        built, det_type="fp", gt_idx=None, pred_idx=0, rule_admitted=True))
    assert resp.status_code == 200, resp.text
    gt_path = Path(_accept_payload(built, det_type="fp", gt_idx=None, pred_idx=0,
                                   rule_admitted=True)["gt_path"])
    raw = json.loads(gt_path.read_text(encoding="utf-8"))
    marker = raw["annotations"][0]["accepted_by_rule"]

    parsed = parse_validation_reference(marker)
    assert parsed is not None
    experiment_id, record_digest = parsed
    row = find_validation(experiment_id, record_digest)
    assert row is not None


def test_the_rail_admits_the_directory_a_rule_admitted_accept_produced(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rail's admits-valid-work case, fed the directory the button's accept actually
    produced rather than a hand-written record: a signed, rule-admitted record is admitted."""
    from tcip_annotation.json_io import require_reference_ground_truth

    built = _earned_bucket(tmp_path, monkeypatch, experiment_id="exp-admit-rail")
    resp = client.post("/api/review/action", json=_accept_payload(
        built, det_type="fp", gt_idx=None, pred_idx=0, rule_admitted=True))
    assert resp.status_code == 200, resp.text
    gt_path = Path(_accept_payload(built, det_type="fp", gt_idx=None, pred_idx=0,
                                   rule_admitted=True)["gt_path"])
    require_reference_ground_truth(gt_path.parent)
