"""Review verdicts are scoped to the prediction bucket, and the dataset, they were recorded against.

A camera that restarts its numbering gives two capture dates the same filename. Verdicts keyed by
image name alone put both dates' reviews in one place: the reference for one date is fed the
other's adjudications, and the immutability guard freezes a bucket nobody reviewed. These drive
the real staging path and the real review route, so what freezes a bucket is what a reviewer
actually recorded against it.

The same holds across roots: the review routes take the dataset root, so a breeder working out of a
project directory that is not the dataset directory still writes verdicts into the store the
inference-side guard and the promotion both read, and a promotion is earned over the prediction
documents that were actually reviewed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import tcip_store
from tcip_annotation import Annotation, BBox
from tcip_mcp.dataset_layout import prediction_dir
from tcip_mcp.pipelines.resolution import sidecar_key
from tcip_mcp.prediction_buckets import stage_prediction_shapes
from tcip_web.app import app

DATE_A = "2026-02-11"
DATE_B = "2026-03-04"
STEM = "IMG_0007"
IMG_W, IMG_H = 640, 480
BOX = (12.0, 20.0, 52.0, 44.0)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _stage(dataset_root: Path, model: str, date: str, stem: str = STEM) -> dict:
    """Stage one prediction into ``model``'s bucket for ``date``."""
    return stage_prediction_shapes(
        str(dataset_root), model, date, stem,
        annotations=[Annotation(subject="bud", geometry=BBox(*BOX), score=0.83)],
        img_w=IMG_W, img_h=IMG_H,
    )


def _image(dataset_root: Path, stem: str = STEM) -> Path:
    """One readable source image the review route can size its context from."""
    img_dir = dataset_root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    path = img_dir / f"{stem}.jpg"
    Image.new("RGB", (IMG_W, IMG_H), color=(80, 120, 90)).save(path)
    return path


def _review(client: TestClient, dataset_root: Path, img: Path, pred_path: str, gt_path: Path,
            stem: str = STEM):
    """Record one accepted verdict through the route a breeder's canvas uses."""
    return client.post("/api/review/action", json={
        "dataset_root": str(dataset_root),
        "image_name": f"{stem}.jpg",
        "image_path": str(img),
        "gt_path": str(gt_path),
        "pred_path": pred_path,
        "det_type": "fp", "class_name": "bud",
        "conf": 0.83, "iou": None,
        "gt_idx": None, "pred_idx": 0,
        "bbox": list(BOX),
        "action": "accepted",
    })


def _sidecar(bucket: Path) -> None:
    """The producing run's own stamp on a staged bucket.

    A verdict resolves its producer identity from this, and with it the content identity of the
    prediction document the reviewer is looking at, so the promotion has something recorded to
    compare the file on disk against.
    """
    tcip_store.replace(sidecar_key(bucket, "operating_point"), {
        "checkpoint_sha256": "sha-detector", "experiment_id": None, "validated": False,
        "id_map": {"bud": 0}, "subject": "bud", "attribute": None,
        "operating_point": {"tiled": {"value": False}, "conf": {"value": 0.25}},
    }, expect=tcip_store.Version.ABSENT)


def _project_root(tmp_path: Path) -> Path:
    """A project directory that is genuinely not the dataset directory."""
    root = tmp_path / "workspace" / "proj"
    root.mkdir(parents=True)
    return root


def test_same_basename_on_two_dates_keeps_separate_verdicts(
    client: TestClient, tmp_path: Path
) -> None:
    """Two dates of one camera filename, one dataset, one model: a verdict on one date's bucket
    freezes that bucket and leaves the other date's alone."""
    dataset_root = tmp_path / "data"
    img = _image(dataset_root)
    staged_a = _stage(dataset_root, "detector", DATE_A)
    staged_b = _stage(dataset_root, "detector", DATE_B)
    assert staged_a["redirected"] is False and staged_b["redirected"] is False

    resp = _review(client, dataset_root, img, staged_a["path"], tmp_path / "gt.json")
    assert resp.status_code == 200

    # The reviewed date's bucket is frozen and redirects.
    again_a = _stage(dataset_root, "detector", DATE_A)
    assert again_a["redirected"] is True
    assert again_a["bucket"] == "detector@r2"
    assert again_a["verdict_count"] == 1

    # The other date shares the filename and nothing else: it is written where it was asked for.
    again_b = _stage(dataset_root, "detector", DATE_B)
    assert again_b["redirected"] is False
    assert again_b["bucket"] == "detector"
    assert again_b["verdict_count"] == 0
    assert again_b["path"] == str(Path(prediction_dir(dataset_root, "detector", DATE_B))
                                 / f"{STEM}.json")


def test_a_multi_bucket_datasets_verdicts_enumerate_under_the_bucket_they_were_recorded_on(
    client: TestClient, tmp_path: Path
) -> None:
    """The admit case: a dataset holding several reviewed and unreviewed buckets reads back one
    set of verdicts per bucket, and the unreviewed bucket keeps a count of zero."""
    from tcip_annotation.review_engine import ReviewEngine
    from tcip_mcp.prediction_buckets import bucket_key_of, review_state_dir_of, verdict_count

    dataset_root = tmp_path / "data"
    img = _image(dataset_root)
    staged_a = _stage(dataset_root, "detector", DATE_A)
    staged_b = _stage(dataset_root, "detector", DATE_B)

    assert _review(client, dataset_root, img, staged_a["path"], tmp_path / "gt.json").status_code == 200

    key_a = bucket_key_of(prediction_dir(dataset_root, "detector", DATE_A))
    key_b = bucket_key_of(prediction_dir(dataset_root, "detector", DATE_B))
    state_dir = review_state_dir_of(dataset_root)
    engine = ReviewEngine(state_dir)

    assert engine.reviewed_buckets() == [key_a]
    assert list(engine.image_states(key_a)) == [f"{STEM}.jpg"]
    assert engine.image_states(key_b) == {}
    assert verdict_count(state_dir, key_a, [STEM]) == 1
    assert verdict_count(state_dir, key_b, [STEM]) == 0
    # The reviewed image reads as reviewed under the bucket it was reviewed on, and only there.
    assert engine.get_image_review_status(key_a, f"{STEM}.jpg") != "not_started"
    assert engine.get_image_review_status(key_b, f"{STEM}.jpg") == "not_started"
    # Both buckets' prediction files are still on disk, neither overwritten by the other's review.
    assert json.loads(Path(staged_b["path"]).read_text(encoding="utf-8"))["annotations"]


# ── The dataset root, when it is not the project root ───────────────────────


def test_gui_verdict_is_visible_to_the_bucket_guard(client: TestClient, tmp_path: Path) -> None:
    """A verdict recorded through the browser lands in the store the inference-side immutability
    guard counts, with the breeder working out of a project directory that is not the dataset. The
    two roots being one directory is what let this look true without being tested."""
    project_root = _project_root(tmp_path)
    dataset_root = tmp_path / "data"
    img = _image(dataset_root)
    staged = _stage(dataset_root, "detector", DATE_A)

    assert _review(client, dataset_root, img, staged["path"],
                   dataset_root / "annotations" / DATE_A / f"{STEM}.json").status_code == 200

    again = _stage(dataset_root, "detector", DATE_A)
    assert again["redirected"] is True
    assert again["verdict_count"] == 1
    assert again["bucket"] == "detector@r2"
    # The verdict lives in the dataset's own store; the project directory holds no review state.
    from tcip_annotation.review_engine import REVIEW_VERDICTS_STORE

    assert tcip_store.keys(REVIEW_VERDICTS_STORE, str(dataset_root / ".tcip" / "state"))
    assert not (project_root / ".tcip").exists()


@pytest.mark.usefixtures("seed_bud_trait_spec")
def test_promotion_refuses_a_changed_prediction_file(client: TestClient, tmp_path: Path) -> None:
    """A promotion is earned over the prediction documents the reviewer actually saw. Replacing one,
    removing one, or adding one for an image that had none all break that, each leaving a reference
    built partly from evidence nobody reviewed, so each is refused rather than promoted."""
    dataset_root = tmp_path / "data"
    other = "IMG_0008"
    img = _image(dataset_root)
    img_other = _image(dataset_root, other)
    staged = _stage(dataset_root, "detector", DATE_A)
    staged_other = _stage(dataset_root, "detector", DATE_A, other)
    bucket = Path(prediction_dir(dataset_root, "detector", DATE_A))
    _sidecar(bucket)
    gt_dir = dataset_root / "annotations" / DATE_A
    assert _review(client, dataset_root, img, staged["path"],
                   gt_dir / f"{STEM}.json").status_code == 200
    assert _review(client, dataset_root, img_other, staged_other["path"],
                   gt_dir / f"{other}.json", stem=other).status_code == 200

    def _promote() -> dict:
        resp = client.post("/api/review/validate_reference", json={
            "dataset_root": str(dataset_root), "trait": "bud_opening", "pred_dir": str(bucket),
            "subject": "bud"})
        assert resp.status_code == 200, resp.text
        return resp.json()

    original = Path(staged["path"]).read_bytes()
    Path(staged["path"]).write_text(
        json.dumps({"img_width": IMG_W, "img_height": IMG_H, "annotations": []}), encoding="utf-8")
    replaced = _promote()
    assert replaced["validated"] is False
    assert f"{STEM}.jpg" in replaced["reason"]
    assert replaced["buckets_stamped"] == []
    assert tcip_store.read(sidecar_key(bucket, "operating_point"))["validated"] is False

    Path(staged["path"]).write_bytes(original)
    original_other = Path(staged_other["path"]).read_bytes()
    Path(staged_other["path"]).unlink()
    removed = _promote()
    assert removed["validated"] is False
    assert f"{other}.jpg" in removed["reason"]

    # Written back by hand: the staging path would redirect around the reviewed bucket under test.
    Path(staged_other["path"]).write_bytes(original_other)
    third = "IMG_0009"
    _image(dataset_root, third)
    assert client.post("/api/review/mark_complete", json={
        "dataset_root": str(dataset_root), "image_name": f"{third}.jpg",
        "pred_dir": str(bucket)}).status_code == 200
    (bucket / f"{third}.json").write_bytes(original)  # a document appears for a reviewed negative
    appeared = _promote()
    assert appeared["validated"] is False
    assert f"{third}.jpg" in appeared["reason"]


def test_promotion_refuses_a_bucket_from_another_dataset_root(
    client: TestClient, tmp_path: Path
) -> None:
    """The record, its covered buckets and its reference all hang off one dataset root, so a bucket
    whose own root positively disagrees with the one stated is refused naming both, rather than
    validated against a dataset that holds none of its predictions."""
    here = tmp_path / "data"
    elsewhere = tmp_path / "other_data"
    (here / "images").mkdir(parents=True)
    _image(elsewhere)
    _stage(elsewhere, "detector", DATE_A)
    bucket = Path(prediction_dir(elsewhere, "detector", DATE_A))
    _sidecar(bucket)

    resp = client.post("/api/review/validate_reference", json={
        "dataset_root": str(here), "trait": "bud_opening", "pred_dir": str(bucket),
        "subject": "bud"})
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert str(here) in detail and str(elsewhere) in detail
    assert tcip_store.read(sidecar_key(bucket, "operating_point"))["validated"] is False


@pytest.mark.usefixtures("seed_bud_trait_spec")
def test_every_review_surface_reads_the_dataset_root_the_request_states(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One fixture, a project root and a dataset root that are different directories, and every
    review surface driven across it: matches, a verdict, mark-complete, the batch image-status
    query, the priority-queue launch, the promotion and the inference-side bucket guard. One route
    standing in for the rest is what let the two roots look interchangeable."""
    import time

    pytest.importorskip("torch")  # the promotion's gate resolves through the evaluation stack
    project_root = _project_root(tmp_path)
    dataset_root = tmp_path / "data"
    img = _image(dataset_root)
    staged = _stage(dataset_root, "detector", DATE_A)
    bucket = Path(prediction_dir(dataset_root, "detector", DATE_A))
    _sidecar(bucket)
    gt = dataset_root / "annotations" / DATE_A / f"{STEM}.json"

    matches = client.post("/api/review/matches", json={
        "dataset_root": str(dataset_root), "image_name": f"{STEM}.jpg", "image_path": str(img),
        "pred_path": staged["path"], "iou_threshold": 0.3, "conf_threshold": 0.1})
    assert matches.status_code == 200, matches.text
    assert matches.json()["image_status"] == "not_started"

    assert _review(client, dataset_root, img, staged["path"], gt).status_code == 200

    done = client.post("/api/review/mark_complete", json={
        "dataset_root": str(dataset_root), "image_name": f"{STEM}.jpg",
        "gt_path": str(gt), "pred_dir": str(bucket)})
    assert done.status_code == 200
    assert done.json()["image_status"] == "completed"

    batch = client.get("/api/review/image_statuses", params={
        "dataset_root": str(dataset_root), "pred_dir": str(bucket)})
    assert batch.json()["statuses"][f"{STEM}.jpg"] == "completed"

    calls: list[dict] = []

    def _fake_queue(**kwargs):
        calls.append(kwargs)
        return {"queue": [], "total_candidates": 0, "reviewed_skipped": 1}

    import tcip_mcp.tools.feedback_tools as feedback_tools_mod
    monkeypatch.setattr(feedback_tools_mod, "prioritize_review_queue", _fake_queue)
    ckpt = dataset_root / "models" / "best.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"not a real checkpoint")
    launch = client.post("/api/review/queue/launch", json={
        "dataset_root": str(dataset_root), "checkpoint_path": str(ckpt),
        "images_dir": str(img.parent)})
    assert launch.status_code == 200, launch.text
    job_id = launch.json()["job_id"]
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and client.get(
            f"/api/review/queue/{job_id}").json()["status"] not in ("completed", "failed"):
        time.sleep(0.02)
    assert calls and calls[0]["dataset_root"] == str(dataset_root)

    promo = client.post("/api/review/validate_reference", json={
        "dataset_root": str(dataset_root), "trait": "bud_opening", "pred_dir": str(bucket),
        "subject": "bud"})
    assert promo.status_code == 200, promo.text
    assert promo.json()["reviewed_image_count"] == 1

    again = _stage(dataset_root, "detector", DATE_A)
    assert again["redirected"] is True

    # Every record these routes changed travels with the dataset; the project directory is untouched.
    from tcip_mcp.audit import audit_log_key

    assert tcip_store.read_log(audit_log_key(dataset_root)).records
    assert not (project_root / ".tcip").exists()
