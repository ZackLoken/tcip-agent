"""Review verdicts are scoped to the prediction bucket, and the dataset, they were recorded against.

A camera that restarts its numbering gives two capture dates the same filename. Verdicts keyed by
image name alone put both dates' reviews in one place: the reference for one date is fed the
other's adjudications. These drive the real staging door and the real review route, so what a
bucket holds is what a reviewer actually recorded against it.

The same holds across roots: the review routes take the dataset root, so a breeder working out of a
project directory that is not the dataset directory still writes verdicts into the dataset's own
store.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import tcip_store
from tcip_mcp.buckets import bucket_key_of
from tcip_web.app import app

DATE_A = "2026-02-11"
DATE_B = "2026-03-04"
STEM = "IMG_0007"
IMG_W, IMG_H = 512, 512
BOX = (16.0, 32.0, 48.0, 64.0)


@pytest.fixture
def client(opened_project) -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _image(dataset_root: Path, date: str, stem: str = STEM) -> Path:
    """One readable source image under ``date``, the review route can size its context from."""
    img_dir = dataset_root / "images" / date
    img_dir.mkdir(parents=True, exist_ok=True)
    path = img_dir / f"{stem}.jpg"
    Image.new("RGB", (IMG_W, IMG_H), color=(80, 120, 90)).save(path)
    return path


def _stage(project: Path, image: Path, model: str = "detector") -> dict:
    """One box over ``BOX`` staged for ``image`` under ``model`` through ``stage_proposals``."""
    from tcip_mcp.tools.proposal_tools import stage_proposals

    x1, y1, x2, y2 = BOX
    return stage_proposals(project, str(image), model_name=model, boxes=[{
        "subject": "bud", "conf": 0.83, "cx": (x1 + x2) / 2 / IMG_W, "cy": (y1 + y2) / 2 / IMG_H,
        "w": (x2 - x1) / IMG_W, "h": (y2 - y1) / IMG_H}])


def _review(client: TestClient, dataset_root: Path, img: Path, pred_path: str, gt_path: Path):
    """Record one accepted verdict through the route a breeder's canvas uses."""
    return client.post("/api/review/action", json={
        "dataset_root": str(dataset_root), "image_name": img.name, "image_path": str(img),
        "gt_path": str(gt_path), "pred_path": pred_path,
        "det_type": "fp", "class_name": "bud", "conf": 0.83, "iou": None,
        "gt_idx": None, "pred_idx": 0, "bbox": list(BOX), "action": "accepted",
    })


def _engine(dataset_root: Path):
    from tcip_annotation.review_engine import ReviewEngine
    from tcip_mcp.project_paths import project_state_dir

    return ReviewEngine(project_state_dir(dataset_root))


def test_same_basename_on_two_dates_keeps_separate_verdicts(
    client: TestClient, tmp_path: Path
) -> None:
    """Two dates of one camera filename, one dataset, one model: a verdict on one date's staged
    bucket lands under that bucket alone, and restaging the reviewed image refuses rather than
    overwrite what was reviewed, while the other date's document stands untouched."""
    dataset_root = tmp_path / "data"
    img_a = _image(dataset_root, DATE_A)
    img_b = _image(dataset_root, DATE_B)
    staged_a = _stage(tmp_path, img_a)
    staged_b = _stage(tmp_path, img_b)
    assert "error" not in staged_a and "error" not in staged_b

    resp = _review(client, dataset_root, img_a, staged_a["path"],
                   dataset_root / "annotations" / DATE_A / f"{STEM}.json")
    assert resp.status_code == 200, resp.text

    key_a = bucket_key_of(Path(staged_a["path"]).parent)
    key_b = bucket_key_of(Path(staged_b["path"]).parent)
    assert key_a != key_b
    engine = _engine(dataset_root)
    assert engine.reviewed_buckets() == [key_a]
    assert list(engine.image_states(key_a)) == [f"{STEM}.jpg"]
    assert engine.image_states(key_b) == {}
    assert engine.get_image_review_status(key_a, f"{STEM}.jpg") != "not_started"
    assert engine.get_image_review_status(key_b, f"{STEM}.jpg") == "not_started"

    again = _stage(tmp_path, img_a)
    assert "already exists" in again["error"]
    assert json.loads(Path(staged_b["path"]).read_text(encoding="utf-8"))["annotations"]


def test_a_gui_verdict_lands_in_the_datasets_own_store(client: TestClient, tmp_path: Path) -> None:
    """A verdict recorded through the browser lands in the dataset's own store, with the breeder
    working out of a project directory that is not the dataset."""
    from tcip_annotation.review_engine import REVIEW_VERDICTS_STORE

    project_root = tmp_path / "workspace" / "proj"
    project_root.mkdir(parents=True)
    dataset_root = tmp_path / "data"
    img = _image(dataset_root, DATE_A)
    staged = _stage(tmp_path, img)

    assert _review(client, dataset_root, img, staged["path"],
                   dataset_root / "annotations" / DATE_A / f"{STEM}.json").status_code == 200

    assert tcip_store.keys(REVIEW_VERDICTS_STORE, str(dataset_root / ".tcip" / "state"))
    assert not (project_root / ".tcip").exists()


def test_every_review_surface_reads_the_dataset_root_the_request_states(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One fixture, a project root and a dataset root that are different directories, and every
    review surface driven across it: matches, a verdict, mark-complete, the batch image-status
    query and the priority-queue launch. One route standing in for the rest is what let the two
    roots look interchangeable."""
    import time

    project_root = tmp_path / "workspace" / "proj"
    project_root.mkdir(parents=True)
    dataset_root = tmp_path / "data"
    img = _image(dataset_root, DATE_A)
    staged = _stage(tmp_path, img)
    bucket = Path(staged["path"]).parent
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
    assert done.status_code == 200, done.text
    assert done.json()["image_status"] == "completed"

    batch = client.get("/api/review/image_statuses", params={
        "dataset_root": str(dataset_root), "pred_dir": str(bucket)})
    assert batch.json()["statuses"][f"{STEM}.jpg"] == "completed"

    calls: list[dict] = []

    def _fake_queue(project, **kwargs):
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

    from tcip_mcp.audit import audit_log_key

    assert tcip_store.read_log(audit_log_key(dataset_root)).records
    assert not (project_root / ".tcip").exists()
