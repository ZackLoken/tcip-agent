"""What a zero-verdict Complete records about false-negative adjudication.

Marking an image Reviewed with no individual verdicts means one of two very different things.
If the model predicted nothing on that image, Complete is itself the confirming act and the image
is a genuine negative whose misses have been adjudicated. If the model did predict on it, the
breeder bulk-accepted without walking the detections, so nothing was adjudicated and the image
must not be counted as covered when a review is promoted into a validation reference.

The per-image prediction file is addressed by the image's stem, so ``IMG_0007.JPG`` is answered by
``IMG_0007.json`` in the bucket.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_annotation.json_io import write_annotations
from tcip_annotation.state import Annotation, BBox
from tcip_web.app import app

IMG_W, IMG_H = 160, 100
CHECKPOINT_SHA = "3f9c1ab27e"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _bucket(tmp_path: Path) -> Path:
    """A prediction bucket: one image predicted on, one left empty, one never written."""
    d = tmp_path / "predictions" / "baseline" / "2-11-26"
    d.mkdir(parents=True)
    write_annotations(
        str(d / "IMG_0007.json"),
        [Annotation(subject="catkin", geometry=BBox(12.0, 20.0, 52.0, 44.0), score=0.71)],
        IMG_W, IMG_H,
    )
    write_annotations(str(d / "IMG_0031.json"), [], IMG_W, IMG_H, keep_empty=True)
    (d / "operating_point.json").write_text(
        json.dumps({"checkpoint_sha256": CHECKPOINT_SHA, "experiment_id": "exp-17"}),
        encoding="utf-8")
    return d


def _dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / ".tcip" / "state").mkdir(parents=True)
    return root


def _shard(dataset_root: Path, image_name: str) -> dict:
    """The one shard written for ``image_name``, wherever its bucket directory put it."""
    found = sorted((dataset_root / ".tcip" / "state" / "review").rglob(f"{image_name}.json"))
    assert len(found) == 1, found
    return json.loads(found[0].read_text(encoding="utf-8"))["state"]


def test_bulk_complete_on_a_predicted_image_is_not_adjudication_covered(
    client: TestClient, tmp_path: Path
) -> None:
    """The bucket holds a detection for this image and no verdict was recorded on it, so the
    completion is a bulk accept: reviewed, but with its misses unadjudicated."""
    bucket = _bucket(tmp_path)
    dataset_root = _dataset_root(tmp_path)

    resp = client.post("/api/review/mark_complete", json={
        "dataset_root": str(dataset_root),
        "image_name": "IMG_0007.JPG",
        "pred_dir": str(bucket),
    })
    assert resp.status_code == 200
    assert resp.json()["image_status"] == "completed"

    state = _shard(dataset_root, "IMG_0007.JPG")
    assert state["adjudication_covered"] is False
    assert state["producer_identity"]["checkpoint_sha256"] == CHECKPOINT_SHA


def test_complete_on_an_image_the_bucket_left_empty_is_a_covered_negative(
    client: TestClient, tmp_path: Path
) -> None:
    """The bucket ran on this image and found nothing, so there was never a detection to walk and
    Complete confirms the negative outright."""
    bucket = _bucket(tmp_path)
    dataset_root = _dataset_root(tmp_path)

    resp = client.post("/api/review/mark_complete", json={
        "dataset_root": str(dataset_root),
        "image_name": "IMG_0031.JPG",
        "pred_dir": str(bucket),
    })
    assert resp.status_code == 200
    assert _shard(dataset_root, "IMG_0031.JPG")["adjudication_covered"] is True


def test_complete_on_an_image_the_bucket_never_wrote_is_a_covered_negative(
    client: TestClient, tmp_path: Path
) -> None:
    """No prediction file for this stem at all reads the same way: nothing to check."""
    bucket = _bucket(tmp_path)
    dataset_root = _dataset_root(tmp_path)

    resp = client.post("/api/review/mark_complete", json={
        "dataset_root": str(dataset_root),
        "image_name": "IMG_0099.JPG",
        "pred_dir": str(bucket),
    })
    assert resp.status_code == 200
    assert _shard(dataset_root, "IMG_0099.JPG")["adjudication_covered"] is True
