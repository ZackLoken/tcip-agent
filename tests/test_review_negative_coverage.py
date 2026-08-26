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
    return TestClient(app, base_url="http://127.0.0.1")


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
    _seed_sidecar(d, {"checkpoint_sha256": CHECKPOINT_SHA, "experiment_id": "exp-17"})
    return d


def _seed_sidecar(pred_dir: Path, sidecar: dict) -> None:
    """The bucket's ``operating_point.json`` stamp, through the seam the route reads it from."""
    import tcip_store
    from tcip_mcp.pipelines.resolution import sidecar_key

    tcip_store.replace(sidecar_key(pred_dir, "operating_point"), sidecar,
                       expect=tcip_store.Version.ABSENT)


def _dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / ".tcip" / "state").mkdir(parents=True)
    return root


def _shard(dataset_root: Path, image_name: str) -> dict:
    """The one shard written for ``image_name``, through the seam wherever its bucket put it."""
    import tcip_store
    from tcip_annotation.review_engine import REVIEW_VERDICTS_STORE

    state_dir = str(dataset_root / ".tcip" / "state")
    found = [k for k in tcip_store.keys(REVIEW_VERDICTS_STORE, state_dir) if k.parts[1] == image_name]
    assert len(found) == 1, found
    return tcip_store.read(found[0])["state"]


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
    assert state["adjudication_covered"] == {"*": False}
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
    assert _shard(dataset_root, "IMG_0031.JPG")["adjudication_covered"] == {"*": True}


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
    assert _shard(dataset_root, "IMG_0099.JPG")["adjudication_covered"] == {"*": True}


def test_complete_named_subject_on_an_image_with_only_another_subjects_gt_is_a_scoped_negative(
    client: TestClient, tmp_path: Path
) -> None:
    """A Complete naming 'catkin' on an image whose GT holds only 'leaf' objects is a genuine
    negative of catkin, not the 'complete' a whole-file (subject-blind) check would produce."""
    dataset_root = _dataset_root(tmp_path)
    gt_dir = dataset_root / "annotations" / "2-11-26"
    gt_dir.mkdir(parents=True)
    gt_path = gt_dir / "IMG_0050.json"
    write_annotations(
        str(gt_path), [Annotation(subject="leaf", geometry=BBox(5.0, 5.0, 40.0, 40.0))],
        IMG_W, IMG_H)

    resp = client.post("/api/review/mark_complete", json={
        "dataset_root": str(dataset_root),
        "image_name": "IMG_0050.JPG",
        "gt_path": str(gt_path),
        "subject": "catkin",
    })
    assert resp.status_code == 200
    assert resp.json()["annotation_status"] == "negative"


def test_complete_named_subject_over_a_file_holding_only_another_subjects_predictions_is_covered(
    client: TestClient, tmp_path: Path
) -> None:
    """A subject's own predictions being absent is a genuine negative for it, even when the same
    file holds another subject's boxes."""
    d = tmp_path / "predictions" / "baseline" / "2-11-26"
    d.mkdir(parents=True)
    write_annotations(
        str(d / "IMG_0060.json"),
        [Annotation(subject="leaf", geometry=BBox(12.0, 20.0, 52.0, 44.0), score=0.71)],
        IMG_W, IMG_H,
    )
    _seed_sidecar(d, {"checkpoint_sha256": CHECKPOINT_SHA, "id_map": {"catkin": 0, "leaf": 1}})
    dataset_root = _dataset_root(tmp_path)

    resp = client.post("/api/review/mark_complete", json={
        "dataset_root": str(dataset_root),
        "image_name": "IMG_0060.JPG",
        "pred_dir": str(d),
        "subject": "catkin",
    })
    assert resp.status_code == 200
    assert _shard(dataset_root, "IMG_0060.JPG")["adjudication_covered"] == {"catkin": True}


def test_complete_named_subject_the_bucket_never_assessed_omits_the_coverage_entry(
    client: TestClient, tmp_path: Path
) -> None:
    """A subject not among the bucket's own recorded class map cannot be judged negative or
    positive: the coverage entry is omitted, and the Complete and its status write still
    proceed."""
    bucket = _bucket(tmp_path)  # a sidecar with no id_map at all
    dataset_root = _dataset_root(tmp_path)

    resp = client.post("/api/review/mark_complete", json={
        "dataset_root": str(dataset_root),
        "image_name": "IMG_0007.JPG",
        "pred_dir": str(bucket),
        "subject": "catkin",
    })
    assert resp.status_code == 200
    assert resp.json()["image_status"] == "completed"
    state = _shard(dataset_root, "IMG_0007.JPG")
    assert "catkin" not in (state.get("adjudication_covered") or {})


def test_a_second_complete_under_another_subject_leaves_the_firsts_claim_intact(
    client: TestClient, tmp_path: Path
) -> None:
    """A Complete confirming one subject and a later Complete confirming another, on the same
    image, both land: the second must not overwrite the first's coverage claim."""
    d = tmp_path / "predictions" / "baseline" / "2-11-26"
    d.mkdir(parents=True)
    write_annotations(str(d / "IMG_0070.json"), [], IMG_W, IMG_H, keep_empty=True)
    _seed_sidecar(d, {"checkpoint_sha256": CHECKPOINT_SHA, "id_map": {"catkin": 0, "leaf": 1}})
    dataset_root = _dataset_root(tmp_path)

    client.post("/api/review/mark_complete", json={
        "dataset_root": str(dataset_root), "image_name": "IMG_0070.JPG",
        "pred_dir": str(d), "subject": "catkin",
    })
    resp = client.post("/api/review/mark_complete", json={
        "dataset_root": str(dataset_root), "image_name": "IMG_0070.JPG",
        "pred_dir": str(d), "subject": "leaf",
    })
    assert resp.status_code == 200
    assert _shard(dataset_root, "IMG_0070.JPG")["adjudication_covered"] == {
        "catkin": True, "leaf": True}


def test_complete_with_no_subject_records_completion_with_a_null_status(
    client: TestClient, tmp_path: Path
) -> None:
    """A Complete naming no subject still records the review completion, but derives and returns
    no subject-scoped status: there is nothing to scope it to."""
    dataset_root = _dataset_root(tmp_path)

    resp = client.post("/api/review/mark_complete", json={
        "dataset_root": str(dataset_root),
        "image_name": "IMG_0080.JPG",
    })
    assert resp.status_code == 200
    assert resp.json()["annotation_status"] is None
    assert _shard(dataset_root, "IMG_0080.JPG")["adjudication_covered"] == {"*": True}
