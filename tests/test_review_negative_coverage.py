"""What a zero-verdict Complete records about false-negative adjudication.

Marking an image Reviewed with no individual verdicts means one of two very different things.
If the model predicted nothing on that image, Complete is itself the confirming act and the image
is a genuine negative whose misses have been adjudicated. If the model did predict on it, the
breeder bulk-accepted without walking the detections, so nothing was adjudicated and the image
must not be counted as covered.

The per-image prediction file is addressed by the image's stem, so ``IMG_0007.JPG`` is answered by
``IMG_0007.json`` in the bucket.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_annotation.json_io import write_annotations
from tcip_annotation.state import Annotation, BBox
from tcip_web.app import app

from tests._web_fixtures import open_new_project

OPENING = {"subject": "bud", "attribute": "opening", "id_map": {"open": 0, "closed": 1}}
BUD = {"subject": "bud", "attribute": None, "id_map": {"bud": 0}}
LEAF = {"subject": "leaf", "attribute": None, "id_map": {"leaf": 0}}
IMG_W, IMG_H = 160, 100
BOX = [12.0, 20.0, 52.0, 44.0]


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _published(project: Path, scope: dict, detections: dict[str, int]) -> Path:
    """The bucket ``predictions/baseline/2-11-26`` published under ``scope``, holding
    ``detections[stem]`` boxes of its first class on each stem's document, with ``project``
    opened in the web backend."""
    pytest.importorskip("torch")
    from tests._chain_fixtures import published

    bucket = project / "predictions" / "baseline" / "2-11-26"
    published(project, bucket, [
        {"image": f"{stem}.JPG", "width": IMG_W, "height": IMG_H, "boxes": [BOX] * n,
         "scores": [0.71] * n, "labels": [1] * n} for stem, n in detections.items()],
        scope=scope)
    open_new_project(project)
    return bucket


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


def _complete(client: TestClient, dataset_root: Path, image_name: str, **extra):
    resp = client.post("/api/review/mark_complete", json={
        "dataset_root": str(dataset_root), "image_name": image_name, **extra})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_bulk_complete_on_a_predicted_image_is_not_adjudication_covered(
    client: TestClient, tmp_path: Path
) -> None:
    """The bucket holds a detection for this image and no verdict was recorded on it, so the
    completion is a bulk accept: reviewed, but with its misses unadjudicated."""
    from tcip_mcp.buckets import read_bucket

    bucket = _published(tmp_path, BUD, {"IMG_0007": 1, "IMG_0031": 0})
    dataset_root = _dataset_root(tmp_path)

    body = _complete(client, dataset_root, "IMG_0007.JPG", pred_dir=str(bucket))

    assert body["image_status"] == "completed"
    state = _shard(dataset_root, "IMG_0007.JPG")
    assert state["adjudication_covered"] == {"*": False}
    assert state["producer_identity"]["checkpoint_sha256"] == read_bucket(
        bucket).producer["checkpoint_sha256"]


def test_complete_on_an_image_the_bucket_found_nothing_on_is_a_covered_negative(
    client: TestClient, tmp_path: Path,
) -> None:
    """The bucket ran on this image and published an empty document: there was never a detection
    to walk, and Complete confirms the negative outright."""
    bucket = _published(tmp_path, BUD, {"IMG_0007": 1, "IMG_0031": 0})
    dataset_root = _dataset_root(tmp_path)

    _complete(client, dataset_root, "IMG_0031.JPG", pred_dir=str(bucket))

    assert _shard(dataset_root, "IMG_0031.JPG")["adjudication_covered"] == {"*": True}


def test_complete_on_an_image_the_bucket_never_predicted_records_no_negative(
    client: TestClient, tmp_path: Path,
) -> None:
    """A bucket whose record names no document for the image never predicted it: its absence is
    unknown, so the Complete is recorded and no negative is claimed for it."""
    bucket = _published(tmp_path, BUD, {"IMG_0007": 1, "IMG_0031": 0})
    dataset_root = _dataset_root(tmp_path)

    body = _complete(client, dataset_root, "IMG_0099.JPG", pred_dir=str(bucket))

    assert body["image_status"] == "completed"
    assert "adjudication_covered" not in _shard(dataset_root, "IMG_0099.JPG")


def test_complete_named_subject_on_an_image_with_only_another_subjects_gt_is_a_scoped_negative(
    client: TestClient, tmp_path: Path
) -> None:
    """A Complete naming 'bud' on an image whose GT holds only 'leaf' objects is a genuine
    negative of bud, not the 'complete' a whole-file (subject-blind) check would produce."""
    open_new_project(tmp_path)
    dataset_root = _dataset_root(tmp_path)
    gt_path = dataset_root / "annotations" / "2-11-26" / "IMG_0050.json"
    write_annotations(
        str(gt_path), [Annotation(subject="leaf", geometry=BBox(5.0, 5.0, 40.0, 40.0))],
        IMG_W, IMG_H)

    body = _complete(client, dataset_root, "IMG_0050.JPG", gt_path=str(gt_path), subject="bud")

    assert body["annotation_status"] == "negative"


def test_complete_naming_a_classified_buckets_own_subject_with_nothing_found_is_covered(
    client: TestClient, tmp_path: Path
) -> None:
    bucket = _published(tmp_path, OPENING, {"IMG_0060": 0})
    dataset_root = _dataset_root(tmp_path)

    _complete(client, dataset_root, "IMG_0060.JPG", pred_dir=str(bucket), subject="bud")

    assert _shard(dataset_root, "IMG_0060.JPG")["adjudication_covered"] == {"bud": True}


def test_complete_named_subject_the_bucket_never_assessed_omits_the_coverage_entry(
    client: TestClient, tmp_path: Path
) -> None:
    """A subject not among a detector bucket's own recorded class map cannot be judged negative
    or positive: the coverage entry is omitted, and the Complete still proceeds."""
    bucket = _published(tmp_path, LEAF, {"IMG_0007": 1})
    dataset_root = _dataset_root(tmp_path)

    body = _complete(client, dataset_root, "IMG_0007.JPG", pred_dir=str(bucket), subject="bud")

    assert body["image_status"] == "completed"
    assert "bud" not in (_shard(dataset_root, "IMG_0007.JPG").get("adjudication_covered") or {})


def test_a_second_complete_naming_a_subject_the_classified_bucket_cannot_resolve_leaves_the_firsts_claim_intact(
    client: TestClient, tmp_path: Path
) -> None:
    """A Complete confirming the classified bucket's own subject, and a later Complete on the same
    image naming a subject the bucket cannot resolve, both land: the second's unresolvable name
    must not overwrite the first's coverage claim."""
    bucket = _published(tmp_path, OPENING, {"IMG_0070": 0})
    dataset_root = _dataset_root(tmp_path)

    first = _complete(client, dataset_root, "IMG_0070.JPG", pred_dir=str(bucket), subject="bud")
    second = _complete(client, dataset_root, "IMG_0070.JPG", pred_dir=str(bucket),
                       subject="leaf")

    assert first["image_status"] == second["image_status"] == "completed"
    assert _shard(dataset_root, "IMG_0070.JPG")["adjudication_covered"] == {"bud": True}


def test_complete_with_no_subject_records_completion_with_a_null_status(
    client: TestClient, tmp_path: Path
) -> None:
    """A Complete naming no subject still records the review completion, but derives and returns
    no subject-scoped status: there is nothing to scope it to."""
    open_new_project(tmp_path)
    dataset_root = _dataset_root(tmp_path)

    body = _complete(client, dataset_root, "IMG_0080.JPG")

    assert body["annotation_status"] is None
    assert _shard(dataset_root, "IMG_0080.JPG")["adjudication_covered"] == {"*": True}


def test_is_negative_for_subject_agrees_across_branches_after_a_same_size_edit(
    tmp_path: Path,
) -> None:
    """Both the subject-less and the named-subject branch read the prediction file through the
    same memo, so an in-place edit forced onto the file's prior timestamp and byte count is
    answered the same way by both, never one from a parse made before the edit and the other
    fresh."""
    from tcip_mcp.buckets import read_bucket
    from tcip_web.routes.review import _is_negative_for_subject

    bucket = _published(tmp_path, BUD, {"IMG_0007": 1})
    record = read_bucket(bucket)
    pred_file = bucket / "IMG_0007.json"
    populated = pred_file.read_bytes()
    os.utime(pred_file, (1_000_000, 1_000_000))

    assert _is_negative_for_subject("IMG_0007.JPG", None, record) is False
    assert _is_negative_for_subject("IMG_0007.JPG", "bud", record) is False

    write_annotations(str(pred_file), [], IMG_W, IMG_H, keep_empty=True)
    emptied = pred_file.read_bytes()
    # Trailing whitespace is not significant JSON content; padding to the populated document's
    # exact byte count reproduces a same-size in-place edit without hand-authoring the document.
    assert len(emptied) < len(populated)
    pred_file.write_bytes(emptied + b" " * (len(populated) - len(emptied)))
    os.utime(pred_file, (1_000_000, 1_000_000))  # identical mtime and byte count as the populated write

    subject_less = _is_negative_for_subject("IMG_0007.JPG", None, record)
    named = _is_negative_for_subject("IMG_0007.JPG", "bud", record)
    assert subject_less is True
    assert named is True
