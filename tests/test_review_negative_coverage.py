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

import os
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
        [Annotation(subject="bud", geometry=BBox(12.0, 20.0, 52.0, 44.0), score=0.71)],
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
    """A Complete naming 'bud' on an image whose GT holds only 'leaf' objects is a genuine
    negative of bud, not the 'complete' a whole-file (subject-blind) check would produce."""
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
        "subject": "bud",
    })
    assert resp.status_code == 200
    assert resp.json()["annotation_status"] == "negative"


def test_complete_named_subject_over_a_file_holding_only_another_subjects_predictions_is_covered(
    client: TestClient, tmp_path: Path
) -> None:
    """A classified bucket's own subject with nothing recorded for it on this file is a genuine
    negative, even when the same file holds another subject's boxes."""
    from tcip_mcp.pipelines.resolution import operating_point_stamp, write_sidecar

    d = tmp_path / "predictions" / "baseline" / "2-11-26"
    d.mkdir(parents=True)
    write_annotations(
        str(d / "IMG_0060.json"),
        [Annotation(subject="leaf", geometry=BBox(12.0, 20.0, 52.0, 44.0), score=0.71)],
        IMG_W, IMG_H,
    )
    stamp = operating_point_stamp(
        {"conf": {"value": 0.5}}, validated=False, validated_by=None, tile_size_validated=None,
        shippable_issues=[], id_map={"open": 0, "closed": 1}, trait="bud_opening",
        dataset_hash="H", checkpoint="m", checkpoint_sha256=CHECKPOINT_SHA,
        experiment_id="exp-17", images_dir=None, raster_path=None,
        produced_at="2026-01-01T00:00:00Z", subject="bud", attribute="opening",
    )
    write_sidecar(d, stamp)
    dataset_root = _dataset_root(tmp_path)

    resp = client.post("/api/review/mark_complete", json={
        "dataset_root": str(dataset_root),
        "image_name": "IMG_0060.JPG",
        "pred_dir": str(d),
        "subject": "bud",
    })
    assert resp.status_code == 200
    assert _shard(dataset_root, "IMG_0060.JPG")["adjudication_covered"] == {"bud": True}


def test_complete_named_subject_the_bucket_never_assessed_omits_the_coverage_entry(
    client: TestClient, tmp_path: Path
) -> None:
    """A subject not among a detector bucket's own recorded class map cannot be judged negative
    or positive: the coverage entry is omitted, and the Complete and its status write still
    proceed."""
    from tcip_mcp.pipelines.resolution import operating_point_stamp, write_sidecar

    d = tmp_path / "predictions" / "baseline" / "2-11-26"
    d.mkdir(parents=True)
    write_annotations(
        str(d / "IMG_0007.json"),
        [Annotation(subject="leaf", geometry=BBox(12.0, 20.0, 52.0, 44.0), score=0.71)],
        IMG_W, IMG_H,
    )
    stamp = operating_point_stamp(
        {"conf": {"value": 0.5}}, validated=False, validated_by=None, tile_size_validated=None,
        shippable_issues=[], id_map={"leaf": 0}, trait="leaf", dataset_hash="H", checkpoint="m",
        checkpoint_sha256=CHECKPOINT_SHA, experiment_id="exp-17", images_dir=None,
        raster_path=None, produced_at="2026-01-01T00:00:00Z", subject="leaf", attribute=None,
    )
    write_sidecar(d, stamp)
    dataset_root = _dataset_root(tmp_path)

    resp = client.post("/api/review/mark_complete", json={
        "dataset_root": str(dataset_root),
        "image_name": "IMG_0007.JPG",
        "pred_dir": str(d),
        "subject": "bud",
    })
    assert resp.status_code == 200
    assert resp.json()["image_status"] == "completed"
    state = _shard(dataset_root, "IMG_0007.JPG")
    assert "bud" not in (state.get("adjudication_covered") or {})


def test_a_second_complete_naming_a_subject_the_classified_bucket_cannot_resolve_leaves_the_firsts_claim_intact(
    client: TestClient, tmp_path: Path
) -> None:
    """A Complete confirming the classified bucket's own subject, and a later Complete on the same
    image naming a subject the bucket cannot resolve (a classified stamp admits only its own
    object class), both land: the second's unresolvable name must not overwrite the first's
    coverage claim."""
    from tcip_mcp.pipelines.resolution import operating_point_stamp, write_sidecar

    d = tmp_path / "predictions" / "baseline" / "2-11-26"
    d.mkdir(parents=True)
    write_annotations(str(d / "IMG_0070.json"), [], IMG_W, IMG_H, keep_empty=True)
    stamp = operating_point_stamp(
        {"conf": {"value": 0.5}}, validated=False, validated_by=None, tile_size_validated=None,
        shippable_issues=[], id_map={"open": 0, "closed": 1}, trait="bud_opening",
        dataset_hash="H", checkpoint="m", checkpoint_sha256=CHECKPOINT_SHA,
        experiment_id="exp-17", images_dir=None, raster_path=None,
        produced_at="2026-01-01T00:00:00Z", subject="bud", attribute="opening",
    )
    write_sidecar(d, stamp)
    dataset_root = _dataset_root(tmp_path)

    first = client.post("/api/review/mark_complete", json={
        "dataset_root": str(dataset_root), "image_name": "IMG_0070.JPG",
        "pred_dir": str(d), "subject": "bud",
    })
    assert first.status_code == 200
    assert first.json()["image_status"] == "completed"
    second = client.post("/api/review/mark_complete", json={
        "dataset_root": str(dataset_root), "image_name": "IMG_0070.JPG",
        "pred_dir": str(d), "subject": "leaf",
    })
    assert second.status_code == 200
    assert second.json()["image_status"] == "completed"
    state = _shard(dataset_root, "IMG_0070.JPG")
    assert state["adjudication_covered"] == {"bud": True}
    assert "leaf" not in state["adjudication_covered"]


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


def test_is_negative_for_subject_agrees_across_branches_after_a_same_size_edit(
    tmp_path: Path,
) -> None:
    """Both the subject-less and the named-subject branch read the prediction file through the
    same memo, so an in-place edit forced onto the file's prior timestamp and byte count is
    answered the same way by both, never one from a parse made before the edit and the other
    fresh."""
    from tcip_web.routes.review import _is_negative_for_subject

    d = tmp_path / "predictions" / "baseline" / "2-11-26"
    d.mkdir(parents=True)
    pred_file = d / "IMG_0007.json"
    write_annotations(
        str(pred_file),
        [Annotation(subject="bud", geometry=BBox(12.0, 20.0, 52.0, 44.0), score=0.71)],
        IMG_W, IMG_H,
    )
    populated = pred_file.read_bytes()
    os.utime(pred_file, (1_000_000, 1_000_000))
    _seed_sidecar(d, {"checkpoint_sha256": CHECKPOINT_SHA, "id_map": {"bud": 0},
                      "subject": "bud", "attribute": None})

    assert _is_negative_for_subject(str(d), "IMG_0007.JPG", None) is False
    assert _is_negative_for_subject(str(d), "IMG_0007.JPG", "bud") is False

    write_annotations(str(pred_file), [], IMG_W, IMG_H, keep_empty=True)
    emptied = pred_file.read_bytes()
    # Trailing whitespace is not significant JSON content; padding to the populated document's
    # exact byte count reproduces a same-size in-place edit without hand-authoring the document.
    assert len(emptied) < len(populated)
    pred_file.write_bytes(emptied + b" " * (len(populated) - len(emptied)))
    os.utime(pred_file, (1_000_000, 1_000_000))  # identical mtime and byte count as the populated write

    subject_less = _is_negative_for_subject(str(d), "IMG_0007.JPG", None)
    named = _is_negative_for_subject(str(d), "IMG_0007.JPG", "bud")
    assert subject_less is True
    assert named is True
    assert subject_less == named
