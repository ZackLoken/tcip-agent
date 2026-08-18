"""Review routes under an armed ``TCIP_IMAGE_ROOTS`` allow-list.

An exposed deployment restricts the absolute, client-supplied paths these routes read and write.
Every case below pins both directions: a path outside the configured roots is refused with 403,
and the equivalent path inside them still does its work. An optional path the client simply did
not send is not an escape and must keep being admitted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tcip_annotation.json_io import read_annotations, write_annotations
from tcip_annotation.state import Annotation, BBox
from tcip_web.app import app

IMG_W, IMG_H = 160, 100


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def allowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An armed allow-list holding exactly one root, with a real image and dataset layout inside."""
    root = tmp_path / "allowed"
    (root / "images").mkdir(parents=True)
    Image.new("RGB", (IMG_W, IMG_H), color=(70, 90, 110)).save(root / "images" / "IMG_0007.JPG")
    monkeypatch.setenv("TCIP_IMAGE_ROOTS", str(root.resolve()))
    return root


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    """A directory deliberately left out of the allow-list."""
    d = tmp_path / "elsewhere"
    d.mkdir()
    return d


def _image(allowed: Path) -> Path:
    return allowed / "images" / "IMG_0007.JPG"


def _dataset_root(allowed: Path) -> Path:
    """The dataset the review store, the labels and any prediction bucket all hang off."""
    return allowed


def test_save_gt_confines_the_label_file_it_writes(
    client: TestClient, allowed: Path, outside: Path
) -> None:
    escaped = outside / "stolen.json"
    payload = {
        "dataset_root": str(_dataset_root(allowed)),
        "image_name": "IMG_0007.JPG",
        "image_path": str(_image(allowed)),
        "annotations": [{"subject": "catkin", "bbox": [12.0, 20.0, 52.0, 44.0]}],
    }

    refused = client.post("/api/review/save_gt", json={**payload, "label_path": str(escaped)})
    assert refused.status_code == 403
    assert not escaped.exists()

    inside = allowed / "annotations" / "2-11-26" / "IMG_0007.json"
    accepted = client.post("/api/review/save_gt", json={**payload, "label_path": str(inside)})
    assert accepted.status_code == 200
    (written,) = read_annotations(str(inside))
    assert written.subject == "catkin"


def test_mark_complete_confines_the_paths_it_reads(
    client: TestClient, allowed: Path, outside: Path
) -> None:
    base = {"dataset_root": str(_dataset_root(allowed)), "image_name": "IMG_0007.JPG"}

    assert client.post("/api/review/mark_complete",
                       json={**base, "gt_path": str(outside / "gt.json")}).status_code == 403
    assert client.post("/api/review/mark_complete",
                       json={**base, "pred_dir": str(outside)}).status_code == 403

    gt = allowed / "annotations" / "2-11-26" / "IMG_0007.json"
    gt.parent.mkdir(parents=True)
    write_annotations(str(gt), [Annotation(subject="catkin", geometry=BBox(12.0, 20.0, 52.0, 44.0))],
                      IMG_W, IMG_H)
    accepted = client.post("/api/review/mark_complete", json={**base, "gt_path": str(gt)})
    assert accepted.status_code == 200
    assert accepted.json()["annotation_status"] == "complete"


def test_an_optional_path_the_client_omits_is_not_an_escape(
    client: TestClient, allowed: Path
) -> None:
    """A completion with no label file and no bucket named is a legitimate call, and the armed
    allow-list must keep admitting it rather than reading the absent path as an escape."""
    resp = client.post("/api/review/mark_complete", json={
        "dataset_root": str(_dataset_root(allowed)),
        "image_name": "IMG_0007.JPG",
    })
    assert resp.status_code == 200
    assert resp.json()["annotation_status"] == "negative"


def test_backup_labels_confines_the_directories_it_walks(
    client: TestClient, allowed: Path, outside: Path
) -> None:
    (outside / "IMG_0007.json").write_text('{"annotations": []}', encoding="utf-8")
    base = {"dataset_root": str(_dataset_root(allowed))}

    refused = client.post("/api/review/backup_labels",
                          json={**base, "label_dirs": [str(outside)]})
    assert refused.status_code == 403
    assert not (outside / ".original").exists()

    label_dir = allowed / "annotations" / "2-11-26"
    label_dir.mkdir(parents=True)
    write_annotations(str(label_dir / "IMG_0007.json"),
                      [Annotation(subject="catkin", geometry=BBox(12.0, 20.0, 52.0, 44.0))],
                      IMG_W, IMG_H)
    accepted = client.post("/api/review/backup_labels",
                           json={**base, "label_dirs": [str(label_dir)]})
    assert accepted.status_code == 200
    assert accepted.json()["files_backed_up"] == 1
    assert (label_dir / ".original" / "IMG_0007.json").is_file()


def test_image_statuses_confines_the_label_directories_it_lists(
    client: TestClient, allowed: Path, outside: Path
) -> None:
    write_annotations(str(outside / "IMG_0007.json"),
                      [Annotation(subject="catkin", geometry=BBox(12.0, 20.0, 52.0, 44.0),
                                  score=0.71)],
                      IMG_W, IMG_H)
    dataset_root = str(_dataset_root(allowed))

    refused = client.get("/api/review/image_statuses",
                         params={"dataset_root": dataset_root, "pred_dir": str(outside)})
    assert refused.status_code == 403

    bucket = allowed / "predictions" / "baseline" / "2-11-26"
    bucket.mkdir(parents=True)
    write_annotations(str(bucket / "IMG_0007.json"),
                      [Annotation(subject="catkin", geometry=BBox(12.0, 20.0, 52.0, 44.0),
                                  score=0.71)],
                      IMG_W, IMG_H)
    accepted = client.get("/api/review/image_statuses",
                          params={"dataset_root": dataset_root, "pred_dir": str(bucket)})
    assert accepted.status_code == 200
    assert accepted.json()["detection_stems"] == ["IMG_0007"]


def test_generation_conf_confines_the_bucket_it_reads(
    client: TestClient, allowed: Path, outside: Path
) -> None:
    (outside / "operating_point.json").write_text(
        json.dumps({"operating_point": {"conf": {"value": 0.44}}}), encoding="utf-8")

    refused = client.get("/api/review/generation_conf", params={"pred_dir": str(outside)})
    assert refused.status_code == 403

    bucket = allowed / "predictions" / "baseline" / "2-11-26"
    bucket.mkdir(parents=True)
    import tcip_store
    from tcip_mcp.pipelines.resolution import sidecar_key

    tcip_store.replace(sidecar_key(bucket, "operating_point"),
                       {"operating_point": {"conf": {"value": 0.44}}},
                       expect=tcip_store.Version.ABSENT)
    accepted = client.get("/api/review/generation_conf", params={"pred_dir": str(bucket)})
    assert accepted.status_code == 200
    assert accepted.json()["generation_conf"] == pytest.approx(0.44)


def test_validate_reference_confines_the_bucket_it_stamps(
    client: TestClient, allowed: Path, outside: Path
) -> None:
    base = {"dataset_root": str(_dataset_root(allowed)), "trait": "catkin"}

    refused = client.post("/api/review/validate_reference",
                          json={**base, "pred_dir": str(outside)})
    assert refused.status_code == 403
    assert not (outside / "operating_point.json").exists()

    bucket = allowed / "predictions" / "baseline" / "2-11-26"
    bucket.mkdir(parents=True)
    accepted = client.post("/api/review/validate_reference",
                           json={**base, "pred_dir": str(bucket)})
    assert accepted.status_code == 200
    body = accepted.json()
    assert body["validated"] is False
    assert body["reviewed_image_count"] == 0


def test_priority_queue_launch_confines_the_checkpoint_it_loads(
    client: TestClient, allowed: Path, outside: Path
) -> None:
    (outside / "best.pt").write_bytes(b"not a real checkpoint")
    images_dir = allowed / "images"

    refused = client.post("/api/review/queue/launch", json={
        "dataset_root": str(_dataset_root(allowed)),
        "checkpoint_path": str(outside / "best.pt"),
        "images_dir": str(images_dir),
    })
    assert refused.status_code == 403

    # An in-root checkpoint clears the guard and is answered on its own merits.
    missing = client.post("/api/review/queue/launch", json={
        "dataset_root": str(_dataset_root(allowed)),
        "checkpoint_path": str(allowed / "models" / "best.pt"),
        "images_dir": str(images_dir),
    })
    assert missing.status_code == 404
    assert "checkpoint not found" in missing.json()["detail"]
