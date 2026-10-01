"""``mark_complete``'s classified-scope membership admission: a classified bucket admits exactly its
own object class as a Complete's stated subject, never one of its attribute's values, and a bucket
this door cannot resolve a subject for (a directory of no bucket) omits the coverage entry rather
than refusing the Complete, while a record that will not read refuses it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_annotation.json_io import write_annotations
from tcip_web.app import app

IMG_W, IMG_H = 160, 100
SUBJECT = "bud"
ATTRIBUTE = "opening"


@pytest.fixture
def client(opened_project: Path) -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _classified_bucket(project: Path, date: str) -> Path:
    """A classified bucket published over ``date`` holding an empty document for each of two
    images."""
    pytest.importorskip("torch")
    from tests._chain_fixtures import published

    bucket = project / "predictions" / "classifier" / date
    published(project, bucket, [
        {"image": name, "width": 64, "height": 64, "boxes": [], "scores": [], "labels": []}
        for name in ("IMG_0010.JPG", "IMG_0011.JPG")],
        scope={"subject": SUBJECT, "attribute": ATTRIBUTE, "id_map": {"open": 0, "closed": 1}})
    return bucket


def _dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / ".tcip" / "state").mkdir(parents=True)
    return root


def _shard(dataset_root: Path, image_name: str) -> dict:
    import tcip_store
    from tcip_annotation.review_engine import REVIEW_VERDICTS_STORE

    state_dir = str(dataset_root / ".tcip" / "state")
    found = [k for k in tcip_store.keys(REVIEW_VERDICTS_STORE, state_dir) if k.parts[1] == image_name]
    assert len(found) == 1, found
    return tcip_store.read(found[0])["state"]


def _complete(client: TestClient, dataset_root: Path, image_name: str, pred_dir: Path,
              subject: str):
    return client.post("/api/review/mark_complete", json={
        "dataset_root": str(dataset_root), "image_name": image_name, "pred_dir": str(pred_dir),
        "subject": subject})


def test_a_classified_bucket_admits_its_own_subject_and_omits_a_value_name(
    client: TestClient, tmp_path: Path,
) -> None:
    d = _classified_bucket(tmp_path, "2026-05-10")
    dataset_root = _dataset_root(tmp_path)

    own_subject = _complete(client, dataset_root, "IMG_0010.JPG", d, SUBJECT)
    assert own_subject.status_code == 200
    assert _shard(dataset_root, "IMG_0010.JPG")["adjudication_covered"] == {SUBJECT: True}

    value_name = _complete(client, dataset_root, "IMG_0011.JPG", d, "open")
    assert value_name.status_code == 200
    state = _shard(dataset_root, "IMG_0011.JPG")
    assert not (state.get("adjudication_covered") or {})


def test_an_undecodable_bucket_record_refuses_the_complete_and_writes_nothing(
    client: TestClient, tmp_path: Path,
) -> None:
    """The record states which model produced the predictions the Complete adjudicates, so one
    that will not read refuses naming it rather than recording a Complete with no producer."""
    import tcip_store
    from tcip_annotation.review_engine import REVIEW_VERDICTS_STORE

    d = _classified_bucket(tmp_path, "2026-05-12")
    (d / "bucket.json").write_bytes(b"{not json")
    dataset_root = _dataset_root(tmp_path)

    resp = _complete(client, dataset_root, "IMG_0030.JPG", d, SUBJECT)

    assert resp.status_code == 400
    assert "does not decode" in resp.json()["detail"]
    assert tcip_store.keys(REVIEW_VERDICTS_STORE, str(dataset_root / ".tcip" / "state")) == []


def test_a_directory_with_no_bucket_record_refuses_the_complete_and_records_nothing(
    client: TestClient, tmp_path: Path,
) -> None:
    """Documents with no record beside them name no producer, so a Complete adjudicating them
    refuses naming the record rather than reading them as some model's predictions."""
    import tcip_store
    from tcip_annotation.review_engine import REVIEW_VERDICTS_STORE

    d = tmp_path / "predictions" / "baseline" / "2026-05-13"
    d.mkdir(parents=True)
    write_annotations(str(d / "IMG_0040.json"), [], IMG_W, IMG_H, keep_empty=True)
    dataset_root = _dataset_root(tmp_path)

    resp = _complete(client, dataset_root, "IMG_0040.JPG", d, SUBJECT)

    assert resp.status_code == 400
    assert "bucket.json" in resp.json()["detail"]
    assert tcip_store.keys(REVIEW_VERDICTS_STORE, str(dataset_root / ".tcip" / "state")) == []
