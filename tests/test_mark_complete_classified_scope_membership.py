"""``mark_complete``'s classified-scope membership admission: a classified stamp admits exactly its
own object class as a Complete's stated subject, never one of its attribute's values, and a stamp
this door cannot resolve at all (a neither-key stamp, an undecodable one, a bare directory) omits
the coverage entry rather than refusing the Complete.
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
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _seed_sidecar(pred_dir: Path, sidecar: dict) -> None:
    import tcip_store
    from tcip_mcp.pipelines.resolution import sidecar_key

    tcip_store.replace(sidecar_key(pred_dir, "operating_point"), sidecar,
                       expect=tcip_store.Version.ABSENT)


def _damage_sidecar(pred_dir: Path) -> None:
    import os

    from tcip_mcp.pipelines.resolution import sidecar_key
    from tcip_store.binding import BACKEND_ENV, DEFAULT_BACKEND, FILE_BACKEND
    from tcip_store.store import _backend

    key = sidecar_key(pred_dir, "operating_point")
    if (os.environ.get(BACKEND_ENV) or DEFAULT_BACKEND) == FILE_BACKEND:
        _backend().path_for(key).write_bytes(b"{not json")
        return
    import sqlite3

    from tcip_store.sqlite_backend import database_path, encode_parts

    conn = sqlite3.connect(str(database_path(str(key.root))), isolation_level=None)
    try:
        conn.execute("update records set value = ? where store = ? and parts = ?",
                    (b"{not json", key.store, encode_parts(key.parts)))
    finally:
        conn.close()


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


def test_a_classified_stamp_admits_its_own_subject_and_omits_a_value_name(
    client: TestClient, tmp_path: Path,
) -> None:
    d = tmp_path / "predictions" / "classifier" / "2026-05-10"
    d.mkdir(parents=True)
    write_annotations(str(d / "IMG_0010.json"), [], IMG_W, IMG_H, keep_empty=True)
    _seed_sidecar(d, {"id_map": {"open": 0, "closed": 1}, "subject": SUBJECT, "attribute": ATTRIBUTE})
    dataset_root = _dataset_root(tmp_path)

    own_subject = client.post("/api/review/mark_complete", json={
        "dataset_root": str(dataset_root), "image_name": "IMG_0010.JPG",
        "pred_dir": str(d), "subject": SUBJECT,
    })
    assert own_subject.status_code == 200
    assert _shard(dataset_root, "IMG_0010.JPG")["adjudication_covered"] == {SUBJECT: True}

    value_name = client.post("/api/review/mark_complete", json={
        "dataset_root": str(dataset_root), "image_name": "IMG_0011.JPG",
        "pred_dir": str(d), "subject": "open",
    })
    assert value_name.status_code == 200
    state = _shard(dataset_root, "IMG_0011.JPG")
    assert not (state.get("adjudication_covered") or {})


def test_a_neither_key_stamp_omits_the_entry_and_still_completes(
    client: TestClient, tmp_path: Path,
) -> None:
    d = tmp_path / "predictions" / "classifier" / "2026-05-11"
    d.mkdir(parents=True)
    write_annotations(str(d / "IMG_0020.json"), [], IMG_W, IMG_H, keep_empty=True)
    _seed_sidecar(d, {"id_map": {"open": 0, "closed": 1}})
    dataset_root = _dataset_root(tmp_path)

    resp = client.post("/api/review/mark_complete", json={
        "dataset_root": str(dataset_root), "image_name": "IMG_0020.JPG",
        "pred_dir": str(d), "subject": SUBJECT,
    })

    assert resp.status_code == 200
    assert resp.json()["image_status"] == "completed"
    state = _shard(dataset_root, "IMG_0020.JPG")
    assert not (state.get("adjudication_covered") or {})


def test_an_undecodable_stamp_omits_the_entry_and_still_completes(
    client: TestClient, tmp_path: Path,
) -> None:
    d = tmp_path / "predictions" / "classifier" / "2026-05-12"
    d.mkdir(parents=True)
    write_annotations(str(d / "IMG_0030.json"), [], IMG_W, IMG_H, keep_empty=True)
    _seed_sidecar(d, {"id_map": {"open": 0, "closed": 1}, "subject": SUBJECT, "attribute": ATTRIBUTE})
    _damage_sidecar(d)
    dataset_root = _dataset_root(tmp_path)

    resp = client.post("/api/review/mark_complete", json={
        "dataset_root": str(dataset_root), "image_name": "IMG_0030.JPG",
        "pred_dir": str(d), "subject": SUBJECT,
    })

    assert resp.status_code == 200
    assert resp.json()["image_status"] == "completed"
    state = _shard(dataset_root, "IMG_0030.JPG")
    assert not (state.get("adjudication_covered") or {})


def test_a_bare_directory_with_a_named_subject_omits_the_entry_and_still_completes(
    client: TestClient, tmp_path: Path,
) -> None:
    d = tmp_path / "predictions" / "baseline" / "2026-05-13"
    d.mkdir(parents=True)
    write_annotations(str(d / "IMG_0040.json"), [], IMG_W, IMG_H, keep_empty=True)
    dataset_root = _dataset_root(tmp_path)

    resp = client.post("/api/review/mark_complete", json={
        "dataset_root": str(dataset_root), "image_name": "IMG_0040.JPG",
        "pred_dir": str(d), "subject": SUBJECT,
    })

    assert resp.status_code == 200
    assert resp.json()["image_status"] == "completed"
    state = _shard(dataset_root, "IMG_0040.JPG")
    assert not (state.get("adjudication_covered") or {})
