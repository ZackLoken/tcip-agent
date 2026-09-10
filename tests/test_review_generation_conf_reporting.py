"""The confidence floor a prediction bucket was exported at, as the Review filter shelf reads it.

Raising the review's own confidence filter above this floor hides low-confidence detections from
the breeder and censors any reference built from the resulting verdicts, so the filter shelf warns
about it live. The warning is only as good as the number behind it: the route reports the bucket's
own recorded value, and reports its absence as absent rather than as a floor of zero.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_web.app import app

RECORDED_CONF = 0.37


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _bucket(tmp_path: Path, name: str, sidecar: dict | None) -> Path:
    d = tmp_path / "predictions" / name
    d.mkdir(parents=True)
    if sidecar is not None:
        import tcip_store
        from tcip_mcp.pipelines.resolution import sidecar_key

        tcip_store.replace(sidecar_key(d, "operating_point"), sidecar,
                           expect=tcip_store.Version.ABSENT)
    return d


def test_generation_conf_reports_the_floor_the_bucket_recorded(
    client: TestClient, tmp_path: Path
) -> None:
    bucket = _bucket(tmp_path, "baseline", {
        "checkpoint_sha256": "3f9c1ab27e",
        "operating_point": {
            "conf": {"value": RECORDED_CONF, "source": "derived"},
            "tile_size": {"value": 1024},
        },
    })

    resp = client.get("/api/review/generation_conf", params={"pred_dir": str(bucket)})
    assert resp.status_code == 200
    assert resp.json()["generation_conf"] == pytest.approx(RECORDED_CONF)


def test_generation_conf_is_absent_when_the_bucket_recorded_none(
    client: TestClient, tmp_path: Path
) -> None:
    """A bucket with no sidecar, and one whose sidecar carries no usable conf value, both report
    an absent floor. A zero would read as a bucket exported with no confidence floor at all, which
    is a different and stronger claim than not knowing."""
    no_sidecar = _bucket(tmp_path, "unstamped", None)
    resp = client.get("/api/review/generation_conf", params={"pred_dir": str(no_sidecar)})
    assert resp.status_code == 200
    assert resp.json()["generation_conf"] is None

    unusable = _bucket(tmp_path, "hand_edited",
                       {"operating_point": {"conf": {"value": "0.37"}}})
    resp = client.get("/api/review/generation_conf", params={"pred_dir": str(unusable)})
    assert resp.status_code == 200
    assert resp.json()["generation_conf"] is None


_SENTINEL = "no-such-key-was-here"


def test_generation_conf_answers_a_null_admission_rule_with_a_naming_reason(
    client: TestClient, tmp_path: Path,
) -> None:
    """admission_rule is null with a reason naming why, for every stamp shape that answers no
    rule, each read as what it is (an absent stamp, a stamp claiming nothing, and a validated
    stamp whose row is gone read as three distinct sentences, never one collapsed into another).
    A guard: the key's own presence (against a sentinel default) is asserted separately from its
    value for all three cases, since the route may omit the admission_rule/admission_reason key
    entirely rather than merely setting it to null.
    The non-numeric-conf case (a successfully bound claim whose stamp still carries an unreadable
    value) is exercised as a unit test of admission_rule_of itself
    (test_admission_rule_of.py), since reaching it through a real bound claim needs a
    validation row and a bucket-content digest to agree with a value the row's own claim never
    recorded, a shape no producer and no hand-written store row can construct honestly."""
    unvalidated = _bucket(tmp_path, "unvalidated", {
        "checkpoint_sha256": "x", "validated": False,
        "operating_point": {"conf": {"value": 0.4}},
    })
    no_stamp = _bucket(tmp_path, "no_stamp", None)
    row_gone = _bucket(tmp_path, "row_gone", {
        "checkpoint_sha256": "x", "validated": True,
        "validated_by": {"experiment_id": "exp-does-not-exist", "record_digest": "0" * 16},
        "operating_point": {
            "conf": {"value": 0.4, "validated_against": "held_out_annotations"},
        },
    })

    for bucket, reason_fragment in (
        (unvalidated, "does not claim validated"),
        (no_stamp, "no operating_point.json"),
        (row_gone, "exp-does-not-exist"),
    ):
        resp = client.get("/api/review/generation_conf", params={"pred_dir": str(bucket)})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body.get("admission_rule", _SENTINEL) is None  # a guard: key absent, not merely null
        assert body["admission_rule"] is None
        assert reason_fragment in body["admission_reason"]
    assert "does not claim validated" not in (
        client.get("/api/review/generation_conf", params={"pred_dir": str(no_stamp)})
        .json()["admission_reason"]
    )


def _damage_stamp(bucket: Path) -> None:
    """Corrupt the already-written operating_point stamp's bytes in place, wherever the bound
    backend keeps it: the write side refuses the same malformed shapes the read side would, so an
    undecodable stamp can only be produced underneath the store, never through it."""
    import os

    from tcip_mcp.pipelines.resolution import sidecar_key
    from tcip_store.binding import BACKEND_ENV, DEFAULT_BACKEND, FILE_BACKEND
    from tcip_store.store import _backend

    key = sidecar_key(bucket, "operating_point")
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


def test_generation_conf_answers_the_decode_error_as_the_reason(
    client: TestClient, tmp_path: Path,
) -> None:
    """A stamp that will not decode reads as its own decode error, never as an absent stamp: the
    route reads the stamp strictly for the rule, unlike the plain generation_conf reading."""
    bucket = _bucket(tmp_path, "undecodable", {"operating_point": {"conf": {"value": 0.4}}})
    _damage_stamp(bucket)

    resp = client.get("/api/review/generation_conf", params={"pred_dir": str(bucket)})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["admission_rule"] is None
    assert "no operating_point.json" not in body["admission_reason"]
    assert "does not claim validated" not in body["admission_reason"]
