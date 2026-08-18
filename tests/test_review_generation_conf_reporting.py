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
    return TestClient(app)


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
