"""The per-image status store is written where the shared locator says, and merged into.

``image_status_path`` is the one locator for the confirmed-negative store: a write resolves through
it rather than rebuilding the path, so every reader finds the human's confirmations, and it folds
into whatever the store already holds, so another subject's or another date's confirmations are
never replaced by the next write.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_mcp.dataset_layout import image_status_path, normalize_status_store
from tcip_web.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _single(client: TestClient, root: Path, image_name: str, status: str, subject: str,
            date: str | None = None) -> None:
    body = {"project_root": str(root), "dataset_root": str(root), "image_name": image_name,
            "status": status, "subject": subject}
    if date:
        body["date"] = date
    resp = client.post("/api/classes/image_status", json=body)
    assert resp.status_code == 200, resp.text


def _bulk(client: TestClient, root: Path, statuses: dict[str, str], subject: str,
          date: str | None = None) -> None:
    body = {"project_root": str(root), "dataset_root": str(root), "subject": subject,
            "statuses": statuses}
    if date:
        body["date"] = date
    resp = client.post("/api/classes/image_status/bulk", json=body)
    assert resp.status_code == 200, resp.text


def _on_disk(root: Path) -> dict[str, dict[str, str]]:
    return normalize_status_store(
        json.loads(image_status_path(root).read_text(encoding="utf-8")))


def test_a_status_write_lands_at_the_shared_locator_and_creates_no_second_store(
    client: TestClient, tmp_path: Path
) -> None:
    """The store the readers resolve through is the store the route wrote, and it is the only
    state file the write produced: a second, locally built path would strand the confirmation."""
    _single(client, tmp_path, "IMG_0001.JPG", "complete", "catkin")

    assert image_status_path(tmp_path).is_file()
    assert _on_disk(tmp_path) == {"catkin": {"IMG_0001.JPG": "complete"}}

    tcip_dir = tmp_path / ".tcip"
    written = sorted(
        p.relative_to(tcip_dir).as_posix() for p in tcip_dir.rglob("*") if p.is_file())
    assert written == ["audit.jsonl", "state/image_status.json"]


def test_confirmations_for_other_subjects_and_dates_survive_a_later_write(
    client: TestClient, tmp_path: Path
) -> None:
    """Every write folds into the store rather than replacing it, whichever route made it, so the
    buckets accumulate instead of the last writer winning the whole file."""
    _bulk(client, tmp_path, {"A.JPG": "complete", "B.JPG": "negative", "C.JPG": "partial"},
          "catkin")
    _single(client, tmp_path, "D.JPG", "negative", "bush")
    _bulk(client, tmp_path, {"A.JPG": "negative", "E.JPG": "complete"}, "catkin", "2026-03-09")
    _single(client, tmp_path, "F.JPG", "unannotated", "catkin")

    assert _on_disk(tmp_path) == {
        "catkin": {"A.JPG": "complete", "B.JPG": "negative", "C.JPG": "partial",
                   "F.JPG": "unannotated"},
        "bush": {"D.JPG": "negative"},
        "catkin/2026-03-09": {"A.JPG": "negative", "E.JPG": "complete"},
    }


def test_the_read_route_returns_the_bucket_the_write_routes_built(
    client: TestClient, tmp_path: Path
) -> None:
    """Reader and writer agree on which bucket a subject and date name, so a confirmation is read
    back under the same scope it was recorded under and never under a neighbouring one."""
    _bulk(client, tmp_path, {"A.JPG": "complete", "B.JPG": "negative", "C.JPG": "partial"},
          "catkin")
    _single(client, tmp_path, "D.JPG", "negative", "bush")
    _bulk(client, tmp_path, {"A.JPG": "negative", "E.JPG": "complete"}, "catkin", "2026-03-09")

    on_disk = _on_disk(tmp_path)
    assert len(on_disk) == 3

    def read(subject: str, date: str | None) -> dict[str, str]:
        params = {"project_root": str(tmp_path), "dataset_root": str(tmp_path),
                  "subject": subject}
        if date:
            params["date"] = date
        resp = client.get("/api/classes/image_status", params=params)
        assert resp.status_code == 200, resp.text
        return resp.json()["statuses"]

    assert read("catkin", None) == on_disk["catkin"]
    assert read("bush", None) == on_disk["bush"]
    assert read("catkin", "2026-03-09") == on_disk["catkin/2026-03-09"]
