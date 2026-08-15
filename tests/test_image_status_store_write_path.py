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
    # The .lock artifact can outlive the write (filelock version, platform); it is not a state file.
    written = sorted(
        p.relative_to(tcip_dir).as_posix()
        for p in tcip_dir.rglob("*") if p.is_file() and p.suffix != ".lock")
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


def test_a_status_the_readers_do_not_understand_is_refused_before_it_reaches_the_store(
    tmp_path: Path,
) -> None:
    """A token outside the store's vocabulary never lands, and every token inside it does.

    A recorded status nothing can interpret is neither a confirmation nor a negative, so it is
    refused; the four the readers do understand still write, which is what keeps the refusal from
    standing between the breeder and an ordinary confirmation.
    """
    from tcip_mcp.dataset_layout import IMAGE_STATUSES, record_image_statuses

    with pytest.raises(ValueError, match="reviewed"):
        record_image_statuses(tmp_path, "catkin", {"A.JPG": "reviewed"})
    assert not image_status_path(tmp_path).exists()

    record_image_statuses(tmp_path, "catkin", {f"{s}.JPG": s for s in IMAGE_STATUSES})
    assert _on_disk(tmp_path)["catkin"] == {f"{s}.JPG": s for s in IMAGE_STATUSES}


def test_a_materialized_output_carries_only_the_negatives_that_run_derived(tmp_path: Path) -> None:
    """A dataset written out by a materializer holds exactly the confirmations that run produced.

    Writing into a directory an earlier run already used replaces its store rather than folding
    into it: a leftover name nobody re-derived would otherwise keep training as an empty image.
    """
    from tcip_mcp.dataset_layout import replace_image_status_store

    replace_image_status_store(tmp_path, {"catkin": {"OLD.JPG": "negative"}})
    replace_image_status_store(tmp_path, {"catkin": {"NEW.JPG": "negative"}})

    assert _on_disk(tmp_path) == {"catkin": {"NEW.JPG": "negative"}}


def test_a_schema_stamp_leaves_every_other_image_stamp_in_place(tmp_path: Path) -> None:
    """Stamping one image merges into the sidecar, per image and per bucket.

    A whole-document write would drop a stamp another image's confirmation was quarantined by,
    which un-quarantines a confirmation made under a schema that has since changed.
    """
    from tcip_mcp.dataset_layout import image_status_digest_path, stamp_image_status_digests

    stamp_image_status_digests(tmp_path, "catkin", ["A.JPG", "B.JPG"], "digest-one")
    stamp_image_status_digests(tmp_path, "bush", ["C.JPG"], "digest-two")
    stamp_image_status_digests(tmp_path, "catkin", ["B.JPG"], "digest-three")

    assert json.loads(image_status_digest_path(tmp_path).read_text(encoding="utf-8")) == {
        "bush": {"C.JPG": "digest-two"},
        "catkin": {"A.JPG": "digest-one", "B.JPG": "digest-three"},
    }
