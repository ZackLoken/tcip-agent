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

import tcip_store
from tcip_mcp.dataset_layout import (
    image_status_digest_key,
    image_status_key,
    image_status_path,
    normalize_status_store,
)
from tcip_web.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


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


def _stored(root: Path) -> dict[str, dict[str, str]]:
    """The store's content through the seam, wherever the selected backend actually keeps it."""
    return normalize_status_store(tcip_store.read(image_status_key(root), default={}))


def test_a_status_write_lands_at_the_shared_locator_and_creates_no_second_store(
    client: TestClient, tmp_path: Path
) -> None:
    """The store the readers resolve through is the store the route wrote, and it is the only
    state file the write produced: a second, locally built path would strand the confirmation.

    Bound to the file backend on purpose: the claim is about the exact set of files the write
    leaves on disk, which only the file backend produces at all.
    """
    from tcip_store.file_backend import FileBackend

    tcip_store.bind(FileBackend())
    _single(client, tmp_path, "IMG_0001.JPG", "complete", "bud")

    assert image_status_path(tmp_path).is_file()
    assert _on_disk(tmp_path) == {"bud": {"IMG_0001.JPG": "complete"}}

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
          "bud")
    _single(client, tmp_path, "D.JPG", "negative", "bush")
    _bulk(client, tmp_path, {"A.JPG": "negative", "E.JPG": "complete"}, "bud", "2026-03-09")
    _single(client, tmp_path, "F.JPG", "unannotated", "bud")

    assert _stored(tmp_path) == {
        "bud": {"A.JPG": "complete", "B.JPG": "negative", "C.JPG": "partial",
                   "F.JPG": "unannotated"},
        "bush": {"D.JPG": "negative"},
        "bud/2026-03-09": {"A.JPG": "negative", "E.JPG": "complete"},
    }


def test_the_read_route_returns_the_bucket_the_write_routes_built(
    client: TestClient, tmp_path: Path
) -> None:
    """Reader and writer agree on which bucket a subject and date name, so a confirmation is read
    back under the same scope it was recorded under and never under a neighbouring one."""
    _bulk(client, tmp_path, {"A.JPG": "complete", "B.JPG": "negative", "C.JPG": "partial"},
          "bud")
    _single(client, tmp_path, "D.JPG", "negative", "bush")
    _bulk(client, tmp_path, {"A.JPG": "negative", "E.JPG": "complete"}, "bud", "2026-03-09")

    on_disk = _stored(tmp_path)
    assert len(on_disk) == 3

    def read(subject: str, date: str | None) -> dict[str, str]:
        params = {"project_root": str(tmp_path), "dataset_root": str(tmp_path),
                  "subject": subject}
        if date:
            params["date"] = date
        resp = client.get("/api/classes/image_status", params=params)
        assert resp.status_code == 200, resp.text
        return resp.json()["statuses"]

    assert read("bud", None) == on_disk["bud"]
    assert read("bush", None) == on_disk["bush"]
    assert read("bud", "2026-03-09") == on_disk["bud/2026-03-09"]


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
        record_image_statuses(tmp_path, "bud", {"A.JPG": "reviewed"}, recorded_by="user:ada")
    assert not tcip_store.exists(image_status_key(tmp_path))

    record_image_statuses(tmp_path, "bud", {f"{s}.JPG": s for s in IMAGE_STATUSES},
                          recorded_by="user:ada")
    assert _stored(tmp_path)["bud"] == {f"{s}.JPG": s for s in IMAGE_STATUSES}


def test_a_status_with_no_actor_behind_it_is_refused(tmp_path: Path) -> None:
    """A status nobody is recorded as setting cannot be told from one a function wrote, so the
    writer refuses it rather than storing an unattributable confirmation."""
    from tcip_mcp.dataset_layout import record_image_statuses

    with pytest.raises(ValueError, match="recorded_by"):
        record_image_statuses(tmp_path, "bud", {"A.JPG": "negative"}, recorded_by="")
    assert not image_status_path(tmp_path).exists()


def test_a_merge_refuses_rather_than_deleting_entries_it_cannot_read(tmp_path: Path) -> None:
    """A merge rewrites the whole document, so entries the reader drops are entries it deletes.

    The writer refuses instead, naming how many are at stake: a store written in some other shape
    holds human statuses, and losing them to an unrelated confirmation is not a recoverable error.
    """
    from tcip_mcp.dataset_layout import record_image_statuses

    tcip_store.replace(image_status_key(tmp_path),
                       {"bud": {"OLD_A.JPG": "negative", "OLD_B.JPG": "complete"}},
                       expect=tcip_store.Version.ABSENT)

    with pytest.raises(ValueError, match="does not recognize"):
        record_image_statuses(tmp_path, "bud", {"NEW.JPG": "negative"},
                              recorded_by="user:breeder")

    still_there = tcip_store.read(image_status_key(tmp_path))
    assert still_there == {"bud": {"OLD_A.JPG": "negative", "OLD_B.JPG": "complete"}}


def test_the_gui_route_records_the_person_whose_confirmation_it_is(
    client: TestClient, tmp_path: Path
) -> None:
    """A Complete-with-nothing posted from the GUI reads back as a negative and says who made it.

    The identity convention marks a person as ``user:<name>``, so a breeder's own confirmation is
    distinguishable from one a harvest transcribed without consulting a second store.
    """
    from tcip_mcp.pipelines.data.label_queries import confirmed_negative_names

    body = {"project_root": str(tmp_path), "dataset_root": str(tmp_path),
            "image_name": "IMG_0009.JPG", "status": "negative", "subject": "bud",
            "user": "rowan"}
    assert client.post("/api/classes/image_status", json=body).status_code == 200

    stored = tcip_store.read(image_status_key(tmp_path))
    record = stored["bud"]["IMG_0009.JPG"]
    assert record["status"] == "negative"
    assert record["recorded_by"] == "user:rowan"
    assert record["recorded_at"]

    (tmp_path / "annotations").mkdir(parents=True, exist_ok=True)
    assert confirmed_negative_names(tmp_path / "annotations", subject="bud", date=None) == {
        "IMG_0009.JPG"}


def test_a_materialized_output_carries_only_the_negatives_that_run_derived(tmp_path: Path) -> None:
    """A dataset written out by a materializer holds exactly the confirmations that run produced.

    Writing into a directory an earlier run already used replaces its store rather than folding
    into it: a leftover name nobody re-derived would otherwise keep training as an empty image.
    """
    from tcip_mcp.dataset_layout import replace_image_status_store, status_records

    def negatives(*names: str) -> dict:
        return status_records(dict.fromkeys(names, "negative"), recorded_by="a_materializer")

    replace_image_status_store(tmp_path, {"bud": negatives("OLD.JPG")})
    replace_image_status_store(tmp_path, {"bud": negatives("NEW.JPG")})

    assert _stored(tmp_path) == {"bud": {"NEW.JPG": "negative"}}


def test_a_schema_stamp_leaves_every_other_image_stamp_in_place(tmp_path: Path) -> None:
    """Stamping one image merges into the sidecar, per image and per bucket.

    A whole-document write would drop a stamp another image's confirmation was quarantined by,
    which un-quarantines a confirmation made under a schema that has since changed.
    """
    from tcip_mcp.dataset_layout import stamp_image_status_digests

    stamp_image_status_digests(tmp_path, "bud", ["A.JPG", "B.JPG"], "digest-one")
    stamp_image_status_digests(tmp_path, "bush", ["C.JPG"], "digest-two")
    stamp_image_status_digests(tmp_path, "bud", ["B.JPG"], "digest-three")

    assert tcip_store.read(image_status_digest_key(tmp_path)) == {
        "bush": {"C.JPG": "digest-two"},
        "bud": {"A.JPG": "digest-one", "B.JPG": "digest-three"},
    }
