"""Tests for WS-A ingestion: the workspace resolver + ``ingest_images`` tool.

All tests use synthetic temp fixtures and an isolated ``TCIP_WORKSPACE`` — they must
never touch the human's real ``~/tcip-projects/`` (the manual end-to-end target).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_mcp import dataset_layout, workspace
from tcip_mcp.tools.ingest_tools import ingest_images


@pytest.fixture(autouse=True)
def _isolate_workspace(tmp_path_factory, monkeypatch):
    """Point TCIP_WORKSPACE at a throwaway dir so no test reaches the real workspace."""
    ws = tmp_path_factory.mktemp("workspace")
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    return ws


def _make_image(path: Path, exif_date: str | None = None) -> None:
    """Write a tiny image; when ``exif_date`` ('YYYY:MM:DD HH:MM:SS') is set, embed it
    as EXIF DateTimeOriginal so the ingest EXIF reader round-trips it."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (32, 24), (100, 110, 120))
    if exif_date:
        exif = Image.Exif()
        exif[0x8769] = {0x9003: exif_date}  # Exif sub-IFD → DateTimeOriginal
        img.save(path, exif=exif)
    else:
        img.save(path)


# ── workspace resolver ──────────────────────────────────────────────────


def test_workspace_root_honours_env(_isolate_workspace):
    assert workspace.workspace_root() == _isolate_workspace.resolve()


def test_project_path_rejects_traversal():
    for bad in ("../escape", "a/b", "a\\b", "", ".", ".."):
        with pytest.raises(ValueError):
            workspace.project_path(bad)


def test_project_path_under_workspace(_isolate_workspace):
    p = workspace.project_path("hazelnut_catkin_valley-farm")
    assert p.parent == _isolate_workspace.resolve()
    assert p.name == "hazelnut_catkin_valley-farm"


def test_active_marker_round_trip(_isolate_workspace):
    assert workspace.read_active_project() is None
    workspace.set_active_project("hazelnut_catkin_valley-farm")
    assert workspace.read_active_project() == "hazelnut_catkin_valley-farm"
    # Second writer wins cleanly (no torn file).
    workspace.set_active_project("chestnut_burr_site-b")
    assert workspace.read_active_project() == "chestnut_burr_site-b"


def test_set_active_project_rejects_bad_name():
    with pytest.raises(ValueError):
        workspace.set_active_project("../evil")


# ── dataset_layout builders ─────────────────────────────────────────────


def test_image_path_builders(tmp_path):
    assert dataset_layout.image_dir(tmp_path, None) == tmp_path / "images"
    assert dataset_layout.image_dir(tmp_path, "2026-02-11") == tmp_path / "images" / "2026-02-11"
    assert (
        dataset_layout.image_path(tmp_path, "2026-02-11", "IMG_1", ".jpg")
        == tmp_path / "images" / "2026-02-11" / "IMG_1.jpg"
    )


def test_list_dates(tmp_path):
    (tmp_path / "images" / "2026-02-11").mkdir(parents=True)
    (tmp_path / "images" / "2026-03-01").mkdir(parents=True)
    dates = dataset_layout.list_dates(tmp_path)
    assert dates == ["2026-02-11", "2026-03-01"]


# ── ingest_images ───────────────────────────────────────────────────────


def test_ingest_exif_buckets_and_undated(tmp_path):
    src = tmp_path / "raw"
    _make_image(src / "a.jpg", exif_date="2026:02:11 10:30:00")
    _make_image(src / "b.jpg", exif_date="2026:02:11 11:00:00")
    _make_image(src / "c.jpg", exif_date="2026:03:01 09:00:00")
    _make_image(src / "no_exif.png")  # no EXIF → undated

    manifest = ingest_images(source=str(src), name="hazelnut_catkin_valley-farm")

    assert "error" not in manifest
    assert manifest["buckets"] == {"2026-02-11": 2, "2026-03-01": 1}
    assert manifest["undated"] == 1
    assert manifest["total"] == 4
    assert manifest["copied"] == 4
    assert manifest["moved"] == 0

    proj = Path(manifest["project_path"])
    assert (proj / "images" / "2026-02-11" / "a.jpg").is_file()
    assert (proj / "images" / "2026-03-01" / "c.jpg").is_file()
    assert (proj / "images" / "undated" / "no_exif.png").is_file()
    # .tcip scaffolding created (the sessions/ event log was retired with its tools)
    assert (proj / ".tcip" / "config.toml").is_file()


def test_ingest_copies_leave_originals_byte_identical(tmp_path):
    src = tmp_path / "raw"
    _make_image(src / "a.jpg", exif_date="2026:02:11 10:30:00")
    before = (src / "a.jpg").read_bytes()

    manifest = ingest_images(source=str(src), name="proj_copy")

    assert (src / "a.jpg").is_file()  # original still there
    assert (src / "a.jpg").read_bytes() == before  # byte-identical
    dest = Path(manifest["project_path"]) / "images" / "2026-02-11" / "a.jpg"
    assert dest.read_bytes() == before  # exact copy (EXIF preserved)


def test_ingest_move_removes_originals(tmp_path):
    src = tmp_path / "raw"
    _make_image(src / "a.jpg", exif_date="2026:02:11 10:30:00")

    manifest = ingest_images(source=str(src), name="proj_move", copy=False)

    assert manifest["moved"] == 1
    assert manifest["move"] is True
    assert not (src / "a.jpg").exists()  # original moved away
    assert (Path(manifest["project_path"]) / "images" / "2026-02-11" / "a.jpg").is_file()


def test_ingest_no_overwrite_reports_stem_collision(tmp_path):
    # Two source subfolders each holding dup.png (no EXIF → both target undated/dup.png).
    src = tmp_path / "raw"
    _make_image(src / "sub1" / "dup.png")
    _make_image(src / "sub2" / "dup.png")

    manifest = ingest_images(source=str(src), name="proj_collision")

    assert manifest["undated"] == 1  # exactly one placed
    assert len(manifest["skipped_collisions"]) == 1
    coll = manifest["skipped_collisions"][0]
    assert coll["stem"] == "dup"
    assert coll["bucket"] == "undated"
    # The placed file is intact.
    assert (Path(manifest["project_path"]) / "images" / "undated" / "dup.png").is_file()


def test_ingest_same_stem_different_ext_is_a_collision(tmp_path):
    # Labels pair by stem alone, so IMG_1.jpg and IMG_1.tif in one bucket must not both
    # land (they'd share one label file). One placed, the other reported.
    src = tmp_path / "raw"
    _make_image(src / "IMG_1.jpg", exif_date="2026:02:11 10:00:00")
    _make_image(src / "IMG_1.png")  # different ext; PNG has no EXIF → but force same bucket

    manifest = ingest_images(source=str(src), name="proj_stem", date_from="2026-02-11")

    assert manifest["copied"] == 1
    assert len(manifest["skipped_collisions"]) == 1
    assert manifest["skipped_collisions"][0]["stem"] == "IMG_1"


def test_ingest_survives_a_bad_file_mid_batch(tmp_path, monkeypatch):
    src = tmp_path / "raw"
    _make_image(src / "good1.png")
    _make_image(src / "bad.png")
    _make_image(src / "good2.png")

    import tcip_mcp.tools.ingest_tools as it

    real_write = it.atomic_write_bytes

    def flaky_write(path, data):
        if Path(path).stem == "bad":
            raise OSError("simulated locked file")
        return real_write(path, data)

    monkeypatch.setattr(it, "atomic_write_bytes", flaky_write)

    manifest = ingest_images(source=str(src), name="proj_resilient")

    assert manifest["copied"] == 2  # the two good files still landed
    assert len(manifest["errors"]) == 1
    assert manifest["errors"][0]["source"].endswith("bad.png")


def test_ingest_reingest_is_idempotent_via_collision_skip(tmp_path):
    src = tmp_path / "raw"
    _make_image(src / "a.jpg", exif_date="2026:02:11 10:30:00")
    ingest_images(source=str(src), name="proj_reingest")
    # Second run: dest already exists → skipped, nothing copied.
    second = ingest_images(source=str(src), name="proj_reingest")
    assert second["copied"] == 0
    assert len(second["skipped_collisions"]) == 1


def test_ingest_date_from_none_all_undated(tmp_path):
    src = tmp_path / "raw"
    _make_image(src / "a.jpg", exif_date="2026:02:11 10:30:00")  # has EXIF but ignored
    _make_image(src / "b.png")

    manifest = ingest_images(source=str(src), name="proj_none", date_from="none")

    assert manifest["undated"] == 2
    assert manifest["buckets"] == {}


def test_ingest_date_from_literal_bucket(tmp_path):
    src = tmp_path / "raw"
    _make_image(src / "a.png")
    _make_image(src / "b.png")

    manifest = ingest_images(source=str(src), name="proj_literal", date_from="2026-05-15")

    assert manifest["buckets"] == {"2026-05-15": 2}
    assert manifest["undated"] == 0
    assert (Path(manifest["project_path"]) / "images" / "2026-05-15" / "a.png").is_file()


def test_ingest_literal_bucket_rejects_traversal(tmp_path):
    src = tmp_path / "raw"
    _make_image(src / "a.png")
    manifest = ingest_images(source=str(src), name="proj_bad", date_from="../escape")
    assert "error" in manifest


def test_ingest_no_images_found(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    manifest = ingest_images(source=str(empty), name="proj_empty")
    assert "error" in manifest
    # No project should be scaffolded when there's nothing to ingest.
    assert not (workspace.project_path("proj_empty")).exists()


def test_ingest_explicit_project_path_overrides_workspace(tmp_path):
    src = tmp_path / "raw"
    _make_image(src / "a.png")
    dest = tmp_path / "custom_dest"

    manifest = ingest_images(source=str(src), name="ignored", project_path=str(dest))

    assert Path(manifest["project_path"]) == dest
    assert (dest / "images" / "undated" / "a.png").is_file()


def test_ingest_non_recursive_skips_subfolders(tmp_path):
    src = tmp_path / "raw"
    _make_image(src / "top.png")
    _make_image(src / "sub" / "deep.png")

    manifest = ingest_images(source=str(src), name="proj_flat", recursive=False)

    assert manifest["total"] == 1  # only top.png


def test_ingest_writes_audit_entry(tmp_path, monkeypatch):
    import tcip_mcp.audit as audit_mod

    audit_path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(audit_mod, "AUDIT_PATH", audit_path)

    src = tmp_path / "raw"
    _make_image(src / "a.png")
    ingest_images(source=str(src), name="proj_audit")

    assert audit_path.is_file()
    lines = audit_path.read_text().strip().splitlines()
    import json

    entries = [json.loads(ln) for ln in lines]
    assert any(e["tool"] == "ingest_images" and e["status"] == "ok" for e in entries)


def test_inspect_project_counts_canonical_images(tmp_path):
    from tcip_mcp.tools.project_tools import inspect_project

    src = tmp_path / "raw"
    _make_image(src / "a.jpg", exif_date="2026:02:11 10:30:00")
    _make_image(src / "b.png")
    manifest = ingest_images(source=str(src), name="proj_status")
    proj = manifest["project_path"]

    status = inspect_project(proj)
    assert status["initialized"] is True
    assert status["image_count"] == 2
    assert "2026-02-11" in status["dates"]
    assert "undated" in status["dates"]
