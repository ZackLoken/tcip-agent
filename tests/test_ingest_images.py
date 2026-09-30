"""Tests for ingestion: the workspace resolver + ``ingest_images`` tool.

All tests use synthetic temp fixtures under the test's own ``TCIP_WORKSPACE``, never the
operator's real workspace.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tcip_mcp import dataset_layout, workspace
from tcip_mcp.tools.ingest_tools import ingest_images


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


def _make_raster(path: Path, **metadata: str) -> None:
    """Write a tiny GeoTIFF carrying ``metadata`` in GDAL's default metadata domain, where a
    stitching engine writes an orthomosaic's own capture date."""
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(str(path), "w", driver="GTiff", width=8, height=8,
                       count=3, dtype="uint8") as ds:
        ds.update_tags(**metadata)


def _make_tagged_tiff(path: Path, datetime_tag: str) -> None:
    """Write a tiny TIFF whose DateTime tag (306) states ``datetime_tag``, the standard tag rather
    than any one engine's metadata item."""
    import numpy as np
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), np.zeros((8, 8, 3), np.uint8),
                     extratags=[(306, "s", 0, datetime_tag, True)])


# ── workspace resolver ──────────────────────────────────────────────────


def test_the_workspace_from_the_environment_is_the_one_tcip_workspace_names():
    assert workspace.workspace_from_environment() == Path(os.environ["TCIP_WORKSPACE"]).resolve()


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


def test_ingest_exif_buckets_and_undated(project, tmp_path):
    src = tmp_path / "raw"
    _make_image(src / "a.jpg", exif_date="2026:02:11 10:30:00")
    _make_image(src / "b.jpg", exif_date="2026:02:11 11:00:00")
    _make_image(src / "c.jpg", exif_date="2026:03:01 09:00:00")
    _make_image(src / "no_exif.png")  # no EXIF → undated

    manifest = ingest_images(project, source=str(src))

    assert "error" not in manifest
    assert manifest["buckets"] == {"2026-02-11": 2, "2026-03-01": 1}
    assert manifest["undated"] == 1
    assert manifest["total"] == 4
    assert manifest["copied"] == 4
    assert manifest["moved"] == 0
    assert Path(manifest["image_root"]) == project / "images"

    assert (project / "images" / "2026-02-11" / "a.jpg").is_file()
    assert (project / "images" / "2026-03-01" / "c.jpg").is_file()
    assert (project / "images" / "undated" / "no_exif.png").is_file()


def test_ingested_bytes_read_back_through_the_image_key(project, tmp_path):
    """The writer is the real ingest_images tool. The reader is tcip_store's blob read through
    dataset_layout.image_key(root, date, stem, ext), which refuses a falsy date, so this uses a
    dated capture: the bytes the store hands back through the key are byte-identical to the
    source file ingest copied from, not merely present."""
    import tcip_store as ts

    src = tmp_path / "raw"
    _make_image(src / "a.jpg", exif_date="2026:02:11 10:30:00")

    manifest = ingest_images(project, source=str(src))
    assert "error" not in manifest

    key = dataset_layout.image_key(project, "2026-02-11", "a", ".jpg")
    stored = ts.read_blob_versioned(key).value
    assert stored == (src / "a.jpg").read_bytes()


def test_ingest_buckets_a_raster_by_the_capture_date_its_metadata_states(project, tmp_path):
    """A raster states its capture date in raster metadata, not in EXIF: an orthomosaic lands in a
    real date bucket the same as a photo does."""
    src = tmp_path / "raw"
    _make_raster(src / "mosaic.tif", capture_date="2025-09-16")

    manifest = ingest_images(project, source=str(src))

    assert manifest["buckets"] == {"2025-09-16": 1}
    assert manifest["undated"] == 0
    assert manifest["unreadable_dates"] == []
    assert (project / "images" / "2025-09-16" / "mosaic.tif").is_file()


def test_ingest_buckets_a_tiff_by_its_own_datetime_tag(project, tmp_path):
    src = tmp_path / "raw"
    _make_tagged_tiff(src / "plot.tif", "2026:04:05 08:00:00")

    manifest = ingest_images(project, source=str(src))

    assert manifest["buckets"] == {"2026-04-05": 1}
    assert manifest["unreadable_dates"] == []


def test_ingest_leaves_a_raster_that_states_no_date_undated_and_unreported(project, tmp_path):
    """Read, and it says nothing about when it was captured: a fact, not a failure."""
    src = tmp_path / "raw"
    _make_raster(src / "plain.tif")

    manifest = ingest_images(project, source=str(src))

    assert manifest["undated"] == 1
    assert manifest["buckets"] == {}
    assert manifest["unreadable_dates"] == []


def test_ingest_leaves_a_photo_without_an_exif_date_undated_and_unreported(project, tmp_path):
    src = tmp_path / "raw"
    _make_image(src / "no_exif.png")

    manifest = ingest_images(project, source=str(src))

    assert manifest["undated"] == 1
    assert manifest["unreadable_dates"] == []


@pytest.mark.parametrize("name", ["broken.jpg", "broken.tif"])
def test_ingest_still_ingests_a_file_whose_capture_date_cannot_be_read(project, tmp_path, name):
    """The probe never gates ingestion: an unreadable container is copied and counted like any
    other, buckets undated, and is named in the report so the difference from a file that simply
    states no date stays visible."""
    src = tmp_path / "raw"
    src.mkdir(parents=True)
    (src / name).write_bytes(b"this is not an image at all")

    manifest = ingest_images(project, source=str(src))

    assert manifest["copied"] == 1
    assert manifest["total"] == 1
    assert manifest["undated"] == 1
    assert manifest["errors"] == []
    assert (project / "images" / "undated" / name).is_file()

    assert len(manifest["unreadable_dates"]) == 1
    entry = manifest["unreadable_dates"][0]
    assert entry["source"].endswith(name)
    assert entry["bucket"] == "undated"
    assert entry["reason"]


def test_ingest_reports_an_exif_date_it_cannot_read_as_a_date(project, tmp_path):
    src = tmp_path / "raw"
    _make_image(src / "odd.jpg", exif_date="whenever")

    manifest = ingest_images(project, source=str(src))

    assert manifest["undated"] == 1
    assert len(manifest["unreadable_dates"]) == 1
    assert "whenever" in manifest["unreadable_dates"][0]["reason"]


@pytest.mark.parametrize("date_from", ["none", "2026-05-15"])
def test_a_date_mode_that_names_its_own_bucket_opens_no_file(project, tmp_path, monkeypatch,
                                                             date_from):
    """Only the per-file mode reads a file; the modes that name the bucket from the caller's own
    argument never open one, so an unreadable file costs nothing and reports nothing."""
    src = tmp_path / "raw"
    _make_image(src / "a.jpg", exif_date="2026:02:11 10:30:00")
    (src / "broken.tif").write_bytes(b"this is not an image at all")

    import PIL.Image
    import rasterio

    opened: list[str] = []
    monkeypatch.setattr(PIL.Image, "open", lambda *a, **k: opened.append("pil"))
    monkeypatch.setattr(rasterio, "open", lambda *a, **k: opened.append("gdal"))

    manifest = ingest_images(project, source=str(src), date_from=date_from)

    assert opened == []
    assert manifest["copied"] == 2
    assert manifest["unreadable_dates"] == []


def test_ingest_copies_leave_originals_byte_identical(project, tmp_path):
    src = tmp_path / "raw"
    _make_image(src / "a.jpg", exif_date="2026:02:11 10:30:00")
    before = (src / "a.jpg").read_bytes()

    ingest_images(project, source=str(src))

    assert (src / "a.jpg").is_file()  # original still there
    assert (src / "a.jpg").read_bytes() == before  # byte-identical
    dest = project / "images" / "2026-02-11" / "a.jpg"
    assert dest.read_bytes() == before  # exact copy (EXIF preserved)


def test_ingest_move_removes_originals(project, tmp_path):
    src = tmp_path / "raw"
    _make_image(src / "a.jpg", exif_date="2026:02:11 10:30:00")

    manifest = ingest_images(project, source=str(src), copy=False)

    assert manifest["moved"] == 1
    assert manifest["move"] is True
    assert not (src / "a.jpg").exists()  # original moved away
    assert (project / "images" / "2026-02-11" / "a.jpg").is_file()


def test_ingest_two_sources_with_one_stem_refuse_the_whole_call(project, tmp_path):
    # Two source subfolders each holding dup.png (no EXIF → both target undated/dup.png).
    src = tmp_path / "raw"
    _make_image(src / "sub1" / "dup.png")
    _make_image(src / "sub2" / "dup.png")

    manifest = ingest_images(project, source=str(src))

    assert "error" in manifest
    assert "dup.png" in manifest["error"]
    assert "undated" in manifest["error"]
    assert not (project / "images").exists()


def test_ingest_same_stem_in_two_different_buckets_is_admitted(project, tmp_path):
    """A rail must admit valid work: the collision key is scoped per bucket, so foo.jpg into one
    date and foo.png into another land side by side, no collision."""
    src = tmp_path / "raw"
    _make_image(src / "foo.jpg", exif_date="2026:02:11 10:00:00")
    _make_image(src / "foo.png", exif_date="2026:03:01 10:00:00")

    manifest = ingest_images(project, source=str(src))

    assert "error" not in manifest
    assert (project / "images" / "2026-02-11" / "foo.jpg").is_file()
    assert (project / "images" / "2026-03-01" / "foo.png").is_file()


def test_ingest_refuses_a_stem_reserved_for_a_bucket_provenance_stamp(project, tmp_path):
    """An image whose stem names a prediction bucket's own provenance stamp would produce a
    label file no bucket walk can tell apart from that stamp; it is reported and not placed."""
    src = tmp_path / "raw"
    _make_image(src / "operating_point.png")
    _make_image(src / "ordinary.png")

    manifest = ingest_images(project, source=str(src))

    assert manifest["undated"] == 1
    assert len(manifest["reserved_name_skips"]) == 1
    assert manifest["reserved_name_skips"][0]["stem"] == "operating_point"
    assert not (project / "images" / "undated" / "operating_point.png").is_file()
    assert (project / "images" / "undated" / "ordinary.png").is_file()


def test_ingest_refuses_a_case_variant_of_a_reserved_stem(project, tmp_path):
    """The reserved-name check is case-insensitive: a source stem differing only in case from a
    bucket's own provenance stamp would still collide with it on a case-insensitive filesystem."""
    src = tmp_path / "raw"
    _make_image(src / "Operating_Point.png")
    _make_image(src / "ordinary.png")

    manifest = ingest_images(project, source=str(src))

    assert len(manifest["reserved_name_skips"]) == 1
    assert manifest["reserved_name_skips"][0]["stem"] == "Operating_Point"
    assert not (project / "images" / "undated" / "Operating_Point.png").is_file()
    assert (project / "images" / "undated" / "ordinary.png").is_file()


def test_ingest_same_stem_different_ext_refuses_the_whole_call(project, tmp_path):
    # Labels pair by stem alone, so IMG_1.jpg and IMG_1.png in one bucket would share one
    # label file: neither has landed yet, so the whole call refuses rather than picking one.
    src = tmp_path / "raw"
    _make_image(src / "IMG_1.jpg", exif_date="2026:02:11 10:00:00")
    _make_image(src / "IMG_1.png")  # different ext; PNG has no EXIF → but force same bucket

    manifest = ingest_images(project, source=str(src), date_from="2026-02-11")

    assert "error" in manifest
    assert "IMG_1" in manifest["error"]
    assert not (project / "images").exists()


def test_ingest_stem_collision_across_two_calls_refuses_naming_both_files(project, tmp_path):
    """The classic collision: one call places foo.jpg, a later call offers foo.png into the
    same bucket. The bucket keeps holding foo.jpg alone; nothing from the second call lands."""
    src1 = tmp_path / "raw1"
    _make_image(src1 / "foo.jpg", exif_date="2026:02:11 10:00:00")
    first = ingest_images(project, source=str(src1))
    assert "error" not in first

    placed = project / "images" / "2026-02-11" / "foo.jpg"
    original_bytes = placed.read_bytes()

    src2 = tmp_path / "raw2"
    _make_image(src2 / "foo.png")
    second = ingest_images(project, source=str(src2), date_from="2026-02-11")

    assert "error" in second
    assert "foo.jpg" in second["error"] and "foo.png" in second["error"]
    assert placed.is_file()
    assert not (project / "images" / "2026-02-11" / "foo.png").exists()
    assert placed.read_bytes() == original_bytes


def test_ingest_case_variant_stem_collision_across_two_calls_refuses(project, tmp_path):
    """Foo.jpg then foo.png: different exact stems, the same case-folded key."""
    src1 = tmp_path / "raw1"
    _make_image(src1 / "Foo.jpg", exif_date="2026:02:11 10:00:00")
    first = ingest_images(project, source=str(src1))
    assert "error" not in first

    placed = project / "images" / "2026-02-11" / "Foo.jpg"
    original_bytes = placed.read_bytes()

    src2 = tmp_path / "raw2"
    _make_image(src2 / "foo.png")
    second = ingest_images(project, source=str(src2), date_from="2026-02-11")

    assert "error" in second
    assert "Foo.jpg" in second["error"] and "foo.png" in second["error"]
    assert placed.is_file()
    assert not (project / "images" / "2026-02-11" / "foo.png").exists()
    assert placed.read_bytes() == original_bytes


def test_ingest_admits_a_source_after_a_band_group_manifest_with_a_different_stem(project,
                                                                                  tmp_path):
    """A rail must admit valid work: a manifest at one stem never blocks an ordinary source at a
    genuinely distinct stem in the same bucket."""
    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest

    src = tmp_path / "raw"
    _make_image(src / "plain.png")
    manifest = ingest_images(project, source=str(src))
    assert "error" not in manifest

    bucket_dir = project / "images" / "undated"
    band_a = bucket_dir / "cap_G.tif"
    band_b = bucket_dir / "cap_R.tif"
    band_a.write_bytes(b"a")
    band_b.write_bytes(b"b")
    write_band_group_manifest(bucket_dir, "cap", {"Green": band_a, "Red": band_b})

    src2 = tmp_path / "raw2"
    _make_image(src2 / "other.png")
    second = ingest_images(project, source=str(src2))
    assert "error" not in second
    assert (bucket_dir / "other.png").is_file()


def test_ingest_after_a_band_group_manifest_refuses_naming_the_manifest(project, tmp_path):
    """cap.jpg after a cap.bandgroup manifest: the door refuses rather than admitting a source
    that would mint the manifest-versus-raw ambiguity every reader already refuses."""
    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest

    src = tmp_path / "raw"
    _make_image(src / "plain.png")
    first = ingest_images(project, source=str(src))
    assert "error" not in first

    bucket_dir = project / "images" / "undated"
    band_a = bucket_dir / "cap_G.tif"
    band_b = bucket_dir / "cap_R.tif"
    band_a.write_bytes(b"a")
    band_b.write_bytes(b"b")
    write_band_group_manifest(bucket_dir, "cap", {"Green": band_a, "Red": band_b})

    src2 = tmp_path / "raw2"
    _make_image(src2 / "cap.jpg")
    second = ingest_images(project, source=str(src2))

    assert "error" in second
    assert "cap.bandgroup" in second["error"]
    assert not (bucket_dir / "cap.jpg").exists()


def test_ingest_survives_a_bad_file_mid_batch(project, tmp_path, monkeypatch):
    src = tmp_path / "raw"
    _make_image(src / "good1.png")
    _make_image(src / "bad.png")
    _make_image(src / "good2.png")

    import tcip_mcp.tools.ingest_tools as it

    real_put = it.store.put_blob

    def flaky_put(key, data, **kwargs):
        if Path(key.parts[-1]).stem == "bad":
            raise OSError("simulated locked file")
        return real_put(key, data, **kwargs)

    monkeypatch.setattr(it.store, "put_blob", flaky_put)

    manifest = ingest_images(project, source=str(src))

    assert manifest["copied"] == 2  # the two good files still landed
    assert len(manifest["errors"]) == 1
    assert manifest["errors"][0]["source"].endswith("bad.png")


def test_ingest_reingest_is_idempotent_via_collision_skip(project, tmp_path):
    src = tmp_path / "raw"
    _make_image(src / "a.jpg", exif_date="2026:02:11 10:30:00")
    ingest_images(project, source=str(src))
    # Second run: dest already exists → skipped, nothing copied.
    second = ingest_images(project, source=str(src))
    assert second["copied"] == 0
    assert len(second["skipped_collisions"]) == 1


def test_ingest_date_from_none_all_undated(project, tmp_path):
    src = tmp_path / "raw"
    _make_image(src / "a.jpg", exif_date="2026:02:11 10:30:00")  # has EXIF but ignored
    _make_image(src / "b.png")

    manifest = ingest_images(project, source=str(src), date_from="none")

    assert manifest["undated"] == 2
    assert manifest["buckets"] == {}


def test_ingest_date_from_literal_bucket(project, tmp_path):
    src = tmp_path / "raw"
    _make_image(src / "a.png")
    _make_image(src / "b.png")

    manifest = ingest_images(project, source=str(src), date_from="2026-05-15")

    assert manifest["buckets"] == {"2026-05-15": 2}
    assert manifest["undated"] == 0
    assert (project / "images" / "2026-05-15" / "a.png").is_file()


def test_ingest_literal_bucket_rejects_traversal(project, tmp_path):
    src = tmp_path / "raw"
    _make_image(src / "a.png")
    manifest = ingest_images(project, source=str(src), date_from="../escape")
    assert "error" in manifest


def test_ingest_no_images_found(project, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    manifest = ingest_images(project, source=str(empty))
    assert "error" in manifest
    assert not (project / "images").exists()


def test_ingest_non_recursive_skips_subfolders(project, tmp_path):
    src = tmp_path / "raw"
    _make_image(src / "top.png")
    _make_image(src / "sub" / "deep.png")

    manifest = ingest_images(project, source=str(src), recursive=False)

    assert manifest["total"] == 1  # only top.png


def test_ingest_writes_audit_entry(project, tmp_path):
    import tcip_store as ts
    import tcip_mcp.audit as audit_mod

    src = tmp_path / "raw"
    _make_image(src / "a.png")
    ingest_images(project, source=str(src))

    entries = ts.read_log(audit_mod.audit_log_key(project)).records
    assert any(e["tool"] == "ingest_images" and e["status"] == "ok" for e in entries)


def test_inspect_project_counts_canonical_images(project, tmp_path):
    from tcip_mcp.tools.project_tools import inspect_project

    src = tmp_path / "raw"
    _make_image(src / "a.jpg", exif_date="2026:02:11 10:30:00")
    _make_image(src / "b.png")
    ingest_images(project, source=str(src))

    status = inspect_project(project)
    assert status["record_problem"] is None
    assert status["image_count"] == 2
    assert "2026-02-11" in status["dates"]
    assert "undated" in status["dates"]


def test_ingest_after_import_admits_a_second_date(tmp_path):
    """The import door adopts a fresh root itself when the process is bound to the database
    backend, so a project it lands is usable at once: no operator ``tcip adopt-store``
    run sits between ``import_project`` and ``ingest_images``. Import, ingest is the admit case.
    Bound to the database backend explicitly, since that is what the door's own adoption step
    is conditional on."""
    import tcip_store
    from tcip_store.sqlite_backend import SqliteBackend
    from tcip_store.store import _backend

    from tcip_mcp.project_record import read_record
    from tcip_mcp.tools.project_tools import archive_project, import_project
    from tests._web_fixtures import new_project

    previous = _backend()
    backend = SqliteBackend()
    tcip_store.bind(backend)
    dest = tmp_path.parent / "reopened"
    try:
        project = new_project(tmp_path)
        src1 = tmp_path / "raw1"
        _make_image(src1 / "a.jpg", exif_date="2026:02:11 10:30:00")
        first = ingest_images(project, source=str(src1))
        assert "error" not in first

        zip_path = tmp_path.parent / "export.zip"
        exported = archive_project(project, str(zip_path))
        assert "error" not in exported

        imported = import_project(str(zip_path), str(dest))
        assert "error" not in imported
        assert imported["database_built"] is True

        src2 = tmp_path / "raw2"
        _make_image(src2 / "b.jpg", exif_date="2026:03:01 10:30:00")
        second = ingest_images(dest, source=str(src2))
        assert "error" not in second
        assert read_record(dest)["id"] == read_record(project)["id"]
    finally:
        tcip_store.bind(previous)
        backend.close()

    assert (dest / "images" / "2026-03-01" / "b.jpg").is_file()
