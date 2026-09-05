"""Tests for ingestion: the workspace resolver + ``ingest_images`` tool.

All tests use synthetic temp fixtures and an isolated ``TCIP_WORKSPACE``. They must
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


def test_workspace_root_honours_env(_isolate_workspace):
    assert workspace.workspace_root() == _isolate_workspace.resolve()


def test_project_path_rejects_traversal():
    for bad in ("../escape", "a/b", "a\\b", "", ".", ".."):
        with pytest.raises(ValueError):
            workspace.project_path(bad)


def test_project_path_under_workspace(_isolate_workspace):
    p = workspace.project_path("currant_bud_valley-farm")
    assert p.parent == _isolate_workspace.resolve()
    assert p.name == "currant_bud_valley-farm"


def test_active_marker_round_trip(_isolate_workspace):
    (workspace.project_path("currant_bud_valley-farm") / ".tcip").mkdir(parents=True)
    (workspace.project_path("chestnut_burr_site-b") / ".tcip").mkdir(parents=True)

    assert workspace.read_active_project() is None
    workspace.activate_project("currant_bud_valley-farm")
    assert workspace.read_active_project() == "currant_bud_valley-farm"
    # Second writer wins cleanly (no torn file).
    workspace.activate_project("chestnut_burr_site-b")
    assert workspace.read_active_project() == "chestnut_burr_site-b"


def test_activate_project_rejects_bad_name():
    with pytest.raises(ValueError):
        workspace.activate_project("../evil")


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

    manifest = ingest_images(source=str(src), name="currant_bud_valley-farm", site="north orchard")

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
    assert (proj / ".tcip" / "artifacts").is_dir()
    assert (proj / ".tcip" / "models").is_dir()


def test_ingested_bytes_read_back_through_the_image_key(tmp_path):
    """The writer is the real ingest_images tool, driven the same way
    test_ingest_exif_buckets_and_undated drives it. The reader is tcip_store's blob read through
    dataset_layout.image_key(root, date, stem, ext), which refuses a falsy date, so this uses a
    dated capture. The existing test only asserts the copied file is_file(); this asserts the
    bytes the store hands back through the key are byte-identical to the source file ingest
    copied from, not merely present."""
    import tcip_store as ts

    src = tmp_path / "raw"
    _make_image(src / "a.jpg", exif_date="2026:02:11 10:30:00")

    manifest = ingest_images(source=str(src), name="currant_bud_valley-farm", site="north orchard")
    assert "error" not in manifest

    proj = Path(manifest["project_path"])
    key = dataset_layout.image_key(proj, "2026-02-11", "a", ".jpg")
    stored = ts.read_blob_versioned(key).value
    assert stored == (src / "a.jpg").read_bytes()


def test_ingest_buckets_a_raster_by_the_capture_date_its_metadata_states(tmp_path):
    """A raster states its capture date in raster metadata, not in EXIF: an orthomosaic lands in a
    real date bucket the same as a photo does."""
    src = tmp_path / "raw"
    _make_raster(src / "mosaic.tif", capture_date="2025-09-16")

    manifest = ingest_images(source=str(src), name="proj_raster_date", site="north orchard")

    assert manifest["buckets"] == {"2025-09-16": 1}
    assert manifest["undated"] == 0
    assert manifest["unreadable_dates"] == []
    assert (Path(manifest["project_path"]) / "images" / "2025-09-16" / "mosaic.tif").is_file()


def test_ingest_buckets_a_tiff_by_its_own_datetime_tag(tmp_path):
    src = tmp_path / "raw"
    _make_tagged_tiff(src / "plot.tif", "2026:04:05 08:00:00")

    manifest = ingest_images(source=str(src), name="proj_tiff_tag", site="north orchard")

    assert manifest["buckets"] == {"2026-04-05": 1}
    assert manifest["unreadable_dates"] == []


def test_ingest_leaves_a_raster_that_states_no_date_undated_and_unreported(tmp_path):
    """Read, and it says nothing about when it was captured: a fact, not a failure."""
    src = tmp_path / "raw"
    _make_raster(src / "plain.tif")

    manifest = ingest_images(source=str(src), name="proj_raster_undated", site="north orchard")

    assert manifest["undated"] == 1
    assert manifest["buckets"] == {}
    assert manifest["unreadable_dates"] == []


def test_ingest_leaves_a_photo_without_an_exif_date_undated_and_unreported(tmp_path):
    src = tmp_path / "raw"
    _make_image(src / "no_exif.png")

    manifest = ingest_images(source=str(src), name="proj_photo_undated", site="north orchard")

    assert manifest["undated"] == 1
    assert manifest["unreadable_dates"] == []


@pytest.mark.parametrize("name", ["broken.jpg", "broken.tif"])
def test_ingest_still_ingests_a_file_whose_capture_date_cannot_be_read(tmp_path, name):
    """The probe never gates ingestion: an unreadable container is copied and counted like any
    other, buckets undated, and is named in the report so the difference from a file that simply
    states no date stays visible."""
    src = tmp_path / "raw"
    src.mkdir(parents=True)
    (src / name).write_bytes(b"this is not an image at all")

    manifest = ingest_images(source=str(src), name="proj_unreadable_case", site="north orchard")

    assert manifest["copied"] == 1
    assert manifest["total"] == 1
    assert manifest["undated"] == 1
    assert manifest["errors"] == []
    assert (Path(manifest["project_path"]) / "images" / "undated" / name).is_file()

    assert len(manifest["unreadable_dates"]) == 1
    entry = manifest["unreadable_dates"][0]
    assert entry["source"].endswith(name)
    assert entry["bucket"] == "undated"
    assert entry["reason"]


def test_ingest_reports_an_exif_date_it_cannot_read_as_a_date(tmp_path):
    src = tmp_path / "raw"
    _make_image(src / "odd.jpg", exif_date="whenever")

    manifest = ingest_images(source=str(src), name="proj_odd_exif", site="north orchard")

    assert manifest["undated"] == 1
    assert len(manifest["unreadable_dates"]) == 1
    assert "whenever" in manifest["unreadable_dates"][0]["reason"]


@pytest.mark.parametrize("date_from", ["none", "2026-05-15"])
def test_a_date_mode_that_names_its_own_bucket_opens_no_file(tmp_path, monkeypatch, date_from):
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

    manifest = ingest_images(source=str(src), name="proj_no_open", site="north orchard", date_from=date_from)

    assert opened == []
    assert manifest["copied"] == 2
    assert manifest["unreadable_dates"] == []


def test_ingest_copies_leave_originals_byte_identical(tmp_path):
    src = tmp_path / "raw"
    _make_image(src / "a.jpg", exif_date="2026:02:11 10:30:00")
    before = (src / "a.jpg").read_bytes()

    manifest = ingest_images(source=str(src), name="proj_copy_mode", site="north orchard")

    assert (src / "a.jpg").is_file()  # original still there
    assert (src / "a.jpg").read_bytes() == before  # byte-identical
    dest = Path(manifest["project_path"]) / "images" / "2026-02-11" / "a.jpg"
    assert dest.read_bytes() == before  # exact copy (EXIF preserved)


def test_ingest_move_removes_originals(tmp_path):
    src = tmp_path / "raw"
    _make_image(src / "a.jpg", exif_date="2026:02:11 10:30:00")

    manifest = ingest_images(source=str(src), name="proj_move_mode", site="north orchard", copy=False)

    assert manifest["moved"] == 1
    assert manifest["move"] is True
    assert not (src / "a.jpg").exists()  # original moved away
    assert (Path(manifest["project_path"]) / "images" / "2026-02-11" / "a.jpg").is_file()


def test_ingest_two_sources_with_one_stem_refuse_the_whole_call(tmp_path):
    # Two source subfolders each holding dup.png (no EXIF → both target undated/dup.png).
    src = tmp_path / "raw"
    _make_image(src / "sub1" / "dup.png")
    _make_image(src / "sub2" / "dup.png")

    manifest = ingest_images(source=str(src), name="proj_collision_case", site="north orchard")

    assert "error" in manifest
    assert "dup.png" in manifest["error"]
    assert "undated" in manifest["error"]
    assert not workspace.project_path("proj_collision_case").exists()


def test_ingest_same_stem_in_two_different_buckets_is_admitted(tmp_path):
    """A rail must admit valid work: the collision key is scoped per bucket, so foo.jpg into one
    date and foo.png into another land side by side, no collision."""
    src = tmp_path / "raw"
    _make_image(src / "foo.jpg", exif_date="2026:02:11 10:00:00")
    _make_image(src / "foo.png", exif_date="2026:03:01 10:00:00")

    manifest = ingest_images(source=str(src), name="proj_twobuckets_case", site="north orchard")

    assert "error" not in manifest
    proj = Path(manifest["project_path"])
    assert (proj / "images" / "2026-02-11" / "foo.jpg").is_file()
    assert (proj / "images" / "2026-03-01" / "foo.png").is_file()


def test_ingest_refuses_a_stem_reserved_for_a_bucket_provenance_stamp(tmp_path):
    """An image whose stem names a prediction bucket's own provenance stamp would produce a
    label file no bucket walk can tell apart from that stamp; it is reported and not placed."""
    src = tmp_path / "raw"
    _make_image(src / "operating_point.png")
    _make_image(src / "ordinary.png")

    manifest = ingest_images(source=str(src), name="proj_reserved_case", site="north orchard")

    assert manifest["undated"] == 1
    assert len(manifest["reserved_name_skips"]) == 1
    assert manifest["reserved_name_skips"][0]["stem"] == "operating_point"
    assert not (Path(manifest["project_path"]) / "images" / "undated" / "operating_point.png").is_file()
    assert (Path(manifest["project_path"]) / "images" / "undated" / "ordinary.png").is_file()


def test_ingest_refuses_a_case_variant_of_a_reserved_stem(tmp_path):
    """The reserved-name check is case-insensitive: a source stem differing only in case from a
    bucket's own provenance stamp would still collide with it on a case-insensitive filesystem."""
    src = tmp_path / "raw"
    _make_image(src / "Operating_Point.png")
    _make_image(src / "ordinary.png")

    manifest = ingest_images(source=str(src), name="proj_reserved_variant", site="north orchard")

    assert len(manifest["reserved_name_skips"]) == 1
    assert manifest["reserved_name_skips"][0]["stem"] == "Operating_Point"
    assert not (Path(manifest["project_path"]) / "images" / "undated" / "Operating_Point.png").is_file()
    assert (Path(manifest["project_path"]) / "images" / "undated" / "ordinary.png").is_file()


def test_ingest_same_stem_different_ext_refuses_the_whole_call(tmp_path):
    # Labels pair by stem alone, so IMG_1.jpg and IMG_1.png in one bucket would share one
    # label file: neither has landed yet, so the whole call refuses rather than picking one.
    src = tmp_path / "raw"
    _make_image(src / "IMG_1.jpg", exif_date="2026:02:11 10:00:00")
    _make_image(src / "IMG_1.png")  # different ext; PNG has no EXIF → but force same bucket

    manifest = ingest_images(source=str(src), name="proj_stem_case", site="north orchard", date_from="2026-02-11")

    assert "error" in manifest
    assert "IMG_1" in manifest["error"]
    assert not workspace.project_path("proj_stem_case").exists()


def test_ingest_stem_collision_across_two_calls_refuses_naming_both_files(tmp_path):
    """The classic collision: one call places foo.jpg, a later call offers foo.png into the
    same bucket. The bucket keeps holding foo.jpg alone; nothing from the second call lands."""
    src1 = tmp_path / "raw1"
    _make_image(src1 / "foo.jpg", exif_date="2026:02:11 10:00:00")
    first = ingest_images(source=str(src1), name="proj_crosscall_case", site="north orchard")
    assert "error" not in first

    proj = Path(first["project_path"])
    placed = proj / "images" / "2026-02-11" / "foo.jpg"
    original_bytes = placed.read_bytes()

    src2 = tmp_path / "raw2"
    _make_image(src2 / "foo.png")
    second = ingest_images(
        source=str(src2), name="proj_crosscall_case", site="north orchard", date_from="2026-02-11",
    )

    assert "error" in second
    assert "foo.jpg" in second["error"] and "foo.png" in second["error"]
    assert placed.is_file()
    assert not (proj / "images" / "2026-02-11" / "foo.png").exists()
    assert placed.read_bytes() == original_bytes


def test_ingest_case_variant_stem_collision_across_two_calls_refuses(tmp_path):
    """Foo.jpg then foo.png: different exact stems, the same case-folded key."""
    src1 = tmp_path / "raw1"
    _make_image(src1 / "Foo.jpg", exif_date="2026:02:11 10:00:00")
    first = ingest_images(source=str(src1), name="proj_casevariant_case", site="north orchard")
    assert "error" not in first

    proj = Path(first["project_path"])
    placed = proj / "images" / "2026-02-11" / "Foo.jpg"
    original_bytes = placed.read_bytes()

    src2 = tmp_path / "raw2"
    _make_image(src2 / "foo.png")
    second = ingest_images(
        source=str(src2), name="proj_casevariant_case", site="north orchard", date_from="2026-02-11",
    )

    assert "error" in second
    assert "Foo.jpg" in second["error"] and "foo.png" in second["error"]
    assert placed.is_file()
    assert not (proj / "images" / "2026-02-11" / "foo.png").exists()
    assert placed.read_bytes() == original_bytes


def test_ingest_admits_a_source_after_a_band_group_manifest_with_a_different_stem(tmp_path):
    """A rail must admit valid work: a manifest at one stem never blocks an ordinary source at a
    genuinely distinct stem in the same bucket."""
    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest

    src = tmp_path / "raw"
    _make_image(src / "plain.png")
    manifest = ingest_images(source=str(src), name="proj_manifestdistinct_case", site="north orchard")
    assert "error" not in manifest

    bucket_dir = Path(manifest["project_path"]) / "images" / "undated"
    band_a = bucket_dir / "cap_G.tif"
    band_b = bucket_dir / "cap_R.tif"
    band_a.write_bytes(b"a")
    band_b.write_bytes(b"b")
    write_band_group_manifest(bucket_dir, "cap", {"Green": band_a, "Red": band_b})

    src2 = tmp_path / "raw2"
    _make_image(src2 / "other.png")
    second = ingest_images(
        source=str(src2), name="proj_manifestdistinct_case", site="north orchard",
    )
    assert "error" not in second
    assert (bucket_dir / "other.png").is_file()


def test_ingest_after_a_band_group_manifest_refuses_naming_the_manifest(tmp_path):
    """cap.jpg after a cap.bandgroup manifest: the door refuses rather than admitting a source
    that would mint the manifest-versus-raw ambiguity every reader already refuses."""
    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest

    src = tmp_path / "raw"
    _make_image(src / "plain.png")
    first = ingest_images(source=str(src), name="proj_manifest_case", site="north orchard")
    assert "error" not in first

    bucket_dir = Path(first["project_path"]) / "images" / "undated"
    band_a = bucket_dir / "cap_G.tif"
    band_b = bucket_dir / "cap_R.tif"
    band_a.write_bytes(b"a")
    band_b.write_bytes(b"b")
    write_band_group_manifest(bucket_dir, "cap", {"Green": band_a, "Red": band_b})

    src2 = tmp_path / "raw2"
    _make_image(src2 / "cap.jpg")
    second = ingest_images(source=str(src2), name="proj_manifest_case", site="north orchard")

    assert "error" in second
    assert "cap.bandgroup" in second["error"]
    assert not (bucket_dir / "cap.jpg").exists()


def test_ingest_a_collision_and_a_conflicting_site_reports_the_collision(tmp_path):
    """The pre-scan runs before the site check, so a call carrying both faults names the
    collision, not the site conflict."""
    src1 = tmp_path / "raw1"
    _make_image(src1 / "foo.jpg", exif_date="2026:02:11 10:00:00")
    first = ingest_images(source=str(src1), name="proj_collisionsite_case", site="north orchard")
    assert "error" not in first

    src2 = tmp_path / "raw2"
    _make_image(src2 / "foo.png")
    second = ingest_images(
        source=str(src2), name="proj_collisionsite_case", site="south orchard",
        date_from="2026-02-11",
    )

    assert "error" in second
    assert "collision" in second["error"]
    assert "orchard" not in second["error"]


def test_ingest_survives_a_bad_file_mid_batch(tmp_path, monkeypatch):
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

    manifest = ingest_images(source=str(src), name="proj_resilient_case", site="north orchard")

    assert manifest["copied"] == 2  # the two good files still landed
    assert len(manifest["errors"]) == 1
    assert manifest["errors"][0]["source"].endswith("bad.png")


def test_ingest_reingest_is_idempotent_via_collision_skip(tmp_path):
    src = tmp_path / "raw"
    _make_image(src / "a.jpg", exif_date="2026:02:11 10:30:00")
    ingest_images(source=str(src), name="proj_reingest_case", site="north orchard")
    # Second run: dest already exists → skipped, nothing copied.
    second = ingest_images(source=str(src), name="proj_reingest_case", site="north orchard")
    assert second["copied"] == 0
    assert len(second["skipped_collisions"]) == 1


def test_ingest_date_from_none_all_undated(tmp_path):
    src = tmp_path / "raw"
    _make_image(src / "a.jpg", exif_date="2026:02:11 10:30:00")  # has EXIF but ignored
    _make_image(src / "b.png")

    manifest = ingest_images(source=str(src), name="proj_none_mode", site="north orchard", date_from="none")

    assert manifest["undated"] == 2
    assert manifest["buckets"] == {}


def test_ingest_date_from_literal_bucket(tmp_path):
    src = tmp_path / "raw"
    _make_image(src / "a.png")
    _make_image(src / "b.png")

    manifest = ingest_images(source=str(src), name="proj_literal_mode", site="north orchard", date_from="2026-05-15")

    assert manifest["buckets"] == {"2026-05-15": 2}
    assert manifest["undated"] == 0
    assert (Path(manifest["project_path"]) / "images" / "2026-05-15" / "a.png").is_file()


def test_ingest_literal_bucket_rejects_traversal(tmp_path):
    src = tmp_path / "raw"
    _make_image(src / "a.png")
    manifest = ingest_images(source=str(src), name="proj_bad_mode", site="north orchard", date_from="../escape")
    assert "error" in manifest


def test_ingest_no_images_found(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    manifest = ingest_images(source=str(empty), name="proj_empty_case", site="north orchard")
    assert "error" in manifest
    # No project should be scaffolded when there's nothing to ingest.
    assert not (workspace.project_path("proj_empty_case")).exists()


def test_ingest_explicit_project_path_overrides_workspace(tmp_path):
    src = tmp_path / "raw"
    _make_image(src / "a.png")
    dest = tmp_path / "custom_dest"

    manifest = ingest_images(source=str(src), name="ignored", site="north orchard", project_path=str(dest))

    assert Path(manifest["project_path"]) == dest
    assert (dest / "images" / "undated" / "a.png").is_file()


def test_ingest_refuses_a_non_conforming_name_under_the_workspace(tmp_path, _isolate_workspace):
    src = tmp_path / "raw"
    _make_image(src / "a.png")

    manifest = ingest_images(source=str(src), name="two_segments", site="north orchard")

    assert "error" in manifest
    assert not workspace.project_path("two_segments").exists()


def test_ingest_admits_a_conforming_name_under_the_workspace(tmp_path):
    src = tmp_path / "raw"
    _make_image(src / "a.png")

    manifest = ingest_images(source=str(src), name="proj_conforms_case", site="north orchard")

    assert "error" not in manifest


def test_ingest_refuses_a_non_conforming_override_basename_under_the_workspace(
    tmp_path, _isolate_workspace
):
    src = tmp_path / "raw"
    _make_image(src / "a.png")
    dest = _isolate_workspace / "two_segments"

    manifest = ingest_images(source=str(src), name="ignored", site="north orchard", project_path=str(dest))

    assert "error" in manifest
    assert not dest.exists()


def test_ingest_a_second_date_into_an_existing_non_conforming_project_opens_it_by_name(
    tmp_path, _isolate_workspace
):
    from tcip_store.binding import bind_default

    # A bound process holds a root's database open, so a project directory is renamed only
    # with the backend closed and rebound.
    backend = bind_default()
    src1 = tmp_path / "raw1"
    _make_image(src1 / "a.jpg", exif_date="2026:02:11 10:30:00")
    ingest_images(source=str(src1), name="proj_reused_case", site="north orchard")

    backend.close()
    bind_default()

    # Rename to a name that doesn't fit crop_subject_phenotype: what an existing project made
    # before this rail (or outside the platform) looks like.
    non_conforming = _isolate_workspace / "two_segments"
    workspace.project_path("proj_reused_case").rename(non_conforming)

    src2 = tmp_path / "raw2"
    _make_image(src2 / "b.jpg", exif_date="2026:03:01 10:30:00")
    manifest = ingest_images(source=str(src2), name="two_segments", site="north orchard")

    assert "error" not in manifest
    assert (non_conforming / "images" / "2026-03-01" / "b.jpg").is_file()


def test_ingest_admits_a_non_conforming_override_outside_the_workspace(tmp_path):
    src = tmp_path / "raw"
    _make_image(src / "a.png")
    dest = tmp_path / "two_segments"

    manifest = ingest_images(source=str(src), name="ignored", site="north orchard", project_path=str(dest))

    assert "error" not in manifest
    assert Path(manifest["project_path"]) == dest


def test_ingest_non_recursive_skips_subfolders(tmp_path):
    src = tmp_path / "raw"
    _make_image(src / "top.png")
    _make_image(src / "sub" / "deep.png")

    manifest = ingest_images(source=str(src), name="proj_flat_case", site="north orchard", recursive=False)

    assert manifest["total"] == 1  # only top.png


def test_ingest_writes_audit_entry(tmp_path, monkeypatch):
    import tcip_store as ts
    import tcip_mcp.audit as audit_mod

    monkeypatch.setattr(audit_mod, "AUDIT_ROOT", tmp_path)

    src = tmp_path / "raw"
    _make_image(src / "a.png")
    ingest_images(source=str(src), name="proj_audit_case", site="north orchard")

    entries = ts.read_log(audit_mod.audit_log_key(tmp_path)).records
    assert any(e["tool"] == "ingest_images" and e["status"] == "ok" for e in entries)


def test_inspect_project_counts_canonical_images(tmp_path):
    from tcip_mcp.tools.project_tools import inspect_project

    src = tmp_path / "raw"
    _make_image(src / "a.jpg", exif_date="2026:02:11 10:30:00")
    _make_image(src / "b.png")
    manifest = ingest_images(source=str(src), name="proj_status_case", site="north orchard")
    proj = manifest["project_path"]

    status = inspect_project(proj)
    assert status["initialized"] is True
    assert status["image_count"] == 2
    assert "2026-02-11" in status["dates"]
    assert "undated" in status["dates"]


# ── the project record's authored site ────────────────────────────────────────


def test_ingest_records_the_site_and_a_second_date_with_the_same_site_is_idempotent(tmp_path):
    """The ordinary re-run: a second season's images land in the same project, once with
    trailing whitespace on the offered site, and nothing about the recorded site changes."""
    from tcip_mcp.project_record import read_record

    src1 = tmp_path / "raw1"
    _make_image(src1 / "a.jpg", exif_date="2026:02:11 10:30:00")
    first = ingest_images(source=str(src1), name="proj_site_case", site="north orchard")
    assert "error" not in first

    src2 = tmp_path / "raw2"
    _make_image(src2 / "b.jpg", exif_date="2026:03:01 10:30:00")
    second = ingest_images(source=str(src2), name="proj_site_case", site="north orchard  ")

    assert "error" not in second
    proj = second["project_path"]
    assert (Path(proj) / "images" / "2026-03-01" / "b.jpg").is_file()
    assert read_record(proj)["site"] == "north orchard"


def test_ingest_refuses_a_conflicting_site_before_copying_a_byte(tmp_path):
    src1 = tmp_path / "raw1"
    _make_image(src1 / "a.jpg", exif_date="2026:02:11 10:30:00")
    first = ingest_images(source=str(src1), name="proj_conflict_case", site="north orchard")
    assert "error" not in first

    src2 = tmp_path / "raw2"
    _make_image(src2 / "b.jpg", exif_date="2026:03:01 10:30:00")
    second = ingest_images(source=str(src2), name="proj_conflict_case", site="south orchard")

    assert "error" in second
    assert "north orchard" in second["error"]
    assert "south orchard" in second["error"]
    proj = Path(first["project_path"])
    assert not (proj / "images" / "2026-03-01").exists()  # nothing copied


def test_ingest_refuses_an_empty_site(tmp_path):
    src = tmp_path / "raw"
    _make_image(src / "a.jpg")

    manifest = ingest_images(source=str(src), name="proj_empty_site", site="   ")

    assert "error" in manifest
    assert "empty" in manifest["error"]
    assert not (workspace.project_path("proj_empty_site")).exists()


def test_ingest_refuses_a_present_but_invalid_record(tmp_path):
    """The door surfaces the reader's own refusal rather than the store's raw exception."""
    import tcip_store

    from tcip_mcp.project_record import project_record_key

    src = tmp_path / "raw"
    _make_image(src / "a.jpg")
    dest = tmp_path / "custom_dest"
    tcip_store.replace(
        project_record_key(str(dest)), {"not_site": "whatever"}, expect=tcip_store.Version.ABSENT
    )

    manifest = ingest_images(
        source=str(src), name="ignored", site="north orchard", project_path=str(dest)
    )

    assert "error" in manifest
    assert "does not hold a site" in manifest["error"]


def test_ingest_refuses_an_undecodable_record(tmp_path):
    """The store's own DecodeError is a StoreError, caught and returned as the door's error."""
    import tcip_store

    from tcip_mcp.project_record import project_record_key
    from tests._record_damage_fixtures import damage_record

    src = tmp_path / "raw"
    _make_image(src / "a.jpg")
    dest = tmp_path / "custom_dest"
    key = project_record_key(str(dest))
    tcip_store.replace(key, {"site": "north orchard"}, expect=tcip_store.Version.ABSENT)
    damage_record(key, b"{not valid json")

    manifest = ingest_images(
        source=str(src), name="ignored", site="north orchard", project_path=str(dest)
    )

    assert "error" in manifest
    assert "does not decode" in manifest["error"]


def test_ingest_refuses_an_unadopted_root(tmp_path, monkeypatch):
    """A root whose records are still loose files: the store's conform rail refuses
    ingest_images's site write there until scripts/adopt_store.py has run. The file backend
    legitimately produces that state (import_project no longer does: it adopts a fresh root
    under the database backend), so the unadopted root here is built by writing through the
    file backend directly, through initialize_project, and then judged under the database backend
    explicitly, since that is a fact about that backend's conform rail, not about whichever
    backend the suite happens to run this file on."""
    import tcip_store
    from tcip_store.file_backend import FileBackend
    from tcip_store.sqlite_backend import SqliteBackend
    from tcip_store.store import _backend

    from tcip_mcp.tools.project_tools import initialize_project

    previous = _backend()
    dest = tmp_path / "unadopted"
    # initialize_project's own audit entry lands at the platform root, not dest; a throwaway root here
    # keeps it off tmp_path, which ingest_images's own audit write below needs pristine.
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "scratch_platform_root"))
    file_backend = FileBackend()
    tcip_store.bind(file_backend)
    try:
        initialize_project(str(dest), site="north orchard")
    finally:
        tcip_store.bind(previous)
        file_backend.close()

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    backend = SqliteBackend()
    tcip_store.bind(backend)
    try:
        src2 = tmp_path / "raw2"
        _make_image(src2 / "b.jpg")
        manifest = ingest_images(
            source=str(src2), name="ignored", site="north orchard", project_path=str(dest)
        )
    finally:
        tcip_store.bind(previous)
        backend.close()

    assert "error" in manifest
    assert "scripts/adopt_store.py" in manifest["error"]


def test_ingest_after_import_and_adopt_admits_a_second_date(tmp_path, _isolate_workspace):
    """The import door adopts a fresh root itself when the process is bound to the database
    backend, so a project it lands is usable at once: no operator ``scripts/adopt_store.py``
    run sits between ``import_project`` and ``ingest_images``. Import, ingest is the admit case.
    Bound to the database backend explicitly, since that is what the door's own adoption step
    is conditional on."""
    import tcip_store
    from tcip_store.sqlite_backend import SqliteBackend
    from tcip_store.store import _backend

    from tcip_mcp.project_record import read_record
    from tcip_mcp.tools.project_tools import archive_project, import_project

    previous = _backend()
    backend = SqliteBackend()
    tcip_store.bind(backend)
    try:
        src1 = tmp_path / "raw1"
        _make_image(src1 / "a.jpg", exif_date="2026:02:11 10:30:00")
        first = ingest_images(source=str(src1), name="proj_import_case", site="north orchard")
        assert "error" not in first
        original_root = Path(first["project_path"])

        zip_path = tmp_path / "export.zip"
        exported = archive_project(str(original_root), str(zip_path))
        assert "error" not in exported

        dest = _isolate_workspace / "proj_reopened_case"
        imported = import_project(str(zip_path), str(dest))
        assert "error" not in imported
        assert imported["database_built"] is True

        src2 = tmp_path / "raw2"
        _make_image(src2 / "b.jpg", exif_date="2026:03:01 10:30:00")
        second = ingest_images(
            source=str(src2), name="proj_reopened_case", site="north orchard",
            project_path=str(dest),
        )
        assert "error" not in second
        assert read_record(str(dest))["site"] == "north orchard"
    finally:
        tcip_store.bind(previous)
        backend.close()

    assert (dest / "images" / "2026-03-01" / "b.jpg").is_file()
    assert read_record(str(dest))["site"] == "north orchard"
