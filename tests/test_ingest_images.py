"""Tests for ingestion: the workspace resolver + ``ingest_images`` tool.

All tests use synthetic temp fixtures under the test's own ``TCIP_WORKSPACE``, never the
operator's real workspace.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pytest

from tcip_mcp import dataset_layout, workspace
from tcip_mcp.tools.ingest_tools import ingest_images


def _make_image(path: Path, exif_date: str | None = None) -> None:
    """Write a tiny image; when ``exif_date`` ('YYYY:MM:DD HH:MM:SS') is set, embed it
    as EXIF DateTimeOriginal so the ingest EXIF reader round-trips it.
    """
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
    stitching engine writes an orthomosaic's own capture date.
    """
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=3,
        dtype="uint8",
    ) as ds:
        ds.update_tags(**metadata)


def _make_tagged_tiff(path: Path, datetime_tag: str) -> None:
    """Write a tiny TIFF whose DateTime tag (306) states ``datetime_tag``, the standard tag rather
    than any one engine's metadata item.
    """
    import numpy as np
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(path),
        np.zeros((8, 8, 3), np.uint8),
        extratags=[(306, "s", 0, datetime_tag, True)],
    )


# ── workspace resolver ──────────────────────────────────────────────────


def test_the_workspace_from_the_environment_is_the_one_tcip_workspace_names():
    assert (
        workspace.workspace_from_environment()
        == Path(os.environ["TCIP_WORKSPACE"]).resolve()
    )


# ── dataset_layout builders ─────────────────────────────────────────────


def test_image_path_builders(tmp_path):
    assert dataset_layout.image_root(tmp_path) == tmp_path / "images"
    assert (
        dataset_layout.image_dir(tmp_path, "2026-02-11")
        == tmp_path / "images" / "2026-02-11"
    )
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


def _entries(manifest: dict) -> dict[str, tuple[str, str | None]]:
    """Each ingested file's bucket and date source, keyed by its source file name."""
    return {
        Path(f["source"]).name: (f["bucket"], f["date_source"])
        for f in manifest["files"]
    }


def _unreadable(manifest: dict) -> list[str]:
    """Each ingested file's reason its capture date could not be read, where it has one."""
    return [
        f["date_unreadable"] for f in manifest["files"] if f["date_unreadable"]
    ]


def test_ingest_exif_buckets_and_undated(project, tmp_path):
    """Each ingested file has one entry naming its bucket and what dated it, a file no source
    dated in the undated bucket with no source.
    """
    src = tmp_path / "raw"
    _make_image(src / "a.jpg", exif_date="2026:02:11 10:30:00")
    _make_image(src / "b.jpg", exif_date="2026:02:11 11:00:00")
    _make_image(src / "c.jpg", exif_date="2026:03:01 09:00:00")
    _make_image(src / "no_exif.png")  # no EXIF → undated

    manifest = ingest_images(project, source=str(src))

    assert "error" not in manifest
    assert _entries(manifest) == {
        "a.jpg": ("2026-02-11", "EXIF_DateTimeOriginal"),
        "b.jpg": ("2026-02-11", "EXIF_DateTimeOriginal"),
        "c.jpg": ("2026-03-01", "EXIF_DateTimeOriginal"),
        "no_exif.png": ("undated", None),
    }
    assert manifest["copied"] == 4
    assert manifest["moved"] == 0
    assert Path(manifest["image_root"]) == project / "images"

    assert (project / "images" / "2026-02-11" / "a.jpg").is_file()
    assert (project / "images" / "2026-03-01" / "c.jpg").is_file()
    assert (project / "images" / "undated" / "no_exif.png").is_file()


def test_ingested_bytes_read_back_through_the_image_path(project, tmp_path):
    """The writer is the real ingest_images tool. The reader is tcip_store's blob read at
    dataset_layout.image_path(root, date, stem, ext): the bytes it hands back are byte-identical
    to the source file ingest copied from, not merely present.
    """
    import tcip_store as ts

    src = tmp_path / "raw"
    _make_image(src / "a.jpg", exif_date="2026:02:11 10:30:00")

    manifest = ingest_images(project, source=str(src))
    assert "error" not in manifest

    path = dataset_layout.image_path(project, "2026-02-11", "a", ".jpg")
    stored = ts.read_blob_versioned(path).value
    assert stored == (src / "a.jpg").read_bytes()


def test_ingest_buckets_a_raster_by_the_capture_date_its_metadata_states(
    project, tmp_path
):
    """A raster whose writer states its capture date in a metadata item the ingest names lands in
    that date's bucket, the manifest naming the item that answered; the same raster with no item
    named states no date the ingest reads, so it lands undated, unreported.
    """
    src = tmp_path / "raw"
    _make_raster(src / "mosaic.tif", capture_date="2025-09-16")

    manifest = ingest_images(
        project, source=str(src), raster_date_item="capture_date"
    )

    assert _entries(manifest) == {"mosaic.tif": ("2025-09-16", "capture_date")}
    assert _unreadable(manifest) == []
    assert (project / "images" / "2025-09-16" / "mosaic.tif").is_file()

    unnamed = ingest_images(project, source=str(src))
    assert _entries(unnamed) == {"mosaic.tif": ("undated", None)}
    assert _unreadable(unnamed) == []


def test_ingest_buckets_a_tiff_by_its_own_datetime_tag(project, tmp_path):
    src = tmp_path / "raw"
    _make_tagged_tiff(src / "plot.tif", "2026:04:05 08:00:00")

    manifest = ingest_images(project, source=str(src))

    assert _entries(manifest) == {
        "plot.tif": ("2026-04-05", "TIFFTAG_DATETIME")
    }
    assert _unreadable(manifest) == []


def test_ingest_leaves_a_raster_that_states_no_date_undated_and_unreported(
    project, tmp_path
):
    """Read, and it says nothing about when it was captured: a fact, not a failure."""
    src = tmp_path / "raw"
    _make_raster(src / "plain.tif")

    manifest = ingest_images(project, source=str(src))

    assert _entries(manifest) == {"plain.tif": ("undated", None)}
    assert _unreadable(manifest) == []


def test_ingest_leaves_a_photo_without_an_exif_date_undated_and_unreported(
    project, tmp_path
):
    src = tmp_path / "raw"
    _make_image(src / "no_exif.png")

    manifest = ingest_images(project, source=str(src))

    assert _entries(manifest) == {"no_exif.png": ("undated", None)}
    assert _unreadable(manifest) == []


@pytest.mark.parametrize("name", ["broken.jpg", "broken.tif"])
def test_ingest_still_ingests_a_file_whose_capture_date_cannot_be_read(
    project, tmp_path, name
):
    """The probe never gates ingestion: an unreadable container is copied and counted like any
    other, buckets undated, and is named in the report so the difference from a file that simply
    states no date stays visible.
    """
    src = tmp_path / "raw"
    src.mkdir(parents=True)
    (src / name).write_bytes(b"this is not an image at all")

    manifest = ingest_images(project, source=str(src))

    assert manifest["copied"] == 1
    assert manifest["errors"] == []
    assert (project / "images" / "undated" / name).is_file()

    assert _entries(manifest) == {name: ("undated", None)}
    assert len(_unreadable(manifest)) == 1


def test_ingest_reports_an_exif_date_it_cannot_read_as_a_date(
    project, tmp_path
):
    src = tmp_path / "raw"
    _make_image(src / "odd.jpg", exif_date="whenever")

    manifest = ingest_images(project, source=str(src))

    assert _entries(manifest) == {"odd.jpg": ("undated", None)}
    (reason,) = _unreadable(manifest)
    assert "whenever" in reason


@pytest.mark.parametrize("stated", ["date_from", "detect_band_groups"])
def test_ingest_takes_no_bucket_or_detection_choice(project, tmp_path, stated):
    """A file's bucket is its own date or undated, and band groups are detected in every touched
    bucket: neither is a choice the call states.
    """
    src = tmp_path / "raw"
    _make_image(src / "a.jpg", exif_date="2026:02:11 10:30:00")

    with pytest.raises(TypeError, match=stated):
        ingest_images(project, source=str(src), **{stated: "none"})
    assert not (project / "images").exists()


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


def test_ingest_two_sources_with_one_stem_refuse_the_whole_call(
    project, tmp_path
):
    # Two source subfolders each holding dup.png (no EXIF → both target undated/dup.png).
    src = tmp_path / "raw"
    _make_image(src / "sub1" / "dup.png")
    _make_image(src / "sub2" / "dup.png")

    manifest = ingest_images(project, source=str(src))

    assert "error" in manifest
    assert "dup.png" in manifest["error"]
    assert "undated" in manifest["error"]
    assert not (project / "images").exists()


def test_ingest_same_stem_in_two_different_buckets_is_admitted(
    project, tmp_path
):
    """foo.jpg into one date and foo.png into another land side by side: the collision key is
    scoped per bucket.
    """
    src = tmp_path / "raw"
    _make_image(src / "foo.jpg", exif_date="2026:02:11 10:00:00")
    _make_image(src / "foo.png", exif_date="2026:03:01 10:00:00")

    manifest = ingest_images(project, source=str(src))

    assert "error" not in manifest
    assert (project / "images" / "2026-02-11" / "foo.jpg").is_file()
    assert (project / "images" / "2026-03-01" / "foo.png").is_file()


def test_ingest_same_stem_different_ext_refuses_the_whole_call(
    project, tmp_path
):
    # Labels pair by stem alone, so IMG_1.jpg and IMG_1.png in one bucket would share one
    # label file: neither has landed yet, so the whole call refuses rather than picking one.
    src = tmp_path / "raw"
    _make_image(src / "IMG_1.jpg", exif_date="2026:02:11 10:00:00")
    _make_image(src / "IMG_1.png", exif_date="2026:02:11 10:05:00")

    manifest = ingest_images(project, source=str(src))

    assert "error" in manifest
    assert "IMG_1" in manifest["error"]
    assert not (project / "images").exists()


def test_ingest_stem_collision_across_two_calls_refuses_naming_both_files(
    project, tmp_path
):
    """The classic collision: one call places foo.jpg, a later call offers foo.png into the
    same bucket. The bucket keeps holding foo.jpg alone; nothing from the second call lands.
    """
    src1 = tmp_path / "raw1"
    _make_image(src1 / "foo.jpg", exif_date="2026:02:11 10:00:00")
    first = ingest_images(project, source=str(src1))
    assert "error" not in first

    placed = project / "images" / "2026-02-11" / "foo.jpg"
    original_bytes = placed.read_bytes()

    src2 = tmp_path / "raw2"
    _make_image(src2 / "foo.png", exif_date="2026:02:11 11:00:00")
    second = ingest_images(project, source=str(src2))

    assert "error" in second
    assert "foo.jpg" in second["error"] and "foo.png" in second["error"]
    assert placed.is_file()
    assert not (project / "images" / "2026-02-11" / "foo.png").exists()
    assert placed.read_bytes() == original_bytes


def test_ingest_case_variant_stem_collision_across_two_calls_refuses(
    project, tmp_path
):
    """Foo.jpg then foo.png: different exact stems, the same case-folded key."""
    src1 = tmp_path / "raw1"
    _make_image(src1 / "Foo.jpg", exif_date="2026:02:11 10:00:00")
    first = ingest_images(project, source=str(src1))
    assert "error" not in first

    placed = project / "images" / "2026-02-11" / "Foo.jpg"
    original_bytes = placed.read_bytes()

    src2 = tmp_path / "raw2"
    _make_image(src2 / "foo.png", exif_date="2026:02:11 11:00:00")
    second = ingest_images(project, source=str(src2))

    assert "error" in second
    assert "Foo.jpg" in second["error"] and "foo.png" in second["error"]
    assert placed.is_file()
    assert not (project / "images" / "2026-02-11" / "foo.png").exists()
    assert placed.read_bytes() == original_bytes


def test_ingest_admits_a_source_after_a_band_group_manifest_with_a_different_stem(
    project, tmp_path
):
    """A manifest at one stem never blocks an ordinary source at another stem in the same
    bucket.
    """
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
    write_band_group_manifest(
        bucket_dir,
        "cap",
        {"Green": band_a, "Red": band_b},
        central_wavelength_nm=None,
        identity_tolerance={},
    )

    src2 = tmp_path / "raw2"
    _make_image(src2 / "other.png")
    second = ingest_images(project, source=str(src2))
    assert "error" not in second
    assert (bucket_dir / "other.png").is_file()


def test_ingest_after_a_band_group_manifest_refuses_naming_the_manifest(
    project, tmp_path
):
    """cap.jpg after a cap.bandgroup manifest: the door refuses rather than admitting a source
    that would mint the manifest-versus-raw ambiguity every reader already refuses.
    """
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
    write_band_group_manifest(
        bucket_dir,
        "cap",
        {"Green": band_a, "Red": band_b},
        central_wavelength_nm=None,
        identity_tolerance={},
    )

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

    import tcip_store

    real_put = tcip_store.put_blob

    def flaky_put(path, data, **kwargs):
        if Path(path).stem == "bad":
            raise OSError("simulated locked file")
        return real_put(path, data, **kwargs)

    monkeypatch.setattr(tcip_store, "put_blob", flaky_put)

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


def _undated_labeled(project, tmp_path, name: str = "a.png") -> Path:
    """One image ingested undated, its label document holding one box; the image's path."""
    from tests._producer_fixtures import box_annotation, label_image

    src = tmp_path / "raw"
    _make_image(src / name)
    (entry,) = ingest_images(project, source=str(src))["files"]
    assert entry["bucket"] == dataset_layout.UNDATED_BUCKET
    label_image(entry["dest"], [box_annotation(2, 2, 10, 10)], 32, 24)
    return Path(entry["dest"])


STATED = date(2026, 5, 15)
"""The capture date the stated-date tests state."""


def _image_records(image: Path) -> list:
    """The keys of every record keyed by ``image``'s capture and stem: its label document, its
    flags and its staged proposals.
    """
    from tcip_annotation.flags import flag_key
    from tcip_mcp.tools.proposal_tools import proposal_staging_key

    label = dataset_layout.label_key_of(image)
    return [
        label,
        flag_key(label),
        proposal_staging_key(*dataset_layout.parse_image_path(image)),
    ]


def test_a_stated_date_moves_an_undated_image_with_every_record_keyed_by_it(
    project, tmp_path, monkeypatch
):
    """The label document, the flags record and the staged proposals of the moved image all
    answer at its new capture, unchanged, and none is left at the old one.
    """
    import tcip_mcp.audit as audit_mod
    import tcip_store as ts
    from tcip_annotation.flags import FlagRequest
    from tcip_mcp.pipelines import proposal
    from tcip_mcp.tools.ingest_tools import state_capture_date
    from tcip_mcp.tools.proposal_tools import propose_annotations
    from tests._producer_fixtures import saved_annotations

    image = _undated_labeled(project, tmp_path)
    dataset_layout.save_label_document(
        project,
        dataset_layout.label_key_of(image),
        [{"subject": "bud", "bbox": [2, 2, 10, 10]}],
        width=32,
        height=24,
        author="user:tester",
        actor=None,
        gestures=dataset_layout.Gestures(flag=(FlagRequest(text="check"),)),
    )

    class _Engine:
        def propose(self, image_path: str, **params: object) -> list[dict]:
            return [
                {
                    "candidate_id": 0,
                    "bbox": [4.0, 4.0, 12.0, 12.0],
                    "area": 64,
                    "score": 0.9,
                    "engine": "stub",
                    "engine_meta": {},
                    "rings": [
                        [(4.0, 4.0), (12.0, 4.0), (12.0, 12.0), (4.0, 12.0)]
                    ],
                }
            ]

    monkeypatch.setattr(proposal, "resolve_proposer", lambda engine: _Engine())
    staged = propose_annotations(project, image_path=str(image), engine="stub")
    assert "error" not in staged, staged
    before = saved_annotations(image)
    records = [ts.read(key) for key in _image_records(image)]

    moved = state_capture_date(
        project,
        images=[str(image)],
        date=STATED,
        rationale="the field book dates this walk",
    )

    dest = project / "images" / "2026-05-15" / "a.png"
    assert moved == {
        "date": "2026-05-15",
        "moved": [{"source": str(image), "dest": str(dest)}],
    }
    assert dest.is_file() and not image.exists()
    assert saved_annotations(dest) == before
    assert [ts.read(key) for key in _image_records(dest)] == records
    assert not any(ts.exists(key) for key in _image_records(image))
    (line,) = [
        e
        for e in ts.read_log(audit_mod.audit_log_key(project)).records
        if e["tool"] == "state_capture_date"
    ]
    assert line["status"] == "ok"
    assert line["arguments"] == {
        "images": [str(image)],
        "date": "2026-05-15",
        "rationale": "the field book dates this walk",
    }


def test_a_stated_date_whose_records_fail_to_commit_changes_nothing(
    project, tmp_path, monkeypatch
):
    """A records transaction that raises after the files were copied leaves no copy in the new
    bucket, the image and its label document where they were, and no audit line.
    """
    import tcip_mcp.audit as audit_mod
    import tcip_store as ts
    from tcip_mcp.tools.ingest_tools import state_capture_date
    from tcip_store.sqlite_backend import Txn

    image = _undated_labeled(project, tmp_path)

    def refused(self, key, value):
        raise ts.StoreBusyError(key, 0.0)

    monkeypatch.setattr(Txn, "write", refused)
    with pytest.raises(ts.StoreBusyError):
        state_capture_date(
            project, images=[str(image)], date=STATED, rationale="known"
        )
    monkeypatch.undo()

    assert image.is_file()
    assert not (project / "images" / "2026-05-15").exists()
    assert ts.exists(dataset_layout.label_key_of(image))
    assert not any(
        e["tool"] == "state_capture_date"
        for e in ts.read_log(audit_mod.audit_log_key(project)).records
    )


def test_a_failed_stated_date_keeps_another_writers_file_in_the_bucket_it_made(
    project, tmp_path, monkeypatch
):
    """A file another writer lands in the bucket the call made survives the call's cleanup, and
    so does the bucket; the call's own copy is gone.
    """
    import tcip_store as ts
    from tcip_mcp.tools.ingest_tools import state_capture_date
    from tcip_store.sqlite_backend import Txn

    image = _undated_labeled(project, tmp_path)
    bucket_dir = project / "images" / "2026-05-15"

    def refused(self, key, value):
        (bucket_dir / "stranger.png").write_bytes(b"another writer's")
        raise ts.StoreBusyError(key, 0.0)

    monkeypatch.setattr(Txn, "write", refused)
    with pytest.raises(ts.StoreBusyError):
        state_capture_date(
            project, images=[str(image)], date=STATED, rationale="known"
        )
    monkeypatch.undo()

    assert (bucket_dir / "stranger.png").is_file()
    assert not (bucket_dir / "a.png").exists()
    assert image.is_file()


def test_a_stated_date_makes_its_bucket_through_the_store_route(
    project, tmp_path, monkeypatch
):
    """The bucket directory the move needs is made by the store's parent-making route, whose
    answer names it.
    """
    import tcip_store as ts
    from tcip_mcp.tools.ingest_tools import state_capture_date

    image = _undated_labeled(project, tmp_path)
    route = ts.ensure_parent
    answers = []

    def recorded(path):
        answers.append(route(path))
        return answers[-1]

    monkeypatch.setattr(ts, "ensure_parent", recorded)
    state_capture_date(
        project, images=[str(image)], date=STATED, rationale="known"
    )

    assert answers == [[project / "images" / "2026-05-15"]]


def test_a_failed_stated_date_leaves_the_empty_bucket_a_racer_made(
    project, tmp_path, monkeypatch
):
    """A bucket directory another writer makes between the store route's walk and its own
    ``mkdir`` is not the call's to remove: it survives the call's failure, empty.
    """
    import tcip_store as ts
    from tcip_mcp.tools.ingest_tools import state_capture_date
    from tcip_store.sqlite_backend import Txn

    image = _undated_labeled(project, tmp_path)
    bucket_dir = project / "images" / "2026-05-15"
    make = Path.mkdir
    raced = []

    def refused(self, key, value):
        raise ts.StoreBusyError(key, 0.0)

    def racer_first(self, *args, **kwargs):
        if self == bucket_dir and not raced:
            raced.append(self)
            make(self)
        make(self, *args, **kwargs)

    monkeypatch.setattr(Txn, "write", refused)
    monkeypatch.setattr(Path, "mkdir", racer_first)
    with pytest.raises(ts.StoreBusyError):
        state_capture_date(
            project, images=[str(image)], date=STATED, rationale="known"
        )
    monkeypatch.undo()

    assert bucket_dir.is_dir()
    assert list(bucket_dir.iterdir()) == []
    assert image.is_file()


def test_a_failed_stated_date_raises_its_cause_when_its_bucket_fills_before_removal(
    project, tmp_path, monkeypatch
):
    """Another writer's file landing in the bucket the call made just before the call removes it
    keeps the bucket and the file, and the transaction's own error propagates with no removal
    error in front of it.
    """
    import tcip_store as ts
    from tcip_mcp.tools.ingest_tools import state_capture_date
    from tcip_store.sqlite_backend import Txn

    image = _undated_labeled(project, tmp_path)
    bucket_dir = project / "images" / "2026-05-15"
    remove = Path.rmdir

    def refused(self, key, value):
        raise ts.StoreBusyError(key, 0.0)

    def landed_first(self):
        if self == bucket_dir:
            (bucket_dir / "stranger.png").write_bytes(b"another writer's")
        remove(self)

    monkeypatch.setattr(Txn, "write", refused)
    monkeypatch.setattr(Path, "rmdir", landed_first)
    with pytest.raises(ts.StoreBusyError) as raised:
        state_capture_date(
            project, images=[str(image)], date=STATED, rationale="known"
        )
    monkeypatch.undo()

    assert raised.value.__context__ is None
    assert (bucket_dir / "stranger.png").is_file()
    assert not (bucket_dir / "a.png").exists()


def test_a_failed_stated_date_raises_a_refused_bucket_removal(
    project, tmp_path, monkeypatch
):
    """A removal of the bucket the call made refused for any reason but another writer's file
    propagates, chained to the transaction's error.
    """
    import tcip_store as ts
    from tcip_mcp.tools.ingest_tools import state_capture_date
    from tcip_store.sqlite_backend import Txn

    image = _undated_labeled(project, tmp_path)

    def refused(self, key, value):
        raise ts.StoreBusyError(key, 0.0)

    def denied(self):
        raise PermissionError("access denied")

    monkeypatch.setattr(Txn, "write", refused)
    monkeypatch.setattr(Path, "rmdir", denied)
    with pytest.raises(PermissionError) as raised:
        state_capture_date(
            project, images=[str(image)], date=STATED, rationale="known"
        )
    monkeypatch.undo()

    assert isinstance(raised.value.__context__, ts.StoreBusyError)


def test_a_stated_date_whose_first_copy_fails_raises_that_cause(
    project, tmp_path, monkeypatch
):
    """The first copy failing propagates its own error and leaves no bucket behind."""
    import tcip_store as ts
    from tcip_mcp.tools.ingest_tools import state_capture_date

    image = _undated_labeled(project, tmp_path)

    def unavailable(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(ts, "put_blob", unavailable)
    with pytest.raises(OSError, match="disk unavailable"):
        state_capture_date(
            project, images=[str(image)], date=STATED, rationale="known"
        )
    monkeypatch.undo()

    assert not (project / "images" / "2026-05-15").exists()
    assert image.is_file()


def test_a_stated_date_refuses_an_image_a_published_bucket_names(
    project, tmp_path
):
    from tcip_mcp.buckets import Document, publish
    from tcip_mcp.pipelines.data.selection import ClassScope
    from tcip_mcp.tools.ingest_tools import state_capture_date

    image = _undated_labeled(project, tmp_path)
    publish(
        project,
        project,
        "proposals",
        [Document(str(image), {"annotations": []})],
        producer={"proposed_by": "agent:test"},
        scope=ClassScope(),
        execution=None,
        raster_path=None,
        raster_identity=None,
        assessment_id=None,
        actor=None,
    )

    refused = state_capture_date(
        project, images=[str(image)], date=STATED, rationale="known"
    )

    assert "proposals" in refused["error"]
    assert image.is_file()
    assert not (project / "images" / "2026-05-15").exists()


def test_a_stated_date_is_refused_by_its_type_unless_it_is_a_date():
    """The MCP call refuses a ``date`` that is no calendar date by the parameter's own type, and
    admits an ISO date.
    """
    import asyncio

    from mcp.server.mcpserver.exceptions import ToolError

    from tcip_mcp.server import build_server

    server = build_server(None)

    def called(stated: str) -> str:
        arguments = {"images": ["a.png"], "date": stated, "rationale": "known"}
        try:
            asyncio.run(server.call_tool("state_capture_date", arguments))
        except ToolError as exc:
            return str(exc)
        return ""

    refused = called("../escape")
    assert "validation error" in refused
    assert "date" in refused.split("Arguments")[-1]
    assert "validation error" not in called("2026-05-15")


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

    assert list(_entries(manifest)) == ["top.png"]


def test_ingest_writes_audit_entry(project, tmp_path):
    import tcip_mcp.audit as audit_mod
    import tcip_store as ts

    src = tmp_path / "raw"
    _make_image(src / "a.png")
    ingest_images(project, source=str(src))

    entries = ts.read_log(audit_mod.audit_log_key(project)).records
    assert any(
        e["tool"] == "ingest_images" and e["status"] == "ok" for e in entries
    )


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
    """A project the import door lands is usable at once: import, then ingest, is the admit
    case.
    """
    from tcip_mcp.project_record import read_record
    from tcip_mcp.tools.project_tools import archive_project, import_project
    from tests._web_fixtures import new_project

    dest = tmp_path.parent / "reopened"
    made = new_project(tmp_path)
    project = made.root
    src1 = tmp_path / "raw1"
    _make_image(src1 / "a.jpg", exif_date="2026:02:11 10:30:00")
    first = ingest_images(project, source=str(src1))
    assert "error" not in first

    zip_path = tmp_path.parent / "export.zip"
    exported = archive_project(project, str(zip_path))
    assert "error" not in exported

    imported = import_project(str(zip_path), str(dest))
    assert "error" not in imported

    src2 = tmp_path / "raw2"
    _make_image(src2 / "b.jpg", exif_date="2026:03:01 10:30:00")
    second = ingest_images(dest, source=str(src2))
    assert "error" not in second
    assert read_record(dest)["id"] == made.id

    assert (dest / "images" / "2026-03-01" / "b.jpg").is_file()
