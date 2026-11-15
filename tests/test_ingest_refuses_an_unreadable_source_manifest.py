"""``ingest_images`` reads each band-group manifest it selected before it copies anything.

Every source folder is built through the platform's producers: the band files and the manifest are
written by ``write_grouped_capture`` (the band-group writer) into the source folder, and a manifest
the cases refuse is then made unreadable by ``unreadable_manifest``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir, image_root
from tcip_mcp.pipelines.data.band_groups import read_band_group_manifest
from tcip_mcp.pipelines.image_utils import (
    UnreadableBandGroupError,
    scan_identities,
)
from tcip_mcp.tools.ingest_tools import IngestManifest, ingest_images
from tests._audit_fixtures import audit_rows
from tests._producer_fixtures import (
    unreadable_manifest,
    without_bands,
    write_grouped_capture,
    write_image,
)

INGEST_CLAUSE = "repair the manifest and ingest again."


def _unreadable_group(source: Path, stem: str = "cap") -> Path:
    """Write ``stem``'s Green and Red band group into ``source``, its manifest made unreadable.

    Returns the manifest.
    """
    manifest, _bands = write_grouped_capture(source, stem, ("Green", "Red"), 8)
    return unreadable_manifest(manifest, without_bands)


def _files(root: Path) -> dict[str, bytes]:
    """Return the bytes of each file under ``root`` by its relative POSIX path."""
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _destination(project: Path) -> list[str]:
    """Return every directory and file under the project's ``images`` tree, relative and sorted."""
    root = image_root(project)
    if not root.exists():
        return []
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


def _refused(
    project: Path, source: str | Path, **choices: bool
) -> UnreadableBandGroupError:
    """Return the :class:`UnreadableBandGroupError` ``ingest_images`` raises for ``source``."""
    with pytest.raises(UnreadableBandGroupError) as refused:
        ingest_images(project, source=str(source), actor=None, **choices)
    return refused.value


@pytest.mark.parametrize(
    ("copy", "group_dir", "suffix"),
    [
        (True, "", ".bandgroup"),
        (False, "", ".bandgroup"),
        (True, "sub", ".bandgroup"),
        (True, "sub", ".BANDGROUP"),
    ],
    ids=["copy", "move", "nested", "upper-case-suffix"],
)
def test_an_unreadable_source_manifest_refuses_before_a_copy_or_a_move(
    project: Path, tmp_path: Path, copy: bool, group_dir: str, suffix: str
) -> None:
    """The refusal names the source's manifest and leaves destination and sources as they were.

    Photographs sit in the source beside the band group (outside its subfolder when nested), so
    a copy or move that began before the read would show in the destination or the sources.
    """
    source = tmp_path / "raw"
    write_image(source / "a_plain.png")
    write_image(source / "z_plain.png")
    manifest = _unreadable_group(source / group_dir)
    manifest = manifest.rename(manifest.with_suffix(suffix))
    destination_before = _destination(project)
    sources_before = _files(source)
    audit_before = len(audit_rows(project, "ingest_images"))

    refusal = _refused(project, source, copy=copy)
    sentence = str(refusal)

    assert _destination(project) == destination_before
    assert _files(source) == sources_before
    (line,) = audit_rows(project, "ingest_images")[audit_before:]
    assert (line["status"], line["error"]) == ("exception", sentence)
    assert str(manifest) in sentence
    assert refusal.__cause__ is not None
    assert str(refusal.__cause__) in sentence
    assert sentence.endswith(INGEST_CLAUSE)


def test_the_scan_and_the_ingest_refuse_one_manifest_with_one_head(
    project: Path, tmp_path: Path
) -> None:
    source = tmp_path / "raw"
    manifest = _unreadable_group(source)

    with pytest.raises(UnreadableBandGroupError) as scanned:
        scan_identities(source)
    ingested = _refused(project, source)

    scan_head, _, scan_clause = str(scanned.value).partition(", so ")
    ingest_head, _, ingest_clause = str(ingested).partition(", so ")
    assert scan_head == ingest_head
    assert str(manifest) in scan_head
    assert scan_clause != ingest_clause


def test_a_glob_that_selects_a_readable_manifest_does_not_read_an_unselected_one(
    project: Path, tmp_path: Path
) -> None:
    source = tmp_path / "raw"
    write_grouped_capture(source, "a", ("Green", "Red"), 8)
    _unreadable_group(source, "b")

    manifest = ingest_images(project, source=str(source / "a*"), actor=None)

    assert isinstance(manifest, IngestManifest), manifest
    assert manifest.found == manifest.copied == 3
    assert sorted(
        p.name for p in image_dir(project, UNDATED_BUCKET).iterdir()
    ) == [
        "a.bandgroup",
        "a_Green.tif",
        "a_Red.tif",
    ]


def test_a_source_folder_whose_manifest_reads_is_ingested_with_its_bands_beside_it(
    project: Path, tmp_path: Path
) -> None:
    source = tmp_path / "raw"
    manifest, _bands = write_grouped_capture(
        source, "cap", ("Green", "Red"), 8
    )
    sources_before = _files(source)

    ingested = ingest_images(project, source=str(source), actor=None)

    assert isinstance(ingested, IngestManifest), ingested
    assert ingested.found == ingested.copied == 3
    placed = image_dir(project, UNDATED_BUCKET)
    reread = read_band_group_manifest(placed / manifest.name)
    assert {name: path.name for name, path in reread.bands.items()} == {
        "Green": "cap_Green.tif",
        "Red": "cap_Red.tif",
    }
    assert all(
        path.parent == placed and path.is_file()
        for path in reread.bands.values()
    )
    assert _files(source) == sources_before
