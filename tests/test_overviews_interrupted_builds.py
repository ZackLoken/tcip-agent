"""Overview pyramids that were never finished: sidecars whose tiles are only partly written,
the progress fraction a watcher reads while a build runs, and the cleanup a cancelled build owes.

A pyramid is structurally complete long before its pixels are, so these cover the states a build
that stopped early leaves behind, where a wrong answer is a silent one: a reduced-resolution read
of an unwritten tile returns zeros rather than failing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from tcip_mcp.pipelines import overviews as overviews_module
from tcip_mcp.pipelines.overviews import (
    _predicted_sidecar_bytes,
    build_overviews,
    has_overviews,
    overview_levels,
    overview_sidecar,
    sidecar_valid,
)


def _deep_pyramid_raster(tmp_path: Path, *, width: int = 32768, height: int = 8) -> Path:
    """A raster wide enough to need several decimation levels, so its sidecar holds one page per
    level rather than the single page a two-level pyramid would."""
    path = tmp_path / "deep.tif"
    arr = (np.arange(height * width, dtype=np.int64) % 251).astype(np.uint8)
    tifffile.imwrite(str(path), arr.reshape(height, width), rowsperstrip=8)
    return path


def _growing_raster(tmp_path: Path) -> Path:
    """A raster whose pyramid takes long enough to build that the parent polls the sidecar's size
    many times while it grows, and whose every level tiles exactly so the finished sidecar matches
    the predicted uncompressed size."""
    path = tmp_path / "growing.tif"
    arr = (np.arange(1024 * 32768, dtype=np.int64) % 251).astype(np.uint8)
    tifffile.imwrite(str(path), arr.reshape(1024, 32768), rowsperstrip=8)
    return path


def _tile_byte_counts(sidecar: Path, page_index: int) -> list[int]:
    with tifffile.TiffFile(str(sidecar)) as tif:
        return [int(count) for count in tif.pages[page_index].databytecounts]


def _page_count(sidecar: Path) -> int:
    with tifffile.TiffFile(str(sidecar)) as tif:
        return len(tif.pages)


def _zero_byte_counts(sidecar: Path, page_index: int, first: int, last: int) -> None:
    """Zero the byte counts of tiles ``[first, last)`` on one pyramid level, the on-disk state a
    build leaves for tiles it never reached."""
    with tifffile.TiffFile(str(sidecar)) as tif:
        page = tif.pages[page_index]
        tag = page.tags.get("TileByteCounts") or page.tags["StripByteCounts"]
        offset, nbytes = tag.valueoffset, tag.valuebytecount
        entries = len(page.databytecounts)
    itemsize = nbytes // entries
    with open(sidecar, "r+b") as fh:
        fh.seek(offset + first * itemsize)
        fh.write(b"\x00" * ((last - first) * itemsize))


def test_a_sidecar_whose_tiles_are_only_partly_written_is_invalid_and_is_rebuilt(
        tmp_path: Path) -> None:
    """A build killed partway leaves written tiles beside unwritten ones. The written half is no
    evidence the pyramid is usable: the unwritten tiles read back as zeros, so the sidecar is
    invalid and the next build replaces it rather than refusing as if it were sound."""
    path = _deep_pyramid_raster(tmp_path)
    sidecar = build_overviews(path)
    assert sidecar_valid(path)

    written = _tile_byte_counts(sidecar, 0)
    assert len(written) >= 4, "fixture needs several tiles per level to be partly written"
    _zero_byte_counts(sidecar, 0, 0, len(written) // 2)

    after = _tile_byte_counts(sidecar, 0)
    assert any(count == 0 for count in after), "fixture must leave some tiles unwritten"
    assert any(count > 0 for count in after), "fixture must leave some tiles written"

    assert sidecar_valid(path) is False

    rebuilt = build_overviews(path)
    assert rebuilt == sidecar
    assert sidecar_valid(path)


def test_a_sidecar_is_invalid_when_a_deeper_pyramid_level_is_unwritten(tmp_path: Path) -> None:
    """Validity covers every decimation level, not just the first. A build that wrote the coarsest
    level and stopped leaves the deeper ones empty, and a display-scale read served from one of
    those levels returns zeros."""
    path = _deep_pyramid_raster(tmp_path)
    sidecar = build_overviews(path)
    assert _page_count(sidecar) >= 3, "fixture needs several levels for a deeper one to be empty"
    assert sidecar_valid(path)

    deeper = _tile_byte_counts(sidecar, 1)
    assert len(deeper) >= 2
    _zero_byte_counts(sidecar, 1, 0, len(deeper))

    first_level = _tile_byte_counts(sidecar, 0)
    assert first_level and all(count > 0 for count in first_level), (
        "the first level must stay fully written, or the deeper level is not what fails validity")
    assert all(count == 0 for count in _tile_byte_counts(sidecar, 1))

    assert sidecar_valid(path) is False


def test_the_progress_denominator_scales_with_the_pyramid_area() -> None:
    """Progress is reported against the pyramid's pixel count, so quadrupling a raster's pixels
    quadruples the denominator. A denominator that grew with the raster's edges instead would be
    passed by the sidecar's first few kilobytes and report a fresh build as nearly finished."""
    levels = [2, 4, 8]
    short = _predicted_sidecar_bytes(32768, 256, 1, 1, levels)
    tall = _predicted_sidecar_bytes(32768, 1024, 1, 1, levels)
    assert tall == 4 * short

    wider_and_taller = _predicted_sidecar_bytes(65536, 512, 1, 1, levels)
    assert wider_and_taller == 4 * short

    assert _predicted_sidecar_bytes(32768, 256, 3, 2, levels) == 6 * short


def test_reported_progress_never_overstates_how_much_of_the_sidecar_is_written(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The fraction handed to a watcher is a floor on real progress. Every fraction reported while
    the build runs stays at or under the share of the finished sidecar that was on disk at the
    moment it was reported, so a watcher never sees a build it must wait minutes for as done."""
    monkeypatch.setattr(overviews_module, "_BUILD_POLL_SECONDS", 0.01)
    path = _growing_raster(tmp_path)
    sidecar = overview_sidecar(path)
    samples: list[tuple[float, int]] = []

    def watch(fraction: float) -> None:
        samples.append((fraction, sidecar.stat().st_size if sidecar.exists() else 0))

    build_overviews(path, progress_cb=watch)
    finished = sidecar.stat().st_size

    part_way = [(fraction, size) for fraction, size in samples if 30_000 < size < 0.9 * finished]
    assert part_way, "the build finished without ever being sampled part way through"

    for fraction, size in samples:
        assert fraction <= size / finished + 0.02, (
            f"reported {fraction} with {size} of {finished} sidecar bytes written")


def test_cancelling_a_build_that_has_started_writing_deletes_the_sidecar(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancelling once tiles are on disk must take the half-written sidecar with it. Left behind,
    it opens as a structurally complete pyramid whose unreached tiles read back as zeros, and the
    next open serves those zeros instead of rebuilding."""
    monkeypatch.setattr(overviews_module, "_BUILD_POLL_SECONDS", 0.01)
    path = _growing_raster(tmp_path)
    sidecar = overview_sidecar(path)
    cancelled_at: list[int] = []

    def cancel_once_started(_fraction: float) -> bool:
        size = sidecar.stat().st_size if sidecar.exists() else 0
        if size < 1_000_000:
            return True
        cancelled_at.append(size)
        return False

    with pytest.raises(RuntimeError, match="cancelled"):
        build_overviews(path, progress_cb=cancel_once_started)

    assert cancelled_at, "the build was never cancelled while its sidecar held written tiles"
    assert not sidecar.exists()
    assert not has_overviews(path)


def test_a_deep_pyramid_carries_one_level_per_decimation_step(tmp_path: Path) -> None:
    """A raster far above the display bound gets a level per power-of-2 step down to it, and the
    sidecar holds a page for each, halving in both dimensions as it goes."""
    path = _deep_pyramid_raster(tmp_path)
    levels = overview_levels(32768, 8)
    assert levels == [2, 4, 8]

    sidecar = build_overviews(path)
    with tifffile.TiffFile(str(sidecar)) as tif:
        shapes = [tuple(page.shape) for page in tif.pages]
    assert shapes == [(4, 16384), (2, 8192), (1, 4096)]
    assert sidecar_valid(path)
