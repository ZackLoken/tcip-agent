"""Tests of overview pyramids a build left unfinished.

Covered: sidecars whose tiles are only partly written, the sidecar bytes a watcher reads while a
build runs, and the cleanup a canceled build owes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from tcip_mcp.pipelines import overviews as overviews_module
from tcip_mcp.pipelines.overviews import (
    build_overviews,
    overview_dims,
    overview_levels,
    overview_sidecar,
    sidecar_valid,
)


def _deep_pyramid_raster(
    tmp_path: Path, *, width: int = 32768, height: int = 8
) -> Path:
    """Write a ``height`` by ``width`` uint8 array of ``arange % 251``; return its path.

    The values run row-major, and ``tifffile.imwrite`` writes them in eight-row strips.
    """
    path = tmp_path / "deep.tif"
    arr = (np.arange(height * width, dtype=np.int64) % 251).astype(np.uint8)
    tifffile.imwrite(str(path), arr.reshape(height, width), rowsperstrip=8)
    return path


def _growing_raster(tmp_path: Path) -> Path:
    """Write a 1024 by 32768 raster of row-major ``arange % 251`` values; return its path."""
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


def _zero_byte_counts(
    sidecar: Path, page_index: int, first: int, last: int
) -> None:
    """Zero the tile or strip byte counts ``[first, last)`` of ``sidecar``'s page ``page_index``."""
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
    tmp_path: Path,
) -> None:
    """A sidecar whose tiles are only partly written is invalid and the next build replaces it.

    With half the first level's tile byte counts zeroed and the rest written, the sidecar reads
    invalid, and the next build writes a valid sidecar at the same path.
    """
    path = _deep_pyramid_raster(tmp_path)
    sidecar = build_overviews(path)
    assert sidecar_valid(path)

    written = _tile_byte_counts(sidecar, 0)
    assert len(written) >= 4, (
        "fixture needs several tiles per level to be partly written"
    )
    _zero_byte_counts(sidecar, 0, 0, len(written) // 2)

    after = _tile_byte_counts(sidecar, 0)
    assert any(count == 0 for count in after), (
        "fixture must leave some tiles unwritten"
    )
    assert any(count > 0 for count in after), (
        "fixture must leave some tiles written"
    )

    assert sidecar_valid(path) is False

    rebuilt = build_overviews(path)
    assert rebuilt == sidecar
    assert sidecar_valid(path)


def test_a_sidecar_is_invalid_when_a_deeper_pyramid_level_is_unwritten(
    tmp_path: Path,
) -> None:
    """A sidecar is invalid when a deeper pyramid level is unwritten.

    With the first level's tiles written and the second level's byte counts all zeroed, the
    sidecar reads invalid.
    """
    path = _deep_pyramid_raster(tmp_path)
    sidecar = build_overviews(path)
    assert _page_count(sidecar) >= 3, (
        "fixture needs several levels for a deeper one to be empty"
    )
    assert sidecar_valid(path)

    deeper = _tile_byte_counts(sidecar, 1)
    assert len(deeper) >= 2
    _zero_byte_counts(sidecar, 1, 0, len(deeper))

    first_level = _tile_byte_counts(sidecar, 0)
    assert first_level and all(count > 0 for count in first_level), (
        "the first level must stay fully written, or the deeper level is not what fails validity"
    )
    assert all(count == 0 for count in _tile_byte_counts(sidecar, 1))

    assert sidecar_valid(path) is False


def test_reported_progress_is_the_sidecar_bytes_on_disk_and_ends_at_the_finished_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reported progress is the sidecar's bytes on disk and ends at the finished file's size.

    Every value reported while the build runs is at most the sidecar's size when the watcher
    reads it, and the last is the finished sidecar's size.
    """
    monkeypatch.setattr(overviews_module, "_BUILD_POLL_SECONDS", 0.01)
    path = _growing_raster(tmp_path)
    sidecar = overview_sidecar(path)
    samples: list[tuple[int, int]] = []

    def watch(written: int) -> None:
        samples.append(
            (written, sidecar.stat().st_size if sidecar.exists() else 0)
        )

    build_overviews(path, progress_cb=watch)
    finished = sidecar.stat().st_size

    part_way = [
        (written, size)
        for written, size in samples
        if 30_000 < size < 0.9 * finished
    ]
    assert part_way, (
        "the build finished without ever being sampled part way through"
    )

    for written, size in samples:
        assert written <= size, (
            f"reported {written} with {size} sidecar bytes on disk"
        )
    assert samples[-1][0] == finished


def test_canceling_a_build_that_has_started_writing_deletes_the_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Canceling a build that has started writing deletes the partly written sidecar.

    Canceling once the sidecar holds at least 1,000,000 bytes raises as canceled, and leaves no
    sidecar and no overview levels.
    """
    monkeypatch.setattr(overviews_module, "_BUILD_POLL_SECONDS", 0.01)
    path = _growing_raster(tmp_path)
    sidecar = overview_sidecar(path)
    canceled_at: list[int] = []

    def cancel_once_started(_written: int) -> bool:
        size = sidecar.stat().st_size if sidecar.exists() else 0
        if size < 1_000_000:
            return True
        canceled_at.append(size)
        return False

    with pytest.raises(RuntimeError, match="canceled"):
        build_overviews(path, progress_cb=cancel_once_started)

    assert canceled_at, (
        "the build was never canceled while its sidecar held written tiles"
    )
    assert not sidecar.exists()
    assert not overview_dims(path)


def test_a_deep_pyramid_carries_one_level_per_decimation_step(
    tmp_path: Path,
) -> None:
    """A deep pyramid carries one level per power-of-2 step down to the floor.

    A raster far above the pyramid floor gets a level per step, and the sidecar holds a page for
    each, halving in both dimensions, rounded up, as it goes.
    """
    path = _deep_pyramid_raster(tmp_path)
    levels = overview_levels(32768, 8)
    assert levels == [2, 4, 8, 16, 32]

    sidecar = build_overviews(path)
    with tifffile.TiffFile(str(sidecar)) as tif:
        shapes = [tuple(page.shape) for page in tif.pages]
    assert shapes == [(4, 16384), (2, 8192), (1, 4096), (1, 2048), (1, 1024)]
    assert sidecar_valid(path)
