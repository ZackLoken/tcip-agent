"""Tests of overview pyramids.

Covered: building the external .ovr sidecar, validating it header-only, and the
reduced-resolution reads GDAL serves from it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile

from tcip_mcp.pipelines import raster_source
from tcip_mcp.pipelines.overviews import (
    build_overviews,
    overview_dims,
    overview_levels,
    overview_sidecar,
    sidecar_valid,
)
from tcip_mcp.pipelines.raster_source import Rect, TiffWholeSource, open_raster

if TYPE_CHECKING:
    from rasterio.io import DatasetReader, DatasetWriter


def _wide_raster(
    tmp_path: Path, *, width: int = 8192, height: int = 8
) -> tuple[Path, np.ndarray]:
    """Write a ``height`` by ``width`` array of row-major ``arange % 251``; return path and array.

    The array is uint8, written by ``tifffile.imwrite`` in four-row strips.
    """
    path = tmp_path / "wide.tif"
    arr = (
        (np.arange(height * width) % 251)
        .astype(np.uint8)
        .reshape(height, width)
    )
    tifffile.imwrite(str(path), arr, rowsperstrip=4)
    return path, arr


def test_overview_levels_are_powers_of_two_down_to_the_pyramid_floor() -> None:
    """Overview levels are powers of two down to the pyramid floor.

    The deepest level is the first whose longest edge, in GDAL's rounded-up whole pixels, is at
    most the floor: 1025 pixels halve to 513, not 512.5.
    """
    assert overview_levels(239921, 141130) == [2, 4, 8, 16, 32, 64, 128, 256]
    assert overview_levels(8192, 8) == [2, 4, 8]
    assert overview_levels(1025, 3) == [2]
    assert overview_levels(2049, 3) == [2, 4]
    assert overview_levels(1024, 100) == []


def test_build_overviews_writes_a_sidecar_gdal_serves_reduced_reads_from(
    tmp_path: Path,
) -> None:
    path, arr = _wide_raster(tmp_path)
    assert not overview_dims(path)
    assert not sidecar_valid(path)

    written: list[int] = []
    sidecar = build_overviews(path, progress_cb=written.append)

    assert sidecar == overview_sidecar(path)
    assert sidecar.is_file()
    assert sidecar_valid(path)
    assert overview_dims(path) == [(4096, 4), (2048, 2), (1024, 1)]
    assert written and written[-1] == sidecar.stat().st_size

    with open_raster(path, 1) as src:
        # The path-holding answer and the open reader's own agree on the levels it serves from.
        assert src.level_dims() == overview_dims(path)
        region, spec = src.read_region(
            Rect(0, 0, 8192, 8), target_size=(4096, 4)
        )
    assert region.shape == (4, 4096, 1)
    assert spec.resample == "average"
    blocks = arr.reshape(4, 2, 4096, 2).mean(axis=(1, 3))
    assert np.allclose(np.squeeze(region, -1), blocks, atol=1.0)


def test_a_raster_the_hint_sends_whole_serves_no_level_off_its_own_pyramid(
    tmp_path: Path,
) -> None:
    """A raster the hint sends to the whole decode serves no level off its own pyramid.

    A three-row, four-band TIFF whose sidecar holds a level, opened at three channels, is served
    by the whole decode, answers no level, and its header opens no windowed reader; opened at four
    channels it answers the sidecar's level.
    """
    path = tmp_path / "three_row.tif"
    arr = np.zeros((3, 2048, 4), dtype=np.uint8)
    arr[..., 0] = np.arange(2048, dtype=np.uint16).reshape(1, 2048) % 251
    tifffile.imwrite(
        str(path),
        arr,
        photometric="rgb",
        extrasamples=["unassalpha"],
        rowsperstrip=1,
    )
    build_overviews(path)
    assert overview_dims(path) == [(1024, 2)]

    with open_raster(path, 3) as src:
        assert isinstance(src, TiffWholeSource)
        assert src.level_dims() == []
    assert raster_source.SourceHeader(path).windowed(3) is None
    with open_raster(path, 4) as src:
        assert src.level_dims() == overview_dims(path) == [(1024, 2)]


def test_an_open_reader_enumerates_its_levels_without_reopening_the_raster(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An open reader enumerates its levels without reopening the raster.

    Asked for its levels, an open GDAL reader opens one dataset per overview level and no further
    main dataset.
    """
    path, _arr = _wide_raster(tmp_path)
    build_overviews(path)
    opened: list[int | None] = []
    real = raster_source.open_gdal_dataset

    def counting(
        dataset_path: str | Path, overview: int | None = None
    ) -> DatasetReader:
        opened.append(overview)
        return real(dataset_path, overview)

    with open_raster(path, 1) as src:
        monkeypatch.setattr(raster_source, "open_gdal_dataset", counting)
        assert src.level_dims() == [(4096, 4), (2048, 2), (1024, 1)]
    assert opened == [0, 1, 2]


class _InProcessChild:
    """A stand-in for the build's child process that runs the child program, unchanged, here."""

    def __init__(self, args: list, **_kwargs: object) -> None:
        import contextlib
        import io
        import sys

        _exe, _flag, program, *argv = args
        out = io.StringIO()
        saved = sys.argv
        sys.argv = ["-c", *argv]
        try:
            with contextlib.redirect_stdout(out):
                exec(
                    compile(program, "<build child>", "exec"),
                    {"__name__": "__main__"},
                )
        finally:
            sys.argv = saved
        self.stdout = io.StringIO(out.getvalue())
        self.stderr = io.StringIO("")
        self.returncode = 0

    def poll(self) -> int:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode


def test_a_build_acquires_the_raster_once_across_parent_and_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Building a pyramid acquires the raster once across parent and child.

    One TIFF header and one GDAL open of its main dataset plan the levels and write them,
    whichever process they run in.
    """
    import rasterio

    from tcip_mcp.pipelines import overviews

    path, _arr = _wide_raster(tmp_path)
    headers: list = []
    mains: list = []
    real_init, real_open = tifffile.TiffFile.__init__, rasterio.open

    def header(
        self: tifffile.TiffFile,
        file: str,
        *args: str | int | bool | None,
        **kwargs: str | int | bool | None,
    ) -> None:
        if Path(str(file)) == path:
            headers.append(file)
        real_init(self, file, *args, **kwargs)

    def gdal(
        fp: str, *args: str, **kwargs: int | None
    ) -> DatasetReader | DatasetWriter:
        if Path(str(fp)) == path and "overview_level" not in kwargs:
            mains.append(fp)
        return real_open(fp, *args, **kwargs)

    monkeypatch.setattr(tifffile.TiffFile, "__init__", header)
    monkeypatch.setattr(rasterio, "open", gdal)
    monkeypatch.setattr(overviews.subprocess, "Popen", _InProcessChild)
    build_overviews(path)
    assert len(headers) == 1
    assert len(mains) == 1
    assert sidecar_valid(path)


def test_a_decimated_read_is_served_from_the_pyramid_not_by_decoding_the_base(
    tmp_path: Path,
) -> None:
    """A decimated read is served from the pyramid, not by decoding the base.

    With the base rewritten to zeros after the build, the native read returns zeros while the
    average-resampled decimated read returns the original content's block averages within 1.0;
    with the sidecar deleted, the same decimated read returns zeros.
    """
    path, arr = _wide_raster(tmp_path)
    build_overviews(path)

    tifffile.imwrite(str(path), np.zeros_like(arr), rowsperstrip=4)
    assert sidecar_valid(path)
    assert overview_dims(path)

    with open_raster(path, 1) as src:
        native, _ = src.read_region(Rect(0, 0, 8192, 8))
        reduced, spec = src.read_region(
            Rect(0, 0, 8192, 8), target_size=(4096, 4)
        )
    assert not native.any()
    assert reduced.shape == (4, 4096, 1)
    assert spec.resample == "average"
    blocks = arr.reshape(4, 2, 4096, 2).mean(axis=(1, 3))
    assert np.allclose(np.squeeze(reduced, -1), blocks, atol=1.0)

    overview_sidecar(path).unlink()
    with open_raster(path, 1) as src:
        off_base, _ = src.read_region(
            Rect(0, 0, 8192, 8), target_size=(4096, 4)
        )
    assert not off_base.any()


def test_build_refuses_to_rebuild_over_a_valid_sidecar(tmp_path: Path) -> None:
    path, _ = _wide_raster(tmp_path)
    build_overviews(path)
    with pytest.raises(ValueError, match="valid overview"):
        build_overviews(path)


def test_a_canceled_build_deletes_the_sidecar(tmp_path: Path) -> None:
    """A canceled build raises, leaving no sidecar and no overview levels."""
    path, _ = _wide_raster(tmp_path)
    with pytest.raises(RuntimeError):
        build_overviews(path, progress_cb=lambda _written: False)
    assert not overview_sidecar(path).exists()
    assert not overview_dims(path)


def test_a_partial_sidecar_is_invalid_and_is_rebuilt(tmp_path: Path) -> None:
    """A partial sidecar reads as invalid and a new build replaces it.

    A sidecar whose tile byte counts are zeroed in place reads as invalid, and a new build writes
    a valid sidecar at the same path.
    """
    path, _ = _wide_raster(tmp_path)
    build_overviews(path)
    partial = overview_sidecar(path)
    assert partial.is_file()
    assert sidecar_valid(path)

    # Zero the sidecar's own tile byte counts in place: the structurally complete but
    # never-written state a killed build leaves, which reads back as silent zeros.
    with tifffile.TiffFile(str(partial)) as tif:
        tag = (
            tif.pages[0].tags.get("TileByteCounts")
            or tif.pages[0].tags["StripByteCounts"]
        )
        offset, nbytes = tag.valueoffset, tag.valuebytecount
    with open(partial, "r+b") as fh:
        fh.seek(offset)
        fh.write(b"\x00" * nbytes)

    assert not sidecar_valid(path)

    sidecar = build_overviews(path)
    assert sidecar_valid(path)
    assert sidecar == partial


def test_building_overviews_for_a_raster_within_the_pyramid_floor_refuses(
    tmp_path: Path,
) -> None:
    """Building overviews for a raster within the pyramid floor refuses, saying no level exists."""
    path = tmp_path / "small.tif"
    tifffile.imwrite(str(path), np.zeros((32, 32), dtype=np.uint8))
    with pytest.raises(ValueError, match="no overview level"):
        build_overviews(path)
