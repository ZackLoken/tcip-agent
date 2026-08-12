"""Overview pyramids: building the external .ovr sidecar, validating it header-only, and the
reduced-resolution reads GDAL serves from it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from tcip_mcp.pipelines.overviews import (
    build_overviews,
    has_overviews,
    overview_levels,
    overview_sidecar,
    sidecar_valid,
)
from tcip_mcp.pipelines.raster_source import Rect, open_raster


def _wide_raster(tmp_path: Path, *, width: int = 8192, height: int = 8) -> tuple[Path, np.ndarray]:
    """A raster whose longest edge exceeds the display bound, small enough to build in tests."""
    path = tmp_path / "wide.tif"
    arr = (np.arange(height * width) % 251).astype(np.uint8).reshape(height, width)
    tifffile.imwrite(str(path), arr, rowsperstrip=4)
    return path, arr


def test_overview_levels_are_powers_of_two_down_to_the_display_bound() -> None:
    assert overview_levels(239921, 141130, max_edge=4096) == [2, 4, 8, 16, 32, 64]
    assert overview_levels(8192, 8, max_edge=4096) == [2]
    assert overview_levels(4096, 100, max_edge=4096) == []


def test_build_overviews_writes_a_sidecar_gdal_serves_reduced_reads_from(tmp_path: Path) -> None:
    path, arr = _wide_raster(tmp_path)
    assert not has_overviews(path)
    assert not sidecar_valid(path)

    fractions: list[float] = []
    sidecar = build_overviews(path, progress_cb=fractions.append)

    assert sidecar == overview_sidecar(path)
    assert sidecar.is_file()
    assert sidecar_valid(path)
    assert has_overviews(path)
    assert fractions and fractions[-1] == pytest.approx(1.0)

    with open_raster(path, 1) as src:
        region, spec = src.read_region(Rect(0, 0, 8192, 8), target_size=(4096, 4))
    assert region.shape == (4, 4096, 1)
    assert (spec.scale, spec.resample) == (0.5, "average")
    blocks = arr.reshape(4, 2, 4096, 2).mean(axis=(1, 3))
    assert np.allclose(np.squeeze(region, -1), blocks, atol=1.0)


def test_build_refuses_to_rebuild_over_a_valid_sidecar(tmp_path: Path) -> None:
    path, _ = _wide_raster(tmp_path)
    build_overviews(path)
    with pytest.raises(ValueError, match="valid overview"):
        build_overviews(path)


def test_a_cancelled_build_deletes_the_sidecar(tmp_path: Path) -> None:
    """An interrupted build must not leave a sidecar behind: its unwritten tiles would read back
    as silent zeros on the next open."""
    path, _ = _wide_raster(tmp_path)
    with pytest.raises(RuntimeError):
        build_overviews(path, progress_cb=lambda _fraction: False)
    assert not overview_sidecar(path).exists()
    assert not has_overviews(path)


def test_a_partial_sidecar_is_invalid_and_is_rebuilt(tmp_path: Path) -> None:
    """A build interrupted outside this module's own cleanup (a crash, a kill) leaves a
    structurally complete sidecar whose unwritten tiles have zero-length byte counts; it must
    read as invalid, and a new build must replace it rather than trust or refuse it."""
    path, _ = _wide_raster(tmp_path)
    build_overviews(path)
    partial = overview_sidecar(path)
    assert partial.is_file()
    assert sidecar_valid(path)

    # Zero the sidecar's own tile byte counts in place: the structurally complete but
    # never-written state a killed build leaves, which reads back as silent zeros.
    with tifffile.TiffFile(str(partial)) as tif:
        tag = tif.pages[0].tags.get("TileByteCounts") or tif.pages[0].tags["StripByteCounts"]
        offset, nbytes = tag.valueoffset, tag.valuebytecount
    with open(partial, "r+b") as fh:
        fh.seek(offset)
        fh.write(b"\x00" * nbytes)

    assert not sidecar_valid(path)

    sidecar = build_overviews(path)
    assert sidecar_valid(path)
    assert sidecar == partial


def test_building_overviews_for_a_raster_within_the_display_bound_refuses(tmp_path: Path) -> None:
    """An empty level list would clear existing overviews instead of building any, so a raster
    already at display scale is refused with the bound named."""
    path = tmp_path / "small.tif"
    tifffile.imwrite(str(path), np.zeros((32, 32), dtype=np.uint8))
    with pytest.raises(ValueError, match="display bound"):
        build_overviews(path)
