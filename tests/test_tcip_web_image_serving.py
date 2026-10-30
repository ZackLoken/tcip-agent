"""Image serving through one raster read (routes/images.py).

The tests cover regions, the display caps, the plain-serve rule, the stretch bounds a render
uses, and the overview build a scaled read of an oversized raster needs first.
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient
from PIL import Image

from tcip_mcp.dataset_layout import image_dir, image_root
from tcip_web.routes import images as images_route
from tests._producer_fixtures import write_image

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from rasterio.io import DatasetReader
    from starlette.testclient import httpx

    from tcip_mcp.pipelines.raster_source import (
        BandGroupRef,
        RasterSource,
        SourceHeader,
        _TiffHeader,
    )
    from tcip_web.routes.images import _RasterStats


@pytest.fixture(autouse=True)
def _clear_stats_cache() -> Iterator[None]:
    """Clear the per-raster stats and band-count caches around each test.

    The caches are process-global; a fresh test starts from empty ones.
    """
    images_route._stats_cache.clear()
    images_route._count_cache.clear()
    yield
    images_route._stats_cache.clear()
    images_route._count_cache.clear()


def _quadrant_rgb(
    path: Path, width: int = 400, height: int = 300
) -> np.ndarray:
    """Write a uint8 RGB ``painted_array`` raster as a TIFF at ``path``; return the array.

    Split at ``width // 2`` and ``height // 2``, it is painted (200, 20, 20) top left,
    (20, 200, 20) top right, (20, 20, 200) bottom left and (200, 200, 20) bottom right.
    """
    from tests._producer_fixtures import painted_array

    x, y = width // 2, height // 2
    arr = painted_array(
        width,
        height,
        [
            ((0, 0, x, y), (200, 20, 20)),
            ((x, 0, width, y), (20, 200, 20)),
            ((0, y, x, height), (20, 20, 200)),
            ((x, y, width, height), (200, 200, 20)),
        ],
        background=(0, 0, 0),
        mode="RGB",
    )
    tifffile.imwrite(str(path), arr)
    return arr


def _multiband(
    path: Path,
    *,
    channels: int = 4,
    height: int = 24,
    width: int = 40,
    dtype: str = "uint16",
) -> np.ndarray:
    """Write a seeded multi-band raster as a TIFF at ``path``; return its array.

    The array is an integer draw in [0, 1000) at seed 3, ``height`` by ``width`` by
    ``channels``, cast to ``dtype``; 3000 is added to its right half, and the result is cast to
    ``dtype`` again.
    """
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 1000, size=(height, width, channels)).astype(dtype)
    arr[:, width // 2 :] = arr[:, width // 2 :] + 3000
    tifffile.imwrite(str(path), arr.astype(dtype))
    return arr.astype(dtype)


def _wide_raster(
    path: Path, *, width: int = 5000, height: int = 64
) -> np.ndarray:
    """Write ``arange(height * width) % 251`` as a uint8 TIFF at ``path``; return the array.

    The array is ``height`` by ``width``, 64 by 5000 by default, written in strips of 8 rows.
    """
    arr = (
        (np.arange(height * width) % 251)
        .astype(np.uint8)
        .reshape(height, width)
    )
    tifffile.imwrite(str(path), arr, rowsperstrip=8)
    return arr


def _wide_multiband(
    path: Path, *, width: int = 5000, height: int = 64, channels: int = 4
) -> np.ndarray:
    """Write a uint8 draw at seed 11 as a band-stack TIFF at ``path``; return the array.

    The array is ``height`` by ``width`` by ``channels``, 64 by 5000 by 4 by default, written
    in strips of 8 rows, no band of it named a color.
    """
    rng = np.random.default_rng(11)
    arr = rng.integers(0, 256, size=(height, width, channels)).astype(np.uint8)
    tifffile.imwrite(
        str(path), arr, rowsperstrip=8, photometric="minisblack", planarconfig="contig"
    )
    return arr


def _served(resp: httpx.Response) -> np.ndarray:
    assert resp.status_code == 200, resp.text
    return np.asarray(Image.open(io.BytesIO(resp.content)))


def _region(read: Mapping[str, int | float | str]) -> list[int | float | str]:
    """Return a view read's (or a rect's) corners as the image route's ``region`` parameter."""
    return [read[k] for k in ("x0", "y0", "x1", "y1")]


DISPLAY = 3840 * 2160
"""The display pixel count most requests here report: a fixture 3840x2160 screen at device pixel
ratio 1, larger than every raster built here, unless a test reports a smaller display."""

SMALL_DISPLAY = 100_000
"""A fixture display pixel count under the 5000x64 wide rasters' 320,000 pixels, so a whole view
of one is a display read."""


ENCODINGS = {
    "image/jpeg": ("JPEG", {"quality": 95, "subsampling": 0}),
    "image/png": ("PNG", {"compress_level": 1}),
}
"""The encoding policy each media type is served under, stated here rather than read from the
route, so a change to the route's policy fails these renders instead of moving both sides."""


def _renders_as(resp: httpx.Response, owed: np.ndarray) -> None:
    """Assert the served image is ``owed``'s pixels, encoded byte for byte.

    The encoding is the policy :data:`ENCODINGS` states for the response's media type.
    """
    assert resp.status_code == 200, resp.text
    pil_format, options = ENCODINGS[resp.headers["content-type"]]
    buf = io.BytesIO()
    Image.fromarray(owed, mode="RGB").save(buf, pil_format, **options)
    assert resp.content == buf.getvalue()


def _composite(
    arr: np.ndarray,
    stretch: str,
    bounds: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Return ``composite_display_rgb(arr, (0, 1, 2), stretch, bounds)``.

    ``bounds`` is per band ``(low, high)``, or ``None``.
    """
    from tcip_mcp.pipelines.band_stats import composite_display_rgb

    return composite_display_rgb(arr, (0, 1, 2), stretch, bounds)


# ── Regions ──────────────────────────────────────────────────────────────────────────────


def test_a_region_serves_that_regions_own_pixels(
    client: TestClient, tmp_path: Path
) -> None:
    path = tmp_path / "quads.tif"
    _quadrant_rgb(path)
    top_right = _served(
        client.get(
            "/api/images",
            params={
                "path": str(path),
                "display_pixels": DISPLAY,
                "region": [200, 0, 400, 150],
            },
        )
    )
    bottom_left = _served(
        client.get(
            "/api/images",
            params={
                "path": str(path),
                "display_pixels": DISPLAY,
                "region": [0, 150, 200, 300],
            },
        )
    )
    assert top_right.shape == (150, 200, 3)
    assert bottom_left.shape == (150, 200, 3)
    assert np.allclose(top_right.mean(axis=(0, 1)), (20, 200, 20), atol=6)
    assert np.allclose(bottom_left.mean(axis=(0, 1)), (20, 20, 200), atol=6)


def test_a_region_outside_the_raster_is_refused(
    client: TestClient, tmp_path: Path
) -> None:
    path = tmp_path / "quads.tif"
    _quadrant_rgb(path)
    resp = client.get(
        "/api/images",
        params={
            "path": str(path),
            "display_pixels": DISPLAY,
            "region": [0, 0, 401, 300],
        },
    )
    assert resp.status_code == 400
    assert "out of bounds for a 300x400 raster" in resp.json()["detail"]


@pytest.mark.parametrize(
    "region",
    [[10, 10, 10, 20], [10, 20, 20, 20], [20, 0, 10, 20]],
)
def test_an_empty_region_is_refused_rather_than_raising(
    client: TestClient, tmp_path: Path, region: list[int]
) -> None:
    """An empty or inverted rectangle is refused as a bad request by the raster's reader."""
    path = tmp_path / "quads.tif"
    _quadrant_rgb(path)
    resp = client.get(
        "/api/images",
        params={"path": str(path), "display_pixels": DISPLAY, "region": region},
    )
    assert resp.status_code == 400
    assert "out of bounds for a 300x400 raster" in resp.json()["detail"]


def test_a_partial_region_is_refused_by_its_type(
    client: TestClient, tmp_path: Path
) -> None:
    path = tmp_path / "quads.tif"
    _quadrant_rgb(path)
    resp = client.get(
        "/api/images",
        params={
            "path": str(path),
            "display_pixels": DISPLAY,
            "region": [0, 0, 100],
        },
    )
    assert resp.status_code == 422
    assert "region" in resp.text


def test_a_selection_of_two_bands_is_refused_by_its_type(
    client: TestClient, tmp_path: Path
) -> None:
    path = tmp_path / "quads.tif"
    _quadrant_rgb(path)
    resp = client.get(
        "/api/images",
        params={"path": str(path), "display_pixels": DISPLAY, "bands": ["0", "1"]},
    )
    assert resp.status_code == 422
    assert "bands" in resp.text


def test_a_view_of_an_ingested_raster_reads_its_cells_or_one_display_read(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """A view of an ingested raster is served by its native cells or by one display read.

    A view within the display's cap is served by the native cells it intersects, a larger one
    by one display read of itself; a path naming no image refuses by name.
    """
    from tcip_mcp.pipelines.reference_grid import (
        derive_serving_tile_size,
        reference_cells,
    )
    from tcip_mcp.tools.ingest_tools import ingest_images

    source = tmp_path / "source"
    source.mkdir()
    _wide_raster(source / "strip_01.tif")
    ingested = ingest_images(tmp_path / "proj", str(source), actor=None)
    assert "error" not in ingested, ingested
    (image,) = (Path(f["dest"]) for f in ingested["files"])

    def view(x1: int) -> dict:
        resp = client.get(
            "/api/images/view",
            params={
                "path": str(image),
                "display_pixels": SMALL_DISPLAY,
                "x0": 0,
                "y0": 0,
                "x1": x1,
                "y1": 64,
            },
        )
        assert resp.status_code == 200, resp.text
        return resp.json()

    edge = derive_serving_tile_size(5000, 64, SMALL_DISPLAY)
    cells = reference_cells(5000, 64, edge, 0.0, clamp=True)
    assert [(r["name"], r["level"]) for r in view(300)["reads"]] == [
        (c.name, 0) for c in cells if c.x0 < 300
    ]
    assert view(5000)["reads"] == [
        {
            "name": "view",
            "level": 0,
            "x0": 0,
            "y0": 0,
            "x1": 5000,
            "y1": 64,
            "nx0": 0,
            "ny0": 0,
            "nx1": 5000,
            "ny1": 64,
        }
    ]

    missing = client.get(
        "/api/images/view",
        params={
            "path": str(image.with_name("absent_01.tif")),
            "display_pixels": SMALL_DISPLAY,
            "x0": 0,
            "y0": 0,
            "x1": 10,
            "y1": 10,
        },
    )
    assert missing.status_code == 404
    assert "absent_01" in missing.json()["detail"]


# ── Display caps ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("size", [(400, 300), (1, 100_000), (100_000, 1)])
def test_a_whole_view_over_the_area_cap_scales_to_fit_whichever_axis_limits(
    client: TestClient,
    tmp_path: Path,
    size: tuple[int, int],
) -> None:
    """A whole view renders within the display's area cap, whichever axis limits it.

    The shapes include a one-pixel-wide frame on either axis, and the served size header says
    what was served.
    """
    path = write_image(tmp_path / "frame.png", size, (30, 60, 90))
    resp = client.get(
        "/api/images", params={"path": str(path), "display_pixels": 20_000}
    )
    served = _served(resp)
    assert 0 < served.shape[0] * served.shape[1] <= 20_000
    assert (
        resp.headers[images_route.SERVED_SIZE_HEADER]
        == f"{served.shape[1]}x{served.shape[0]}"
    )
    if min(size) > 1:
        assert served.shape[1] / served.shape[0] == pytest.approx(
            size[0] / size[1], abs=0.02
        )


def test_a_region_over_the_area_cap_is_a_display_read_within_it(
    client: TestClient,
    tmp_path: Path,
) -> None:
    path = tmp_path / "quads.tif"
    _quadrant_rgb(path)
    resp = client.get(
        "/api/images",
        params={
            "path": str(path),
            "display_pixels": 20_000,
            "region": [0, 0, 400, 300],
        },
    )
    served = _served(resp)
    assert (
        resp.headers["content-type"]
        == images_route._DISPLAY_ENCODING.media_type
    )
    assert served.shape[0] * served.shape[1] <= 20_000


def test_a_region_within_the_area_cap_is_served_natively(
    client: TestClient, tmp_path: Path
) -> None:
    path = tmp_path / "quads.tif"
    _quadrant_rgb(path)
    resp = client.get(
        "/api/images",
        params={
            "path": str(path),
            "display_pixels": 20_000,
            "region": [0, 0, 100, 100],
        },
    )
    assert _served(resp).shape == (100, 100, 3)
    assert (
        resp.headers["content-type"]
        == images_route._NATIVE_ENCODING.media_type
    )


@pytest.mark.parametrize("shape", [(70, 100), (1, 16_384), (1, 70_000)])
def test_a_native_region_decodes_to_exactly_the_source_pixels(
    client: TestClient,
    tmp_path: Path,
    shape: tuple[int, int],
) -> None:
    """A native region decodes to exactly the source pixels.

    Seeded uint8 noise comes back value for value, including at edges past the limits WebP
    (16,383) and JPEG (65,500) accept.
    """
    height, width = shape
    path = tmp_path / "noise.tif"
    arr = (
        np.random.default_rng(2)
        .integers(0, 256, size=(height + 20, width + 30, 3))
        .astype(np.uint8)
    )
    tifffile.imwrite(str(path), arr)
    served = _served(
        client.get(
            "/api/images",
            params={
                "path": str(path),
                "display_pixels": DISPLAY,
                "region": [30, 20, 30 + width, 20 + height],
            },
        )
    )
    assert np.array_equal(served, arr[20:, 30:])


@pytest.mark.parametrize(
    "raster_shape,display,rows",
    [
        ((300, 400), 10_000, 100),
        ((300, 400), 15_000, 100),
        ((1, 100_000), 20_000, 1),
    ],
)
def test_the_view_routes_cells_and_the_image_routes_cap_are_one_derivation(
    client: TestClient,
    tmp_path: Path,
    raster_shape: tuple[int, int],
    display: int,
    rows: int,
) -> None:
    """The view route's native cells and the image route's area cap agree for one display.

    For one display, a view the view route serves by native cells is one the image route
    serves natively, and a view one column past the cap is a display read on both routes; each
    cell the view route names serves natively at its own size. The caps are a square, a
    non-square area, and one a single-row raster reaches.
    """
    from tcip_mcp.pipelines.overviews import build_overviews, overview_levels

    path = tmp_path / "raster.tif"
    tifffile.imwrite(str(path), np.full(raster_shape, 90, dtype=np.uint8))
    if overview_levels(raster_shape[1], raster_shape[0]):
        build_overviews(path)
    for width, native in (
        (display // rows, True),
        (display // rows + 1, False),
    ):
        rect = {"x0": 0, "y0": 0, "x1": width, "y1": rows}
        reads = client.get(
            "/api/images/view",
            params={"path": str(path), "display_pixels": display, **rect},
        ).json()["reads"]
        assert (
            all(r["level"] == 0 and r["name"] != "view" for r in reads)
            is native
        )
        served = client.get(
            "/api/images",
            params={
                "path": str(path),
                "display_pixels": display,
                "region": _region(rect),
            },
        )
        assert served.status_code == 200, served.text
        assert (
            served.headers["content-type"]
            == images_route._NATIVE_ENCODING.media_type
        ) is native
        if native:
            for read in reads:
                resp = client.get(
                    "/api/images",
                    params={
                        "path": str(path),
                        "display_pixels": display,
                        "region": _region(read),
                    },
                )
                assert (
                    resp.headers["content-type"]
                    == images_route._NATIVE_ENCODING.media_type
                )
                assert _served(resp).shape[:2] == (
                    read["y1"] - read["y0"],
                    read["x1"] - read["x0"],
                )


def test_the_view_and_labels_routes_measure_the_frame_the_image_routes_reader_opens(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """The view and labels routes measure a one-row RGB TIFF at the image route's frame.

    The view route and the annotation route measure it at the count the image route's plain
    read opens it at: the canvas frame the labels route answers is the view route's advertised
    extent, and every read the view route advertises lies inside the reader's frame and serves.
    """
    from tcip_mcp.pipelines.raster_source import SourceHeader

    images = image_dir(tmp_path / "rgb_ds", "2026-01-01")
    images.mkdir(parents=True)
    path = images / "rgb_row.tif"
    tifffile.imwrite(
        str(path),
        np.tile(np.array([20, 100, 220], dtype=np.uint8), (1, 200_000, 1)),
        photometric="rgb",
    )
    with SourceHeader(path).open_at_route_count() as raster:
        frame = (raster.width, raster.height)
    reads = client.get(
        "/api/images/view",
        params={
            "path": str(path),
            "display_pixels": 1_000_000,
            "x0": 0,
            "y0": 0,
            "x1": 200_000,
            "y1": 200_000,
        },
    ).json()["reads"]
    assert (
        max(r["nx1"] for r in reads),
        max(r["ny1"] for r in reads),
    ) == frame
    labels = client.get(
        "/api/annotate/labels", params={"image_path": str(path)}
    )
    assert labels.status_code == 200, labels.text
    assert (labels.json()["img_width"], labels.json()["img_height"]) == frame
    for read in reads:
        resp = client.get(
            "/api/images",
            params={
                "path": str(path),
                "display_pixels": 1_000_000,
                "level": read["level"],
                "region": _region(read),
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.headers[images_route.SERVED_SIZE_HEADER] == (
            f"{read['x1'] - read['x0']}x{read['y1'] - read['y0']}"
        )


def test_a_display_read_comes_off_the_overview_level_it_planned(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """A whole view too large for the cap is read off the overview level it planned.

    The read comes off the pyramid: with the native pixels zeroed after the build, the served
    view still carries the original content the pyramid holds, and a 10,000-pixel display is
    served at 100x100, still from the pyramid.
    """
    from tcip_mcp.pipelines.overviews import build_overviews

    path = tmp_path / "pyramid.tif"
    tifffile.imwrite(
        str(path), np.full((2048, 2048), 200, dtype=np.uint8), rowsperstrip=64
    )
    build_overviews(path)
    tifffile.imwrite(
        str(path), np.zeros((2048, 2048), dtype=np.uint8), rowsperstrip=64
    )
    served = _served(
        client.get(
            "/api/images",
            params={"path": str(path), "display_pixels": 1500 * 1500},
        )
    )
    assert served.mean() == pytest.approx(200, abs=1)

    coarsest = client.get(
        "/api/images", params={"path": str(path), "display_pixels": 10_000}
    )
    assert _served(coarsest).mean() == pytest.approx(200, abs=1)
    assert coarsest.headers[images_route.SERVED_SIZE_HEADER] == "100x100"


@pytest.mark.parametrize(
    "size,display,served_size",
    [
        ((4000, 3006), 2_073_600, "1000x751"),
        ((1920, 1086), 300_000, "728x411"),
    ],
)
def test_a_planned_display_read_is_one_the_raster_reader_admits(
    client: TestClient,
    tmp_path: Path,
    size: tuple[int, int],
    display: int,
    served_size: str,
) -> None:
    """The plan's whole-pixel output size is one the raster reader's own geometry admits.

    The rasters have aspect ratios a quarter or a fitted scale does not divide evenly.
    """
    from tcip_mcp.pipelines.overviews import build_overviews

    path = tmp_path / "odd.tif"
    tifffile.imwrite(
        str(path),
        np.full((size[1], size[0]), 120, dtype=np.uint8),
        rowsperstrip=64,
    )
    build_overviews(path)
    resp = client.get(
        "/api/images", params={"path": str(path), "display_pixels": display}
    )
    assert resp.status_code == 200, resp.text
    assert resp.headers[images_route.SERVED_SIZE_HEADER] == served_size


def test_a_one_pixel_wide_raster_reads_off_its_pyramid(
    client: TestClient, tmp_path: Path
) -> None:
    """A raster one pixel wide is read off the pyramid level that fits the cap.

    With its native pixels zeroed after the build, a 1 by 100,000 raster at a 20,000-pixel
    display is served at 1x12500 carrying the pyramid's 200.
    """
    from tcip_mcp.pipelines.overviews import build_overviews

    path = tmp_path / "tall.tif"
    tifffile.imwrite(str(path), np.full((100_000, 1), 200, dtype=np.uint8))
    build_overviews(path)
    tifffile.imwrite(str(path), np.zeros((100_000, 1), dtype=np.uint8))
    resp = client.get(
        "/api/images", params={"path": str(path), "display_pixels": 20_000}
    )
    assert _served(resp).mean() == pytest.approx(200, abs=1)
    assert resp.headers[images_route.SERVED_SIZE_HEADER] == "1x12500"


def test_a_palette_raster_reads_its_colors_off_its_pyramid(
    client: TestClient, tmp_path: Path
) -> None:
    """A palette raster's display read expands the planned pyramid level's own indices.

    The pyramid holds sampled indices: with the native indices zeroed after the build, the
    served view's color is one of the two colors the raster's indices name.
    """
    from tcip_mcp.pipelines.overviews import build_overviews

    colormap = np.zeros((3, 256), dtype=np.uint16)
    colormap[:, 5] = (200 * 256, 50 * 256, 100 * 256)
    colormap[:, 7] = (110 * 256, 115 * 256, 80 * 256)
    colormap[:, 9] = (20 * 256, 180 * 256, 60 * 256)
    # Columns alternate 5 and 9; index 7 is their mean in both index and color, so a pyramid that
    # averaged either way would hold 7.
    indices = np.tile(np.array([5, 9], dtype=np.uint8), (2048, 1024))
    path = tmp_path / "palette.tif"
    tifffile.imwrite(
        str(path),
        indices,
        photometric="palette",
        colormap=colormap,
        rowsperstrip=64,
    )
    build_overviews(path)
    tifffile.imwrite(
        str(path),
        np.zeros((2048, 2048), dtype=np.uint8),
        photometric="palette",
        colormap=colormap,
        rowsperstrip=64,
    )
    served = _served(
        client.get(
            "/api/images",
            params={"path": str(path), "display_pixels": 1500 * 1500},
        )
    )
    mean = served.mean(axis=(0, 1))
    assert np.allclose(mean, (200, 50, 100), atol=3) or np.allclose(
        mean, (20, 180, 60), atol=3
    ), mean


def test_region_statistics_do_not_depend_on_the_display_that_asked(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two displays asking for one region of one raster get the same stretch.

    The stretch is the one the bands route reports, even where the native sample covers only
    part of the raster.
    """
    monkeypatch.setattr(images_route, "_STATS_WINDOW_SIZE", 64)
    monkeypatch.setattr(images_route, "_STATS_MAX_WINDOWS", 4)
    path = tmp_path / "ramp.npy"
    rng = np.random.default_rng(9)
    ramp = np.arange(1024, dtype=np.uint16)[None, :, None] * 40
    np.save(
        str(path),
        (rng.integers(0, 1000, size=(1024, 1024, 4)) + ramp).astype(np.uint16),
    )

    def region(display: int) -> dict:
        return {
            "path": str(path),
            "bands": ["0", "1", "2"],
            "region": [0, 0, 64, 64],
            "display_pixels": display,
        }

    small = client.get("/api/images", params=region(65_536))
    images_route._stats_cache.clear()  # each display computes its statistics cold
    large = client.get("/api/images", params=region(262_144))
    assert _served(small).tolist() == _served(large).tolist()
    body = client.get("/api/images/bands", params={"path": str(path)}).json()
    assert body["sampled"] is True
    bounds = [(b["min"], b["max"]) for b in body["bands"][:3]]
    arr = np.load(str(path))
    _renders_as(small, _composite(arr[:64, :64], "minmax", bounds))


def _poisoned_pyramid(path: Path) -> None:
    """Write a 4096-pixel square raster of 200, build its pyramid, then zero its native pixels.

    The raster is uint8 in strips of 64 rows; ``build_overviews`` builds the pyramid, and the
    raster is then rewritten as zeros.
    """
    from tcip_mcp.pipelines.overviews import build_overviews

    tifffile.imwrite(
        str(path), np.full((4096, 4096), 200, dtype=np.uint8), rowsperstrip=64
    )
    build_overviews(path)
    tifffile.imwrite(
        str(path), np.zeros((4096, 4096), dtype=np.uint8), rowsperstrip=64
    )


def _paint_levels(path: Path, values: tuple[int, ...]) -> None:
    """Overwrite each level of ``path``'s uncompressed sidecar in place with one value apiece.

    The levels take ``values`` finest first.
    """
    from tcip_mcp.pipelines.overviews import overview_sidecar

    sidecar = overview_sidecar(path)
    with tifffile.TiffFile(str(sidecar)) as tif:
        pages = [
            (
                page.compression,
                list(zip(page.dataoffsets, page.databytecounts)),
            )
            for page in tif.pages
        ]
    assert len(pages) == len(values)
    with open(sidecar, "r+b") as fh:
        for (compression, strips), value in zip(pages, values):
            assert compression == 1, (
                "the fixture paints only an uncompressed sidecar"
            )
            for offset, count in strips:
                fh.seek(offset)
                fh.write(bytes([value]) * count)


@pytest.mark.parametrize(
    "crop,display,served_size",
    [
        ((0, 0, 1, 4096), 100, "1x100"),
        ((0, 0, 1, 4096), 600, "1x600"),
        ((1, 1, 1025, 1025), 512 * 512, "257x257"),
    ],
)
def test_a_crop_is_planned_on_the_window_the_reader_opens(
    client: TestClient,
    tmp_path: Path,
    crop: tuple[int, int, int, int],
    display: int,
    served_size: str,
) -> None:
    """A crop is planned on the window the raster reader opens.

    A one-pixel-wide crop is served at 1x100 or 1x600, and the offset crop (1, 1, 1025, 1025)
    at 257x257, its level window with the origin rounded down and the end rounded up; each
    carries the pyramid's 200.
    """
    path = tmp_path / "pyramid.tif"
    _poisoned_pyramid(path)
    resp = client.get(
        "/api/images",
        params={
            "path": str(path),
            "display_pixels": display,
            "region": list(crop),
        },
    )
    assert _served(resp).mean() == pytest.approx(200, abs=1)
    assert resp.headers[images_route.SERVED_SIZE_HEADER] == served_size


def test_a_scaled_view_is_served_by_tiles_of_its_planned_level(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """A view past the cap is answered with tiles of the overview level its read is planned off.

    Each tile is served at that level's resolution off that physical level (native 0, level 1
    50, level 2 200), and a pan answers the tiles it still covers with the same records and
    etags, beside new tiles.
    """
    path = tmp_path / "pyramid.tif"
    _poisoned_pyramid(path)
    _paint_levels(path, (50, 200))
    display = 512 * 512

    def tiles(x0: int, x1: int) -> list[dict]:
        resp = client.get(
            "/api/images/view",
            params={
                "path": str(path),
                "display_pixels": display,
                "x0": x0,
                "y0": 0,
                "x1": x1,
                "y1": 2048,
            },
        )
        assert resp.status_code == 200, resp.text
        return resp.json()["reads"]

    def fetch(tile: dict) -> httpx.Response:
        return client.get(
            "/api/images",
            params={
                "path": str(path),
                "display_pixels": display,
                "level": tile["level"],
                "region": _region(tile),
            },
        )

    first = tiles(0, 2048)
    assert first and {t["level"] for t in first} == {2}
    tile = first[0]
    served = fetch(tile)
    assert served.headers["content-type"] == "image/jpeg"
    assert _served(served).mean() == pytest.approx(200, abs=1)
    assert served.headers[images_route.SERVED_SIZE_HEADER] == (
        f"{tile['x1'] - tile['x0']}x{tile['y1'] - tile['y0']}"
    )
    assert (tile["nx1"] - tile["nx0"]) == (tile["x1"] - tile["x0"]) * 4

    panned = tiles(512, 2560)
    before = {(t["level"], t["name"]): t for t in first}
    after = {(t["level"], t["name"]): t for t in panned}
    shared = before.keys() & after.keys()
    assert shared and after.keys() - before.keys()
    for key in shared:
        assert after[key] == before[key]
        assert (
            fetch(after[key]).headers["etag"]
            == fetch(before[key]).headers["etag"]
        )


@pytest.mark.parametrize("shape", [(200_000, 1), (1, 200_000)])
def test_every_advertised_tile_of_a_skinny_raster_is_one_its_encoding_admits(
    client: TestClient,
    tmp_path: Path,
    shape: tuple[int, int],
) -> None:
    """A one-pixel level longer than the JPEG edge limit is tiled within that limit.

    The tiles are at the level's own resolution, and every tile the view route names serves; a
    hand-built tile of the whole level, within the area cap but past the edge limit, is refused
    naming the limit.
    """
    from tcip_mcp.pipelines.overviews import build_overviews

    path = tmp_path / "skinny.tif"
    tifffile.imwrite(str(path), np.full(shape, 120, dtype=np.uint8))
    build_overviews(path)
    height, width = shape
    reads = client.get(
        "/api/images/view",
        params={
            "path": str(path),
            "display_pixels": 100_000,
            "x0": 0,
            "y0": 0,
            "x1": width,
            "y1": height,
        },
    ).json()["reads"]
    assert reads and {r["level"] for r in reads} == {1}
    for read in reads:
        resp = client.get(
            "/api/images",
            params={
                "path": str(path),
                "display_pixels": 100_000,
                "level": 1,
                "region": _region(read),
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.headers[images_route.SERVED_SIZE_HEADER] == (
            f"{read['x1'] - read['x0']}x{read['y1'] - read['y0']}"
        )
        assert max(read["x1"] - read["x0"], read["y1"] - read["y0"]) <= 65_500

    whole_level = client.get(
        "/api/images",
        params={
            "path": str(path),
            "display_pixels": 100_000,
            "level": 1,
            "region": [0, 0, max(1, width // 2), max(1, height // 2)],
        },
    )
    assert whole_level.status_code == 400
    assert "65500" in whole_level.json()["detail"]


def test_a_stacked_tiff_the_reader_decodes_whole_is_offered_no_overview_tiles(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """A stacked TIFF whose first page carries overviews is offered no overview tiles.

    For a three-page stack, the view route answers with one display read of the view, and that
    read serves.
    """
    from tcip_mcp.pipelines.overviews import build_overviews, overview_dims

    path = tmp_path / "stack.tif"
    tifffile.imwrite(
        str(path),
        np.full((3, 2048, 2048), 90, dtype=np.uint8),
        photometric="minisblack",
    )
    build_overviews(path)
    assert overview_dims(path), "the fixture's first page must carry overviews"
    reads = client.get(
        "/api/images/view",
        params={
            "path": str(path),
            "display_pixels": 65_536,
            "x0": 0,
            "y0": 0,
            "x1": 2048,
            "y1": 2048,
        },
    ).json()["reads"]
    assert [(r["name"], r["level"]) for r in reads] == [("view", 0)]
    resp = client.get(
        "/api/images",
        params={
            "path": str(path),
            "display_pixels": 65_536,
            "region": _region(reads[0]),
        },
    )
    assert resp.status_code == 200, resp.text


def test_a_tile_request_is_refused_where_the_raster_has_no_such_level(
    client: TestClient,
    tmp_path: Path,
) -> None:
    path = tmp_path / "quads.tif"
    _quadrant_rgb(path)
    resp = client.get(
        "/api/images",
        params={
            "path": str(path),
            "display_pixels": DISPLAY,
            "level": 1,
            "region": [0, 0, 10, 10],
        },
    )
    assert resp.status_code == 400
    assert "no overview level 1" in resp.json()["detail"]
    native = client.get(
        "/api/images",
        params={
            "path": str(path),
            "display_pixels": DISPLAY,
            "region": [0, 0, 10, 10],
        },
    )
    assert native.status_code == 200


# ── Cache keys ───────────────────────────────────────────────────────────────────────────


def test_the_etag_varies_with_the_region(
    client: TestClient, tmp_path: Path
) -> None:
    path = tmp_path / "quads.tif"
    _quadrant_rgb(path)
    whole = client.get(
        "/api/images", params={"path": str(path), "display_pixels": DISPLAY}
    )
    left = client.get(
        "/api/images",
        params={
            "path": str(path),
            "display_pixels": DISPLAY,
            "region": [0, 0, 200, 300],
        },
    )
    right = client.get(
        "/api/images",
        params={
            "path": str(path),
            "display_pixels": DISPLAY,
            "region": [200, 0, 400, 300],
        },
    )
    tags = {whole.headers["etag"], left.headers["etag"], right.headers["etag"]}
    assert len(tags) == 3


def test_the_etag_changes_when_an_overview_sidecar_appears(
    client: TestClient, tmp_path: Path
) -> None:
    """The etag changes when an overview build lands between two requests."""
    from tcip_mcp.pipelines.overviews import build_overviews

    path = tmp_path / "wide.tif"
    _wide_raster(path)
    before = client.get(
        "/api/images", params={"path": str(path), "display_pixels": DISPLAY}
    )
    assert before.status_code == 200
    build_overviews(path)
    after = client.get(
        "/api/images", params={"path": str(path), "display_pixels": DISPLAY}
    )
    assert after.status_code == 200
    assert after.headers["etag"] != before.headers["etag"]


# ── The plain-serve rule ─────────────────────────────────────────────────────────────────


def test_a_uint8_raster_serves_its_own_pixels_with_no_stretch(
    client: TestClient, tmp_path: Path
) -> None:
    """A uint8 raster with no band selection is served at its own levels with no stretch.

    The response carries no stats-source and no display-bounds header.
    """
    path = tmp_path / "flat.tif"
    tifffile.imwrite(
        str(path), np.full((32, 40, 3), (100, 120, 140), dtype=np.uint8)
    )
    resp = client.get(
        "/api/images", params={"path": str(path), "display_pixels": DISPLAY}
    )
    served = _served(resp)
    assert np.allclose(served.mean(axis=(0, 1)), (100, 120, 140), atol=3)
    assert "x-tcip-stats-source" not in resp.headers
    assert "x-tcip-display-bounds" not in resp.headers


def test_a_uint16_raster_serves_on_its_dtypes_full_scale(
    client: TestClient, tmp_path: Path
) -> None:
    """A serve of a uint16 raster at the ``none`` stretch divides by the dtype's own ceiling.

    A half-scale single-band frame reads as mid-gray.
    """
    path = tmp_path / "half.tif"
    tifffile.imwrite(str(path), np.full((32, 40), 32768, dtype=np.uint16))
    resp = client.get(
        "/api/images",
        params={"path": str(path), "display_pixels": DISPLAY, "stretch": "none"},
    )
    served = _served(resp)
    assert np.allclose(served.mean(axis=(0, 1)), 127.5, atol=3)


def test_a_multi_band_uint16_raster_serves_on_that_same_scale(
    client: TestClient, tmp_path: Path
) -> None:
    path = tmp_path / "half_rgb.tif"
    tifffile.imwrite(str(path), np.full((32, 40, 3), 32768, dtype=np.uint16))
    served = _served(
        client.get(
            "/api/images",
            params={"path": str(path), "display_pixels": DISPLAY, "stretch": "none"},
        )
    )
    assert np.allclose(served.mean(axis=(0, 1)), 127.5, atol=3)


def test_a_float_regions_full_scale_is_the_rasters_own_maximum(
    client: TestClient, tmp_path: Path
) -> None:
    """A float region at the ``none`` stretch divides by the maximum the raster's own sample found.

    The left half, at 100, of a raster reaching 1000 serves at 100/1000 of full scale: a dim
    corner stays dim.
    """
    path = tmp_path / "float.tif"
    from tests._producer_fixtures import painted_array

    arr = painted_array(
        40,
        32,
        [((0, 0, 20, 32), 100.0), ((20, 0, 40, 32), 1000.0)],
        background=0.0,
        mode="F",
    )
    tifffile.imwrite(str(path), arr)
    resp = client.get(
        "/api/images",
        params={
            "path": str(path),
            "display_pixels": DISPLAY,
            "region": [0, 0, 20, 32],
            "stretch": "none",
        },
    )
    served = _served(resp)
    assert np.allclose(
        served.mean(axis=(0, 1)), 100.0 / 1000.0 * 255.0, atol=3
    )


def test_a_non_positive_float_band_renders_black(
    client: TestClient, tmp_path: Path
) -> None:
    """A float band with no positive data renders black at the ``none`` stretch."""
    path = tmp_path / "negative.tif"
    arr = (
        -np.abs(np.random.default_rng(0).standard_normal((32, 40))) - 1.0
    ).astype(np.float32)
    tifffile.imwrite(str(path), arr)
    resp = client.get(
        "/api/images",
        params={"path": str(path), "display_pixels": DISPLAY, "stretch": "none"},
    )
    served = _served(resp)
    assert np.allclose(served, 0, atol=2)


def test_a_single_band_raster_serves_as_replicated_gray(
    client: TestClient, tmp_path: Path
) -> None:
    path = tmp_path / "gray.tif"
    tifffile.imwrite(str(path), np.full((32, 40), 90, dtype=np.uint8))
    served = _served(
        client.get(
            "/api/images",
            params={"path": str(path), "display_pixels": DISPLAY},
        )
    )
    assert served.shape == (32, 40, 3)
    assert np.allclose(served.mean(axis=(0, 1)), 90, atol=3)


def test_a_four_band_raster_serves_as_plain_rgb_with_the_fourth_band_dropped(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """The alpha band of an RGBA raster is dropped, not composited and not stretched.

    The file's own colors reach the viewer unshifted.
    """
    path = tmp_path / "rgba.tif"
    arr = np.zeros((32, 40, 4), dtype=np.uint8)
    arr[..., :3] = (100, 120, 140)
    arr[..., 3] = 255
    tifffile.imwrite(str(path), arr)
    resp = client.get(
        "/api/images", params={"path": str(path), "display_pixels": DISPLAY}
    )
    served = _served(resp)
    assert served.shape == (32, 40, 3)
    assert np.allclose(served.mean(axis=(0, 1)), (100, 120, 140), atol=3)


def test_a_five_band_raster_composites_its_first_three_bands(
    client: TestClient, tmp_path: Path
) -> None:
    """A default request for a five-band raster is a composite of its first three bands.

    Past the band counts an RGB reading covers, the composite is a stretched render of the first
    three bands between the served array's own bounds.
    """
    path = tmp_path / "five.tif"
    arr = _multiband(path, channels=5)
    resp = client.get(
        "/api/images", params={"path": str(path), "display_pixels": DISPLAY}
    )
    _renders_as(resp, _composite(arr, "minmax"))


def _pinned_stretch_raster(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Write a non-square 4-band raster and return it with its hand-worked ``minmax`` pixels.

    The three selected bands span different ranges, and each value in them sits on a quarter
    of its band's span, so the display byte each one is owed is an exact number (0, 63, 127,
    191, 255 after the truncating uint8 cast). Returns the raster and the ``3,0,1`` composite
    it is owed.
    """
    bands = [
        [[0, 100, 200], [300, 400, 400]],  # spans 0..400
        [[1400, 1300, 1200], [1100, 1000, 1400]],  # spans 1000..1400
        [[7, 8, 9], [10, 11, 12]],  # never selected: band choice has to matter
        [[20, 40, 60], [80, 100, 20]],  # spans 20..100
    ]
    block = np.stack([np.array(b, dtype="uint16") for b in bands], axis=-1)
    arr = np.tile(block, (2, 2, 1))
    owed_block = np.stack(
        [
            np.array([[0, 63, 127], [191, 255, 0]], dtype="uint8"),  # band 3
            np.array([[0, 63, 127], [191, 255, 255]], dtype="uint8"),  # band 0
            np.array([[255, 191, 127], [63, 0, 255]], dtype="uint8"),  # band 1
        ],
        axis=-1,
    )
    tifffile.imwrite(str(path), arr)
    return arr, np.tile(owed_block, (2, 2, 1))


def test_the_served_composite_is_the_shared_display_primitives_own_pixels(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """The route encodes ``composite_display_rgb``'s output byte for byte.

    ``composite_display_rgb`` itself equals the hand-worked ``minmax`` pixels of a known raster.
    """
    from tcip_mcp.pipelines.band_stats import composite_display_rgb

    path = tmp_path / "capture.tif"
    arr, owed = _pinned_stretch_raster(path)
    composed = composite_display_rgb(arr, (3, 0, 1), "minmax")
    assert composed.tolist() == owed.tolist()

    resp = client.get(
        "/api/images",
        params={
            "path": str(path),
            "display_pixels": DISPLAY,
            "bands": ["3", "0", "1"],
            "stretch": "minmax",
        },
    )
    _renders_as(resp, composed)


# ── Stretch bounds ───────────────────────────────────────────────────────────────────────


def test_two_regions_of_one_raster_stretch_against_the_same_bounds(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """Two regions of one raster render between the whole raster's bounds.

    Each half renders between the whole raster's bounds, not its own.
    """
    path = tmp_path / "capture.tif"
    arr = _multiband(path)
    raster_bounds = [
        (float(arr[:, :, i].min()), float(arr[:, :, i].max()))
        for i in range(3)
    ]
    left = client.get(
        "/api/images",
        params={
            "path": str(path),
            "display_pixels": DISPLAY,
            "bands": ["0", "1", "2"],
            "region": [0, 0, 20, 24],
        },
    )
    right = client.get(
        "/api/images",
        params={
            "path": str(path),
            "display_pixels": DISPLAY,
            "bands": ["0", "1", "2"],
            "region": [20, 0, 40, 24],
        },
    )
    _renders_as(left, _composite(arr[:, :20], "minmax", raster_bounds))
    _renders_as(right, _composite(arr[:, 20:], "minmax", raster_bounds))


def test_a_whole_view_renders_between_the_bounds_of_the_array_it_served(
    client: TestClient, tmp_path: Path
) -> None:
    path = tmp_path / "capture.tif"
    arr = _multiband(path)
    resp = client.get(
        "/api/images",
        params={
            "path": str(path),
            "display_pixels": DISPLAY,
            "bands": ["0", "1", "2"],
        },
    )
    _renders_as(resp, _composite(arr, "minmax"))


def test_a_composited_non_positive_float_band_renders_black_beside_its_lit_bands(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """A composited band with no positive data renders black under ``stretch=none``.

    The two bands with positive data average above 50 in the composite, and the route serves
    that composite byte for byte.
    """
    path = tmp_path / "capture.tif"
    rng = np.random.default_rng(5)
    arr = rng.uniform(1.0, 100.0, size=(24, 40, 3)).astype(np.float32)
    arr[:, :, 0] = -np.abs(arr[:, :, 0]) - 1.0
    tifffile.imwrite(str(path), arr)
    owed = _composite(arr, "none")
    assert (owed[:, :, 0] == 0).all()
    assert owed[:, :, 1].mean() > 50 and owed[:, :, 2].mean() > 50
    _renders_as(
        client.get(
            "/api/images",
            params={
                "path": str(path),
                "display_pixels": DISPLAY,
                "bands": ["0", "1", "2"],
                "stretch": "none",
            },
        ),
        owed,
    )


def test_a_percent_clip_region_stretches_between_the_cached_cut_points(
    client: TestClient,
    tmp_path: Path,
) -> None:
    path = tmp_path / "capture.tif"
    arr = _multiband(path)
    resp = client.get(
        "/api/images",
        params={
            "path": str(path),
            "display_pixels": DISPLAY,
            "bands": ["0", "1", "2"],
            "stretch": "percent_clip",
            "region": [0, 0, 20, 24],
        },
    )
    stats = _cached_stats(path)
    _renders_as(
        resp, _composite(arr[:, :20], "percent_clip", stats.clip_bounds[:3])
    )


def _cached_stats(path: Path) -> _RasterStats:
    """Return the stats the route cached for ``path`` read at its own band count.

    Fails an assertion when the route cached no stats under that key.
    """
    from tcip_mcp.pipelines.raster_source import SourceHeader

    header = SourceHeader(path)
    stats = images_route._stats_cache.get(
        (*images_route._source_identity(header), header.channels)
    )
    assert stats is not None, "the route left no stats for this raster"
    return stats


def _five_band_float_with_one_nan(
    path: Path, *, height: int = 24, width: int = 40
) -> np.ndarray:
    """Write and return a 5-band float32 raster whose band 0 holds one NaN pixel.

    The NaN sits at band 0's top-left pixel, and the rest of the raster is a seeded uniform
    draw in [0, 1000).
    """
    rng = np.random.default_rng(7)
    arr = rng.uniform(0, 1000, size=(height, width, 5)).astype(np.float32)
    arr[0, 0, 0] = np.nan
    tifffile.imwrite(str(path), arr)
    return arr


def test_a_nan_pixel_still_serves_the_raster(
    client: TestClient, tmp_path: Path
) -> None:
    """A five-band float raster holding one NaN pixel still serves at its own size."""
    path = tmp_path / "nan.tif"
    _five_band_float_with_one_nan(path)
    resp = client.get(
        "/api/images",
        params={
            "path": str(path),
            "display_pixels": DISPLAY,
            "bands": ["0", "1", "2"],
        },
    )
    assert _served(resp).shape == (24, 40, 3)


# ── /api/images/bands ────────────────────────────────────────────────────────────────────


def test_get_bands_reports_exact_bounds_when_the_sample_covered_every_pixel(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """The bands route reports exact bounds for a raster its window budget covers whole.

    The response says the sample was not partial, covering every pixel at seed 0, and gives
    each band's own min and max.
    """
    path = tmp_path / "capture.tif"
    arr = _multiband(path)
    resp = client.get("/api/images/bands", params={"path": str(path)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["band_count"] == 4
    assert body["sampled"] is False
    assert body["pixel_fraction"] == 1.0
    assert body["seed"] == 0
    assert [b["min"] for b in body["bands"]] == [
        float(arr[:, :, i].min()) for i in range(4)
    ]
    assert [b["max"] for b in body["bands"]] == [
        float(arr[:, :, i].max()) for i in range(4)
    ]
    assert {b["dtype"] for b in body["bands"]} == {"uint16"}


def test_get_bands_says_so_when_it_read_only_part_of_the_raster(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bands route reports a sample when its window budget is below the raster's grid.

    The response marks the bounds sampled, with a pixel fraction under one and its seed. The
    raster is a numpy stack.
    """
    monkeypatch.setattr(images_route, "_STATS_WINDOW_SIZE", 4)
    monkeypatch.setattr(images_route, "_STATS_MAX_WINDOWS", 2)
    path = tmp_path / "capture.npy"
    np.save(
        str(path), np.random.default_rng(3).integers(0, 1000, size=(24, 40, 4))
    )
    body = client.get("/api/images/bands", params={"path": str(path)}).json()
    assert body["sampled"] is True
    assert 0.0 < body["pixel_fraction"] < 1.0
    assert body["seed"] == 0


def test_get_bands_never_loads_a_tiff_through_load_image(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Describing a windowed TIFF's bands never calls ``image_utils.load_image``.

    ``load_image`` is the whole-image load.
    """
    from tcip_mcp.pipelines import image_utils

    calls: list = []
    real = image_utils.load_image

    def counted(
        *args: str | Path | BandGroupRef | int,
        **kwargs: str | Path | BandGroupRef | int,
    ) -> Image.Image | np.ndarray:
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(image_utils, "load_image", counted)
    path = tmp_path / "capture.tif"
    _multiband(path)
    resp = client.get("/api/images/bands", params={"path": str(path)})
    assert resp.status_code == 200
    assert calls == []


def test_cold_band_stats_open_the_raster_once(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stacked numpy raster is opened once for its cold band stats.

    The bands route answers four bands, every pixel sampled, with one open of the raster.
    """
    from tcip_mcp.pipelines import raster_source

    opened: list = []
    real = raster_source.SourceHeader.open

    def counted(*args: SourceHeader | int, **kwargs: int) -> RasterSource:
        opened.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(raster_source.SourceHeader, "open", counted)
    path = tmp_path / "capture.npz"
    np.savez(
        str(path),
        bands=np.random.default_rng(3).integers(0, 1000, size=(24, 40, 4)),
    )
    body = client.get("/api/images/bands", params={"path": str(path)}).json()
    assert body["band_count"] == 4 and body["pixel_fraction"] == 1.0
    assert len(opened) == 1


def test_a_described_raster_is_described_again_without_an_open(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asking for one raster's bands twice opens it once in total."""
    from tcip_mcp.pipelines import raster_source

    opened: list = []
    real = raster_source.SourceHeader.open

    def counted(*args: SourceHeader | int, **kwargs: int) -> RasterSource:
        opened.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(raster_source.SourceHeader, "open", counted)
    path = tmp_path / "capture.tif"
    _multiband(path)
    first = client.get("/api/images/bands", params={"path": str(path)}).json()
    assert (
        client.get("/api/images/bands", params={"path": str(path)}).json()
        == first
    )
    assert len(opened) == 1


@pytest.mark.parametrize("bands", [5, 3], ids=["with_stats", "count_only"])
def test_a_warm_description_reads_nothing_off_the_file(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bands: int,
) -> None:
    """A repeated description of an unchanged container answers from the caches.

    The band count comes from its own cache, the statistics (where the count calls for them)
    from theirs, so not even the array that states the count is loaded.
    """
    path = tmp_path / "stack.npz"
    np.savez(str(path), image=np.zeros((24, 40, bands), dtype=np.uint8))
    first = client.get("/api/images/bands", params={"path": str(path)}).json()
    assert first["band_count"] == bands
    assert bool(first["bands"]) == (bands > 3)

    loads: list = []
    real = np.load

    def counted(
        *args: str, **kwargs: str
    ) -> np.ndarray | np.lib.npyio.NpzFile:
        loads.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(np, "load", counted)
    assert (
        client.get("/api/images/bands", params={"path": str(path)}).json()
        == first
    )
    assert loads == []


def test_one_reading_of_a_raster_is_sampled_once_whichever_route_asks_first(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A description and a plain region of one raster share one statistics computation.

    The float four-band raster's statistics are computed and cached once for both requests.
    """
    path = tmp_path / "float4.tif"
    rng = np.random.default_rng(5)
    tifffile.imwrite(str(path), rng.random((24, 40, 4)).astype(np.float32))
    computed: list = []
    real = images_route._stats_of

    def counted(*args: RasterSource, **kwargs: RasterSource) -> _RasterStats:
        computed.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(images_route, "_stats_of", counted)
    assert (
        client.get("/api/images/bands", params={"path": str(path)}).status_code
        == 200
    )
    resp = client.get(
        "/api/images",
        params={
            "path": str(path),
            "display_pixels": DISPLAY,
            "region": [0, 0, 20, 24],
        },
    )
    assert resp.status_code == 200, resp.text
    assert len(computed) == 1
    assert len(images_route._stats_cache) == 1


def test_a_band_group_rewritten_to_name_another_member_is_described_afresh(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """A band group rewritten to name another member is described afresh.

    With band A rewritten from a file of 10 to one of 90 with the same size and timestamp, the
    described maximum of band A goes from 10 to 90.
    """
    import os

    from tests._producer_fixtures import write_band_group

    d = image_root(tmp_path)
    d.mkdir()
    for name, value in (("cap_a", 10), ("cap_b", 50), ("cap_c", 90)):
        np.save(
            str(d / f"{name}.npy"), np.full((8, 12), value, dtype=np.uint8)
        )
    stamp = (d / "cap_a.npy").stat().st_mtime_ns
    os.utime(d / "cap_c.npy", ns=(stamp, stamp))
    manifest = write_band_group(
        d,
        "cap",
        {"A": d / "cap_a.npy", "B": d / "cap_b.npy"},
        central_wavelength_nm=None,
    )
    first = client.get(
        "/api/images/bands", params={"path": str(manifest)}
    ).json()
    assert first["bands"][0]["max"] == 10

    write_band_group(
        d,
        "cap",
        {"A": d / "cap_c.npy", "B": d / "cap_b.npy"},
        central_wavelength_nm=None,
    )
    again = client.get(
        "/api/images/bands", params={"path": str(manifest)}
    ).json()
    assert again["bands"][0]["max"] == 90


def test_a_cold_region_stretch_samples_through_the_reader_that_served_the_region(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cold region stretch opens the raster once for the whole request.

    The region is served, and its raster's statistics are cached once.
    """
    from tcip_mcp.pipelines import raster_source

    opened: list = []
    real = raster_source.SourceHeader.open

    def counted(*args: SourceHeader | int, **kwargs: int) -> RasterSource:
        opened.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(raster_source.SourceHeader, "open", counted)
    path = tmp_path / "capture.tif"
    _multiband(path)
    resp = client.get(
        "/api/images",
        params={
            "path": str(path),
            "display_pixels": DISPLAY,
            "bands": ["0", "1", "2"],
            "region": [0, 0, 20, 24],
        },
    )
    assert resp.status_code == 200, resp.text
    assert len(images_route._stats_cache) == 1
    assert len(opened) == 1


def test_the_view_route_reads_one_header_and_opens_the_raster_once(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The view route reads one header and opens the raster once to plan a view.

    Planning a view off a pyramid takes one header read and one GDAL open of the raster
    itself; a view within the cap takes one header read and no open.
    """
    from tcip_mcp.pipelines import raster_source
    from tcip_mcp.pipelines.overviews import build_overviews

    path = tmp_path / "wide.tif"
    _wide_raster(path)
    build_overviews(path)
    headers: list = []
    rasters: list = []
    real_header, real_open = (
        raster_source.tiff_header,
        raster_source.open_gdal_dataset,
    )

    def header(tif: Path) -> _TiffHeader:
        headers.append(tif)
        return real_header(tif)

    def gdal(
        dataset_path: str | Path, overview: int | None = None
    ) -> DatasetReader:
        if overview is None:
            rasters.append(dataset_path)
        return real_open(dataset_path, overview)

    monkeypatch.setattr(raster_source, "tiff_header", header)
    monkeypatch.setattr(raster_source, "open_gdal_dataset", gdal)
    reads = client.get(
        "/api/images/view",
        params={
            "path": str(path),
            "display_pixels": SMALL_DISPLAY,
            "x0": 0,
            "y0": 0,
            "x1": 5000,
            "y1": 64,
        },
    ).json()["reads"]
    assert reads and all(r["level"] > 0 for r in reads)
    assert len(headers) == 1
    assert len(rasters) == 1

    headers.clear()
    rasters.clear()
    native = client.get(
        "/api/images/view",
        params={
            "path": str(path),
            "display_pixels": DISPLAY,
            "x0": 0,
            "y0": 0,
            "x1": 100,
            "y1": 64,
        },
    ).json()["reads"]
    assert native and all(r["level"] == 0 for r in native)
    assert len(headers) == 1
    assert rasters == []


def test_get_bands_reads_an_oversized_rasters_stats_off_its_overviews(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bands route reads an oversized raster's stats off its overviews.

    Past the native-sampling budget the stats come off the overviews: each band's maximum is
    positive though the native pixels are zeroed, and the response states the overview size,
    1024 by 13 for the 5000x64 frame, in place of a pixel fraction and seed.
    """
    from tcip_mcp.pipelines.overviews import build_overviews

    monkeypatch.setattr(images_route, "_STATS_MAX_WINDOWS", 1)
    path = tmp_path / "wide_ms.tif"
    arr = _wide_multiband(path)
    build_overviews(path)
    tifffile.imwrite(
        str(path),
        np.zeros_like(arr),
        rowsperstrip=8,
        photometric="minisblack",
        planarconfig="contig",
    )

    body = client.get("/api/images/bands", params={"path": str(path)}).json()
    assert body["band_count"] == 4
    assert body["sampled"] is False
    assert body["overview_size"] == [1024, 13]
    assert body["pixel_fraction"] is None and body["seed"] is None
    assert all(0 <= b["min"] <= b["max"] <= 255 for b in body["bands"])
    assert all(b["max"] > 0 for b in body["bands"]), (
        "the native pixels, zeroed, were read"
    )


def test_an_oversized_raster_without_overviews_names_the_build_endpoint_for_its_stats(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bands route refuses an oversized raster without overviews, naming the build endpoint.

    The refusal carries the overviews-required error header and names
    ``POST /api/images/overviews``.
    """
    monkeypatch.setattr(images_route, "_STATS_MAX_WINDOWS", 1)
    path = tmp_path / "wide_ms.tif"
    _wide_multiband(path)

    resp = client.get("/api/images/bands", params={"path": str(path)})
    assert resp.status_code == 400
    assert (
        resp.headers[images_route.IMAGE_ERROR_HEADER]
        == images_route.OVERVIEWS_REQUIRED
    )
    assert "POST /api/images/overviews" in resp.json()["detail"]


def test_a_raster_within_the_sampling_budget_keeps_reading_native_pixels(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raster within the sampling budget keeps its native-pixel stats.

    Under the threshold the response keeps each band's exact maximum and the sampling facts and
    reports no overview size.
    """
    monkeypatch.setattr(images_route, "_STATS_MAX_WINDOWS", 1)
    path = tmp_path / "capture.tif"
    arr = _multiband(path)
    body = client.get("/api/images/bands", params={"path": str(path)}).json()
    assert body["sampled"] is False
    assert body["pixel_fraction"] == 1.0 and body["seed"] == 0
    assert body["overview_size"] is None
    assert [b["max"] for b in body["bands"]] == [
        float(arr[:, :, i].max()) for i in range(4)
    ]


def test_a_region_of_an_oversized_raster_stretches_by_its_overview_bounds(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tcip_mcp.pipelines.overviews import build_overviews

    monkeypatch.setattr(images_route, "_STATS_MAX_WINDOWS", 1)
    path = tmp_path / "wide_ms.tif"
    arr = _wide_multiband(path)
    build_overviews(path)

    resp = client.get(
        "/api/images",
        params={
            "path": str(path),
            "display_pixels": SMALL_DISPLAY,
            "bands": ["0", "1", "2"],
            "region": [0, 0, 256, 64],
        },
    )
    stats = _cached_stats(path)
    assert stats.overview_size == (1024, 13)
    bounds = [(r.minimum, r.maximum) for r in stats.ranges[:3]]
    _renders_as(resp, _composite(arr[:, :256], "minmax", bounds))


def test_get_bands_carries_the_band_interpretations_a_backend_reads(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """The bands route carries the band interpretations a backend reads.

    A composited raster's bands are reported with what each holds where a backend reads it and
    ``None`` where nothing does: a 16-bit RGBA TIFF names its colors and its alpha.
    """
    rgba = tmp_path / "rgba.tif"
    tifffile.imwrite(
        str(rgba), np.zeros((32, 40, 4), dtype=np.uint16), rowsperstrip=8
    )
    body = client.get("/api/images/bands", params={"path": str(rgba)}).json()
    assert [b["interpretation"] for b in body["bands"]] == [
        "red",
        "green",
        "blue",
        "alpha",
    ]

    stack = tmp_path / "stack.npy"
    np.save(str(stack), np.zeros((32, 40, 4), dtype=np.uint8))
    images_route._stats_cache.clear()
    plain = client.get("/api/images/bands", params={"path": str(stack)}).json()
    assert plain["band_count"] == 4
    assert [b["interpretation"] for b in plain["bands"]] == [None] * 4


def test_get_bands_keeps_the_three_band_early_return(
    client: TestClient, tmp_path: Path
) -> None:
    """The bands route answers an ordinary RGB frame with its band count and its verdict alone.

    An ordinary RGB frame displays as its own pixels and has no per-band symbology to show, so the
    response carries no band entries and no sampling facts.
    """
    from tcip_mcp.pipelines.band_stats import DEFAULT_STRETCH

    path = write_image(tmp_path / "plain.jpg", (40, 32), (5, 5, 5))
    body = client.get("/api/images/bands", params={"path": str(path)}).json()
    assert body == {
        "band_count": 3,
        "bands": [],
        "plain": True,
        "default_bands": ["0", "1", "2"],
        "default_stretch": DEFAULT_STRETCH,
        "pixel_fraction": None,
        "seed": None,
        "overview_size": None,
        "sampled": False,
    }


# ── Overview builds ──────────────────────────────────────────────────────────────────────


def _await_job(
    client: TestClient, job_id: str, timeout: float = 120.0
) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(
            "/api/images/overviews/status", params={"job_id": job_id}
        ).json()
        if not body["live"]:
            return body
        time.sleep(0.05)
    raise AssertionError(
        f"overview job {job_id} did not finish within {timeout}s"
    )


def test_a_scaled_read_of_an_oversized_raster_without_overviews_names_the_build_endpoint(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """A scaled read of an oversized raster without overviews is refused, naming the endpoint.

    The refusal carries the overviews-required error header and names
    ``POST /api/images/overviews``.
    """
    path = tmp_path / "wide.tif"
    _wide_raster(path)
    resp = client.get(
        "/api/images",
        params={"path": str(path), "display_pixels": SMALL_DISPLAY},
    )
    assert resp.status_code == 400
    assert (
        resp.headers[images_route.IMAGE_ERROR_HEADER]
        == images_route.OVERVIEWS_REQUIRED
    )
    assert "POST /api/images/overviews" in resp.json()["detail"]


def test_a_refused_read_of_an_oversized_raster_leaves_its_file_closed(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """A read refused for want of overviews leaves no handle on the raster: it can be removed.

    On Windows a file a reader still holds open refuses its removal.
    """
    path = tmp_path / "wide.tif"
    _wide_raster(path)
    resp = client.get(
        "/api/images",
        params={"path": str(path), "display_pixels": SMALL_DISPLAY},
    )
    assert resp.status_code == 400

    os.remove(path)
    assert not path.exists()


def test_the_overview_job_makes_that_same_request_servable(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """The overview job makes a refused oversized view servable.

    The view is refused 400; the overview job completes, writing a valid sidecar of the bytes
    it reports, and the same view then serves within the display's pixel count.
    """
    path = tmp_path / "wide.tif"
    _wide_raster(path)
    view = {"path": str(path), "display_pixels": SMALL_DISPLAY}
    assert client.get("/api/images", params=view).status_code == 400

    started = client.post("/api/images/overviews", json={"path": str(path)})
    assert started.status_code == 200
    job = _await_job(client, started.json()["job_id"])
    assert job["status"] == "completed", job

    from tcip_mcp.pipelines.overviews import overview_sidecar, sidecar_valid

    assert job["bytes_written"] == overview_sidecar(path).stat().st_size
    assert sidecar_valid(path)
    served = _served(client.get("/api/images", params=view))
    assert served.shape[0] * served.shape[1] <= SMALL_DISPLAY


def test_a_build_request_joins_the_one_already_running_for_that_raster(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """A build request for a raster whose build is running joins that build."""
    path = tmp_path / "wide.tif"
    _wide_raster(path)
    running = images_route.OverviewJob(
        job_id="ovr-running", path=str(path), status="running"
    )
    images_route._overview_registry.launch(running, lambda _job: None)
    try:
        joined = client.post(
            "/api/images/overviews", json={"path": str(path)}
        ).json()
        assert joined["job_id"] == "ovr-running"
        assert joined["status"] == "running"
    finally:
        images_route._overview_registry.jobs.pop(running.job_id, None)


def test_a_build_on_an_unreadable_raster_reaches_a_terminal_failure(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """A build that cannot open its raster ends as a recorded failure.

    The job's status reads failed, and it carries an error.
    """
    path = tmp_path / "broken.tif"
    path.write_bytes(b"this is not a raster")
    started = client.post("/api/images/overviews", json={"path": str(path)})
    assert started.status_code == 200
    job = _await_job(client, started.json()["job_id"], timeout=30.0)
    assert job["status"] == "failed"
    assert job["error"]


def test_an_overview_job_id_that_does_not_exist_is_a_404(
    client: TestClient,
) -> None:
    assert (
        client.get(
            "/api/images/overviews/status", params={"job_id": "ovr-nope"}
        ).status_code
        == 404
    )


def test_a_deep_zoom_region_within_the_cap_is_served_without_overviews(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """A region of a raster past the pyramid floor that fits the cap serves without overviews."""
    path = tmp_path / "wide.tif"
    _wide_raster(path)
    served = _served(
        client.get(
            "/api/images",
            params={
                "path": str(path),
                "display_pixels": SMALL_DISPLAY,
                "region": [0, 0, 256, 64],
            },
        )
    )
    assert served.shape == (64, 256, 3)


# ── Rendered-variant cache: the version key ───────────────────────────────────────────────


def test_a_render_cached_under_an_older_version_key_is_not_reused(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A render cached under an older ``RENDER_CACHE_VERSION`` is not reused.

    After a version bump the render is served under a new etag and cached as a second entry.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(images_route, "_render_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(images_route, "RENDER_CACHE_VERSION", 1)

    path = tmp_path / "flat.tif"
    tifffile.imwrite(
        str(path), np.full((32, 40, 3), (10, 20, 30), dtype=np.uint8)
    )

    whole = f"*{images_route._DISPLAY_ENCODING.suffix}"
    first = client.get(
        "/api/images", params={"path": str(path), "display_pixels": DISPLAY}
    )
    assert first.status_code == 200
    assert len(list(cache_dir.glob(whole))) == 1

    monkeypatch.setattr(images_route, "RENDER_CACHE_VERSION", 2)
    second = client.get(
        "/api/images", params={"path": str(path), "display_pixels": DISPLAY}
    )
    assert second.status_code == 200
    assert second.headers["etag"] != first.headers["etag"]
    assert len(list(cache_dir.glob(whole))) == 2


# ── Rendered-variant cache: byte-budget LRU ──────────────────────────────────────────────

_SUFFIXES = (
    images_route._DISPLAY_ENCODING.suffix,
    images_route._NATIVE_ENCODING.suffix,
)


def _cache_entry(cache_dir: Path, name: str, size: int, mtime: float) -> Path:
    """Write a ``size``-byte cache entry ``name`` and an empty header sidecar to ``cache_dir``.

    The entry takes the display suffix when ``name`` ends in an even digit and the native
    suffix when odd, and its modification time is set to ``mtime``.
    """
    image = cache_dir / f"{name}{_SUFFIXES[int(name[-1]) % 2]}"
    image.write_bytes(b"\xff" * size)
    (cache_dir / f"{name}.json").write_text("{}", encoding="utf-8")
    import os

    os.utime(image, (mtime, mtime))
    return image


def _cached_images(cache_dir: Path) -> list[str]:
    return sorted(p.stem for p in cache_dir.iterdir() if p.suffix in _SUFFIXES)


def test_eviction_respects_the_byte_budget_and_keeps_the_newest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Eviction removes the least recently used entries until the cache fits its byte budget.

    Entries go whichever encoding they hold, each with its sidecar.
    """
    sidecar = len("{}")
    monkeypatch.setattr(
        images_route, "_cache_budget_bytes", 2 * (1000 + sidecar)
    )
    for i, mtime in enumerate([100.0, 200.0, 300.0, 400.0]):
        _cache_entry(tmp_path, f"entry{i}", 1000, mtime)

    images_route._evict_lru(tmp_path)

    assert _cached_images(tmp_path) == ["entry2", "entry3"]
    assert sorted(p.stem for p in tmp_path.glob("*.json")) == [
        "entry2",
        "entry3",
    ]


def test_a_cache_within_budget_is_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(images_route, "_cache_budget_bytes", 10_000)
    for i in range(3):
        _cache_entry(tmp_path, f"entry{i}", 1000, 100.0 + i)
    images_route._evict_lru(tmp_path)
    assert len(_cached_images(tmp_path)) == 3


def test_the_budget_derives_once_per_process_from_free_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import collections
    import shutil

    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(images_route, "_cache_budget_bytes", None)
    monkeypatch.setattr(
        shutil, "disk_usage", lambda _p: usage(100, 20, 80_000)
    )
    first = images_route._cache_byte_budget(tmp_path)
    assert first == 80_000 // images_route._CACHE_BUDGET_DIVISOR

    def _no_more_reads(_path: Path) -> NoReturn:
        raise AssertionError(
            "the budget must not be re-derived after the first read"
        )

    monkeypatch.setattr(shutil, "disk_usage", _no_more_reads)
    assert images_route._cache_byte_budget(tmp_path) == first


def test_a_file_that_fails_its_header_probe_answers_400_and_a_readable_one_still_serves(
    client: TestClient, tmp_path: Path
) -> None:
    """A file that fails its header probe answers 400, and a readable one still serves.

    A truncated capture answers 400 saying the image could not be opened.
    """
    truncated = tmp_path / "truncated.jpg"
    truncated.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01")

    refused = client.get(
        "/api/images",
        params={"path": str(truncated), "display_pixels": DISPLAY},
    )
    assert refused.status_code == 400, refused.text
    assert "could not open this image" in refused.json()["detail"]

    readable = tmp_path / "readable.jpg"
    Image.fromarray(np.full((16, 20, 3), 90, dtype=np.uint8)).save(readable)
    assert (
        client.get(
            "/api/images",
            params={"path": str(readable), "display_pixels": DISPLAY},
        ).status_code
        == 200
    )


@pytest.mark.parametrize(
    ("dtype", "channels", "photometric", "plain"),
    [
        ("uint16", 3, "rgb", False),
        ("uint8", 4, "rgb", True),
        ("uint8", 4, "minisblack", False),
    ],
    ids=["16-bit RGB", "8-bit RGBA", "8-bit four spectral bands"],
)
def test_the_image_route_and_the_vision_tools_display_a_raster_alike(
    client: TestClient,
    tmp_path: Path,
    dtype: str,
    channels: int,
    photometric: str,
    plain: bool,
) -> None:
    """The route's default render of a whole raster is the vision tools' display of it.

    A 16-bit three-band raster and a four-band 8-bit one that names no alpha band are composited
    by both; a four-band 8-bit one whose fourth band is alpha is shown as its own pixels by both,
    and the bands route describes each as the two display it.
    """
    from tcip_mcp.pipelines.raster_source import SourceHeader
    from tcip_mcp.tools.vision_tools import _display_of

    path = tmp_path / f"{dtype}_{channels}_{photometric}.tif"
    top = np.iinfo(dtype).max
    pixels = (
        np.random.default_rng(5).integers(0, top // 3, size=(24, 40, channels))
        + np.arange(channels) * (top // 3)
    ).astype(dtype)
    tifffile.imwrite(
        str(path),
        pixels,
        photometric=photometric,
        planarconfig="contig",
        extrasamples=["unassalpha"] if photometric == "rgb" and channels == 4 else None,
    )

    served = _served(
        client.get(
            "/api/images",
            params={
                "path": str(path),
                "display_pixels": DISPLAY,
                "region": [0, 0, 40, 24],
            },
        )
    )

    assert np.array_equal(served, _display_of(SourceHeader(path)).pixels)
    assert np.array_equal(served, pixels[:, :, :3]) is plain
    described = client.get("/api/images/bands", params={"path": str(path)}).json()
    assert described["plain"] is plain
