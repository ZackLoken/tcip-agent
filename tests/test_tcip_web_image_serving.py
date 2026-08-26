"""Image serving through one raster read: regions, the display caps, the plain-serve rule, the
stretch bounds a response reports, and the overview build a scaled read of an oversized raster
needs first (routes/images.py).
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient
from PIL import Image

from tcip_web.app import app
from tcip_web.routes import images as images_route


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


@pytest.fixture(autouse=True)
def _clear_stats_cache():
    """The per-raster stats cache is process-global; a fresh test starts from an empty one."""
    images_route._stats_cache.clear()
    yield
    images_route._stats_cache.clear()


def _quadrant_rgb(path: Path, width: int = 400, height: int = 300) -> np.ndarray:
    """A uint8 RGB raster whose four quadrants are flat, distinct colors, so a served region can
    be told from any other region and a transposed one from an upright one."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[: height // 2, : width // 2] = (200, 20, 20)
    arr[: height // 2, width // 2:] = (20, 200, 20)
    arr[height // 2:, : width // 2] = (20, 20, 200)
    arr[height // 2:, width // 2:] = (200, 200, 20)
    tifffile.imwrite(str(path), arr)
    return arr


def _multiband(path: Path, *, channels: int = 4, height: int = 24, width: int = 40,
               dtype="uint16") -> np.ndarray:
    """A small multi-band raster whose left and right halves hold different value ranges, so a
    region rendered against its own bounds looks different from one rendered against the raster's.
    """
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 1000, size=(height, width, channels)).astype(dtype)
    arr[:, width // 2:] = arr[:, width // 2:] + 3000
    tifffile.imwrite(str(path), arr.astype(dtype))
    return arr.astype(dtype)


def _wide_raster(path: Path, *, width: int = 5000, height: int = 64) -> np.ndarray:
    """A raster whose longest edge is past the display edge bound, small enough to build in
    tests: an overview level exists for it."""
    arr = (np.arange(height * width) % 251).astype(np.uint8).reshape(height, width)
    tifffile.imwrite(str(path), arr, rowsperstrip=8)
    return arr


def _wide_multiband(path: Path, *, width: int = 5000, height: int = 64,
                    channels: int = 4) -> np.ndarray:
    """A multi-band raster past the display edge bound, so a pyramid can be built for it, with a
    band count that reaches the per-band stats rather than the plain-RGB early return."""
    rng = np.random.default_rng(11)
    arr = rng.integers(0, 256, size=(height, width, channels)).astype(np.uint8)
    tifffile.imwrite(str(path), arr, rowsperstrip=8)
    return arr


def _served(resp) -> np.ndarray:
    assert resp.status_code == 200, resp.text
    return np.asarray(Image.open(io.BytesIO(resp.content)))


def _reject_non_finite_token(token: str) -> float:
    """``json.loads``'s ``parse_constant`` hook: raises on ``NaN``/``Infinity``/``-Infinity``, the
    three tokens Python's own JSON extension accepts and a browser's ``JSON.parse`` refuses, so a
    header that parses here is one a real browser would parse too."""
    raise ValueError(f"header carries a JSON token no strict parser accepts: {token!r}")


def _stats_source(resp) -> dict:
    """``X-TCIP-Stats-Source`` parsed as the ``StatsSource`` JSON it now carries, under a strict
    parser that rejects what ``JSON.parse`` rejects (see ``_reject_non_finite_token``)."""
    import json

    return json.loads(resp.headers["x-tcip-stats-source"], parse_constant=_reject_non_finite_token)


def _display_bounds(resp) -> list[list[float]]:
    """``X-TCIP-Display-Bounds`` parsed as the JSON list of pairs it now carries, under the same
    strict parser as ``_stats_source``."""
    import json

    return json.loads(
        resp.headers["x-tcip-display-bounds"], parse_constant=_reject_non_finite_token)


# ── Regions ──────────────────────────────────────────────────────────────────────────────


def test_a_region_serves_that_regions_own_pixels(client: TestClient, tmp_path: Path):
    path = tmp_path / "quads.tif"
    _quadrant_rgb(path)
    top_right = _served(client.get("/api/images", params={
        "path": str(path), "x0": 200, "y0": 0, "x1": 400, "y1": 150}))
    bottom_left = _served(client.get("/api/images", params={
        "path": str(path), "x0": 0, "y0": 150, "x1": 200, "y1": 300}))
    assert top_right.shape == (150, 200, 3)
    assert bottom_left.shape == (150, 200, 3)
    assert np.allclose(top_right.mean(axis=(0, 1)), (20, 200, 20), atol=6)
    assert np.allclose(bottom_left.mean(axis=(0, 1)), (20, 20, 200), atol=6)


def test_a_region_outside_the_raster_is_refused(client: TestClient, tmp_path: Path):
    path = tmp_path / "quads.tif"
    _quadrant_rgb(path)
    resp = client.get("/api/images", params={
        "path": str(path), "x0": 0, "y0": 0, "x1": 401, "y1": 300})
    assert resp.status_code == 400
    assert "outside" in resp.json()["detail"]


@pytest.mark.parametrize("corners", [
    {"x0": 10, "y0": 10, "x1": 10, "y1": 20},
    {"x0": 10, "y0": 20, "x1": 20, "y1": 20},
    {"x0": 20, "y0": 0, "x1": 10, "y1": 20},
])
def test_an_empty_region_is_refused_rather_than_raising(client: TestClient, tmp_path: Path,
                                                        corners):
    """An empty or inverted rectangle is a bad request, never a 500 through the raster layer's
    own out-of-bounds error."""
    path = tmp_path / "quads.tif"
    _quadrant_rgb(path)
    resp = client.get("/api/images", params={"path": str(path), **corners})
    assert resp.status_code == 400
    assert "x0 < x1" in resp.json()["detail"]


def test_a_partial_region_is_refused(client: TestClient, tmp_path: Path):
    path = tmp_path / "quads.tif"
    _quadrant_rgb(path)
    resp = client.get("/api/images", params={"path": str(path), "x0": 0, "y0": 0, "x1": 100})
    assert resp.status_code == 400
    assert "all four" in resp.json()["detail"]


# ── Display caps ─────────────────────────────────────────────────────────────────────────


def test_max_width_defaults_to_the_display_edge_bound(client: TestClient, tmp_path: Path):
    """A client that names no width gets the platform's own bound applied at the route, so the
    number lives in one place instead of in every caller."""
    from tcip_mcp.pipelines.display_bounds import DISPLAY_MAX_EDGE

    path = tmp_path / "wide.tif"
    _wide_raster(path)
    served = _served(client.get("/api/images", params={"path": str(path)}))
    assert served.shape[1] == DISPLAY_MAX_EDGE


def test_a_whole_view_over_the_area_cap_scales_to_fit_instead_of_refusing(
    client: TestClient, tmp_path: Path, monkeypatch,
):
    """Whatever an image's shape, asking for the whole of it renders it: the area bound scales the
    result down rather than refusing the request."""
    monkeypatch.setattr(images_route, "DISPLAY_MAX_PIXELS", 20_000)
    path = tmp_path / "frame.jpg"
    Image.new("RGB", (400, 300), (30, 60, 90)).save(path)
    served = _served(client.get("/api/images", params={"path": str(path)}))
    assert served.shape[0] * served.shape[1] <= 20_000
    assert served.shape[1] / served.shape[0] == pytest.approx(400 / 300, abs=0.02)


def test_a_region_over_the_area_cap_is_refused_with_the_cap_named(
    client: TestClient, tmp_path: Path, monkeypatch,
):
    monkeypatch.setattr(images_route, "DISPLAY_MAX_PIXELS", 20_000)
    path = tmp_path / "quads.tif"
    _quadrant_rgb(path)
    resp = client.get("/api/images", params={
        "path": str(path), "x0": 0, "y0": 0, "x1": 400, "y1": 300})
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "20000" in detail and "max_width" in detail


def test_a_region_within_the_area_cap_is_served(client: TestClient, tmp_path: Path, monkeypatch):
    """The refusal above must not close the door on the regions it was written to admit."""
    monkeypatch.setattr(images_route, "DISPLAY_MAX_PIXELS", 20_000)
    path = tmp_path / "quads.tif"
    _quadrant_rgb(path)
    served = _served(client.get("/api/images", params={
        "path": str(path), "x0": 0, "y0": 0, "x1": 100, "y1": 100}))
    assert served.shape == (100, 100, 3)


# ── Cache keys ───────────────────────────────────────────────────────────────────────────


def test_the_etag_varies_with_the_region(client: TestClient, tmp_path: Path):
    path = tmp_path / "quads.tif"
    _quadrant_rgb(path)
    whole = client.get("/api/images", params={"path": str(path)})
    left = client.get("/api/images", params={
        "path": str(path), "x0": 0, "y0": 0, "x1": 200, "y1": 300})
    right = client.get("/api/images", params={
        "path": str(path), "x0": 200, "y0": 0, "x1": 400, "y1": 300})
    tags = {whole.headers["etag"], left.headers["etag"], right.headers["etag"]}
    assert len(tags) == 3


def test_the_etag_changes_when_an_overview_sidecar_appears(client: TestClient, tmp_path: Path):
    """Overview-served pixels are not the pixels a native read resamples to the same size, so a
    build that lands between two requests has to invalidate what the first one cached."""
    from tcip_mcp.pipelines.overviews import build_overviews

    path = tmp_path / "wide.tif"
    _wide_raster(path)
    before = client.get("/api/images", params={"path": str(path)})
    assert before.status_code == 200
    build_overviews(path)
    after = client.get("/api/images", params={"path": str(path)})
    assert after.status_code == 200
    assert after.headers["etag"] != before.headers["etag"]


# ── The plain-serve rule ─────────────────────────────────────────────────────────────────


def test_a_uint8_raster_serves_its_own_pixels_with_no_stretch(client: TestClient, tmp_path: Path):
    """No band selection, no stretch: a flat frame would come back black through a min-max span
    and comes back at its own levels instead."""
    path = tmp_path / "flat.tif"
    tifffile.imwrite(str(path), np.full((32, 40, 3), (100, 120, 140), dtype=np.uint8))
    resp = client.get("/api/images", params={"path": str(path)})
    served = _served(resp)
    assert np.allclose(served.mean(axis=(0, 1)), (100, 120, 140), atol=3)
    assert _stats_source(resp) == {"read": "none", "seed": None, "pixel_fraction": None,
                                   "overview_scale": None}


def test_a_uint16_raster_serves_on_its_dtypes_full_scale(client: TestClient, tmp_path: Path):
    """A plain serve of a non-uint8 raster divides by the dtype's own ceiling, so a half-scale
    frame reads as mid-grey. A PIL decode of a single-band uint16 frame clips every value at 255
    instead, which renders the same raster white."""
    path = tmp_path / "half.tif"
    tifffile.imwrite(str(path), np.full((32, 40), 32768, dtype=np.uint16))
    resp = client.get("/api/images", params={"path": str(path)})
    served = _served(resp)
    assert np.allclose(served.mean(axis=(0, 1)), 127.5, atol=3)
    assert _stats_source(resp) == {"read": "dtype_full_scale", "seed": None,
                                   "pixel_fraction": None, "overview_scale": None}


def test_a_multi_band_uint16_raster_serves_on_that_same_scale(client: TestClient, tmp_path: Path):
    path = tmp_path / "half_rgb.tif"
    tifffile.imwrite(str(path), np.full((32, 40, 3), 32768, dtype=np.uint16))
    served = _served(client.get("/api/images", params={"path": str(path)}))
    assert np.allclose(served.mean(axis=(0, 1)), 127.5, atol=3)


def test_a_float_regions_full_scale_is_the_rasters_own_maximum(client: TestClient, tmp_path: Path):
    """A float raster has no dtype ceiling to divide by, so a region divides by the maximum the
    raster's own sample found, not by the brightest value in the region in hand: a dim corner
    stays dim instead of being lifted to full brightness."""
    path = tmp_path / "float.tif"
    arr = np.zeros((32, 40), dtype=np.float32)
    arr[:, :20] = 100.0
    arr[:, 20:] = 1000.0
    tifffile.imwrite(str(path), arr)
    resp = client.get("/api/images", params={
        "path": str(path), "x0": 0, "y0": 0, "x1": 20, "y1": 32})
    served = _served(resp)
    assert np.allclose(served.mean(axis=(0, 1)), 100.0 / 1000.0 * 255.0, atol=3)
    source = _stats_source(resp)
    assert source["read"] == "window_sample" and source["seed"] == 0
    assert _display_bounds(resp) == [[0.0, 1000.0]]


def test_a_non_positive_float_bands_display_bounds_are_its_own_divisor(
    client: TestClient, tmp_path: Path,
):
    """A float band with no positive data renders black and reports (0.0, divisor), the pair it
    actually stretched between, not its own negative sampled range."""
    path = tmp_path / "negative.tif"
    arr = (-np.abs(np.random.default_rng(0).standard_normal((32, 40))) - 1.0).astype(np.float32)
    tifffile.imwrite(str(path), arr)
    resp = client.get("/api/images", params={"path": str(path)})
    served = _served(resp)
    assert np.allclose(served, 0, atol=2)
    assert _display_bounds(resp) == [[0.0, float(-arr.min())]]


def test_a_single_band_raster_serves_as_replicated_grey(client: TestClient, tmp_path: Path):
    path = tmp_path / "grey.tif"
    tifffile.imwrite(str(path), np.full((32, 40), 90, dtype=np.uint8))
    served = _served(client.get("/api/images", params={"path": str(path)}))
    assert served.shape == (32, 40, 3)
    assert np.allclose(served.mean(axis=(0, 1)), 90, atol=3)


def test_a_four_band_raster_serves_as_plain_rgb_with_the_fourth_band_dropped(
    client: TestClient, tmp_path: Path,
):
    """The alpha band of an RGBA raster is dropped, not composited and not stretched: the file's
    own colors reach the viewer unshifted."""
    path = tmp_path / "rgba.tif"
    arr = np.zeros((32, 40, 4), dtype=np.uint8)
    arr[..., :3] = (100, 120, 140)
    arr[..., 3] = 255
    tifffile.imwrite(str(path), arr)
    resp = client.get("/api/images", params={"path": str(path)})
    served = _served(resp)
    assert served.shape == (32, 40, 3)
    assert np.allclose(served.mean(axis=(0, 1)), (100, 120, 140), atol=3)
    assert _stats_source(resp) == {"read": "none", "seed": None, "pixel_fraction": None,
                                   "overview_scale": None}


def test_a_five_band_raster_composites_its_first_three_bands(client: TestClient, tmp_path: Path):
    """Past the band counts an RGB reading covers, a default request is a composite, which is a
    stretched render and says so."""
    path = tmp_path / "five.tif"
    _multiband(path, channels=5)
    resp = client.get("/api/images", params={"path": str(path)})
    assert _served(resp).shape == (24, 40, 3)
    assert _stats_source(resp) == {"read": "served_array", "seed": None, "pixel_fraction": None,
                                   "overview_scale": None}


def _pinned_stretch_raster(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """A non-square 4-band raster whose bands span different ranges, with the display pixels a
    ``minmax`` stretch owes it worked out by hand.

    Every value sits on a quarter of its own band's span, so the display byte each one is owed is
    an exact number (0, 63, 127, 191, 255 after the truncating uint8 cast) rather than something
    re-derived from the stretch's own arithmetic. Returns the raster and the ``3,0,1`` composite it
    is owed.
    """
    bands = [
        [[0, 100, 200], [300, 400, 400]],           # spans 0..400
        [[1400, 1300, 1200], [1100, 1000, 1400]],   # spans 1000..1400
        [[7, 8, 9], [10, 11, 12]],                  # never selected: band choice has to matter
        [[20, 40, 60], [80, 100, 20]],              # spans 20..100
    ]
    block = np.stack([np.array(b, dtype="uint16") for b in bands], axis=-1)
    arr = np.tile(block, (2, 2, 1))
    owed_block = np.stack([
        np.array([[0, 63, 127], [191, 255, 0]], dtype="uint8"),      # band 3
        np.array([[0, 63, 127], [191, 255, 255]], dtype="uint8"),    # band 0
        np.array([[255, 191, 127], [63, 0, 255]], dtype="uint8"),    # band 1
    ], axis=-1)
    tifffile.imwrite(str(path), arr)
    return arr, np.tile(owed_block, (2, 2, 1))


def test_the_served_composite_is_the_shared_display_primitives_own_pixels(
    client: TestClient, tmp_path: Path,
):
    """What the route encodes is ``composite_display_rgb``'s output byte for byte: the viewer and
    any other consumer of that primitive see one set of pixels, not two matching expressions.

    The pixels that primitive owes a known raster are pinned here independently of how it computes
    them, so the two sides agreeing is evidence about the display pixels and not just about both
    calling one function.
    """
    from tcip_mcp.pipelines.band_stats import composite_display_rgb

    path = tmp_path / "capture.tif"
    arr, owed = _pinned_stretch_raster(path)
    composed = composite_display_rgb(arr, [3, 0, 1], "minmax")
    assert composed.tolist() == owed.tolist()

    resp = client.get("/api/images", params={
        "path": str(path), "bands": "3,0,1", "stretch": "minmax", "quality": 90})
    assert resp.status_code == 200
    buf = io.BytesIO()
    Image.fromarray(composed, mode="RGB").save(buf, "JPEG", quality=90)
    assert resp.content == buf.getvalue()


# ── Stretch bounds and what a response reports about them ────────────────────────────────


def test_two_regions_of_one_raster_stretch_against_the_same_bounds(
    client: TestClient, tmp_path: Path,
):
    """Region renders read their bounds from the raster's own sample, never from the region in
    hand, so a viewer panning across a raster is not looking at a stretch that moves under them.
    """
    path = tmp_path / "capture.tif"
    _multiband(path)
    left = client.get("/api/images", params={
        "path": str(path), "bands": "0,1,2", "x0": 0, "y0": 0, "x1": 20, "y1": 24})
    right = client.get("/api/images", params={
        "path": str(path), "bands": "0,1,2", "x0": 20, "y0": 0, "x1": 40, "y1": 24})
    assert left.status_code == right.status_code == 200
    assert _display_bounds(left) == _display_bounds(right)
    left_source, right_source = _stats_source(left), _stats_source(right)
    assert left_source["read"] == "window_sample" and left_source["seed"] == 0
    assert right_source == left_source


def test_a_regions_bounds_are_the_rasters_sampled_bounds(client: TestClient, tmp_path: Path):
    path = tmp_path / "capture.tif"
    arr = _multiband(path)
    resp = client.get("/api/images", params={
        "path": str(path), "bands": "0,1,2", "x0": 0, "y0": 0, "x1": 20, "y1": 24})
    assert resp.status_code == 200
    reported = [tuple(pair) for pair in _display_bounds(resp)]
    assert reported == [(float(arr[:, :, i].min()), float(arr[:, :, i].max())) for i in range(3)]


def test_a_whole_view_reports_the_bounds_of_the_array_it_served(client: TestClient, tmp_path: Path):
    path = tmp_path / "capture.tif"
    arr = _multiband(path)
    resp = client.get("/api/images", params={"path": str(path), "bands": "0,1,2"})
    assert resp.status_code == 200
    assert _stats_source(resp) == {"read": "served_array", "seed": None, "pixel_fraction": None,
                                   "overview_scale": None}
    reported = [tuple(pair) for pair in _display_bounds(resp)]
    assert reported == [(float(arr[:, :, i].min()), float(arr[:, :, i].max())) for i in range(3)]


def test_a_composited_non_positive_float_bands_display_bounds_are_its_own_divisor(
    client: TestClient, tmp_path: Path,
):
    """The composite route under ``stretch=none`` reports (0.0, divisor) for a selected band with
    no positive data, the same rule the plain serve reports it under, not its sampled (min, max)."""
    path = tmp_path / "capture.tif"
    rng = np.random.default_rng(5)
    arr = rng.uniform(1.0, 100.0, size=(24, 40, 3)).astype(np.float32)
    arr[:, :, 0] = -np.abs(arr[:, :, 0]) - 1.0
    tifffile.imwrite(str(path), arr)
    resp = client.get("/api/images", params={
        "path": str(path), "bands": "0,1,2", "stretch": "none"})
    assert resp.status_code == 200
    reported = [tuple(pair) for pair in _display_bounds(resp)]
    assert reported[0] == (0.0, float(-arr[:, :, 0].min()))
    assert reported[1] == (0.0, float(arr[:, :, 1].max()))
    assert reported[2] == (0.0, float(arr[:, :, 2].max()))


def test_a_percent_clip_region_stretches_between_the_cached_cut_points(
    client: TestClient, tmp_path: Path,
):
    from tcip_mcp.pipelines import raster_source

    path = tmp_path / "capture.tif"
    _multiband(path)
    resp = client.get("/api/images", params={
        "path": str(path), "bands": "0,1,2", "stretch": "percent_clip",
        "x0": 0, "y0": 0, "x1": 20, "y1": 24})
    assert resp.status_code == 200
    stats = images_route._raster_stats(path, 4, raster_source.source_pool_key(path, 4))
    reported = [v for pair in _display_bounds(resp) for v in pair]
    expected = [v for i in range(3) for v in stats.clip_bounds[i]]
    assert reported == pytest.approx(expected, rel=1e-5)


def _five_band_float_with_one_nan(path: Path, *, height: int = 24, width: int = 40) -> np.ndarray:
    """A 5-band float32 raster whose band 0 holds one NaN pixel among otherwise ordinary values:
    ``.min()``/``.max()`` propagate that single NaN across the whole band, so band 0's bounds have
    no finite value to report while bands 1 and 2 (not touched) still do."""
    rng = np.random.default_rng(7)
    arr = rng.uniform(0, 1000, size=(height, width, 5)).astype(np.float32)
    arr[0, 0, 0] = np.nan
    tifffile.imwrite(str(path), arr)
    return arr


def test_a_nan_pixel_reports_a_null_bound_under_a_strict_parser(
    client: TestClient, tmp_path: Path,
):
    """A NaN pixel would otherwise poison ``X-TCIP-Display-Bounds`` with Python's own ``NaN``
    token, a JSON extension a browser's ``JSON.parse`` refuses outright, which is what left the
    canvas blank on a raster the server had in fact rendered. The route still serves (200), the
    poisoned band's bound comes back ``null`` rather than that token, and the header parses under
    the same strict parser ``_display_bounds`` uses (refusing exactly what ``JSON.parse`` refuses).
    """
    path = tmp_path / "nan.tif"
    arr = _five_band_float_with_one_nan(path)
    resp = client.get("/api/images", params={"path": str(path), "bands": "0,1,2"})
    assert resp.status_code == 200, resp.text
    bounds = _display_bounds(resp)
    assert bounds[0] == [None, None]
    assert bounds[1] == [float(arr[:, :, 1].min()), float(arr[:, :, 1].max())]
    assert bounds[2] == [float(arr[:, :, 2].min()), float(arr[:, :, 2].max())]


# ── /api/images/bands ────────────────────────────────────────────────────────────────────


def test_get_bands_reports_exact_bounds_when_the_sample_covered_every_pixel(
    client: TestClient, tmp_path: Path,
):
    """Exactness is a reported fact, not a size branch: a raster the window budget covers whole
    gets its own min/max and says the sample was not partial."""
    path = tmp_path / "capture.tif"
    arr = _multiband(path)
    resp = client.get("/api/images/bands", params={"path": str(path)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["band_count"] == 4
    assert body["sampled"] is False
    assert body["pixel_fraction"] == 1.0
    assert body["seed"] == 0
    assert [b["min"] for b in body["bands"]] == [float(arr[:, :, i].min()) for i in range(4)]
    assert [b["max"] for b in body["bands"]] == [float(arr[:, :, i].max()) for i in range(4)]
    assert {b["dtype"] for b in body["bands"]} == {"uint16"}


def test_get_bands_says_so_when_it_read_only_part_of_the_raster(
    client: TestClient, tmp_path: Path, monkeypatch,
):
    """With a window budget below the raster's own grid, the reported bounds describe a sample and
    the response says which one."""
    monkeypatch.setattr(images_route, "_STATS_WINDOW_SIZE", 4)
    monkeypatch.setattr(images_route, "_STATS_MAX_WINDOWS", 2)
    path = tmp_path / "capture.tif"
    _multiband(path)
    body = client.get("/api/images/bands", params={"path": str(path)}).json()
    assert body["sampled"] is True
    assert 0.0 < body["pixel_fraction"] < 1.0
    assert body["seed"] == 0


def test_get_bands_reads_no_whole_decode(client: TestClient, tmp_path: Path, monkeypatch):
    """The per-band stats come from sampled windows, so describing a raster never costs a decode
    of all of it (which is what makes a raster too large to decode describable at all)."""
    from tcip_mcp.pipelines import image_utils

    calls: list = []
    real = image_utils.load_image

    def counted(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(image_utils, "load_image", counted)
    path = tmp_path / "capture.tif"
    _multiband(path)
    assert client.get("/api/images/bands", params={"path": str(path)}).status_code == 200
    assert calls == []


def test_get_bands_reads_an_oversized_rasters_stats_off_its_overviews(
    client: TestClient, tmp_path: Path, monkeypatch,
):
    """Past the native-sampling budget the stats come from one reduced read of the whole frame,
    and the response says at what scale rather than presenting them as the raster's own bounds."""
    from tcip_mcp.pipelines.overviews import build_overviews

    monkeypatch.setattr(images_route, "_STATS_SAMPLE_BUDGET", 100_000)
    path = tmp_path / "wide_ms.tif"
    _wide_multiband(path)
    build_overviews(path)

    body = client.get("/api/images/bands", params={"path": str(path)}).json()
    assert body["band_count"] == 4
    assert body["sampled"] is False
    assert body["overview_scale"] == pytest.approx(1024 / 5000)
    assert "pixel_fraction" not in body and "seed" not in body
    assert all(0 <= b["min"] <= b["max"] <= 255 for b in body["bands"])


def test_an_oversized_raster_without_overviews_names_the_build_endpoint_for_its_stats(
    client: TestClient, tmp_path: Path, monkeypatch,
):
    """Describing it from native windows would decode most of the file to read a fraction of it,
    so the same refusal a whole view gets applies, in the same words."""
    monkeypatch.setattr(images_route, "_STATS_SAMPLE_BUDGET", 100_000)
    path = tmp_path / "wide_ms.tif"
    _wide_multiband(path)

    resp = client.get("/api/images/bands", params={"path": str(path)})
    assert resp.status_code == 400
    assert resp.headers[images_route.IMAGE_ERROR_HEADER] == images_route.OVERVIEWS_REQUIRED
    assert "POST /api/images/overviews" in resp.json()["detail"]


def test_a_raster_within_the_sampling_budget_keeps_reading_native_pixels(
    client: TestClient, tmp_path: Path, monkeypatch,
):
    """The threshold admits everything under it unchanged: same exact bounds, same reported
    sampling facts, no overview needed."""
    monkeypatch.setattr(images_route, "_STATS_SAMPLE_BUDGET", 100_000)
    path = tmp_path / "capture.tif"
    arr = _multiband(path)
    body = client.get("/api/images/bands", params={"path": str(path)}).json()
    assert body["sampled"] is False
    assert body["pixel_fraction"] == 1.0 and body["seed"] == 0
    assert "overview_scale" not in body
    assert [b["max"] for b in body["bands"]] == [float(arr[:, :, i].max()) for i in range(4)]


def test_a_region_of_an_oversized_raster_reports_the_overview_scale_it_stretched_by(
    client: TestClient, tmp_path: Path, monkeypatch,
):
    from tcip_mcp.pipelines.overviews import build_overviews

    monkeypatch.setattr(images_route, "_STATS_SAMPLE_BUDGET", 100_000)
    path = tmp_path / "wide_ms.tif"
    _wide_multiband(path)
    build_overviews(path)

    resp = client.get("/api/images", params={
        "path": str(path), "bands": "0,1,2", "x0": 0, "y0": 0, "x1": 256, "y1": 64})
    assert resp.status_code == 200
    source = _stats_source(resp)
    assert source["read"] == "overview"
    assert source["overview_scale"] == pytest.approx(1024 / 5000, rel=1e-5)
    assert "x-tcip-display-bounds" in resp.headers


def test_get_bands_carries_the_band_interpretations_a_backend_reads(
    client: TestClient, tmp_path: Path,
):
    """What each band holds is the fact that tells an ordinary colour frame from a four-band
    capture; it is reported where a backend reads it and absent where nothing does."""
    rgba = tmp_path / "rgba.tif"
    tifffile.imwrite(str(rgba), np.zeros((32, 40, 4), dtype=np.uint8), rowsperstrip=8)
    body = client.get("/api/images/bands", params={"path": str(rgba)}).json()
    assert [b["interpretation"] for b in body["bands"]] == ["red", "green", "blue", "alpha"]

    stack = tmp_path / "stack.npy"
    np.save(str(stack), np.zeros((32, 40, 4), dtype=np.uint8))
    images_route._stats_cache.clear()
    plain = client.get("/api/images/bands", params={"path": str(stack)}).json()
    assert plain["band_count"] == 4
    assert all("interpretation" not in b for b in plain["bands"])


def test_get_bands_keeps_the_three_band_early_return(client: TestClient, tmp_path: Path):
    """An ordinary RGB frame has no per-band symbology to show, so it costs no pixel read and the
    response carries no sampling facts to report."""
    path = tmp_path / "plain.jpg"
    Image.new("RGB", (40, 32), (5, 5, 5)).save(path)
    body = client.get("/api/images/bands", params={"path": str(path)}).json()
    assert body == {"band_count": 3, "bands": []}


# ── Overview builds ──────────────────────────────────────────────────────────────────────


def _await_job(client: TestClient, job_id: str, timeout: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get("/api/images/overviews/status", params={"job_id": job_id}).json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"overview job {job_id} did not finish within {timeout}s")


def test_a_scaled_read_of_an_oversized_raster_without_overviews_names_the_build_endpoint(
    client: TestClient, tmp_path: Path, monkeypatch,
):
    """Reading the whole of an oversized raster natively is what the display bound exists to
    prevent, so the request is refused, naming the endpoint that makes it servable."""
    monkeypatch.setattr(images_route, "DISPLAY_MAX_PIXELS", 100_000)
    path = tmp_path / "wide.tif"
    _wide_raster(path)
    resp = client.get("/api/images", params={"path": str(path)})
    assert resp.status_code == 400
    assert resp.headers[images_route.IMAGE_ERROR_HEADER] == images_route.OVERVIEWS_REQUIRED
    assert "POST /api/images/overviews" in resp.json()["detail"]


def test_the_overview_job_makes_that_same_request_servable(
    client: TestClient, tmp_path: Path, monkeypatch,
):
    """The refusal admits the work it asked for: build the pyramid it named and the same view
    serves."""
    monkeypatch.setattr(images_route, "DISPLAY_MAX_PIXELS", 100_000)
    path = tmp_path / "wide.tif"
    _wide_raster(path)
    assert client.get("/api/images", params={"path": str(path)}).status_code == 400

    started = client.post("/api/images/overviews", json={"path": str(path)})
    assert started.status_code == 200
    job = _await_job(client, started.json()["job_id"])
    assert job["status"] == "completed", job
    assert job["progress"] == 1.0

    from tcip_mcp.pipelines.overviews import sidecar_valid

    assert sidecar_valid(path)
    served = _served(client.get("/api/images", params={"path": str(path)}))
    assert served.shape[1] <= 5000


def test_a_build_request_joins_the_one_already_running_for_that_raster(
    client: TestClient, tmp_path: Path,
):
    """One build per raster: two builds over the same sidecar would race each other's writes."""
    path = tmp_path / "wide.tif"
    _wide_raster(path)
    running = images_route.OverviewJob(job_id="ovr-running", path=str(path), status="running")
    images_route._overview_jobs[running.job_id] = running
    try:
        joined = client.post("/api/images/overviews", json={"path": str(path)}).json()
        assert joined["job_id"] == "ovr-running"
        assert joined["status"] == "running"
    finally:
        images_route._overview_jobs.pop(running.job_id, None)


def test_a_build_on_an_unreadable_raster_reaches_a_terminal_failure(
    client: TestClient, tmp_path: Path,
):
    """A build that cannot even open its raster has to end as a recorded failure: a caller polls
    this job until it reaches a terminal status, so a worker that dies mid-flight strands it."""
    path = tmp_path / "broken.tif"
    path.write_bytes(b"this is not a raster")
    started = client.post("/api/images/overviews", json={"path": str(path)})
    assert started.status_code == 200
    job = _await_job(client, started.json()["job_id"], timeout=30.0)
    assert job["status"] == "failed"
    assert job["error"]


def test_an_overview_job_id_that_does_not_exist_is_a_404(client: TestClient):
    assert client.get("/api/images/overviews/status",
                      params={"job_id": "ovr-nope"}).status_code == 404


def test_a_deep_zoom_region_within_the_cap_is_served_without_overviews(
    client: TestClient, tmp_path: Path, monkeypatch,
):
    """A region small enough to read natively needs no pyramid, however large the raster is."""
    monkeypatch.setattr(images_route, "DISPLAY_MAX_PIXELS", 100_000)
    path = tmp_path / "wide.tif"
    _wide_raster(path)
    served = _served(client.get("/api/images", params={
        "path": str(path), "x0": 0, "y0": 0, "x1": 256, "y1": 64}))
    assert served.shape == (64, 256, 3)


# ── Rendered-variant cache: the version key ───────────────────────────────────────────────


def test_a_render_cached_under_an_older_version_key_is_not_reused(
    client: TestClient, tmp_path: Path, monkeypatch,
):
    """The render cache key carries ``RENDER_CACHE_VERSION``, so bumping the constant makes
    every entry cached under the old value unreachable: a bumped version renders and caches
    fresh, under its own key, rather than replaying what an older version wrote."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(images_route, "_render_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(images_route, "RENDER_CACHE_VERSION", 1)

    path = tmp_path / "flat.tif"
    tifffile.imwrite(str(path), np.full((32, 40, 3), (10, 20, 30), dtype=np.uint8))

    first = client.get("/api/images", params={"path": str(path)})
    assert first.status_code == 200
    assert len(list(cache_dir.glob("*.jpg"))) == 1

    monkeypatch.setattr(images_route, "RENDER_CACHE_VERSION", 2)
    second = client.get("/api/images", params={"path": str(path)})
    assert second.status_code == 200
    assert second.headers["etag"] != first.headers["etag"]
    assert len(list(cache_dir.glob("*.jpg"))) == 2


# ── Rendered-variant cache: byte-budget LRU ──────────────────────────────────────────────


def _cache_entry(cache_dir: Path, name: str, size: int, mtime: float) -> Path:
    """One rendered variant on disk: the JPEG plus its header sidecar, backdated."""
    jpg = cache_dir / f"{name}.jpg"
    jpg.write_bytes(b"\xff" * size)
    (cache_dir / f"{name}.json").write_text("{}", encoding="utf-8")
    import os

    os.utime(jpg, (mtime, mtime))
    return jpg


def test_eviction_respects_the_byte_budget_and_keeps_the_newest(tmp_path: Path, monkeypatch):
    """Least recently used entries go first, each with its sidecar, until the cache's
    total bytes fit the budget."""
    sidecar = len("{}")
    monkeypatch.setattr(images_route, "_cache_budget_bytes", 2 * (1000 + sidecar))
    for i, mtime in enumerate([100.0, 200.0, 300.0, 400.0]):
        _cache_entry(tmp_path, f"entry{i}", 1000, mtime)

    images_route._evict_lru(tmp_path)

    kept = sorted(p.name for p in tmp_path.glob("*.jpg"))
    assert kept == ["entry2.jpg", "entry3.jpg"]
    assert sorted(p.stem for p in tmp_path.glob("*.json")) == ["entry2", "entry3"]


def test_a_cache_within_budget_is_left_alone(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(images_route, "_cache_budget_bytes", 10_000)
    for i in range(3):
        _cache_entry(tmp_path, f"entry{i}", 1000, 100.0 + i)
    images_route._evict_lru(tmp_path)
    assert len(list(tmp_path.glob("*.jpg"))) == 3


def test_the_budget_derives_once_per_process_from_free_space(tmp_path: Path, monkeypatch):
    import collections
    import shutil

    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(images_route, "_cache_budget_bytes", None)
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: usage(100, 20, 80_000))
    first = images_route._cache_byte_budget(tmp_path)
    assert first == 80_000 // images_route._CACHE_BUDGET_DIVISOR

    def _no_more_reads(_path):
        raise AssertionError("the budget must not be re-derived after the first read")

    monkeypatch.setattr(shutil, "disk_usage", _no_more_reads)
    assert images_route._cache_byte_budget(tmp_path) == first
