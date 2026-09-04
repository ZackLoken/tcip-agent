"""The one GeoTIFF-writing helper every suite that needs a real (or deliberately incomplete or
rotated) georeferenced raster on disk imports, so test_region_completeness.py,
test_coverage_routes.py, test_orthomosaic_mapping.py, test_orthomosaic_tools.py and
test_segment_attribution.py never drift into slightly different recipes for "the same" fixture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

# UTM zone 15N: the real projected CRS every default below uses. Central meridian is -93
# degrees exactly, an independently hand-verifiable reference point (test_orthomosaic_mapping.py).
UTM_15N_EPSG = 32615
TIEPOINT_NATIVE_X = 500_000.0  # UTM zone 15N's own false easting = the central meridian
TIEPOINT_NATIVE_Y = 4_800_000.0
PIXEL_SCALE = 0.5  # native-CRS units (m) per pixel


def build_geokeys(*, model_type: int = 1, projected_epsg: int | None = UTM_15N_EPSG) -> tuple[int, ...]:
    """A ``GeoKeyDirectoryTag``'s flat uint16 array naming ``model_type`` (1 == Projected) and,
    when given, a projected CRS by EPSG code."""
    entries: list[int] = [1024, 0, 1, model_type]
    if projected_epsg is not None:
        entries += [3072, 0, 1, projected_epsg]
    return (1, 1, 0, len(entries) // 4, *entries)


def write_geotiff(
    path: Path,
    *,
    width: int | None = None,
    height: int | None = None,
    shape: tuple[int, int, int] = (5, 5, 4),
    pixel_scale: tuple[float, float, float] = (PIXEL_SCALE, PIXEL_SCALE, 0.0),
    tiepoint: tuple[float, ...] = (0.0, 0.0, 0.0, TIEPOINT_NATIVE_X, TIEPOINT_NATIVE_Y, 0.0),
    projected_epsg: int | None = UTM_15N_EPSG,
    model_type: int = 1,
    geokeys: tuple[int, ...] | None = None,
    include_transformation_tag: bool = False,
    random: bool = False,
    rowsperstrip: int | None = None,
) -> None:
    """A striped GeoTIFF at ``path`` carrying real georeferencing tags, or a deliberately
    incomplete/rotated set for a refusal test.

    ``shape`` (height, width, channels) is the base frame; ``width``/``height`` override its
    first two dimensions when given (the small-fixture callers that never need a fourth channel).
    ``geokeys`` overrides the whole ``GeoKeyDirectoryTag`` array directly, for a caller building
    one by hand (an out-of-line key, say); ``model_type``/``projected_epsg`` build one through
    :func:`build_geokeys` otherwise. A channel count above 3 gets an unassociated-alpha
    ``extrasamples`` tag per extra channel, since ``photometric="rgb"`` alone leaves them
    undeclared.
    """
    if width is not None or height is not None:
        h = height if height is not None else shape[0]
        w = width if width is not None else shape[1]
        shape = (h, w, shape[2])
    arr = (
        np.random.default_rng(0).integers(0, 255, size=shape, dtype=np.uint8)
        if random else np.zeros(shape, dtype=np.uint8)
    )
    keys = geokeys if geokeys is not None else build_geokeys(
        model_type=model_type, projected_epsg=projected_epsg)
    extratags = [
        (33550, "d", 3, pixel_scale, False),
        (33922, "d", len(tiepoint), tiepoint, False),
        (34735, "H", len(keys), keys, False),
    ]
    if include_transformation_tag:
        # Presence alone is what read_geotransform refuses on, regardless of the values it
        # carries; a flat placeholder is exactly as effective as a computed affine here.
        extratags.append((34264, "d", 16, tuple(1.0 for _ in range(16)), False))
    kwargs: dict = {}
    if shape[-1] > 3:
        kwargs["extrasamples"] = ["unassalpha"] * (shape[-1] - 3)
    if rowsperstrip is not None:
        kwargs["rowsperstrip"] = rowsperstrip
    tifffile.imwrite(str(path), arr, photometric="rgb", extratags=extratags, **kwargs)


def write_canonical_dataset_raster(
    dataset_root: Path, *, date: str = "2024-06-01", stem: str = "mosaic",
    width: int = 64, height: int = 64, crop: str = "chestnut", channels: int = 3,
) -> Path:
    """A georeferenced raster at ``<dataset_root>/images/<date>/<stem>.tif``, its dataset
    registered (``register_dataset``), so its own label document resolves through
    ``annotation_path_for_image`` and its identity is a real, registered one.

    The one canonical-position raster fixture every canopy-segment and whole-raster delivery
    test that needs a registered dataset behind its raster builds through, so none of them
    re-derives the layout by hand.
    """
    from tcip_mcp.tools.project_tools import register_dataset

    raster_path = dataset_root / "images" / date / f"{stem}.tif"
    raster_path.parent.mkdir(parents=True, exist_ok=True)
    write_geotiff(raster_path, width=width, height=height, shape=(height, width, channels))
    result = register_dataset(str(dataset_root), crop=crop, project_root=str(dataset_root))
    assert "error" not in result, result
    return raster_path
