"""The raster-georeferencing-to-metres-per-pixel resolver: what it accepts, what it refuses, and
the short clause it names a refusal by (never a filesystem path)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tcip_mcp.pipelines.pixel_size import raster_pixel_size, raster_pixel_size_reason
from tests._geotiff_fixtures import UTM_15N_EPSG as _UTM_15N_EPSG
from tests._geotiff_fixtures import write_geotiff as _write_shared_geotiff


def _write_geotiff(
    path: Path, *, pixel_scale: tuple = (0.5, 0.5, 0.0), projected_epsg: int | None = _UTM_15N_EPSG,
    model_type: int = 1, include_transformation_tag: bool = False,
) -> None:
    """A striped GeoTIFF written the way tests/test_orthomosaic_mapping.py's own fixtures are,
    through the shared writer every GeoTIFF-fixture-needing suite calls."""
    _write_shared_geotiff(
        path, width=5, height=5, shape=(5, 5, 3), pixel_scale=pixel_scale,
        projected_epsg=projected_epsg, model_type=model_type,
        include_transformation_tag=include_transformation_tag)


class TestRasterPixelSize:
    def test_a_projected_geotiff_resolves_its_pixel_size(self, tmp_path):
        path = tmp_path / "mosaic.tif"
        _write_geotiff(path, pixel_scale=(0.5, 0.5, 0.0))
        size = raster_pixel_size(path)
        assert size is not None
        assert size.metres_per_px == pytest.approx(0.5)
        assert raster_pixel_size_reason(path) is None

    def test_a_foot_unit_raster_converts_through_the_pyproj_factor(self, tmp_path):
        path = tmp_path / "mosaic.tif"
        _write_geotiff(path, pixel_scale=(1.0, 1.0, 0.0), projected_epsg=2264)
        size = raster_pixel_size(path)
        assert size is not None
        assert size.metres_per_px == pytest.approx(0.3048, abs=1e-3)

    def test_no_georeferencing_tags_has_no_pixel_size(self, tmp_path):
        path = tmp_path / "plain.tif"
        Image.fromarray(np.zeros((5, 5, 3), dtype=np.uint8)).save(path)
        assert raster_pixel_size(path) is None
        assert raster_pixel_size_reason(path)

    def test_a_rotated_raster_has_no_pixel_size(self, tmp_path):
        path = tmp_path / "rotated.tif"
        _write_geotiff(path, include_transformation_tag=True)
        assert raster_pixel_size(path) is None
        assert raster_pixel_size_reason(path) == "it is rotated or sheared"

    def test_missing_georeferencing_tags_names_a_short_clause_never_the_server_path(
        self, tmp_path,
    ):
        path = tmp_path / "plain.tif"
        Image.fromarray(np.zeros((5, 5, 3), dtype=np.uint8)).save(path)
        reason = raster_pixel_size_reason(path)
        assert reason == "its georeferencing tags are incomplete"
        assert str(path.parent) not in reason
        assert "\\" not in reason and "/" not in reason

    def test_a_geographic_model_type_is_refused_by_read_geotransforms_own_check(self, tmp_path):
        """Distinct from the projected-model-type-but-geographic-CRS case above: here
        GTModelTypeGeoKey itself names the geographic model type (2), so read_geotransform
        refuses before this module ever reaches pyproj."""
        path = tmp_path / "geographic_model_type.tif"
        _write_geotiff(path, model_type=2)
        assert raster_pixel_size(path) is None
        assert raster_pixel_size_reason(path) == "its georeferencing tags are incomplete"

    def test_a_zero_pixel_scale_has_no_pixel_size(self, tmp_path):
        path = tmp_path / "zero.tif"
        _write_geotiff(path, pixel_scale=(0.0, 0.0, 0.0))
        assert raster_pixel_size(path) is None
        assert raster_pixel_size_reason(path) == "its pixel scale is zero or negative"

    def test_a_negative_pixel_scale_has_no_pixel_size(self, tmp_path):
        path = tmp_path / "negative.tif"
        _write_geotiff(path, pixel_scale=(-0.5, 0.5, 0.0))
        assert raster_pixel_size(path) is None
        assert raster_pixel_size_reason(path) == "its pixel scale is zero or negative"

    def test_a_geographic_crs_under_a_projected_model_type_has_no_pixel_size(self, tmp_path):
        """The raster's own GTModelTypeGeoKey says Projected (read_geotransform admits it), but
        the EPSG it names resolves to a geographic CRS: a disagreement this module's own
        is_projected check catches, distinct from read_geotransform's own model-type refusal."""
        path = tmp_path / "geo.tif"
        _write_geotiff(path, projected_epsg=4326)
        assert raster_pixel_size(path) is None
        assert raster_pixel_size_reason(path) == "its georeferencing is not projected"

    def test_an_unresolvable_epsg_has_no_pixel_size_naming_it_user_defined(self, tmp_path):
        path = tmp_path / "userdef.tif"
        _write_geotiff(path, projected_epsg=32767)
        assert raster_pixel_size(path) is None
        reason = raster_pixel_size_reason(path)
        assert "32767" in reason
        assert "user-defined" in reason

    def test_a_compound_epsg_has_no_pixel_size_naming_it_compound(self, tmp_path):
        path = tmp_path / "compound.tif"
        _write_geotiff(path, projected_epsg=7415)
        assert raster_pixel_size(path) is None
        reason = raster_pixel_size_reason(path)
        assert "7415" in reason
        assert "compound" in reason

    def test_an_anisotropic_raster_has_no_pixel_size(self, tmp_path):
        path = tmp_path / "aniso.tif"
        _write_geotiff(path, pixel_scale=(0.5, 0.6, 0.0))
        assert raster_pixel_size(path) is None
        assert raster_pixel_size_reason(path) == "its pixel scales differ by axis"

    def test_pixel_scales_within_the_isotropy_slack_still_resolve(self, tmp_path):
        path = tmp_path / "slack.tif"
        _write_geotiff(path, pixel_scale=(0.03, 0.030000001, 0.0))
        size = raster_pixel_size(path)
        assert size is not None
        assert size.metres_per_px == pytest.approx(0.03)

    def test_a_npy_raster_is_skipped_as_not_a_tiff(self, tmp_path):
        path = tmp_path / "array.npy"
        np.save(path, np.zeros((5, 5, 3), dtype=np.uint8))
        assert raster_pixel_size(path) is None
        assert raster_pixel_size_reason(path) == "it is not a TIFF"

    def test_a_photographic_capture_has_no_pixel_size(self, tmp_path):
        path = tmp_path / "photo.jpg"
        Image.fromarray(np.zeros((5, 5, 3), dtype=np.uint8)).save(path)
        assert raster_pixel_size(path) is None
        assert raster_pixel_size_reason(path) == "it is not a raster"
