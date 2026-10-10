"""The raster-georeferencing-to-meters-per-pixel resolver: what it accepts, what it refuses, and
the short clause it names a refusal by (never a filesystem path)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tcip_mcp.pipelines.pixel_size import resolve_pixel_size as _resolve_header
from tcip_mcp.pipelines.raster_source import SourceHeader
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


def resolve_pixel_size(path: Path):
    """The resolver over ``path``'s header."""
    return _resolve_header(SourceHeader(path))


def _refused(path: Path) -> str:
    """The clause ``path`` resolves no pixel size for."""
    size, reason = resolve_pixel_size(path)
    assert size is None
    return reason


class TestRasterPixelSize:
    def test_a_projected_geotiff_resolves_its_pixel_size(self, tmp_path):
        path = tmp_path / "mosaic.tif"
        _write_geotiff(path, pixel_scale=(0.5, 0.5, 0.0))
        size, reason = resolve_pixel_size(path)
        assert size is not None
        assert size.meters_per_px == pytest.approx(0.5)
        assert reason == ""

    def test_a_foot_unit_raster_converts_through_the_pyproj_factor(self, tmp_path):
        path = tmp_path / "mosaic.tif"
        _write_geotiff(path, pixel_scale=(1.0, 1.0, 0.0), projected_epsg=2264)
        size, _reason = resolve_pixel_size(path)
        assert size is not None
        assert size.meters_per_px == pytest.approx(0.3048, abs=1e-3)

    def test_a_rotated_raster_has_no_pixel_size(self, tmp_path):
        path = tmp_path / "rotated.tif"
        _write_geotiff(path, include_transformation_tag=True)
        assert _refused(path) == "it is rotated or sheared"

    def test_missing_georeferencing_tags_names_a_short_clause_never_the_server_path(
        self, tmp_path,
    ):
        path = tmp_path / "plain.tif"
        Image.fromarray(np.zeros((5, 5, 3), dtype=np.uint8)).save(path)
        reason = _refused(path)
        assert reason == "its georeferencing tags are incomplete"
        assert str(path.parent) not in reason
        assert "\\" not in reason and "/" not in reason

    def test_a_geographic_model_type_is_refused_by_the_tag_reads_own_check(self, tmp_path):
        """Distinct from the projected-model-type-but-geographic-CRS case below: here
        GTModelTypeGeoKey itself names the geographic model type (2), so the header's
        georeference read refuses before this module ever reaches pyproj."""
        path = tmp_path / "geographic_model_type.tif"
        _write_geotiff(path, model_type=2)
        assert _refused(path) == "its georeferencing tags are incomplete"

    def test_a_zero_pixel_scale_has_no_pixel_size(self, tmp_path):
        path = tmp_path / "zero.tif"
        _write_geotiff(path, pixel_scale=(0.0, 0.0, 0.0))
        assert _refused(path) == "its pixel scale is zero or negative"

    def test_a_negative_pixel_scale_has_no_pixel_size(self, tmp_path):
        path = tmp_path / "negative.tif"
        _write_geotiff(path, pixel_scale=(-0.5, 0.5, 0.0))
        assert _refused(path) == "its pixel scale is zero or negative"

    def test_a_geographic_crs_under_a_projected_model_type_has_no_pixel_size(self, tmp_path):
        """The raster's own GTModelTypeGeoKey says Projected (the tag read admits it), but the
        EPSG it names resolves to a geographic CRS: a disagreement this module's own is_projected
        check catches, distinct from the tag read's own model-type refusal."""
        path = tmp_path / "geo.tif"
        _write_geotiff(path, projected_epsg=4326)
        assert _refused(path) == "its georeferencing is not projected"

    def test_an_unresolvable_epsg_has_no_pixel_size_naming_it_user_defined(self, tmp_path):
        path = tmp_path / "userdef.tif"
        _write_geotiff(path, projected_epsg=32767)
        reason = _refused(path)
        assert "32767" in reason
        assert "user-defined" in reason

    def test_a_compound_epsg_has_no_pixel_size_naming_it_compound(self, tmp_path):
        path = tmp_path / "compound.tif"
        _write_geotiff(path, projected_epsg=7415)
        reason = _refused(path)
        assert "7415" in reason
        assert "compound" in reason

    def test_an_anisotropic_raster_has_no_pixel_size(self, tmp_path):
        path = tmp_path / "aniso.tif"
        _write_geotiff(path, pixel_scale=(0.5, 0.6, 0.0))
        assert _refused(path) == "its pixel scales differ by axis"

    def test_pixel_scales_within_the_isotropy_slack_still_resolve(self, tmp_path):
        path = tmp_path / "slack.tif"
        _write_geotiff(path, pixel_scale=(0.03, 0.030000001, 0.0))
        size, _reason = resolve_pixel_size(path)
        assert size is not None
        assert size.meters_per_px == pytest.approx(0.03)

    def test_a_npy_raster_is_skipped_as_not_a_tiff(self, tmp_path):
        path = tmp_path / "array.npy"
        np.save(path, np.zeros((5, 5, 3), dtype=np.uint8))
        assert _refused(path) == "it is not a TIFF"

    def test_an_unreadable_tiff_could_not_be_read(self, tmp_path):
        path = tmp_path / "broken.tif"
        path.write_bytes(b"not a real tiff")
        assert _refused(path) == "it could not be read"

    def test_a_photographic_capture_has_no_pixel_size(self, tmp_path):
        path = tmp_path / "photo.jpg"
        Image.fromarray(np.zeros((5, 5, 3), dtype=np.uint8)).save(path)
        assert _refused(path) == "it is not a raster"
