"""Reference-grid geometry: partition and overlap semantics, derived tile sizes, naming,
and the serializable geometry dict every consumer echoes."""

from __future__ import annotations

import pytest

from tcip_mcp.pipelines.display_bounds import DISPLAY_MAX_EDGE, VIZ_ARTIFACT_MAX_EDGE
from tcip_mcp.pipelines.reference_grid import (
    derive_lattice_tile_size,
    derive_pointing_tile_size,
    derive_serving_tile_size,
    grid_geometry,
    reference_cells,
)


class TestReferenceCells:
    @pytest.mark.parametrize("width,height,tile", [
        (640, 480, 80),      # exact multiple
        (100, 80, 64),       # truncated edge cells
        (4096, 4096, 4096),  # single cell
        (10, 3, 4),          # sliver edges
    ])
    def test_clamped_zero_overlap_is_an_exact_partition(self, width, height, tile):
        """Union of the clamped cells is the extent and no two cells share a pixel."""
        cells = reference_cells(width, height, tile, 0.0, clamp=True)
        covered = 0
        for c in cells:
            assert 0 <= c.x0 < c.x1 <= width
            assert 0 <= c.y0 < c.y1 <= height
            covered += (c.x1 - c.x0) * (c.y1 - c.y0)
        assert covered == width * height
        rects = [(c.x0, c.y0, c.x1, c.y1) for c in cells]
        for i, a in enumerate(rects):
            for b in rects[i + 1:]:
                assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1], \
                    f"cells overlap: {a} and {b}"

    def test_unclamped_overlap_keeps_training_semantics(self):
        """overlap > 0 under clamp=False reproduces the training tiler's own origins and
        full-size rects, so a caller composing training-style tiles is admitted as-is."""
        from tcip_mcp.pipelines.data.tiling import compute_stride, tile_positions

        width, height, tile, overlap = 500, 300, 224, 0.2
        stride = compute_stride(tile, overlap)
        cells = reference_cells(width, height, tile, overlap)
        assert [(c.x0, c.y0) for c in cells] == tile_positions(height, width, tile, stride)
        assert all(c.x1 - c.x0 == tile and c.y1 - c.y0 == tile for c in cells)
        assert any(c.x1 > width or c.y1 > height for c in cells)

    def test_names_are_column_letter_plus_row(self):
        from tcip_annotation.sam_wrapper import column_index, column_label

        cells = reference_cells(2800, 200, 100, clamp=True)
        assert cells[0].name == "A1"
        for c in cells:
            assert c.name == f"{column_label(c.col)}{c.row + 1}"
            assert column_index(c.name.rstrip("0123456789")) == c.col
        assert cells[-1].name == "AB2"

    def test_refusals_name_the_bad_input(self):
        with pytest.raises(ValueError, match="tile_size"):
            reference_cells(100, 100, 0)
        with pytest.raises(ValueError, match="overlap"):
            reference_cells(100, 100, 10, overlap=1.0)
        with pytest.raises(ValueError, match="1x1"):
            reference_cells(0, 100, 10)


class TestDerivations:
    def test_serving_tile_fits_one_display_serve(self):
        for dims in [(141130, 239921), (4000, 3000), (5000, 64), (640, 480)]:
            tile = derive_serving_tile_size(*dims)
            assert 1 <= tile <= DISPLAY_MAX_EDGE

    def test_image_inside_display_bound_is_one_cell(self):
        tile = derive_serving_tile_size(4096, 3000)
        assert tile == 4096
        assert len(reference_cells(4096, 3000, tile, clamp=True)) == 1

    def test_mosaic_scale_geometry(self):
        """The real-mosaic numeric case: 141130 x 239921 derives tile 4067, a 35 x 59
        serving grid."""
        tile = derive_serving_tile_size(141130, 239921)
        assert tile == 4067
        geometry = grid_geometry(141130, 239921, tile)
        assert geometry["cols"] == 35
        assert geometry["rows"] == 59

    def test_derivations_are_deterministic(self):
        assert derive_serving_tile_size(7000, 5000) == derive_serving_tile_size(7000, 5000)
        assert derive_pointing_tile_size(7000, 5000) == derive_pointing_tile_size(7000, 5000)

    def test_pointing_grain_is_fixed_in_render_space(self):
        """Images at or past the artifact bound share one pointing grain: the cell count
        along the long edge is a function of the render, not of native size, so the
        mosaic's overlay and an ordinary photo's read at the same label density."""
        import math

        from tcip_mcp.pipelines.reference_grid import POINTING_LEGIBLE_EDGE

        expected = math.ceil(VIZ_ARTIFACT_MAX_EDGE / POINTING_LEGIBLE_EDGE)
        for dims in [(141130, 239921), (4000, 3000), (3000, 2000)]:
            long_native = max(dims)
            tile = derive_pointing_tile_size(*dims)
            assert math.ceil(long_native / tile) == expected

    def test_pointing_tile_numeric_case_inside_the_artifact_bound(self):
        """An image inside the artifact bound renders at native scale, so the grain is
        chosen against its own long edge: 640 splits into 14 cells of edge 46."""
        assert derive_pointing_tile_size(640, 480) == 46

    def test_serving_tile_is_a_fixed_derivation_regardless_of_zoom(self):
        """The serving grid never depends on a set zoom or the coverage lattice: hardcoded
        expected values, not a self-comparison, so a shared-code-path regression would actually
        be caught."""
        assert derive_serving_tile_size(141130, 239921) == 4067
        assert derive_serving_tile_size(4000, 3000) == 4000
        assert derive_serving_tile_size(5000, 64) == 2500
        assert derive_serving_tile_size(640, 480) == 640
        assert derive_serving_tile_size(4096, 3000) == 4096

    def test_lattice_tile_is_one_screenful_at_the_set_zoom(self):
        """One screenful of native pixels at the zoom, off the short viewport dimension: at 1.5x
        zoom on a 1416x903 viewport (the render's own mockup numbers), the short dimension (903)
        derives a 602 px cell edge."""
        import math

        assert derive_lattice_tile_size(1416, 903, 1.5) == math.ceil(903 / 1.5)
        assert derive_lattice_tile_size(1416, 903, 1.5) == 602

    def test_lattice_tile_uses_the_short_viewport_dimension(self):
        assert derive_lattice_tile_size(2000, 500, 1.0) == 500
        assert derive_lattice_tile_size(500, 2000, 1.0) == 500

    def test_lattice_tile_scales_inversely_with_zoom(self):
        assert derive_lattice_tile_size(1000, 800, 2.0) < derive_lattice_tile_size(1000, 800, 1.0)

    def test_lattice_tile_refuses_a_non_positive_zoom(self):
        with pytest.raises(ValueError, match="zoom"):
            derive_lattice_tile_size(1000, 800, 0)
        with pytest.raises(ValueError, match="zoom"):
            derive_lattice_tile_size(1000, 800, -1.5)

    def test_lattice_tile_refuses_a_sub_pixel_viewport(self):
        with pytest.raises(ValueError, match="viewport"):
            derive_lattice_tile_size(0, 800, 1.0)


class TestGridGeometry:
    def test_round_trips_through_reference_cells(self):
        geometry = grid_geometry(100, 80, 64)
        cells = reference_cells(geometry["width"], geometry["height"],
                                geometry["tile_size"], geometry["overlap"], clamp=True)
        assert geometry == {"width": 100, "height": 80, "tile_size": 64, "overlap": 0.0,
                            "cols": 2, "rows": 2}
        assert len(cells) == geometry["cols"] * geometry["rows"]

    def test_tile_size_is_required(self):
        with pytest.raises(TypeError, match="'tile_size'"):
            grid_geometry(100, 80)  # type: ignore[call-arg]  # the omission is the subject; the raises pins it to tile_size
