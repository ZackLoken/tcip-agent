"""Reference-grid geometry: partition, overlap, derived tile sizes and naming.

Covers partition and overlap semantics, derived tile sizes, naming, and the serializable
geometry dict.
"""

from __future__ import annotations

import inspect

import pytest

from tcip_mcp.pipelines.display_bounds import VIZ_ARTIFACT_MAX_EDGE
from tcip_mcp.pipelines.reference_grid import (
    derive_pointing_tile_size,
    derive_serving_tile_size,
    grid_geometry,
    reference_cells,
)


class TestReferenceCells:
    @pytest.mark.parametrize(
        "width,height,tile",
        [
            (640, 480, 80),  # exact multiple
            (100, 80, 64),  # truncated edge cells
            (4096, 4096, 4096),  # single cell
            (10, 3, 4),  # sliver edges
        ],
    )
    def test_clamped_zero_overlap_is_an_exact_partition(
        self, width: int, height: int, tile: int
    ) -> None:
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
            for b in rects[i + 1 :]:
                assert (
                    a[2] <= b[0]
                    or b[2] <= a[0]
                    or a[3] <= b[1]
                    or b[3] <= a[1]
                ), f"cells overlap: {a} and {b}"

    def test_unclamped_overlap_steps_origins_and_keeps_full_size_rects(
        self,
    ) -> None:
        """Unclamped overlapping cells step by the overlapped stride and keep the full edge.

        Overlap > 0 under clamp=False steps origins by the overlapped stride while they lie
        in the frame, and every rect keeps the full cell edge, past the extent at the far side.
        """
        width, height, tile, overlap = 500, 300, 224, 0.2
        cells = reference_cells(width, height, tile, overlap)
        assert sorted({(c.x0, c.y0) for c in cells}) == [
            (0, 0),
            (0, 179),
            (179, 0),
            (179, 179),
            (358, 0),
            (358, 179),
        ]
        assert all(c.x1 - c.x0 == tile and c.y1 - c.y0 == tile for c in cells)
        assert any(c.x1 > width or c.y1 > height for c in cells)

    def test_clamped_overlapping_cells_cover_the_whole_frame(self) -> None:
        """Clamped overlapping cells reach the far edge on both axes.

        Origins step while they lie inside the frame, so a stride that does not divide the
        frame still places a last origin short of the far edge: 200px at 100 and 0.2 steps 0, 80,
        160, and the clamped cells reach 200 on both axes.
        """
        cells = reference_cells(200, 200, 100, 0.2, clamp=True)

        assert (
            sorted({c.x0 for c in cells})
            == sorted({c.y0 for c in cells})
            == [0, 80, 160]
        )
        assert max(c.x1 for c in cells) == max(c.y1 for c in cells) == 200

    def test_names_are_column_letter_plus_row(self) -> None:
        from tcip_annotation.grid import column_index, column_label

        cells = reference_cells(2800, 200, 100, 0.0, clamp=True)
        assert cells[0].name == "A1"
        for c in cells:
            assert c.name == f"{column_label(c.col)}{c.row + 1}"
            assert column_index(c.name.rstrip("0123456789")) == c.col
        assert cells[-1].name == "AB2"

    def test_refusals_name_the_bad_input(self) -> None:
        with pytest.raises(ValueError, match="tile_size"):
            reference_cells(100, 100, 0, 0.0)
        with pytest.raises(ValueError, match="overlap"):
            reference_cells(100, 100, 10, overlap=1.0)
        with pytest.raises(ValueError, match="1x1"):
            reference_cells(0, 100, 10, 0.0)


CAP = 2560 * 1440
"""The area cap of a fixture 2560x1440 display at device pixel ratio 1, its own pixel count: its
square edge is 1920 pixels."""


class TestDerivations:
    @pytest.mark.parametrize(
        "width,height",
        [
            (141130, 239921),
            (4000, 3000),
            (5000, 64),
            (640, 480),
            (1, 100_000),
            (100_000, 1),
        ],
    )
    def test_every_serving_cell_fits_the_cap(
        self, width: int, height: int
    ) -> None:
        tile = derive_serving_tile_size(width, height, CAP)
        cells = reference_cells(width, height, tile, 0.0, clamp=True)
        assert all((c.x1 - c.x0) * (c.y1 - c.y0) <= CAP for c in cells)
        assert len(cells) == 1 or width * height > CAP

    def test_image_inside_the_cap_is_one_cell(self) -> None:
        for width, height in [(1920, 1000), (5000, 64)]:
            tile = derive_serving_tile_size(width, height, CAP)
            assert (
                len(reference_cells(width, height, tile, 0.0, clamp=True)) == 1
            )

    def test_mosaic_scale_geometry(self) -> None:
        """An orthomosaic-sized extent derives tile 1920 and a 74 x 125 serving grid.

        The numeric case is 141130 x 239921 under the fixture display's cap.
        """
        tile = derive_serving_tile_size(141130, 239921, CAP)
        assert tile == 1920
        geometry = grid_geometry(141130, 239921, tile, 0.0)
        assert geometry["cols"] == 74
        assert geometry["rows"] == 125

    def test_derivations_are_deterministic(self) -> None:
        assert derive_serving_tile_size(
            7000, 5000, CAP
        ) == derive_serving_tile_size(7000, 5000, CAP)
        assert derive_pointing_tile_size(
            7000, 5000
        ) == derive_pointing_tile_size(7000, 5000)

    def test_pointing_grain_is_fixed_in_render_space(self) -> None:
        """Images at or past the artifact bound share one pointing grain.

        Images of different native sizes at or past the bound split their long edge into the
        same cell count.
        """
        import math

        from tcip_mcp.pipelines.reference_grid import POINTING_LEGIBLE_EDGE

        expected = math.ceil(VIZ_ARTIFACT_MAX_EDGE / POINTING_LEGIBLE_EDGE)
        for dims in [(141130, 239921), (4000, 3000), (3000, 2000)]:
            long_native = max(dims)
            tile = derive_pointing_tile_size(*dims)
            assert math.ceil(long_native / tile) == expected

    def test_pointing_tile_numeric_case_inside_the_artifact_bound(
        self,
    ) -> None:
        """An image inside the artifact bound takes its pointing grain from its own long edge.

        A 640 by 480 image takes a pointing tile of edge 46.
        """
        assert derive_pointing_tile_size(640, 480) == 46

    def test_serving_tile_is_a_fixed_derivation_of_the_extent(self) -> None:
        """Five extents derive their known serving tile edges under the fixture cap."""
        assert derive_serving_tile_size(141130, 239921, CAP) == 1920
        assert derive_serving_tile_size(4000, 3000, CAP) == 1334
        assert derive_serving_tile_size(5000, 64, CAP) == 5000
        assert derive_serving_tile_size(640, 480, CAP) == 640
        assert derive_serving_tile_size(1920, 1000, CAP) == 1920

    def test_an_encodings_edge_limit_bounds_the_tile_edge(self) -> None:
        """An encoding's edge limit splits a one-pixel level that fits the cap into cells.

        A one-pixel level within the cap but longer than the encoding's edge limit splits
        into cells that limit admits; the same level with no limit stays one cell.
        """
        assert derive_serving_tile_size(1, 100_000, 100_000, 65_500) == 50_000
        assert derive_serving_tile_size(100_000, 1, 100_000, 65_500) == 50_000
        assert derive_serving_tile_size(1, 100_000, 100_000) == 100_000


class TestGridGeometry:
    def test_round_trips_through_reference_cells(self) -> None:
        geometry = grid_geometry(100, 80, 64, 0.0)
        cells = reference_cells(
            geometry["width"],
            geometry["height"],
            geometry["tile_size"],
            geometry["overlap"],
            clamp=True,
        )
        assert geometry == {
            "width": 100,
            "height": 80,
            "tile_size": 64,
            "overlap": 0.0,
            "cols": 2,
            "rows": 2,
        }
        assert len(cells) == geometry["cols"] * geometry["rows"]

    def test_tile_size_and_overlap_are_required(self) -> None:
        with pytest.raises(TypeError, match="'tile_size' and 'overlap'"):
            grid_geometry(100, 80)  # type: ignore[call-arg]  # the omission is the subject; the raises pins it to both
        for grid in (grid_geometry, reference_cells):
            assert (
                inspect.signature(grid).parameters["overlap"].default
                is inspect.Parameter.empty
            )
