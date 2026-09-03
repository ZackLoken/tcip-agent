"""The region-completeness store's pure pieces: the locators, the shape-guarding normalizer,
and the per-cell content digest that detects a stale attestation (an annotation edited or
deleted inside an already-attested cell)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox, Point, Polygon

from tcip_mcp.dataset_layout import (
    region_completeness_digest_path,
    region_completeness_path,
    normalize_region_completeness_store,
)
from tcip_mcp.pipelines.reference_grid import reference_cells
from tcip_mcp.pipelines.region_completeness import (
    DatasetExtent,
    PixelSize,
    annotation_counts_by_cell,
    annotations_by_cell,
    cell_annotation_digest,
    cell_annotation_digests,
    dataset_extent_source,
    dataset_physical_extent,
    dataset_working_scale_bar,
    default_working_scale_source,
    raster_pixel_size,
    raster_pixel_size_reason,
    saved_extents,
    stale_cells,
    working_scale_bar,
)
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


def test_region_completeness_path_locator(tmp_path):
    assert region_completeness_path(tmp_path) == (
        tmp_path / ".tcip" / "state" / "region_completeness.json")


def test_region_completeness_digest_path_locator(tmp_path):
    assert region_completeness_digest_path(tmp_path) == (
        tmp_path / ".tcip" / "state" / "region_completeness_digest.json")


class TestNormalizeRegionCompletenessStore:
    def test_valid_records_pass_through(self):
        raw = {"catkin/mosaic": {"grid": {"width": 100}, "cells_complete": ["A1", "B2"],
                                 "attested_by": "user:z", "attested_at": "t", "stem": "mosaic",
                                 "date": None, "subject": "catkin"}}
        assert normalize_region_completeness_store(raw) == raw

    def test_non_dict_input_yields_empty(self):
        assert normalize_region_completeness_store(None) == {}
        assert normalize_region_completeness_store([1, 2]) == {}

    def test_entry_missing_grid_is_dropped(self):
        raw = {"catkin/mosaic": {"cells_complete": ["A1"]}}
        assert normalize_region_completeness_store(raw) == {}

    def test_entry_with_non_dict_grid_is_dropped(self):
        raw = {"catkin/mosaic": {"grid": "not-a-dict", "cells_complete": []}}
        assert normalize_region_completeness_store(raw) == {}

    def test_entry_with_non_list_cells_complete_is_dropped(self):
        raw = {"catkin/mosaic": {"grid": {}, "cells_complete": "A1"}}
        assert normalize_region_completeness_store(raw) == {}

    def test_entry_with_non_string_cell_names_is_dropped(self):
        raw = {"catkin/mosaic": {"grid": {}, "cells_complete": ["A1", 2]}}
        assert normalize_region_completeness_store(raw) == {}

    def test_one_bad_entry_does_not_drop_a_good_sibling(self):
        raw = {
            "catkin/mosaic": {"grid": {}, "cells_complete": ["A1"]},
            "bush/other": {"cells_complete": ["A1"]},  # missing grid
        }
        got = normalize_region_completeness_store(raw)
        assert list(got) == ["catkin/mosaic"]


class TestCellAnnotationDigest:
    def _cell(self, name="A1", x0=0, y0=0, x1=64, y1=64):
        cells = reference_cells(128, 128, 64, clamp=True)
        return next(c for c in cells if c.name == name)

    def test_empty_cell_is_deterministic(self):
        cell = self._cell()
        assert cell_annotation_digest([], "catkin", cell) == cell_annotation_digest(
            [], "catkin", cell)

    def test_two_calls_over_identical_content_agree(self):
        cell = self._cell()
        anns = [Annotation(subject="catkin", geometry=BBox(1, 1, 9, 9))]
        assert cell_annotation_digest(anns, "catkin", cell) == cell_annotation_digest(
            anns, "catkin", cell)

    def test_a_moved_annotation_changes_the_digest(self):
        cell = self._cell()
        before = [Annotation(subject="catkin", geometry=BBox(1, 1, 9, 9))]
        after = [Annotation(subject="catkin", geometry=BBox(1, 1, 20, 20))]
        assert cell_annotation_digest(before, "catkin", cell) != cell_annotation_digest(
            after, "catkin", cell)

    def test_a_deleted_annotation_changes_the_digest(self):
        cell = self._cell()
        before = [Annotation(subject="catkin", geometry=BBox(1, 1, 9, 9))]
        assert cell_annotation_digest(before, "catkin", cell) != cell_annotation_digest(
            [], "catkin", cell)

    def test_a_different_subjects_annotation_in_the_same_cell_does_not_count(self):
        cell = self._cell()
        anns = [Annotation(subject="bush", geometry=BBox(1, 1, 9, 9))]
        assert cell_annotation_digest(anns, "catkin", cell) == cell_annotation_digest(
            [], "catkin", cell)

    def test_an_annotation_outside_the_cell_does_not_count(self):
        # A1 is [0,64)x[0,64); this box centers well outside it.
        cell = self._cell()
        far = [Annotation(subject="catkin", geometry=BBox(90, 90, 100, 100))]
        assert cell_annotation_digest(far, "catkin", cell) == cell_annotation_digest(
            [], "catkin", cell)


class TestCellAnnotationDigests:
    """The binned, one-pass-over-annotations sibling of cell_annotation_digest: must agree with
    it exactly, for every cell, in the same call -- the only property that matters here, since a
    binning bug that silently assigns an annotation to the wrong cell is a real staleness-check
    correctness bug, not just a performance regression."""

    def _cells(self, width=128, height=128, tile_size=64, overlap=0.0):
        return reference_cells(width, height, tile_size, overlap, clamp=True)

    def test_agrees_with_the_per_cell_digest_across_a_populated_grid(self):
        cells = self._cells()
        anns = [
            Annotation(subject="catkin", geometry=BBox(1, 1, 9, 9)),      # A1
            Annotation(subject="catkin", geometry=BBox(70, 1, 78, 9)),    # B1
            Annotation(subject="catkin", geometry=BBox(1, 70, 9, 78)),    # A2
            Annotation(subject="catkin", geometry=BBox(70, 70, 78, 78)),  # B2
            Annotation(subject="bush", geometry=BBox(1, 1, 9, 9)),        # A1, wrong subject
        ]
        got = cell_annotation_digests(anns, "catkin", cells, tile_size=64)
        expected = {c.name: cell_annotation_digest(anns, "catkin", c) for c in cells}
        assert got == expected
        assert len(got) == 4  # A1, A2, B1, B2 -- the grid this fixture actually derives

    def test_empty_cell_list_returns_empty(self):
        assert cell_annotation_digests([], "catkin", [], tile_size=64) == {}

    def test_a_cell_with_no_annotations_still_gets_a_real_digest(self):
        cells = self._cells()
        got = cell_annotation_digests([], "catkin", cells, tile_size=64)
        assert set(got) == {c.name for c in cells}
        assert all(got.values())  # every value is a real, non-empty digest string

    def test_nonzero_overlap_falls_back_to_the_per_cell_path_and_still_agrees(self):
        cells = self._cells(overlap=0.2)
        anns = [Annotation(subject="catkin", geometry=BBox(1, 1, 9, 9))]
        got = cell_annotation_digests(anns, "catkin", cells, tile_size=64, overlap=0.2)
        expected = {c.name: cell_annotation_digest(anns, "catkin", c) for c in cells}
        assert got == expected


class TestAnnotationsByCell:
    """The shared binning cell_annotation_digests now calls: subject's annotations grouped by
    cell name, one pass over annotations."""

    def _cells(self, width=128, height=128, tile_size=64, overlap=0.0):
        return reference_cells(width, height, tile_size, overlap, clamp=True)

    def test_bins_by_cell_and_filters_by_subject(self):
        cells = self._cells()
        catkin_a1 = Annotation(subject="catkin", geometry=BBox(1, 1, 9, 9))
        catkin_b1 = Annotation(subject="catkin", geometry=BBox(70, 1, 78, 9))
        bush_a1 = Annotation(subject="bush", geometry=BBox(2, 2, 10, 10))
        got = annotations_by_cell([catkin_a1, catkin_b1, bush_a1], "catkin", cells, tile_size=64)
        assert got["A1"] == [catkin_a1]
        assert got["B1"] == [catkin_b1]
        assert got["A2"] == []
        assert got["B2"] == []

    def test_every_cell_in_the_grid_gets_an_entry_even_when_empty(self):
        cells = self._cells()
        got = annotations_by_cell([], "catkin", cells, tile_size=64)
        assert set(got) == {c.name for c in cells}
        assert all(v == [] for v in got.values())

    def test_feeds_cell_annotation_digests_identically_to_before(self):
        """cell_annotation_digests, refactored to call this, must still agree with the per-cell
        digest exactly -- the property TestCellAnnotationDigests already proves, restated here
        against the shared binning directly."""
        cells = self._cells()
        anns = [
            Annotation(subject="catkin", geometry=BBox(1, 1, 9, 9)),
            Annotation(subject="catkin", geometry=BBox(70, 1, 78, 9)),
        ]
        digests = cell_annotation_digests(anns, "catkin", cells, tile_size=64)
        by_cell = annotations_by_cell(anns, "catkin", cells, tile_size=64)
        for cell in cells:
            expected = cell_annotation_digest(by_cell[cell.name], "catkin", cell)
            assert digests[cell.name] == expected

    def test_nonzero_overlap_falls_back_to_per_cell_containment(self):
        cells = self._cells(overlap=0.2)
        ann = Annotation(subject="catkin", geometry=BBox(1, 1, 9, 9))
        got = annotations_by_cell([ann], "catkin", cells, tile_size=64, overlap=0.2)
        assert got["A1"] == [ann]


class TestAnnotationCountsByCell:
    """Every subject's per-cell annotation count, one pass over annotations regardless of how
    many subjects are present: the completeness route's ``annotation_counts`` field."""

    def _cells(self, width=128, height=128, tile_size=64, overlap=0.0):
        return reference_cells(width, height, tile_size, overlap, clamp=True)

    def test_counts_every_subject_present_in_one_pass(self):
        cells = self._cells()
        anns = [
            Annotation(subject="catkin", geometry=BBox(1, 1, 9, 9)),
            Annotation(subject="catkin", geometry=BBox(2, 2, 10, 10)),
            Annotation(subject="bush", geometry=BBox(70, 1, 78, 9)),
        ]
        got = annotation_counts_by_cell(anns, cells, tile_size=64)
        assert got == {"catkin": {"A1": 2}, "bush": {"B1": 1}}

    def test_no_annotations_yields_no_subjects(self):
        cells = self._cells()
        assert annotation_counts_by_cell([], cells, tile_size=64) == {}


class TestStaleCells:
    def _grid_and_cells(self):
        cells = reference_cells(128, 128, 64, clamp=True)
        grid = {"width": 128, "height": 128, "tile_size": 64, "overlap": 0.0,
                "cols": 2, "rows": 2}
        return grid, cells

    def _write_label(self, ann_dir: Path, stem: str, annotations, w=128, h=128, **kwargs):
        ann_dir.mkdir(parents=True, exist_ok=True)
        json_io.write_annotations(ann_dir / f"{stem}.json", annotations, w, h, **kwargs)

    def _digest_as_stamped(self, ann_dir: Path, stem: str, subject: str, cell) -> str:
        """The digest a real attestation stamps: computed from the annotations *as read back
        from disk* (json_io round-trips coordinates), the same source ``stale_cells`` reads."""
        anns = json_io.read_annotations(str(ann_dir / f"{stem}.json"))
        return cell_annotation_digest(anns, subject, cell)

    def test_no_stamp_at_all_is_reported_stale(self, tmp_path):
        grid, _cells = self._grid_and_cells()
        self._write_label(tmp_path / "annotations", "mosaic",
                          [Annotation(subject="catkin", geometry=BBox(1, 1, 9, 9))])
        record = {"grid": grid, "cells_complete": ["A1"], "stem": "mosaic", "date": None}
        assert stale_cells(tmp_path, record, {}, "catkin") == ["A1"]

    def test_matching_stamp_is_not_stale(self, tmp_path):
        grid, cells = self._grid_and_cells()
        anns = [Annotation(subject="catkin", geometry=BBox(1, 1, 9, 9))]
        self._write_label(tmp_path / "annotations", "mosaic", anns)
        a1 = next(c for c in cells if c.name == "A1")
        digest = self._digest_as_stamped(tmp_path / "annotations", "mosaic", "catkin", a1)
        record = {"grid": grid, "cells_complete": ["A1"], "stem": "mosaic", "date": None}
        assert stale_cells(tmp_path, record, {"A1": digest}, "catkin") == []

    def test_an_edit_inside_an_attested_cell_is_detected(self, tmp_path):
        grid, cells = self._grid_and_cells()
        a1 = next(c for c in cells if c.name == "A1")
        original = [Annotation(subject="catkin", geometry=BBox(1, 1, 9, 9))]
        self._write_label(tmp_path / "annotations", "mosaic", original)
        digest = self._digest_as_stamped(tmp_path / "annotations", "mosaic", "catkin", a1)
        # The label file is edited after attestation: the box moves.
        edited = [Annotation(subject="catkin", geometry=BBox(1, 1, 30, 30))]
        self._write_label(tmp_path / "annotations", "mosaic", edited)
        record = {"grid": grid, "cells_complete": ["A1"], "stem": "mosaic", "date": None}
        assert stale_cells(tmp_path, record, {"A1": digest}, "catkin") == ["A1"]

    def test_a_deletion_inside_an_attested_cell_is_detected(self, tmp_path):
        grid, cells = self._grid_and_cells()
        a1 = next(c for c in cells if c.name == "A1")
        original = [Annotation(subject="catkin", geometry=BBox(1, 1, 9, 9))]
        self._write_label(tmp_path / "annotations", "mosaic", original)
        digest = self._digest_as_stamped(tmp_path / "annotations", "mosaic", "catkin", a1)
        self._write_label(tmp_path / "annotations", "mosaic", [], keep_empty=True)
        record = {"grid": grid, "cells_complete": ["A1"], "stem": "mosaic", "date": None}
        assert stale_cells(tmp_path, record, {"A1": digest}, "catkin") == ["A1"]

    def test_an_edit_outside_the_attested_cell_does_not_flag_it(self, tmp_path):
        grid, cells = self._grid_and_cells()
        a1 = next(c for c in cells if c.name == "A1")
        original = [Annotation(subject="catkin", geometry=BBox(1, 1, 9, 9))]
        self._write_label(tmp_path / "annotations", "mosaic", original)
        digest = self._digest_as_stamped(tmp_path / "annotations", "mosaic", "catkin", a1)
        # A new annotation lands in a different cell (B2, [64,128)x[64,128)); A1's content is
        # untouched, so A1's own attestation must not be disturbed by an edit elsewhere.
        edited = original + [Annotation(subject="catkin", geometry=BBox(70, 70, 80, 80))]
        self._write_label(tmp_path / "annotations", "mosaic", edited)
        record = {"grid": grid, "cells_complete": ["A1"], "stem": "mosaic", "date": None}
        assert stale_cells(tmp_path, record, {"A1": digest}, "catkin") == []


class TestSavedExtents:
    def test_a_box_contributes_its_longer_side(self):
        anns = [Annotation(subject="leaf", geometry=BBox(0, 0, 10, 4))]
        assert saved_extents(anns, "leaf") == [10.0]

    def test_a_polygon_contributes_its_bounding_boxs_longer_side(self):
        anns = [Annotation(subject="leaf", geometry=Polygon([[(0, 0), (6, 0), (3, 20)]]))]
        assert saved_extents(anns, "leaf") == [20.0]

    def test_a_multi_ring_polygon_spans_every_ring(self):
        rings = [[(0, 0), (5, 0), (5, 5), (0, 5)], [(50, 50), (60, 50), (60, 55), (50, 55)]]
        anns = [Annotation(subject="leaf", geometry=Polygon(rings))]
        assert saved_extents(anns, "leaf") == [60.0]

    def test_a_point_contributes_nothing(self):
        anns = [Annotation(subject="leaf", geometry=Point(5, 5))]
        assert saved_extents(anns, "leaf") == []

    def test_a_geometry_less_annotation_contributes_nothing(self):
        anns = [Annotation(subject="leaf", geometry=None)]
        assert saved_extents(anns, "leaf") == []

    def test_filters_by_subject(self):
        anns = [Annotation(subject="fruit", geometry=BBox(0, 0, 10, 10))]
        assert saved_extents(anns, "leaf") == []

    def test_mixed_geometries_yield_one_extent_per_box_or_polygon(self):
        anns = [
            Annotation(subject="leaf", geometry=BBox(0, 0, 10, 4)),
            Annotation(subject="leaf", geometry=Point(1, 1)),
            Annotation(subject="leaf", geometry=Polygon([[(0, 0), (6, 0), (3, 30)]])),
            Annotation(subject="leaf", geometry=None),
        ]
        assert sorted(saved_extents(anns, "leaf")) == [10.0, 30.0]


class TestWorkingScaleBar:
    def test_no_extents_yields_no_bar(self):
        assert working_scale_bar([], judged_span_px=46, source="s", from_this_image=True) is None

    def test_one_extent_is_the_exact_quotient(self):
        bar = working_scale_bar([100.0], judged_span_px=46, source="s", from_this_image=True)
        assert bar == {
            "value": 46 / 100.0, "median_extent_native_px": 100.0, "annotation_count": 1,
            "judged_span_px": 46, "source": "s", "from_this_image": True,
        }

    def test_an_odd_count_takes_the_middle_value(self):
        bar = working_scale_bar(
            [10.0, 100.0, 30.0], judged_span_px=46, source="s", from_this_image=True)
        assert bar is not None
        assert bar["median_extent_native_px"] == 30.0
        assert bar["value"] == 46 / 30.0
        assert bar["annotation_count"] == 3

    def test_an_even_count_averages_the_two_middle_values(self):
        bar = working_scale_bar(
            [10.0, 20.0, 30.0, 40.0], judged_span_px=46, source="s", from_this_image=True)
        assert bar is not None
        assert bar["median_extent_native_px"] == 25.0
        assert bar["value"] == 46 / 25.0

    def test_a_single_whole_frame_annotation_yields_a_bar_every_ordinary_view_meets(self):
        # A 4000px-wide annotation spanning nearly the whole frame yields a tiny scale value,
        # below any real zoom level: the accepted consequence for a large object.
        bar = working_scale_bar([4000.0], judged_span_px=46, source="s", from_this_image=True)
        assert bar is not None
        assert bar["value"] < 0.05

    def test_from_this_image_is_set_false_on_the_dataset_branch(self):
        bar = working_scale_bar([100.0], judged_span_px=46, source="s", from_this_image=False)
        assert bar is not None
        assert bar["from_this_image"] is False


def test_default_working_scale_source_names_the_span_and_disclaims_measurement():
    source = default_working_scale_source(46)
    assert "46" in source
    assert "not a measurement" in source


class TestDatasetExtentSource:
    def _extent(self, **overrides) -> DatasetExtent:
        fields = {
            "median_extent_m": 5.0, "annotation_count": 3, "image_count": 2,
            "metres_per_px_min": 0.5, "metres_per_px_max": 1.0,
            "dates": ("2026-02-11", "2026-03-02"),
        }
        fields.update(overrides)
        return DatasetExtent(**fields)

    def test_names_the_contributing_dates(self):
        extent = self._extent()
        pixel_size = PixelSize(metres_per_px=0.5, source_clause="a projected geotransform "
                                "(EPSG:32615, 0.5 m/px)")
        source = dataset_extent_source("catkin", extent, pixel_size, 46)
        assert "2026-02-11 and 2026-03-02" in source
        assert "46" in source

    def test_uses_the_pixel_sizes_own_source_clause(self):
        extent = self._extent()
        pixel_size = PixelSize(metres_per_px=0.5, source_clause="a projected geotransform "
                                "(EPSG:32615, 0.5 m/px)")
        source = dataset_extent_source("catkin", extent, pixel_size, 46)
        assert pixel_size.source_clause in source

    def test_a_single_contributing_date_carries_no_conjunction(self):
        extent = self._extent(dates=("2026-02-11",))
        pixel_size = PixelSize(metres_per_px=0.5, source_clause="a projected geotransform "
                                "(EPSG:32615, 0.5 m/px)")
        source = dataset_extent_source("catkin", extent, pixel_size, 46)
        assert "2026-02-11" in source
        assert " and 2026-02-11" not in source

    def test_a_dateless_contributor_reads_as_an_undated_capture(self):
        extent = self._extent(dates=(None,))
        pixel_size = PixelSize(metres_per_px=0.5, source_clause="a projected geotransform "
                                "(EPSG:32615, 0.5 m/px)")
        source = dataset_extent_source("catkin", extent, pixel_size, 46)
        assert "an undated capture" in source


class TestDatasetWorkingScaleBar:
    def test_divides_the_median_through_this_images_pixel_size(self):
        extent = DatasetExtent(
            median_extent_m=5.0, annotation_count=2, image_count=2,
            metres_per_px_min=0.5, metres_per_px_max=0.5, dates=("2026-01-01",))
        pixel_size = PixelSize(metres_per_px=0.5, source_clause="a projected geotransform "
                                "(EPSG:32615, 0.5 m/px)")
        bar = dataset_working_scale_bar("catkin", extent, pixel_size, 46)
        assert bar["median_extent_native_px"] == pytest.approx(10.0)  # 5.0m / 0.5 m/px
        assert bar["from_this_image"] is False
        assert bar["annotation_count"] == 2


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


class TestDatasetPhysicalExtent:
    def _dataset(self, tmp_path: Path) -> tuple[Path, Path, Path]:
        root = tmp_path / "ds"
        img_dir = root / "images" / "2026-01-01"
        ann_dir = root / "annotations" / "2026-01-01"
        img_dir.mkdir(parents=True)
        ann_dir.mkdir(parents=True)
        return root, img_dir, ann_dir

    def test_pools_across_rasters_at_different_pixel_sizes_and_differs_from_the_pixel_median(
        self, tmp_path,
    ):
        root, img_dir, ann_dir = self._dataset(tmp_path)
        _write_geotiff(img_dir / "a.tif", pixel_scale=(0.5, 0.5, 0.0))
        json_io.write_annotations(
            ann_dir / "a.json", [Annotation(subject="leaf", geometry=BBox(0, 0, 10, 4))], 5, 5)
        _write_geotiff(img_dir / "b.tif", pixel_scale=(1.0, 1.0, 0.0))
        json_io.write_annotations(
            ann_dir / "b.json", [Annotation(subject="leaf", geometry=BBox(0, 0, 20, 4))], 5, 5)
        # A negative georeferenced raster of a third pixel size: contributes no annotation, and
        # its own pixel size must not skew the median of the two that do contribute.
        _write_geotiff(img_dir / "c.tif", pixel_scale=(2.0, 2.0, 0.0))
        json_io.write_annotations(ann_dir / "c.json", [], 5, 5, keep_empty=True)

        extent = dataset_physical_extent(root, "leaf", pixel_sizes={})
        assert extent is not None
        assert extent.median_extent_m == pytest.approx(12.5)  # median of [5.0, 20.0] metres
        assert extent.annotation_count == 2
        assert extent.image_count == 2
        # The pooled native-pixel median (10, 20 -> 15.0) is a different number: the metres
        # conversion is applied per contributor before pooling, never after.
        assert extent.median_extent_m != 15.0
        assert extent.metres_per_px_min == pytest.approx(0.5)
        assert extent.metres_per_px_max == pytest.approx(1.0)
        assert extent.dates == ("2026-01-01",)

    def test_a_zero_pixel_scale_contributor_is_skipped_and_the_median_is_unaffected(
        self, tmp_path,
    ):
        root, img_dir, ann_dir = self._dataset(tmp_path)
        _write_geotiff(img_dir / "a.tif", pixel_scale=(0.5, 0.5, 0.0))
        json_io.write_annotations(
            ann_dir / "a.json", [Annotation(subject="leaf", geometry=BBox(0, 0, 10, 4))], 5, 5)
        _write_geotiff(img_dir / "zero.tif", pixel_scale=(0.0, 0.0, 0.0))
        json_io.write_annotations(
            ann_dir / "zero.json",
            [Annotation(subject="leaf", geometry=BBox(0, 0, 1000, 4))], 5, 5)

        extent = dataset_physical_extent(root, "leaf", pixel_sizes={})
        assert extent is not None
        assert extent.annotation_count == 1
        assert extent.image_count == 1
        assert extent.median_extent_m == pytest.approx(5.0)  # 10px * 0.5 m/px, the zero one skipped

    def test_a_contributor_with_no_known_pixel_size_does_not_contribute(self, tmp_path):
        root, img_dir, ann_dir = self._dataset(tmp_path)
        _write_geotiff(img_dir / "a.tif", pixel_scale=(0.5, 0.5, 0.0))
        json_io.write_annotations(
            ann_dir / "a.json", [Annotation(subject="leaf", geometry=BBox(0, 0, 10, 4))], 5, 5)
        Image.fromarray(np.zeros((5, 5, 3), dtype=np.uint8)).save(img_dir / "photo.jpg")
        json_io.write_annotations(
            ann_dir / "photo.json",
            [Annotation(subject="leaf", geometry=BBox(0, 0, 40, 4))], 5, 5)

        extent = dataset_physical_extent(root, "leaf", pixel_sizes={})
        assert extent is not None
        assert extent.annotation_count == 1
        assert extent.image_count == 1
        assert extent.median_extent_m == pytest.approx(5.0)

    def test_a_npy_raster_is_skipped_while_the_walk_continues(self, tmp_path):
        """A ``.npy`` container is a raster by ``capture_kind`` but carries no TIFF tags to read a
        pixel size from (:func:`raster_pixel_size` names it "it is not a TIFF"); its own
        annotation must not contribute, and the walk must still reach the sibling that does."""
        root, img_dir, ann_dir = self._dataset(tmp_path)
        _write_geotiff(img_dir / "a.tif", pixel_scale=(0.5, 0.5, 0.0))
        json_io.write_annotations(
            ann_dir / "a.json", [Annotation(subject="leaf", geometry=BBox(0, 0, 10, 4))], 5, 5)
        np.save(img_dir / "b.npy", np.zeros((5, 5, 3), dtype=np.uint8))
        json_io.write_annotations(
            ann_dir / "b.json", [Annotation(subject="leaf", geometry=BBox(0, 0, 40, 4))], 5, 5)

        extent = dataset_physical_extent(root, "leaf", pixel_sizes={})
        assert extent is not None
        assert extent.annotation_count == 1
        assert extent.image_count == 1
        assert extent.median_extent_m == pytest.approx(5.0)

    def test_no_annotation_with_a_known_pixel_size_yields_none(self, tmp_path):
        root, img_dir, ann_dir = self._dataset(tmp_path)
        Image.fromarray(np.zeros((5, 5, 3), dtype=np.uint8)).save(img_dir / "photo.jpg")
        json_io.write_annotations(
            ann_dir / "photo.json",
            [Annotation(subject="leaf", geometry=BBox(0, 0, 10, 4))], 5, 5)
        assert dataset_physical_extent(root, "leaf", pixel_sizes={}) is None

    def test_a_dateless_dataset_walks_the_flat_bucket(self, tmp_path):
        root = tmp_path / "ds"
        img_dir = root / "images"
        ann_dir = root / "annotations"
        img_dir.mkdir(parents=True)
        ann_dir.mkdir(parents=True)
        _write_geotiff(img_dir / "a.tif", pixel_scale=(0.5, 0.5, 0.0))
        json_io.write_annotations(
            ann_dir / "a.json", [Annotation(subject="leaf", geometry=BBox(0, 0, 10, 4))], 5, 5)

        extent = dataset_physical_extent(root, "leaf", pixel_sizes={})
        assert extent is not None
        assert extent.dates == (None,)

    def test_an_ambiguous_image_stem_ends_the_derivation(self, tmp_path):
        from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest
        from tcip_mcp.pipelines.image_utils import AmbiguousImageStem

        root, img_dir, _ann_dir = self._dataset(tmp_path)
        band_a = img_dir / "cap_G.tif"
        band_b = img_dir / "cap_R.tif"
        _write_geotiff(band_a, pixel_scale=(0.5, 0.5, 0.0))
        _write_geotiff(band_b, pixel_scale=(0.5, 0.5, 0.0))
        write_band_group_manifest(img_dir, "cap", {"Green": band_a, "Red": band_b})
        _write_geotiff(img_dir / "cap.tif", pixel_scale=(0.5, 0.5, 0.0))

        with pytest.raises(AmbiguousImageStem):
            dataset_physical_extent(root, "leaf", pixel_sizes={})
