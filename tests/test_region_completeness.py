"""The region-completeness store's pure pieces: the locators, the shape-guarding normalizer,
and the per-cell content digest that detects a stale attestation (an annotation edited or
deleted inside an already-attested cell)."""

from __future__ import annotations

from pathlib import Path

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox, Point, Polygon

from tcip_mcp.dataset_layout import (
    region_completeness_digest_path,
    region_completeness_path,
    normalize_region_completeness_store,
)
from tcip_mcp.pipelines.reference_grid import reference_cells
from tcip_mcp.pipelines.region_completeness import (
    annotation_counts_by_cell,
    annotations_by_cell,
    cell_annotation_digest,
    cell_annotation_digests,
    default_working_scale_source,
    saved_extents,
    stale_cells,
    working_scale_bar,
)


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
        assert working_scale_bar([], judged_span_px=46, source="s") is None

    def test_one_extent_is_the_exact_quotient(self):
        bar = working_scale_bar([100.0], judged_span_px=46, source="s")
        assert bar == {
            "value": 46 / 100.0, "median_extent_native_px": 100.0, "annotation_count": 1,
            "judged_span_px": 46, "source": "s",
        }

    def test_an_odd_count_takes_the_middle_value(self):
        bar = working_scale_bar([10.0, 100.0, 30.0], judged_span_px=46, source="s")
        assert bar is not None
        assert bar["median_extent_native_px"] == 30.0
        assert bar["value"] == 46 / 30.0
        assert bar["annotation_count"] == 3

    def test_an_even_count_averages_the_two_middle_values(self):
        bar = working_scale_bar([10.0, 20.0, 30.0, 40.0], judged_span_px=46, source="s")
        assert bar is not None
        assert bar["median_extent_native_px"] == 25.0
        assert bar["value"] == 46 / 25.0

    def test_a_single_whole_frame_annotation_yields_a_bar_every_ordinary_view_meets(self):
        # A 4000px-wide annotation spanning nearly the whole frame yields a tiny scale value,
        # below any real zoom level: the accepted consequence for a large object.
        bar = working_scale_bar([4000.0], judged_span_px=46, source="s")
        assert bar is not None
        assert bar["value"] < 0.05


def test_default_working_scale_source_names_the_span_and_disclaims_measurement():
    source = default_working_scale_source(46)
    assert "46" in source
    assert "not a measurement" in source
