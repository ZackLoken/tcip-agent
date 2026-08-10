"""The region-completeness store's pure pieces: the locators, the shape-guarding normalizer,
and the per-cell content digest that detects a stale attestation (an annotation edited or
deleted inside an already-attested cell)."""

from __future__ import annotations

from pathlib import Path

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox

from tcip_mcp.dataset_layout import (
    region_completeness_digest_path,
    region_completeness_path,
    normalize_region_completeness_store,
)
from tcip_mcp.pipelines.reference_grid import reference_cells
from tcip_mcp.pipelines.region_completeness import cell_annotation_digest, stale_cells


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
