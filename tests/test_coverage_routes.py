"""View-coverage routes: grid serving against the shared geometry, the per-image record's
merge semantics, its bucketing and refusals, and the audited write (routes/coverage.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tcip_web.app import app
from tcip_web.routes import coverage as coverage_route


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


@pytest.fixture(autouse=True)
def _fresh_audit_coalescing():
    """The per-process audit-coalescing set must not leak notes across tests."""
    coverage_route._audited_coverage_keys.clear()
    yield
    coverage_route._audited_coverage_keys.clear()


@pytest.fixture
def dated_dataset(tmp_path: Path) -> tuple[Path, str]:
    """A canonical dataset with one 100x80 image under a date bucket."""
    img_dir = tmp_path / "ds" / "images" / "2026-03-01"
    img_dir.mkdir(parents=True)
    path = img_dir / "plot.tif"
    Image.fromarray(np.zeros((80, 100, 3), dtype=np.uint8)).save(path)
    return tmp_path / "ds", str(path)


def _audit_entries(root: Path, tool: str) -> list[dict]:
    """That root's audit entries for one tool, read through the seam rather than off disk."""
    import tcip_store as ts
    from tcip_mcp.audit import audit_log_key

    page = ts.read_log(audit_log_key(root))
    return [record for record in page.records if record["tool"] == tool]


def _grid_only(grid: dict) -> dict:
    """The six ``GridGeometry`` keys, stripped of ``cells`` and ``derivation``: what
    ``useCoverageGrid.ts`` posts (it destructures both off the grid route's response before
    storing the rest, since neither is one of the six fields ``GridGeometry`` forbids extras
    beyond)."""
    return {k: v for k, v in grid.items() if k not in ("cells", "derivation")}


def _post_body(image_path: str, cells: list[str], grid: dict, **overrides) -> dict:
    body = {
        "image_path": image_path,
        "subject": "bush",
        "date": "2026-03-01",
        "grid": _grid_only(grid),
        "cells_served_at_native": cells,
        "viewing": {"bands": None, "stretch": "minmax", "stats_source": {"read": "none"},
                    "base_served_size": "100x80", "display_bounds": None,
                    "working_scale_bar": None},
    }
    body.update(overrides)
    return body


def _grid(client: TestClient, path: str, tile_size: int | None = None) -> dict:
    params = {"path": path}
    if tile_size is not None:
        params["tile_size"] = tile_size
    resp = client.get("/api/coverage/grid", params=params)
    assert resp.status_code == 200, resp.text
    return resp.json()


class TestGridRoute:
    def test_grid_equals_reference_cells(self, client, dated_dataset):
        from tcip_mcp.pipelines.reference_grid import grid_geometry, reference_cells

        _root, path = dated_dataset
        got = _grid(client, path, tile_size=64)
        assert {k: got[k] for k in ("width", "height", "tile_size", "overlap", "cols", "rows")} \
            == grid_geometry(100, 80, 64)
        assert got["cells"] == [
            {"name": c.name, "x0": c.x0, "y0": c.y0, "x1": c.x1, "y1": c.y1}
            for c in reference_cells(100, 80, 64, clamp=True)
        ]

    def test_tile_size_omitted_derives_the_coverage_lattice(self, client, tmp_path):
        """An ordinary (non-large-raster) source still resolves ``derive_coverage_tile_size``
        unchanged. A small single-page TIFF also opens windowed via GDAL
        (``raster_source.opens_windowed`` is a backend-capability predicate, not a size one), so
        the shared TIFF-based ``dated_dataset`` fixture is not the right fixture to prove this
        contract; a genuinely non-windowed photographic source (PNG) is used instead."""
        from tcip_mcp.pipelines.reference_grid import derive_coverage_tile_size

        img_dir = tmp_path / "ds" / "images" / "2026-03-01"
        img_dir.mkdir(parents=True)
        path = img_dir / "plot.png"
        Image.fromarray(np.zeros((80, 100, 3), dtype=np.uint8)).save(path)

        got = _grid(client, str(path))
        assert got["tile_size"] == derive_coverage_tile_size(100, 80)
        assert got["cols"] == 1 and got["rows"] == 1

    def test_tile_size_omitted_for_a_small_windowed_source_still_derives_the_coverage_lattice(
        self, client, dated_dataset,
    ):
        """A windowed source (``opens_windowed`` is a decode-cost predicate, true for any
        GDAL-servable TIFF regardless of size) with no georeferencing must still resolve the
        ordinary lattice, not the large-raster one -- the exact regression an independent review
        caught: an ordinary drone/ground TIFF capture is windowed too, and must not be misrouted
        just because it's a TIFF."""
        from tcip_mcp.pipelines.raster_source import is_georeferenced, opens_windowed
        from tcip_mcp.pipelines.reference_grid import derive_coverage_tile_size

        _root, path = dated_dataset
        assert opens_windowed(path, 3) is True, \
            "fixture must genuinely be windowed to prove the georeferencing gate, not the backend gate"
        assert is_georeferenced(path) is False

        got = _grid(client, path)
        assert got["tile_size"] == derive_coverage_tile_size(100, 80)

    def test_tile_size_omitted_for_a_large_ungeoreferenced_source_still_derives_the_coverage_lattice(
        self, client, tmp_path,
    ):
        """A large, windowed, but ungeoreferenced TIFF (an ordinary high-resolution drone/ground
        photo saved as TIFF, no stitching) must resolve the ordinary lattice too: the decider is
        real per-pixel georeferencing, never image size, per Zack's explicit direction."""
        import tifffile

        from tcip_mcp.pipelines.raster_source import is_georeferenced, opens_windowed
        from tcip_mcp.pipelines.reference_grid import derive_coverage_tile_size

        width, height = 8000, 4500
        img_dir = tmp_path / "ds" / "images" / "2026-03-01"
        img_dir.mkdir(parents=True)
        path = img_dir / "big_photo.tif"
        arr = np.random.default_rng(0).integers(
            0, 255, size=(height, width, 3), dtype=np.uint8)
        tifffile.imwrite(str(path), arr, photometric="rgb", rowsperstrip=8)

        assert opens_windowed(path, 3) is True
        assert is_georeferenced(path) is False, \
            "fixture must genuinely lack georeferencing to prove size alone doesn't route here"

        got = _grid(client, str(path))
        assert got["tile_size"] == derive_coverage_tile_size(width, height)

    def test_tile_size_omitted_for_a_large_raster_source_derives_the_large_raster_lattice(
        self, client, tmp_path,
    ):
        """A large, windowed, genuinely georeferenced source (a stitched, georectified
        orthomosaic) resolves ``derive_large_raster_grid_tile_size`` instead, the fixed-
        subdivision lattice, not the display-derived one."""
        width, height = 4200, 2100
        path = self._write_georeferenced_tiff(tmp_path, width, height)

        from tcip_mcp.pipelines.raster_source import is_georeferenced, opens_windowed
        from tcip_mcp.pipelines.reference_grid import (
            derive_coverage_tile_size,
            derive_large_raster_grid_tile_size,
        )

        assert opens_windowed(path, 3) is True, \
            "fixture must genuinely trigger the windowed branch, not merely assert an untested case"
        assert is_georeferenced(path) is True, \
            "fixture must genuinely carry a real geotransform, not merely assert an untested case"

        got = _grid(client, str(path))
        expected_tile = derive_large_raster_grid_tile_size(width, height)
        assert got["tile_size"] == expected_tile
        assert got["tile_size"] != derive_coverage_tile_size(width, height)

    def test_tile_size_omitted_for_a_small_georeferenced_source_derives_the_large_raster_lattice(
        self, client, tmp_path,
    ):
        """A small georeferenced source still resolves the large-raster lattice: this platform's
        own decision is that image size never drives this choice, only whether the raster is a
        real georectified mosaic."""
        width, height = 400, 300
        path = self._write_georeferenced_tiff(tmp_path, width, height)

        from tcip_mcp.pipelines.reference_grid import (
            derive_coverage_tile_size,
            derive_large_raster_grid_tile_size,
        )

        got = _grid(client, str(path))
        assert got["tile_size"] == derive_large_raster_grid_tile_size(width, height)
        assert got["tile_size"] != derive_coverage_tile_size(width, height)

    @staticmethod
    def _write_georeferenced_tiff(tmp_path: Path, width: int, height: int) -> str:
        """A striped GeoTIFF GDAL serves windowed, carrying a real UTM affine geotransform."""
        import tifffile

        img_dir = tmp_path / "ds" / "images" / "2026-03-01"
        img_dir.mkdir(parents=True, exist_ok=True)
        path = img_dir / "mosaic.tif"
        arr = np.random.default_rng(0).integers(0, 255, size=(height, width, 3), dtype=np.uint8)
        geokeys = (1, 1, 0, 2, 1024, 0, 1, 1, 3072, 0, 1, 32615)  # UTM zone 15N
        tifffile.imwrite(
            str(path), arr, photometric="rgb", rowsperstrip=8,
            extratags=[
                (33550, "d", 3, (1.0, 1.0, 0.0), False),
                (33922, "d", 6, (0.0, 0.0, 0.0, 500_000.0, 4_800_000.0, 0.0), False),
                (34735, "H", len(geokeys), geokeys, False),
            ],
        )
        return str(path)

    def test_overlap_is_refused_naming_the_partition_contract(self, client, dated_dataset):
        _root, path = dated_dataset
        resp = client.get("/api/coverage/grid",
                          params={"path": path, "tile_size": 64, "overlap": 0.2})
        assert resp.status_code == 400
        assert "exact-partition" in resp.json()["detail"]

    def test_served_cells_are_accepted_verbatim_across_packages(self, client, dated_dataset,
                                                                tmp_path):
        """The cell dicts the route serves feed render_grid_overlay and grid_to_pixel
        unchanged: the one shape both packages document."""
        from tcip_annotation.sam_wrapper import grid_to_pixel
        from tcip_annotation.viz import render_grid_overlay

        _root, path = dated_dataset
        cells = _grid(client, path, tile_size=64)["cells"]
        assert grid_to_pixel("B2", cells) == ((64 + 100) / 2, (64 + 80) / 2)
        out = render_grid_overlay(np.zeros((80, 100, 3), dtype=np.uint8), cells,
                                  native_size=(100, 80),
                                  output_path=str(tmp_path / "overlay.png"))
        assert Path(out).is_file()


class TestGridDerivation:
    """The /grid route's ``derivation`` line: one of the three values naming how the tile size
    was chosen, so the panel can state it rather than leave a cell's origin unexplained."""

    def test_display_bounded_derivation_for_an_ordinary_source(self, client, dated_dataset):
        _root, path = dated_dataset
        got = _grid(client, path)
        assert got["derivation"] == "cells sized to one full-resolution screenful"

    def test_large_raster_derivation_for_a_georeferenced_source(self, client, tmp_path):
        width, height = 4200, 2100
        path = TestGridRoute._write_georeferenced_tiff(tmp_path, width, height)
        got = _grid(client, path)
        assert got["derivation"] == "the long edge in 16 equal divisions"

    def test_chosen_derivation_for_an_explicit_tile_size(self, client, dated_dataset):
        _root, path = dated_dataset
        got = _grid(client, path, tile_size=40)
        assert got["derivation"] == "a chosen cell edge of 40 px"


class TestAnnotationCounts:
    """The completeness read's ``annotation_counts`` field: one implementation
    (``annotation_counts_by_cell``/``_grid_for_raster``) shared with the digest and the grid
    route, so a saved-annotation count is always binned against the grid the breeder sees."""

    @staticmethod
    def _cell_center(cell: dict) -> tuple[float, float]:
        return (cell["x0"] + cell["x1"]) / 2, (cell["y0"] + cell["y1"]) / 2

    def test_counts_are_served_for_an_unattested_raster(self, client, tmp_path):
        """A raster with no attestation at all still reports per-cell saved-annotation counts:
        the field is read off the grid and the label file, never folded into an attestation
        record. A multi-cell large-raster fixture is used since the small ``dated_dataset``
        fixture derives a single whole-image cell, leaving no second cell to prove binning
        against."""
        from tcip_annotation import json_io
        from tcip_annotation.state import Annotation, BBox

        width, height = 4200, 2100
        path = TestGridRoute._write_georeferenced_tiff(tmp_path, width, height)
        root = tmp_path / "ds"
        served = _grid(client, path)
        cell_a, cell_b = served["cells"][0], served["cells"][1]
        ax, ay = self._cell_center(cell_a)
        bx, by = self._cell_center(cell_b)
        json_io.write_annotations(
            root / "annotations" / "2026-03-01" / "mosaic.json",
            [Annotation(subject="catkin", geometry=BBox(ax - 1, ay - 1, ax + 1, ay + 1)),
             Annotation(subject="catkin", geometry=BBox(bx - 1, by - 1, bx + 1, by + 1))],
            width, height)

        got = client.get("/api/coverage/completeness", params={
            "path": path, "dataset_root": str(root)})
        assert got.status_code == 200, got.text
        body = got.json()
        assert body["by_subject"] == {}
        assert body["annotation_counts"] == {"catkin": {cell_a["name"]: 1, cell_b["name"]: 1}}
        assert body["counts_error"] is None
        assert {k: body["counts_grid"][k] for k in
                ("width", "height", "tile_size", "overlap", "cols", "rows")} == \
            {k: served[k] for k in ("width", "height", "tile_size", "overlap", "cols", "rows")}

    def test_counts_use_the_same_grid_the_grid_route_serves(self, client, tmp_path):
        """``_grid_for_raster`` is the one implementation both routes call: the completeness
        read's grid-derived cell names must match the /grid route's own, not a second
        derivation that could drift."""
        from tcip_annotation import json_io
        from tcip_annotation.state import Annotation, BBox

        width, height = 4200, 2100
        path = TestGridRoute._write_georeferenced_tiff(tmp_path, width, height)
        root = tmp_path / "ds"
        served = _grid(client, path)
        one_cell = served["cells"][0]
        cx, cy = self._cell_center(one_cell)
        json_io.write_annotations(
            root / "annotations" / "2026-03-01" / "mosaic.json",
            [Annotation(subject="catkin", geometry=BBox(cx - 1, cy - 1, cx + 1, cy + 1))],
            width, height)

        counts = client.get("/api/coverage/completeness", params={
            "path": path, "dataset_root": str(root)}).json()["annotation_counts"]
        assert counts == {"catkin": {one_cell["name"]: 1}}

    def test_incomplete_band_group_reports_counts_error_and_serves_by_subject_with_its_record(
        self, client, tmp_path,
    ):
        """A band group missing a member cannot derive a grid at all; the read still serves
        ``by_subject`` with the record attested while the group was complete, the refusal named
        in ``counts_error``, never blanked."""
        import numpy as np
        import tifffile

        from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest

        img_dir = tmp_path / "ds" / "images" / "2026-03-01"
        img_dir.mkdir(parents=True)
        band_a = img_dir / "cap_G.tif"
        band_b = img_dir / "cap_R.tif"
        tifffile.imwrite(str(band_a), np.full((16, 16), 1, dtype=np.uint16))
        tifffile.imwrite(str(band_b), np.full((16, 16), 2, dtype=np.uint16))
        write_band_group_manifest(img_dir, "cap", {"Green": band_a, "Red": band_b})
        manifest_path = str(img_dir / "cap.bandgroup")

        grid = _grid(client, manifest_path, tile_size=8)
        resp = client.post("/api/coverage/completeness", json={
            "image_path": manifest_path, "subject": "bush", "grid": _grid_only(grid),
            "cell": "A1", "complete": True, "user": "breeder"})
        assert resp.status_code == 200, resp.text

        band_b.unlink()

        resp = client.get("/api/coverage/completeness", params={"path": manifest_path})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["by_subject"]["bush"]["cells_complete"] == ["A1"]
        assert body["annotation_counts"] == {}
        assert body["counts_grid"] is None
        assert body["counts_error"]
        assert "cap" in body["counts_error"]


class TestCompletenessCountsAreBestEffort:
    """The counts computation (grid derivation, label read, binning) is best-effort beside
    ``by_subject``, which needs no raster at all: a raster-side failure costs ``counts_error``,
    never the records."""

    def test_a_missing_raster_serves_the_records_with_a_counts_error(self, client, dated_dataset):
        root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        resp = client.post("/api/coverage/completeness", json={
            "image_path": path, "subject": "catkin", "grid": _grid_only(grid), "cell": "A1",
            "complete": True, "user": "breeder"})
        assert resp.status_code == 200, resp.text

        Path(path).unlink()

        got = client.get("/api/coverage/completeness", params={"path": path})
        assert got.status_code == 200, got.text
        body = got.json()
        assert body["by_subject"]["catkin"]["cells_complete"] == ["A1"]
        assert body["annotation_counts"] == {}
        assert body["counts_grid"] is None
        assert body["counts_error"]

    def test_an_unreadable_label_costs_only_the_counts_when_no_record_needs_it(
        self, client, dated_dataset,
    ):
        root, path = dated_dataset
        label_path = root / "annotations" / "2026-03-01" / "plot.json"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("not json {][", encoding="utf-8")

        resp = client.get("/api/coverage/completeness", params={"path": path})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["by_subject"] == {}
        assert body["annotation_counts"] == {}
        assert body["counts_grid"] is None
        assert body["counts_error"]
        assert str(label_path) in body["counts_error"]

    def test_an_unreadable_label_still_refuses_when_a_record_depends_on_it(
        self, client, dated_dataset,
    ):
        """The pre-existing rule stays: staleness cannot be computed for an existing record
        without the label file, so that path still refuses rather than reporting a counts
        error and serving a record whose staleness is unknown."""
        root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        client.post("/api/coverage/completeness", json={
            "image_path": path, "subject": "catkin", "grid": _grid_only(grid), "cell": "A1",
            "complete": True, "user": "breeder"})

        label_path = root / "annotations" / "2026-03-01" / "plot.json"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("not json {][", encoding="utf-8")

        resp = client.get("/api/coverage/completeness", params={"path": path})
        assert resp.status_code == 400
        assert str(label_path) in resp.json()["detail"]


class TestCompletenessPathConfinement:
    """``get_completeness`` opens the raster (for its counts) through the same allow-list
    ``get_grid`` uses, on every branch."""

    def test_a_path_outside_the_allowed_roots_is_refused_even_with_an_allowed_dataset_root(
        self, client, dated_dataset, tmp_path_factory,
    ):
        root, _path = dated_dataset
        outside_dir = tmp_path_factory.mktemp("outside") / "images" / "2026-03-01"
        outside_dir.mkdir(parents=True)
        outside_path = outside_dir / "secret.tif"
        Image.fromarray(np.zeros((80, 100, 3), dtype=np.uint8)).save(outside_path)

        resp = client.get("/api/coverage/completeness", params={
            "path": str(outside_path), "dataset_root": str(root)})
        assert resp.status_code == 403

    def test_an_in_root_read_still_answers(self, client, dated_dataset):
        root, path = dated_dataset
        resp = client.get("/api/coverage/completeness", params={
            "path": path, "dataset_root": str(root)})
        assert resp.status_code == 200, resp.text


class TestCoverageRecord:
    def test_post_get_roundtrip(self, client, dated_dataset):
        _root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        resp = client.post("/api/coverage", json=_post_body(path, ["A1", "B2"], grid))
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["replaced"] is False
        assert body["cells_served_at_native"] == 2
        assert body["total_cells"] == 4

        got = client.get("/api/coverage", params={
            "path": path, "subject": "bush", "date": "2026-03-01"})
        assert got.status_code == 200
        record = got.json()["coverage"]
        assert record["cells_served_at_native"] == ["A1", "B2"]
        assert record["grid"]["tile_size"] == 64
        assert record["viewing"]["base_served_size"] == "100x80"
        assert record["updated_at"]

    def test_matching_grid_union_merges(self, client, dated_dataset):
        _root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        client.post("/api/coverage", json=_post_body(path, ["A1"], grid))
        resp = client.post("/api/coverage", json=_post_body(path, ["B1", "A1"], grid))
        assert resp.json()["replaced"] is False
        record = client.get("/api/coverage", params={
            "path": path, "subject": "bush", "date": "2026-03-01"}).json()["coverage"]
        assert record["cells_served_at_native"] == ["A1", "B1"]

    def test_mismatched_grid_replaces_wholesale_and_flags(self, client, dated_dataset):
        _root, path = dated_dataset
        client.post("/api/coverage",
                    json=_post_body(path, ["A1"], _grid(client, path, tile_size=64)))
        resp = client.post("/api/coverage",
                           json=_post_body(path, ["A1"], _grid(client, path, tile_size=100)))
        assert resp.json()["replaced"] is True
        record = client.get("/api/coverage", params={
            "path": path, "subject": "bush", "date": "2026-03-01"}).json()["coverage"]
        assert record["grid"]["tile_size"] == 100
        assert record["cells_served_at_native"] == ["A1"]

    def test_subject_is_required(self, client, dated_dataset):
        _root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        resp = client.post("/api/coverage", json=_post_body(path, ["A1"], grid, subject=None))
        assert resp.status_code == 400
        assert "subject" in resp.json()["detail"]
        assert client.get("/api/coverage", params={"path": path}).status_code == 400

    def test_date_must_be_explicit(self, client, dated_dataset):
        """``date`` carries no default on ``CoveragePayload``, so an omitted key is the model's
        own required-field refusal rather than a route-level check."""
        _root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        body = _post_body(path, ["A1"], grid)
        del body["date"]
        resp = client.post("/api/coverage", json=body)
        assert resp.status_code == 422
        assert any("date" in str(err.get("loc")) for err in resp.json()["detail"])

    def test_explicit_null_date_is_the_dateless_bucket(self, client, tmp_path):
        """A non-dated dataset passes date null and lands under the subject-only bucket."""
        import tcip_store as ts
        from tcip_mcp.dataset_layout import view_coverage_key

        img_dir = tmp_path / "flat" / "images"
        img_dir.mkdir(parents=True)
        path = img_dir / "frame.tif"
        Image.fromarray(np.zeros((80, 100, 3), dtype=np.uint8)).save(path)
        grid = _grid(TestClient(app, base_url="http://127.0.0.1"), str(path), tile_size=64)
        resp = TestClient(app, base_url="http://127.0.0.1").post("/api/coverage",
                                    json=_post_body(str(path), ["A1"], grid, date=None))
        assert resp.status_code == 200, resp.text
        store = ts.read(view_coverage_key(tmp_path / "flat"))
        assert list(store) == ["bush"]

    def test_cells_outside_the_grid_are_refused(self, client, dated_dataset):
        _root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        resp = client.post("/api/coverage", json=_post_body(path, ["Z9"], grid))
        assert resp.status_code == 400
        assert "Z9" in resp.json()["detail"]

    def test_swept_cells_merge_independently_of_served(self, client, dated_dataset):
        _root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        client.post("/api/coverage",
                    json=_post_body(path, ["A1"], grid, cells_swept=["A1", "B1"]))
        resp = client.post("/api/coverage",
                           json=_post_body(path, [], grid, cells_swept=["A2"]))
        assert resp.status_code == 200, resp.text
        assert resp.json()["cells_swept"] == 3
        record = client.get("/api/coverage", params={
            "path": path, "subject": "bush", "date": "2026-03-01"}).json()["coverage"]
        assert record["cells_swept"] == ["A1", "A2", "B1"]
        assert record["cells_served_at_native"] == ["A1"]

    def test_a_post_may_carry_either_fact_alone(self, client, dated_dataset):
        _root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        resp = client.post("/api/coverage", json=_post_body(path, [], grid,
                                                            cells_swept=["B2"]))
        assert resp.status_code == 200, resp.text
        assert resp.json()["cells_served_at_native"] == 0
        assert resp.json()["cells_swept"] == 1

    def test_working_scale_bar_round_trips_in_viewing(self, client, dated_dataset):
        _root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        viewing = {"bands": None, "stretch": "minmax", "stats_source": {"read": "none"},
                   "base_served_size": "100x80", "display_bounds": None,
                   "working_scale_bar": {
                       "value": 0.125,
                       "source": "minimum view scale at annotation-authoring events"}}
        client.post("/api/coverage",
                    json=_post_body(path, [], grid, cells_swept=["A1"], viewing=viewing))
        record = client.get("/api/coverage", params={
            "path": path, "subject": "bush", "date": "2026-03-01"}).json()["coverage"]
        assert record["viewing"]["working_scale_bar"]["value"] == 0.125

    def test_post_from_a_plain_rgb_view_round_trips(self, client, dated_dataset):
        """The exact shape ``coverageTracker.ts`` sends for a plain RGB view: ``bands``/
        ``stretch`` are absent (``compositeParams`` returns ``{}`` for a <=3-band raster, and
        ``JSON.stringify`` drops the resulting ``undefined`` keys), the other four viewing keys
        are always present."""
        _root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        body = {
            "image_path": path,
            "dataset_root": None,
            "subject": "bush",
            "date": "2026-03-01",
            "grid": _grid_only(grid),
            "cells_served_at_native": ["A1"],
            "cells_swept": [],
            "viewing": {
                "stats_source": {"read": "none"},
                "display_bounds": None,
                "base_served_size": "100x80",
                "working_scale_bar": None,
            },
        }
        resp = client.post("/api/coverage", json=body)
        assert resp.status_code == 200, resp.text
        record = client.get("/api/coverage", params={
            "path": path, "subject": "bush", "date": "2026-03-01"}).json()["coverage"]
        assert record["viewing"]["bands"] is None
        assert record["viewing"]["stretch"] is None
        assert record["viewing"]["stats_source"] == {
            "read": "none", "seed": None, "pixel_fraction": None, "overview_scale": None}

    def test_post_from_a_composite_view_round_trips(self, client, dated_dataset):
        """The shape a multiband composite view sends: ``bands``/``stretch`` carry real values
        and ``display_bounds`` carries one pair per displayed band."""
        _root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        body = {
            "image_path": path,
            "dataset_root": None,
            "subject": "bush",
            "date": "2026-03-01",
            "grid": _grid_only(grid),
            "cells_served_at_native": [],
            "cells_swept": ["A1"],
            "viewing": {
                "bands": "3,2,1",
                "stretch": "percent_clip",
                "stats_source": {"read": "window_sample", "seed": 0, "pixel_fraction": 0.5,
                                 "overview_scale": None},
                "display_bounds": [[0.0, 1000.0], [5.0, 20.0], [1.0, 2.0]],
                "base_served_size": "100x80",
                "working_scale_bar": None,
            },
        }
        resp = client.post("/api/coverage", json=body)
        assert resp.status_code == 200, resp.text
        record = client.get("/api/coverage", params={
            "path": path, "subject": "bush", "date": "2026-03-01"}).json()["coverage"]
        assert record["viewing"]["bands"] == "3,2,1"
        assert record["viewing"]["stretch"] == "percent_clip"
        assert record["viewing"]["display_bounds"] == [[0.0, 1000.0], [5.0, 20.0], [1.0, 2.0]]

    def test_a_viewing_with_an_undeclared_key_is_refused(self, client, dated_dataset):
        _root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        body = _post_body(path, ["A1"], grid)
        body["viewing"]["mystery_key"] = "oops"
        resp = client.post("/api/coverage", json=body)
        assert resp.status_code == 422

    def test_a_bare_string_stats_source_is_refused(self, client, dated_dataset):
        _root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        body = _post_body(path, ["A1"], grid)
        body["viewing"]["stats_source"] = "none"
        resp = client.post("/api/coverage", json=body)
        assert resp.status_code == 422

    def test_a_top_level_undeclared_key_is_refused(self, client, dated_dataset):
        """``CoveragePayload`` forbids extra keys the same way its nested ``viewing`` does,
        rather than dropping a top-level key the client sent and no one intended silently
        discarded."""
        _root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        body = _post_body(path, ["A1"], grid)
        body["mystery_top_level_key"] = "oops"
        resp = client.post("/api/coverage", json=body)
        assert resp.status_code == 422

    def test_viewing_must_be_explicit(self, client, dated_dataset):
        """``viewing`` carries no default, the same way ``date`` does not: an omitted key is the
        model's own required-field refusal, so a post can never silently record a viewing context
        no browser ever chose."""
        _root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        body = _post_body(path, ["A1"], grid)
        del body["viewing"]
        resp = client.post("/api/coverage", json=body)
        assert resp.status_code == 422
        assert any("viewing" in str(err.get("loc")) for err in resp.json()["detail"])

    def _seed_old_shape_record(self, root, bucket: str, image_name: str, grid: dict) -> None:
        import tcip_store as ts
        from tcip_mcp.dataset_layout import view_coverage_key

        normalized_grid = {k: grid[k] for k in
                           ("width", "height", "tile_size", "overlap", "cols", "rows")}
        old_record = {
            "grid": normalized_grid,
            "cells_served_at_native": ["A1"],
            "cells_swept": [],
            "viewing": {"stats_source": "none", "display_bounds": None,
                        "base_served_size": "100x80"},
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
        ts.replace(view_coverage_key(root), {bucket: {image_name: old_record}},
                  expect=ts.Version.ABSENT)

    def test_get_coverage_refuses_an_old_shape_stored_record_naming_the_conform_script(
        self, client, dated_dataset,
    ):
        root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        self._seed_old_shape_record(root, "bush/2026-03-01", "plot.tif", grid)

        resp = client.get("/api/coverage", params={
            "path": path, "subject": "bush", "date": "2026-03-01"})
        assert resp.status_code == 400
        assert "conform_view_coverage_viewing.py" in resp.json()["detail"]

    def test_post_coverage_merge_path_refuses_an_old_shape_stored_record(
        self, client, dated_dataset,
    ):
        root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        self._seed_old_shape_record(root, "bush/2026-03-01", "plot.tif", grid)

        resp = client.post("/api/coverage", json=_post_body(path, ["B1"], grid))
        assert resp.status_code == 400
        assert "conform_view_coverage_viewing.py" in resp.json()["detail"]

    def test_post_coverage_replace_path_over_an_old_shape_record_succeeds(
        self, client, dated_dataset,
    ):
        """A mismatched grid replaces the record wholesale, with nothing to merge into, so an
        old-shape stored record does not block it."""
        root, path = dated_dataset
        grid64 = _grid(client, path, tile_size=64)
        self._seed_old_shape_record(root, "bush/2026-03-01", "plot.tif", grid64)
        grid100 = _grid(client, path, tile_size=100)

        resp = client.post("/api/coverage", json=_post_body(path, ["A1"], grid100))
        assert resp.status_code == 200, resp.text
        assert resp.json()["replaced"] is True
        record = client.get("/api/coverage", params={
            "path": path, "subject": "bush", "date": "2026-03-01"}).json()["coverage"]
        assert record["grid"]["tile_size"] == 100

    def test_mismatched_grid_replaces_both_facts(self, client, dated_dataset):
        _root, path = dated_dataset
        client.post("/api/coverage",
                    json=_post_body(path, ["A1"], _grid(client, path, tile_size=64),
                                    cells_swept=["B1"]))
        resp = client.post("/api/coverage",
                           json=_post_body(path, [], _grid(client, path, tile_size=100),
                                           cells_swept=["A1"]))
        assert resp.json()["replaced"] is True
        record = client.get("/api/coverage", params={
            "path": path, "subject": "bush", "date": "2026-03-01"}).json()["coverage"]
        assert record["cells_served_at_native"] == []
        assert record["cells_swept"] == ["A1"]

    def test_unknown_swept_cell_is_refused(self, client, dated_dataset):
        _root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        resp = client.post("/api/coverage", json=_post_body(path, [], grid,
                                                            cells_swept=["Q7"]))
        assert resp.status_code == 400
        assert "Q7" in resp.json()["detail"]

    def test_audit_coalesces_to_one_entry_per_image(self, client, dated_dataset):
        root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        client.post("/api/coverage", json=_post_body(path, ["A1"], grid))
        client.post("/api/coverage", json=_post_body(path, ["B1"], grid))
        entries = _audit_entries(root, "gui_view_coverage")
        assert len(entries) == 1
        assert entries[0]["arguments"]["image_name"] == "plot.tif"


def test_view_coverage_path_locator(tmp_path):
    from tcip_mcp.dataset_layout import view_coverage_path

    assert view_coverage_path(tmp_path) == tmp_path / ".tcip" / "state" / "view_coverage.json"


class TestCompletenessRoute:
    def _toggle(self, client, path, grid, cell, subject="catkin", complete=True, **overrides):
        body = {"image_path": path, "subject": subject, "grid": _grid_only(grid), "cell": cell,
               "complete": complete, "user": "breeder"}
        body.update(overrides)
        return client.post("/api/coverage/completeness", json=body)

    def test_attest_then_unattest(self, client, dated_dataset):
        _root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        resp = self._toggle(client, path, grid, "A1", complete=True)
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "status": "ok", "complete": True, "cells_complete": ["A1"], "replaced": None}

        resp = self._toggle(client, path, grid, "A1", complete=False)
        assert resp.json() == {
            "status": "ok", "complete": False, "cells_complete": [], "replaced": None}

    def test_re_attesting_a_stale_cell_keeps_it_complete_with_a_fresh_digest(
        self, client, dated_dataset,
    ):
        """Explicit direction distinguishes a re-attest from a toggle: attest a cell, edit its
        annotation content so it reads stale, then post complete=True again. The cell must stay
        complete with a fresh (non-stale) digest, never flip to not-complete the way a toggle
        reading the stored state (rather than the posted direction) would."""
        from tcip_annotation import json_io
        from tcip_annotation.state import Annotation, BBox

        root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        assert self._toggle(client, path, grid, "A1", complete=True).status_code == 200

        label_path = root / "annotations" / "2026-03-01" / "plot.json"
        json_io.write_annotations(
            label_path, [Annotation(subject="catkin", geometry=BBox(1, 1, 9, 9))], 100, 80)
        stale_before = client.get(
            "/api/coverage/completeness", params={"path": path}).json()["by_subject"]["catkin"]
        assert stale_before["stale_cells"] == ["A1"]

        resp = self._toggle(client, path, grid, "A1", complete=True)
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "status": "ok", "complete": True, "cells_complete": ["A1"], "replaced": None}

        record = client.get(
            "/api/coverage/completeness", params={"path": path}).json()["by_subject"]["catkin"]
        assert record["cells_complete"] == ["A1"]
        assert record["stale_cells"] == []

    def test_attest_unattest_reattest_each_succeed_through_the_route(self, client, dated_dataset):
        """Admits valid work: every explicit direction the control can post succeeds, in the
        order a breeder would actually use them."""
        _root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)

        attest = self._toggle(client, path, grid, "A1", complete=True)
        assert attest.status_code == 200, attest.text
        assert attest.json()["complete"] is True

        unattest = self._toggle(client, path, grid, "A1", complete=False)
        assert unattest.status_code == 200, unattest.text
        assert unattest.json()["complete"] is False

        reattest = self._toggle(client, path, grid, "A1", complete=True)
        assert reattest.status_code == 200, reattest.text
        assert reattest.json()["complete"] is True
        assert reattest.json()["cells_complete"] == ["A1"]

    def test_unattest_of_a_never_attested_cell_writes_nothing(self, client, dated_dataset):
        """complete=false on a bucket with no existing record is an idempotent no-op: nothing
        to unattest, so no record is minted and block calibration's view is unchanged."""
        from tcip_mcp.pipelines.region_completeness import incomplete_cells_for_rect

        root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        resp = self._toggle(client, path, grid, "A1", complete=False)
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "status": "ok", "complete": False, "cells_complete": [], "replaced": None}

        assert client.get(
            "/api/coverage/completeness", params={"path": path}).json()["by_subject"] == {}
        assert incomplete_cells_for_rect(root, "catkin", Path(path).stem, (0, 0, 64, 64)) is None

    def test_attest_then_unattest_still_works(self, client, dated_dataset):
        """Admits valid work: the never-attested no-op does not disturb the ordinary
        attest/unattest sequence."""
        _root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        assert self._toggle(client, path, grid, "A1", complete=True).status_code == 200
        resp = self._toggle(client, path, grid, "A1", complete=False)
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "status": "ok", "complete": False, "cells_complete": [], "replaced": None}

    def test_complete_direction_is_required(self, client, dated_dataset):
        _root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        body = {"image_path": path, "subject": "catkin", "grid": _grid_only(grid), "cell": "A1",
               "user": "breeder"}
        resp = client.post("/api/coverage/completeness", json=body)
        assert resp.status_code == 422

    def test_subject_is_required(self, client, dated_dataset):
        _root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        resp = self._toggle(client, path, grid, "A1", subject="")
        assert resp.status_code == 400
        assert "subject" in resp.json()["detail"]

    def test_unknown_cell_is_refused(self, client, dated_dataset):
        _root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        resp = self._toggle(client, path, grid, "Z9")
        assert resp.status_code == 400
        assert "Z9" in resp.json()["detail"]

    def test_marking_a_cell_complete_refuses_an_unreadable_label(self, client, dated_dataset):
        """Marking a cell complete stamps a digest of the label file's current content; a
        document that will not read must refuse rather than stamp a digest of nothing."""
        from tcip_mcp.dataset_layout import annotation_path

        root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        label = annotation_path(root, "2026-03-01", Path(path).stem)
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("not json {][", encoding="utf-8")

        resp = self._toggle(client, path, grid, "A1")
        assert resp.status_code == 400
        assert str(label) in resp.json()["detail"]

    def test_by_stem_reads_back_the_attested_cell(self, client, dated_dataset):
        _root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        self._toggle(client, path, grid, "A1")
        self._toggle(client, path, grid, "B2")

        got = client.get("/api/coverage/completeness", params={"path": path})
        assert got.status_code == 200, got.text
        record = got.json()["by_subject"]["catkin"]
        assert record["cells_complete"] == ["A1", "B2"]
        assert record["attested_by"] == "user:breeder"
        assert record["stale_cells"] == []
        assert record["subject"] == "catkin"
        assert record["stem"] == Path(path).stem

    def test_different_subjects_do_not_collide(self, client, dated_dataset):
        _root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        self._toggle(client, path, grid, "A1", subject="catkin")
        self._toggle(client, path, grid, "B2", subject="bush")

        by_subject = client.get(
            "/api/coverage/completeness", params={"path": path}).json()["by_subject"]
        assert by_subject["catkin"]["cells_complete"] == ["A1"]
        assert by_subject["bush"]["cells_complete"] == ["B2"]

    def test_mismatched_grid_replaces_wholesale(self, client, dated_dataset):
        _root, path = dated_dataset
        self._toggle(client, path, _grid(client, path, tile_size=64), "A1")
        resp = self._toggle(client, path, _grid(client, path, tile_size=100), "A1")
        assert resp.json()["cells_complete"] == ["A1"]
        record = client.get(
            "/api/coverage/completeness", params={"path": path}).json()["by_subject"]["catkin"]
        assert record["grid"]["tile_size"] == 100

    def test_attest_on_a_new_lattice_returns_and_audits_the_replaced_cells(
        self, client, dated_dataset,
    ):
        """The previous-lattice discard is recorded, not just enacted: the response and the
        audit line both name the grid and cells a lattice-changing attest just overwrote, the
        way post_coverage already states its own replacement."""
        root, path = dated_dataset
        grid64 = _grid(client, path, tile_size=64)
        self._toggle(client, path, grid64, "A1")
        self._toggle(client, path, grid64, "B2")

        grid100 = _grid(client, path, tile_size=100)
        resp = self._toggle(client, path, grid100, "A1")
        assert resp.status_code == 200, resp.text
        replaced = resp.json()["replaced"]
        assert replaced["cells_complete"] == ["A1", "B2"]
        assert replaced["grid"]["tile_size"] == 64

        entries = _audit_entries(root, "gui_set_region_completeness")
        audited = entries[-1]["arguments"]["replaced"]
        assert audited["cells_complete"] == ["A1", "B2"]
        assert audited["grid"]["tile_size"] == 64

    def test_attest_on_the_same_lattice_carries_replaced_null(self, client, dated_dataset):
        _root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        self._toggle(client, path, grid, "A1")
        resp = self._toggle(client, path, grid, "B2")
        assert resp.status_code == 200, resp.text
        assert resp.json()["replaced"] is None

    def test_toggle_refuses_when_the_store_holds_an_unrecognized_entry(self, client, dated_dataset):
        """An entry the normalizer cannot read (a legacy or corrupt shape) must not be silently
        dropped by a write into an unrelated bucket; the write refuses, naming it, and the entry
        is still in the store afterwards."""
        import tcip_store as ts
        from tcip_mcp.dataset_layout import region_completeness_key

        root, path = dated_dataset
        stray_bucket = "orchard/2026-02-01"
        ts.replace(region_completeness_key(root), {stray_bucket: {"cells_complete": ["A1"]}},
                  expect=ts.Version.ABSENT)

        grid = _grid(client, path, tile_size=64)
        resp = self._toggle(client, path, grid, "A1")
        assert resp.status_code == 400
        assert stray_bucket in resp.json()["detail"]

        store = ts.read(region_completeness_key(root))
        assert store[stray_bucket] == {"cells_complete": ["A1"]}

    def test_toggle_refuses_over_a_non_dict_stored_document(self, client, dated_dataset):
        """A list- or string-shaped stored document is not something the per-bucket unreadable
        check names (it has no buckets to enumerate); the write must still refuse rather than
        read it as an empty store and replace it wholesale."""
        import tcip_store as ts
        from tcip_mcp.dataset_layout import region_completeness_key

        root, path = dated_dataset
        ts.replace(region_completeness_key(root), ["not", "a", "dict"], expect=ts.Version.ABSENT)

        grid = _grid(client, path, tile_size=64)
        resp = self._toggle(client, path, grid, "A1")
        assert resp.status_code == 400
        assert "list" in resp.json()["detail"]

        store = ts.read(region_completeness_key(root))
        assert store == ["not", "a", "dict"]

    def test_a_stale_attestation_is_detected_on_read(self, client, dated_dataset):
        """A cell is attested complete, then an annotation is added inside it: the stamped
        digest no longer matches, so the attestation reads back as stale (see
        tcip_mcp.pipelines.region_completeness.stale_cells) rather than silently trusted."""
        from tcip_annotation import json_io
        from tcip_annotation.state import Annotation, BBox

        root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        self._toggle(client, path, grid, "A1")

        label_path = root / "annotations" / "2026-03-01" / "plot.json"
        json_io.write_annotations(
            label_path, [Annotation(subject="catkin", geometry=BBox(1, 1, 9, 9))], 100, 80)

        record = client.get(
            "/api/coverage/completeness", params={"path": path}).json()["by_subject"]["catkin"]
        assert record["cells_complete"] == ["A1"]
        assert record["stale_cells"] == ["A1"]

    def test_reading_completeness_refuses_an_unreadable_label(self, client, dated_dataset):
        """Staleness is recomputed from the label file on every read; a document that will not
        read must refuse rather than silently score every cell fresh or every cell stale."""
        root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        self._toggle(client, path, grid, "A1")

        label_path = root / "annotations" / "2026-03-01" / "plot.json"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("not json {][", encoding="utf-8")

        resp = client.get("/api/coverage/completeness", params={"path": path})
        assert resp.status_code == 400
        assert str(label_path) in resp.json()["detail"]

    def test_audit_records_the_write(self, client, dated_dataset):
        root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        self._toggle(client, path, grid, "A1")
        entries = _audit_entries(root, "gui_set_region_completeness")
        assert len(entries) == 1
        assert entries[0]["arguments"] == {
            "image_name": "plot.tif", "subject": "catkin", "cell": "A1",
            "complete": True, "stem": "plot", "date": "2026-03-01", "replaced": None}

    def test_the_digest_is_named_before_the_record_that_attests_it(
        self, client, dated_dataset, monkeypatch,
    ):
        """One transaction over both stores, digest key named first.

        A transaction applies its staged writes in the order the keys were named, and
        ``stale_cells`` reports a cell carrying no stamp as stale, so an attestation applied
        ahead of its own stamp would read back stale the moment a breeder made it.
        """
        import tcip_store

        from tcip_mcp.dataset_layout import (
            region_completeness_digest_key,
            region_completeness_key,
        )

        root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        declared: list[tuple] = []
        real_transaction = tcip_store.transaction

        def recording(*keys, **kwargs):
            declared.append(keys)
            return real_transaction(*keys, **kwargs)

        monkeypatch.setattr(tcip_store, "transaction", recording)
        assert self._toggle(client, path, grid, "A1").status_code == 200

        assert declared == [
            (region_completeness_digest_key(root), region_completeness_key(root))]

    def test_region_completeness_path_locator(self, tmp_path):
        from tcip_mcp.dataset_layout import region_completeness_path

        assert region_completeness_path(tmp_path) == (
            tmp_path / ".tcip" / "state" / "region_completeness.json")
