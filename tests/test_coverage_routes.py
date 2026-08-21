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


def _post_body(image_path: str, cells: list[str], grid: dict, **overrides) -> dict:
    body = {
        "image_path": image_path,
        "subject": "bush",
        "date": "2026-03-01",
        "grid": grid,
        "cells_served_at_native": cells,
        "viewing": {"bands": None, "stretch": "minmax", "stats_source": "none",
                    "base_served_size": "100x80", "display_bounds": None},
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
        _root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        body = _post_body(path, ["A1"], grid)
        del body["date"]
        resp = client.post("/api/coverage", json=body)
        assert resp.status_code == 400
        assert "date" in resp.json()["detail"]

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
        viewing = {"bands": None, "stretch": "minmax", "stats_source": "none",
                   "base_served_size": "100x80", "display_bounds": None,
                   "working_scale_bar": {
                       "value": 0.125,
                       "source": "minimum view scale at annotation-authoring events"}}
        client.post("/api/coverage",
                    json=_post_body(path, [], grid, cells_swept=["A1"], viewing=viewing))
        record = client.get("/api/coverage", params={
            "path": path, "subject": "bush", "date": "2026-03-01"}).json()["coverage"]
        assert record["viewing"]["working_scale_bar"]["value"] == 0.125

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
    def _toggle(self, client, path, grid, cell, subject="catkin", **overrides):
        body = {"image_path": path, "subject": subject, "grid": grid, "cell": cell,
               "user": "breeder"}
        body.update(overrides)
        return client.post("/api/coverage/completeness", json=body)

    def test_toggle_on_then_off(self, client, dated_dataset):
        _root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        resp = self._toggle(client, path, grid, "A1")
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"status": "ok", "complete": True, "cells_complete": ["A1"]}

        resp = self._toggle(client, path, grid, "A1")
        assert resp.json() == {"status": "ok", "complete": False, "cells_complete": []}

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

    def test_audit_records_the_write(self, client, dated_dataset):
        root, path = dated_dataset
        grid = _grid(client, path, tile_size=64)
        self._toggle(client, path, grid, "A1")
        entries = _audit_entries(root, "gui_set_region_completeness")
        assert len(entries) == 1
        assert entries[0]["arguments"] == {
            "image_name": "plot.tif", "subject": "catkin", "cell": "A1",
            "complete": True, "stem": "plot", "date": "2026-03-01"}

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
