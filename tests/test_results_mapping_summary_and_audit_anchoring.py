"""What the Results plant-mapping doors report, and where the mapping they persist is audited.

Three separate facts travel back from a build: how many images a date holds, how many of them
resolved to a plant, and how far the GPS matches sat from the plant they resolved to. None of them
stands in for another, and a date's numbers are its own. The mapping a build persists is audited
into the project the GUI has open, the same project the persist path itself must sit under.

The mapping name a phenology door is handed is the door's own input to validate: a name naming no
mapping is refused by name, never carried forward into a phenology computed over nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tcip_web.app import app
from tcip_web.state import store

from tests.test_tcip_web_results_routes import _phenology_fixture

pytestmark = pytest.mark.usefixtures("seed_bud_operationalization")

PLANTS = (
    ("PLOT1", "AccA", 43.20000, -90.00000),
    ("PLOT2", "AccB", 43.20000, -90.00015),
    ("PLOT3", "AccC", 43.20000, -90.00030),
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _dms(deg: float) -> tuple[float, float, float]:
    d = int(abs(deg))
    minutes_full = (abs(deg) - d) * 60
    m = int(minutes_full)
    return float(d), float(m), (minutes_full - m) * 60


def _write_image(path: Path, when: str, lat: float | None = None, lon: float | None = None) -> None:
    """A JPEG carrying EXIF capture time and, unless the fix is missing, a GPS position."""
    img = Image.new("RGB", (8, 8), (10, 20, 30))
    exif = img.getexif()
    exif[306] = when
    exif[36867] = when
    if lat is not None and lon is not None:
        gps = exif.get_ifd(0x8825)
        gps[1] = "N" if lat >= 0 else "S"
        gps[2] = _dms(lat)
        gps[3] = "E" if lon >= 0 else "W"
        gps[4] = _dms(lon)
    img.save(path, exif=exif)


def _capture_fixture(root: Path) -> dict:
    """Three plants on one row, then two capture dates of unequal size against them.

    The first date holds a close match, a looser match, an image taken far from any plant, and an
    image with no GPS fix at all; the second holds a single exact match.
    """
    csv_path = root / "plants.csv"
    csv_path.write_text(
        "plot_name,accession_name,WGS84_centroid_x,WGS84_centroid_y\n"
        + "".join(f"{plot},{acc},{lon},{lat}\n" for plot, acc, lat, lon in PLANTS),
        encoding="utf-8",
    )
    images = root / "images"
    (images / "2026-02-11").mkdir(parents=True)
    (images / "2026-02-25").mkdir(parents=True)
    _write_image(images / "2026-02-11" / "a.jpg", "2026:02:11 09:00:00", 43.20000, -90.0000123)
    _write_image(images / "2026-02-11" / "b.jpg", "2026:02:11 09:01:00", 43.200027, -90.00015)
    _write_image(images / "2026-02-11" / "c.jpg", "2026:02:11 09:02:00", 43.30000, -90.00030)
    _write_image(images / "2026-02-11" / "e.jpg", "2026:02:11 09:03:00")
    _write_image(images / "2026-02-25" / "d.jpg", "2026:02:25 09:00:00", 43.20000, -90.00030)
    from tcip_mcp.tools.project_tools import register_dataset
    from tcip_mcp.traits import registered_crops

    register_dataset(str(root), crop=sorted(registered_crops())[0])
    from tests._binding_fixtures import register_plant_registry_for

    registry = register_plant_registry_for([csv_path])
    # The mapping doors build for the project the GUI has open, the one these captures belong to.
    store.open_project(root.resolve())
    return {
        "name": "valley", "images_root": str(images), "plant_registry": registry,
        # Not a BuildMappingPayload field (ignored by the route); carried for a test that needs
        # the registry's own source file, e.g. to prove it may live outside the project.
        "csv_path": str(csv_path),
    }


def test_build_reports_image_count_mapped_count_and_mean_distance_as_three_answers(
    client: TestClient, tmp_path: Path,
) -> None:
    """A date's summary keeps the three quantities apart. The fixture's first date makes them three
    different numbers (four images, two plants resolved, three GPS distances recorded), so a summary
    that reused one of them for another would read wrong rather than merely coincide."""
    from tcip_mcp.pipelines.postprocessing.plant_mapping import NEAREST_MATCH_FACTOR

    payload = _capture_fixture(tmp_path)
    body = client.post("/api/results/plant_mapping/build", json=payload).json()

    rows = body["mapping"]["2026-02-11"]
    assert len(rows) == 4
    assert {r["plot_name"] for r in rows if r["plot_name"]} == {"PLOT1", "PLOT2"}
    distances = [r["distance_m"] for r in rows if r["distance_m"] is not None]
    assert len(distances) == 3

    summary = body["summary"]["per_date"]["2026-02-11"]
    assert summary["n_images"] == 4
    assert summary["n_mapped"] == 2
    assert summary["n_unattributed"] == 2
    # summary() rounds to 2 places, the same as build_plant_mapping's own tool-side per_date,
    # so the two doors' summaries agree exactly rather than merely approximately.
    assert summary["avg_distance_m"] == round(sum(distances) / len(distances), 2)
    assert len({summary["n_images"], summary["n_mapped"], len(distances)}) == 3

    other = body["summary"]["per_date"]["2026-02-25"]
    assert other["n_images"] == 1
    assert other["n_mapped"] == 1
    assert other["n_unattributed"] == 0
    assert other["avg_distance_m"] == pytest.approx(0.0, abs=1e-6)

    totals = body["summary"]["totals"]
    assert totals == {
        "n_dates": 2, "n_images": summary["n_images"] + other["n_images"],
        "n_mapped": summary["n_mapped"] + other["n_mapped"],
        "n_unattributed": summary["n_unattributed"] + other["n_unattributed"],
    }

    assert set(body["nn_tolerance_m"]) == {"value", "source"}
    assert isinstance(body["nn_tolerance_m"]["value"], float)
    assert body["nn_tolerance_m"]["source"] in {"grid_pitch", "fallback", "stated", "stated_capped"}
    assert body["max_match_distance_m"] == pytest.approx(
        body["nn_tolerance_m"]["value"] * NEAREST_MATCH_FACTOR)


def test_build_without_a_stated_tolerance_derives_it_from_the_plot_grid_pitch(
    client: TestClient, tmp_path: Path,
) -> None:
    """A payload naming no ``nn_tolerance_m`` derives the match radius from the plot's own grid
    pitch, the same as the tool's own default; a route that pinned a numeric default instead
    would always land on ``stated``/``stated_capped``, never ``grid_pitch``."""
    from tcip_mcp.pipelines.postprocessing import plant_mapping

    payload = _capture_fixture(tmp_path)
    resp = client.post("/api/results/plant_mapping/build", json=payload)
    assert resp.status_code == 200, resp.text

    build = plant_mapping.load_mapping(tmp_path, payload["name"])
    assert build is not None
    assert build.nn_tolerance_m["source"] == "grid_pitch"


def test_build_response_carries_the_persisted_record_own_tolerance_dict(
    client: TestClient, tmp_path: Path,
) -> None:
    """The response's ``nn_tolerance_m`` equals the persisted record's own value: this cannot
    prove the route never recomputes it, only that whatever it answers agrees with what
    ``persist_mapping`` actually wrote."""
    from tcip_mcp.pipelines.postprocessing import plant_mapping

    payload = _capture_fixture(tmp_path)
    resp = client.post("/api/results/plant_mapping/build", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    build = plant_mapping.load_mapping(tmp_path, payload["name"])
    assert build is not None
    assert body["nn_tolerance_m"] == build.nn_tolerance_m


def test_a_mapping_persisted_into_platform_state_is_audited_into_the_owning_project(
    client: TestClient, tmp_path: Path,
) -> None:
    """A mapping is project state: the build always lands under the open project's own
    ``.tcip/state/plant_mappings/<name>.json``, by the name the payload names, and the audit
    row for it lands in that same project's log. There is no caller-chosen path left to anchor
    it elsewhere: the payload carries a name, never a location."""
    import tcip_store

    from tcip_mcp.audit import audit_log_key
    from tcip_mcp.pipelines.postprocessing import plant_mapping

    payload = _capture_fixture(tmp_path)
    resp = client.post("/api/results/plant_mapping/build", json=payload)
    assert resp.status_code == 200, resp.text
    build = plant_mapping.load_mapping(tmp_path, payload["name"])
    assert build is not None
    assert set(build.assignments.keys()) == {"2026-02-11", "2026-02-25"}

    page = tcip_store.read_log(audit_log_key(tmp_path))
    built = [r for r in page.records if r["tool"] == "gui_build_plant_mapping"]
    assert len(built) == 1
    assert built[0]["arguments"]["name"] == payload["name"]
    assert built[0]["arguments"]["n_dates"] == 2
    assert not (tmp_path / ".tcip" / ".tcip").exists()


def test_every_phenology_door_refuses_a_mapping_name_that_names_no_mapping(
    client: TestClient, tmp_path: Path,
) -> None:
    """With prediction buckets whose evidence is fully in order, a mapping name that names
    nothing at all is refused by name at every door: a legitimately-built mapping covering no
    dates refuses the requested dates it does not cover, and a name never built refuses as
    absent. Without either, the doors would answer with a phenology computed over no plants at
    all, which reads like a project with nothing to show."""
    from tests._binding_fixtures import write_plant_mapping

    body = _phenology_fixture(tmp_path, validated=True, detections=4)
    assert client.post("/api/results/phenology_measurement", json=body).status_code == 200

    write_plant_mapping(tmp_path, "empty", {}, dataset_root=tmp_path / "ds")
    for mapping_name, expected_status, expected_detail in (
        ("empty", 400, "does not cover"),
        ("not_written_yet", 404, "not_written_yet"),
    ):
        broken = {**body, "mapping_name": mapping_name}
        resp = client.post("/api/results/phenology_measurement", json=broken)
        assert resp.status_code == expected_status, mapping_name
        assert expected_detail in resp.json()["detail"]
        resp = client.post(
            "/api/results/export_csv",
            json={**broken, "payload": "milestones", "filename": "x.csv"})
        assert resp.status_code == expected_status
        assert expected_detail in resp.json()["detail"]
