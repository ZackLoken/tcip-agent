"""What the Results plant-mapping doors report, and where the mapping they persist is audited.

Three separate facts travel back from a build: how many images a date holds, how many of them
resolved to a plant, and how far the GPS matches sat from the plant they resolved to. None of them
stands in for another, and a date's numbers are its own. The mapping a build persists into platform
state is audited into the project that owns that state directory, resolved from the ``.tcip`` marker
in the path rather than from a fixed depth beneath it.

The mapping path a phenology door is handed is the door's own input to validate: a path naming no
mapping is refused by name, never carried forward into a phenology computed over nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tcip_web.app import app

from tests.test_tcip_web_results_routes import _phenology_fixture

pytestmark = pytest.mark.usefixtures("seed_catkin_trait_spec")

PLANTS = (
    ("PLOT1", "AccA", 43.20000, -90.00000),
    ("PLOT2", "AccB", 43.20000, -90.00015),
    ("PLOT3", "AccC", 43.20000, -90.00030),
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


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
    return {"images_root": str(images), "plant_csv_paths": [str(csv_path)]}


def test_build_reports_image_count_mapped_count_and_mean_distance_as_three_answers(
    client: TestClient, tmp_path: Path,
) -> None:
    """A date's summary keeps the three quantities apart. The fixture's first date makes them three
    different numbers (four images, two plants resolved, three GPS distances recorded), so a summary
    that reused one of them for another would read wrong rather than merely coincide."""
    payload = _capture_fixture(tmp_path)
    body = client.post("/api/results/plant_mapping/build", json=payload).json()

    rows = body["mapping"]["2026-02-11"]
    assert len(rows) == 4
    assert {r["plot_name"] for r in rows if r["plot_name"]} == {"PLOT1", "PLOT2"}
    distances = [r["distance_m"] for r in rows if r["distance_m"] is not None]
    assert len(distances) == 3

    summary = body["summary"]["2026-02-11"]
    assert summary["n_images"] == 4
    assert summary["n_mapped"] == 2
    assert summary["avg_distance_m"] == pytest.approx(sum(distances) / len(distances))
    assert len({summary["n_images"], summary["n_mapped"], len(distances)}) == 3

    other = body["summary"]["2026-02-25"]
    assert other["n_images"] == 1
    assert other["n_mapped"] == 1
    assert other["avg_distance_m"] == pytest.approx(0.0, abs=1e-6)


def test_a_mapping_persisted_into_platform_state_is_audited_into_the_owning_project(
    client: TestClient, tmp_path: Path,
) -> None:
    """The audit row lands in the project that owns the state directory the mapping was written
    into, found from the ``.tcip`` marker itself, so a mapping stored deeper under state is still
    attributed to the project rather than to some directory a fixed number of levels up."""
    payload = _capture_fixture(tmp_path)
    persist_path = tmp_path / ".tcip" / "state" / "mappings" / "valley" / "plant_mapping.json"
    resp = client.post(
        "/api/results/plant_mapping/build", json={**payload, "persist_path": str(persist_path)})
    assert resp.status_code == 200
    assert json.loads(persist_path.read_text(encoding="utf-8")).keys() == {
        "2026-02-11", "2026-02-25"}

    audit_path = tmp_path / ".tcip" / "audit.jsonl"
    assert audit_path.is_file()
    rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line]
    built = [r for r in rows if r["tool"] == "gui_build_plant_mapping"]
    assert len(built) == 1
    assert built[0]["arguments"]["persist_path"] == str(persist_path)
    assert built[0]["arguments"]["n_dates"] == 2
    assert not (tmp_path / ".tcip" / ".tcip").exists()


def test_a_mapping_persisted_outside_platform_state_writes_no_audit_row(
    client: TestClient, tmp_path: Path,
) -> None:
    """The audit seam belongs to platform state. A mapping the breeder parks somewhere else is
    still built and persisted, and no project is credited with an audit row for it."""
    payload = _capture_fixture(tmp_path)
    persist_path = tmp_path / "exports" / "plant_mapping.json"
    resp = client.post(
        "/api/results/plant_mapping/build", json={**payload, "persist_path": str(persist_path)})
    assert resp.status_code == 200
    assert persist_path.is_file()
    assert list(tmp_path.rglob("audit.jsonl")) == []


def test_every_phenology_door_refuses_a_mapping_path_that_names_no_mapping(
    client: TestClient, tmp_path: Path,
) -> None:
    """With prediction buckets whose evidence is fully in order, a mapping path holding no
    assignments is refused by name at every door. Without that the doors would answer with a
    phenology computed over no plants at all, which reads like a project with nothing to show."""
    body = _phenology_fixture(tmp_path, validated=True, detections=4)
    assert client.post("/api/results/onset_dates", json=body).status_code == 200

    empty = tmp_path / "empty_mapping.json"
    empty.write_text("{}", encoding="utf-8")
    absent = tmp_path / "not_written_yet.json"
    for mapping_path in (empty, absent):
        broken = {**body, "mapping_path": str(mapping_path)}
        for route in ("per_plant_curves", "onset_dates"):
            resp = client.post(f"/api/results/{route}", json=broken)
            assert resp.status_code == 404, (route, mapping_path.name)
            assert str(mapping_path) in resp.json()["detail"], route
        resp = client.post(
            "/api/results/export_csv",
            json={**broken, "payload": "milestones", "filename": "x.csv"})
        assert resp.status_code == 404
        assert str(mapping_path) in resp.json()["detail"]
