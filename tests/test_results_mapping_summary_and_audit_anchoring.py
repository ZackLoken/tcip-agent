"""What the Results plant-mapping doors report, and where the mapping they persist is audited.

Three separate facts travel back from a build: how many images a date holds, how many of them
resolved to a plant, and how far the GPS matches sat from the plant they resolved to. None of them
stands in for another, and a date's numbers are its own. The mapping a build persists is audited
into the project the GUI has open.

A phenology route handed a mapping that does not cover its buckets refuses naming the date, and
one handed a name no build made refuses it as absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests._audit_fixtures import AUDIT_ENTRY_KEYS
from tests._mapping_fixtures import write_plant_csv

PLANTS = [
    {"plot": "PLOT1", "accession": "AccA", "lat": 43.20000, "lon": -90.00000},
    {"plot": "PLOT2", "accession": "AccB", "lat": 43.20000, "lon": -90.00015},
    {"plot": "PLOT3", "accession": "AccC", "lat": 43.20000, "lon": -90.00030},
]


def _dms(deg: float) -> tuple[float, float, float]:
    d = int(abs(deg))
    minutes_full = (abs(deg) - d) * 60
    m = int(minutes_full)
    return float(d), float(m), (minutes_full - m) * 60


def _write_image(
    path: Path,
    when: str,
    lat: float | None = None,
    lon: float | None = None,
    h_pos_err: float | None = None,
) -> None:
    """Write an image at ``path`` carrying EXIF capture time ``when``.

    With ``lat`` and ``lon`` given, its GPS block carries that position; with ``h_pos_err``
    given, the horizontal positioning error (meters) it reports.
    """
    from PIL import TiffImagePlugin

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
    if h_pos_err is not None:
        exif.get_ifd(0x8825)[31] = TiffImagePlugin.IFDRational(h_pos_err)
    img.save(path, exif=exif)


FIRST_DATE_ERRORS = {"a.jpg": 1.5, "b.jpg": 4.0, "c.jpg": 2.0}
"""The positioning errors (meters) the first date's photographs report, sample data."""


def _capture_fixture(root: Path) -> dict:
    """Write :data:`PLANTS` and two capture dates of photographs under ``root``; return the payload.

    It writes the plant CSV, four photographs on 2026-02-11 (three geotagged with
    :data:`FIRST_DATE_ERRORS`, one with no GPS) and one on 2026-02-25, opens ``root`` as the
    project, registers the dataset and plant registry, and returns the build payload named
    ``valley``.
    """
    csv_path = write_plant_csv(root / "plants.csv", PLANTS)
    images = root / "images"
    (images / "2026-02-11").mkdir(parents=True)
    (images / "2026-02-25").mkdir(parents=True)
    _write_image(
        images / "2026-02-11" / "a.jpg",
        "2026:02:11 09:00:00",
        43.20000,
        -90.0000123,
        FIRST_DATE_ERRORS["a.jpg"],
    )
    _write_image(
        images / "2026-02-11" / "b.jpg",
        "2026:02:11 09:01:00",
        43.200027,
        -90.00015,
        FIRST_DATE_ERRORS["b.jpg"],
    )
    _write_image(
        images / "2026-02-11" / "c.jpg",
        "2026:02:11 09:02:00",
        43.30000,
        -90.00030,
        FIRST_DATE_ERRORS["c.jpg"],
    )
    _write_image(images / "2026-02-11" / "e.jpg", "2026:02:11 09:03:00")
    _write_image(
        images / "2026-02-25" / "d.jpg",
        "2026:02:25 09:00:00",
        43.20000,
        -90.00030,
        1.0,
    )
    from tcip_mcp.tools.project_tools import register_dataset
    from tcip_mcp.traits import registered_crops
    from tests._mapping_fixtures import register_plant_registry_for
    from tests._web_fixtures import open_new_project

    # The mapping doors build for the project the GUI has open, the one these captures belong to.
    open_new_project(root)
    register_dataset(root, str(root), sorted(registered_crops())[0])
    registry = register_plant_registry_for(root, [csv_path])
    return {
        "name": "valley",
        "images_root": str(images),
        "plant_registry": registry,
        "user": "tester",
        # Not a BuildMappingPayload field (ignored by the route); carried for a test that needs
        # the registry's own source file, e.g. to prove it may live outside the project.
        "csv_path": str(csv_path),
    }


def test_build_reports_image_count_mapped_count_and_mean_distance_as_three_answers(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """A date's summary keeps the three quantities apart.

    The fixture's first date makes them three different numbers (four images, two plants
    resolved, three GPS distances recorded), and the summary reports each as its own.
    """
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        load_mapping_rows,
    )

    payload = _capture_fixture(tmp_path)
    body = client.post("/api/results/plant_mapping/build", json=payload).json()

    rows = load_mapping_rows(tmp_path, payload["name"])["2026-02-11"]
    assert len(rows) == 4
    assert {r["plant_id"] for r in rows if r["plant_id"]} == {
        "PLOT1",
        "PLOT2",
    }
    distances = [r["distance_m"] for r in rows if r["distance_m"] is not None]
    assert len(distances) == 3

    summary = body["summary"]["per_date"]["2026-02-11"]
    assert summary["n_images"] == 4
    assert summary["n_mapped"] == 2
    assert summary["n_unattributed"] == 2
    # summary() rounds to 2 places, the same as build_plant_mapping's own tool-side per_date,
    # so the two doors' summaries agree exactly rather than merely approximately.
    assert summary["avg_distance_m"] == round(
        sum(distances) / len(distances), 2
    )
    assert len({summary["n_images"], summary["n_mapped"], len(distances)}) == 3

    other = body["summary"]["per_date"]["2026-02-25"]
    assert other["n_images"] == 1
    assert other["n_mapped"] == 1
    assert other["n_unattributed"] == 0
    assert other["avg_distance_m"] == pytest.approx(0.0, abs=1e-6)

    totals = body["summary"]["totals"]
    assert totals == {
        "n_dates": 2,
        "n_images": summary["n_images"] + other["n_images"],
        "n_mapped": summary["n_mapped"] + other["n_mapped"],
        "n_unattributed": summary["n_unattributed"] + other["n_unattributed"],
    }

    assert body["match_radius"] == {
        "2026-02-11": {
            "value": max(FIRST_DATE_ERRORS.values()),
            "source": "h_pos_err",
            "rationale": None,
        },
        "2026-02-25": {"value": 1.0, "source": "h_pos_err", "rationale": None},
    }


def test_build_without_a_stated_radius_derives_each_dates_from_its_photographs(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """A build naming no ``match_radius`` derives each date's from its photographs.

    A payload naming no ``match_radius`` records each date's radius as derived from the
    positioning errors its photographs report (``h_pos_err``).
    """
    from tcip_mcp.pipelines.postprocessing import plant_mapping

    payload = _capture_fixture(tmp_path)
    resp = client.post("/api/results/plant_mapping/build", json=payload)
    assert resp.status_code == 200, resp.text

    build = plant_mapping.load_mapping(tmp_path, payload["name"])
    assert build is not None
    assert {r.source for r in build.match_radius.values()} == {"h_pos_err"}


def test_build_response_carries_the_persisted_record_own_match_radius(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """The build response's ``match_radius`` equals the persisted record's own."""
    from tcip_mcp.pipelines.postprocessing import plant_mapping

    payload = _capture_fixture(tmp_path)
    resp = client.post("/api/results/plant_mapping/build", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    build = plant_mapping.load_mapping(tmp_path, payload["name"])
    assert build is not None
    assert body["match_radius"] == build.to_record()["match_radius"]


def test_the_route_and_the_tool_build_one_mapping_with_one_audit_line_each(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """Identical builds through the HTTP route and the MCP tool answer, persist and audit alike.

    Their answers, their persisted records and their audit lines' arguments agree with
    ``name``, ``built_at`` and ``record_sha256`` set aside. The two builds leave two
    ``plant_mapping_built`` lines in the open project's log, each naming its build and the
    digest of the record it wrote. The route's line carries ``actor`` ``user:tester``, the
    tool's carries no ``actor``, and nothing else on the route's line names the tester.
    """
    import json

    import tcip_store
    from tcip_mcp.audit import audit_log_key
    from tcip_mcp.pipelines.postprocessing import plant_mapping
    from tcip_mcp.tools.phenology_tools import build_plant_mapping

    payload = _capture_fixture(tmp_path)
    before = len(tcip_store.read_log(audit_log_key(tmp_path)).records)
    resp = client.post("/api/results/plant_mapping/build", json=payload)
    assert resp.status_code == 200, resp.text
    tool_answer = build_plant_mapping(
        tmp_path,
        name="ridge",
        images_root=payload["images_root"],
        plant_registry=payload["plant_registry"],
    )
    assert "error" not in tool_answer, tool_answer

    def own(record: dict) -> dict:
        return {
            k: v
            for k, v in record.items()
            if k not in ("name", "built_at", "record_sha256")
        }

    assert own(resp.json()) == own(tool_answer)
    route_record = tcip_store.read(
        plant_mapping.plant_mapping_key(tmp_path, payload["name"])
    )
    tool_record = tcip_store.read(
        plant_mapping.plant_mapping_key(tmp_path, "ridge")
    )
    assert own(route_record) == own(tool_record)

    lines = tcip_store.read_log(audit_log_key(tmp_path)).records[before:]
    assert [line["tool"] for line in lines] == ["plant_mapping_built"] * 2
    assert [line["arguments"]["name"] for line in lines] == [
        payload["name"],
        "ridge",
    ]
    assert [line["arguments"]["record_sha256"] for line in lines] == [
        plant_mapping.record_digest(route_record),
        plant_mapping.record_digest(tool_record),
    ]
    assert own(lines[0]["arguments"]) == own(lines[1]["arguments"])
    assert lines[0]["actor"] == "user:tester" and "actor" not in lines[1]
    assert (
        set(lines[0]) - {"actor"} <= AUDIT_ENTRY_KEYS
        and set(lines[1]) <= AUDIT_ENTRY_KEYS
    )
    assert "tester" not in json.dumps(
        {k: v for k, v in lines[0].items() if k != "actor"}
    )
    assert not (tmp_path / ".tcip" / ".tcip").exists()


def test_a_mapping_or_registry_name_outside_the_segment_rule_refuses_at_its_key(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """A mapping or registry name outside the segment rule is refused at its key producer.

    Both key producers refuse the name, and the build route, the build tool and the registry
    registration each refuse it by name with no mapping written; a legal name and an archived
    one key.
    """
    from tcip_mcp.pipelines.postprocessing import plant_mapping
    from tcip_mcp.tools.phenology_tools import (
        build_plant_mapping,
        register_plant_registry,
    )

    payload = _capture_fixture(tmp_path)
    with pytest.raises(ValueError, match="Bad_Name"):
        plant_mapping.plant_mapping_key(tmp_path, "Bad_Name")
    with pytest.raises(ValueError, match="Bad_Name"):
        plant_mapping.plant_registry_key(tmp_path, "Bad_Name")
    plant_mapping.plant_mapping_key(tmp_path, "valley")
    plant_mapping.plant_mapping_key(tmp_path, f"valley@{'0' * 12}")

    resp = client.post(
        "/api/results/plant_mapping/build",
        json={**payload, "name": "Bad_Name"},
    )
    assert resp.status_code == 400 and "Bad_Name" in resp.json()["detail"]
    answer = build_plant_mapping(
        tmp_path,
        name="Bad_Name",
        images_root=payload["images_root"],
        plant_registry=payload["plant_registry"],
    )
    assert "Bad_Name" in answer["error"]
    registered = register_plant_registry(
        tmp_path,
        name="Bad_Name",
        csv_paths=[payload["csv_path"]],
        crop="c",
        site="s",
    )
    assert "Bad_Name" in registered["error"]
    assert plant_mapping.plant_mapping_names(tmp_path) == []


def test_every_phenology_door_refuses_a_mapping_name_that_names_no_mapping(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """Two phenology routes refuse by name a mapping that does not cover the buckets.

    With assessed prediction buckets, a mapping that does not cover them is refused by name at
    the measurement and CSV export routes: one built over the first date alone refuses the date
    it does not cover, and a name no build made refuses as absent.
    """
    pytest.importorskip("torch")
    from tcip_mcp.tools.phenology_tools import build_plant_mapping
    from tests._chain_fixtures import attributed_series

    series = attributed_series(tmp_path, fractions=(0.0, 1.0))
    body = series.body()
    assert (
        client.post(
            "/api/results/phenology_measurement", json=body
        ).status_code
        == 200
    )

    built = build_plant_mapping(
        tmp_path,
        name="partial",
        images_root=str(series.root / "images"),
        plant_registry="reg",
        dates=sorted(series.buckets)[:1],
    )
    assert "error" not in built, built
    for mapping_name, expected_status, expected_detail in (
        ("partial", 400, "does not cover"),
        ("not-written-yet", 404, "not-written-yet"),
    ):
        broken = {**body, "mapping_name": mapping_name}
        resp = client.post("/api/results/phenology_measurement", json=broken)
        assert resp.status_code == expected_status, mapping_name
        assert expected_detail in resp.json()["detail"]
        resp = client.post(
            "/api/results/export_csv",
            json={
                **broken,
                "payload": "milestones",
                "filename": "x.csv",
                "user": "tester",
            },
        )
        assert resp.status_code == expected_status
        assert expected_detail in resp.json()["detail"]
