"""A second registered trait, driven through both phenology doors (the measurement and
export_csv), asserting its own schema and that an unassessed delivery is refused, so the delivery
path is exercised against more than one registered trait.

``currant_bloom`` is authored here, in each test's own project, honestly
tentative: no domain expert has confirmed it, and it exists to prove the delivery mechanism
generalizes to a real *second* trait, not to describe a validated measurement. It deliberately
leaves ``majority_milestone``/``majority_label`` empty rather than copied from bud_opening, since
crops.yml names no "majority" bloom date for currant, proving ``_milestone_columns`` produces the
smaller, no-majority column set for a real second trait, not only for local ``TraitEntry`` shapes
constructed to prove the structural invariant.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_mcp import subject_registry as cr
from tcip_web.app import app

BLOOM_STATE = cr.Attribute("bloom_state", "categorical", ("closed", "open"))
"""The attribute of ``flower`` the second trait's positive state names."""
FLOWERS = cr.SubjectRegistry(subjects=(cr.Subject(name="flower", attributes=(BLOOM_STATE,)),))


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _seed_currant_bloom_trait(tmp_path: Path) -> None:
    from tests._trait_fixtures import entry, propose, seed_confirmed_crossing

    propose(tmp_path, entry(
        "currant_bloom", ("bloom_05per_date", "bloom_50per_date", "bloom_95per_date"),
        count_objective="count_unbiased", localization="center_match",
        positive_state={"attribute": BLOOM_STATE.name, "value": "open"},
        milestone_fractions=(0.05, 0.50, 0.95), milestone_on="positive_fraction",
        # No majority alias: crops.yml names no single "most blooms open" date for currant,
        # unlike bud_opening's bud_majority_date. Left empty rather than copied from bud_opening.
        phenology_prefix="bloom", count_bias_tolerance_frac=0.01,
        notes="Test-only, provisional: proves the delivery mechanism generalizes to a second "
              "trait. Not a domain-expert-confirmed measurement.",
    ))
    # A second trait needs its own confirmed meaning too: nothing about the record is bud_opening-shaped.
    seed_confirmed_crossing(tmp_path, "currant_bloom", measured_subject="flower")


def _currant_bloom_body(tmp_path: Path) -> dict:
    """A registered dataset of geolocated captures over two dates, one unassessed bucket per date
    whose scope declares the positive state's attribute, the mapping over them, and ``currant_bloom`` confirmed, with ``tmp_path`` open in the
    web backend; the request body a phenology door takes."""
    import asyncio

    from tcip_mcp.tools.phenology_tools import build_plant_mapping
    from tcip_web.state import store
    from tests._mapping_fixtures import register_plant_registry_for
    from tests.test_plant_mapping_binding import DATES, PLANTS, _dataset, _init, _write_scene

    _init(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(_dataset(tmp_path), dates=DATES)
    registry = register_plant_registry_for(tmp_path, [plant_csv])
    built = build_plant_mapping(tmp_path, name="valley", images_root=str(images_root),
                                plant_registry=registry)
    assert "error" not in built, built
    _seed_currant_bloom_trait(tmp_path)
    asyncio.run(store.open_project(tmp_path.resolve()))
    return {"mapping_name": "valley", "buckets": list(preds_by_date.values()),
            "trait": "currant_bloom", "plants": [p["plot"] for p in PLANTS]}


def _export(client: TestClient, body: dict, payload: str = "milestones", **extra):
    return client.post("/api/results/export_csv",
                       json={**body, "payload": payload, "filename": "x.csv", **extra})


def test_currant_bloom_is_registered_and_distinct_from_bud_opening(tmp_path: Path):
    from tcip_mcp.traits import read_trait, trait_names

    _seed_currant_bloom_trait(tmp_path)
    assert "currant_bloom" in trait_names(tmp_path)
    t = read_trait("currant_bloom", tmp_path).latest.entry
    assert t.delivers == ("bloom_05per_date", "bloom_50per_date", "bloom_95per_date")
    assert t.majority_milestone == ""  # no majority alias, unlike bud_opening


def test_currant_bloom_measurement_carries_its_own_milestone_columns(
    client: TestClient, tmp_path: Path,
) -> None:
    body = _currant_bloom_body(tmp_path)

    resp = client.post("/api/results/phenology_measurement",
                       json={**body, "show_unvalidated": True})

    assert resp.status_code == 200, resp.text[:300]
    out = resp.json()
    assert out["curves"]["rows"] and out["milestones"]["rows"]
    assert out["milestones"]["columns"] == [
        {"date": f"bloom_{p}per_date", "bound": f"bloom_{p}per_date_bound"}
        for p in ("05", "50", "95")]
    onset = out["milestones"]["rows"][0]
    assert "bud_majority_date" not in onset
    assert "bloom_opening_date" not in onset


def test_currant_bloom_export_csv_delivers_its_own_schema(client: TestClient, tmp_path: Path) -> None:
    body = _currant_bloom_body(tmp_path)

    from tests._web_fixtures import acknowledged_post

    resp = acknowledged_post(client, "/api/results/export_csv",
                             {**body, "payload": "milestones", "filename": "x.csv"},
                             reason="a second trait's schema")

    assert resp.status_code == 200, resp.text[:300]
    header = resp.text.splitlines()[0].split(",")
    assert "bloom_05per_date" in header and "bloom_95per_date" in header
    assert not any(c.startswith("bud_") for c in header)


def test_currant_bloom_refuses_unassessed_evidence_on_every_door(
    client: TestClient, tmp_path: Path,
) -> None:
    body = _currant_bloom_body(tmp_path)

    resp = client.post("/api/results/phenology_measurement", json=body)

    assert resp.status_code == 400
    assert resp.json()["detail"]["kind"] == "delivery"
    assert "no assessment answers" in resp.json()["detail"]["message"]
    for payload in ("curves", "milestones"):
        assert _export(client, body, payload).status_code == 400, payload
