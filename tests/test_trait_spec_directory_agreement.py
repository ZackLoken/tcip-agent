"""The registry, the breeder's trait picker, and the data-state doctor read one spec directory.

``traits`` owns where a project's trait specs live. The Results tab's trait list and
``scripts/doctor.py`` both report on that same directory, and each of them looking somewhere else
fails the same silent way: zero traits and zero broken-spec findings, which is exactly what a
project with nothing authored looks like. So the two surfaces are checked against the registry's
own resolution rather than against a path spelled out beside them.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tcip_web.app import app

from tcip_mcp import traits
from tcip_mcp.traits import load_trait_specs_with_errors, registered_traits_for

DOCTOR = str(Path(__file__).parent.parent / "scripts" / "doctor.py")


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _seed_registry(project_root: Path) -> Path:
    """Two valid specs with different vocabularies plus one that cannot load, all written into the
    directory the registry itself resolves, never a path spelled out here."""
    specs_dir = project_root / traits._TRAIT_SPECS_RELPATH
    specs_dir.mkdir(parents=True, exist_ok=True)
    (specs_dir / "leaf.yml").write_text(
        "name: leaf\ndelivers: [leaf_length, leaf_width]\nmilestone_fractions: [0.5]\n",
        encoding="utf-8")
    (specs_dir / "bloom.yml").write_text(
        "name: bloom\ndelivers: [bloom_50per_date]\nmilestone_fractions: [0.05, 0.5, 0.95]\n",
        encoding="utf-8")
    (specs_dir / "unicorn.yml").write_text(
        "name: unicorn\ndelivers: [unicorn_horn_length]\n", encoding="utf-8")
    return specs_dir


def test_the_traits_route_serves_what_the_registry_resolves(client: TestClient, tmp_path: Path):
    """Same project root, same answer: the route's trait names and broken-spec files are the
    registry's, so a breeder facing an empty picker is seeing an empty registry."""
    _seed_registry(tmp_path)
    registered = registered_traits_for(tmp_path)
    _specs, errors = load_trait_specs_with_errors(
        specs_dir=tmp_path / traits._TRAIT_SPECS_RELPATH)
    assert registered == ["bloom", "leaf"]
    assert [e["file"] for e in errors] == ["unicorn.yml"]

    body = client.get("/api/results/traits", params={"project_root": str(tmp_path)}).json()
    assert body["traits"] == registered
    assert [e["file"] for e in body["invalid_specs"]] == [e["file"] for e in errors]
    assert body["milestone_fractions_by_trait"] == {"bloom": [0.05, 0.5, 0.95], "leaf": [0.5]}


def test_the_doctor_reports_every_spec_the_registry_could_not_load(tmp_path: Path):
    """The session-start ritual reads the same directory: a spec the registry drops is named by
    file and reason, rather than the doctor finding nothing because it looked elsewhere."""
    _seed_registry(tmp_path)
    _specs, errors = load_trait_specs_with_errors(
        specs_dir=tmp_path / traits._TRAIT_SPECS_RELPATH)
    assert errors, "the seeded registry must contain a spec that fails to load"

    res = subprocess.run([sys.executable, DOCTOR, str(tmp_path)], capture_output=True, text=True)
    assert res.returncode == 2, res.stdout
    for e in errors:
        assert e["file"] in res.stdout
        assert "unicorn_horn_length" in res.stdout


def test_a_registry_with_nothing_broken_reports_nothing_broken(client: TestClient, tmp_path: Path):
    """The agreement above must not turn every project into a complaint: a registry whose specs all
    load serves its traits with an empty invalid list and no doctor finding."""
    specs_dir = tmp_path / traits._TRAIT_SPECS_RELPATH
    specs_dir.mkdir(parents=True, exist_ok=True)
    (specs_dir / "leaf.yml").write_text("name: leaf\ndelivers: [leaf_length]\n", encoding="utf-8")

    body = client.get("/api/results/traits", params={"project_root": str(tmp_path)}).json()
    assert body["traits"] == ["leaf"]
    assert body["invalid_specs"] == []

    res = subprocess.run([sys.executable, DOCTOR, str(tmp_path)], capture_output=True, text=True)
    assert "trait spec" not in res.stdout, res.stdout
