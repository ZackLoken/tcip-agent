"""The registry, the breeder's trait picker, and the data-state doctor read one spec directory.

``traits`` owns where a project's trait specs live. The Results tab's trait list and
``tcip doctor`` both report on that same directory, and each of them looking somewhere else
fails the same silent way: zero traits and zero broken-spec findings, which is exactly what a
project with nothing authored looks like. So the two surfaces are checked against the registry's
own resolution rather than against a path spelled out beside them.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import tcip_store as ts
from fastapi.testclient import TestClient
from tcip_web.app import app

from tcip_mcp import traits
from tcip_mcp.traits import load_trait_specs_with_errors, registered_traits_for

@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _seed_registry(project_root: Path) -> Path:
    """Two valid specs with different vocabularies plus one that cannot load, all written into the
    directory the registry itself resolves, never a path spelled out here."""
    specs_dir = project_root / traits._TRAIT_SPECS_RELPATH
    ts.replace(
        traits.trait_spec_key(specs_dir, "leaf"),
        {"name": "leaf", "delivers": ["leaf_length", "leaf_width"], "milestone_fractions": [0.5]},
        expect=ts.Version.ABSENT)
    ts.replace(
        traits.trait_spec_key(specs_dir, "bloom"),
        {"name": "bloom", "delivers": ["bloom_50per_date"], "milestone_fractions": [0.05, 0.5, 0.95]},
        expect=ts.Version.ABSENT)
    ts.replace(
        traits.trait_spec_key(specs_dir, "unicorn"),
        {"name": "unicorn", "delivers": ["unicorn_horn_length"]},
        expect=ts.Version.ABSENT)
    return specs_dir


def test_the_traits_route_serves_what_the_registry_resolves(client: TestClient, tmp_path: Path):
    """Same project root, same answer: the route's trait names and broken-spec files are the
    registry's, so a breeder facing an empty picker is seeing an empty registry."""
    _seed_registry(tmp_path)
    registered = registered_traits_for(tmp_path)
    _specs, errors = load_trait_specs_with_errors(
        specs_dir=tmp_path / traits._TRAIT_SPECS_RELPATH)
    assert registered == ["bloom", "leaf"]
    assert [e["file"] for e in errors] == ["unicorn.json"]

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

    res = subprocess.run(
        [sys.executable, "-m", "tcip_web.cli", "doctor", str(tmp_path)],
        capture_output=True, text=True)
    assert res.returncode == 2, res.stdout
    for e in errors:
        assert e["file"] in res.stdout
        assert "unicorn_horn_length" in res.stdout


def _load_doctor():
    from tcip_mcp.cli import doctor

    return doctor


def test_the_traits_route_follows_the_registry_when_the_registry_moves(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """The route resolves the spec directory through ``trait_specs_dir()`` rather than a path
    spelled out beside it. Renaming the store's own final directory segment is no longer
    supported (the locator's ``trait_specs`` prefix is fixed, matching every sibling STATE-rooted
    store); this proves centralization instead by redirecting ``trait_specs_dir()`` itself to a
    different, still canonically-named directory and checking the route follows it there."""
    redirected = tmp_path / "elsewhere" / ".tcip" / "state" / "trait_specs"
    monkeypatch.setattr(traits, "trait_specs_dir", lambda project_root=None: redirected)
    ts.replace(
        traits.trait_spec_key(redirected, "leaf"), {"name": "leaf", "delivers": ["leaf_length"]},
        expect=ts.Version.ABSENT)

    body = client.get("/api/results/traits", params={"project_root": str(tmp_path)}).json()
    assert body["traits"] == registered_traits_for(tmp_path) == ["leaf"]


def test_the_doctor_follows_the_registry_when_the_registry_moves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Same demand on the session-start ritual: it reports on the directory ``trait_specs_dir()``
    resolves, so a redirected registry cannot turn every broken spec into a silent clean bill."""
    redirected = tmp_path / "elsewhere" / ".tcip" / "state" / "trait_specs"
    monkeypatch.setattr(traits, "trait_specs_dir", lambda project_root=None: redirected)
    ts.replace(
        traits.trait_spec_key(redirected, "unicorn"),
        {"name": "unicorn", "delivers": ["unicorn_horn_length"]}, expect=ts.Version.ABSENT)

    findings: list[tuple[str, str]] = []
    _load_doctor().check_trait_specs(tmp_path, findings)

    assert [f for f in findings if "unicorn.json" in f[1]]


def test_the_doctor_names_a_broken_spec_without_touching_it(tmp_path: Path):
    """The ritual diagnoses, it never edits. A spec record the registry cannot decode is reported
    by name and reason, and its bytes are still on disk exactly as they were afterwards."""
    specs_dir = tmp_path / traits._TRAIT_SPECS_RELPATH
    specs_dir.mkdir(parents=True)
    broken = specs_dir / "broken.json"
    broken.write_text("not valid json {", encoding="utf-8")
    before = broken.read_bytes()

    # A malformed record's bytes exist only as a loose file, which the file backend serves
    # directly; the database backend refuses to read a root holding files it did not write.
    env = {**os.environ, "TCIP_STORE_BACKEND": "file"}
    res = subprocess.run(
        [sys.executable, "-m", "tcip_web.cli", "doctor", str(tmp_path)],
        capture_output=True, text=True, env=env)

    assert res.returncode == 2, f"the doctor did not report the broken spec: {res.stdout}"
    assert "broken.json" in res.stdout
    assert broken.read_bytes() == before


def test_a_registry_with_nothing_broken_reports_nothing_broken(client: TestClient, tmp_path: Path):
    """The agreement above must not turn every project into a complaint: a registry whose specs all
    load serves its traits with an empty invalid list and no doctor finding."""
    specs_dir = tmp_path / traits._TRAIT_SPECS_RELPATH
    ts.replace(
        traits.trait_spec_key(specs_dir, "leaf"), {"name": "leaf", "delivers": ["leaf_length"]},
        expect=ts.Version.ABSENT)
    # A statement makes this registry genuinely nothing-broken under check_trait_spec_statements too.
    ts.replace(
        traits.trait_spec_statement_key(traits.trait_spec_statements_scope(tmp_path), "leaf"),
        {"trait": "leaf", "statement_fields": {"delivers": ["leaf_length"]},
         "rationale": "test fixture", "stated_by": "test", "stated_at": "2026-03-04T00:00:00+00:00",
         "relayed_note": "", "confirmed_by": None, "confirmed_at": None,
         "identity_from_request": None, "record_seen": None},
        expect=ts.Version.ABSENT,
    )

    body = client.get("/api/results/traits", params={"project_root": str(tmp_path)}).json()
    assert body["traits"] == ["leaf"]
    assert body["invalid_specs"] == []

    res = subprocess.run(
        [sys.executable, "-m", "tcip_web.cli", "doctor", str(tmp_path)],
        capture_output=True, text=True)
    assert "trait spec" not in res.stdout, res.stdout
