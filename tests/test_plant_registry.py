"""Coverage for ``register_plant_registry``: a named, project-scoped record of a
plant-locations CSV set, read back by ``build_plant_mapping`` and
``deliver_orthomosaic_plant_counts`` in place of a re-asserted path list on every call.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

import tcip_store as ts
from tcip_mcp.pipelines.postprocessing import plant_mapping
from tcip_mcp.tools.phenology_tools import build_plant_mapping, register_plant_registry

from tests.test_plant_mapping_binding import PLANTS, _dataset, _init, _write_scene


def _plant_csv(path: Path, plants: list[dict] | None = None) -> Path:
    plants = PLANTS if plants is None else plants
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["plot_name", "accession_name", "WGS84_centroid_x", "WGS84_centroid_y"])
        for p in plants:
            w.writerow([p["plot"], p["accession"], p["lon"], p["lat"]])
    return path


def test_registers_a_csv_and_persists_the_frozen_record(tmp_path: Path) -> None:
    """The recorded shape: per-file {path, sha256, n_plants}, crop, site, registered_by,
    registered_at, and a digest over the parsed rows."""
    csv_path = _plant_csv(tmp_path / "plants.csv")

    res = register_plant_registry(
        name="valley-plants", csv_paths=[str(csv_path)], crop="currant", site="north orchard")

    assert "error" not in res, res
    assert res["name"] == "valley-plants"
    assert res["n_plants"] == len(PLANTS)
    assert res["registered_by"] == "register_plant_registry"
    assert len(res["digest"]) == 64
    assert res["csvs"] == [{
        "path": str(csv_path),
        "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "n_plants": len(PLANTS),
    }]

    stored = plant_mapping.load_registry(tmp_path, "valley-plants")
    assert stored is not None
    assert stored["digest"] == res["digest"]
    assert stored["csvs"] == res["csvs"]


def test_refuses_a_name_outside_name_segment(tmp_path: Path) -> None:
    csv_path = _plant_csv(tmp_path / "plants.csv")

    res = register_plant_registry(
        name="Not Legal!", csv_paths=[str(csv_path)], crop="currant", site="orchard")

    assert "error" in res
    assert "lowercase" in res["error"]


def test_refuses_a_missing_file(tmp_path: Path) -> None:
    res = register_plant_registry(
        name="valley", csv_paths=[str(tmp_path / "missing.csv")], crop="currant", site="orchard")

    assert "error" in res
    assert "plant CSV" in res["error"]


def test_refuses_naming_a_file_that_parses_no_georeferenced_plant(tmp_path: Path) -> None:
    """A per-file check: one good file beside one that parses to nothing still refuses, naming
    only the failing file."""
    good = _plant_csv(tmp_path / "good.csv")
    bad = tmp_path / "bad.csv"
    bad.write_text("plot_name,accession_name,WGS84_centroid_x,WGS84_centroid_y\n", encoding="utf-8")

    res = register_plant_registry(
        name="valley", csv_paths=[str(good), str(bad)], crop="currant", site="orchard")

    assert "error" in res
    assert bad.name in res["error"]
    assert good.name not in res["error"]
    assert not ts.exists(plant_mapping.plant_registry_key(tmp_path, "valley"))


def test_a_second_registration_under_the_same_name_and_content_is_a_no_op(tmp_path: Path) -> None:
    csv_path = _plant_csv(tmp_path / "plants.csv")

    first = register_plant_registry(
        name="valley", csv_paths=[str(csv_path)], crop="currant", site="orchard")
    assert "error" not in first, first

    second = register_plant_registry(
        name="valley", csv_paths=[str(csv_path)], crop="currant", site="orchard")

    assert "error" not in second, second
    assert second["digest"] == first["digest"]


def test_a_second_registration_under_the_same_name_and_different_plants_refuses(
    tmp_path: Path,
) -> None:
    csv_path = _plant_csv(tmp_path / "plants.csv")
    other_csv = _plant_csv(
        tmp_path / "other.csv", [{"plot": "P9", "accession": "acc-Z", "lat": 1.0, "lon": 1.0}])

    first = register_plant_registry(
        name="valley", csv_paths=[str(csv_path)], crop="currant", site="orchard")
    assert "error" not in first, first

    conflict = register_plant_registry(
        name="valley", csv_paths=[str(other_csv)], crop="currant", site="orchard")

    assert "error" in conflict
    assert first["digest"] in conflict["error"]
    stored = plant_mapping.load_registry(tmp_path, "valley")
    assert stored["digest"] == first["digest"], "the taken name's record must not have moved"


def test_build_plant_mapping_reads_a_registered_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admits valid work: a mapping built by build_plant_mapping over a name
    register_plant_registry minted, through both platform doors."""
    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, _ = _write_scene(dataset_root, dates=["2026-02-11"])

    reg = register_plant_registry(
        name="valley-plants", csv_paths=[str(plant_csv)], crop="currant", site="orchard")
    assert "error" not in reg, reg

    res = build_plant_mapping(
        name="valley", images_root=str(images_root), plant_registry="valley-plants")

    assert "error" not in res, res
    build = plant_mapping.load_mapping(tmp_path, "valley")
    assert build is not None
    assert build.plant_registry == {"name": "valley-plants", "digest": reg["digest"]}


def test_build_plant_mapping_refuses_naming_register_plant_registry_when_registry_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    images_root, _, _ = _write_scene(dataset_root, dates=["2026-02-11"])

    res = build_plant_mapping(
        name="valley", images_root=str(images_root), plant_registry="nope")

    assert "error" in res
    assert "register_plant_registry" in res["error"]


def _deliver_scene(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, dict[str, str]]:
    """A real, delivered phenology scene through the platform's own producers, for the registry
    doors below to then interfere with. Returns the registered registry's own name and
    predictions_by_date."""
    from tests.test_second_trait_acceptance import _seed_currant_bloom_trait

    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root, dates=["2026-02-11"])
    reg = register_plant_registry(
        name="valley-plants", csv_paths=[str(plant_csv)], crop="currant", site="orchard")
    assert "error" not in reg, reg
    build_res = build_plant_mapping(
        name="valley", images_root=str(images_root), plant_registry="valley-plants")
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)
    return "valley-plants", preds_by_date


def test_the_happy_path_through_the_platforms_own_producers_refuses_at_the_classifier_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admits valid work: a registered registry that still loads and still hashes to what the
    mapping recorded is admitted by the delivery door, which resolves the mapping, runs the
    per-plant phenology over it and reaches the measurement gate. The scene's classifier is
    unvalidated and the MCP door takes no acknowledgement, so the gate is where this delivery
    stops, naming the classifier rather than the registry or the mapping."""
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones

    registry_name, preds_by_date = _deliver_scene(tmp_path, monkeypatch)
    out_csv = tmp_path / "out.csv"

    res = deliver_phenology_milestones(
        trait="currant_bloom", mapping_name="valley", predictions_by_date=preds_by_date,
        output_csv_path=str(out_csv))

    assert "error" in res, res
    assert "validated positive-state classifier" in res["error"]
    assert registry_name not in res["error"]
    assert res["n_plants"] > 0


def test_a_deleted_registry_refuses_at_delivery_naming_the_registry_and_the_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GUARDS: a registry deleted after the mapping was built refuses the delivery by name,
    rather than verifying an empty plant_csvs_unverified list against nothing."""
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones

    registry_name, preds_by_date = _deliver_scene(tmp_path, monkeypatch)
    key = plant_mapping.plant_registry_key(tmp_path, registry_name)
    ts.delete(key, expect=ts.read_versioned(key).version)
    out_csv = tmp_path / "out.csv"

    res = deliver_phenology_milestones(
        trait="currant_bloom", mapping_name="valley", predictions_by_date=preds_by_date,
        output_csv_path=str(out_csv))

    assert "error" in res
    assert registry_name in res["error"]
    assert "valley" in res["error"]
    assert not out_csv.exists()


def test_a_registry_digest_mismatch_refuses_at_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GUARDS: a registry whose stored digest no longer matches what the mapping was built
    against (a hand-edited or store-corrupted record) refuses rather than verifying against the
    wrong plants."""
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones

    registry_name, preds_by_date = _deliver_scene(tmp_path, monkeypatch)
    key = plant_mapping.plant_registry_key(tmp_path, registry_name)
    versioned = ts.read_versioned(key)
    ts.replace(key, {**versioned.value, "digest": "0" * 64}, expect=versioned.version)
    out_csv = tmp_path / "out.csv"

    res = deliver_phenology_milestones(
        trait="currant_bloom", mapping_name="valley", predictions_by_date=preds_by_date,
        output_csv_path=str(out_csv))

    assert "error" in res
    assert registry_name in res["error"]
    assert not out_csv.exists()


def _call_arg_blocks(text: str, name: str) -> list[str]:
    """Every argument block of a call to ``name(`` in ``text``, skipping ``name``'s own ``def``,
    tracking paren depth so a nested call inside the arguments (``str(images_root)``, say) never
    closes the block early the way a single non-nested regex would."""
    blocks = []
    marker = name + "("
    search_from = 0
    while True:
        idx = text.find(marker, search_from)
        if idx == -1:
            break
        search_from = idx + len(marker)
        if text[max(0, idx - 4):idx] == "def ":
            continue
        depth = 1
        pos = search_from
        while pos < len(text) and depth > 0:
            if text[pos] == "(":
                depth += 1
            elif text[pos] == ")":
                depth -= 1
            pos += 1
        blocks.append(text[search_from:pos - 1])
    return blocks


def test_no_build_plant_mapping_call_site_names_the_retired_plant_csv_paths_argument() -> None:
    """Collection-time guard against smoke_phenology_e2e.py's own regression: every
    build_plant_mapping( call site under scripts/ and packages/ passes plant_registry and never
    the retired plant_csv_paths keyword, so a fresh script cannot silently reintroduce it."""
    import subprocess

    repo_root = Path(__file__).resolve().parents[1]
    tracked = subprocess.run(
        ["git", "ls-files", "scripts", "packages"], cwd=repo_root, capture_output=True,
        text=True, check=True,
    ).stdout.splitlines()

    call_sites: list[tuple[str, str]] = []
    for rel in tracked:
        if not rel.endswith(".py"):
            continue
        text = (repo_root / rel).read_text(encoding="utf-8")
        for block in _call_arg_blocks(text, "build_plant_mapping"):
            call_sites.append((rel, block))

    stale = [rel for rel, block in call_sites if "plant_csv_paths" in block]
    assert not stale, f"build_plant_mapping call(s) still pass plant_csv_paths: {stale}"
    missing = [
        rel for rel, block in call_sites
        if "plant_registry" not in block and "**" not in block
    ]
    assert not missing, f"build_plant_mapping call(s) do not pass plant_registry: {missing}"


