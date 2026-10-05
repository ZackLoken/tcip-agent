"""Coverage for the same-name ``build_plant_mapping`` rebuild that supersedes rather than
silently replaces a mapping a delivery event still cites.
"""

from __future__ import annotations

from pathlib import Path

import tcip_store as ts

from tcip_mcp.delivery import read_delivery_events
from tcip_mcp.pipelines.delivery_events_schema import PlantMappingDisclosure
from tcip_mcp.pipelines.postprocessing import plant_mapping
from tcip_mcp.tools.phenology_tools import build_plant_mapping

from tests._mapping_fixtures import register_plant_registry_for
from tests.test_plant_mapping_binding import DATES, PLANTS, _dataset, _init, _write_scene
from tests.test_plant_mapping_binding import _deliver as _deliver_through
from tests.test_second_trait_acceptance import _seed_currant_bloom_trait


def _deliver(project: Path, preds_by_date: dict[str, str], out_csv: Path) -> dict:
    """The delivery over the buckets of :func:`_dataset`'s default root."""
    return _deliver_through(
        project, trait="currant_bloom", mapping_name="valley",
        plants=[p["plot"] for p in PLANTS], dataset_root=project / "ds",
        buckets=preds_by_date.values(), output_csv_path=str(out_csv))


def _cited_mapping(tmp_path: Path) -> tuple[str, dict[str, str]]:
    """A mapping built, delivered from (so a delivery event cites its digest), and the plant CSV
    it was built over, for a rebuild under the same name to then be tried against."""
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root, dates=[DATES[0]])
    registry = register_plant_registry_for(tmp_path, [plant_csv])
    build_res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root), plant_registry=registry)
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    res = _deliver(tmp_path, preds_by_date, tmp_path / "out.csv")
    assert "error" not in res, res
    return str(images_root), preds_by_date


def test_a_cited_rebuild_refuses_naming_the_citing_events(tmp_path: Path) -> None:
    """A guard: a same-name rebuild whose current record a delivery event cites refuses unless
    supersede=True."""
    images_root, _ = _cited_mapping(tmp_path)
    before = plant_mapping.load_mapping(tmp_path, "valley")
    assert before is not None

    citing_ids = [
        r.event_id for r in read_delivery_events(tmp_path)
        if isinstance(r.plant_mapping, PlantMappingDisclosure)
        and r.plant_mapping.record_sha256 == before.record_sha256
    ]
    assert citing_ids

    res = build_plant_mapping(tmp_path, name="valley", images_root=images_root, plant_registry=(
        before.plant_registry["name"]))

    assert "error" in res
    assert "citing_events" in res
    assert set(res["citing_events"]) == set(citing_ids)
    for event_id in citing_ids:
        assert event_id in res["error"]

    after = plant_mapping.load_mapping(tmp_path, "valley")
    assert after is not None
    assert after.record_sha256 == before.record_sha256, "the cited record must be untouched"


def test_a_cited_rebuild_with_supersede_archives_the_old_record_and_keeps_it_readable(
    tmp_path: Path,
) -> None:
    """Admits valid work: supersede=True archives the current record under
    <name>@<digest[:12]>, the new record's own supersedes names the archived digest, the
    archived record stays readable on its own build's receipt, the supersede writing one receipt
    alone, and plant_mapping_names never lists it."""
    from tcip_mcp.audit import audit_log_key

    def receipts() -> list[dict]:
        return [entry["arguments"] for entry in ts.read_log(audit_log_key(tmp_path)).records
                if entry.get("tool") == "plant_mapping_built"]

    images_root, preds_by_date = _cited_mapping(tmp_path)
    before = plant_mapping.load_mapping(tmp_path, "valley")
    assert before is not None
    archived_digest = before.record_sha256
    archived_name = f"valley@{archived_digest[:12]}"
    receipts_before = receipts()

    res = build_plant_mapping(
        tmp_path, name="valley", images_root=images_root,
        plant_registry=before.plant_registry["name"], supersede=True)

    assert "error" not in res, res
    [receipt] = receipts()[len(receipts_before):]
    assert (receipt["name"], receipt["supersedes"]) == ("valley", archived_digest)
    after = plant_mapping.load_mapping(tmp_path, "valley")
    assert after is not None
    assert after.record_sha256 != archived_digest
    assert after.supersedes == archived_digest

    archived = plant_mapping.load_mapping(tmp_path, archived_name)
    assert archived is not None
    assert archived.record_sha256 == archived_digest
    assert archived.assignments == before.assignments

    assert archived_name not in plant_mapping.plant_mapping_names(tmp_path)
    assert "valley" in plant_mapping.plant_mapping_names(tmp_path)

    # The delivery event's own citation still resolves to the archived record's own content.
    res2 = _deliver(tmp_path, preds_by_date, tmp_path / "out2.csv")
    assert "error" not in res2, res2


def test_resolved_mapping_key_for_citation_names_the_archive_once_superseded(
    tmp_path: Path,
) -> None:
    """Coverage: a delivery event's own cited digest resolves to the current name while
    unmoved, and to the archived key once a supersede rebuild has moved the name on."""
    images_root, _ = _cited_mapping(tmp_path)
    before = plant_mapping.load_mapping(tmp_path, "valley")
    assert before is not None

    assert plant_mapping.resolved_mapping_key_for_citation(
        tmp_path, "valley", before.record_sha256) == "valley"

    res = build_plant_mapping(
        tmp_path, name="valley", images_root=images_root,
        plant_registry=before.plant_registry["name"], supersede=True)
    assert "error" not in res, res

    archived_name = f"valley@{before.record_sha256[:12]}"
    assert plant_mapping.resolved_mapping_key_for_citation(
        tmp_path, "valley", before.record_sha256) == archived_name


def test_the_delivery_events_route_resolves_a_superseded_citation_to_the_archive(
    tmp_path: Path, client,
) -> None:
    """The panel route's own plant_mapping_resolved_key names the archive once a supersede
    rebuild has moved the cited name on, and the plain name while it has not."""
    import asyncio

    from tcip_web.state import store

    images_root, _ = _cited_mapping(tmp_path)
    before = plant_mapping.load_mapping(tmp_path, "valley")
    assert before is not None

    asyncio.run(store.open_project(tmp_path))
    resp = client.get("/api/results/delivery-events")
    assert resp.status_code == 200, resp.text
    record = next(r for r in resp.json()["records"] if r.get("plant_mapping"))
    assert record["plant_mapping_resolved_key"] == "valley"

    res = build_plant_mapping(
        tmp_path, name="valley", images_root=images_root,
        plant_registry=before.plant_registry["name"], supersede=True)
    assert "error" not in res, res

    resp2 = client.get("/api/results/delivery-events")
    assert resp2.status_code == 200, resp2.text
    record2 = next(r for r in resp2.json()["records"] if r.get("plant_mapping"))
    assert record2["plant_mapping_resolved_key"] == f"valley@{before.record_sha256[:12]}"


def _built(tmp_path: Path) -> tuple[Path, str]:
    """A mapping ``valley`` built through the platform's own door; its images root and registry."""
    _init(tmp_path)
    images_root, plant_csv, _ = _write_scene(_dataset(tmp_path), dates=[DATES[0]])
    registry = register_plant_registry_for(tmp_path, [plant_csv])
    built = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root), plant_registry=registry)
    assert "error" not in built, built
    return images_root, registry


def test_a_mapping_record_without_assignments_refuses_to_decode_by_name(tmp_path: Path) -> None:
    import pytest

    _built(tmp_path)
    raw = ts.read(plant_mapping.plant_mapping_key(tmp_path, "valley"))
    del raw["assignments"]
    with pytest.raises(ValueError, match="assignments"):
        plant_mapping.MappingBuild.from_record(raw, tmp_path, "valley")


def test_a_rebuild_over_a_record_that_will_not_decode_proceeds_on_its_raw_digest(
    tmp_path: Path, monkeypatch,
) -> None:
    """The record being replaced is identified by its raw digest, whatever its shape: the rebuild
    asks the citation check about that digest and lands."""
    images_root, registry = _built(tmp_path)
    key = plant_mapping.plant_mapping_key(tmp_path, "valley")
    malformed = {"name": "valley", "an old shape": True}
    ts.replace(key, malformed)
    asked: list[str] = []
    real = plant_mapping._citing_delivery_event_ids

    def citing(project, name, digest):
        asked.append(digest)
        return real(project, name, digest)

    monkeypatch.setattr(plant_mapping, "_citing_delivery_event_ids", citing)
    rebuilt = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root), plant_registry=registry)
    assert "error" not in rebuilt, rebuilt
    assert asked == [plant_mapping.record_digest(malformed)]
    assert plant_mapping.load_mapping(tmp_path, "valley") is not None


def test_an_uncited_rebuild_replaces_as_today_recording_nothing_extra(tmp_path: Path) -> None:
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, _ = _write_scene(dataset_root, dates=[DATES[0]])
    registry = register_plant_registry_for(tmp_path, [plant_csv])

    first = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root), plant_registry=registry)
    assert "error" not in first, first
    before = plant_mapping.load_mapping(tmp_path, "valley")
    assert before is not None

    second = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root), plant_registry=registry)
    assert "error" not in second, second

    after = plant_mapping.load_mapping(tmp_path, "valley")
    assert after is not None
    assert after.supersedes is None
    assert plant_mapping.plant_mapping_names(tmp_path) == ["valley"]
