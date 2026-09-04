"""Coverage for the same-name ``build_plant_mapping`` rebuild that supersedes rather than
silently replaces a mapping a delivery event still cites (additions-design section 7c).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tcip_store as ts
from tcip_mcp.pipelines import resolution
from tcip_mcp.pipelines.postprocessing import plant_mapping
from tcip_mcp.tools.phenology_tools import build_plant_mapping, deliver_phenology_milestones

from tests._binding_fixtures import register_plant_registry_for
from tests.test_plant_mapping_binding import DATES, _dataset, _init, _write_scene
from tests.test_second_trait_acceptance import _seed_currant_bloom_trait


def _cited_mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, dict[str, str]]:
    """A mapping built, delivered from (so a delivery event cites its digest), and the plant CSV
    it was built over, for a rebuild under the same name to then be tried against."""
    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root, dates=[DATES[0]])
    registry = register_plant_registry_for([plant_csv])
    build_res = build_plant_mapping(
        name="valley", images_root=str(images_root), plant_registry=registry)
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    out_csv = tmp_path / "out.csv"
    res = deliver_phenology_milestones(
        trait="currant_bloom", mapping_name="valley", predictions_by_date=preds_by_date,
        output_csv_path=str(out_csv), acknowledge_unvalidated=True)
    assert "error" not in res, res
    return str(images_root), preds_by_date


def test_a_cited_rebuild_refuses_naming_the_citing_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GUARDS: a same-name rebuild whose current record a delivery event cites refuses unless
    supersede=True."""
    images_root, _ = _cited_mapping(tmp_path, monkeypatch)
    before = plant_mapping.load_mapping(tmp_path, "valley")
    assert before is not None

    scope = resolution.delivery_events_scope(tmp_path)
    citing_ids = [
        r["event_id"] for k in ts.keys(resolution.DELIVERY_EVENTS_STORE, str(scope))
        for r in [ts.read(k)]
        if r.get("plant_mapping") and r["plant_mapping"]["record_sha256"] == before.record_sha256
    ]
    assert citing_ids

    res = build_plant_mapping(name="valley", images_root=images_root, plant_registry=(
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admits valid work: supersede=True archives the current record under
    <name>@<digest[:12]>, the new record's own supersedes names the archived digest, the
    archived record stays readable, and plant_mapping_names never lists it."""
    images_root, preds_by_date = _cited_mapping(tmp_path, monkeypatch)
    before = plant_mapping.load_mapping(tmp_path, "valley")
    assert before is not None
    archived_digest = before.record_sha256
    archived_name = f"valley@{archived_digest[:12]}"

    res = build_plant_mapping(
        name="valley", images_root=images_root, plant_registry=before.plant_registry["name"],
        supersede=True)

    assert "error" not in res, res
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
    out_csv2 = tmp_path / "out2.csv"
    res2 = deliver_phenology_milestones(
        trait="currant_bloom", mapping_name="valley", predictions_by_date=preds_by_date,
        output_csv_path=str(out_csv2), acknowledge_unvalidated=True)
    assert "error" not in res2, res2


def test_an_uncited_rebuild_replaces_as_today_recording_nothing_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, _ = _write_scene(dataset_root, dates=[DATES[0]])
    registry = register_plant_registry_for([plant_csv])

    first = build_plant_mapping(
        name="valley", images_root=str(images_root), plant_registry=registry)
    assert "error" not in first, first
    before = plant_mapping.load_mapping(tmp_path, "valley")
    assert before is not None

    second = build_plant_mapping(
        name="valley", images_root=str(images_root), plant_registry=registry)
    assert "error" not in second, second

    after = plant_mapping.load_mapping(tmp_path, "valley")
    assert after is not None
    assert after.supersedes is None
    assert plant_mapping.plant_mapping_names(tmp_path) == ["valley"]
