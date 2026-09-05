"""``scripts/conform_plant_mapping_records.py``: rewrites a stored plant-mapping record's old
``plant_csvs`` field to the new ``plant_registry`` reference and adds ``supersedes: null``.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

import tcip_store as ts
from tcip_store.binding import bind_default

from tcip_mcp.pipelines.postprocessing import plant_mapping

SCRIPT = Path(__file__).parent.parent / "scripts" / "conform_plant_mapping_records.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("conform_plant_mapping_records_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_old_shaped_record(project_root: Path, name: str, csv_path: Path) -> dict:
    """The pre-change writer's own record shape: ``plant_csvs`` (not ``plant_registry``), no
    ``supersedes`` field at all, exactly the ten fields ``_REQUIRED_TOP_KEYS`` named before this
    family. Built by hand from that old field list, never from the current dataclass, so this
    fixture cannot drift forward silently when the class changes again.
    """
    record = {
        "name": name, "project_root": str(project_root), "dataset_root": str(project_root / "ds"),
        "dataset_id": "ds-1", "built_by": "build_plant_mapping",
        "built_at": "2026-02-11T00:00:00+00:00", "dates_requested": None, "dates": ["2026-02-11"],
        "nn_tolerance_m": {"value": 10.0, "source": "fallback"},
        "plant_csvs": [{"path": str(csv_path), "sha256": "0" * 64, "n_plants": 1}],
        "capture_identity": {"2026-02-11": "0" * 16},
        "capture_digests": {"2026-02-11": {}}, "unreadable": {"2026-02-11": []},
        "assignments": {"2026-02-11": []},
    }
    key = plant_mapping.plant_mapping_key(project_root, name)
    ts.replace(key, record)
    return record


def _write_plant_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["plot_name", "accession_name", "WGS84_centroid_x", "WGS84_centroid_y"])
        w.writerow(["P1", "acc-1", "-90.05", "43.19"])


@pytest.fixture(autouse=True)
def _bind(tmp_path: Path) -> None:
    bind_default()
    (tmp_path / ".tcip").mkdir(parents=True, exist_ok=True)


def test_plan_previews_without_writing(tmp_path: Path) -> None:
    module = _load_script()
    csv_path = tmp_path / "plants.csv"
    _write_plant_csv(csv_path)
    before = _write_old_shaped_record(tmp_path, "valley", csv_path)

    outcomes, writes = module.plan_root(tmp_path, "valley-plants", crop="currant", site="north")

    assert any("conformed" in o for o in outcomes)
    assert writes
    stored = ts.read(plant_mapping.plant_mapping_key(tmp_path, "valley"))
    assert stored == before, "plan_root computes writes but must not apply them"
    assert plant_mapping.load_registry(tmp_path, "valley-plants") is None, (
        "plan_root must not register the CSVs either")


def test_main_conforms_the_record_and_registers_the_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    csv_path = tmp_path / "plants.csv"
    _write_plant_csv(csv_path)
    _write_old_shaped_record(tmp_path, "valley", csv_path)

    monkeypatch.setattr(sys, "argv", [
        "conform_plant_mapping_records.py", str(tmp_path), "valley-plants",
        "--crop", "currant", "--site", "north orchard",
    ])
    exit_code = module.main()
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "conformed" in output

    stored = ts.read(plant_mapping.plant_mapping_key(tmp_path, "valley"))
    assert "plant_csvs" not in stored
    assert stored["plant_registry"]["name"] == "valley-plants"
    assert stored["supersedes"] is None

    registry = plant_mapping.load_registry(tmp_path, "valley-plants")
    assert registry is not None
    assert registry["digest"] == stored["plant_registry"]["digest"]
    assert registry["crop"] == "currant"
    assert registry["site"] == "north orchard"

    build = plant_mapping.load_mapping(tmp_path, "valley")
    assert build is not None
    assert build.plant_registry == stored["plant_registry"]


def test_a_record_already_conformed_is_reported_unchanged(tmp_path: Path) -> None:
    module = _load_script()
    key = plant_mapping.plant_mapping_key(tmp_path, "valley")
    ts.replace(key, {
        "name": "valley", "project_root": str(tmp_path), "dataset_root": str(tmp_path / "ds"),
        "dataset_id": "ds-1", "built_by": "build_plant_mapping",
        "built_at": "2026-02-11T00:00:00+00:00", "dates_requested": None, "dates": [],
        "nn_tolerance_m": {"value": 10.0, "source": "fallback"},
        "plant_registry": {"name": "already-there", "digest": "0" * 64},
        "capture_identity": {}, "capture_digests": {}, "unreadable": {}, "assignments": {},
        "supersedes": None,
    })

    outcomes, writes = module.plan_root(tmp_path, "valley-plants", crop="currant", site="north")

    assert writes == []
    assert any("already conformed" in o for o in outcomes)


def test_a_stored_path_no_longer_on_disk_refuses(tmp_path: Path) -> None:
    module = _load_script()
    missing_csv = tmp_path / "gone.csv"
    _write_old_shaped_record(tmp_path, "valley", missing_csv)

    outcomes, writes = module.plan_root(tmp_path, "valley-plants", crop="currant", site="north")

    assert writes == []
    assert any("refused" in o for o in outcomes)


def test_a_registry_name_outside_name_segment_refuses(tmp_path: Path) -> None:
    module = _load_script()
    csv_path = tmp_path / "plants.csv"
    _write_plant_csv(csv_path)
    _write_old_shaped_record(tmp_path, "valley", csv_path)

    outcomes, writes = module.plan_root(tmp_path, "Not Legal!", crop="currant", site="north")

    assert writes == []
    assert any("refused" in o and "lowercase" in o for o in outcomes)
    assert plant_mapping.load_registry(tmp_path, "Not Legal!") is None


def test_main_prints_delivery_events_a_conform_would_strand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Rewriting plant_csvs to plant_registry moves the record's own digest, so a delivery event
    minted against the old digest now cites a record nothing hashes to; the conform names it."""
    from tcip_mcp.pipelines import resolution

    module = _load_script()
    csv_path = tmp_path / "plants.csv"
    _write_plant_csv(csv_path)
    old_record = _write_old_shaped_record(tmp_path, "valley", csv_path)
    old_digest = plant_mapping.record_digest(old_record)

    event_key = resolution.delivery_event_key(resolution.delivery_events_scope(tmp_path), "evt-1")
    ts.replace(event_key, {
        "event_id": "evt-1", "trait": "currant_bloom", "delivery_kind": "state_crossing_dates",
        "door": "deliver_phenology_milestones", "output_path": None, "output_sha256": None,
        "measurement_documents": ["operating_point"], "scale_document": None,
        "acknowledged_by": None, "acknowledgement_reason": None,
        "plant_mapping": {
            "name": "valley", "project_root": str(tmp_path), "dataset_id": "ds-1",
            "dataset_root": str(tmp_path / "ds"), "built_at": "2026-02-11T00:00:00+00:00",
            "record_sha256": old_digest, "nn_tolerance_m": {"value": 10.0, "source": "fallback"},
            "capture_identity": {}, "captures_unverified": [], "plant_csvs_unverified": [],
            "dates_delivered": ["2026-02-11"], "images_unattributed": 0,
            "images_unattributed_scope": "delivered_dates", "plant_attribution": "image",
        },
        "documents": {}, "produced_at": "2026-02-11T00:00:00+00:00",
    })

    monkeypatch.setattr(sys, "argv", [
        "conform_plant_mapping_records.py", str(tmp_path), "valley-plants",
        "--crop", "currant", "--site", "north orchard", "--plan",
    ])
    exit_code = module.main()
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "evt-1" in output
    assert old_digest in output
