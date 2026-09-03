"""``scripts/conform_delivery_events.py``: the one-off check for a project's stored
``delivery_events`` records against the current ``DeliveryEventRecord`` shape. Unlike
``conform_view_coverage_viewing.py`` this script never rewrites a record: none of the three
``plant_mapping`` disclosure keys a refused record lacks has a value derivable from the rest of
that record.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import tcip_store as ts
from tcip_store.binding import bind_default

from tcip_mcp.pipelines import resolution
from tcip_mcp.pipelines.resolution import record_delivery_binding_event

SCRIPT = Path(__file__).parent.parent / "scripts" / "conform_delivery_events.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("conform_delivery_events_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _events(root: Path) -> dict[str, dict]:
    scope = resolution.delivery_events_scope(root)
    keys = ts.keys(resolution.DELIVERY_EVENTS_STORE, str(scope))
    return {key.parts[0]: ts.read(key) for key in keys}


def _write_valid_event(root: Path) -> None:
    record_delivery_binding_event(
        "test_door", None, [], {}, measurement_documents=["operating_point"],
        scale_document=None, trait="astringency", delivery_kind="state_crossing_dates",
        project_root=root, plant_mapping=None,
    )


def _write_old_shaped_event(root: Path, event_id: str = "old-shaped") -> None:
    """A ``plant_mapping`` written before ``dates_delivered``, ``images_unattributed`` and
    ``plant_attribution`` existed: exactly the gap this script names rather than fills."""
    key = resolution.delivery_event_key(resolution.delivery_events_scope(root), event_id)
    ts.replace(
        key,
        {
            "event_id": event_id,
            "trait": "astringency",
            "delivery_kind": "state_crossing_dates",
            "door": "compute_phenology",
            "output_path": None,
            "measurement_documents": ["operating_point"],
            "scale_document": None,
            "plant_mapping": {
                "name": "valley",
                "project_root": str(root),
                "dataset_id": "ds-1",
                "dataset_root": "C:/data",
                "built_at": datetime.now(timezone.utc).isoformat(),
                "record_sha256": "0" * 64,
                "nn_tolerance_m": {"value": 3, "source": "stated"},
                "capture_identity": {},
                "captures_unverified": [],
                "plant_csvs_unverified": [],
                "images_unattributed_scope": "delivered_dates",
            },
            "documents": {},
            "produced_at": datetime.now(timezone.utc).isoformat(),
        },
        expect=ts.Version.ABSENT,
    )


def test_a_root_with_nothing_stored_reports_no_outcomes_and_is_not_refused(tmp_path: Path):
    bind_default()
    module = _load_script()

    outcomes, refused = module.check_root(tmp_path)

    assert outcomes == []
    assert refused is False


def test_check_root_names_a_valid_record_and_refuses_an_old_shaped_one(tmp_path: Path):
    bind_default()
    module = _load_script()
    _write_valid_event(tmp_path)
    _write_old_shaped_event(tmp_path)

    outcomes, refused = module.check_root(tmp_path)

    assert refused is True
    assert len(outcomes) == 2
    assert any(o.endswith(": validates, unchanged") for o in outcomes)
    refusal = next(o for o in outcomes if o.startswith("old-shaped: refused"))
    assert "dates_delivered" in refusal
    assert "re-deliver, or remove the record by hand" in refusal


def test_main_over_an_empty_root_exits_zero_and_reports_nothing_stored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    bind_default()
    module = _load_script()

    monkeypatch.setattr(sys, "argv", ["conform_delivery_events.py", str(tmp_path)])
    exit_code = module.main()
    output = capsys.readouterr().out

    assert exit_code == 0
    assert f"{tmp_path.resolve()}: nothing stored" in output


def test_main_plan_mode_over_a_mixed_root_prints_both_lines_exits_two_and_rewrites_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    bind_default()
    module = _load_script()
    _write_valid_event(tmp_path)
    _write_old_shaped_event(tmp_path)
    before = _events(tmp_path)

    monkeypatch.setattr(sys, "argv", ["conform_delivery_events.py", "--plan", str(tmp_path)])
    exit_code = module.main()
    output = capsys.readouterr().out

    assert exit_code == 2
    assert "validates, unchanged" in output
    assert "old-shaped: refused" in output
    assert _events(tmp_path) == before
