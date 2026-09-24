"""The audit log records mutations and nothing else.

A read-only door (a status poll a browser drives once a second, a listing, a knowledge document
served back) leaves no line; a mutating door still leaves exactly one. The decorator infers
nothing from a body's return value: a body that returned reads ``ok`` whatever dict it
answered, and only a body that raised reads ``exception``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tcip_mcp.audit as audit_module
import tcip_store as ts


@pytest.fixture
def platform_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "platform"
    root.mkdir()
    monkeypatch.setattr(audit_module, "AUDIT_ROOT", root)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(root))
    return root


def _rows() -> list[dict]:
    return list(ts.read_log(audit_module.audit_log_key()).records)


def test_a_successful_monitor_result_with_a_null_error_audits_nothing(platform_root: Path) -> None:
    """The stream's own read: a run whose status carries ``error: None`` polled through
    ``monitor_training`` leaves the log exactly as it found it."""
    from tcip_mcp.experiments import create_experiment, update_status
    from tcip_mcp.tools.training_tools import monitor_training

    create_experiment("exp-polled", {"model_source": {"builder": "m:f"}})
    update_status("exp-polled", "running")
    before = len(_rows())

    status = monitor_training("exp-polled")

    assert status["status"] == "running"
    assert status["error"] is None
    assert len(_rows()) == before


def test_read_only_doors_leave_no_line(platform_root: Path) -> None:
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.tools.experiment_tools import get_experiment, list_experiments
    from tcip_mcp.tools.knowledge_tools import serve_domain_knowledge
    from tcip_mcp.tools.meta_tools import read_audit_log
    from tcip_mcp.tools.project_tools import view_gui_state
    from tcip_mcp.tools.training_tools import inspect_compute_resources, monitor_training

    create_experiment("exp-read", {"model_source": {"builder": "m:f"}})
    before = len(_rows())

    get_experiment("exp-read")
    list_experiments()
    list_experiments(launched_only=True)
    monitor_training("exp-read")
    monitor_training("no-such-run")
    inspect_compute_resources()
    view_gui_state()
    serve_domain_knowledge()
    serve_domain_knowledge("no such document")
    read_audit_log()

    assert len(_rows()) == before


def test_a_phenology_look_on_screen_leaves_no_line(tmp_path: Path) -> None:
    """The Results tab's measurement door computes and shows; it ships nothing and writes nothing,
    so the project's log reads the same before and after a successful look."""
    from fastapi.testclient import TestClient

    from tcip_web.app import app

    from tests.test_tcip_web_results_routes import _phenology_fixture

    client = TestClient(app, base_url="http://127.0.0.1")
    body = _phenology_fixture(tmp_path, validated=True, detections=100)
    before = list(ts.read_log(audit_module.audit_log_key(tmp_path)).records)

    resp = client.post("/api/results/phenology_measurement", json=body)

    assert resp.status_code == 200, resp.text
    assert resp.json()["milestones"]["rows"]
    assert list(ts.read_log(audit_module.audit_log_key(tmp_path)).records) == before


def test_ranking_a_review_queue_leaves_no_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ranking candidates for review reads a checkpoint and a manifest and writes nothing, so a
    successful ranking through a registered run leaves the platform log as it found it."""
    from tcip_mcp.tools.feedback_tools import prioritize_review_queue

    from tests.test_feedback_tools import _registered_checkpoint_from_experiment, _stub_scorer
    from tests.test_selection_disjointness_label_movement import DATES, _bind_run, _dataset, _draw

    root = _dataset(tmp_path / "data")
    manifest_dir = tmp_path / "manifest"
    _draw(root, manifest_dir)
    _bind_run(root, manifest_dir, "exp-ranked", date=DATES[0])
    ckpt_path = _registered_checkpoint_from_experiment(tmp_path, "exp-ranked")
    _stub_scorer(monkeypatch)
    before = _rows()

    result = prioritize_review_queue(
        checkpoint_path=ckpt_path, images_dir=str(root / "images" / DATES[0]),
        project_path=str(tmp_path))

    assert "error" not in result, result
    assert result["queue"]
    assert _rows() == before


def test_a_mutating_door_still_leaves_one_line(platform_root: Path) -> None:
    """The rail admits the work it exists for: a mutation through the platform's own door leaves
    exactly one line, status ok, after the record it wrote landed."""
    from tcip_mcp.tools.meta_tools import report_friction

    before = len(_rows())
    result = report_friction(str(platform_root), "unexpected_behavior", "a real mutation")

    assert "error" not in result
    rows = _rows()
    assert len(rows) == before + 1
    assert rows[-1]["tool"] == "report_friction"
    assert rows[-1]["status"] == "ok"


def test_a_return_is_ok_a_raise_is_an_exception_and_a_refusal_is_no_line(platform_root: Path) -> None:
    """A refusal returned as the error dict every tool returns is no act and leaves no line; a
    dict whose ``error`` is null is an ordinary answer."""
    from tcip_mcp.audit import audited

    @audited
    def refuse() -> dict:
        return {"error": "refused by name"}

    @audited
    def answer() -> dict:
        return {"error": None, "status": "running"}

    @audited
    def explode() -> dict:
        raise RuntimeError("boom")

    refuse()
    answer()
    with pytest.raises(RuntimeError):
        explode()

    statuses = [(r["tool"], r["status"]) for r in _rows()]
    assert statuses == [("answer", "ok"), ("explode", "exception")]
