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


def _rows(project: Path) -> list[dict]:
    return list(ts.read_log(audit_module.audit_log_key(project)).records)


def test_a_successful_monitor_result_with_a_null_error_audits_nothing(tmp_path: Path) -> None:
    """The stream's own read: a run whose status carries ``error: None`` polled through
    ``monitor_training`` leaves the log exactly as it found it."""
    from tcip_mcp.tools.training_tools import monitor_training
    from tests._verified_checkpoint_fixtures import detection_config, opened_run

    opened_run(tmp_path, detection_config(tmp_path / "ds"), experiment_id="exp-polled")
    before = len(_rows(tmp_path))

    status = monitor_training(tmp_path, "exp-polled")["run"]

    assert status["state"] == "running"
    assert status["error"] is None
    assert len(_rows(tmp_path)) == before


def test_read_only_doors_leave_no_line(tmp_path: Path) -> None:
    from tcip_mcp.tools.experiment_tools import get_experiment, list_experiments
    from tcip_mcp.tools.knowledge_tools import serve_domain_knowledge
    from tcip_mcp.tools.meta_tools import read_audit_log
    from tcip_mcp.tools.project_tools import view_gui_state
    from tcip_mcp.tools.training_tools import inspect_compute_resources, monitor_training
    from tests._verified_checkpoint_fixtures import detection_config, opened_run

    opened_run(tmp_path, detection_config(tmp_path / "ds"), experiment_id="exp-read")
    before = len(_rows(tmp_path))

    get_experiment(tmp_path, "exp-read")
    list_experiments(tmp_path)
    monitor_training(tmp_path, "exp-read")
    monitor_training(tmp_path, "no-such-run")
    inspect_compute_resources(tmp_path)
    view_gui_state(tmp_path)
    serve_domain_knowledge()
    serve_domain_knowledge("no such document")
    read_audit_log(tmp_path)

    assert len(_rows(tmp_path)) == before


def test_a_phenology_look_on_screen_leaves_no_line(tmp_path: Path, client) -> None:
    """The Results tab's measurement door computes and shows; it ships nothing and writes nothing,
    so the project's log reads the same before and after a successful look."""
    pytest.importorskip("torch")
    from tests._chain_fixtures import attributed_series

    body = attributed_series(tmp_path).body()
    before = _rows(tmp_path)

    resp = client.post("/api/results/phenology_measurement", json=body)

    assert resp.status_code == 200, resp.text
    assert resp.json()["milestones"]["rows"]
    assert _rows(tmp_path) == before


def test_ranking_a_review_queue_leaves_no_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ranking candidates for review reads a checkpoint and a manifest and writes nothing, so a
    successful ranking through a registered run leaves the project's log as it found it."""
    from tcip_mcp.tools.feedback_tools import prioritize_review_queue

    from tests.test_feedback_tools import _bound_checkpoint, _stub_scorer
    from tests.test_selection_ground_truth_digests import DATES, _dataset, _draw

    root = _dataset(tmp_path / "data")
    manifest_dir = tmp_path / "manifest"
    _draw(tmp_path, root, manifest_dir)
    _run_dir, ckpt_path = _bound_checkpoint(tmp_path, manifest_dir, "exp-ranked")
    _stub_scorer(monkeypatch)
    before = _rows(tmp_path)

    result = prioritize_review_queue(
        tmp_path, checkpoint_path=ckpt_path, images_dir=str(root / "images" / DATES[0]))

    assert "error" not in result, result
    assert result["queue"]
    assert _rows(tmp_path) == before


def test_a_mutating_door_still_leaves_one_line(tmp_path: Path) -> None:
    """The rail admits the work it exists for: a mutation through the platform's own door leaves
    exactly one line, status ok, after the record it wrote landed."""
    from tcip_mcp.tools.meta_tools import report_friction

    before = len(_rows(tmp_path))
    result = report_friction(tmp_path, "unexpected_behavior", "a real mutation")

    assert "error" not in result
    rows = _rows(tmp_path)
    assert len(rows) == before + 1
    assert rows[-1]["tool"] == "report_friction"
    assert rows[-1]["status"] == "ok"


def test_a_return_is_ok_a_raise_is_an_exception_and_a_refusal_is_no_line(tmp_path: Path) -> None:
    """A refusal returned as the error dict every tool returns is no act and leaves no line; a
    dict whose ``error`` is null is an ordinary answer."""
    from tcip_mcp.audit import audited

    @audited
    def refuse(project: Path) -> dict:
        return {"error": "refused by name"}

    @audited
    def answer(project: Path) -> dict:
        return {"error": None, "status": "running"}

    @audited
    def explode(project: Path) -> dict:
        raise RuntimeError("boom")

    refuse(tmp_path)
    answer(tmp_path)
    with pytest.raises(RuntimeError):
        explode(tmp_path)

    statuses = [(r["tool"], r["status"]) for r in _rows(tmp_path)]
    assert statuses == [("answer", "ok"), ("explode", "exception")]
