"""A crash in the training worker's own pre-envelope setup (config read, dataset build, split
manifest write) never leaves the experiment record ``running`` and never ends the subprocess
without a ``training_run`` audit event. ``run_training_envelope``, the one place that opens that
event, is not reached from there, so the worker itself reconciles the record to ``failed`` and
opens the event before letting the crash propagate, except when the crash is a terminal-lock
refusal, which the reconciler leaves untouched.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import tcip_store as ts
from tcip_mcp import audit as audit_module
from tcip_mcp import experiments as exp
from tcip_mcp.audit import audit_log_key
from tcip_mcp.pipelines.data import split_construction as sc
from tcip_mcp.pipelines.training import subprocess_worker as worker
from tcip_mcp.tools import training_tools as ttools


def _training_run_events(root: Path) -> list[dict]:
    events = ts.read_log(audit_log_key(root)).records
    return [e for e in events if e.get("tool") == "training_run"]


def _write_launch_config(tmp_path: Path, out: Path) -> dict:
    config = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "task": "detection"},
        "data": {"images_dir": str(tmp_path / "images"), "labels_dir": str(tmp_path / "labels"),
                 "subject": "shoot"},
        "batch_size": 1,
    }
    ts.replace(ttools.launch_config_key(out), config)
    return config


def test_a_pre_envelope_crash_marks_the_run_failed_and_opens_a_training_run_event(tmp_path):
    eid = "exp-worker-crash"
    out = tmp_path / "run"
    out.mkdir()
    config = _write_launch_config(tmp_path, out)
    exp.create_experiment(eid, config)
    exp.update_status(eid, "running")

    def _boom(*args, **kwargs):
        raise RuntimeError("dataset build exploded")

    # auto_train_val is imported inside _prepare_run_context at call time, so patching the
    # source module's own name is what the worker's own lazy import resolves.
    original_ref = sc.auto_train_val
    sc.auto_train_val = _boom
    try:
        with pytest.raises(RuntimeError, match="dataset build exploded"):
            worker.run(eid, str(out), "")
    finally:
        sc.auto_train_val = original_ref

    status = ts.read(exp.status_key(eid, root=tmp_path))
    assert status["state"] == "failed"
    assert status["error"] == "dataset build exploded"

    events = _training_run_events(tmp_path)
    assert len(events) == 1
    assert events[0]["status"] == "failed"
    assert events[0]["arguments"]["experiment_id"] == eid


def test_a_terminal_lock_refusal_reaching_run_leaves_the_record_and_the_log_alone(tmp_path):
    """A pre-envelope crash that is already an ExperimentTerminal (a provenance patch's own
    terminal-lock refusal) is not re-marked failed and leaves no line at all; the record keeps the
    state and reason its own earlier writer (a wall-clock watchdog, say) recorded."""
    eid = "exp-worker-terminal"
    out = tmp_path / "run"
    out.mkdir()
    config = _write_launch_config(tmp_path, out)
    exp.create_experiment(eid, config)
    exp.update_status(eid, "running")
    exp.update_status(eid, "failed", error="exceeded max_wall_clock_seconds (5)")

    def _already_audited_refusal(*args, **kwargs):
        raise exp.ExperimentTerminal(f"Experiment {eid} is failed (terminal); refusing.")

    original_ref = sc.auto_train_val
    sc.auto_train_val = _already_audited_refusal
    try:
        with pytest.raises(exp.ExperimentTerminal):
            worker.run(eid, str(out), "")
    finally:
        sc.auto_train_val = original_ref

    status = ts.read(exp.status_key(eid, root=tmp_path))
    assert status["state"] == "failed"
    assert status["error"] == "exceeded max_wall_clock_seconds (5)"  # untouched

    assert ts.read_log(audit_log_key(tmp_path)).records == []  # the crash-audit branch never ran


def test_a_terminal_records_refusal_still_gets_a_training_run_event(tmp_path, monkeypatch):
    """A pre-envelope crash landing on a record already terminal in a different state
    (``completed``) makes update_status refuse the ``failed`` transition, a refusal that writes
    nothing even with the audit log refused, and the training_run event for the crash itself is
    still written."""
    eid = "exp-worker-refusal-append-fails"
    out = tmp_path / "run"
    out.mkdir()
    config = _write_launch_config(tmp_path, out)
    exp.create_experiment(eid, config)
    exp.update_status(eid, "running")
    exp.update_status(eid, "completed")

    def _boom(*args, **kwargs):
        raise RuntimeError("dataset build exploded again")

    def _broken_record_event_or_raise(tool, arguments=None, **kwargs):
        raise audit_module.AuditEntryNotWritten(tool, RuntimeError("audit log unwritable"))

    monkeypatch.setattr(audit_module, "record_event_or_raise", _broken_record_event_or_raise)

    original_ref = sc.auto_train_val
    sc.auto_train_val = _boom
    try:
        with pytest.raises(RuntimeError, match="dataset build exploded again"):
            worker.run(eid, str(out), "")
    finally:
        sc.auto_train_val = original_ref

    status = ts.read(exp.status_key(eid, root=tmp_path))
    assert status["state"] == "completed"  # the refused failed-transition never wrote

    events = _training_run_events(tmp_path)
    assert len(events) == 1
    assert events[0]["status"] == "failed"
    assert events[0]["arguments"]["experiment_id"] == eid
