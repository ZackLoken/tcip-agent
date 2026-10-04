"""A crash in the training worker's own pre-envelope setup (run record read, dataset build from
what the launch resolved) never leaves the run reading ``running`` and never ends the process
without a ``training_run`` audit event: the run is closed ``failed`` before the crash
propagates.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import tcip_store as ts
from tcip_mcp import experiments as exp
from tcip_mcp.audit import audit_log_key
from tcip_mcp.pipelines.data import split_construction as sc
from tcip_mcp.pipelines.training import subprocess_worker as worker
from tests._verified_checkpoint_fixtures import detection_config, opened_run


def _training_run_events(root: Path) -> list[dict]:
    events = ts.read_log(audit_log_key(root)).records
    return [e for e in events if e.get("tool") == "training_run"]


def test_a_pre_envelope_crash_marks_the_run_failed_and_opens_a_training_run_event(
        tmp_path, monkeypatch):
    run_dir = opened_run(tmp_path, detection_config(tmp_path / "data", batch_size=1),
                         experiment_id="exp-worker-crash")

    def _boom(*args, **kwargs):
        raise RuntimeError("dataset build exploded")

    # recorded_datasets is imported inside prepare_run_context at call time, so patching the
    # source module's own name is what the worker's own lazy import resolves.
    monkeypatch.setattr(sc, "recorded_datasets", _boom)
    with pytest.raises(RuntimeError, match="dataset build exploded"):
        worker.run_directory(run_dir)

    observation = exp.observe(run_dir)
    final = observation.final
    assert final["state"] == "failed"
    assert final["error"] == "dataset build exploded"
    assert final["checkpoint"] is None
    assert observation.state == "failed"

    events = _training_run_events(tmp_path)
    assert len(events) == 1
    assert events[0]["status"] == "failed"
    assert events[0]["arguments"]["experiment_id"] == "exp-worker-crash"


def test_a_pre_envelope_crash_whose_audit_line_is_refused_names_both_causes(
        tmp_path, monkeypatch):
    """Setup fails and the run's audit line is refused: the setup error propagates, and the
    final status names it and the unwritten line."""
    import tcip_mcp.audit as audit

    run_dir = opened_run(tmp_path, detection_config(tmp_path / "data", batch_size=1),
                         experiment_id="exp-worker-crash-unaudited")

    def _boom(*args, **kwargs):
        raise RuntimeError("dataset build exploded")

    def _refuse(key, entry):
        raise OSError("the log's disk is full")

    monkeypatch.setattr(sc, "recorded_datasets", _boom)
    monkeypatch.setattr(audit, "append", _refuse)
    with pytest.raises(RuntimeError, match="dataset build exploded"):
        worker.run_directory(run_dir)

    final = exp.observe(run_dir).final
    assert final["state"] == "failed"
    assert "dataset build exploded" in final["error"]
    assert "audit entry could not be written" in final["error"]
