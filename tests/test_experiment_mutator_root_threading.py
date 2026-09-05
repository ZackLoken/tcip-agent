"""``experiments.py``'s per-mutator functions (``complete_run``, ``log_metrics``,
``record_artifact``, ``update_lineage``, ``overwrite_config_if_pristine``) took no caller
``root`` at all, unlike ``update_status``, which already lets a launch's own wall-clock watchdog
scope its write to the root it captured at launch rather than wherever this process's platform
root has since moved to. A parent-side writer resolving one of these other mutators against a
record under a root other than the process's own current pin had no way to reach it, or its own
refusal's audit line, scoped correctly. Each now threads ``root`` through to its own key
resolution and to its own ``_audit_refused`` call the same way ``update_status`` already does.
"""

from __future__ import annotations

from pathlib import Path

import tcip_store as ts
from tcip_mcp import experiments as exp
from tcip_mcp.audit import audit_log_key


def _refusals(root: Path) -> list[dict]:
    events = ts.read_log(audit_log_key(root)).records
    return [e for e in events if e.get("tool") == "experiment_mutation_refused"]


def test_complete_run_refusal_audits_the_named_root_not_the_current_one(tmp_path, monkeypatch):
    launch_root = tmp_path / "launch"
    other_root = tmp_path / "other"
    launch_root.mkdir()
    other_root.mkdir()
    eid = "exp-root-complete"

    monkeypatch.setenv("TCIP_STATE_ROOT", str(launch_root))
    exp.create_experiment(eid, {"a": 1})
    exp.update_status(eid, "running")
    exp.update_status(eid, "failed", error="watchdog reason")
    weights = launch_root / "model_best.pt"
    weights.write_bytes(b"weights")

    monkeypatch.setenv("TCIP_STATE_ROOT", str(other_root))
    result = exp.complete_run(eid, str(weights), root=launch_root)

    assert "error" in result
    assert result["state"] == "failed"
    assert _refusals(launch_root)
    assert _refusals(other_root) == []


def test_record_artifact_refusal_audits_the_named_root_not_the_current_one(tmp_path, monkeypatch):
    launch_root = tmp_path / "launch"
    other_root = tmp_path / "other"
    launch_root.mkdir()
    other_root.mkdir()
    eid = "exp-root-artifact"

    monkeypatch.setenv("TCIP_STATE_ROOT", str(launch_root))
    exp.create_experiment(eid, {"a": 1})
    exp.update_status(eid, "running")
    exp.record_artifact(eid, "model_final", "/runs/first.pt", root=launch_root)
    exp.update_status(eid, "failed", error="watchdog reason")

    monkeypatch.setenv("TCIP_STATE_ROOT", str(other_root))
    result = exp.record_artifact(eid, "model_final", "/runs/second.pt", root=launch_root)

    assert "error" in result
    assert _refusals(launch_root)
    assert _refusals(other_root) == []


def test_log_metrics_refusal_audits_the_named_root_not_the_current_one(tmp_path, monkeypatch):
    launch_root = tmp_path / "launch"
    other_root = tmp_path / "other"
    launch_root.mkdir()
    other_root.mkdir()
    eid = "exp-root-metrics"

    monkeypatch.setenv("TCIP_STATE_ROOT", str(launch_root))
    exp.create_experiment(eid, {"a": 1})
    exp.update_status(eid, "running")
    exp.update_status(eid, "failed", error="watchdog reason")

    monkeypatch.setenv("TCIP_STATE_ROOT", str(other_root))
    result = exp.log_metrics(eid, 1, {"loss": 0.5}, root=launch_root)

    assert "error" in result
    assert _refusals(launch_root)
    assert _refusals(other_root) == []


def test_update_lineage_identity_refusal_audits_the_named_root_not_the_current_one(
    tmp_path, monkeypatch,
):
    """dataset_id/dataset_fingerprint are complete_run's alone via update_lineage: refused
    unconditionally, no terminal precondition needed to reach the refusal."""
    launch_root = tmp_path / "launch"
    other_root = tmp_path / "other"
    launch_root.mkdir()
    other_root.mkdir()
    eid = "exp-root-lineage"

    monkeypatch.setenv("TCIP_STATE_ROOT", str(launch_root))
    exp.create_experiment(eid, {"a": 1})

    monkeypatch.setenv("TCIP_STATE_ROOT", str(other_root))
    exp.update_lineage(eid, dataset_id="ds-1", root=launch_root)

    assert _refusals(launch_root)
    assert _refusals(other_root) == []


def test_overwrite_config_if_pristine_refusal_audits_the_named_root_not_the_current_one(
    tmp_path, monkeypatch,
):
    launch_root = tmp_path / "launch"
    other_root = tmp_path / "other"
    launch_root.mkdir()
    other_root.mkdir()
    eid = "exp-root-pristine"

    monkeypatch.setenv("TCIP_STATE_ROOT", str(launch_root))
    exp.create_experiment(eid, {"a": 1})
    exp.update_status(eid, "running")  # no longer "created": no longer pristine

    monkeypatch.setenv("TCIP_STATE_ROOT", str(other_root))
    result = exp.overwrite_config_if_pristine(eid, {"a": 2}, root=launch_root)

    assert "error" in result
    assert _refusals(launch_root)
    assert _refusals(other_root) == []
