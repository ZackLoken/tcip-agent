"""``scripts/gate_baseline.py`` runs the steps ``.github/workflows/ci.yml`` actually declares.

These tests hold the parsed plan to the workflow file itself, loaded independently here, so a
job or step CI gains that the script silently drops (or wrongly skips) is caught rather than
assumed away.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "gate_baseline.py"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _load():
    spec = importlib.util.spec_from_file_location("gate_baseline", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gate_baseline"] = mod
    spec.loader.exec_module(mod)
    return mod


def _workflow_job_names() -> set:
    return set(yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))["jobs"])


def test_parsed_job_set_matches_the_workflows_job_set_with_environment_out_of_scope():
    gate_baseline = _load()
    workflow_jobs = _workflow_job_names()
    assert gate_baseline.OUT_OF_SCOPE_JOB in workflow_jobs
    assert workflow_jobs - {gate_baseline.OUT_OF_SCOPE_JOB} == set(gate_baseline.PARSED_JOBS)


def test_every_declared_step_runs_or_is_skipped_by_the_stated_rule():
    gate_baseline = _load()
    plan = gate_baseline.build_plan()
    by_key = {stage.key: stage for stage in plan}

    skipped_by_uses = {
        "actions/checkout@v4", "actions/setup-python@v5", "actions/cache@v4",
        "actions/checkout@v4 [sqlite]", "actions/setup-python@v5 [sqlite]",
        "actions/cache@v4 [sqlite]", "actions/checkout@v4 [file]",
        "actions/setup-python@v5 [file]", "actions/cache@v4 [file]",
        "actions/checkout@v4", "actions/setup-node@v4",
    }
    skipped_by_run_prefix = {
        "Install CPU-only torch", "Install packages",
        "Install CPU-only torch [sqlite]", "Install packages [sqlite]",
        "Install CPU-only torch [file]", "Install packages [file]",
        "Install frontend dependencies",
    }
    run_steps = {
        "Type check (mypy, permissive; see mypy.ini)",
        "Lint (ruff) [sqlite]", "ARCHITECTURE.md matches the tree [sqlite]",
        "ARCHITECTURE.md citations match what they quote [sqlite]", "Run tests [sqlite]",
        "Lint (ruff) [file]", "ARCHITECTURE.md matches the tree [file]",
        "ARCHITECTURE.md citations match what they quote [file]", "Run tests [file]",
        "Format check", "Lint", "Type check", "Test", "Build",
    }

    assert set(by_key) == skipped_by_uses | skipped_by_run_prefix | run_steps
    for key in skipped_by_uses:
        assert by_key[key].skip_reason is not None and by_key[key].skip_reason.startswith("uses:")
    for key in skipped_by_run_prefix:
        assert by_key[key].skip_reason is not None and "run: starts with" in by_key[key].skip_reason
    for key in run_steps:
        assert by_key[key].skip_reason is None


def test_python_jobs_matrix_leg_env_carries_the_resolved_store_backend():
    gate_baseline = _load()
    by_key = {stage.key: stage for stage in gate_baseline.build_plan()}
    assert by_key["Run tests [sqlite]"].env["TCIP_STORE_BACKEND"] == "sqlite"
    assert by_key["Run tests [file]"].env["TCIP_STORE_BACKEND"] == "file"


def test_main_refuses_without_an_out_directory(monkeypatch):
    gate_baseline = _load()
    # --timeout bounds a pre-change script that lacked this refusal and would otherwise run its
    # whole default stage list for real; the refusal here is expected to fire before any of that.
    monkeypatch.setattr(sys, "argv", ["gate_baseline.py", "--timeout", "1"])
    with pytest.raises(SystemExit):
        gate_baseline.main()


def test_main_refuses_the_phase0_baseline_directory(monkeypatch):
    gate_baseline = _load()
    monkeypatch.setattr(sys, "argv", [
        "gate_baseline.py", "--out", str(gate_baseline.PHASE0_BASELINE), "--timeout", "1",
    ])
    with pytest.raises(SystemExit):
        gate_baseline.main()


def test_main_runs_a_selected_stage_to_a_fresh_out_directory(tmp_path, monkeypatch):
    # The rail must admit valid work, not only refuse an invalid --out: a fresh scratch
    # directory with a real stage selected runs to completion and records a summary.
    gate_baseline = _load()
    out = tmp_path / "gate-out"
    monkeypatch.setattr(sys, "argv", [
        "gate_baseline.py", "--out", str(out), "--only", "Lint (ruff) [sqlite]",
    ])
    assert gate_baseline.main() == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["stages"][0]["stage"] == "Lint (ruff) [sqlite]"
    assert summary["stages"][0]["exit_code"] == 0
