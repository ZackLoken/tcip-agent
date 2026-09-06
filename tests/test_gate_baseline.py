"""``tools/gate_baseline.py`` runs the steps ``.github/workflows/ci.yml`` actually declares.

These tests hold the parsed plan to the workflow file itself, loaded independently here, so a
job or step CI gains that the script silently drops (or wrongly skips) is caught rather than
assumed away.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from collections import Counter

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "gate_baseline.py"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _load():
    spec = importlib.util.spec_from_file_location("gate_baseline", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gate_baseline"] = mod
    spec.loader.exec_module(mod)
    return mod


def _workflow_job_names() -> set:
    return set(yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))["jobs"])


def test_parsed_job_set_matches_the_workflows_job_set_with_the_out_of_scope_jobs_removed():
    gate_baseline = _load()
    workflow_jobs = _workflow_job_names()
    out_of_scope = set(gate_baseline.OUT_OF_SCOPE_JOBS)
    assert out_of_scope <= workflow_jobs
    assert workflow_jobs - out_of_scope == set(gate_baseline.PARSED_JOBS)


def test_every_declared_step_runs_or_is_skipped_by_the_stated_rule():
    # A Counter over the full plan, not a dict keyed by stage.key, so two stages sharing a key
    # (mypy's two unnamed actions/cache@v4 steps) are both counted, not one silently overwritten.
    gate_baseline = _load()
    plan = gate_baseline.build_plan()

    skipped_by_uses = [
        "mypy:actions/checkout@v4", "mypy:actions/setup-python@v5",
        "mypy:actions/cache@v4", "mypy:actions/cache@v4",
        "python:actions/checkout@v4 [sqlite]", "python:actions/setup-python@v5 [sqlite]",
        "python:actions/cache@v4 [sqlite]",
        "python:actions/checkout@v4 [file]", "python:actions/setup-python@v5 [file]",
        "python:actions/cache@v4 [file]",
        "typescript:actions/checkout@v4", "typescript:actions/setup-node@v4",
    ]
    skipped_by_run_prefix = [
        "mypy:Install CPU-only torch", "mypy:Install packages",
        "python:Install CPU-only torch [sqlite]", "python:Install packages [sqlite]",
        "python:Install CPU-only torch [file]", "python:Install packages [file]",
        "typescript:Install frontend dependencies",
    ]
    run_steps = [
        "mypy:Type check (mypy; see mypy.ini)",
        "python:Lint (ruff) [sqlite]", "python:ARCHITECTURE.md matches the tree [sqlite]",
        "python:ARCHITECTURE.md citations match what they quote [sqlite]",
        "python:Run tests [sqlite]",
        "python:Lint (ruff) [file]", "python:ARCHITECTURE.md matches the tree [file]",
        "python:ARCHITECTURE.md citations match what they quote [file]", "python:Run tests [file]",
        "typescript:Format check", "typescript:Lint", "typescript:Type check",
        "typescript:Test", "typescript:Build",
    ]

    expected = Counter(skipped_by_uses + skipped_by_run_prefix + run_steps)
    assert Counter(stage.key for stage in plan) == expected

    by_key: dict = {}
    for stage in plan:
        by_key.setdefault(stage.key, []).append(stage)
    for key in skipped_by_uses:
        for stage in by_key[key]:
            assert stage.skip_reason is not None and stage.skip_reason.startswith("uses:")
    for key in skipped_by_run_prefix:
        for stage in by_key[key]:
            assert stage.skip_reason is not None and "run: starts with" in stage.skip_reason
    for key in run_steps:
        for stage in by_key[key]:
            assert stage.skip_reason is None


def test_every_stage_key_is_unique_per_job_step_and_leg():
    # mypy's two unnamed cache steps are the one case job/name/leg cannot separate; both share a
    # skip reason and never run, so that single duplicate is named here, not asserted away.
    gate_baseline = _load()
    plan = gate_baseline.build_plan()
    counts = Counter(stage.key for stage in plan)
    duplicated = {key: n for key, n in counts.items() if n > 1}
    assert duplicated == {"mypy:actions/cache@v4": 2}


def test_resolve_git_bash_prefers_the_derived_path_when_it_exists(tmp_path, monkeypatch):
    """The Windows derivation, exercised on any host by pinning the host predicate."""
    gate_baseline = _load()
    monkeypatch.setattr(gate_baseline, "_host_is_windows", lambda: True)
    exec_path = tmp_path / "Git" / "mingw64" / "libexec" / "git-core"
    exec_path.mkdir(parents=True)
    bash_exe = exec_path.parent.parent / "bin" / "bash.exe"
    bash_exe.parent.mkdir(parents=True)
    bash_exe.write_text("", encoding="utf-8")

    class _Completed:
        returncode = 0
        stdout = str(exec_path) + "\n"
        stderr = ""

    monkeypatch.setattr(gate_baseline.subprocess, "run", lambda *a, **k: _Completed())
    assert gate_baseline._resolve_git_bash() == str(bash_exe)


def test_resolve_git_bash_on_a_posix_host_is_the_bash_on_path(monkeypatch):
    """A Linux or macOS host runs each step through the bash on PATH, the shell the CI runner
    itself gives the run: text, with no Git-for-Windows lookup."""
    gate_baseline = _load()
    monkeypatch.setattr(gate_baseline, "_host_is_windows", lambda: False)
    monkeypatch.setattr(gate_baseline.shutil, "which", lambda name: "/usr/bin/bash")
    monkeypatch.setattr(
        gate_baseline.subprocess, "run",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no git probe on a posix host")),
    )
    assert gate_baseline._resolve_git_bash() == "/usr/bin/bash"


def test_resolve_git_bash_on_a_posix_host_refuses_with_no_bash_on_path(monkeypatch):
    gate_baseline = _load()
    monkeypatch.setattr(gate_baseline, "_host_is_windows", lambda: False)
    monkeypatch.setattr(gate_baseline.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit, match="bash was not found on PATH"):
        gate_baseline._resolve_git_bash()


def test_python_jobs_matrix_leg_env_carries_the_resolved_store_backend():
    gate_baseline = _load()
    by_key = {stage.key: stage for stage in gate_baseline.build_plan()}
    assert by_key["python:Run tests [sqlite]"].env["TCIP_STORE_BACKEND"] == "sqlite"
    assert by_key["python:Run tests [file]"].env["TCIP_STORE_BACKEND"] == "file"


def test_unresolved_expression_refuses_by_name(tmp_path, monkeypatch):
    # ci.yml itself carries no ${{ }} expression build_plan cannot resolve, so this drives the
    # refusal with a fixture workflow instead: a stray expression in a job's own env.
    gate_baseline = _load()
    workflow = tmp_path / "ci.yml"
    workflow.write_text(
        "jobs:\n"
        "  mypy:\n"
        "    env:\n"
        "      TOKEN: ${{ secrets.NOT_A_MATRIX_VALUE }}\n"
        "    steps: []\n"
        "  python:\n"
        "    steps: []\n"
        "  typescript:\n"
        "    steps: []\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gate_baseline, "CI_WORKFLOW", workflow)
    with pytest.raises(gate_baseline.UnresolvedExpressionError):
        gate_baseline.build_plan()


def test_main_refuses_without_an_out_directory(monkeypatch):
    gate_baseline = _load()
    # --timeout 1 is a property of this test, not the refusal under test: it bounds how long a
    # regression that let this run for real (rather than refusing) could hold up the suite.
    monkeypatch.setattr(sys, "argv", ["gate_baseline.py", "--timeout", "1"])
    with pytest.raises(SystemExit):
        gate_baseline.main()


def test_main_refuses_the_recorded_baseline_directory(monkeypatch):
    gate_baseline = _load()
    monkeypatch.setattr(sys, "argv", [
        "gate_baseline.py", "--out", str(gate_baseline.RECORDED_BASELINE), "--timeout", "1",
    ])
    with pytest.raises(SystemExit):
        gate_baseline.main()


def test_main_runs_a_selected_stage_to_a_fresh_out_directory(tmp_path, monkeypatch):
    # The rail must admit valid work, not only refuse an invalid --out: a fresh scratch
    # directory with a real stage selected runs to completion and records a summary.
    gate_baseline = _load()
    out = tmp_path / "gate-out"
    monkeypatch.setattr(sys, "argv", [
        "gate_baseline.py", "--out", str(out), "--only", "python:Lint (ruff) [sqlite]",
    ])
    assert gate_baseline.main() == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["stages"][0]["stage"] == "python:Lint (ruff) [sqlite]"
    assert summary["stages"][0]["exit_code"] == 0


def test_only_accepts_a_stage_key_carrying_a_comma(tmp_path, monkeypatch):
    # The mypy step's own name carries a comma, so --only must not split on one; the stage's own
    # subprocess is stubbed so this stays fast rather than running a real mypy pass.
    gate_baseline = _load()

    class _Completed:
        stdout = ""
        stderr = ""
        returncode = 0

    monkeypatch.setattr(gate_baseline.subprocess, "run", lambda *a, **k: _Completed())
    out = tmp_path / "gate-out"
    key = "mypy:Type check (mypy; see mypy.ini)"
    monkeypatch.setattr(sys, "argv", ["gate_baseline.py", "--out", str(out), "--only", key])
    assert gate_baseline.main() == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["stages"][0]["stage"] == key
