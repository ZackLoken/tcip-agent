"""The frontend's dependency audit excepts advisories only through a static exceptions file.

CI's ``typescript`` job audits the frontend's full dependency set with ``better-npm-audit``, which
reads the exceptions file ``.nsprc`` beside the frontend's ``package.json`` and refuses an entry
without a note or an expiry. The tests read the frontend's directory for a script standing in for
the file and the local gate's own plan of the real workflow; the audit itself needs the registry
and is not run.
"""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

import yaml

from tests import FRONTEND
from tools import gate_baseline

if TYPE_CHECKING:
    from tools.gate_baseline import Stage

AUDIT_TOOL = "better-npm-audit"


def _audit_stage() -> Stage:
    """Return the one stage of the plan over ci.yml that runs the audit tool in ``typescript``."""
    stages = [
        stage
        for stage in gate_baseline.build_plan()
        if stage.job == "typescript" and AUDIT_TOOL in (stage.run or "")
    ]
    assert len(stages) == 1, [stage.key for stage in stages]
    return stages[0]


def test_the_exceptions_are_a_static_file_never_a_script() -> None:
    # The tool prefers a JavaScript exceptions file beside it to the static one and executes it.
    for script in (".nsprc.cjs", ".nsprc.js"):
        assert not (FRONTEND / script).exists(), script


def test_the_audit_stage_runs_the_pinned_tool_over_the_full_set_requiring_notes_and_expiry() -> (
    None
):
    stage = _audit_stage()

    assert stage.cwd == FRONTEND
    assert shlex.split(stage.run or "") == [
        "npx",
        "--yes",
        f"{AUDIT_TOOL}@4.2.1",
        "audit",
        "--level",
        "high",
        "--require-notes",
        "--require-expiry",
    ]


def test_the_audit_step_does_not_continue_on_error() -> None:
    stage = _audit_stage()
    workflow = yaml.safe_load(
        gate_baseline.CI_WORKFLOW.read_text(encoding="utf-8")
    )
    job = workflow["jobs"][stage.job]
    (step,) = [s for s in job["steps"] if s.get("name") == stage.name]

    assert "continue-on-error" not in job
    assert "continue-on-error" not in step
