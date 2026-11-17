"""The frontend's dependency audit excepts advisories only through its committed exceptions file.

CI's ``typescript`` job audits the frontend's full dependency set with ``better-npm-audit``, which
reads the exceptions file ``.nsprc`` beside the frontend's ``package.json``. The tests read that
file and the local gate's own plan of the real workflow; the audit itself needs the registry and
is not run.
"""

from __future__ import annotations

import argparse
import datetime
import shlex
from typing import TYPE_CHECKING, Any

import yaml

from tests import FRONTEND
from tools import gate_baseline

if TYPE_CHECKING:
    from tools.gate_baseline import Stage

AUDIT_TOOL = "better-npm-audit"
EXCEPTIONS_FILE = FRONTEND / ".nsprc"


def _audit_stage() -> Stage:
    """Return the one stage of the plan over ci.yml that runs the audit tool in ``typescript``."""
    stages = [
        stage
        for stage in gate_baseline.build_plan()
        if stage.job == "typescript" and AUDIT_TOOL in (stage.run or "")
    ]
    assert len(stages) == 1, [stage.key for stage in stages]
    return stages[0]


def test_every_exception_is_active_with_a_note_and_a_calendar_date_expiry() -> (
    None
):
    assert EXCEPTIONS_FILE.is_file(), f"{EXCEPTIONS_FILE} does not exist"
    # Per the tool's readme, a JavaScript exceptions file beside it wins over it.
    for script in (".nsprc.cjs", ".nsprc.js"):
        assert not (FRONTEND / script).exists(), script
    # The tool reads the file as YAML, which also accepts the JSON the file is written in.
    exceptions: dict[str, Any] = yaml.safe_load(
        EXCEPTIONS_FILE.read_text(encoding="utf-8")
    )

    assert exceptions, "the file holds no exception"
    for advisory, entry in exceptions.items():
        assert isinstance(entry, dict), (advisory, entry)
        assert entry.get("active") is True, advisory
        assert str(entry.get("notes", "")).strip(), advisory
        expiry = entry.get("expiry")
        assert isinstance(expiry, str), (advisory, expiry)
        datetime.date.fromisoformat(expiry)


def test_the_audit_stage_fails_at_level_high_over_the_full_set_in_the_frontend() -> (
    None
):
    stage = _audit_stage()
    tokens = shlex.split(stage.run or "")
    tool_arguments = next(
        tokens[i + 1 :] for i, t in enumerate(tokens) if AUDIT_TOOL in t
    )
    # The options below are the tool's own, from its readme and `audit --help`.
    options = argparse.ArgumentParser(add_help=False)
    options.add_argument("-l", "--level")
    options.add_argument("-p", "--production", action="store_true")
    options.add_argument("--skip-optional", action="store_true")
    options.add_argument("-x", "--exclude")
    options.add_argument("-m", "--module-ignore")
    options.add_argument("--ignore-dependencies-of")
    parsed, _ = options.parse_known_args(tool_arguments)

    assert stage.cwd == FRONTEND
    assert parsed.level == "high"
    assert not parsed.production
    assert not parsed.skip_optional
    assert parsed.exclude is None
    assert parsed.module_ignore is None
    assert parsed.ignore_dependencies_of is None


def test_the_audit_step_does_not_continue_on_error() -> None:
    stage = _audit_stage()
    workflow = yaml.safe_load(
        gate_baseline.CI_WORKFLOW.read_text(encoding="utf-8")
    )
    job = workflow["jobs"][stage.job]
    (step,) = [s for s in job["steps"] if s.get("name") == stage.name]

    assert "continue-on-error" not in job
    assert "continue-on-error" not in step
