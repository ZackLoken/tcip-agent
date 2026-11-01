"""The local gate plans every job the CI workflow declares but those named out of scope."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

from tools import gate_baseline

if TYPE_CHECKING:
    from pathlib import Path


def test_a_job_added_to_the_workflow_and_named_nowhere_is_planned(
    tmp_path: Path,
) -> None:
    """Every job of a copy of ci.yml carrying one added job is planned or named out of scope."""
    workflow = yaml.safe_load(
        gate_baseline.CI_WORKFLOW.read_text(encoding="utf-8")
    )
    workflow["jobs"]["added"] = {
        "runs-on": "ubuntu-latest",
        "steps": [{"name": "Echo", "run": "echo added"}],
    }
    copy = tmp_path / "ci.yml"
    copy.write_text(yaml.safe_dump(workflow), encoding="utf-8")

    planned = {stage.job for stage in gate_baseline.build_plan(copy)}

    assert (
        set(workflow["jobs"]) - planned - set(gate_baseline.OUT_OF_SCOPE_JOBS)
        == set()
    )
    assert "added" in planned


def test_no_two_stages_of_the_ci_plan_share_a_key() -> None:
    """Each stage key of the plan over ci.yml names one stage, so --only selects one."""
    keys = [stage.key for stage in gate_baseline.build_plan()]

    assert sorted(k for k in set(keys) if keys.count(k) > 1) == []
