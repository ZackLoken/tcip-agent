"""``tcip diagnose-project`` reports each run record that will not read as its own finding.

A run is opened or run to completion by the platform's own writers; the damaged ones are then
rewritten on disk (no writer replaces a published ``run.json``), so the doctor meets records the
platform produced and a later edit spoiled.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tcip_mcp import experiments
from tcip_mcp.experiments import METRICS_FILE, RUN_FILE
from tcip_store import StoreError, encode_record
from tests._verified_checkpoint_fixtures import (
    detection_config,
    finished_run,
    opened_run,
)
from tests._web_fixtures import new_project
from tests.test_diagnose_project_selections import _findings

REGISTRY_INDEX_FINDING = "model registry index will not"
"""What the doctor's finding says when the registry's foreign index will not read."""


def _rewritten(run_dir: Path, change: Callable[[dict], None]) -> None:
    """Rewrite the run's published ``run.json`` on disk with ``change`` applied to its record."""
    path = run_dir / RUN_FILE
    record = experiments.read_record(path)
    assert isinstance(record, dict)
    change(record)
    path.write_bytes(encode_record(record))


def _without_key(run_dir: Path) -> None:
    _rewritten(run_dir, lambda record: record.pop("launched_as"))


def _undecodable(run_dir: Path) -> None:
    (run_dir / RUN_FILE).write_bytes(b"{not valid json")


def _traversing(run_dir: Path) -> None:
    _rewritten(run_dir, lambda record: record.update(resume_from="../last.pt"))


def _empty_path(run_dir: Path) -> None:
    _rewritten(run_dir, lambda record: record.update(resume_from=""))


def _without_metrics(run_dir: Path) -> None:
    (run_dir / METRICS_FILE).unlink()


DAMAGES = {
    "missing key": _without_key,
    "undecodable": _undecodable,
    "traversing path": _traversing,
    "empty path": _empty_path,
    "no metrics log": _without_metrics,
}
"""Each way a run the platform opened is spoiled on disk, by the id it is damaged under."""


def _damaged_run(
    tmp_path: Path, project: Path, damage: Callable[[Path], None]
) -> str:
    """Open a run under ``project``, spoil it with ``damage``, and return its id."""
    run_dir = opened_run(
        project,
        detection_config(tmp_path / "ds", project=project),
        experiment_id="exp-damaged",
    )
    damage(run_dir)
    return run_dir.name


@pytest.mark.parametrize("damage", DAMAGES.values(), ids=DAMAGES.keys())
def test_the_doctor_command_reports_a_run_that_will_not_read_as_its_own_finding(
    tmp_path: Path, damage: Callable[[Path], None]
) -> None:
    """A spoiled run is one error finding naming it, and the command answers with no traceback.

    The registry index is not blamed for it.
    """
    project = tmp_path / "proj"
    new_project(project)
    run_id = _damaged_run(tmp_path, project, damage)

    found = _findings(project)

    assert [level for level, msg in found if run_id in msg] == ["error"], found
    assert not [msg for _, msg in found if REGISTRY_INDEX_FINDING in msg], (
        found
    )


def test_the_registry_checks_go_on_over_the_runs_that_read(
    tmp_path: Path,
) -> None:
    """A completed run whose checkpoint is gone is still checked beside a run that will not read."""
    project = tmp_path / "proj"
    new_project(project)
    _damaged_run(tmp_path, project, _undecodable)
    completed = finished_run(project, experiment_id="exp-completed")
    checkpoint = experiments.observe_run(completed).checkpoint
    assert checkpoint is not None
    Path(checkpoint.path).unlink()

    found = _findings(project)

    assert [
        level
        for level, msg in found
        if "registry entry 'exp-completed'" in msg
        and "checkpoint missing" in msg
    ] == ["error"], found


def test_a_reader_of_the_runs_still_refuses_on_a_run_that_will_not_read(
    tmp_path: Path,
) -> None:
    """``run_observations`` raises the refusal of a run that will not read."""
    project = tmp_path / "proj"
    new_project(project)
    _damaged_run(tmp_path, project, _undecodable)

    with pytest.raises(StoreError, match="exp-damaged"):
        experiments.run_observations(project)
