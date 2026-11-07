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


def _final_rewritten(run_dir: Path, change: Callable[[dict], None]) -> None:
    """Rewrite the run's published ``final_status.json`` on disk with ``change`` applied."""
    path = run_dir / experiments.FINAL_STATUS_FILE
    record = experiments.read_record(path)
    assert isinstance(record, dict)
    change(record)
    path.write_bytes(encode_record(record))


def _final_without_key(run_dir: Path) -> None:
    _final_rewritten(run_dir, lambda record: record.pop("ended"))


def _final_traversing(run_dir: Path) -> None:
    def change(record: dict) -> None:
        record["checkpoint"]["path"] = "../last.pt"

    _final_rewritten(run_dir, change)


FINAL_DAMAGES = {
    "missing key": _final_without_key,
    "traversing path": _final_traversing,
}
"""Each way a completed run's final status is spoiled on disk, by the id it is damaged under."""


def _damaged_run(
    tmp_path: Path,
    project: Path,
    damage: Callable[[Path], None],
    experiment_id: str = "exp-damaged",
) -> str:
    """Open a run under ``project``, spoil it with ``damage``, and return its id."""
    run_dir = opened_run(
        project,
        detection_config(tmp_path / "ds", project=project),
        experiment_id=experiment_id,
    )
    damage(run_dir)
    return run_dir.name


def _completed_without_checkpoint(project: Path, experiment_id: str) -> str:
    """Run a training run to completion under ``project``, delete its checkpoint, return its id."""
    run_dir = finished_run(project, experiment_id=experiment_id)
    checkpoint = experiments.observe_run(run_dir).checkpoint
    assert checkpoint is not None
    Path(checkpoint.path).unlink()
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


@pytest.mark.parametrize(
    "damage", FINAL_DAMAGES.values(), ids=FINAL_DAMAGES.keys()
)
def test_the_doctor_command_reports_a_final_status_that_will_not_read_as_its_own_finding(
    tmp_path: Path, damage: Callable[[Path], None]
) -> None:
    """A completed run whose final status is spoiled is one error finding naming the run."""
    project = tmp_path / "proj"
    new_project(project)
    run_dir = finished_run(project, experiment_id="exp-final")
    damage(run_dir)

    found = _findings(project)

    assert [level for level, msg in found if "exp-final" in msg] == [
        "error"
    ], found
    assert not [msg for _, msg in found if REGISTRY_INDEX_FINDING in msg], (
        found
    )


def test_every_run_that_will_not_read_is_named_and_the_runs_that_read_are_checked(
    tmp_path: Path,
) -> None:
    """Two runs spoiled two ways are two findings, and a completed run beside them is checked.

    The completed run's checkpoint is gone, so the per-entry check names it.
    """
    project = tmp_path / "proj"
    new_project(project)
    spoiled = [
        _damaged_run(tmp_path, project, _undecodable, "exp-undecodable"),
        _damaged_run(tmp_path, project, _without_metrics, "exp-unlogged"),
    ]
    completed = _completed_without_checkpoint(project, "exp-completed")

    found = _findings(project)

    for run_id in spoiled:
        assert [level for level, msg in found if run_id in msg] == ["error"], (
            found
        )
    assert [
        level
        for level, msg in found
        if f"registry entry {completed!r}" in msg
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
