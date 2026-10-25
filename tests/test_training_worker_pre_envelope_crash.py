"""A crash in the training worker's pre-envelope setup closes the run ``failed``.

A crash in the dataset build closes the run ``failed`` with a final status naming the crash,
and the crash propagates.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import tcip_store as ts
from tcip_mcp import experiments as exp
from tcip_mcp.audit import audit_log_key
from tcip_mcp.pipelines.data import split_construction as sc
from tcip_mcp.pipelines.training import subprocess_worker as worker
from tests._audit_fixtures import refuse_audit_appends
from tests._verified_checkpoint_fixtures import detection_config, opened_run

if TYPE_CHECKING:
    from typing import NoReturn


def _training_run_events(root: Path) -> list[dict]:
    events = ts.read_log(audit_log_key(root)).records
    return [e for e in events if e.get("tool") == "training_run"]


def test_a_pre_envelope_crash_marks_the_run_failed_and_opens_a_training_run_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = opened_run(
        tmp_path,
        detection_config(tmp_path / "data", project=tmp_path, batch_size=1),
        experiment_id="exp-worker-crash",
    )

    def _boom(*args: object, **kwargs: object) -> NoReturn:
        raise RuntimeError("dataset build exploded")

    # recorded_datasets is imported inside prepare_run_context at call time, so patching the
    # source module's own name is what the worker's own lazy import resolves.
    monkeypatch.setattr(sc, "recorded_datasets", _boom)
    with pytest.raises(RuntimeError, match="dataset build exploded"):
        worker.run_directory(run_dir)

    observation = exp.observe_run(run_dir)
    final = observation.final
    assert final is not None
    assert final.state == "failed"
    assert final.status_error == "dataset build exploded"
    assert final.checkpoint is None
    assert observation.state == "failed"

    events = _training_run_events(tmp_path)
    assert len(events) == 1
    assert events[0]["status"] == "failed"
    assert events[0]["arguments"]["experiment_id"] == "exp-worker-crash"


def test_a_pre_envelope_crash_whose_audit_line_is_refused_names_both_causes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-envelope crash whose audit line is refused names both causes.

    The setup error propagates, and the final status names it and the unwritten line.
    """
    run_dir = opened_run(
        tmp_path,
        detection_config(tmp_path / "data", project=tmp_path, batch_size=1),
        experiment_id="exp-worker-crash-unaudited",
    )

    def _boom(*args: object, **kwargs: object) -> NoReturn:
        raise RuntimeError("dataset build exploded")

    monkeypatch.setattr(sc, "recorded_datasets", _boom)
    refuse_audit_appends(monkeypatch, error=OSError("the log's disk is full"))
    with pytest.raises(RuntimeError, match="dataset build exploded"):
        worker.run_directory(run_dir)

    final = exp.observe_run(run_dir).final
    assert final is not None and final.status_error is not None
    assert final.state == "failed"
    assert "dataset build exploded" in final.status_error
    assert "audit entry could not be written" in final.status_error
