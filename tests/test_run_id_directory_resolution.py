"""A custom-named record, launched by one process, monitored/cancelled/streamed by its own id
from a second process that never held it in its own in-memory registry.

There is one id: a training run's id is always its experiment id (no record, no run), so a
second process reaches a record it never launched by that id alone, straight off the status
record, with no resolver to fall back to and no second, minted id to reconcile against it.
"""

from __future__ import annotations


def _launch_custom_named_record(experiment_id: str, output_dir: str) -> None:
    """The exact create_experiment/stamp_run_identity sequence _ensure_experiment drives for a
    pre-created record, leaving nothing in this process's own in-memory registry: a second
    process's monitor_training/cancel_training/stream calls below see only the disk record,
    never a TrainRun for this id."""
    from tcip_mcp.experiments import create_experiment, stamp_run_identity

    create_experiment(experiment_id, {"model_source": {"builder": "my_models:bud_det"}})
    stamp_run_identity(experiment_id, output_dir, launched_by={"launcher": "process"})


def test_a_custom_named_record_is_monitored_by_its_own_id_with_no_registry_entry(
    tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import log_metrics
    from tcip_mcp.pipelines.training.run_registry import get_run
    from tcip_mcp.tools.training_tools import monitor_training

    eid = "exp-001-bud-det"
    output_dir = tmp_path / "runs" / eid
    output_dir.mkdir(parents=True)
    _launch_custom_named_record(eid, str(output_dir))
    log_metrics(eid, 3, {"val_map50": 0.4})

    assert get_run(eid) is None  # the registry this process holds never saw this id

    result = monitor_training(eid)
    assert result["status"] == "running"
    assert result["epoch"] == 3
    assert result["output_dir"] == str(output_dir)


def test_a_custom_named_record_is_cancelled_by_its_own_id_with_no_registry_entry(
    tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import cancel_training

    eid = "exp-002-bud-det"
    output_dir = tmp_path / "runs" / eid
    output_dir.mkdir(parents=True)
    _launch_custom_named_record(eid, str(output_dir))

    result = cancel_training(eid)
    assert result["cancel_requested"] is True
    assert result["experiment_id"] == eid
    assert (output_dir / ".cancel_requested").is_file()


def test_a_custom_named_record_streams_by_its_own_id_with_no_registry_entry(tmp_path, monkeypatch):
    """The web route's own metrics key resolves straight from the id, exactly like
    monitor_training/cancel_training above: no resolver, no in-memory entry needed."""
    monkeypatch.chdir(tmp_path)
    from tcip_store import read_log

    from tcip_mcp.experiments import log_metrics
    from tcip_web.routes.training import _metrics_key

    eid = "exp-003-bud-det"
    output_dir = tmp_path / "runs" / eid
    output_dir.mkdir(parents=True)
    _launch_custom_named_record(eid, str(output_dir))
    log_metrics(eid, 1, {"val_map50": 0.2})

    key = _metrics_key(str(tmp_path), eid)
    rows = [dict(r) for r in read_log(key).records]
    assert rows and rows[0]["val_map50"] == 0.2


def test_an_id_no_record_ever_stamped_resolves_to_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import reconstruct_run_status

    _launch_custom_named_record("exp-004-bud-det", str(tmp_path / "runs" / "a"))

    assert reconstruct_run_status("exp-never-launched") is None
