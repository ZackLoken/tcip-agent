"""The terminal lock protects an experiment's provenance writers, not just its own members.

subprocess_worker's two config.json patches and _persist_split_manifest's split.json write had no
state precondition at all: a run whose experiment record turned terminal mid-flight (the wall-clock
watchdog marking it failed while the child was still building its dataset) would have those writes
land anyway. They now share experiments.refuse_if_terminal with log_metrics/record_artifact, and a
refusal there raises ExperimentTerminal rather than degrading to a logged warning, so the worker
exits non-zero with the reason on stderr instead of training against, and silently patching, a
record that already closed.
"""

from __future__ import annotations

import tcip_store as ts
from tcip_mcp import experiments as exp
from tcip_mcp.audit import audit_log_key


def _refusals(root):
    events = ts.read_log(audit_log_key(root)).records
    return [e for e in events if e.get("tool") == "experiment_mutation_refused"]


class _StemDataset:
    """The minimal shape _persist_split_manifest reads off a built dataset."""

    def __init__(self, stems):
        self.stems = stems


def test_split_write_refused_against_a_watchdog_failed_record_leaves_it_failed(tmp_path):
    from tcip_mcp.experiments import ExperimentTerminal, create_experiment, update_status
    from tcip_mcp.tools.training_tools import _persist_split_manifest

    import pytest

    eid = "exp-020-hazelnut-catkin-det"
    create_experiment(eid, {"model_source": {"builder": "my_models:catkin_det"}})
    update_status(eid, "running")
    # The reachable trigger: the wall-clock watchdog marks the run failed while the child worker
    # is still alive and mid dataset-build, before it ever reaches _persist_split_manifest.
    update_status(eid, "failed", error="exceeded max_wall_clock_seconds (5)")

    with pytest.raises(ExperimentTerminal):
        _persist_split_manifest(
            eid, _StemDataset(["img_001"]), _StemDataset(["img_002"]),
            {"labels_dir": ""},
        )

    status = ts.read(exp.status_key(eid, root=tmp_path))
    assert status["state"] == "failed"
    assert status["error"] == "exceeded max_wall_clock_seconds (5)"  # the watchdog's own reason
    assert not ts.exists(exp.split_key(eid, root=tmp_path))  # the write never landed

    refusals = _refusals(tmp_path)
    assert len(refusals) == 1
    assert refusals[0]["arguments"]["op"] == "persist_split_manifest"
    assert refusals[0]["arguments"]["experiment_id"] == eid
    assert refusals[0]["status"] == "refused"


def test_update_status_refusal_audits_the_launch_root_not_the_current_one(tmp_path, monkeypatch):
    """A launch's wall-clock watchdog passes the root it captured at launch; its refused write's
    audit line must land on that root's own log, not whatever this process has since adopted."""
    from tcip_mcp.experiments import create_experiment, update_status

    launch_root = tmp_path / "launch"
    other_root = tmp_path / "other"
    launch_root.mkdir()
    other_root.mkdir()

    eid = "exp-021-hazelnut-catkin-det"
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(launch_root))
    create_experiment(eid, {"model_source": {"builder": "my_models:catkin_det"}})
    update_status(eid, "completed")

    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(other_root))
    result = update_status(
        eid, "failed", error="exceeded max_wall_clock_seconds (5)", root=launch_root
    )
    assert "error" in result

    assert _refusals(launch_root)
    assert not _refusals(other_root)


def test_split_write_still_lands_against_a_running_record(tmp_path):
    """The guard admits the ordinary case: a live run's own split write still succeeds."""
    from tcip_mcp.experiments import create_experiment, update_status
    from tcip_mcp.tools.training_tools import _persist_split_manifest

    eid = "exp-021-chestnut-burr-det"
    create_experiment(eid, {"model_source": {"builder": "my_models:burr_det"}})
    update_status(eid, "running")

    _persist_split_manifest(
        eid, _StemDataset(["img_001", "img_003"]), _StemDataset(["img_002"]),
        {"labels_dir": ""},
    )

    manifest = ts.read(exp.split_key(eid, root=tmp_path))
    assert manifest["train"] == ["img_001", "img_003"]
    assert manifest["val"] == ["img_002"]
    assert _refusals(tmp_path) == []


def test_tiling_patch_refused_against_a_terminal_record_raises_and_audits(tmp_path):
    from tcip_mcp.experiments import ExperimentTerminal, create_experiment, update_status

    import pytest

    from tcip_mcp.pipelines.training.subprocess_worker import _patch_experiment_config_tiling

    eid = "exp-022-currant-cluster-det"
    create_experiment(eid, {"model_source": {"builder": "my_models:cluster_det"}})
    update_status(eid, "running")
    update_status(eid, "completed")
    config_before = ts.read(exp.config_key(eid, root=tmp_path))

    with pytest.raises(ExperimentTerminal):
        _patch_experiment_config_tiling(eid, {"tile_size": 224})

    assert ts.read(exp.config_key(eid, root=tmp_path)) == config_before  # untouched
    refusals = _refusals(tmp_path)
    assert refusals and refusals[0]["arguments"]["op"] == "patch_experiment_config_tiling"


def test_tiling_patch_still_lands_against_a_running_record(tmp_path):
    from tcip_mcp.experiments import create_experiment, update_status
    from tcip_mcp.pipelines.training.subprocess_worker import _patch_experiment_config_tiling

    eid = "exp-023-elderberry-umbel-det"
    create_experiment(eid, {"model_source": {"builder": "my_models:umbel_det"}, "data": {}})
    update_status(eid, "running")

    _patch_experiment_config_tiling(eid, {"tile_size": 224})

    config = ts.read(exp.config_key(eid, root=tmp_path))
    assert config["data"]["tiling"]["tile_size"] == 224


def test_id_map_patch_refused_against_a_terminal_record(tmp_path):
    from tcip_mcp.experiments import ExperimentTerminal, create_experiment, update_status

    import pytest

    from tcip_mcp.pipelines.training.subprocess_worker import _patch_experiment_config_id_map

    eid = "exp-024-persimmon-fruit-det"
    create_experiment(eid, {"model_source": {"builder": "my_models:fruit_det"}})
    update_status(eid, "running")
    update_status(eid, "failed", error="dataloader raised")

    with pytest.raises(ExperimentTerminal):
        _patch_experiment_config_id_map(eid, "catkin", None, {"catkin": 0})

    refusals = _refusals(tmp_path)
    assert refusals and refusals[0]["arguments"]["op"] == "patch_experiment_config_id_map"


def test_split_write_raises_when_the_refusal_audit_append_fails(tmp_path, monkeypatch):
    """A refusal's own audit line failing to write must not swallow the refusal: the write still
    never lands and ExperimentTerminal still reaches the caller, chaining the append failure
    rather than losing it to a logged warning."""
    from tcip_mcp.experiments import ExperimentTerminal, create_experiment, update_status
    from tcip_mcp.tools.training_tools import _persist_split_manifest

    import pytest

    eid = "exp-027-hazelnut-catkin-det-3"
    create_experiment(eid, {"model_source": {"builder": "my_models:catkin_det"}})
    update_status(eid, "running")
    update_status(eid, "failed", error="exceeded max_wall_clock_seconds (5)")

    def _boom(*a, **k):
        raise OSError("simulated audit append failure")

    monkeypatch.setattr("tcip_mcp.audit.record_event_or_raise", _boom)

    with pytest.raises(ExperimentTerminal) as excinfo:
        _persist_split_manifest(
            eid, _StemDataset(["img_001"]), _StemDataset(["img_002"]),
            {"labels_dir": ""},
        )
    assert isinstance(excinfo.value.__cause__, OSError)

    assert not ts.exists(exp.split_key(eid, root=tmp_path))  # the write still never landed


def test_tiling_patch_raises_when_the_refusal_audit_append_fails(tmp_path, monkeypatch):
    from tcip_mcp.experiments import ExperimentTerminal, create_experiment, update_status

    import pytest

    from tcip_mcp.pipelines.training.subprocess_worker import _patch_experiment_config_tiling

    eid = "exp-028-currant-cluster-det-2"
    create_experiment(eid, {"model_source": {"builder": "my_models:cluster_det"}})
    update_status(eid, "running")
    update_status(eid, "completed")
    config_before = ts.read(exp.config_key(eid, root=tmp_path))

    def _boom(*a, **k):
        raise OSError("simulated audit append failure")

    monkeypatch.setattr("tcip_mcp.audit.record_event_or_raise", _boom)

    with pytest.raises(ExperimentTerminal) as excinfo:
        _patch_experiment_config_tiling(eid, {"tile_size": 224})
    assert isinstance(excinfo.value.__cause__, OSError)

    assert ts.read(exp.config_key(eid, root=tmp_path)) == config_before  # untouched


def test_id_map_patch_raises_when_the_refusal_audit_append_fails(tmp_path, monkeypatch):
    from tcip_mcp.experiments import ExperimentTerminal, create_experiment, update_status

    import pytest

    from tcip_mcp.pipelines.training.subprocess_worker import _patch_experiment_config_id_map

    eid = "exp-029-persimmon-fruit-det-2"
    create_experiment(eid, {"model_source": {"builder": "my_models:fruit_det"}})
    update_status(eid, "running")
    update_status(eid, "failed", error="dataloader raised")

    def _boom(*a, **k):
        raise OSError("simulated audit append failure")

    monkeypatch.setattr("tcip_mcp.audit.record_event_or_raise", _boom)

    with pytest.raises(ExperimentTerminal) as excinfo:
        _patch_experiment_config_id_map(eid, "catkin", None, {"catkin": 0})
    assert isinstance(excinfo.value.__cause__, OSError)


def test_overwrite_config_if_pristine_still_succeeds_over_a_pristine_experiment(tmp_path):
    """P6-26's transaction narrowing changes no accepted call: a genuinely pristine experiment's
    config is still overwritten."""
    from tcip_mcp.experiments import create_experiment, overwrite_config_if_pristine

    eid = "exp-025-black_locust-raceme-det"
    create_experiment(eid, {"a": 1})

    result = overwrite_config_if_pristine(eid, {"a": 2, "data": {"tiling": {"tile_size": 512}}})

    assert "error" not in result
    assert ts.read(exp.config_key(eid, root=tmp_path)) == {"a": 2, "data": {"tiling": {"tile_size": 512}}}


def test_overwrite_config_if_pristine_still_refuses_a_non_pristine_experiment(tmp_path):
    from tcip_mcp.experiments import create_experiment, log_metrics, overwrite_config_if_pristine, update_status

    eid = "exp-026-hazelnut-catkin-det-2"
    create_experiment(eid, {"a": 1})
    update_status(eid, "running")
    log_metrics(eid, 1, {"loss": 0.5})

    result = overwrite_config_if_pristine(eid, {"a": 2})

    assert "error" in result
    assert ts.read(exp.config_key(eid, root=tmp_path)) == {"a": 1}
