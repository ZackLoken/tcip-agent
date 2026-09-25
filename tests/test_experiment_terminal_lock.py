"""The terminal lock protects an experiment's provenance writers, not just its own members.

subprocess_worker's config.json mirror and persist_run_partition's split.json write share
experiments.rewrite_live_member's terminal refusal with log_metrics and record_artifact, so a run whose experiment
record turned terminal mid-flight (the wall-clock watchdog marking it failed while the child was
still building its dataset) cannot have those writes land anyway. A refusal there raises
ExperimentTerminal rather than degrading to a logged warning, so the worker exits non-zero with
the reason on stderr instead of training against, and silently patching, a record that closed.
"""

from __future__ import annotations

import pytest
import tcip_store as ts
from tcip_mcp import experiments as exp
from tcip_mcp.audit import audit_log_key


def _refusals(root):
    """Every line the root's audit log holds: a refused write changes nothing and writes none."""
    return list(ts.read_log(audit_log_key(root)).records)


class _StemDataset:
    """The minimal shape persist_run_partition reads off a built dataset."""

    def __init__(self, stems):
        self.stems = stems


def test_split_write_refused_against_a_watchdog_failed_record_leaves_it_failed(tmp_path):
    from tcip_mcp.experiments import ExperimentTerminal, create_experiment, update_status
    from tcip_mcp.pipelines.data.split_construction import persist_run_partition

    eid = "exp-020-currant-bud-det"
    create_experiment(eid, {"model_source": {"builder": "my_models:bud_det"}})
    update_status(eid, "running")
    # The reachable trigger: the wall-clock watchdog marks the run failed while the child worker
    # is still alive and mid dataset-build, before it ever reaches persist_run_partition.
    update_status(eid, "failed", error="exceeded max_wall_clock_seconds (5)")

    with pytest.raises(ExperimentTerminal):
        persist_run_partition(
            eid,
            {"labels_dir": "", "split": {"resolved_seed": 0, "resolved_group_by": "stem"}},
        )

    status = ts.read(exp.status_key(eid, root=tmp_path))
    assert status["state"] == "failed"
    assert status["error"] == "exceeded max_wall_clock_seconds (5)"  # the watchdog's own reason
    assert not ts.exists(exp.split_key(eid, root=tmp_path))  # the write never landed

    assert _refusals(tmp_path) == []


def test_update_status_refusal_reads_the_launch_root_and_writes_no_line(tmp_path, monkeypatch):
    """A launch's wall-clock watchdog passes the root it captured at launch; the refusal is
    decided against that root's record and writes no line under either root."""
    from tcip_mcp.experiments import create_experiment, update_status

    launch_root = tmp_path / "launch"
    other_root = tmp_path / "other"
    launch_root.mkdir()
    other_root.mkdir()

    eid = "exp-021-currant-bud-det"
    monkeypatch.setenv("TCIP_STATE_ROOT", str(launch_root))
    create_experiment(eid, {"model_source": {"builder": "my_models:bud_det"}})
    update_status(eid, "completed")

    monkeypatch.setenv("TCIP_STATE_ROOT", str(other_root))
    result = update_status(
        eid, "failed", error="exceeded max_wall_clock_seconds (5)", root=launch_root
    )
    assert "error" in result

    assert result["state"] == "completed"
    assert not _refusals(launch_root)
    assert not _refusals(other_root)


def test_split_write_still_lands_against_a_running_record(tmp_path):
    """The guard admits the ordinary case: a live run's own split write still succeeds.

    The membership is the one a producer named, built here through ``_recorded_partition``, the
    same function every route hands this writer: the record's members never come from a loader.
    """
    from tcip_mcp.dataset_layout import status_bucket
    from tcip_mcp.experiments import create_experiment, update_status
    from tcip_mcp.pipelines.data.selection import Sample
    from tcip_mcp.pipelines.data.split_construction import (
        _recorded_partition, persist_run_partition, recorded_side,
    )

    eid = "exp-021-chestnut-burr-det"
    create_experiment(eid, {"model_source": {"builder": "my_models:burr_det"}})
    update_status(eid, "running")

    labels_dir = tmp_path / "annotations"

    def _sample(stem: str, side: str) -> Sample:
        return Sample(member=stem, source=str(tmp_path / "images" / f"{stem}.png"),
                      ground_truth=str(labels_dir / f"{stem}.json"), group=stem, side=side,
                      confirmation_bucket=status_bucket("burr", None))

    train = [_sample("img_001", "train"), _sample("img_003", "train")]
    val = [_sample("img_002", "val")]
    persist_run_partition(
        eid,
        {"labels_dir": str(labels_dir),
         "split": {"resolved_seed": 0, "resolved_group_by": "stem"}},
        partition=_recorded_partition(train, val, train + val),
    )

    manifest = ts.read(exp.split_key(eid, root=tmp_path))
    assert recorded_side(manifest["members"], "train") == ["img_001", "img_003"]
    assert recorded_side(manifest["members"], "val") == ["img_002"]
    assert _refusals(tmp_path) == []


def test_the_data_mirror_refused_against_a_terminal_record_raises(tmp_path):
    from tcip_mcp.experiments import ExperimentTerminal, create_experiment, update_status
    from tcip_mcp.pipelines.training.subprocess_worker import _mirror_data_section

    eid = "exp-022-quince-cluster-det"
    create_experiment(eid, {"model_source": {"builder": "my_models:cluster_det"}})
    update_status(eid, "running")
    update_status(eid, "completed")
    config_before = ts.read(exp.config_key(eid, root=tmp_path))

    with pytest.raises(ExperimentTerminal):
        _mirror_data_section(eid, {"tiling": {"tile_size": 224}})

    assert ts.read(exp.config_key(eid, root=tmp_path)) == config_before  # untouched
    assert _refusals(tmp_path) == []


def test_the_data_mirror_lands_against_a_running_record(tmp_path):
    from tcip_mcp.experiments import create_experiment, update_status
    from tcip_mcp.pipelines.training.subprocess_worker import _mirror_data_section

    eid = "exp-023-quince-umbel-det"
    create_experiment(eid, {"model_source": {"builder": "my_models:umbel_det"}, "data": {}})
    update_status(eid, "running")

    _mirror_data_section(eid, {"tiling": {"tile_size": 224}})

    config = ts.read(exp.config_key(eid, root=tmp_path))
    assert config["data"]["tiling"]["tile_size"] == 224


def test_overwrite_config_if_pristine_still_succeeds_over_a_pristine_experiment(tmp_path):
    """The narrowed transaction changes no accepted call: a genuinely pristine experiment's
    config is still overwritten."""
    from tcip_mcp.experiments import create_experiment, overwrite_config_if_pristine

    eid = "exp-025-black_locust-raceme-det"
    create_experiment(eid, {"a": 1})

    result = overwrite_config_if_pristine(eid, {"a": 2, "data": {"tiling": {"tile_size": 512}}})

    assert "error" not in result
    assert ts.read(exp.config_key(eid, root=tmp_path)) == {"a": 2, "data": {"tiling": {"tile_size": 512}}}


def test_overwrite_config_if_pristine_still_refuses_a_non_pristine_experiment(tmp_path):
    from tcip_mcp.experiments import create_experiment, log_metrics, overwrite_config_if_pristine, update_status

    eid = "exp-026-currant-bud-det-2"
    create_experiment(eid, {"a": 1})
    update_status(eid, "running")
    log_metrics(eid, 1, {"loss": 0.5})

    result = overwrite_config_if_pristine(eid, {"a": 2})

    assert "error" in result
    assert ts.read(exp.config_key(eid, root=tmp_path)) == {"a": 1}
