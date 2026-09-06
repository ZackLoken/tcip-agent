"""A cancelled experiment is a finished record, not a slot to relaunch into.

Cancellation is a terminal outcome the audit log distinguishes from failure, so the record of what
a cancelled run did (its config, its status, its metrics) has to survive the next launch that names
the same id. ``_ensure_experiment`` mints a fresh parented id instead of reopening it.
"""

from __future__ import annotations

import tcip_store as ts
from tcip_mcp import experiments as exp


def test_relaunch_into_a_cancelled_id_mints_a_fresh_parented_id(tmp_path) -> None:
    """A cancelled experiment that never logged a metric is still finished: the relaunch gets its
    own id and the cancelled record keeps the config and status it was cancelled with."""
    from tcip_mcp.experiments import create_experiment, update_status
    from tcip_mcp.tools.training_tools import _ensure_experiment

    create_experiment("stopped", {"optimizer": {"head_lr": 0.001}}, data_source="imgs_v1")
    update_status("stopped", "running")
    update_status("stopped", "cancelled")
    status_before = ts.read(exp.status_key("stopped"))

    eid = _ensure_experiment("stopped", {"optimizer": {"head_lr": 0.05}}, "imgs_v2",
                             resume_from="", run_id="run_c1", output_dir="out/c1",
                             launched_by={"launcher": "process"})

    assert eid == "stopped_run_c1"
    assert ts.read(exp.config_key("stopped")) == {"optimizer": {"head_lr": 0.001}}
    assert ts.read(exp.status_key("stopped")) == status_before

    assert ts.read(exp.lineage_key("stopped_run_c1"))["parent_experiment"] == "stopped"
    fresh_status = ts.read(exp.status_key("stopped_run_c1"))
    assert fresh_status["run_id"] == "run_c1"
    assert fresh_status["output_dir"] == "out/c1"


def test_resuming_from_a_cancelled_runs_checkpoint_does_not_reopen_its_record(tmp_path) -> None:
    """Resuming a cancelled run continues the training, never the experiment record: the metrics
    the cancelled id already holds stay exactly as they were, and the resumed run's own history
    accumulates under a fresh id parented to it."""
    from tcip_mcp.experiments import create_experiment, log_metrics, update_status
    from tcip_mcp.tools.training_tools import _ensure_experiment

    create_experiment("stopped_mid", {"seed": 7})
    update_status("stopped_mid", "running")
    log_metrics("stopped_mid", 1, {"val_loss": 0.9})
    log_metrics("stopped_mid", 2, {"val_loss": 0.4})
    update_status("stopped_mid", "cancelled")
    metrics_before = list(ts.read_log(exp.metrics_key("stopped_mid")).records)

    eid = _ensure_experiment("stopped_mid", {"seed": 7}, None,
                             resume_from="out/checkpoint_epoch_2.pt", run_id="run_c2",
                             output_dir="out/c2", launched_by={"launcher": "process"})

    assert eid == "stopped_mid_run_c2"
    assert list(ts.read_log(exp.metrics_key("stopped_mid")).records) == metrics_before
    assert ts.read_log(exp.metrics_key("stopped_mid_run_c2")).records == []
    assert ts.read(exp.lineage_key("stopped_mid_run_c2"))["parent_experiment"] == "stopped_mid"
