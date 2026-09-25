"""A canceled experiment is a finished record, not a slot to relaunch into.

Cancellation is a terminal outcome the audit log distinguishes from failure, so the record of what
a canceled run did (its config, its status, its metrics) has to survive the next launch that names
the same id. ``_ensure_experiment`` mints a fresh parented id instead of reopening it.
"""

from __future__ import annotations

import tcip_store as ts
from tcip_mcp import experiments as exp


def test_relaunch_into_a_canceled_id_mints_a_fresh_parented_id(tmp_path) -> None:
    """A canceled experiment that never logged a metric is still finished: the relaunch gets its
    own id and the canceled record keeps the config and status it was canceled with."""
    from tcip_mcp.experiments import create_experiment, update_status
    from tcip_mcp.tools.training_tools import _ensure_experiment

    create_experiment("stopped", {"optimizer": {"head_lr": 0.001}}, data_source="imgs_v1")
    update_status("stopped", "running")
    update_status("stopped", "canceled")
    status_before = ts.read(exp.status_key("stopped"))

    eid, out_dir = _ensure_experiment("stopped", {"optimizer": {"head_lr": 0.05}}, "imgs_v2",
                                      resume_from="", output_base="out",
                                      launched_by={"launcher": "process"})

    assert eid.startswith("stopped_run_") and eid != "stopped"
    assert ts.read(exp.config_key("stopped")) == {"optimizer": {"head_lr": 0.001}}
    assert ts.read(exp.status_key("stopped")) == status_before

    assert ts.read(exp.lineage_key(eid))["parent_experiment"] == "stopped"
    fresh_status = ts.read(exp.status_key(eid))
    assert fresh_status["output_dir"] == out_dir


def test_resuming_from_a_canceled_runs_checkpoint_does_not_reopen_its_record(tmp_path) -> None:
    """Resuming a canceled run continues the training, never the experiment record: the metrics
    the canceled id already holds stay exactly as they were, and the resumed run's own history
    accumulates under a fresh id parented to it."""
    from tcip_mcp.experiments import create_experiment, log_metrics, update_status
    from tcip_mcp.tools.training_tools import _ensure_experiment

    create_experiment("stopped_mid", {"seed": 7})
    update_status("stopped_mid", "running")
    log_metrics("stopped_mid", 1, {"val_loss": 0.9})
    log_metrics("stopped_mid", 2, {"val_loss": 0.4})
    update_status("stopped_mid", "canceled")
    metrics_before = list(ts.read_log(exp.metrics_key("stopped_mid")).records)

    eid, _out_dir = _ensure_experiment("stopped_mid", {"seed": 7}, None,
                                       resume_from="out/checkpoint_epoch_2.pt",
                                       output_base="out",
                                       launched_by={"launcher": "process"})

    assert eid.startswith("stopped_mid_run_") and eid != "stopped_mid"
    assert list(ts.read_log(exp.metrics_key("stopped_mid")).records) == metrics_before
    assert ts.read_log(exp.metrics_key(eid)).records == []
    assert ts.read(exp.lineage_key(eid))["parent_experiment"] == "stopped_mid"
