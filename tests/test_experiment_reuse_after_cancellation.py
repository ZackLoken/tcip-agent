"""A cancelled experiment is a finished record, not a slot to relaunch into.

Cancellation is a terminal outcome the audit log distinguishes from failure, so the record of what
a cancelled run did (its config, its status, its metrics) has to survive the next launch that names
the same id. ``_ensure_experiment`` mints a fresh parented id instead of reopening it.
"""

from __future__ import annotations

import json
from pathlib import Path


def _experiments(root: Path) -> Path:
    return root / ".tcip" / "experiments"


def test_relaunch_into_a_cancelled_id_mints_a_fresh_parented_id(tmp_path: Path) -> None:
    """A cancelled experiment that never logged a metric is still finished: the relaunch gets its
    own id and the cancelled record keeps the config and status it was cancelled with."""
    from tcip_mcp.experiments import create_experiment, update_status
    from tcip_mcp.tools.training_tools import _ensure_experiment

    create_experiment("stopped", {"optimizer": {"head_lr": 0.001}}, data_source="imgs_v1")
    update_status("stopped", "running")
    update_status("stopped", "cancelled")
    cancelled_dir = _experiments(tmp_path) / "stopped"
    status_before = (cancelled_dir / "status.json").read_text()

    eid = _ensure_experiment("stopped", {"optimizer": {"head_lr": 0.05}}, "imgs_v2",
                             resume_from="", run_id="run_c1", output_dir="out/c1")

    assert eid == "stopped_run_c1"
    assert json.loads((cancelled_dir / "config.json").read_text()) == {
        "optimizer": {"head_lr": 0.001}}
    assert (cancelled_dir / "status.json").read_text() == status_before

    fresh_dir = _experiments(tmp_path) / "stopped_run_c1"
    assert json.loads((fresh_dir / "lineage.json").read_text())["parent_experiment"] == "stopped"
    fresh_status = json.loads((fresh_dir / "status.json").read_text())
    assert fresh_status["run_id"] == "run_c1"
    assert fresh_status["output_dir"] == "out/c1"


def test_resuming_from_a_cancelled_runs_checkpoint_does_not_reopen_its_record(
        tmp_path: Path) -> None:
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
    cancelled_dir = _experiments(tmp_path) / "stopped_mid"
    metrics_before = (cancelled_dir / "metrics.jsonl").read_text()

    eid = _ensure_experiment("stopped_mid", {"seed": 7}, None,
                             resume_from="out/checkpoint_epoch_2.pt", run_id="run_c2",
                             output_dir="out/c2")

    assert eid == "stopped_mid_run_c2"
    assert (cancelled_dir / "metrics.jsonl").read_text() == metrics_before
    assert not (_experiments(tmp_path) / "stopped_mid_run_c2" / "metrics.jsonl").exists()
    assert json.loads(
        (_experiments(tmp_path) / "stopped_mid_run_c2" / "lineage.json").read_text()
    )["parent_experiment"] == "stopped_mid"
