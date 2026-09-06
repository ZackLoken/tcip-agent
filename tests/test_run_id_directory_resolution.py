"""Resolving a run id to the experiment directory that actually produced it.

The directory-name shortcuts (an exact match, then the fresh-id relaunch suffix) are guesses;
the stamped ``status.json["run_id"]`` is the fact. When more than one directory name fits the
suffix pattern, resolution has to come from the stamp, otherwise a run's epochs, heartbeat and
reconstructed status are attributed to a different experiment. A run id nothing stamped resolves
to nothing at all rather than to whichever directory happens to be nearby.
"""

from __future__ import annotations

from pathlib import Path


def _experiment(experiment_id: str, run_id: str, output_dir: str) -> None:
    from tcip_mcp.experiments import create_experiment, stamp_run_identity

    create_experiment(experiment_id, {"model_source": {"builder": "my_models:bud_det"}})
    stamp_run_identity(experiment_id, run_id, output_dir, launched_by={"launcher": "process"})


def test_ambiguous_relaunch_suffix_resolves_through_the_stamped_run_id(tmp_path):
    from tcip_mcp import experiments as exp
    from tcip_mcp.experiments import resolve_experiment_dir_for_run

    run_id = "run_20260114_7f3c"
    decoy = f"alpha_{run_id}"
    stamped = f"zeta_{run_id}"
    _experiment(decoy, "run_20251203_11ab", str(tmp_path / "runs" / "alpha"))
    _experiment(stamped, run_id, str(tmp_path / "runs" / "zeta"))

    # The candidates the ambiguous suffix match actually walks, from the store the resolver
    # itself reads, not a directory listing the store's own backend need not keep.
    candidates = exp.experiment_ids_with_status(None)
    assert sorted(name for name in candidates if name.endswith(f"_{run_id}")) == [decoy, stamped]

    resolved = resolve_experiment_dir_for_run(run_id)
    assert resolved is not None
    assert resolved.name == stamped


def test_reconstructed_run_attributes_its_epochs_to_the_stamped_experiment(tmp_path):
    from tcip_mcp.experiments import log_metrics, reconstruct_run_status, update_status

    run_id = "run_20260114_7f3c"
    decoy = f"alpha_{run_id}"
    stamped = f"zeta_{run_id}"
    _experiment(decoy, "run_20251203_11ab", str(tmp_path / "runs" / "alpha"))
    _experiment(stamped, run_id, str(tmp_path / "runs" / "zeta"))

    update_status(decoy, "running")
    log_metrics(decoy, 1, {"val_map50": 0.10})
    log_metrics(decoy, 2, {"val_map50": 0.12})

    update_status(stamped, "running")
    for epoch, score in ((5, 0.55), (6, 0.61), (7, 0.63)):
        log_metrics(stamped, epoch, {"val_map50": score})

    result = reconstruct_run_status(run_id)
    assert result is not None
    assert result["experiment_id"] == stamped
    assert result["run_id"] == run_id
    assert result["current_epoch"] == 7
    assert result["output_dir"] == str(tmp_path / "runs" / "zeta")


def test_reconstructed_run_reports_the_stamped_output_dir_not_the_experiment_dir(tmp_path):
    """A run's artifact directory is computed separately from its experiment directory and only
    coincides with it by convention, so a custom-named experiment (pre-created before any run id
    existed) has to report the directory its launch actually stamped."""
    from tcip_mcp.experiments import (
        experiments_dir,
        log_metrics,
        reconstruct_run_status,
        update_status,
    )

    eid = "exp-001-currant-bud-det"
    run_id = "run_20260114_9d21"
    output_dir = tmp_path / "training_runs" / run_id
    output_dir.mkdir(parents=True)
    _experiment(eid, run_id, str(output_dir))
    update_status(eid, "running")
    log_metrics(eid, 6, {"val_map50": 0.48})
    update_status(eid, "completed")

    result = reconstruct_run_status(run_id)
    assert result is not None
    assert result["experiment_id"] == eid
    assert result["status"] == "completed"
    assert result["current_epoch"] == 6
    assert Path(result["output_dir"]) != experiments_dir() / eid
    assert result["output_dir"] == str(output_dir)


def test_run_id_no_directory_stamped_resolves_to_nothing(tmp_path):
    from tcip_mcp.experiments import reconstruct_run_status, resolve_experiment_dir_for_run

    _experiment("exp-002-currant-cluster-det", "run_20260101_aa01", str(tmp_path / "runs" / "a"))
    _experiment("exp-003-currant-cluster-det", "run_20260102_bb02", str(tmp_path / "runs" / "b"))

    assert resolve_experiment_dir_for_run("run_20260113_never") is None
    assert reconstruct_run_status("run_20260113_never") is None
