"""Experiment-tool coverage (list / compare / lineage)."""

import pytest


@pytest.mark.parametrize("backend_name", ["file", "sqlite"])
def test_a_created_experiment_is_listed_whichever_backend_holds_its_record(
    tmp_path, monkeypatch, backend_name
):
    """What names an experiment is a status record, not a directory some backend happens to make.

    The listing and a direct lookup by id have to answer over the same set: an experiment
    findable by its own id but omitted from the listing is a run the breeder cannot see in the
    GUI's experiment list while the tools reach it fine.
    """
    import tcip_store as ts
    from tcip_store.binding import BACKEND_ENV, bind_default

    monkeypatch.setenv(BACKEND_ENV, backend_name)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    backend = bind_default()
    try:
        from tcip_mcp.experiments import create_experiment, experiment_exists, list_experiments

        create_experiment("e1", {"model_source": {"builder": "my_models:fcos_det"}})

        assert experiment_exists("e1")
        listed = list_experiments()
        assert [e["experiment_id"] for e in listed] == ["e1"]
        assert listed[0]["state"] == "created"
    finally:
        ts.unbind()
        backend.close()


def test_experiment_list_compare_lineage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import (
        compare_experiments,
        create_experiment,
        get_experiment_lineage,
        list_experiments,
        log_metrics,
        update_lineage,
    )

    create_experiment("e1", {"model_source": {"builder": "my_models:tv_resnet50_det"}}, data_source="imgs")
    log_metrics("e1", 1, {"map50": 0.6})
    update_lineage("e1", predictions="w.pt")
    create_experiment("e2", {"model_source": {"builder": "my_models:fcos_det"}})

    assert {e["experiment_id"] for e in list_experiments()} == {"e1", "e2"}

    cmp = compare_experiments(["e1", "e2", "missing"])
    assert cmp["count"] == 3
    e1 = next(c for c in cmp["experiments"] if c["experiment_id"] == "e1")
    assert e1["model"] == "my_models:tv_resnet50_det" and e1["last_logged_metrics"]["map50"] == 0.6
    assert any("error" in c for c in cmp["experiments"])   # the missing experiment is reported

    lin = get_experiment_lineage("e1")
    assert lin["lineage"]["predictions"] == "w.pt"
    assert "error" in get_experiment_lineage("nope")


# ── get_experiment tool: pagination + view='lineage' refusal ──────────────


def test_get_experiment_tool_pages_metrics_and_exposes_n_rows(tmp_path, monkeypatch):
    """The MCP tool accepts metrics_limit/metrics_offset under view='full'; n_rows (the row
    count) is the paging bound, always present alongside n_epochs."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment, log_metrics
    from tcip_mcp.tools.experiment_tools import get_experiment

    create_experiment("exp-paged", {"model_source": {"builder": "my_models:chestnut_burr_det"}})
    for e in range(5):
        log_metrics("exp-paged", e, {"loss": float(e)})

    full = get_experiment("exp-paged")
    assert full["n_rows"] == 5 and full["n_epochs"] == 5

    page = get_experiment("exp-paged", metrics_limit=2, metrics_offset=1)
    assert page["n_rows"] == 5
    assert [r["epoch"] for r in page["metrics"]] == [1, 2]


def test_get_experiment_tool_lineage_view_admits_defaults_refuses_pagination(tmp_path, monkeypatch):
    """A rail must admit valid work: the ordinary view='lineage' call (no pagination args)
    still succeeds; a non-default pagination arg under that view is refused as meaningless."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.tools.experiment_tools import get_experiment

    create_experiment("exp-lineage", {"model_source": {"builder": "my_models:currant_cluster_det"}},
                      data_source="imgs")

    ok = get_experiment("exp-lineage", view="lineage")
    assert "error" not in ok

    refused = get_experiment("exp-lineage", view="lineage", metrics_limit=3)
    assert "error" in refused
    refused_offset = get_experiment("exp-lineage", view="lineage", metrics_offset=2)
    assert "error" in refused_offset


# ── list_experiments MCP tool ──────────────────────────────────────────────


def test_list_experiments_tool_carries_has_model_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment, stamp_run_identity
    from tcip_mcp.tools.experiment_tools import list_experiments

    create_experiment("exp-run", {"model_source": {"builder": "my_models:fcos_det"}})
    stamp_run_identity("exp-run", "out_dir", launched_by={"launcher": "process"})
    create_experiment("exp-precreated", {"a": 1})

    listed = {e["experiment_id"]: e for e in list_experiments()["experiments"]}
    assert listed["exp-run"]["has_model_source"] is True
    assert listed["exp-precreated"]["has_model_source"] is False
    assert "run_id" not in listed["exp-run"]


def test_list_experiments_launched_only_serves_the_absorbed_runs_view(tmp_path, monkeypatch):
    """launched_only=True switches list_experiments to the absorbed door's view: launched runs
    only, keyed by experiment_id, in the shape _all_training_runs builds."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import create_experiment, update_status
    from tcip_mcp.tools.experiment_tools import list_experiments
    from tcip_mcp.tools.training_tools import _all_training_runs

    create_experiment("exp-launched-view", {"model_source": {"builder": "my_models:chestnut_burr_det"}})
    update_status("exp-launched-view", "running")
    create_experiment("exp-not-a-run", {"a": 1})

    default_view = list_experiments()
    assert "experiments" in default_view and "runs" not in default_view

    launched_view = list_experiments(launched_only=True)
    assert launched_view == {"runs": _all_training_runs(read_progress=True)}
    by_id = {r["experiment_id"]: r for r in launched_view["runs"]}
    assert "exp-launched-view" in by_id
    assert "exp-not-a-run" not in by_id


# ── compare_experiments: derived state, log lock, last row, post-end rows, refusals ────


def test_compare_experiments_reports_lock_last_row_and_post_end_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import tcip_store as ts
    from tcip_mcp.experiments import (
        compare_experiments, create_experiment, log_metrics, metrics_key, update_status,
    )

    create_experiment("exp-locked", {"model_source": {"builder": "my_models:chestnut_burr_det"}})
    update_status("exp-locked", "running")
    log_metrics("exp-locked", 0, {"loss": 0.5})
    update_status("exp-locked", "completed")

    result = compare_experiments(["exp-locked"])
    c = result["experiments"][0]
    assert c["recorded_state"] == "completed"
    assert c["state"] == "completed"
    assert c["log_locked"] is True
    assert c["last_logged_metrics"]["loss"] == 0.5
    assert c["rows_after_end"] == 0
    assert c["n_epochs"] == 1 and c["n_rows"] == 1

    # An outside writer's row landing with a later timestamp than the terminal mark: the field's
    # own residual, not the mutation lock's job to catch (log_metrics itself would refuse).
    ts.append(metrics_key("exp-locked"),
             {"epoch": 1, "timestamp": "2099-01-01T00:00:00+00:00", "loss": 0.1})
    result2 = compare_experiments(["exp-locked"])
    assert result2["experiments"][0]["rows_after_end"] == 1


def test_compare_experiments_cancelled_run_is_not_log_locked(tmp_path, monkeypatch):
    """A rail must admit valid work: the lock admits rows on a cancelled record, so log_locked
    reads False there even though no production flow appends to one."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import compare_experiments, create_experiment, update_status

    create_experiment("exp-cancelled-cmp", {"model_source": {"builder": "my_models:chestnut_burr_det"}})
    update_status("exp-cancelled-cmp", "running")
    update_status("exp-cancelled-cmp", "cancelled")

    result = compare_experiments(["exp-cancelled-cmp"])
    assert result["experiments"][0]["log_locked"] is False


def test_compare_experiments_stale_heartbeat_compares_interrupted(tmp_path, monkeypatch):
    """A crashed run's recorded state still reads "running", but the derived state a caller
    should trust reads "interrupted" once the heartbeat goes stale."""
    monkeypatch.chdir(tmp_path)
    from datetime import datetime, timedelta, timezone

    import tcip_store as ts
    from tcip_mcp.experiments import compare_experiments, create_experiment, status_key, update_status

    create_experiment("exp-stale", {"model_source": {"builder": "my_models:chestnut_burr_det"}})
    update_status("exp-stale", "running")
    key = status_key("exp-stale")
    with ts.transaction(key) as txn:
        s = txn.read(key)
        s["heartbeat"] = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        txn.write(key, s)

    result = compare_experiments(["exp-stale"])
    c = result["experiments"][0]
    assert c["recorded_state"] == "running"
    assert c["state"] == "interrupted"


def test_compare_experiments_reads_no_refusal_history(tmp_path, monkeypatch):
    """A refused write changes nothing and leaves no line, so the comparison has no refusal
    history to report and carries no field for one."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import compare_experiments, create_experiment, log_metrics, update_status

    create_experiment("exp-refused", {"model_source": {"builder": "my_models:chestnut_burr_det"}})
    update_status("exp-refused", "running")
    update_status("exp-refused", "completed")
    assert "error" in log_metrics("exp-refused", 1, {"loss": 0.9})

    assert "refused_mutations" not in compare_experiments(["exp-refused"])["experiments"][0]


def test_compare_experiments_running_with_fresh_heartbeat(tmp_path, monkeypatch):
    """A rail must admit valid work: a launched, running run with a fresh heartbeat compares
    state="running", unlocked, with no rows-after-end count since it hasn't ended."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import compare_experiments, create_experiment, update_status

    create_experiment("exp-fresh-running", {"model_source": {"builder": "my_models:chestnut_burr_det"}})
    update_status("exp-fresh-running", "running")

    result = compare_experiments(["exp-fresh-running"])
    c = result["experiments"][0]
    assert c["state"] == "running"
    assert c["log_locked"] is False
    assert c["rows_after_end"] is None


def test_compare_experiments_never_launched_reports_recorded_state(tmp_path, monkeypatch):
    """A pre-created experiment never launched carries no heartbeat; compare must report its own
    recorded state ("created") rather than deriving "interrupted" from the absent heartbeat, which
    would misreport a run that never started as one that crashed."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import compare_experiments, create_experiment

    create_experiment("exp-never-launched", {"a": 1})

    result = compare_experiments(["exp-never-launched"])
    c = result["experiments"][0]
    assert c["recorded_state"] == "created"
    assert c["state"] == "created"


def test_compare_experiments_rows_after_end_compares_instants_across_offsets(tmp_path, monkeypatch):
    """rows_after_end parses each row's timestamp (and the record's own ``ended``) as an instant
    and compares strictly-after, so a row stamped in a different UTC offset, or a bare "Z", still
    compares on the instant it actually names rather than on its ISO text; a row whose timestamp
    isn't a parseable string (a bespoke loop's own integer counter, say) is skipped, not raised on."""
    monkeypatch.chdir(tmp_path)
    import tcip_store as ts
    from tcip_mcp.experiments import (
        compare_experiments, create_experiment, metrics_key, status_key, update_status,
    )

    create_experiment("exp-instants", {"model_source": {"builder": "my_models:chestnut_burr_det"}})
    update_status("exp-instants", "running")
    update_status("exp-instants", "completed")

    ended = "2024-01-01T12:00:00+00:00"
    key = status_key("exp-instants")
    with ts.transaction(key) as txn:
        s = txn.read(key)
        s["ended"] = ended
        txn.write(key, s)

    mkey = metrics_key("exp-instants")
    rows = [
        {"epoch": 1, "timestamp": "2024-01-01T07:00:01-06:00", "loss": 0.1},  # 13:00:01 UTC: after
        {"epoch": 2, "timestamp": "2024-01-01T16:59:59+05:00", "loss": 0.2},  # 11:59:59 UTC: before
        {"epoch": 3, "timestamp": "2024-01-01T11:59:59Z", "loss": 0.3},       # before
        {"epoch": 4, "timestamp": "2024-01-01T12:00:00+00:00", "loss": 0.4},  # same instant: not after
        {"epoch": 5, "timestamp": "2024-01-01T12:00:01+00:00", "loss": 0.5},  # after
        {"epoch": 6, "timestamp": 1704110401, "loss": 0.6},                   # not a string: skipped
    ]
    for row in rows:
        ts.append(mkey, row)

    result = compare_experiments(["exp-instants"])
    assert result["experiments"][0]["rows_after_end"] == 2
