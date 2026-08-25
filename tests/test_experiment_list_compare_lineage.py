"""Experiment-tool coverage (list / compare / lineage)."""

import pytest


@pytest.mark.parametrize("backend_name", ["file", "sqlite"])
def test_a_created_experiment_is_listed_whichever_backend_holds_its_record(
    tmp_path, monkeypatch, backend_name
):
    """What names an experiment is a status record, not a directory some backend happens to make.

    The listing and the run resolver have to answer over the same set: an experiment the resolver
    finds and the listing omits is a run the breeder cannot see in the GUI's experiment list while
    the tools resolve it fine.
    """
    import tcip_store as ts
    from tcip_store.binding import BACKEND_ENV, bind_default

    monkeypatch.setenv(BACKEND_ENV, backend_name)
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    backend = bind_default()
    try:
        from tcip_mcp.experiments import (
            create_experiment,
            list_experiments,
            resolve_experiment_for_run,
        )

        create_experiment("e1", {"model_source": {"builder": "my_models:fcos_det"}})

        assert resolve_experiment_for_run("e1") == "e1"
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
    update_lineage("e1", model_weights="w.pt")
    create_experiment("e2", {"model_source": {"builder": "my_models:fcos_det"}})

    assert {e["experiment_id"] for e in list_experiments()} == {"e1", "e2"}

    cmp = compare_experiments(["e1", "e2", "missing"])
    assert cmp["count"] == 3
    e1 = next(c for c in cmp["experiments"] if c["experiment_id"] == "e1")
    assert e1["model"] == "my_models:tv_resnet50_det" and e1["last_logged_metrics"]["map50"] == 0.6
    assert any("error" in c for c in cmp["experiments"])   # the missing experiment is reported

    lin = get_experiment_lineage("e1")
    assert lin["lineage"]["model_weights"] == "w.pt"
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


def test_list_experiments_tool_carries_run_id_and_has_model_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment, stamp_run_identity
    from tcip_mcp.tools.experiment_tools import list_experiments

    create_experiment("exp-run", {"model_source": {"builder": "my_models:chestnut_burr_det"}})
    stamp_run_identity("exp-run", "run-abc", "out_dir")
    create_experiment("exp-precreated", {"a": 1})

    listed = {e["experiment_id"]: e for e in list_experiments()["experiments"]}
    assert listed["exp-run"]["run_id"] == "run-abc"
    assert listed["exp-run"]["has_model_source"] is True
    assert listed["exp-precreated"]["run_id"] is None
    assert listed["exp-precreated"]["has_model_source"] is False


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


def test_compare_experiments_reports_refused_mutations(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import compare_experiments, create_experiment, log_metrics, update_status

    create_experiment("exp-refused", {"model_source": {"builder": "my_models:chestnut_burr_det"}})
    update_status("exp-refused", "running")
    update_status("exp-refused", "completed")
    log_metrics("exp-refused", 1, {"loss": 0.9})  # refused: the record is terminal

    result = compare_experiments(["exp-refused"])
    refusals = result["experiments"][0]["refused_mutations"]
    assert len(refusals) == 1
    assert refusals[0]["arguments"]["op"] == "log_metrics"
    assert refusals[0]["arguments"]["experiment_id"] == "exp-refused"
