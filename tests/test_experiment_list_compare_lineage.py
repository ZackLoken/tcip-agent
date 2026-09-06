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
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
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


def test_list_experiments_tool_carries_run_id_and_has_model_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment, stamp_run_identity
    from tcip_mcp.tools.experiment_tools import list_experiments

    create_experiment("exp-run", {"model_source": {"builder": "my_models:chestnut_burr_det"}})
    stamp_run_identity("exp-run", "run-abc", "out_dir", launched_by={"launcher": "process"})
    create_experiment("exp-precreated", {"a": 1})

    listed = {e["experiment_id"]: e for e in list_experiments()["experiments"]}
    assert listed["exp-run"]["run_id"] == "run-abc"
    assert listed["exp-run"]["has_model_source"] is True
    assert listed["exp-precreated"]["run_id"] is None
    assert listed["exp-precreated"]["has_model_source"] is False


def test_list_experiments_launched_only_serves_the_absorbed_runs_view(tmp_path, monkeypatch):
    """launched_only=True switches list_experiments to the view the door it absorbed used to
    serve: launched runs only, keyed by run_id, in the shape _all_training_runs builds."""
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
    by_id = {r["run_id"]: r for r in launched_view["runs"]}
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


def test_compare_experiments_scans_the_audit_log_once_for_many_experiments(tmp_path, monkeypatch):
    """_index_refused_mutations reads the platform audit log once per compare call, indexed by
    experiment id, not once per experiment compared: five experiments cost one scan, not five."""
    monkeypatch.chdir(tmp_path)
    import tcip_store
    from tcip_mcp.experiments import compare_experiments, create_experiment

    ids = [f"exp-scan-{i}" for i in range(5)]
    for eid in ids:
        create_experiment(eid, {"model_source": {"builder": "my_models:chestnut_burr_det"}})

    calls = {"n": 0}
    real_read_log = tcip_store.read_log

    def _counting(*a, **k):
        calls["n"] += 1
        return real_read_log(*a, **k)

    monkeypatch.setattr(tcip_store, "read_log", _counting)

    result = compare_experiments(ids)
    assert result["count"] == 5
    assert calls["n"] == 1


def test_compare_experiments_refused_mutations_absent_when_the_read_raises(tmp_path, monkeypatch):
    """refused_mutations is absent, never an empty list, when the read of the platform audit log
    itself raises: "no refusals" and "couldn't read the log" must never be told apart by an empty
    list."""
    monkeypatch.chdir(tmp_path)
    import tcip_store
    from tcip_mcp.experiments import compare_experiments, create_experiment

    create_experiment("exp-log-unreadable", {"model_source": {"builder": "my_models:chestnut_burr_det"}})

    def _boom(*a, **k):
        raise OSError("simulated audit log read failure")

    monkeypatch.setattr(tcip_store, "read_log", _boom)

    result = compare_experiments(["exp-log-unreadable"])
    assert "refused_mutations" not in result["experiments"][0]


def test_compare_experiments_refused_mutations_absent_when_the_page_reports_corrupt_entries(
    tmp_path, monkeypatch
):
    """read_log folds a corrupt entry onto page.corrupt rather than raising, so a page that reads
    fine but reports corruption must still leave refused_mutations absent for every experiment,
    not present as an incomplete list: an unreadable log and a corrupt one are the same "can't
    trust this" fact."""
    monkeypatch.chdir(tmp_path)
    import tcip_store
    from tcip_store import LogPage
    from tcip_mcp.experiments import compare_experiments, create_experiment

    create_experiment("exp-log-corrupt", {"model_source": {"builder": "my_models:chestnut_burr_det"}})

    def _corrupt_page(*a, **k):
        return LogPage(records=[], cursor="", corrupt=(3,))

    monkeypatch.setattr(tcip_store, "read_log", _corrupt_page)

    result = compare_experiments(["exp-log-corrupt"])
    assert "refused_mutations" not in result["experiments"][0]


def test_compare_experiments_refused_mutations_absent_when_the_page_reports_version_refused(
    tmp_path, monkeypatch
):
    """A page carrying a version-refused entry is the same "can't trust this" fact as a corrupt
    one: refused_mutations must be absent, not an incomplete list built from what did decode."""
    monkeypatch.chdir(tmp_path)
    import tcip_store
    from tcip_store import LogPage
    from tcip_mcp.experiments import compare_experiments, create_experiment

    create_experiment("exp-log-version-refused",
                       {"model_source": {"builder": "my_models:chestnut_burr_det"}})

    def _version_refused_page(*a, **k):
        return LogPage(records=[], cursor="", version_refused=(2,))

    monkeypatch.setattr(tcip_store, "read_log", _version_refused_page)

    result = compare_experiments(["exp-log-version-refused"])
    assert "refused_mutations" not in result["experiments"][0]


def test_compare_experiments_finds_a_refusal_under_the_pinned_root(tmp_path, monkeypatch):
    """Coverage of the one-root invariant: a refusal update_status itself records for an
    experiment under the platform root this process is pinned to appears in refused_mutations
    when comparing that experiment, produced through the real producer (a terminal-to-terminal
    move) rather than a raw internal write. An output_dir naming a second root, even one whose
    store.db holds garbage bytes, does not make the field absent: a refusal lands only under the
    root that holds the record, so the platform log this reader scans is complete for an
    experiment that resolves under it at all, and nothing about a second root ever gets read."""
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("TCIP_STATE_ROOT", str(root))

    from tcip_mcp.experiments import (
        compare_experiments, create_experiment, stamp_run_identity, update_status,
    )

    create_experiment("exp-terminal", {"model_source": {"builder": "my_models:chestnut_burr_det"}})
    completed = update_status("exp-terminal", "completed")
    assert completed["state"] == "completed"
    refused = update_status("exp-terminal", "failed")
    assert "error" in refused

    other_root = tmp_path / "other_root"
    (other_root / ".tcip").mkdir(parents=True)
    (other_root / ".tcip" / "store.db").write_bytes(b"not a real sqlite database")
    stamp_run_identity(
        "exp-terminal", "run-1", str(other_root / ".tcip" / "experiments" / "exp-terminal"),
        launched_by={"launcher": "process"},
    )

    result = compare_experiments(["exp-terminal"])
    refusals = result["experiments"][0]["refused_mutations"]
    assert len(refusals) == 1
    assert refusals[0]["arguments"]["op"] == "update_status"


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
