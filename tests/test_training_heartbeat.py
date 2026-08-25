"""A training run reconstructed from experiment records reads as 'running' while its
heartbeat is fresh (still training in another process, e.g. the MCP agent) and only
'interrupted' once the heartbeat goes stale. Prevents the GUI mislabelling an
agent-launched run as dead and inviting a duplicate launch."""

from datetime import datetime, timedelta, timezone

import tcip_store as ts


def test_heartbeat_fresh_helper():
    from tcip_mcp.experiments import derived_state
    from tcip_mcp.tools.training_tools import TCIP_HEARTBEAT_STALE_SECONDS

    now = datetime.now(timezone.utc)
    fresh = {"state": "running", "heartbeat": now.isoformat()}
    assert derived_state(fresh, TCIP_HEARTBEAT_STALE_SECONDS) == "running"

    almost_stale = {"state": "running", "heartbeat": (
        now - timedelta(seconds=TCIP_HEARTBEAT_STALE_SECONDS - 30)).isoformat()}
    assert derived_state(almost_stale, TCIP_HEARTBEAT_STALE_SECONDS) == "running"

    stale = {"state": "running", "heartbeat": (
        now - timedelta(seconds=TCIP_HEARTBEAT_STALE_SECONDS + 60)).isoformat()}
    assert derived_state(stale, TCIP_HEARTBEAT_STALE_SECONDS) == "interrupted"

    assert derived_state({"state": "running", "heartbeat": None},
                         TCIP_HEARTBEAT_STALE_SECONDS) == "interrupted"
    assert derived_state({"state": "running", "heartbeat": "not-a-timestamp"},
                         TCIP_HEARTBEAT_STALE_SECONDS) == "interrupted"


def test_reconstructed_run_running_vs_interrupted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # EXPERIMENTS_DIR is .tcip/experiments (cwd-relative)
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    from tcip_mcp.experiments import create_experiment, log_metrics, status_key, update_status
    from tcip_mcp.tools.training_tools import list_training_runs

    # Live run: marked running, heartbeat refreshed by a fresh metric log.
    create_experiment("live", {"model_source": {"builder": "x:y"}}, data_source="imgs")
    update_status("live", "running")
    log_metrics("live", 1, {"val_map50": 0.5})

    # Dead run: running state but a forced-stale heartbeat (process gone).
    create_experiment("dead", {"model_source": {"builder": "x:y"}}, data_source="imgs")
    update_status("dead", "running")
    key = status_key("dead")
    with ts.transaction(key) as txn:
        s = txn.read(key)
        s["heartbeat"] = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        txn.write(key, s)

    by_id = {r["run_id"]: r for r in list_training_runs()["runs"]}
    assert by_id["live"]["status"] == "running"
    assert by_id["live"]["external"] is True
    assert by_id["dead"]["status"] == "interrupted"
