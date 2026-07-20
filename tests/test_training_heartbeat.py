"""L21 — a training run reconstructed from experiment records reads as 'running' while its
heartbeat is fresh (still training in another process, e.g. the MCP agent) and only
'interrupted' once the heartbeat goes stale. Prevents the GUI mislabelling an
agent-launched run as dead and inviting a duplicate launch."""

import json
from datetime import datetime, timedelta, timezone


def test_heartbeat_fresh_helper():
    from tcip_web.routes.training import _HEARTBEAT_STALE_SECONDS, _heartbeat_fresh

    now = datetime.now(timezone.utc)
    assert _heartbeat_fresh(now.isoformat())
    assert _heartbeat_fresh((now - timedelta(seconds=_HEARTBEAT_STALE_SECONDS - 30)).isoformat())
    assert not _heartbeat_fresh((now - timedelta(seconds=_HEARTBEAT_STALE_SECONDS + 60)).isoformat())
    assert not _heartbeat_fresh(None)
    assert not _heartbeat_fresh("not-a-timestamp")


def test_reconstructed_run_running_vs_interrupted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # EXPERIMENTS_DIR is .tcip/experiments (cwd-relative)
    from tcip_mcp.experiments import create_experiment, log_metrics, update_status
    from tcip_web.routes.training import _historical_training_runs

    # Live run: marked running, heartbeat refreshed by a fresh metric log.
    create_experiment("live", {"model_source": {"builder": "x:y"}}, data_source="imgs")
    update_status("live", "running")
    log_metrics("live", 1, {"val_map50": 0.5})

    # Dead run: running state but a forced-stale heartbeat (process gone).
    create_experiment("dead", {"model_source": {"builder": "x:y"}}, data_source="imgs")
    update_status("dead", "running")
    sp = tmp_path / ".tcip" / "experiments" / "dead" / "status.json"
    s = json.loads(sp.read_text())
    s["heartbeat"] = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    sp.write_text(json.dumps(s))

    by_id = {r["run_id"]: r for r in _historical_training_runs()}
    assert by_id["live"]["status"] == "running"
    assert by_id["live"]["external"] is True
    assert by_id["dead"]["status"] == "interrupted"
