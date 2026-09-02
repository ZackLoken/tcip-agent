"""A sweep's manifest carries a driver heartbeat the way a training run's status record does,
so a listing can tell a live driver from a dead one instead of trusting a recorded ``status``
verbatim (``sweep_state``, in ``training_tools.py``, calling ``experiments.derived_state`` rather
than a second copy of the freshness rule)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def test_sweep_state_helper():
    from tcip_mcp.tools.training_tools import TCIP_HEARTBEAT_STALE_SECONDS, sweep_state

    now = datetime.now(timezone.utc)
    fresh = {"status": "running", "heartbeat": now.isoformat()}
    assert sweep_state(fresh, stale_seconds=TCIP_HEARTBEAT_STALE_SECONDS) == "running"

    stale = {"status": "running", "heartbeat": (
        now - timedelta(seconds=TCIP_HEARTBEAT_STALE_SECONDS + 60)).isoformat()}
    assert sweep_state(stale, stale_seconds=TCIP_HEARTBEAT_STALE_SECONDS) == "interrupted"

    no_heartbeat = {"status": "running"}
    assert sweep_state(no_heartbeat, stale_seconds=TCIP_HEARTBEAT_STALE_SECONDS) == "interrupted"

    done = {"status": "completed", "heartbeat": None}
    assert sweep_state(done, stale_seconds=TCIP_HEARTBEAT_STALE_SECONDS) == "completed"


def test_sweep_state_driver_live_wins_over_a_stale_or_missing_heartbeat():
    """A process that can vouch for the driver directly (the Tuning route's own worker thread,
    still alive) reads "running" even the instant before its next heartbeat write lands."""
    from tcip_mcp.tools.training_tools import TCIP_HEARTBEAT_STALE_SECONDS, sweep_state

    assert sweep_state({"status": "running"}, stale_seconds=TCIP_HEARTBEAT_STALE_SECONDS,
                       driver_live=True) == "running"

    stale = {"status": "running", "heartbeat": (
        datetime.now(timezone.utc) - timedelta(seconds=TCIP_HEARTBEAT_STALE_SECONDS + 60)
    ).isoformat()}
    assert sweep_state(stale, stale_seconds=TCIP_HEARTBEAT_STALE_SECONDS, driver_live=True) == "running"


def test_sweep_state_driver_live_never_overrides_a_recorded_done_state():
    """driver_live proves the thread is alive, not that the sweep is still running: a thread
    lingering after its own terminal write must not resurrect a completed sweep as running."""
    from tcip_mcp.tools.training_tools import TCIP_HEARTBEAT_STALE_SECONDS, sweep_state

    for status in ("completed", "failed", "cancelled"):
        manifest = {"status": status}
        assert sweep_state(manifest, stale_seconds=TCIP_HEARTBEAT_STALE_SECONDS,
                           driver_live=True) == status


def test_run_hpo_stamps_a_fresh_heartbeat_at_or_after_started_at(
    tmp_path, real_hpo_base_config, monkeypatch
):
    import tcip_store as ts
    import tcip_mcp.tools.training_tools as tt

    def fake_search(**kw):
        return {"best_params": {"lr": 0.1}, "best_value": 0.25, "n_trials": 1,
                "study_name": kw["study_name"]}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    result = tt.run_hpo(base_config=real_hpo_base_config, n_trials=1, output_dir=str(tmp_path))
    manifest = ts.read(tt.sweep_manifest_key(result["study_name"], str(tmp_path)))

    started_at = datetime.fromisoformat(manifest["started_at"])
    heartbeat = datetime.fromisoformat(manifest["heartbeat"])
    assert heartbeat >= started_at
    assert manifest["status"] == "completed"


def test_run_hpo_heartbeat_thread_stops_before_the_terminal_write(
    tmp_path, real_hpo_base_config, monkeypatch
):
    """The heartbeat thread is stopped and joined before run_hpo's own terminal manifest write,
    so nothing restamps the manifest again once run_hpo has returned: a listing reading it
    afterward sees a status that stands, not one a straggling heartbeat write could revert."""
    import time

    import tcip_mcp.tools.training_tools as tt

    monkeypatch.setattr(tt, "sweep_heartbeat_seconds", lambda: 0.02)

    writes: list[dict] = []
    real_replace = tt.store.replace

    def counting_replace(key, value, **kw):
        if key.store == tt.SWEEP_MANIFEST_STORE:
            writes.append(dict(value))
        return real_replace(key, value, **kw)

    monkeypatch.setattr(tt.store, "replace", counting_replace)

    def fake_search(**kw):
        time.sleep(0.09)  # long enough for several heartbeat restamps at a 0.02s interval
        return {"best_params": {"lr": 0.1}, "best_value": 0.25, "n_trials": 1,
                "study_name": kw["study_name"]}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    tt.run_hpo(base_config=real_hpo_base_config, n_trials=1, output_dir=str(tmp_path))
    count_at_return = len(writes)
    assert count_at_return >= 3  # the initial write, at least one restamp, the terminal write
    assert writes[-1]["status"] == "completed"

    time.sleep(0.06)  # well past the heartbeat interval
    assert len(writes) == count_at_return  # nothing wrote again after run_hpo returned


def test_heartbeat_loop_survives_a_store_error_and_keeps_restamping(
    tmp_path, real_hpo_base_config, monkeypatch, caplog
):
    """A store error on one heartbeat restamp costs one beat, not the rest of the sweep: the
    loop keeps stamping afterward, and ``run_hpo`` still returns its normal result rather than
    a live sweep silently losing its heartbeat for good."""
    import threading

    import tcip_mcp.tools.training_tools as tt
    from tcip_store import StoreError

    monkeypatch.setattr(tt, "sweep_heartbeat_seconds", lambda: 0.01)

    writes: list[dict] = []
    raised_once = False
    recovered = threading.Event()
    real_replace = tt.store.replace

    def flaky_replace(key, value, **kw):
        nonlocal raised_once
        if key.store == tt.SWEEP_MANIFEST_STORE:
            writes.append(dict(value))
            if len(writes) == 2 and not raised_once:
                raised_once = True
                raise StoreError("simulated lock timeout")
            if raised_once and len(writes) >= 4:
                recovered.set()
        return real_replace(key, value, **kw)

    monkeypatch.setattr(tt.store, "replace", flaky_replace)

    def fake_search(**kw):
        assert recovered.wait(timeout=5), \
            "the heartbeat loop never restamped again after the store error"
        return {"best_params": {"lr": 0.1}, "best_value": 0.25, "n_trials": 1,
                "study_name": kw["study_name"]}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)

    with caplog.at_level("WARNING", logger=tt.logger.name):
        result = tt.run_hpo(base_config=real_hpo_base_config, n_trials=1, output_dir=str(tmp_path))

    assert result["best_value"] == 0.25  # run_hpo returned normally past the failed restamp
    manifest = tt.store.read(tt.sweep_manifest_key(result["study_name"], str(tmp_path)))
    assert manifest["status"] == "completed"
    assert any("could not restamp the heartbeat" in r.getMessage() for r in caplog.records)
