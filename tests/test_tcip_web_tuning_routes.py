"""Tests for the Tuning routes: launching a sweep, reading sweeps/trials off disk, and the
live-monitoring surfaces (Ray's dashboard, per-sweep and per-trial TensorBoards)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_web.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def hpo_root(tmp_path, monkeypatch) -> Path:
    """Point the platform state root at a tmp dir so the routes read this test's sweeps."""
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    root = tmp_path / ".tcip" / "hpo"
    root.mkdir(parents=True)
    return root


def _write_sweep(root: Path, study: str, **manifest_fields) -> Path:
    sweep = root / study
    sweep.mkdir(parents=True, exist_ok=True)
    manifest = {"study_name": study, "status": "running", "n_trials": 2, **manifest_fields}
    (sweep / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return sweep


def _write_trial(sweep: Path, trial_id: str, *, metrics: list[dict] | None = None,
                 params: dict | None = None) -> Path:
    trial = sweep / f"trial_{trial_id}"
    trial.mkdir(parents=True, exist_ok=True)
    if params is not None:
        (trial / "resolved_config.json").write_text(
            json.dumps({"training": {}, "trial_params": params, "unconsumed_params": []}),
            encoding="utf-8")
    if metrics is not None:
        (trial / "metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in metrics), encoding="utf-8")
    return trial


def test_list_sweeps_empty(client: TestClient) -> None:
    resp = client.get("/api/tuning/sweeps")
    assert resp.status_code == 200
    body = resp.json()
    assert "sweeps" in body


def test_get_sweep_404(client: TestClient) -> None:
    resp = client.get("/api/tuning/sweeps/nope")
    assert resp.status_code == 404


def test_launch_creates_sweep_then_listed(client: TestClient) -> None:
    resp = client.post(
        "/api/tuning/launch",
        json={
            "base_config": {"model_source": {"builder": "x:y"}, "data": {}, "training": {}},
            "param_space": {"training.batch_size": [2, 4]},
            "n_trials": 1,
            "output_dir": "",
            "search_alg": "random",
            "scheduler": "asha",
        },
    )
    assert resp.status_code == 200
    sweep_id = resp.json()["sweep_id"]
    assert sweep_id.startswith("hpo-")
    # The background thread will fail fast because config is bogus; we only
    # assert the registry got the entry.
    listing = client.get("/api/tuning/sweeps").json()
    assert any(s["sweep_id"] == sweep_id for s in listing["sweeps"])


def test_list_sweeps_finds_a_sweep_that_only_exists_on_disk(client, hpo_root) -> None:
    """An agent-launched sweep never touches this process, so its manifest is the listing."""
    _write_sweep(hpo_root, "hpo_abc12345")

    sweeps = client.get("/api/tuning/sweeps").json()["sweeps"]
    entry = next(s for s in sweeps if s["sweep_id"] == "hpo_abc12345")
    assert entry["status"] == "running"
    assert entry["external"] is True


def test_a_live_sweep_is_not_listed_twice_by_its_own_manifest(client, hpo_root, monkeypatch) -> None:
    from tcip_web.routes import tuning

    _write_sweep(hpo_root, "hpo_dup00001", status="completed")
    monkeypatch.setitem(tuning._sweeps, "hpo_dup00001",
                        tuning.HPOJob(sweep_id="hpo_dup00001", status="running"))

    sweeps = client.get("/api/tuning/sweeps").json()["sweeps"]
    matching = [s for s in sweeps if s["sweep_id"] == "hpo_dup00001"]
    assert len(matching) == 1
    assert matching[0]["status"] == "running"  # the live job wins over its manifest


def test_get_sweep_reads_the_manifest_when_the_sweep_is_not_in_memory(client, hpo_root) -> None:
    _write_sweep(hpo_root, "hpo_done0001", status="completed",
                 result={"best_params": {"lr": 0.01}, "best_value": 0.2})

    body = client.get("/api/tuning/sweeps/hpo_done0001").json()
    assert body["status"] == "completed"
    assert body["result"]["best_params"] == {"lr": 0.01}
    assert body["manifest"]["study_name"] == "hpo_done0001"


def test_list_trials_reports_platform_trial_dirs_only(client, hpo_root) -> None:
    sweep = _write_sweep(hpo_root, "hpo_trials01")
    _write_trial(sweep, "aaa_00000", metrics=[{"epoch": 1}], params={"lr": 0.01})
    _write_trial(sweep, "bbb_00001", params={"lr": 0.5})
    (sweep / "trainable_ccc_00002_0_lr=0.1_2026-01-01_00-00-00").mkdir()

    trials = client.get("/api/tuning/sweeps/hpo_trials01/trials").json()["trials"]
    by_id = {t["trial_id"]: t for t in trials}
    assert set(by_id) == {"aaa_00000", "bbb_00001"}  # Ray's own trial dir is not one of ours
    assert by_id["aaa_00000"]["has_metrics"] is True
    assert by_id["aaa_00000"]["params"] == {"lr": 0.01}
    assert by_id["bbb_00001"]["has_metrics"] is False


def test_get_trial_metrics_returns_every_row(client, hpo_root) -> None:
    sweep = _write_sweep(hpo_root, "hpo_metrics1")
    _write_trial(sweep, "aaa_00000", metrics=[{"epoch": 1, "val_loss": 0.5},
                                              {"epoch": 2, "val_loss": 0.4}])

    body = client.get("/api/tuning/sweeps/hpo_metrics1/trials/aaa_00000/metrics").json()
    assert body["exists"] is True
    assert [row["val_loss"] for row in body["metrics"]] == [0.5, 0.4]


def test_trials_of_an_unknown_sweep_are_a_404(client, hpo_root) -> None:
    assert client.get("/api/tuning/sweeps/hpo_missing/trials").status_code == 404


def test_a_trial_id_cannot_walk_out_of_its_sweep(client, hpo_root) -> None:
    """A trial id arrives as a path segment, so a name carrying a separator is refused rather
    than resolved into a log somewhere else. The ordinary id is still served."""
    sweep = _write_sweep(hpo_root, "hpo_walk0001")
    _write_trial(sweep, "aaa_00000", metrics=[{"epoch": 1, "val_loss": 0.5}])
    outside = hpo_root.parent / "elsewhere.jsonl"
    outside.write_text(json.dumps({"epoch": 99}) + "\n", encoding="utf-8")

    from fastapi import HTTPException

    from tcip_web.routes import tuning

    with pytest.raises(HTTPException) as exc:
        tuning.get_trial_metrics("hpo_walk0001", "../../elsewhere")
    assert exc.value.status_code == 400

    served = client.get("/api/tuning/sweeps/hpo_walk0001/trials/aaa_00000/metrics").json()
    assert served["exists"] is True
    assert [row["epoch"] for row in served["metrics"]] == [1]


def test_a_sweep_id_cannot_walk_out_of_the_hpo_root(hpo_root) -> None:
    from fastapi import HTTPException

    from tcip_web.routes import tuning

    with pytest.raises(HTTPException) as exc:
        tuning._read_manifest("../../elsewhere")
    assert exc.value.status_code == 400


@pytest.fixture
def tb_launches(monkeypatch) -> list[tuple[str, str]]:
    """Record what the routes hand ``launch_tensorboard`` instead of starting a real one."""
    calls: list[tuple[str, str]] = []

    def fake_launch(logdir: str, run_id: str | None = None) -> dict:
        calls.append((logdir, run_id or ""))
        return {"url": "http://localhost:6006", "port": 6006, "pid": 1, "logdir": logdir}

    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", fake_launch
    )
    return calls


def test_ray_dashboard_is_null_when_no_cluster_is_up(client, hpo_root) -> None:
    assert client.get("/api/tuning/ray-dashboard").json() == {"url": None}


def test_ray_dashboard_serves_the_url_the_process_that_started_ray_wrote(
    client, hpo_root, tmp_path
) -> None:
    """The sweep's own process records the URL; this one only reads it back."""
    from tcip_store import store

    from tcip_mcp.pipelines.training.hpo import ray_dashboard_key

    store.replace(ray_dashboard_key(),
                  {"url": "http://127.0.0.1:8265", "pid": os.getpid(),
                   "started_at": "2026-01-01T00:00:00+00:00"})

    assert client.get("/api/tuning/ray-dashboard").json() == {"url": "http://127.0.0.1:8265"}


def test_sweep_tensorboard_launches_over_a_clean_named_trial_view(
    client, hpo_root, tb_launches
) -> None:
    """Not the sweep directory itself: TensorBoard would read each trial's run name off its
    nested ``trial_<id>/tensorboard`` leaf otherwise, showing ``trial_<id>\\tensorboard`` in its
    own run picker instead of ``trial_<id>``."""
    sweep = _write_sweep(hpo_root, "hpo_tbsweep1")
    trial = _write_trial(sweep, "aaa_00000", metrics=[{"epoch": 1}])
    (trial / "tensorboard").mkdir()
    (trial / "tensorboard" / "marker.txt").write_text("x", encoding="utf-8")

    resp = client.post("/api/tuning/sweeps/hpo_tbsweep1/tensorboard")
    assert resp.status_code == 200
    assert resp.json()["url"] == "http://localhost:6006"
    (logdir, key), = tb_launches
    view = Path(logdir)
    assert view != sweep.resolve()
    assert key == "sweep_hpo_tbsweep1"

    link = view / "trial_aaa_00000"
    assert link.is_dir()
    assert (link / "marker.txt").is_file()  # resolves through to the real trial's own dir


def test_sweep_tensorboard_view_gains_a_link_for_a_trial_that_appears_later(
    client, hpo_root, tb_launches
) -> None:
    """A sweep still running writes new trial dirs after the first launch call; the view has
    to pick each one up rather than being built once and left stale."""
    sweep = _write_sweep(hpo_root, "hpo_tbsweep2")
    trial_a = _write_trial(sweep, "aaa_00000")
    (trial_a / "tensorboard").mkdir()

    resp = client.post("/api/tuning/sweeps/hpo_tbsweep2/tensorboard")
    view = Path(resp.json()["logdir"])
    assert sorted(p.name for p in view.iterdir()) == ["trial_aaa_00000"]

    trial_b = _write_trial(sweep, "bbb_00001")
    (trial_b / "tensorboard").mkdir()
    client.post("/api/tuning/sweeps/hpo_tbsweep2/tensorboard")
    assert sorted(p.name for p in view.iterdir()) == ["trial_aaa_00000", "trial_bbb_00001"]


def test_trial_tensorboard_launches_over_that_trial_s_own_logdir(
    client, hpo_root, tb_launches
) -> None:
    sweep = _write_sweep(hpo_root, "hpo_tbtrial1")
    trial = _write_trial(sweep, "aaa_00000", metrics=[{"epoch": 1}])

    resp = client.post("/api/tuning/sweeps/hpo_tbtrial1/trials/aaa_00000/tensorboard")
    assert resp.status_code == 200
    (logdir, key), = tb_launches
    assert Path(logdir) == (trial / "tensorboard").resolve()
    assert key == "sweep_hpo_tbtrial1_trial_aaa_00000"


def test_stopping_a_trial_tensorboard_uses_that_trial_s_own_key(
    client, hpo_root, monkeypatch
) -> None:
    """Trials share a bounded port range, so each one's TensorBoard must be stoppable alone."""
    stopped: list[str] = []

    def fake_stop(run_id: str | None = None, logdir: str | None = None) -> dict:
        stopped.append(run_id or "")
        return {"status": "stopped", "pid": 7}

    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.stop_tensorboard", fake_stop
    )
    _write_trial(_write_sweep(hpo_root, "hpo_tbstop01"), "aaa_00000")

    resp = client.post("/api/tuning/sweeps/hpo_tbstop01/trials/aaa_00000/tensorboard/stop")
    assert resp.status_code == 200
    assert resp.json()["status"] == "stopped"
    assert stopped == ["sweep_hpo_tbstop01_trial_aaa_00000"]


def test_tensorboard_for_an_unknown_sweep_is_a_404(client, hpo_root, tb_launches) -> None:
    for url in ("/api/tuning/sweeps/hpo_missing/tensorboard",
                "/api/tuning/sweeps/hpo_missing/trials/aaa_00000/tensorboard"):
        resp = client.post(url)
        assert resp.status_code == 404
        assert "hpo_missing" in resp.json()["detail"]
    assert tb_launches == []
