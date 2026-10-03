"""Tests for the Tuning routes: relaunching a sweep from its recorded input, reading sweeps and
trials off their directories, cancel, and the live-monitoring surfaces (Ray's dashboard,
per-sweep and per-trial TensorBoards).

Every sweep here is recorded by ``run_hyperparameter_search`` itself, only Ray Tune's search
stubbed, and every trial directory is opened by the launcher's own writer, as the sweep's trial
producer opens it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_web.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _worker_join_bound() -> float:
    """How long a test here waits on a sweep worker: a stubbed search returns at once, so the
    worker's slowest act is one record write."""
    from tcip_store.file_backend import DEFAULT_LOCK_TIMEOUT_S

    return 2 * DEFAULT_LOCK_TIMEOUT_S


@pytest.fixture(autouse=True)
def _join_sweep_workers():
    """Wait out any sweep a test here launched before that test returns."""
    yield
    from tcip_web.routes import tuning

    assert tuning.wait_for_workers(timeout_s=_worker_join_bound()) == ()


def _stub_search(monkeypatch, *, during=None) -> None:
    """Replace Ray Tune's search with one answering the sweep's log directory, first calling
    ``during(study_name, sweep_root)`` while the sweep's driver is live."""
    def fake_search(**kw):
        sweep_root = Path(kw["storage_path"]) / kw["study_name"]
        if during is not None:
            during(kw["study_name"], sweep_root)
        return str(sweep_root)

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)


def _record_sweep(project: Path, study: str, base_config: dict, **kwargs) -> dict:
    """A sweep of ``project`` that ``run_hyperparameter_search`` records over the stubbed search,
    starting no TensorBoard of its own; its answer."""
    import tcip_mcp.tools.training_tools as tt

    arguments = {"n_trials": 1, "scheduler": "none", "search_seed": 0,
                 "auto_tensorboard": False, **kwargs}
    answer = tt.run_hyperparameter_search(project, base_config=base_config, study_name=study,
                                          **arguments)
    assert "error" not in answer, answer
    return answer


def _record_trial(project: Path, sweep_root: Path, trial_id: str, *, params: dict,
                  metrics: tuple[dict, ...] = ()) -> Path:
    """A trial run directory of ``project`` under ``sweep_root`` holding ``params`` and one
    metrics row per entry of ``metrics``, resolved and opened by the launcher's own producer and
    writer over two frames of its own and logged through the envelope's own sink."""
    from tcip_mcp.pipelines.data.split_construction import resolve_run
    from tcip_mcp.tools.training_tools import open_run
    from tests._verified_checkpoint_fixtures import detection_config, log_epoch

    trial = sweep_root / f"trial_{trial_id}"
    config = detection_config(project / "trial-data" / trial_id)
    open_run(trial, config, resolve_run(config, project=project).record,
             trial_params=params)
    for epoch, row in enumerate(metrics, 1):
        log_epoch(trial, epoch, row)
    return trial


def _opened(project: Path, study: str, base_config: dict):
    """A sweep of ``project`` over ``base_config`` whose input its own writer wrote
    (``open_sweep``), not yet run."""
    from tcip_mcp.tools.training_tools import open_sweep

    opened = open_sweep(
        project, base_config, None, n_trials=1, search_alg="random", scheduler="none", grace_period=5,
        reduction_factor=3, warm_start=False, baseline_params=None, max_concurrent=1,
        resources_per_trial=None, study_name=study, split_draws=1, split_draw_seeds=None,
        search_seed=0, trial_budget=None, relaunched_from=None, actor=None)
    assert not isinstance(opened, dict), opened
    return opened


def _sweep_root(project: Path, study: str) -> Path:
    from tcip_mcp.tools.training_tools import sweep_dir

    return sweep_dir(study, project=project)


def test_list_sweeps_empty(client: TestClient, opened_project) -> None:
    resp = client.get("/api/tuning/sweeps")
    assert resp.status_code == 200
    assert resp.json() == {"sweeps": []}


def test_get_sweep_404(client: TestClient, opened_project) -> None:
    assert client.get("/api/tuning/sweeps/nope").status_code == 404


def test_a_recorded_sweep_lists_and_reads_back_from_its_directory(
    client, opened_project, real_hpo_base_config, monkeypatch,
) -> None:
    """An agent-launched sweep never touches this process: its directory is the listing, and
    the detail carries its input, its trials and what they amount to."""
    _stub_search(monkeypatch, during=lambda study, root: _record_trial(
        opened_project, root, "aaa_00000", params={"batch_size": 2}))
    _record_sweep(opened_project, "hpo_done0001", real_hpo_base_config, param_space={"batch_size": [2, 4]})

    row = next(s for s in client.get("/api/tuning/sweeps").json()["sweeps"]
               if s["sweep_id"] == "hpo_done0001")
    assert (row["status"], row["external"]) == ("completed", True)
    assert row["param_space_keys"] == ["batch_size"]

    body = client.get("/api/tuning/sweeps/hpo_done0001").json()
    assert body["status"] == "completed"
    assert [(t["trial_id"], t["params"]) for t in body["trials"]] == [
        ("aaa_00000", {"batch_size": 2})]
    assert body["outcome"]["best_params"] is None  # the trial never completed a checkpoint
    assert body["input"]["base_config"] == real_hpo_base_config
    assert body["input"]["relaunched_from"] is None


def test_a_sweep_reads_running_while_its_driver_beats_and_interrupted_once_it_stops(
    client, opened_project, real_hpo_base_config, monkeypatch,
) -> None:
    """The state is derived from the sweep's own heartbeat: fresh while its driver runs,
    interrupted once no process is left to touch it, on the listing and the detail alike."""
    from tcip_mcp import experiments

    seen: dict[str, tuple[str, str]] = {}

    def during(study, _root):
        def read() -> tuple[str, str]:
            row = next(s for s in client.get("/api/tuning/sweeps").json()["sweeps"]
                       if s["sweep_id"] == study)
            return row["status"], client.get(f"/api/tuning/sweeps/{study}").json()["status"]

        seen["live"] = read()
        stale = experiments.HEARTBEAT_STALE_SECONDS
        experiments.HEARTBEAT_STALE_SECONDS = -1.0
        try:
            seen["stale"] = read()
        finally:
            experiments.HEARTBEAT_STALE_SECONDS = stale

    _stub_search(monkeypatch, during=during)
    _record_sweep(opened_project, "hpo_beat0001", real_hpo_base_config)

    assert seen == {"live": ("running", "running"), "stale": ("interrupted", "interrupted")}


def test_a_relaunched_sweep_is_listed_once_as_this_processs_own(
        client, opened_project, real_hpo_base_config, monkeypatch) -> None:
    """A sweep this process relaunched is one directory like any other: listed once, from that
    directory, marked as not external."""
    from tcip_web.routes import tuning

    _stub_search(monkeypatch)
    _record_sweep(opened_project, "hpo_dup00001", real_hpo_base_config)
    sweep_id = client.post("/api/tuning/sweeps", json={"study_name": "hpo_dup00001", "user": "tester"}).json()[
        "sweep_id"]
    assert tuning.wait_for_workers(timeout_s=_worker_join_bound()) == ()

    matching = [s for s in client.get("/api/tuning/sweeps").json()["sweeps"]
                if s["sweep_id"] == sweep_id]
    assert len(matching) == 1
    assert matching[0]["external"] is False


def test_a_relaunch_records_its_source_and_its_row_agrees_with_the_sources(
    client, opened_project, real_hpo_base_config, monkeypatch,
) -> None:
    """A relaunch through the route opens its sweep through the one operation the tool opens one
    through: the new sweep's id is minted the tool's way, the act leaves one audit line naming
    the sweep and its source, its input names the sweep it replayed, and its row projects the
    same search shape the source's disk row does."""
    import tcip_mcp.tools.training_tools as tt
    from tcip_mcp.audit import audit_log_key
    from tcip_store import read_log
    from tcip_web.routes import tuning

    _stub_search(monkeypatch)
    _record_sweep(opened_project, "hpo_relsrc001", real_hpo_base_config, n_trials=3,
                  param_space={"lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2}})
    minted = tt.run_hyperparameter_search(
        opened_project, base_config=real_hpo_base_config, n_trials=1, scheduler="none",
        search_seed=0, auto_tensorboard=False)["study_name"]
    before = len(read_log(audit_log_key(opened_project)).records)

    resp = client.post("/api/tuning/sweeps", json={"study_name": "hpo_relsrc001", "user": "tester"})
    assert resp.status_code == 200
    sweep_id = resp.json()["sweep_id"]
    assert tuning.wait_for_workers(timeout_s=_worker_join_bound()) == ()
    assert sweep_id.split("_")[0] == minted.split("_")[0] and len(sweep_id) == len(minted)
    lines = read_log(audit_log_key(opened_project)).records[before:]
    assert [(line["tool"], line["arguments"], line["actor"]) for line in lines] == [
        ("open_sweep", {"sweep_id": sweep_id, "relaunched_from": "hpo_relsrc001"}, "user:tester")]

    body = client.get(f"/api/tuning/sweeps/{sweep_id}").json()
    assert body["input"]["relaunched_from"] == "hpo_relsrc001"
    by_id = {s["sweep_id"]: s for s in client.get("/api/tuning/sweeps").json()["sweeps"]}
    assert by_id[sweep_id]["relaunched_from"] == "hpo_relsrc001"
    fields = ("n_trials", "search_alg", "scheduler", "param_space_keys", "split_draws")
    assert {k: by_id[sweep_id][k] for k in fields} == {k: by_id["hpo_relsrc001"][k]
                                                       for k in fields}


def test_a_relaunch_replays_every_argument_its_source_recorded(
    client, opened_project, real_hpo_base_config, monkeypatch,
) -> None:
    """A relaunch reads every argument the source's input holds, not only base_config, so a
    sweep started with a non-default search shape replays that shape exactly."""
    import tcip_mcp.tools.training_tools as tt
    from tcip_web.routes import tuning

    _stub_search(monkeypatch)
    space = {"lr": {"type": "loguniform", "low": 1e-6, "high": 1e-1}}
    _record_sweep(opened_project, "hpo_fields001", real_hpo_base_config, param_space=space, n_trials=7,
                  grace_period=3, reduction_factor=4, max_concurrent=2,
                  resources_per_trial={"cpu": 2.0, "gpu": 0.5}, trial_budget=9, search_seed=713)

    captured: dict = {}
    real_open = tt.open_sweep

    def spy(project, base_config, param_space, **kwargs):
        captured.update(kwargs, base_config=base_config, param_space=param_space)
        return real_open(project, base_config, param_space, **kwargs)

    monkeypatch.setattr(tt, "open_sweep", spy)
    resp = client.post("/api/tuning/sweeps", json={"study_name": "hpo_fields001", "user": "tester"})
    assert resp.status_code == 200
    assert tuning.wait_for_workers(timeout_s=_worker_join_bound()) == ()

    assert captured["search_seed"] == 713
    assert captured["base_config"] == real_hpo_base_config
    assert captured["param_space"] == space
    assert (captured["n_trials"], captured["grace_period"], captured["reduction_factor"],
            captured["max_concurrent"], captured["trial_budget"]) == (7, 3, 4, 2, 9)
    assert captured["resources_per_trial"] == {"cpu": 2.0, "gpu": 0.5}
    assert captured["relaunched_from"] == "hpo_fields001"


def test_a_relaunch_whose_data_moved_is_refused_with_the_refusals_own_words(
    client, opened_project, real_hpo_base_config, monkeypatch,
) -> None:
    """A replay the launch refuses at preflight answers 422 with the refusal's own words before
    any sweep directory or thread exists."""
    import shutil

    from tcip_mcp.experiments import sweeps_dir

    _stub_search(monkeypatch)
    _record_sweep(opened_project, "hpo_moved0001", real_hpo_base_config)
    shutil.rmtree(real_hpo_base_config["data"]["images_dir"])
    before = sorted(p.name for p in sweeps_dir(opened_project).iterdir())

    resp = client.post("/api/tuning/sweeps", json={"study_name": "hpo_moved0001", "user": "tester"})

    assert resp.status_code == 422
    assert resp.json()["detail"]["error"]
    assert sorted(p.name for p in sweeps_dir(opened_project).iterdir()) == before


def test_relaunch_route_404s_for_an_unknown_sweep(client: TestClient, opened_project) -> None:
    assert client.post("/api/tuning/sweeps", json={"study_name": "nope", "user": "tester"}).status_code == 404


def test_the_launch_route_is_not_registered(client: TestClient, opened_project) -> None:
    """A sweep starts only from a recorded input, through ``/api/tuning/sweeps``, never from a
    client-submitted base_config and param_space."""
    resp = client.post(
        "/api/tuning/launch",
        json={"base_config": {"model_source": {"builder": "x:y"}, "data": {}},
              "param_space": {}, "n_trials": 1, "search_alg": "random", "scheduler": "asha"},
    )
    assert resp.status_code == 404


def test_list_trials_reports_the_sweeps_own_trial_directories_only(
    client, opened_project, real_hpo_base_config, monkeypatch,
) -> None:
    def during(study, root):
        _record_trial(opened_project, root, "aaa_00000", params={"lr": 0.01},
                      metrics=({"val_loss": 0.5},))
        _record_trial(opened_project, root, "bbb_00001", params={"lr": 0.5})
        (root / "trainable_ccc_00002_0_lr=0.1_2026-01-01_00-00-00").mkdir()

    _stub_search(monkeypatch, during=during)
    _record_sweep(opened_project, "hpo_trials01", real_hpo_base_config)

    trials = client.get("/api/tuning/sweeps/hpo_trials01/trials").json()["trials"]
    by_id = {t["trial_id"]: t for t in trials}
    assert set(by_id) == {"aaa_00000", "bbb_00001"}  # Ray's own trial dir is not one of ours
    assert (by_id["aaa_00000"]["has_metrics"], by_id["aaa_00000"]["params"]) == (
        True, {"lr": 0.01})
    assert by_id["bbb_00001"]["has_metrics"] is False


def test_get_trial_metrics_returns_every_row(
    client, opened_project, real_hpo_base_config, monkeypatch,
) -> None:
    _stub_search(monkeypatch, during=lambda study, root: _record_trial(
        opened_project, root, "aaa_00000", params={"lr": 0.01},
        metrics=({"val_loss": 0.5}, {"val_loss": 0.4})))
    _record_sweep(opened_project, "hpo_metrics1", real_hpo_base_config)

    body = client.get("/api/tuning/sweeps/hpo_metrics1/trials/aaa_00000/metrics").json()
    assert body["exists"] is True
    assert [row["val_loss"] for row in body["metrics"]] == [0.5, 0.4]


def test_trials_of_an_unknown_sweep_are_a_404(client, opened_project) -> None:
    assert client.get("/api/tuning/sweeps/hpo_missing/trials").status_code == 404


def test_a_trial_id_cannot_walk_out_of_its_sweep(
    client, opened_project, real_hpo_base_config, monkeypatch,
) -> None:
    """A trial id arrives as a path segment, so a name carrying a separator is refused rather
    than resolved into a log somewhere else. The ordinary id is still served."""
    from fastapi import HTTPException

    from tcip_web.routes import tuning

    _stub_search(monkeypatch, during=lambda study, root: _record_trial(
        opened_project, root, "aaa_00000", params={}, metrics=({"val_loss": 0.5},)))
    _record_sweep(opened_project, "hpo_walk0001", real_hpo_base_config)
    (opened_project / "elsewhere.jsonl").write_text(json.dumps({"epoch": 99}) + "\n", encoding="utf-8")

    with pytest.raises(HTTPException) as exc:
        tuning.get_trial_metrics("hpo_walk0001", "../../elsewhere")
    assert exc.value.status_code == 400

    served = client.get("/api/tuning/sweeps/hpo_walk0001/trials/aaa_00000/metrics").json()
    assert [row["epoch"] for row in served["metrics"]] == [1]


def test_a_sweep_id_cannot_walk_out_of_the_sweeps_directory(opened_project) -> None:
    from fastapi import HTTPException

    from tcip_web.routes import tuning

    with pytest.raises(HTTPException) as exc:
        tuning.get_sweep("../../elsewhere")
    assert exc.value.status_code == 400


def test_cancel_route_404s_for_an_unknown_sweep(client: TestClient, opened_project) -> None:
    assert client.post("/api/tuning/sweeps/nope/cancel",
                       json={"user": "tester"}).status_code == 404


def test_cancel_route_records_the_cancellation_its_trials_poll(
    client, opened_project, real_hpo_base_config, monkeypatch,
) -> None:
    """A cancel lands in the sweep's own directory, which every trial of it polls, and its line
    names the person who asked; the listing reports it."""
    from tcip_mcp.audit import audit_log_key
    from tcip_mcp.experiments import cancel_requested
    from tcip_store import read_log

    _opened(opened_project, "hpo_cancel01", real_hpo_base_config)
    before = len(read_log(audit_log_key(opened_project)).records)

    resp = client.post("/api/tuning/sweeps/hpo_cancel01/cancel", json={"user": "tester"})
    assert resp.status_code == 200
    assert resp.json()["cancel_requested"] is True
    (line,) = read_log(audit_log_key(opened_project)).records[before:]
    assert (line["tool"], line["actor"]) == ("cancel_hyperparameter_search", "user:tester")
    assert cancel_requested(_sweep_root(opened_project, "hpo_cancel01"))
    row = next(s for s in client.get("/api/tuning/sweeps").json()["sweeps"]
               if s["sweep_id"] == "hpo_cancel01")
    assert row["cancel_requested"] is True


def test_cancel_reaches_a_relaunch_before_its_first_trial(
    client, opened_project, real_hpo_base_config, monkeypatch,
) -> None:
    """A cancel that arrives once the route has answered but before the relaunch's run started
    reaches the sweep, which ends canceled before its first trial with that reason as its
    final status's error."""
    import threading

    import tcip_mcp.tools.training_tools as tt
    from tcip_web.routes import tuning

    _stub_search(monkeypatch)
    _record_sweep(opened_project, "hpo_precancel1", real_hpo_base_config)
    release = threading.Event()
    real_run = tt.run_sweep

    def held(sweep, **kwargs):
        release.wait(timeout=5)
        return real_run(sweep, **kwargs)

    monkeypatch.setattr(tt, "run_sweep", held)
    sweep_id = client.post("/api/tuning/sweeps", json={"study_name": "hpo_precancel1", "user": "tester"}).json()[
        "sweep_id"]

    cancel_resp = client.post(f"/api/tuning/sweeps/{sweep_id}/cancel", json={"user": "tester"})
    assert cancel_resp.status_code == 200 and cancel_resp.json()["cancel_requested"] is True

    release.set()
    assert tuning.wait_for_workers(timeout_s=_worker_join_bound()) == ()
    body = client.get(f"/api/tuning/sweeps/{sweep_id}").json()
    assert (body["status"], body["error"]) == ("canceled", tt._CANCEL_BEFORE_START_REASON)


def test_ray_dashboard_is_null_when_no_cluster_is_up(client, opened_project) -> None:
    assert client.get("/api/tuning/ray-dashboard").json() == {"url": None}


def test_ray_dashboard_serves_the_url_the_process_that_started_ray_wrote(client, opened_project) -> None:
    """The sweep's own process records the URL; this one only reads it back."""
    from tcip_store import store

    from tcip_mcp.pipelines.training.hpo import ray_dashboard_key

    store.replace(ray_dashboard_key(opened_project),
                  {"url": "http://127.0.0.1:8265", "pid": os.getpid()})

    assert client.get("/api/tuning/ray-dashboard").json() == {"url": "http://127.0.0.1:8265"}


def _trial_with_tensorboard(project: Path, root: Path, trial_id: str) -> Path:
    trial = _record_trial(project, root, trial_id, params={})
    (trial / "tensorboard").mkdir()
    (trial / "tensorboard" / "marker.txt").write_text("x", encoding="utf-8")
    return trial


def test_sweep_tensorboard_launches_over_a_clean_named_trial_view(
    client, opened_project, real_hpo_base_config, monkeypatch, tb_launches,
) -> None:
    """Not the sweep directory itself: TensorBoard would read each trial's run name off its
    nested ``trial_<id>/tensorboard`` leaf otherwise."""
    _stub_search(monkeypatch, during=lambda study, root: _trial_with_tensorboard(
        opened_project, root, "aaa_00000"))
    _record_sweep(opened_project, "hpo_tbsweep1", real_hpo_base_config)

    resp = client.post("/api/tuning/sweeps/hpo_tbsweep1/tensorboard", json={})
    assert resp.status_code == 200
    (logdir, key), = tb_launches
    view = Path(logdir)
    assert view != _sweep_root(opened_project, "hpo_tbsweep1").resolve()
    assert key == "sweep_hpo_tbsweep1"
    assert (view / "trial_aaa_00000" / "marker.txt").is_file()


def test_sweep_tensorboard_view_gains_a_link_for_a_trial_that_appears_later(
    client, opened_project, real_hpo_base_config, monkeypatch, tb_launches,
) -> None:
    _stub_search(monkeypatch, during=lambda study, root: _trial_with_tensorboard(
        opened_project, root, "aaa_00000"))
    _record_sweep(opened_project, "hpo_tbsweep2", real_hpo_base_config)

    view = Path(client.post("/api/tuning/sweeps/hpo_tbsweep2/tensorboard", json={}).json()[
        "logdir"])
    assert sorted(p.name for p in view.iterdir()) == ["trial_aaa_00000"]

    _trial_with_tensorboard(opened_project, _sweep_root(opened_project, "hpo_tbsweep2"),
                            "bbb_00001")
    client.post("/api/tuning/sweeps/hpo_tbsweep2/tensorboard", json={})
    assert sorted(p.name for p in view.iterdir()) == ["trial_aaa_00000", "trial_bbb_00001"]


def test_trial_tensorboard_launches_over_that_trials_own_logdir(
    client, opened_project, real_hpo_base_config, monkeypatch, tb_launches,
) -> None:
    _stub_search(monkeypatch, during=lambda study, root: _record_trial(
        opened_project, root, "aaa_00000", params={}))
    _record_sweep(opened_project, "hpo_tbtrial1", real_hpo_base_config)

    resp = client.post("/api/tuning/sweeps/hpo_tbtrial1/trials/aaa_00000/tensorboard", json={})
    assert resp.status_code == 200
    (logdir, key), = tb_launches
    assert Path(logdir) == (_sweep_root(opened_project, "hpo_tbtrial1") / "trial_aaa_00000"
                            / "tensorboard").resolve()
    assert key == "sweep_hpo_tbtrial1_trial_aaa_00000"


def test_stopping_a_trial_tensorboard_uses_that_trials_own_key(client, opened_project, monkeypatch) -> None:
    """Trials share a bounded port range, so each one's TensorBoard must be stoppable alone."""
    stopped: list[str] = []

    def fake_stop(key: str | None = None, logdir: str | None = None) -> dict:
        stopped.append(key or "")
        return {"status": "stopped", "pid": 7}

    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.stop_tensorboard", fake_stop
    )

    resp = client.post("/api/tuning/sweeps/hpo_tbstop01/trials/aaa_00000/tensorboard/stop",
                       json={})
    assert resp.status_code == 200
    assert stopped == ["sweep_hpo_tbstop01_trial_aaa_00000"]


def test_tensorboard_for_an_unknown_sweep_is_a_404(client, opened_project, tb_launches) -> None:
    for url in ("/api/tuning/sweeps/hpo_missing/tensorboard",
                "/api/tuning/sweeps/hpo_missing/trials/aaa_00000/tensorboard"):
        resp = client.post(url, json={})
        assert resp.status_code == 404
        assert "hpo_missing" in resp.json()["detail"]


def test_a_launched_sweep_is_reached_only_through_its_own_project_once_another_is_open(
    client, opened_project, real_hpo_base_config, monkeypatch, tb_launches,
) -> None:
    """A sweep this process launched runs for the project it launched for, while every HTTP
    lookup reads the open project: once another project is open the sweep is not found there,
    and reopening its own project reaches it again."""
    from tcip_mcp.project_record import read_record
    from tcip_web.routes import tuning
    from tests._web_fixtures import open_new_project

    _stub_search(monkeypatch, during=lambda study, root: _trial_with_tensorboard(
        opened_project, root, "aaa_00000"))
    _record_sweep(opened_project, "hpo_seed00003", real_hpo_base_config)
    sweep_id = client.post("/api/tuning/sweeps", json={"study_name": "hpo_seed00003", "user": "tester"}).json()[
        "sweep_id"]
    assert tuning.wait_for_workers(timeout_s=_worker_join_bound()) == ()

    open_new_project(opened_project.parent / "other")

    assert client.get(f"/api/tuning/sweeps/{sweep_id}").status_code == 404
    assert client.post(f"/api/tuning/sweeps/{sweep_id}/tensorboard", json={}).status_code == 404

    assert client.post("/api/projects/open",
                       json={"id": read_record(opened_project)["id"]}).status_code == 200
    body = client.get(f"/api/tuning/sweeps/{sweep_id}").json()
    assert (body["sweep_id"], body["status"], body["external"]) == (sweep_id, "completed", False)


def test_a_web_launched_sweep_runs_only_the_routes_own_tensorboard(
    client, opened_project, real_hpo_base_config, monkeypatch, tb_launches,
) -> None:
    """A relaunch through the route must not auto-launch a TensorBoard of its own: the route
    serves its own per-sweep view on demand."""
    from tcip_web.routes import tuning

    def during(study, root):
        _trial_with_tensorboard(opened_project, root, "aaa_00000")

    _stub_search(monkeypatch, during=during)
    _record_sweep(opened_project, "hpo_seed00004", real_hpo_base_config, auto_tensorboard=False)

    sweep_id = client.post("/api/tuning/sweeps", json={"study_name": "hpo_seed00004", "user": "tester"}).json()[
        "sweep_id"]
    assert tuning.wait_for_workers(timeout_s=_worker_join_bound()) == ()
    assert tb_launches == []

    assert client.post(f"/api/tuning/sweeps/{sweep_id}/tensorboard", json={}).status_code == 200
    (_, key), = tb_launches
    assert key == f"sweep_{sweep_id}"
