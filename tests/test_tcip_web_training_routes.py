"""Integration tests for the Training routes, each serving the project the backend has open."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests._training_values import sweep_space


def test_the_run_list_refuses_while_no_project_is_open(client: TestClient) -> None:
    resp = client.get("/api/training/runs")
    assert resp.status_code == 409


def _opened(experiment_id: str, tmp_path: Path, builder: str = "my_models:chestnut_burr_det"):
    """A detector run built by ``builder`` over two frames of its own under ``tmp_path``, opened
    by the launcher's own producer and writer."""
    from tests._verified_checkpoint_fixtures import detection_config, opened_run

    config = detection_config(tmp_path / f"{experiment_id}-data",
                              model_source={"builder": builder, "task": "detection"})
    return opened_run(tmp_path, config, experiment_id=experiment_id)


def test_a_listed_run_row_carries_what_the_launch_picker_shows(opened_client: TestClient,
                                                               opened_project) -> None:
    _opened("exp-picker-1", opened_project, builder="my_models:leaf_det")

    resp = opened_client.get("/api/training/runs")

    assert resp.status_code == 200
    (row,) = resp.json()["runs"]
    assert (row["experiment_id"], row["builder"], row["task"]) == (
        "exp-picker-1", "my_models:leaf_det", "detection")
    assert row["state"] == "running"  # launched, its heartbeat still fresh
    assert (row["relaunched_from"], row["sweep"]) == (None, None)
    assert resp.json()["sweeps"] == []


def test_a_run_is_read_by_its_id_and_an_unknown_id_refuses_naming_it(opened_client: TestClient,
                                                                       opened_project) -> None:
    _opened("exp-read", opened_project)

    resp = opened_client.get("/api/training/runs/exp-read")
    assert resp.status_code == 200
    assert (resp.json()["run"]["experiment_id"], resp.json()["run"]["state"]) == (
        "exp-read", "running")

    missing = opened_client.get("/api/training/runs/exp-missing")
    assert missing.status_code == 404
    assert "exp-missing" in missing.json()["detail"]


def test_relaunch_route_404s_for_an_unknown_experiment(opened_client: TestClient) -> None:
    resp = opened_client.post(
        "/api/training/runs", json={"relaunched_from": "nope", "user": "tester"})
    assert resp.status_code == 404


def test_relaunch_route_422s_with_preflight_issues_for_a_refused_config(
    tmp_path, opened_client: TestClient
) -> None:
    _opened("exp-refused", tmp_path, builder="not.a:module")

    resp = opened_client.post("/api/training/runs",
                       json={"relaunched_from": "exp-refused", "user": "tester"})
    assert resp.status_code == 422
    assert resp.json()["detail"]["issues"]


def test_metrics_stream_reports_no_frames_for_a_run_no_record_claims(
    opened_client: TestClient,
) -> None:
    with opened_client.websocket_connect("ws://127.0.0.1/api/training/runs/foo-xxx/stream") as ws:
        msg = ws.receive_json()
    assert msg["type"] == "status"
    assert msg["status"] is None
    assert msg["error"]


def _completed_run_with_rows(project: Path, run_id: str) -> Path:
    """A completed run under ``project`` (a terminal run ends the stream after one tick) whose
    body logged two metrics rows through the envelope's own sink."""
    from tests._verified_checkpoint_fixtures import finished_run

    return finished_run(project, experiment_id=run_id,
                        rows=[{"epoch": 1, "loss": 0.9}, {"epoch": 2, "loss": 0.4}])


def test_metrics_stream_serves_the_rows_the_run_logged(
    opened_client: TestClient, tmp_path: Path
) -> None:
    run_id = "exp-abc"
    _completed_run_with_rows(tmp_path, run_id)

    frames = []
    with opened_client.websocket_connect(f"ws://127.0.0.1/api/training/runs/{run_id}/stream") as ws:
        while True:
            msg = ws.receive_json()
            frames.append(msg)
            if msg["type"] == "status":
                break

    rows = [f["row"] for f in frames if f["type"] == "metric"]
    assert [(r["epoch"], r["loss"]) for r in rows] == [(1, 0.9), (2, 0.4)]
    assert frames[-1]["type"] == "status"
    for frame in frames:
        assert frame["experiment_id"] == run_id
        assert "run_id" not in frame


def test_metrics_stream_sends_a_loss_and_the_selection_logged_after_it_as_one_epoch(
    tmp_path: Path,
) -> None:
    """A loss row followed by a selection row at one epoch streams as one frame carrying both."""
    import asyncio

    from tcip_web.routes.training import _stream_metrics
    from tests._verified_checkpoint_fixtures import finished_run

    finished_run(tmp_path, experiment_id="exp-one-epoch", rows=[
        {"epoch": 1, "loss": 0.9}, {"epoch": 1, "selection": 0.5, "selection_metric": "loss"}])
    sent: list[dict] = []

    class _Socket:
        async def send_json(self, payload: dict) -> None:
            sent.append(payload)

    asyncio.run(_stream_metrics(_Socket(), tmp_path, "exp-one-epoch", poll_seconds=0.0))

    (row,) = [msg["row"] for msg in sent if msg["type"] == "metric"]
    assert (row["epoch"], row["loss"], row["selection"]) == (1, 0.9, 0.5)


def test_a_finite_value_logged_after_a_non_finite_one_at_one_epoch_streams_alone(
    tmp_path: Path,
) -> None:
    """A loss logged non-finite and then finite at one epoch streams as the finite value, its
    earlier non-finite state gone with the value it described."""
    import asyncio

    from tcip_store.values import NOT_FINITE_SUFFIX

    from tcip_mcp.experiments import write_final_status
    from tcip_web.routes.training import _stream_metrics
    from tests._verified_checkpoint_fixtures import log_epoch

    run_dir = _opened("exp-recovered", tmp_path)
    log_epoch(run_dir, 2, {"loss": float("nan")})
    log_epoch(run_dir, 2, {"loss": 0.1})
    write_final_status(run_dir, "failed", "stopped after the second row", checkpoint=None)
    sent: list[dict] = []

    class _Socket:
        async def send_json(self, payload: dict) -> None:
            sent.append(payload)

    asyncio.run(_stream_metrics(_Socket(), tmp_path, "exp-recovered", poll_seconds=0.0))

    (row,) = [msg["row"] for msg in sent if msg["type"] == "metric"]
    assert row["loss"] == 0.1 and f"loss{NOT_FINITE_SUFFIX}" not in row


def test_the_stream_sends_epoch_rows_as_epochs_and_the_latest_batch_row_as_its_own_frame(
    tmp_path: Path,
) -> None:
    import asyncio

    from tcip_mcp.experiments import STEP_KEY, observe, write_final_status
    from tcip_mcp.pipelines.training.envelope import TrainContext
    from tcip_mcp.pipelines.training.run_registry import observed_run
    from tcip_web.routes.training import _stream_metrics

    run_dir = _opened("exp-batches", tmp_path)
    ctx = TrainContext(run=observed_run(observe(run_dir)), train_loader=None)
    ctx.log_batch(1, 1, {"batch_loss": 0.8})
    ctx.log_batch(2, 1, {"batch_loss": 0.7})
    ctx.log_metrics(1, {"train_loss": 0.75})
    ctx.log_batch(3, 2, {"batch_loss": 0.6})
    write_final_status(run_dir, "failed", "stopped mid-epoch", checkpoint=None)
    sent: list[dict] = []

    class _Socket:
        async def send_json(self, payload: dict) -> None:
            sent.append(payload)

    asyncio.run(_stream_metrics(_Socket(), tmp_path, "exp-batches", poll_seconds=0.0))

    (epoch,) = [msg["row"] for msg in sent if msg["type"] == "metric"]
    assert (epoch["epoch"], epoch["train_loss"]) == (1, 0.75) and STEP_KEY not in epoch
    (batch,) = [msg["row"] for msg in sent if msg["type"] == "batch"]
    assert (batch[STEP_KEY], batch["epoch"], batch["batch_loss"]) == (3, 2, 0.6)
    assert sent[-1]["type"] == "status" and sent[-1]["status"]["current_epoch"] == 1


def test_metrics_stream_pushes_complete_entries_and_defers_a_partial_one(tmp_path: Path) -> None:
    """An entry still being appended is held back, never pushed half-formed.

    A row's bytes land on disk before its terminator does, so a stream that consumed the
    fragment would skip the completed row permanently. The stream reads through the log's own
    byte cursor, which holds that fragment back until it is whole.
    """
    import asyncio

    from tcip_web.routes.training import _stream_metrics

    run_id = "exp-streamed"
    run_dir = _completed_run_with_rows(tmp_path, run_id)
    with (run_dir / "metrics.jsonl").open("ab") as f:
        f.write(b'{"epoch": 3')

    sent: list[dict] = []

    class _Socket:
        async def send_json(self, payload: dict) -> None:
            sent.append(payload)

    asyncio.run(_stream_metrics(_Socket(), tmp_path, run_id, poll_seconds=0.0))

    rows = [msg["row"] for msg in sent if msg["type"] == "metric"]
    assert [(r["epoch"], r["loss"]) for r in rows] == [(1, 0.9), (2, 0.4)]
    assert sent[-1]["type"] == "status"


def test_metrics_stream_reads_the_log_off_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stream's log reads must never run on the event loop's own thread: a slow disk read
    running on the loop would stall every request and socket the app serves, not only this
    one."""
    import asyncio
    import threading

    from tcip_mcp import experiments
    from tcip_web.routes.training import _stream_metrics

    run_id = "exp-off-loop"
    _completed_run_with_rows(tmp_path, run_id)

    main_thread = threading.current_thread()
    read_threads: list[threading.Thread] = []
    real_read_rows = experiments.read_rows

    def recording_read_rows(*args: object, **kwargs: object) -> object:
        read_threads.append(threading.current_thread())
        return real_read_rows(*args, **kwargs)

    monkeypatch.setattr(experiments, "read_rows", recording_read_rows)

    class _Socket:
        async def send_json(self, payload: dict) -> None:
            pass

    asyncio.run(_stream_metrics(_Socket(), tmp_path, run_id, poll_seconds=0.0))

    assert read_threads
    assert all(thread is not main_thread for thread in read_threads)


def test_compare_route_compares_two_runs_and_names_an_unknown_id(
    opened_client: TestClient, tmp_path: Path,
) -> None:
    _completed_run_with_rows(tmp_path, "exp-left")
    _completed_run_with_rows(tmp_path, "exp-right")

    resp = opened_client.post("/api/training/compare",
                       json={"experiment_ids": ["exp-left", "exp-right", "exp-gone"]})

    assert resp.status_code == 200
    left, right, gone = resp.json()["experiments"]
    assert (left["experiment_id"], left["state"], left["n_epochs"]) == ("exp-left", "completed", 2)
    assert (right["experiment_id"], right["state"]) == ("exp-right", "completed")
    assert gone == {"experiment_id": "exp-gone", "error": "Experiment not found: exp-gone"}
    assert resp.json()["same_dataset_fingerprint"] is None


def test_metric_directions_route_answers_the_declared_table(opened_client: TestClient) -> None:
    """The route answers evaluation.py's own declared-direction table."""
    from tcip_mcp.pipelines.training.evaluation import HIGHER_IS_BETTER_BY_METRIC

    resp = opened_client.get("/api/training/metric-directions")
    assert resp.status_code == 200
    assert resp.json()["higher_is_better"] == HIGHER_IS_BETTER_BY_METRIC


def test_cancel_unknown_run_returns_404(opened_client: TestClient) -> None:
    resp = opened_client.post("/api/training/runs/does-not-exist/cancel", json={"user": "tester"})
    assert resp.status_code == 404


def test_a_cancel_naming_its_person_is_admitted_and_its_line_names_them(
    tmp_path, opened_client: TestClient
) -> None:
    from tcip_mcp.audit import audit_log_key
    from tcip_store import read_log
    from tests._verified_checkpoint_fixtures import detection_config, opened_run

    opened_run(tmp_path, detection_config(tmp_path / "gui-data"), experiment_id="exp-gui-cancel")
    before = len(read_log(audit_log_key(tmp_path)).records)

    resp = opened_client.post("/api/training/runs/exp-gui-cancel/cancel", json={"user": "tester"})

    assert resp.status_code == 200, resp.text
    (line,) = read_log(audit_log_key(tmp_path)).records[before:]
    assert (line["tool"], line["actor"]) == ("cancel_training", "user:tester")


def test_tensorboard_route_404s_for_unknown_run(opened_client: TestClient) -> None:
    resp = opened_client.post("/api/training/runs/does-not-exist/tensorboard", json={})
    assert resp.status_code == 404
    assert "does-not-exist" in resp.json()["detail"]


def test_tensorboard_route_launches_over_the_run_s_own_board(
    opened_client: TestClient, tmp_path: Path, tb_launches: list[str],
) -> None:
    # The GUI's link comes from a TensorBoard this process started, so the route must reach
    # launch_tensorboard with the run's own log directory and hand back what it returned.
    from tcip_mcp.experiments import board_of

    run_dir = _opened("run-42", tmp_path)
    board_of(run_dir).mkdir()
    (board_of(run_dir) / "events.out.tfevents.1.host").write_bytes(b"")

    resp = opened_client.post("/api/training/runs/run-42/tensorboard", json={})
    assert resp.status_code == 200
    assert resp.json()["url"] == "http://127.0.0.1:6006"
    assert tb_launches == [str(board_of(run_dir))]


def test_tensorboard_route_404s_with_no_logs_for_a_run_with_no_event_file(
    opened_client: TestClient, tmp_path: Path,
) -> None:
    """A run whose body never reached ``SummaryWriter`` has a real directory but no event file
    for TensorBoard to serve; that reads as ``no_logs``, not as a launchable board over an empty
    directory, so the GUI never offers a retry against it."""
    _opened("run-nologs", tmp_path)

    resp = opened_client.post("/api/training/runs/run-nologs/tensorboard", json={})
    assert resp.status_code == 404
    assert resp.json()["detail"]["no_logs"] is True


def test_tensorboard_route_404s_with_no_logs_carrying_the_recorded_error(
    tmp_path, opened_client: TestClient,
) -> None:
    """A run whose status the platform itself recorded carries a real error (a crash during
    data load, say) and whose output directory holds no event file: the refusal must say both
    that it produced no logs (so the tab offers no Try again) and what the recorded error was,
    not the plain error text a run with real logs would still get."""
    from tests._verified_checkpoint_fixtures import finished_run

    run_id = "exp-fails-at-data-load"
    finished_run(tmp_path, experiment_id=run_id, training_source=f"{__name__}:_fails_at_data_load")

    resp = opened_client.post(f"/api/training/runs/{run_id}/tensorboard", json={})
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert detail["no_logs"] is True
    assert detail["message"] == "could not open the dataset's images_dir"


def _fails_at_data_load(ctx) -> None:
    """A training body that dies before its first epoch, the way a data-load failure does."""
    raise RuntimeError("could not open the dataset's images_dir")


def test_list_runs_reads_every_training_run_directory(opened_project, monkeypatch) -> None:
    """The route's rows come from the project's run directories: a run whose process stopped
    touching its heartbeat reads interrupted."""
    from tcip_mcp import experiments
    from tcip_web.routes import training

    _opened("run_1", opened_project)
    monkeypatch.setattr(experiments, "HEARTBEAT_STALE_SECONDS", -1.0)

    by_id = {r.experiment_id: r for r in training.list_runs_route().runs}
    assert by_id["run_1"].state == "interrupted"


def test_list_runs_route_is_a_pure_pass_through_to_the_tool(opened_project) -> None:
    """The route adds nothing of its own: its listing equals the agent's tool's, exactly, so the
    route holds no reconstruction of its own."""
    from tcip_mcp.tools.experiment_tools import list_experiments
    from tcip_web.routes.training import list_runs_route

    _opened("exp-route-parity", opened_project)

    assert list_runs_route().model_dump() == list_experiments(opened_project)


def test_a_route_launch_shows_its_declaration_from_its_launch_event(
    tmp_path, monkeypatch, opened_client: TestClient
) -> None:
    """A run started through the browser-facing door shows the declaration its launch event
    carries: none, since no agent declared itself to the serving process."""
    from tests._producer_fixtures import fake_popen

    monkeypatch.chdir(tmp_path)
    fake_popen(monkeypatch, [])

    from tcip_mcp.experiments import RUN_FILE, experiment_dir, read_record
    from tests._verified_checkpoint_fixtures import detection_config, opened_run

    opened_run(tmp_path, detection_config(tmp_path / "gui-data", batch_size=1),
               experiment_id="exp-gui-relaunch")

    from tcip_mcp.audit import audit_log_key
    from tcip_store import read_log

    before = len(read_log(audit_log_key(tmp_path)).records)
    resp = opened_client.post("/api/training/runs",
                       json={"relaunched_from": "exp-gui-relaunch", "user": "tester"})
    assert resp.status_code == 200, resp.json()

    minted = resp.json()["experiment_id"]
    (line,) = read_log(audit_log_key(tmp_path)).records[before:]
    assert (line["tool"], line["actor"]) == ("launch_training", "user:tester")
    relaunched = read_record(experiment_dir(minted, project=tmp_path) / RUN_FILE)
    assert "launched_by" not in relaunched
    assert relaunched["relaunched_from"] == "exp-gui-relaunch"
    row = next(r for r in opened_client.get("/api/training/runs").json()["runs"]
               if r["experiment_id"] == minted)
    assert row["launch"] == {}


def _regression_config(tmp_path: Path) -> dict:
    from tests.tiny_trainer_fixtures import regressor_config, write_regression_dataset

    images_dir, csv_path = write_regression_dataset(
        tmp_path, intensities=[0.0, 1.0], values=[0.1, 0.9])
    return regressor_config(1, data={"images_dir": str(images_dir), "labels_dir": str(csv_path),
                                     "split": {"seed": 0, "val_ratio": 0.15}})


def test_relaunch_route_forks_a_run_s_config_and_names_the_parent(
    tmp_path, monkeypatch, opened_client: TestClient
) -> None:
    """Relaunching a run's config starts a new run directory through the real launcher and a
    real child, its launch record replaying the picked run's config and naming it as parent."""
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", lambda *a, **k: {})
    from tcip_mcp.tools.training_tools import launch_training
    from tests._verified_checkpoint_fixtures import run_to_end

    first = launch_training(tmp_path, _regression_config(tmp_path), actor=None)
    assert "error" not in first, first
    parent_id = first["experiment_id"]
    assert run_to_end(tmp_path, parent_id)["state"] == "completed"

    resp = opened_client.post(
        "/api/training/runs", json={"relaunched_from": parent_id, "user": "tester"})
    assert resp.status_code == 200, resp.json()
    forked_id = resp.json()["experiment_id"]
    assert forked_id != parent_id
    assert run_to_end(tmp_path, forked_id)["state"] == "completed"

    from tcip_mcp.experiments import RUN_FILE, experiment_dir, read_record

    parent = read_record(experiment_dir(parent_id, project=tmp_path) / RUN_FILE)
    forked = read_record(experiment_dir(forked_id, project=tmp_path) / RUN_FILE)
    assert forked["relaunched_from"] == parent_id
    assert forked["config"] == parent["config"]


def test_list_runs_route_names_the_run_s_selection_metric(
    tmp_path, monkeypatch, opened_client: TestClient
) -> None:
    """A launched (subprocess-delegated) run's row carries the metric the trainer itself
    stamped on its metrics-log rows, and the best value read back beside it, so the Training
    tab can label its best value instead of showing nothing (the parent's placeholder
    ``inf``) or a name with no value (a ``None`` best value)."""
    import math

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", lambda *a, **k: {})
    from tcip_mcp.tools.training_tools import launch_training
    from tests._verified_checkpoint_fixtures import run_to_end

    result = launch_training(tmp_path, _regression_config(tmp_path), actor=None)
    assert "error" not in result, result
    run_to_end(tmp_path, result["experiment_id"])

    from tcip_web.routes.training import list_runs_route

    by_id = {r["experiment_id"]: r for r in list_runs_route().model_dump()["runs"]}
    row = by_id[result["experiment_id"]]
    # Regression selects on its loss by default; there is no evaluation.selection_metric override
    # in this config.
    assert row["best_metric_name"] == "loss"
    assert isinstance(row["best_metric"], (int, float)) and math.isfinite(row["best_metric"])


def test_a_sweep_trial_is_listed_read_canceled_and_streamed_by_its_id(
    opened_client: TestClient, opened_project: Path, real_hpo_base_config: dict,
) -> None:
    """A trial is a run directory under its sweep: the run routes reach it by its own id, and the
    listing groups it under its sweep rather than beside the runs launched on their own."""
    from tcip_mcp.tools.training_tools import open_trial
    from tests._verified_checkpoint_fixtures import opened_sweep

    _opened("run-a", opened_project)
    opened = opened_sweep(opened_project, real_hpo_base_config)
    sweep_id = opened.name
    trial = open_trial(opened, "t0", {"optimizer.head_lr": 0.001})

    listing = opened_client.get("/api/training/runs").json()
    assert [r["experiment_id"] for r in listing["runs"]] == ["run-a"]
    (sweep,) = listing["sweeps"]
    assert sweep["sweep_id"] == sweep_id
    assert [(t["experiment_id"], t["sweep"], t["trial_params"]) for t in sweep["trials"]] == [
        (trial.name, sweep_id, {"optimizer.head_lr": 0.001})]

    read = opened_client.get(f"/api/training/runs/{trial.name}")
    assert (read.status_code, read.json()["run"]["sweep"]) == (200, sweep_id)

    canceled = opened_client.post(
        f"/api/training/runs/{trial.name}/cancel", json={"user": "tester"})
    assert canceled.status_code == 200, canceled.text
    assert canceled.json()["cancel_requested"] is True

    from tcip_mcp.experiments import write_final_status

    write_final_status(trial, "canceled", "canceled by request", checkpoint=None)
    with opened_client.websocket_connect(
            f"ws://127.0.0.1/api/training/runs/{trial.name}/stream") as ws:
        frame = ws.receive_json()
    assert frame["type"] == "status"
    assert (frame["status"]["state"], frame["status"]["sweep"]) == ("canceled", sweep_id)


def test_a_sweep_and_its_trial_each_have_one_board_whichever_door_launches_it(
    opened_client: TestClient, opened_project: Path, real_hpo_base_config: dict, monkeypatch,
    tb_launches: list[str],
) -> None:
    """A sweep's TensorBoard is the sweep's own directory whether its body or the run route
    starts it, and a trial's is its own run board: one key per directory."""
    import tcip_mcp.tools.training_tools as tt
    from tcip_mcp.experiments import board_of, find_sweep
    from tcip_mcp.pipelines.training.tensorboard_manager import _key_of

    trials: list[Path] = []

    def one_trial_search(**kw) -> None:
        trial = tt.open_trial(kw["sweep_dir"], "t0", {"optimizer.head_lr": 0.1})
        board_of(trial).mkdir()
        (board_of(trial) / "events.out.tfevents.1.host").write_bytes(b"")
        trials.append(trial)

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", one_trial_search)
    sweep_id = tt.run_hyperparameter_search(
        opened_project, base_config=real_hpo_base_config, param_space=sweep_space(), n_trials=1,
        search_seed=0)["sweep"]["sweep_id"]
    for name in (sweep_id, trials[0].name):
        resp = opened_client.post(f"/api/training/runs/{name}/tensorboard", json={})
        assert resp.status_code == 200, resp.text

    sweep_dir = find_sweep(sweep_id, project=opened_project)
    assert sweep_dir is not None and board_of(sweep_dir) == sweep_dir
    assert [_key_of(logdir) for logdir in tb_launches] == [
        _key_of(str(sweep_dir)), _key_of(str(sweep_dir)), _key_of(str(board_of(trials[0])))]


def test_a_recorded_sweep_relaunches_through_the_run_launch_door(
    opened_client: TestClient, opened_project: Path, real_hpo_base_config: dict, monkeypatch,
) -> None:
    """A sweep relaunches from its own recorded input through the one launch door: a new sweep
    is opened naming its source, its audit line names the person, and its body starts in a worker
    process of its own over the new sweep's directory; a sweep relaunch never takes a partition."""
    import tcip_store as ts
    from tcip_mcp.audit import audit_log_key
    from tcip_mcp.experiments import find_sweep, observe
    from tests._producer_fixtures import fake_popen
    from tests._verified_checkpoint_fixtures import opened_sweep

    started: list[list[str]] = []
    fake_popen(monkeypatch, started)
    source = opened_sweep(opened_project, real_hpo_base_config).name
    before = len(ts.read_log(audit_log_key(opened_project)).records)

    resp = opened_client.post(
        "/api/training/runs", json={"relaunched_from": source, "user": "tester"})

    assert resp.status_code == 200, resp.text
    minted = find_sweep(resp.json()["sweep_id"], project=opened_project)
    assert minted is not None
    assert observe(minted).record["input"]["relaunched_from"] == source
    (line,) = ts.read_log(audit_log_key(opened_project)).records[before:]
    assert (line["tool"], line["actor"]) == ("open_sweep", "user:tester")
    assert [argv[-2:] for argv in started] == [["--run-dir", str(minted)]]

    refused = opened_client.post("/api/training/runs", json={
        "relaunched_from": source, "selection_dir": str(opened_project), "user": "tester"})
    assert refused.status_code == 422
