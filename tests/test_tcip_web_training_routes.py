"""Integration tests for the Training routes, each serving the project the backend has open."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_web.app import app


@pytest.fixture
def client(opened_project) -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _wait_terminal(project: Path, experiment_id: str, deadline_s: float = 60) -> dict:
    from tcip_mcp.tools.training_tools import monitor_training

    deadline = time.monotonic() + deadline_s
    status: dict = {}
    while time.monotonic() < deadline:
        status = monitor_training(project, experiment_id)
        if status.get("status") in ("failed", "completed", "canceled"):
            return status
        time.sleep(0.2)
    return status


def test_the_validate_route_is_not_registered(client: TestClient) -> None:
    """Nothing serves ``/api/training/validate``: the browser never submits a typed config, so
    no route validates one. ``preflight_config`` keeps its own coverage in
    tests/test_training_tools.py; this pins the route's absence alone."""
    resp = client.post("/api/training/validate", json={"config": {}})
    assert resp.status_code == 404


def test_the_launch_route_is_not_registered(client: TestClient) -> None:
    """Nothing serves ``/api/training/launch``: a run starts only from a recorded config,
    through ``/api/training/runs``, never from a client-submitted one."""
    resp = client.post("/api/training/launch", json={"config": {}, "output_dir": ""})
    assert resp.status_code == 404


def test_the_run_list_refuses_while_no_project_is_open() -> None:
    resp = TestClient(app, base_url="http://127.0.0.1").get("/api/training/runs")
    assert resp.status_code == 409


def _opened(experiment_id: str, tmp_path: Path, builder: str = "my_models:chestnut_burr_det"):
    """A detector run built by ``builder`` over two frames of its own under ``tmp_path``, opened
    by the launcher's own producer and writer."""
    from tests._verified_checkpoint_fixtures import detection_config, opened_run

    config = detection_config(tmp_path / f"{experiment_id}-data",
                              model_source={"builder": builder, "task": "detection"})
    return opened_run(tmp_path, config, experiment_id=experiment_id)


def _calibration(project: Path):
    """A calibration run under ``project`` opened by its own writer."""
    from tcip_mcp.experiments import open_calibration_run

    return open_calibration_run({
        "document": "operating_point", "checkpoint_sha256": None,
        "reference_identity": {"calibration_dataset_hash": "h"}, "trait": "bud_50per_date",
        "derived_from": "a calibration door"}, project=project)


def test_list_configs_route_reports_a_launchable_config(opened_project) -> None:
    from tcip_web.routes.training import list_configs_route

    _opened("exp-picker-1", opened_project)

    by_id = {r["experiment_id"]: r for r in list_configs_route()["configs"]}
    row = by_id["exp-picker-1"]
    assert (row["builder"], row["task"], row["subject"]) == (
        "my_models:chestnut_burr_det", "detection", "bud")
    assert row["state"] == "running"  # launched, its heartbeat still fresh
    assert row["parent_experiment"] is None


def test_relaunch_route_404s_for_an_unknown_experiment(client: TestClient) -> None:
    resp = client.post("/api/training/runs", json={"experiment_id": "nope"})
    assert resp.status_code == 404


def test_relaunch_route_404s_for_a_calibration_run(tmp_path, client: TestClient) -> None:
    """A calibration run's launch record carries no config, so there is nothing to relaunch."""
    calibration = _calibration(tmp_path)

    resp = client.post("/api/training/runs", json={"experiment_id": calibration.name})
    assert resp.status_code == 404


def test_relaunch_route_422s_with_preflight_issues_for_a_refused_config(
    tmp_path, client: TestClient
) -> None:
    _opened("exp-refused", tmp_path, builder="not.a:module")

    resp = client.post("/api/training/runs", json={"experiment_id": "exp-refused"})
    assert resp.status_code == 422
    assert resp.json()["detail"]["issues"]


def test_list_runs_returns_shape(client: TestClient) -> None:
    resp = client.get("/api/training/runs")
    assert resp.status_code == 200
    body = resp.json()
    assert "runs" in body


def test_the_http_metrics_route_is_not_registered(client: TestClient) -> None:
    """No HTTP metrics route is registered: the WebSocket stream (below) is the single serving
    surface a run's metrics rows reach the browser through. The sibling GET pins the router as
    mounted, so the 404 discriminates this one route rather than a router that never mounted."""
    resp = client.get("/api/training/runs/foo-xxx/metrics")
    assert resp.status_code == 404

    registered = client.get("/api/training/runs")
    assert registered.status_code == 200


def test_metrics_stream_reports_no_frames_for_a_run_no_record_claims(client: TestClient) -> None:
    with client.websocket_connect("ws://127.0.0.1/api/training/runs/foo-xxx/stream") as ws:
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


def test_metrics_stream_serves_the_rows_the_run_logged(client: TestClient, tmp_path: Path) -> None:
    run_id = "exp-abc"
    _completed_run_with_rows(tmp_path, run_id)

    frames = []
    with client.websocket_connect(f"ws://127.0.0.1/api/training/runs/{run_id}/stream") as ws:
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


def test_compare_route_handles_empty_ids(client: TestClient) -> None:
    resp = client.post("/api/training/compare", json={"experiment_ids": []})
    assert resp.status_code == 200
    # body schema is up to compare_experiments; we only assert the route returns JSON.
    assert isinstance(resp.json(), dict)


def test_metric_directions_route_answers_the_declared_table(client: TestClient) -> None:
    """A plain read of evaluation.py's own declared-direction table: the comparison's metric
    chooser groups by this on mount, never by calling the audited rank tool with no metric."""
    from tcip_mcp.pipelines.training.evaluation import HIGHER_IS_BETTER_BY_METRIC

    resp = client.get("/api/training/metric-directions")
    assert resp.status_code == 200
    assert resp.json()["higher_is_better"] == HIGHER_IS_BETTER_BY_METRIC


def test_cancel_unknown_run_returns_404(client: TestClient) -> None:
    resp = client.post("/api/training/runs/does-not-exist/cancel", json={})
    assert resp.status_code == 404


def test_tensorboard_route_404s_for_unknown_run(client: TestClient) -> None:
    resp = client.post("/api/training/runs/does-not-exist/tensorboard", json={})
    assert resp.status_code == 404
    assert "does-not-exist" in resp.json()["detail"]


def test_tensorboard_route_launches_under_the_run_output_dir(client: TestClient, monkeypatch,
                                                             tmp_path: Path) -> None:
    # The GUI's link comes from a TensorBoard this process started, so the route must reach
    # launch_tensorboard with the run's own log directory and hand back what it returned.
    calls: list[tuple[str, str]] = []
    tb_dir = tmp_path / "tensorboard"
    tb_dir.mkdir()
    (tb_dir / "events.out.tfevents.1.host").write_bytes(b"")

    def fake_status(project: Path, experiment_id: str) -> dict:
        return {"experiment_id": experiment_id, "status": "running", "output_dir": str(tmp_path)}

    def fake_launch(logdir: str, key: str | None = None) -> dict:
        calls.append((logdir, key or ""))
        return {"url": "http://localhost:6006", "port": 6006, "pid": 1, "logdir": logdir}

    monkeypatch.setattr("tcip_mcp.tools.training_tools.monitor_training", fake_status)
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", fake_launch
    )

    resp = client.post("/api/training/runs/run-42/tensorboard", json={})
    assert resp.status_code == 200
    assert resp.json()["url"] == "http://localhost:6006"
    # No key of its own: keyed by log directory, so a repeat call here reuses the run's own
    # TensorBoard entry rather than starting a second one.
    assert calls == [(f"{tmp_path}/tensorboard", "")]


def test_tensorboard_route_404s_with_no_logs_for_a_run_with_no_output_dir(
    client: TestClient, monkeypatch,
) -> None:
    """A run that failed before writing an output directory has nothing a TensorBoard could
    ever serve; the refusal names that so the GUI never offers a retry against it."""

    def fake_status(project: Path, experiment_id: str) -> dict:
        return {"experiment_id": experiment_id, "status": "failed", "output_dir": "", "error": None}

    monkeypatch.setattr("tcip_mcp.tools.training_tools.monitor_training", fake_status)

    resp = client.post("/api/training/runs/run-nologs/tensorboard", json={})
    assert resp.status_code == 404
    assert resp.json()["detail"]["no_logs"] is True


def test_tensorboard_route_404s_with_no_logs_for_a_stamped_dir_with_no_event_file(
    client: TestClient, monkeypatch, tmp_path: Path,
) -> None:
    """A run whose output directory was stamped before the child crashed (e.g. it never
    reached ``SummaryWriter``) has a real output directory but no event file for TensorBoard
    to serve; that reads as ``no_logs`` too, not as a launchable board over an empty directory."""

    def fake_status(project: Path, experiment_id: str) -> dict:
        return {"experiment_id": experiment_id, "status": "failed", "output_dir": str(tmp_path),
                "error": None}

    monkeypatch.setattr("tcip_mcp.tools.training_tools.monitor_training", fake_status)

    resp = client.post("/api/training/runs/run-nologs-2/tensorboard", json={})
    assert resp.status_code == 404
    assert resp.json()["detail"]["no_logs"] is True


def test_tensorboard_route_404s_with_no_logs_carrying_the_recorded_error(
    tmp_path, client: TestClient,
) -> None:
    """A run whose status the platform itself recorded carries a real error (a crash during
    data load, say) and whose output directory holds no event file: the refusal must say both
    that it produced no logs (so the tab offers no Try again) and what the recorded error was,
    not the plain error text a run with real logs would still get."""
    from tests._verified_checkpoint_fixtures import finished_run

    run_id = "exp-fails-at-data-load"
    finished_run(tmp_path, experiment_id=run_id, training_source=f"{__name__}:_fails_at_data_load")

    resp = client.post(f"/api/training/runs/{run_id}/tensorboard", json={})
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert detail["no_logs"] is True
    assert detail["error"] == "could not open the dataset's images_dir"


def _fails_at_data_load(ctx) -> None:
    """A training body that dies before its first epoch, the way a data-load failure does."""
    raise RuntimeError("could not open the dataset's images_dir")


def test_list_runs_reads_every_training_run_directory(opened_project, monkeypatch) -> None:
    """The route's rows come from the project's run directories: a run whose process stopped
    touching its heartbeat reads interrupted, and a calibration run (no config) is not a
    training run."""
    from tcip_mcp import experiments
    from tcip_web.routes import training

    _opened("run_1", opened_project)
    calibration = _calibration(opened_project)
    monkeypatch.setattr(experiments, "HEARTBEAT_STALE_SECONDS", -1.0)

    by_id = {r["experiment_id"]: r for r in training.list_runs_route()["runs"]}
    assert by_id["run_1"]["status"] == "interrupted"
    assert calibration.name not in by_id


def test_list_runs_route_is_a_pure_pass_through_to_the_tool(opened_project) -> None:
    """The route adds nothing of its own: its rows equal the tool's ``launched_only=True`` view,
    exactly, so the route holds no reconstruction of its own."""
    from tcip_mcp.tools.experiment_tools import list_experiments
    from tcip_web.routes.training import list_runs_route

    _opened("exp-route-parity", opened_project)

    assert list_runs_route()["runs"] == list_experiments(opened_project, launched_only=True)["runs"]


def test_relaunch_route_stamps_the_run_as_launched_through_this_app(
    tmp_path, monkeypatch, client: TestClient
) -> None:
    """The route wraps its call in declare_launcher('gui'), so the run it starts through the
    browser-facing door reads back as launched by this app, not by a bare process. Mocks
    subprocess.Popen so the assertion runs against the launch record without waiting on a real
    child (test_relaunch_route_forks_a_run_s_config_and_names_the_parent covers the real
    subprocess path)."""
    import subprocess

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", lambda *a, **k: {})

    # Forces the mcp package's own real import before Popen is replaced below: the route
    # imports training_tools lazily inside the request, too late to see a real Popen.
    import tcip_mcp.tools.training_tools  # noqa: F401

    class _FakeProc:
        pid = 424242

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _FakeProc())

    from tcip_mcp.experiments import RUN_FILE, experiment_dir, read_record
    from tests._verified_checkpoint_fixtures import detection_config, opened_run

    opened_run(tmp_path, detection_config(
        tmp_path / "gui-data", batch_size=1, stages=[{"freeze_to": -1, "epochs": 1}],
        mixed_precision=False, device="cpu"), experiment_id="exp-gui-relaunch")

    resp = client.post("/api/training/runs", json={"experiment_id": "exp-gui-relaunch"})
    assert resp.status_code == 200, resp.json()

    relaunched = read_record(experiment_dir(resp.json()["experiment_id"], project=tmp_path)
                             / RUN_FILE)
    assert relaunched["launched_by"] == {"launcher": "gui"}
    assert relaunched["parent_experiment"] == "exp-gui-relaunch"


def _regression_config(tmp_path: Path) -> dict:
    from tests.tiny_trainer_fixtures import write_regression_dataset

    images_dir, csv_path = write_regression_dataset(
        tmp_path, intensities=[0.0, 1.0], values=[0.1, 0.9])
    return {
        "model_source": {"builder": "tests.tiny_trainer_fixtures:build_mean_intensity_regressor",
                         "task": "regression"},
        "data": {"images_dir": str(images_dir), "labels_dir": str(csv_path)},
        "batch_size": 2, "stages": [{"freeze_to": 0, "epochs": 1}],
        "mixed_precision": False, "device": "cpu",
        "checkpoint_every_n_epochs": 0, "early_stopping": {"enabled": False},
    }


def test_relaunch_route_forks_a_run_s_config_and_names_the_parent(
    tmp_path, monkeypatch, client: TestClient
) -> None:
    """Relaunching a run's config starts a new run directory through the real launcher and a
    real child, its launch record replaying the picked run's config and naming it as parent."""
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", lambda *a, **k: {})
    from tcip_mcp.tools.training_tools import launch_training

    first = launch_training(tmp_path, _regression_config(tmp_path))
    assert "error" not in first, first
    parent_id = first["experiment_id"]
    assert _wait_terminal(tmp_path, parent_id)["status"] == "completed"

    resp = client.post("/api/training/runs", json={"experiment_id": parent_id})
    assert resp.status_code == 200, resp.json()
    forked_id = resp.json()["experiment_id"]
    assert forked_id != parent_id
    assert _wait_terminal(tmp_path, forked_id)["status"] == "completed"

    from tcip_mcp.experiments import RUN_FILE, experiment_dir, read_record

    parent = read_record(experiment_dir(parent_id, project=tmp_path) / RUN_FILE)
    forked = read_record(experiment_dir(forked_id, project=tmp_path) / RUN_FILE)
    assert forked["parent_experiment"] == parent_id
    assert forked["config"] == parent["config"]


def test_list_runs_route_names_the_run_s_selection_metric(
    tmp_path, monkeypatch, client: TestClient
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

    result = launch_training(tmp_path, _regression_config(tmp_path))
    assert "error" not in result, result
    _wait_terminal(tmp_path, result["experiment_id"])

    from tcip_web.routes.training import list_runs_route

    by_id = {r["experiment_id"]: r for r in list_runs_route()["runs"]}
    row = by_id[result["experiment_id"]]
    # Regression selects on the training loss by default; there is no evaluation.selection_metric
    # override in this config.
    assert row["best_metric_name"] == "loss"
    assert isinstance(row["best_metric"], (int, float)) and math.isfinite(row["best_metric"])


def test_list_runs_excludes_hpo_trials(opened_project) -> None:
    """An HPO trial's run directory lives under its sweep, so the Training-tab list, which reads
    the project's run directories, never carries one."""
    from tcip_mcp.experiments import sweeps_dir
    from tcip_mcp.pipelines.data.split_construction import resolve_run
    from tcip_mcp.tools.training_tools import open_run
    from tcip_web.routes.training import list_runs_route
    from tests._verified_checkpoint_fixtures import detection_config

    _opened("run-a", opened_project)
    config = detection_config(opened_project / "trial-data")
    open_run(sweeps_dir(opened_project) / "hpo_study" / "trial_b", config,
             resolve_run(config, project=opened_project).record,
             launched_by={"launcher": "process"}, trial_params={"lr": 0.01})

    assert [r["experiment_id"] for r in list_runs_route()["runs"]] == ["run-a"]
