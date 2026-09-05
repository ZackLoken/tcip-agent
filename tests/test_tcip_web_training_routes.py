"""Integration tests for the Training routes."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_web.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _wait_terminal(run_id: str, deadline_s: float = 60) -> dict:
    from tcip_mcp.tools.training_tools import monitor_training

    deadline = time.monotonic() + deadline_s
    status: dict = {}
    while time.monotonic() < deadline:
        status = monitor_training(run_id)
        if status.get("status") in ("failed", "completed", "cancelled"):
            return status
        time.sleep(0.2)
    return status


def test_validate_route_no_longer_exists(client: TestClient) -> None:
    """The raw validate door retired with the config picker: the browser never submits a typed
    config again, so nothing serves ``/validate`` any more. ``preflight_config`` itself keeps
    its own coverage (tests/test_training_tools.py); this is only the route's absence."""
    resp = client.post("/api/training/validate", json={"config": {}})
    assert resp.status_code == 404


def test_launch_route_no_longer_exists(client: TestClient) -> None:
    """The raw launch door retired with the config picker: a run starts only from a recorded
    config, through ``/api/training/runs``, never from a client-submitted config again."""
    resp = client.post("/api/training/launch", json={"config": {}, "output_dir": ""})
    assert resp.status_code == 404


def test_list_configs_route_reports_a_launchable_config(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import create_experiment
    from tcip_web.routes.training import list_configs_route

    create_experiment("exp-picker-1", {
        "model_source": {"builder": "my_models:chestnut_burr_det", "task": "detection"},
        "data": {"images_dir": "/data/images", "subject": "bud"},
    })

    rows = list_configs_route()["configs"]
    by_id = {r["experiment_id"]: r for r in rows}
    assert by_id["exp-picker-1"]["builder"] == "my_models:chestnut_burr_det"
    assert by_id["exp-picker-1"]["task"] == "detection"
    assert by_id["exp-picker-1"]["subject"] == "bud"
    assert by_id["exp-picker-1"]["state"] == "created"
    assert by_id["exp-picker-1"]["parent_experiment"] is None


def test_relaunch_route_404s_for_an_unknown_experiment(client: TestClient) -> None:
    resp = client.post("/api/training/runs", json={"experiment_id": "nope"})
    assert resp.status_code == 404


def test_relaunch_route_404s_for_a_config_without_model_source(tmp_path, monkeypatch,
                                                                client: TestClient) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import create_experiment

    create_experiment("exp-not-training", {"source": "review_feedback"})

    resp = client.post("/api/training/runs", json={"experiment_id": "exp-not-training"})
    assert resp.status_code == 404


def test_relaunch_route_409s_for_a_pristine_config_with_an_attached_run(
    tmp_path, monkeypatch, client: TestClient
) -> None:
    """A 'created' experiment whose status already carries a run identity is a launch already
    attached to it; relaunching would silently clobber that stamp rather than forking."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import create_experiment, stamp_run_identity

    create_experiment("exp-attached", {"model_source": {"builder": "m:f"}, "data": {}})
    stamp_run_identity("exp-attached", "run_123", str(tmp_path / "out"))

    resp = client.post("/api/training/runs", json={"experiment_id": "exp-attached"})
    assert resp.status_code == 409


def test_relaunch_route_422s_with_preflight_issues_for_a_refused_config(
    tmp_path, monkeypatch, client: TestClient
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import create_experiment

    create_experiment("exp-refused", {
        "model_source": {"builder": "not.a:module", "task": "detection"}, "data": {},
    })

    resp = client.post("/api/training/runs", json={"experiment_id": "exp-refused"})
    assert resp.status_code == 422
    assert resp.json()["detail"]["issues"]


def test_list_runs_returns_shape(client: TestClient) -> None:
    resp = client.get("/api/training/runs")
    assert resp.status_code == 200
    body = resp.json()
    assert "runs" in body


def test_training_metrics_route_no_longer_exists(client: TestClient, tmp_path: Path) -> None:
    """The HTTP metrics route is gone; the WebSocket stream (below) is the single serving
    surface a run's metrics rows reach the browser through. Pinning the router itself as
    registered (the sibling GET below) is what makes the 404 discriminate the one deleted
    route rather than a router that failed to mount at all."""
    resp = client.get(
        "/api/training/runs/foo-xxx/metrics",
        params={"project_root": str(tmp_path)},
    )
    assert resp.status_code == 404

    registered = client.get("/api/training/runs")
    assert registered.status_code == 200


def test_metrics_stream_reports_no_frames_for_a_run_no_record_claims(
    client: TestClient, tmp_path: Path
) -> None:
    with client.websocket_connect(
        f"ws://127.0.0.1/api/training/runs/foo-xxx/stream?project_root={tmp_path}",
    ) as ws:
        msg = ws.receive_json()
    assert msg["type"] == "status"
    assert msg["status"] is None
    assert msg["error"]


def test_metrics_stream_serves_the_rows_the_run_logged(client: TestClient, tmp_path: Path) -> None:
    from tcip_mcp.experiments import create_experiment, log_metrics, update_status

    run_id = "exp-abc"
    create_experiment(run_id, {"model_source": {"builder": "m:f"}})
    log_metrics(run_id, 1, {"loss": 0.9})
    log_metrics(run_id, 2, {"loss": 0.4})
    update_status(run_id, "completed")  # a terminal run ends the stream after one tick

    frames = []
    with client.websocket_connect(
        f"ws://127.0.0.1/api/training/runs/{run_id}/stream?project_root={tmp_path}",
    ) as ws:
        while True:
            msg = ws.receive_json()
            frames.append(msg)
            if msg["type"] == "status":
                break

    rows = [f["row"] for f in frames if f["type"] == "metric"]
    assert [(r["epoch"], r["loss"]) for r in rows] == [(1, 0.9), (2, 0.4)]
    assert frames[-1]["type"] == "status"


def test_metrics_stream_pushes_complete_entries_and_defers_a_partial_one(tmp_path: Path) -> None:
    """An entry still being appended is held back, never pushed half-formed.

    A row's bytes land on disk before its terminator does, so a stream that consumed the
    fragment would skip the completed row permanently. The stream reads through the log's own
    cursor, which holds that fragment back until it is whole.

    Bound to the file backend on purpose: a torn tail is bytes an appender left mid-write, an
    on-disk file mechanic the fragment below fabricates directly; a database backend commits a
    row whole or not at all and has no such state to defer.
    """
    import asyncio

    import tcip_store
    from tcip_mcp.experiments import (
        create_experiment, experiments_dir, log_metrics, update_status,
    )
    from tcip_store.file_backend import FileBackend
    from tcip_web.routes.training import _stream_metrics

    tcip_store.bind(FileBackend())

    run_id = "exp-streamed"
    create_experiment(run_id, {"model_source": {"builder": "m:f"}})
    log_metrics(run_id, 1, {"loss": 0.9})
    log_metrics(run_id, 2, {"loss": 0.4})
    update_status(run_id, "completed")  # a terminal run ends the poll loop after one tick
    with (experiments_dir() / run_id / "metrics.jsonl").open("ab") as f:
        f.write(b'{"epoch": 3')

    sent: list[dict] = []

    class _Socket:
        async def send_json(self, payload: dict) -> None:
            sent.append(payload)

    asyncio.run(_stream_metrics(_Socket(), str(tmp_path), run_id, poll_seconds=0.0))

    rows = [msg["row"] for msg in sent if msg["type"] == "metric"]
    assert [(r["epoch"], r["loss"]) for r in rows] == [(1, 0.9), (2, 0.4)]
    assert sent[-1]["type"] == "status"


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

    def fake_status(run_id: str) -> dict:
        return {"run_id": run_id, "status": "running", "output_dir": str(tmp_path)}

    def fake_launch(logdir: str, run_id: str | None = None) -> dict:
        calls.append((logdir, run_id or ""))
        return {"url": "http://localhost:6006", "port": 6006, "pid": 1, "logdir": logdir}

    monkeypatch.setattr("tcip_mcp.tools.training_tools.monitor_training", fake_status)
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", fake_launch
    )

    resp = client.post("/api/training/runs/run-42/tensorboard", json={})
    assert resp.status_code == 200
    assert resp.json()["url"] == "http://localhost:6006"
    assert calls == [(f"{tmp_path}/tensorboard", "run-42")]


def test_tensorboard_route_404s_with_no_logs_for_a_run_with_no_output_dir(
    client: TestClient, monkeypatch,
) -> None:
    """A run that failed before writing an output directory has nothing a TensorBoard could
    ever serve; the refusal names that so the GUI never offers a retry against it."""

    def fake_status(run_id: str) -> dict:
        return {"run_id": run_id, "status": "failed", "output_dir": "", "error": None}

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

    def fake_status(run_id: str) -> dict:
        return {"run_id": run_id, "status": "failed", "output_dir": str(tmp_path), "error": None}

    monkeypatch.setattr("tcip_mcp.tools.training_tools.monitor_training", fake_status)

    resp = client.post("/api/training/runs/run-nologs-2/tensorboard", json={})
    assert resp.status_code == 404
    assert resp.json()["detail"]["no_logs"] is True


def test_tensorboard_route_404s_with_no_logs_carrying_the_recorded_error(
    tmp_path, monkeypatch, client: TestClient,
) -> None:
    """A run whose status the platform itself recorded carries a real error (a crash during
    data load, say) and whose output directory holds no event file: the refusal must say both
    that it produced no logs (so the tab offers no Try again) and what the recorded error was,
    not the plain error text a run with real logs would still get."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import create_experiment, stamp_run_identity, update_status

    run_id = "exp-fails-at-data-load"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    create_experiment(run_id, {"model_source": {"builder": "m:f"}})
    stamp_run_identity(run_id, run_id, str(output_dir))
    update_status(run_id, "failed", error="could not open the dataset's images_dir")

    resp = client.post(f"/api/training/runs/{run_id}/tensorboard", json={})
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert detail["no_logs"] is True
    assert detail["error"] == "could not open the dataset's images_dir"


def test_list_runs_reconstructs_from_experiments(tmp_path, monkeypatch) -> None:
    """No live entry for either experiment (this process never held them in ``_RUNS``): the
    route's own rows now come from ``_all_training_runs``'s unified disk enumeration with no
    patch needed. A genuine training experiment left 'running' by a crash resurfaces as
    'interrupted'; a review-feedback experiment (no model_source) is not a training run and is
    excluded.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    import tcip_store
    from tcip_mcp.experiments import create_experiment, status_key, update_status
    from tcip_web.routes import training

    create_experiment("run_1", {"model_source": {"builder": "torchvision:fasterrcnn_resnet50_fpn"}})
    # No heartbeat stamped, unlike update_status: this is the crash this test simulates.
    with tcip_store.transaction(status_key("run_1")) as txn:
        txn.write(status_key("run_1"), {"state": "running", "started": None, "ended": None})

    create_experiment("fb_1", {"source": "review_feedback"})
    update_status("fb_1", "completed")

    body = training.list_runs_route()
    by_id = {r["run_id"]: r for r in body["runs"]}
    assert by_id["run_1"]["status"] == "interrupted"  # dead process -> interrupted
    assert by_id["run_1"]["external"] is True
    assert "fb_1" not in by_id  # review-feedback experiment is not a training run


def test_list_runs_route_is_a_pure_pass_through_to_the_tool(tmp_path, monkeypatch) -> None:
    """Post-unification the route adds nothing of its own: its rows equal the tool's
    ``launched_only=True`` view, exactly. Before unification the route's own reconstruction
    added rows the tool lacked."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import create_experiment, update_status
    from tcip_mcp.tools.experiment_tools import list_experiments
    from tcip_web.routes.training import list_runs_route

    create_experiment("exp-route-parity", {"model_source": {"builder": "my_models:chestnut_burr_det"}})
    update_status("exp-route-parity", "running")

    assert list_runs_route()["runs"] == list_experiments(launched_only=True)["runs"]


def test_never_launched_experiment_is_absent_from_the_route(tmp_path, monkeypatch) -> None:
    """A pre-created experiment (state 'created', no run_id stamp, no metrics logged) never
    launched and must not list as a run at all, interrupted or otherwise."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import create_experiment
    from tcip_web.routes.training import list_runs_route

    create_experiment("exp-never-launched", {"model_source": {"builder": "my_models:chestnut_burr_det"}})

    by_id = {r["run_id"]: r for r in list_runs_route()["runs"]}
    assert "exp-never-launched" not in by_id


def test_relaunch_route_launches_a_pristine_config_as_its_own_first_run(
    tmp_path, monkeypatch, client: TestClient
) -> None:
    """Relaunching a pristine (never-launched) config reuses its own id: no fork."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", lambda *a, **k: {})
    from tcip_mcp.experiments import create_experiment
    from tests.tiny_trainer_fixtures import write_regression_dataset

    images_dir, csv_path = write_regression_dataset(
        tmp_path, intensities=[0.0, 1.0], values=[0.1, 0.9])
    labels_dir = tmp_path / "unused_labels"
    labels_dir.mkdir()
    cfg = {
        "model_source": {"builder": "tests.tiny_trainer_fixtures:build_mean_intensity_regressor",
                         "task": "regression", "in_chans": 3},
        "data": {"images_dir": str(images_dir), "csv_path": str(csv_path), "labels_dir": str(labels_dir)},
        "training": {"batch_size": 2, "stages": [{"freeze_to": 0, "epochs": 1}],
                     "mixed_precision": False, "device": "cpu",
                     "checkpoint_every_n_epochs": 0, "early_stopping": {"enabled": False}},
    }
    create_experiment("exp-pristine-relaunch", cfg)

    resp = client.post("/api/training/runs", json={"experiment_id": "exp-pristine-relaunch"})
    assert resp.status_code == 200, resp.json()
    run_id = resp.json()["run_id"]
    _wait_terminal(run_id)

    from tcip_mcp.experiments import config_key, lineage_key, read_member

    snapshot = read_member(config_key("exp-pristine-relaunch"))
    assert snapshot["experiment_id"] == "exp-pristine-relaunch"
    lineage = read_member(lineage_key("exp-pristine-relaunch"))
    assert lineage["parent_experiment"] is None


def test_relaunch_route_forks_a_run_s_config_and_names_the_parent(
    tmp_path, monkeypatch, client: TestClient
) -> None:
    """Relaunching a config that already has a run attached mints a fresh id, its own snapshot
    naming itself and its lineage naming the picked config as parent."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", lambda *a, **k: {})
    from tests.tiny_trainer_fixtures import write_regression_dataset

    images_dir, csv_path = write_regression_dataset(
        tmp_path, intensities=[0.0, 1.0], values=[0.1, 0.9])
    labels_dir = tmp_path / "unused_labels"
    labels_dir.mkdir()
    cfg = {
        "model_source": {"builder": "tests.tiny_trainer_fixtures:build_mean_intensity_regressor",
                         "task": "regression", "in_chans": 3},
        "data": {"images_dir": str(images_dir), "csv_path": str(csv_path), "labels_dir": str(labels_dir)},
        "training": {"batch_size": 2, "stages": [{"freeze_to": 0, "epochs": 1}],
                     "mixed_precision": False, "device": "cpu",
                     "checkpoint_every_n_epochs": 0, "early_stopping": {"enabled": False}},
    }
    from tcip_mcp.tools.training_tools import launch_training

    first = launch_training(dict(cfg), str(tmp_path / "out1"))
    assert "error" not in first, first
    parent_id = first["experiment_id"]
    _wait_terminal(first["run_id"])

    resp = client.post("/api/training/runs", json={"experiment_id": parent_id})
    assert resp.status_code == 200, resp.json()
    forked_id = resp.json()["experiment_id"]
    assert forked_id != parent_id
    _wait_terminal(resp.json()["run_id"])

    from tcip_mcp.experiments import config_key, lineage_key, read_member

    snapshot = read_member(config_key(forked_id))
    assert snapshot["experiment_id"] == forked_id
    lineage = read_member(lineage_key(forked_id))
    assert lineage["parent_experiment"] == parent_id


def test_list_runs_route_names_the_run_s_selection_metric(
    tmp_path, monkeypatch, client: TestClient
) -> None:
    """A launched (subprocess-delegated) run's row carries the metric the trainer itself
    stamped on its metrics-log rows, and the best value read back beside it, so the Training
    tab can label its best value instead of showing nothing (the parent's placeholder
    ``inf``) or a name with no value (``None``, the pre-fix disk reconstruction)."""
    import math

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", lambda *a, **k: {})
    from tcip_mcp.tools.training_tools import launch_training
    from tests.tiny_trainer_fixtures import write_regression_dataset

    images_dir, csv_path = write_regression_dataset(
        tmp_path, intensities=[0.0, 1.0], values=[0.1, 0.9])
    labels_dir = tmp_path / "unused_labels"
    labels_dir.mkdir()
    cfg = {
        "model_source": {"builder": "tests.tiny_trainer_fixtures:build_mean_intensity_regressor",
                         "task": "regression", "in_chans": 3},
        "data": {"images_dir": str(images_dir), "csv_path": str(csv_path), "labels_dir": str(labels_dir)},
        "training": {"batch_size": 2, "stages": [{"freeze_to": 0, "epochs": 1}],
                     "mixed_precision": False, "device": "cpu",
                     "checkpoint_every_n_epochs": 0, "early_stopping": {"enabled": False}},
    }
    result = launch_training(cfg, str(tmp_path / "out"))
    assert "error" not in result, result
    _wait_terminal(result["run_id"])

    from tcip_web.routes.training import list_runs_route

    by_id = {r["run_id"]: r for r in list_runs_route()["runs"]}
    row = by_id[result["run_id"]]
    # Regression selects on the training loss by default; there is no evaluation.selection_metric
    # override in this config.
    assert row["best_metric_name"] == "loss"
    assert isinstance(row["best_metric"], (int, float)) and math.isfinite(row["best_metric"])


def test_list_runs_excludes_hpo_trials(monkeypatch) -> None:
    # HPO trial runs (origin='hpo_trial') must not leak into the Training-tab list.
    from tcip_mcp.pipelines.training import run_registry as rr

    monkeypatch.setattr(rr, "_RUNS", {})
    rr.create_run({"model_source": {"builder": "x:y"}}, "out_a", origin="training")
    rr.create_run({"model_source": {"builder": "x:y"}}, "out_b", origin="hpo_trial")

    default = {r["run_id"] for r in rr.list_runs()}
    assert len(default) == 1  # only the standalone training run
    assert all(rr._RUNS[rid].origin == "training" for rid in default)
    assert len(rr.list_runs(include_hpo_trials=True)) == 2  # full set on request
