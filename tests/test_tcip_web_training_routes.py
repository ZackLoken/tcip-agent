"""Integration tests for the Training routes."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_web.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def test_validate_flags_missing_sections(client: TestClient) -> None:
    resp = client.post("/api/training/validate", json={"config": {}})
    body = resp.json()
    assert body["valid"] is False
    assert any("model_source" in s for s in body["issues"])


def test_validates_verdict_tracks_the_issues_the_config_actually_raises(
    client: TestClient, tmp_path: Path,
) -> None:
    """The verdict the route reports is the accumulated issue list's own answer, in both
    directions: a buildable config is accepted with nothing against it, and a single defect in an
    otherwise identical config refuses it and says which field."""
    import copy

    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": {"images_dir": str(images), "labels_dir": str(labels), "task": "detection"},
        "training": {"batch_size": 2, "stages": [{"lr": 1e-3, "epochs": 1}]},
    }
    body = client.post("/api/training/validate", json={"config": cfg}).json()
    assert body["issues"] == [], body
    assert body["valid"] is True

    defective = copy.deepcopy(cfg)
    defective["training"]["batch_size"] = 0
    refused = client.post("/api/training/validate", json={"config": defective}).json()
    assert refused["valid"] is False
    assert any("batch_size" in issue for issue in refused["issues"]), refused["issues"]


def test_list_runs_returns_shape(client: TestClient) -> None:
    resp = client.get("/api/training/runs")
    assert resp.status_code == 200
    body = resp.json()
    assert "runs" in body


def test_metrics_route_handles_missing_file(client: TestClient, tmp_path: Path) -> None:
    # When no metrics.jsonl exists yet, the route reports exists=False with empty rows.
    resp = client.get(
        "/api/training/runs/foo-xxx/metrics",
        params={"project_root": str(tmp_path)},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["exists"] is False
    assert body["metrics"] == []


def test_metrics_route_reads_the_rows_the_run_logged(client: TestClient, tmp_path: Path) -> None:
    from tcip_mcp.experiments import create_experiment, log_metrics

    run_id = "exp-abc"
    create_experiment(run_id, {"model_source": {"builder": "m:f"}})
    log_metrics(run_id, 1, {"loss": 0.9})
    log_metrics(run_id, 2, {"loss": 0.4})

    resp = client.get(
        f"/api/training/runs/{run_id}/metrics",
        params={"project_root": str(tmp_path)},
    )
    body = resp.json()
    assert body["exists"] is True
    assert [(r["epoch"], r["loss"]) for r in body["metrics"]] == [(1, 0.9), (2, 0.4)]


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

    def fake_status(run_id: str) -> dict:
        return {"run_id": run_id, "status": "running", "output_dir": str(tmp_path)}

    def fake_launch(logdir: str, run_id: str | None = None) -> dict:
        calls.append((logdir, run_id or ""))
        return {"url": "http://localhost:6006", "port": 6006, "pid": 1, "logdir": logdir}

    monkeypatch.setattr("tcip_mcp.tools.training_tools.check_training_status", fake_status)
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", fake_launch
    )

    resp = client.post("/api/training/runs/run-42/tensorboard", json={})
    assert resp.status_code == 200
    assert resp.json()["url"] == "http://localhost:6006"
    assert calls == [(f"{tmp_path}/tensorboard", "run-42")]


def test_list_runs_reconstructs_from_experiments(tmp_path, monkeypatch) -> None:
    """No live entry for either experiment (this process never held them in ``_RUNS``): the
    route's own rows now come from ``list_training_runs``'s unified disk enumeration with no
    patch needed. A genuine training experiment left 'running' by a crash resurfaces as
    'interrupted'; a review-feedback experiment (no model_source) is not a training run and is
    excluded.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
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
    """Post-unification the route adds nothing of its own: its rows equal the tool's rows,
    exactly. Before unification the route's own reconstruction added rows the tool lacked."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    from tcip_mcp.experiments import create_experiment, update_status
    from tcip_mcp.tools.training_tools import list_training_runs
    from tcip_web.routes.training import list_runs_route

    create_experiment("exp-route-parity", {"model_source": {"builder": "my_models:chestnut_burr_det"}})
    update_status("exp-route-parity", "running")

    assert list_runs_route()["runs"] == list_training_runs()["runs"]


def test_never_launched_experiment_is_absent_from_the_route(tmp_path, monkeypatch) -> None:
    """A pre-created experiment (state 'created', no run_id stamp, no metrics logged) never
    launched and must not list as a run at all, interrupted or otherwise."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    from tcip_mcp.experiments import create_experiment
    from tcip_web.routes.training import list_runs_route

    create_experiment("exp-never-launched", {"model_source": {"builder": "my_models:chestnut_burr_det"}})

    by_id = {r["run_id"]: r for r in list_runs_route()["runs"]}
    assert "exp-never-launched" not in by_id


def test_list_runs_excludes_hpo_trials(monkeypatch) -> None:
    # HPO trial runs (origin='hpo_trial') must not leak into the Training-tab list.
    from tcip_mcp.pipelines.training import generic_trainer as gt

    monkeypatch.setattr(gt, "_RUNS", {})
    gt.create_run({"model_source": {"builder": "x:y"}}, "out_a", origin="training")
    gt.create_run({"model_source": {"builder": "x:y"}}, "out_b", origin="hpo_trial")

    default = {r["run_id"] for r in gt.list_runs()}
    assert len(default) == 1  # only the standalone training run
    assert all(gt._RUNS[rid].origin == "training" for rid in default)
    assert len(gt.list_runs(include_hpo_trials=True)) == 2  # full set on request
