"""Integration tests for the Training routes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_web.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


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


def test_metrics_route_reads_jsonl(client: TestClient, tmp_path: Path) -> None:
    run_id = "exp-abc"
    metrics_path = tmp_path / ".tcip" / "experiments" / run_id / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True)
    rows = [{"epoch": 1, "loss": 0.9}, {"epoch": 2, "loss": 0.4}]
    with metrics_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    resp = client.get(
        f"/api/training/runs/{run_id}/metrics",
        params={"project_root": str(tmp_path)},
    )
    body = resp.json()
    assert body["exists"] is True
    assert body["metrics"] == rows


def test_read_metrics_after_resumes_from_byte_offset(tmp_path: Path) -> None:
    # The websocket poll must remember a byte offset and seek there, not re-parse the file
    # from the start every tick: a resumed read only returns rows written since that offset.
    from tcip_web.routes.training import _read_metrics_after

    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps({"epoch": 1}) + "\n", encoding="utf-8")

    rows, offset = _read_metrics_after(path, 0)
    assert rows == [{"epoch": 1}]
    assert offset == path.stat().st_size

    # Nothing new yet: re-polling from the remembered offset yields no rows.
    rows2, offset2 = _read_metrics_after(path, offset)
    assert rows2 == []
    assert offset2 == offset

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"epoch": 2}) + "\n")

    rows3, offset3 = _read_metrics_after(path, offset2)
    assert rows3 == [{"epoch": 2}]
    assert offset3 == path.stat().st_size


def test_read_metrics_after_defers_partial_trailing_line(tmp_path: Path) -> None:
    # A writer's line lands on disk before its trailing newline flushes; a seek-based reader
    # must not consume that partial line, or it would permanently skip the completed row.
    from tcip_web.routes.training import _read_metrics_after

    path = tmp_path / "metrics.jsonl"
    path.write_bytes(json.dumps({"epoch": 1}).encode("utf-8") + b"\n" + b'{"epoch": 2')

    rows, offset = _read_metrics_after(path, 0)
    assert rows == [{"epoch": 1}]
    assert offset < path.stat().st_size  # the partial line was not consumed

    with path.open("ab") as f:
        f.write(b', "loss": 0.1}\n')

    rows2, offset2 = _read_metrics_after(path, offset)
    assert rows2 == [{"epoch": 2, "loss": 0.1}]
    assert offset2 == path.stat().st_size


def test_compare_route_handles_empty_ids(client: TestClient) -> None:
    resp = client.post("/api/training/compare", json={"experiment_ids": []})
    assert resp.status_code == 200
    # body schema is up to compare_experiments; we only assert the route returns JSON.
    assert isinstance(resp.json(), dict)


def test_cancel_unknown_run_returns_404(client: TestClient) -> None:
    resp = client.post("/api/training/runs/does-not-exist/cancel")
    assert resp.status_code == 404


def test_tensorboard_route_404s_for_unknown_run(client: TestClient) -> None:
    resp = client.post("/api/training/runs/does-not-exist/tensorboard")
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

    resp = client.post("/api/training/runs/run-42/tensorboard")
    assert resp.status_code == 200
    assert resp.json()["url"] == "http://localhost:6006"
    assert calls == [(f"{tmp_path}/tensorboard", "run-42")]


def test_list_runs_reconstructs_from_experiments(tmp_path, monkeypatch) -> None:
    # No live runs (post-restart): the list is rebuilt from .tcip/experiments/. A genuine
    # training experiment left 'running' by a crash resurfaces as 'interrupted'; a
    # review-feedback experiment (no model_source) is not a training run and is excluded.
    monkeypatch.chdir(tmp_path)
    import json as _json

    from tcip_web.routes import training

    exp = tmp_path / ".tcip" / "experiments" / "run_1"
    exp.mkdir(parents=True)
    (exp / "config.json").write_text(
        _json.dumps({"model_source": {"builder": "torchvision:fasterrcnn_resnet50_fpn"}})
    )
    (exp / "status.json").write_text(_json.dumps({"state": "running"}))

    fb = tmp_path / ".tcip" / "experiments" / "fb_1"
    fb.mkdir(parents=True)
    (fb / "config.json").write_text(_json.dumps({"source": "review_feedback"}))
    (fb / "status.json").write_text(_json.dumps({"state": "completed"}))

    monkeypatch.setattr(
        "tcip_mcp.tools.training_tools.list_training_runs", lambda: {"runs": []}
    )
    body = training.list_runs_route()
    by_id = {r["run_id"]: r for r in body["runs"]}
    assert by_id["run_1"]["status"] == "interrupted"  # dead process -> interrupted
    assert "fb_1" not in by_id  # review-feedback experiment is not a training run


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
