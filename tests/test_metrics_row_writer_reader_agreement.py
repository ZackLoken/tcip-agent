"""The metrics rows a run writes are the rows the Training stream serves.

Both sides here are the real implementations: the run's body appends to its own metrics log
through the envelope, and the WebSocket route reads that same log in the run's own directory and
replays it, so a change to the row shape or to where the log lives shows up as a disagreement
instead of passing against a hand-built file.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from tcip_web.app import app
from tests._verified_checkpoint_fixtures import (
    detection_config, finished_run, log_epoch, opened_run,
)


def _client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _drain(ws) -> list[dict]:
    """Every frame the stream sends, up to and including the terminal status frame."""
    frames = []
    while True:
        msg = ws.receive_json()
        frames.append(msg)
        if msg["type"] == "status":
            break
    return frames


def test_logged_rows_reach_the_training_stream_reader_with_their_epoch_and_values(
    tmp_path, opened_project,
):
    run_id = "exp-021-currant-bud-det"
    finished_run(tmp_path, experiment_id=run_id, rows=[
        {"epoch": 3, "loss": 0.94, "val_map50": 0.28},
        {"epoch": 7, "loss": 0.31, "val_map50": 0.66}])

    with _client().websocket_connect(
        f"ws://127.0.0.1/api/training/runs/{run_id}/stream",
    ) as ws:
        frames = _drain(ws)

    rows = [f["row"] for f in frames if f["type"] == "metric"]
    assert len(rows) == 2
    assert [r.get("epoch") for r in rows] == [3, 7]
    assert [r.get("val_map50") for r in rows] == [0.28, 0.66]
    assert rows[-1].get("loss") == 0.31
    assert all(r.get("timestamp") for r in rows)
    assert frames[-1]["status"]["status"] == "completed"


def test_training_stream_serves_a_relaunched_run_by_its_own_minted_id(tmp_path, opened_project):
    """A relaunch's id is minted fresh and is the one id its directory is addressed by
    everywhere, the stream included: no separate run id to resolve it through."""
    from tcip_mcp.experiments import mint_experiment_id

    parent = finished_run(tmp_path)
    relaunched_id = mint_experiment_id()
    finished_run(tmp_path, experiment_id=relaunched_id, rows=[{"epoch": 1, "loss": 1.2}])
    assert relaunched_id != parent.name

    with _client().websocket_connect(
        f"ws://127.0.0.1/api/training/runs/{relaunched_id}/stream",
    ) as ws:
        frames = _drain(ws)

    rows = [f["row"] for f in frames if f["type"] == "metric"]
    assert [r.get("loss") for r in rows] == [1.2]


def test_stream_drains_a_row_that_lands_between_the_read_and_the_terminal_check(
    tmp_path, opened_project, monkeypatch,
):
    """A row the run appends as it ends, after the stream's last log read, still reaches the
    browser ahead of the status frame that ends the stream: the run ends between the stream's
    read and its next observation."""
    from tcip_mcp import experiments
    from tcip_mcp.pipelines.training.envelope import TrainContext, run_training_envelope
    from tcip_mcp.pipelines.training.run_registry import TrainRun

    run_id = "exp-023-walnut-shell-det"
    run_dir = opened_run(tmp_path, detection_config(
        tmp_path / "data", training_source="tests.bespoke_models:save_built_weights",
        fixture_rows=[{"epoch": 2, "loss": 0.2}]), experiment_id=run_id)
    log_epoch(run_dir, 1, {"loss": 0.5})

    real_observe = experiments.observe
    calls = {"n": 0}

    def finish_then_observe(directory, *args):
        calls["n"] += 1
        if calls["n"] == 2:
            record = real_observe(directory).record
            run = TrainRun(id=run_id, config=record["config"],
                           objective=record["resolved"]["objective"], project=tmp_path,
                           output_dir=str(run_dir))
            run_training_envelope(TrainContext(run=run, train_loader=None))
        return real_observe(directory, *args)

    monkeypatch.setattr(experiments, "observe", finish_then_observe)

    with _client().websocket_connect(
        f"ws://127.0.0.1/api/training/runs/{run_id}/stream",
    ) as ws:
        frames = _drain(ws)

    rows = [f["row"] for f in frames if f["type"] == "metric"]
    assert [r.get("epoch") for r in rows] == [1, 2]
    assert frames[-1]["type"] == "status"
    assert frames[-1]["status"]["status"] == "completed"


def test_training_stream_serves_no_metric_frames_for_a_run_no_directory_holds(opened_project):
    """An id no run directory answers for replays nothing and sends only the terminal status
    frame naming the unresolved run."""
    with _client().websocket_connect(
        "ws://127.0.0.1/api/training/runs/never-launched/stream",
    ) as ws:
        frames = _drain(ws)

    assert [f for f in frames if f["type"] == "metric"] == []
    assert frames[-1]["type"] == "status"
    assert frames[-1]["status"] is None
    assert frames[-1]["error"]
