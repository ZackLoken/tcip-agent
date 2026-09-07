"""The metrics rows a run writes are the rows the Training stream serves.

Both sides here are the real implementations: ``log_metrics`` appends to the run's own log and
the WebSocket route reads that same log by the experiment's own id and replays it, so a change
to the row shape or to where the log lives shows up as a disagreement instead of passing against
a hand-built file.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from tcip_web.app import app


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


def test_logged_rows_reach_the_training_stream_reader_with_their_epoch_and_values(tmp_path):
    from tcip_mcp.experiments import create_experiment, log_metrics, update_status

    run_id = "exp-021-currant-bud-det"
    create_experiment(run_id, {"model_source": {"builder": "my_models:bud_det"}})
    log_metrics(run_id, 3, {"loss": 0.94, "val_map50": 0.28})
    log_metrics(run_id, 7, {"loss": 0.31, "val_map50": 0.66})
    update_status(run_id, "completed")  # a terminal run ends the stream after one tick

    with _client().websocket_connect(
        f"ws://127.0.0.1/api/training/runs/{run_id}/stream?project_root={tmp_path}",
    ) as ws:
        frames = _drain(ws)

    rows = [f["row"] for f in frames if f["type"] == "metric"]
    assert len(rows) == 2
    assert [r.get("epoch") for r in rows] == [3, 7]
    assert [r.get("val_map50") for r in rows] == [0.28, 0.66]
    assert rows[-1].get("loss") == 0.31
    assert all(r.get("timestamp") for r in rows)
    assert frames[-1]["status"]["status"] == "completed"


def test_training_stream_serves_a_forked_experiment_by_its_own_minted_id(tmp_path):
    """A fork's id (``<parent>_<minted>``, minted when the parent already has history) is the one
    id the record is addressed by everywhere, the stream included: no separate run id to resolve
    it through."""
    from tcip_mcp.experiments import (
        create_experiment, log_metrics, mint_experiment_id, update_status,
    )

    forked_id = f"exp-022-burr-det_{mint_experiment_id()}"
    create_experiment(forked_id, {"model_source": {"builder": "my_models:burr_det"}})
    log_metrics(forked_id, 1, {"loss": 1.2})
    update_status(forked_id, "completed")

    with _client().websocket_connect(
        f"ws://127.0.0.1/api/training/runs/{forked_id}/stream?project_root={tmp_path}",
    ) as ws:
        frames = _drain(ws)

    rows = [f["row"] for f in frames if f["type"] == "metric"]
    assert [r.get("loss") for r in rows] == [1.2]


def test_stream_drains_a_row_that_lands_between_the_read_and_the_terminal_check(
    tmp_path, monkeypatch,
):
    """A row appended in the window between the stream's last log read and its terminal
    status check must still reach the browser, ahead of the status frame that ends the
    stream, rather than being dropped by it."""
    from tcip_mcp.experiments import create_experiment, log_metrics, update_status
    from tcip_mcp.tools import training_tools

    run_id = "exp-023-walnut-shell-det"
    create_experiment(run_id, {"model_source": {"builder": "my_models:shell_det"}})
    log_metrics(run_id, 1, {"loss": 0.5})

    real_monitor_training = training_tools.monitor_training
    appended = {"done": False}

    def fake_monitor_training(rid):
        if not appended["done"]:
            appended["done"] = True
            log_metrics(run_id, 2, {"loss": 0.2})
            update_status(run_id, "completed")
        return real_monitor_training(rid)

    monkeypatch.setattr(training_tools, "monitor_training", fake_monitor_training)

    with _client().websocket_connect(
        f"ws://127.0.0.1/api/training/runs/{run_id}/stream?project_root={tmp_path}",
    ) as ws:
        frames = _drain(ws)

    rows = [f["row"] for f in frames if f["type"] == "metric"]
    assert [r.get("epoch") for r in rows] == [1, 2]
    assert frames[-1]["type"] == "status"
    assert frames[-1]["status"]["status"] == "completed"


def test_training_stream_serves_no_metric_frames_for_a_run_no_record_claims(tmp_path):
    """An id no experiment record exists for still resolves a metrics key (nothing appended to
    it), so the stream replays nothing and sends only the terminal status frame naming the
    unresolved run."""
    with _client().websocket_connect(
        f"ws://127.0.0.1/api/training/runs/never-launched/stream?project_root={tmp_path}",
    ) as ws:
        frames = _drain(ws)

    assert [f for f in frames if f["type"] == "metric"] == []
    assert frames[-1]["type"] == "status"
    assert frames[-1]["status"] is None
    assert frames[-1]["error"]
