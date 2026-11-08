"""The metrics rows a run writes are the rows the Training stream serves.

Both sides are the real implementations: the run's body appends to its own metrics log through
the envelope, and the WebSocket route replays that log from the run's own directory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests._verified_checkpoint_fixtures import (
    detection_config,
    finished_run,
    log_epoch,
    opened_run,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest
    from fastapi.testclient import TestClient
    from starlette.testclient import WebSocketTestSession

    from tcip_mcp.experiments import RunObservation, RunRecord


def _drain(ws: WebSocketTestSession) -> list[dict]:
    """Return the frames the stream sends, through the status or error frame that ends it."""
    frames = []
    while True:
        msg = ws.receive_json()
        frames.append(msg)
        if msg["type"] in {"status", "error"}:
            break
    return frames


def test_logged_rows_reach_the_training_stream_reader_with_their_epoch_and_values(
    tmp_path: Path,
    opened_project: Path,
    client: TestClient,
) -> None:
    run_id = "exp-021-currant-bud-det"
    finished_run(
        tmp_path,
        experiment_id=run_id,
        rows=[
            {"epoch": 3, "loss": 0.94, "val_map50": 0.28},
            {"epoch": 7, "loss": 0.31, "val_map50": 0.66},
        ],
    )

    with client.websocket_connect(
        f"ws://127.0.0.1/api/training/runs/{run_id}/stream",
    ) as ws:
        frames = _drain(ws)

    rows = [f["row"] for f in frames if f["type"] == "metric"]
    assert len(rows) == 2
    assert [r.get("epoch") for r in rows] == [3, 7]
    assert [r.get("val_map50") for r in rows] == [0.28, 0.66]
    assert rows[-1].get("loss") == 0.31
    assert all(r.get("timestamp") for r in rows)
    assert frames[-1]["status"]["state"] == "completed"


def test_training_stream_serves_a_relaunched_run_by_its_own_minted_id(
    tmp_path: Path,
    opened_project: Path,
    client: TestClient,
) -> None:
    """The training stream serves a relaunched run by its own minted id."""
    from tcip_mcp.experiments import mint_experiment_id

    parent = finished_run(tmp_path)
    relaunched_id = mint_experiment_id()
    finished_run(
        tmp_path, experiment_id=relaunched_id, rows=[{"epoch": 1, "loss": 1.2}]
    )
    assert relaunched_id != parent.name

    with client.websocket_connect(
        f"ws://127.0.0.1/api/training/runs/{relaunched_id}/stream",
    ) as ws:
        frames = _drain(ws)

    rows = [f["row"] for f in frames if f["type"] == "metric"]
    assert [r.get("loss") for r in rows] == [1.2]


def test_stream_drains_a_row_that_lands_between_the_read_and_the_terminal_check(
    tmp_path: Path,
    opened_project: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stream drains a row landing between its read and its terminal check.

    A row the run appends as it ends, after the stream's last log read, still reaches the
    browser ahead of the status frame that ends the stream: the run ends between the stream's
    read and its next observation.
    """
    from tcip_mcp import experiments
    from tcip_mcp.pipelines.training.envelope import (
        TrainContext,
        run_training_envelope,
    )
    from tcip_mcp.pipelines.training.run_registry import observed_run
    from tests._chain_fixtures import SAVE_BUILT_WEIGHTS

    run_id = "exp-023-walnut-shell-det"
    run_dir = opened_run(
        tmp_path,
        detection_config(
            tmp_path / "data",
            project=tmp_path,
            training_source=SAVE_BUILT_WEIGHTS,
            fixture_rows=[{"epoch": 2, "loss": 0.2}],
        ),
        experiment_id=run_id,
    )
    log_epoch(run_dir, 1, {"loss": 0.5})

    real_observe = experiments.observe_run
    calls = {"n": 0}

    def finish_then_observe(directory: Path) -> RunObservation[RunRecord]:
        calls["n"] += 1
        if calls["n"] == 2:
            run_training_envelope(
                TrainContext(
                    run=observed_run(real_observe(directory)),
                    train_loader=None,
                )
            )
        return real_observe(directory)

    monkeypatch.setattr(experiments, "observe_run", finish_then_observe)

    with client.websocket_connect(
        f"ws://127.0.0.1/api/training/runs/{run_id}/stream",
    ) as ws:
        frames = _drain(ws)

    rows = [f["row"] for f in frames if f["type"] == "metric"]
    assert [r.get("epoch") for r in rows] == [1, 2]
    assert frames[-1]["type"] == "status"
    assert frames[-1]["status"]["state"] == "completed"


def test_the_log_reader_refuses_the_non_finite_row_its_writer_refuses(
    tmp_path: Path,
) -> None:
    """A row carrying a non-finite number is refused by the log's writer and by its reader alike.

    The writer (``metrics_row``) refuses it naming the position; a line spelling it, which
    ``json`` decodes, is refused by ``read_rows`` as invalid at its line, naming the position too.
    """
    import pytest

    from tcip_mcp.experiments import metrics_row, read_rows
    from tcip_store import ReadRefusedError

    with pytest.raises(ValueError, match=r"val_per_class_iou\.dict\.0"):
        metrics_row({"val_per_class_iou": {"0": float("nan")}}, 1, None)

    log = tmp_path / "metrics.jsonl"
    log.write_bytes(
        b'{"epoch": 1, "timestamp": "2026-01-01T00:00:00+00:00", '
        b'"val_per_class_iou": {"0": NaN}}\n'
    )
    with pytest.raises(
        ReadRefusedError, match=r"val_per_class_iou\.dict\.0"
    ) as refused:
        read_rows(log)
    assert (refused.value.served.where, refused.value.served.refusal.kind) == (
        f"{log}:1",
        "invalid",
    )


def test_a_row_read_past_an_offset_is_refused_at_its_line_counted_from_the_logs_start(
    tmp_path: Path,
) -> None:
    """A row read past an offset that will not decode is refused at its line in the whole log.

    Two epochs logged through the envelope's sink are read, a line that does not decode is
    appended, and the read from the offset the first read returned names that line's number in
    the file, not its index past the offset.
    """
    import pytest

    from tcip_mcp.experiments import METRICS_FILE, read_rows
    from tcip_store import ReadRefusedError

    run_dir = opened_run(
        tmp_path,
        detection_config(tmp_path / "data", project=tmp_path),
        experiment_id="exp-offset-read",
    )
    log_epoch(run_dir, 1, {"loss": 0.9})
    log_epoch(run_dir, 2, {"loss": 0.5})
    log = run_dir / METRICS_FILE
    rows, offset = read_rows(log)
    assert [row.epoch for row in rows] == [1, 2]
    line = log.read_bytes().count(b"\n") + 1
    with log.open("ab") as handle:
        handle.write(b"{not a row\n")

    with pytest.raises(ReadRefusedError) as refused:
        read_rows(log, after=offset)
    assert line > 1
    assert refused.value.served.where == f"{log}:{line}"


def test_training_stream_serves_no_metric_frames_for_a_run_no_directory_holds(
    opened_project: Path,
    client: TestClient,
) -> None:
    """A stream for an id no run directory holds serves no metric frame.

    Its one frame is the error frame naming the id.
    """
    from tcip_web.routes.training import TrainingErrorFrame

    with client.websocket_connect(
        "ws://127.0.0.1/api/training/runs/never-launched/stream",
    ) as ws:
        frames = _drain(ws)

    (frame,) = frames
    refused = TrainingErrorFrame.model_validate(frame)
    assert refused.experiment_id == "never-launched"
    assert "never-launched" in refused.error
