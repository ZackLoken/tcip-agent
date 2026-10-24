"""Integration tests for the Training routes, each serving the project the backend has open."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from fastapi.testclient import TestClient

from tests._training_values import fifo_search, sweep_space

if TYPE_CHECKING:
    from collections.abc import (
        AsyncIterator,
        Awaitable,
        Callable,
        Mapping,
    )

    import httpx
    from starlette.types import Receive

    from tcip_mcp.pipelines.training.envelope import TrainContext


def test_the_run_list_refuses_while_no_project_is_open(
    client: TestClient,
) -> None:
    resp = client.get("/api/training/runs")
    assert resp.status_code == 409


def _opened(
    experiment_id: str,
    tmp_path: Path,
    builder: str = "my_models:chestnut_burr_det",
) -> Path:
    """Open a detector run built by ``builder`` over two frames of its own under ``tmp_path``.

    The builder's module is an empty file of its own, and the run is opened by the launcher's
    own producer and writer.
    """
    from tests._chain_fixtures import FRAME_STATISTICS
    from tests._verified_checkpoint_fixtures import (
        detection_config,
        opened_run,
    )

    module = Path(*builder.partition(":")[0].split("."))
    code = (tmp_path / f"{experiment_id}-code" / module).with_suffix(".py")
    code.parent.mkdir(parents=True, exist_ok=True)
    code.write_text("", encoding="utf-8")
    config = detection_config(
        tmp_path / f"{experiment_id}-data",
        project=tmp_path,
        model_source={
            "builder": builder,
            "source_files": [str(code)],
            "task": "detection",
            "normalization": FRAME_STATISTICS,
        },
    )
    return opened_run(tmp_path, config, experiment_id=experiment_id)


def test_a_listed_run_row_carries_what_the_launch_picker_shows(
    opened_client: TestClient, opened_project: Path
) -> None:
    _opened("exp-picker-1", opened_project, builder="my_models:leaf_det")

    resp = opened_client.get("/api/training/runs")

    assert resp.status_code == 200
    (row,) = resp.json()["runs"]
    assert (row["experiment_id"], row["builder"], row["task"]) == (
        "exp-picker-1",
        "my_models:leaf_det",
        "detection",
    )
    assert row["state"] == "running"  # launched, its heartbeat still fresh
    assert (row["relaunched_from"], row["sweep"]) == (None, None)
    assert resp.json()["sweeps"] == []


def test_a_run_is_read_by_its_id_and_an_unknown_id_refuses_naming_it(
    opened_client: TestClient, opened_project: Path
) -> None:
    _opened("exp-read", opened_project)

    resp = opened_client.get("/api/training/runs/exp-read")
    assert resp.status_code == 200
    assert (
        resp.json()["run"]["experiment_id"],
        resp.json()["run"]["state"],
    ) == ("exp-read", "running")

    missing = opened_client.get("/api/training/runs/exp-missing")
    assert missing.status_code == 404
    assert "exp-missing" in missing.json()["detail"]


def test_relaunch_route_404s_for_an_unknown_experiment(
    opened_client: TestClient,
) -> None:
    resp = opened_client.post(
        "/api/training/runs",
        json={"relaunched_from": "nope", "user": "tester"},
    )
    assert resp.status_code == 404


def test_relaunch_route_422s_with_preflight_issues_for_a_refused_config(
    tmp_path: Path, opened_client: TestClient
) -> None:
    _opened("exp-refused", tmp_path, builder="not.a:module")

    resp = opened_client.post(
        "/api/training/runs",
        json={"relaunched_from": "exp-refused", "user": "tester"},
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["issues"]


def test_metrics_stream_reports_no_frames_for_a_run_no_record_claims(
    opened_client: TestClient,
) -> None:
    with opened_client.websocket_connect(
        "ws://127.0.0.1/api/training/runs/foo-xxx/stream"
    ) as ws:
        msg = ws.receive_json()
    assert msg["type"] == "status"
    assert msg["status"] is None
    assert msg["error"]


def _completed_run_with_rows(project: Path, run_id: str) -> Path:
    """Return a completed run under ``project`` whose body logged two metrics rows.

    The child's own entry runs it, its body logging epoch 1 at loss 0.9 and epoch 2 at loss 0.4.
    """
    from tests._verified_checkpoint_fixtures import finished_run

    return finished_run(
        project,
        experiment_id=run_id,
        rows=[{"epoch": 1, "loss": 0.9}, {"epoch": 2, "loss": 0.4}],
    )


def test_metrics_stream_serves_the_rows_the_run_logged(
    opened_client: TestClient, tmp_path: Path
) -> None:
    run_id = "exp-abc"
    _completed_run_with_rows(tmp_path, run_id)

    frames = []
    with opened_client.websocket_connect(
        f"ws://127.0.0.1/api/training/runs/{run_id}/stream"
    ) as ws:
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

    finished_run(
        tmp_path,
        experiment_id="exp-one-epoch",
        rows=[
            {"epoch": 1, "loss": 0.9},
            {"epoch": 1, "selection": 0.5, "selection_metric": "loss"},
        ],
    )
    sent: list[dict] = []

    class _Socket:
        async def send_json(self, payload: dict) -> None:
            sent.append(payload)

    asyncio.run(
        _stream_metrics(_Socket(), tmp_path, "exp-one-epoch", poll_seconds=0.0)
    )

    (row,) = [msg["row"] for msg in sent if msg["type"] == "metric"]
    assert (row["epoch"], row["loss"], row["selection"]) == (1, 0.9, 0.5)


def test_a_finite_value_logged_after_a_non_finite_one_at_one_epoch_streams_alone(
    tmp_path: Path,
) -> None:
    """A loss logged non-finite and then finite at one epoch streams as the finite value.

    The one streamed row reads loss 0.1 and carries no not-finite marker for the loss.
    """
    import asyncio

    from tcip_mcp.experiments import write_final_status
    from tcip_store.values import NOT_FINITE_SUFFIX
    from tcip_web.routes.training import _stream_metrics
    from tests._verified_checkpoint_fixtures import log_epoch

    run_dir = _opened("exp-recovered", tmp_path)
    log_epoch(run_dir, 2, {"loss": float("nan")})
    log_epoch(run_dir, 2, {"loss": 0.1})
    write_final_status(
        run_dir, "failed", "stopped after the second row", checkpoint=None
    )
    sent: list[dict] = []

    class _Socket:
        async def send_json(self, payload: dict) -> None:
            sent.append(payload)

    asyncio.run(
        _stream_metrics(_Socket(), tmp_path, "exp-recovered", poll_seconds=0.0)
    )

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

    asyncio.run(
        _stream_metrics(_Socket(), tmp_path, "exp-batches", poll_seconds=0.0)
    )

    (epoch,) = [msg["row"] for msg in sent if msg["type"] == "metric"]
    assert (epoch["epoch"], epoch["train_loss"]) == (
        1,
        0.75,
    ) and STEP_KEY not in epoch
    (batch,) = [msg["row"] for msg in sent if msg["type"] == "batch"]
    assert (batch[STEP_KEY], batch["epoch"], batch["batch_loss"]) == (
        3,
        2,
        0.6,
    )
    assert (
        sent[-1]["type"] == "status"
        and sent[-1]["status"]["current_epoch"] == 1
    )


def test_metrics_stream_pushes_complete_entries_and_defers_a_partial_one(
    tmp_path: Path,
) -> None:
    """The metrics stream pushes whole rows and holds back an entry still being appended.

    With two whole rows and a third lacking its terminator, the stream pushes the two rows and
    then the status frame.
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
    """The metrics stream reads the log off the event loop's own thread."""
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
    opened_client: TestClient,
    tmp_path: Path,
) -> None:
    _completed_run_with_rows(tmp_path, "exp-left")
    _completed_run_with_rows(tmp_path, "exp-right")

    resp = opened_client.post(
        "/api/training/compare",
        json={"experiment_ids": ["exp-left", "exp-right", "exp-gone"]},
    )

    assert resp.status_code == 200
    left, right, gone = resp.json()["experiments"]
    assert (left["experiment_id"], left["state"], left["n_epochs"]) == (
        "exp-left",
        "completed",
        2,
    )
    assert (right["experiment_id"], right["state"]) == (
        "exp-right",
        "completed",
    )
    assert gone == {
        "experiment_id": "exp-gone",
        "error": "Experiment not found: exp-gone",
    }
    assert resp.json()["same_dataset_fingerprint"] is None


def test_metric_directions_route_answers_the_declared_table(
    opened_client: TestClient,
) -> None:
    """The route answers the schema's own declared-direction table."""
    from tcip_mcp.pipelines.schemas import HIGHER_IS_BETTER_BY_METRIC

    resp = opened_client.get("/api/training/metric-directions")
    assert resp.status_code == 200
    assert resp.json()["higher_is_better"] == HIGHER_IS_BETTER_BY_METRIC


def test_cancel_unknown_run_returns_404(opened_client: TestClient) -> None:
    resp = opened_client.post(
        "/api/training/runs/does-not-exist/cancel", json={"user": "tester"}
    )
    assert resp.status_code == 404


def test_a_cancel_naming_its_person_is_admitted_and_its_line_names_them(
    tmp_path: Path, opened_client: TestClient
) -> None:
    from tcip_mcp.audit import audit_log_key
    from tcip_store import read_log
    from tests._verified_checkpoint_fixtures import (
        detection_config,
        opened_run,
    )

    opened_run(
        tmp_path,
        detection_config(tmp_path / "gui-data", project=tmp_path),
        experiment_id="exp-gui-cancel",
    )
    before = len(read_log(audit_log_key(tmp_path)).records)

    resp = opened_client.post(
        "/api/training/runs/exp-gui-cancel/cancel", json={"user": "tester"}
    )

    assert resp.status_code == 200, resp.text
    (line,) = read_log(audit_log_key(tmp_path)).records[before:]
    assert (line["tool"], line["actor"]) == ("cancel_training", "user:tester")


def test_tensorboard_route_404s_for_unknown_run(
    opened_client: TestClient,
) -> None:
    resp = opened_client.post(
        "/api/training/runs/does-not-exist/tensorboard", json={}
    )
    assert resp.status_code == 404
    assert "does-not-exist" in resp.json()["detail"]


def test_tensorboard_route_launches_over_the_run_s_own_board(
    opened_client: TestClient,
    tmp_path: Path,
    tb_launches: list[Path],
) -> None:
    """The route launches over the run's own directory and answers what the launch serves.

    The frame's URL comes from a TensorBoard this process started, so the route reaches
    ``launch_tensorboard`` with the run's directory and answers the TensorBoard it tracks.
    """
    from tcip_mcp.experiments import board_of
    from tcip_mcp.pipelines.training.tensorboard_manager import running

    run_dir = _opened("run-42", tmp_path)
    board_of(run_dir).mkdir()
    (board_of(run_dir) / "events.out.tfevents.1.host").write_bytes(b"")

    resp = opened_client.post("/api/training/runs/run-42/tensorboard", json={})
    assert resp.status_code == 200
    serving = running(run_dir)
    assert serving is not None
    assert resp.json() == serving.model_dump()
    assert tb_launches == [run_dir]


def test_a_board_path_reaches_the_tensorboard_serving_it_as_received(
    opened_client: TestClient, tmp_path: Path
) -> None:
    """The board path answers what the TensorBoard serving it answers on its own port.

    A real TensorBoard the launch route starts over a run's one-scalar event file answers its
    index page and a pre-gzipped asset through the backend byte for byte, each with the
    ``Content-Encoding`` it was served with and without the ``Connection`` header it sends, which
    belongs to its own connection; a form-encoded data post passes whole, so one naming no tag is
    refused by the TensorBoard and one naming the tag is answered.
    """
    import httpx
    from torch.utils.tensorboard import SummaryWriter

    from tcip_mcp.experiments import board_of
    from tcip_mcp.pipelines.training.tensorboard_manager import (
        stop_tensorboard,
    )
    from tcip_mcp.web_client import LOOPBACK_HOST

    run_dir = _opened("run-board", tmp_path)
    with SummaryWriter(str(board_of(run_dir))) as writer:
        writer.add_scalar("loss", 0.5, 1)

    launched = opened_client.post(
        "/api/training/runs/run-board/tensorboard", json={}
    ).json()
    try:
        url = launched["url"]
        own_port = f"http://{LOOPBACK_HOST}:{launched['port']}"
        for page, encoding in ((url, "identity"), (f"{url}index.js", "gzip")):
            with (
                opened_client.stream("GET", page) as forwarded,
                httpx.stream("GET", own_port + page) as served,
            ):
                assert forwarded.status_code == served.status_code == 200
                assert forwarded.headers["content-encoding"] == encoding
                assert served.headers["content-encoding"] == encoding
                assert served.headers["connection"] == "close"
                assert "connection" not in forwarded.headers
                assert b"".join(forwarded.iter_raw()) == b"".join(
                    served.iter_raw()
                )

        scalars = f"{url}data/plugin/scalars/scalars_multirun"
        answered = opened_client.post(
            scalars, data={"tag": "loss", "runs": "."}
        )
        assert answered.status_code == 200, answered.text
        assert (
            opened_client.post(scalars, data={"runs": "."}).status_code == 400
        )
    finally:
        stop_tensorboard(str(board_of(run_dir)))


def test_a_forward_whose_connection_fails_closes_its_client(
    opened_client: TestClient,
    tmp_path: Path,
    tb_launches: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forward whose send to the TensorBoard raises still closes the client it opened."""
    import httpx

    url = _served("run-refused", tmp_path, opened_client)
    opened: list[httpx.AsyncClient] = []

    class _RefusedClient(httpx.AsyncClient):
        def __init__(self) -> None:
            super().__init__()
            opened.append(self)

        async def send(
            self, request: httpx.Request, **kwargs: object
        ) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

    monkeypatch.setattr(httpx, "AsyncClient", _RefusedClient)

    with pytest.raises(httpx.ConnectError):
        opened_client.get(url)
    assert [client.is_closed for client in opened] == [True]


def _served(experiment_id: str, tmp_path: Path, client: TestClient) -> str:
    """Open a run with an event file, launch its TensorBoard through the route; return its URL.

    The TensorBoard is :func:`tb_launches`'s listening stand-in, which the caller requests.
    """
    from tcip_mcp.experiments import board_of

    run_dir = _opened(experiment_id, tmp_path)
    board_of(run_dir).mkdir()
    (board_of(run_dir) / "events.out.tfevents.1.host").write_bytes(b"")
    return client.post(
        f"/api/training/runs/{experiment_id}/tensorboard", json={}
    ).json()["url"]


def _upstream(
    monkeypatch: pytest.MonkeyPatch,
    headers: list[tuple[str, str]],
    chunks: Callable[[], AsyncIterator[bytes]],
    close_error: Exception | None = None,
) -> tuple[list[httpx.AsyncClient], list[httpx.AsyncByteStream]]:
    """Make every forward's client answer through httpx's mock transport.

    Each answer is a 200 with ``headers`` and a body of ``chunks()``, whose close raises
    ``close_error`` when one is given. Returns the clients the forwards opened and the bodies
    they were answered with, each recording whether it closed.
    """
    import httpx

    opened: list[httpx.AsyncClient] = []
    bodies: list[httpx.AsyncByteStream] = []

    class _Body(httpx.AsyncByteStream):
        closed = False

        def __aiter__(self) -> AsyncIterator[bytes]:
            return chunks()

        async def aclose(self) -> None:
            self.closed = True
            if close_error is not None:
                raise close_error

    def answer(request: httpx.Request) -> httpx.Response:
        bodies.append(_Body())
        return httpx.Response(200, headers=headers, stream=bodies[-1])

    class _Upstream(httpx.AsyncClient):
        def __init__(self) -> None:
            super().__init__(transport=httpx.MockTransport(answer))
            opened.append(self)

    monkeypatch.setattr(httpx, "AsyncClient", _Upstream)
    return opened, bodies


_IDENTITY = ("content-encoding", "identity")
"""An upstream answer's own encoding, which the app's compression passes on untouched."""


async def _forward(
    url: str,
    spec_version: str,
    receive: Callable[[], Awaitable[Mapping[str, object]]],
    send: Callable[[Mapping[str, object]], Awaitable[None]],
) -> None:
    """Drive the app's GET of ``url`` as a server speaking ASGI ``spec_version`` drives it.

    The request's one body message is answered first, then ``receive`` answers. A forward still
    running after ten seconds is cancelled and fails the caller with an ``AssertionError`` naming
    the deadline, whatever its cleanup raised while unwinding.
    """
    from tcip_web.app import app

    async def arriving() -> Mapping[str, object]:
        nonlocal requested
        if not requested:
            requested = True
            return {"type": "http.request", "body": b"", "more_body": False}
        return await receive()

    requested = False
    deadline = asyncio.timeout(10)
    try:
        async with deadline:
            await app(
                {
                    "type": "http",
                    "asgi": {"version": "3.0", "spec_version": spec_version},
                    "http_version": "1.1",
                    "method": "GET",
                    "scheme": "http",
                    "path": url,
                    "raw_path": url.encode(),
                    "root_path": "",
                    "query_string": b"",
                    "headers": [(b"host", b"127.0.0.1")],
                    "server": ("127.0.0.1", 80),
                    "client": ("127.0.0.1", 50000),
                },
                arriving,
                send,
            )
    finally:
        assert not deadline.expired(), (
            "the forward was still running at its ten-second deadline"
        )


async def _never() -> Mapping[str, object]:
    """Wait for an arrival that never comes, until cancelled."""
    await asyncio.Event().wait()
    raise AssertionError("an event no one sets was set")


@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
def test_a_forwarded_answer_sends_each_chunk_on_as_the_upstream_sends_it(
    opened_client: TestClient,
    tmp_path: Path,
    tb_launches: list[Path],
    monkeypatch: pytest.MonkeyPatch,
    spec_version: str,
) -> None:
    """A forwarded answer sends each chunk on as it arrives, every repeated field kept.

    The upstream withholds its second chunk until the first has been sent on, and the first is
    seen sent while the second is still withheld; two ``Set-Cookie`` fields arrive as two; once
    the answer ends, its upstream body and client are closed. ``spec_version`` is the ASGI
    version the server speaks, which picks the branch Starlette streams under.
    """
    url = _served(
        f"run-streamed-{spec_version.replace('.', '')}",
        tmp_path,
        opened_client,
    )
    first_sent = asyncio.Event()
    produced: list[bytes] = []
    sent: list[bytes] = []
    fields: list[tuple[bytes, bytes]] = []
    withheld_when_first_sent: list[bool] = []

    async def chunks() -> AsyncIterator[bytes]:
        produced.append(b"one")
        yield b"one"
        await first_sent.wait()
        produced.append(b"two")
        yield b"two"

    async def send(message: Mapping[str, object]) -> None:
        if message["type"] == "http.response.start":
            fields.extend(
                cast("list[tuple[bytes, bytes]]", message["headers"])
            )
        elif message["type"] == "http.response.body":
            sent.append(cast(bytes, message["body"]))
            if message["body"] == b"one":
                withheld_when_first_sent.append(b"two" not in produced)
                first_sent.set()

    opened, bodies = _upstream(
        monkeypatch,
        [_IDENTITY, ("set-cookie", "a=1"), ("set-cookie", "b=2")],
        chunks,
    )

    asyncio.run(_forward(url, spec_version, _never, send))

    assert sent == [b"one", b"two", b""]
    assert withheld_when_first_sent == [True]
    assert [value for name, value in fields if name == b"set-cookie"] == [
        b"a=1",
        b"b=2",
    ]
    assert [client.is_closed for client in opened] == [True]
    assert [getattr(body, "closed") for body in bodies] == [True]


@pytest.mark.parametrize(
    "close_error",
    [None, RuntimeError("the upstream close failed")],
    ids=["upstream-closes", "upstream-close-raises"],
)
def test_a_forward_closes_its_upstream_once_it_hears_the_browser_leave(
    opened_client: TestClient,
    tmp_path: Path,
    tb_launches: list[Path],
    monkeypatch: pytest.MonkeyPatch,
    close_error: Exception | None,
) -> None:
    """Under ASGI 2.3, a disconnect heard after the first chunk was sent closes the upstream.

    The upstream sends one chunk and then withholds the next; the server reports the browser
    gone only once that chunk has been sent on, and that disconnect is consumed. Starlette's own
    disconnect listener is seen returning normally, and the upstream is seen still awaiting its
    withheld chunk when that await is cancelled, so the stream was still running when the heard
    disconnect ended it. The upstream body and the client are then closed, the client also when
    the body's close raises ``close_error``, which the forward then raises.
    """
    from tcip_web.routes.training import _Forwarded

    url = _served("run-heard-leaving", tmp_path, opened_client)
    first_sent = asyncio.Event()
    sent: list[bytes] = []
    disconnects: list[Mapping[str, object]] = []
    withholding: list[str] = []
    listened: list[str] = []
    listen = _Forwarded.listen_for_disconnect

    async def listening(answer: _Forwarded, receive: Receive) -> None:
        await listen(answer, receive)
        listened.append("returned")

    monkeypatch.setattr(_Forwarded, "listen_for_disconnect", listening)

    async def chunks() -> AsyncIterator[bytes]:
        yield b"one"
        withholding.append("awaiting")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            withholding.append("cancelled")
            raise

    async def receive() -> Mapping[str, object]:
        await first_sent.wait()
        disconnects.append({"type": "http.disconnect"})
        return disconnects[-1]

    async def send(message: Mapping[str, object]) -> None:
        if message["type"] == "http.response.body":
            sent.append(cast(bytes, message["body"]))
            first_sent.set()

    opened, bodies = _upstream(
        monkeypatch, [_IDENTITY], chunks, close_error=close_error
    )

    if close_error is None:
        asyncio.run(_forward(url, "2.3", receive, send))
    else:
        with pytest.raises(type(close_error), match=str(close_error)):
            asyncio.run(_forward(url, "2.3", receive, send))

    assert sent == [b"one"]
    assert disconnects == [{"type": "http.disconnect"}]
    assert listened == ["returned"]
    assert withholding == ["awaiting", "cancelled"]
    assert [getattr(body, "closed") for body in bodies] == [True]
    assert [client.is_closed for client in opened] == [True]


def test_a_forward_closes_its_upstream_once_a_send_to_the_browser_fails(
    opened_client: TestClient,
    tmp_path: Path,
    tb_launches: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under ASGI 2.4, a send failing after the first chunk was sent closes the upstream.

    The first chunk is sent on; the send of the second raises as a closed connection's does,
    which Starlette raises as ``ClientDisconnect``, and the upstream body and client are then
    closed.
    """
    from starlette.requests import ClientDisconnect

    url = _served("run-send-fails", tmp_path, opened_client)
    first_sent = asyncio.Event()
    sent: list[bytes] = []

    async def chunks() -> AsyncIterator[bytes]:
        yield b"one"
        await first_sent.wait()
        yield b"two"
        await asyncio.Event().wait()

    async def send(message: Mapping[str, object]) -> None:
        if message["type"] == "http.response.body":
            if sent:
                raise OSError("the browser's connection is closed")
            sent.append(cast(bytes, message["body"]))
            first_sent.set()

    opened, bodies = _upstream(monkeypatch, [_IDENTITY], chunks)

    with pytest.raises(ClientDisconnect):
        asyncio.run(_forward(url, "2.4", _never, send))

    assert sent == [b"one"]
    assert [client.is_closed for client in opened] == [True]
    assert [getattr(body, "closed") for body in bodies] == [True]


def test_tensorboard_route_404s_with_no_logs_for_a_run_with_no_event_file(
    opened_client: TestClient,
    tmp_path: Path,
) -> None:
    """The TensorBoard route answers 404 with ``no_logs`` for a run with no event file."""
    _opened("run-nologs", tmp_path)

    resp = opened_client.post(
        "/api/training/runs/run-nologs/tensorboard", json={}
    )
    assert resp.status_code == 404
    assert resp.json()["detail"]["no_logs"] is True


def test_tensorboard_route_404s_with_no_logs_carrying_the_recorded_error(
    tmp_path: Path,
    opened_client: TestClient,
) -> None:
    """The TensorBoard route's ``no_logs`` refusal carries the run's recorded error.

    A run whose status the platform itself recorded carries a real error (a crash during
    data load, say) and whose output directory holds no event file: the refusal says both that
    it produced no logs and what the recorded error was.
    """
    from tests._verified_checkpoint_fixtures import (
        detector_declaring,
        finished_run,
    )

    run_id = "exp-fails-at-data-load"
    finished_run(
        tmp_path,
        experiment_id=run_id,
        model_source=detector_declaring(__file__),
        training_source=f"{Path(__file__).stem}:_fails_at_data_load",
    )

    resp = opened_client.post(
        f"/api/training/runs/{run_id}/tensorboard", json={}
    )
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert detail["no_logs"] is True
    assert detail["message"] == "could not open the dataset's images_dir"


def _fails_at_data_load(ctx: TrainContext) -> None:
    """Raise a RuntimeError naming the dataset's unopenable images_dir."""
    raise RuntimeError("could not open the dataset's images_dir")


def test_list_runs_reads_every_training_run_directory(
    opened_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run listing's rows come from the project's run directories.

    A run whose process stopped touching its heartbeat reads interrupted.
    """
    from tcip_mcp import experiments
    from tcip_web.routes import training

    _opened("run_1", opened_project)
    monkeypatch.setattr(experiments, "HEARTBEAT_STALE_SECONDS", -1.0)

    by_id = {r.experiment_id: r for r in training.list_runs_route().runs}
    assert by_id["run_1"].state == "interrupted"


def test_list_runs_route_is_a_pure_pass_through_to_the_tool(
    opened_project: Path,
) -> None:
    """The run listing route's answer equals the agent's tool's, exactly."""
    from tcip_mcp.tools.experiment_tools import list_experiments
    from tcip_web.routes.training import list_runs_route

    _opened("exp-route-parity", opened_project)

    assert list_runs_route().model_dump() == list_experiments(opened_project)


def test_a_route_launch_shows_its_declaration_from_its_launch_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    opened_client: TestClient,
) -> None:
    """A run launched through the browser-facing door shows its launch event's declaration.

    The declaration is none, since no agent declared itself to the serving process.
    """
    from tests._producer_fixtures import fake_popen

    monkeypatch.chdir(tmp_path)
    fake_popen(monkeypatch, [])

    from tcip_mcp.experiments import experiment_dir, read_run
    from tests._verified_checkpoint_fixtures import (
        detection_config,
        opened_run,
    )

    opened_run(
        tmp_path,
        detection_config(
            tmp_path / "gui-data", project=tmp_path, batch_size=1
        ),
        experiment_id="exp-gui-relaunch",
    )

    from tcip_mcp.audit import audit_log_key
    from tcip_store import read_log

    before = len(read_log(audit_log_key(tmp_path)).records)
    resp = opened_client.post(
        "/api/training/runs",
        json={"relaunched_from": "exp-gui-relaunch", "user": "tester"},
    )
    assert resp.status_code == 200, resp.json()

    minted = resp.json()["experiment_id"]
    (line,) = read_log(audit_log_key(tmp_path)).records[before:]
    assert (line["tool"], line["actor"]) == ("launch_training", "user:tester")
    relaunched = read_run(experiment_dir(minted, project=tmp_path))
    assert "launched_by" not in relaunched.model_dump()
    assert relaunched.relaunched_from == "exp-gui-relaunch"
    row = next(
        r
        for r in opened_client.get("/api/training/runs").json()["runs"]
        if r["experiment_id"] == minted
    )
    assert row["launch"] == {}


def _regression_config(tmp_path: Path) -> dict:
    from tests._training_values import drawn_split
    from tests._verified_checkpoint_fixtures import table_images
    from tests.tiny_trainer_fixtures import regressor_config

    return regressor_config(
        1,
        data={
            **table_images(tmp_path),
            "split": drawn_split(val_ratio=0.15),
        },
        project=tmp_path,
    )


def test_relaunch_route_forks_a_run_s_config_and_names_the_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    opened_client: TestClient,
) -> None:
    """The relaunch route forks a run's config and names the parent.

    Relaunching starts a new run directory through the real launcher and a real child, its
    launch record replaying the picked run's config and naming it as parent.
    """
    from tcip_mcp.tools.training_tools import launch_training
    from tests._producer_fixtures import no_tensorboard
    from tests._verified_checkpoint_fixtures import run_to_end

    no_tensorboard(monkeypatch)

    first = launch_training(tmp_path, _regression_config(tmp_path), actor=None)
    assert "error" not in first, first
    parent_id = first["experiment_id"]
    assert run_to_end(tmp_path, parent_id)["state"] == "completed"

    resp = opened_client.post(
        "/api/training/runs",
        json={"relaunched_from": parent_id, "user": "tester"},
    )
    assert resp.status_code == 200, resp.json()
    forked_id = resp.json()["experiment_id"]
    assert forked_id != parent_id
    assert run_to_end(tmp_path, forked_id)["state"] == "completed"

    from tcip_mcp.experiments import experiment_dir, read_run

    parent = read_run(experiment_dir(parent_id, project=tmp_path))
    forked = read_run(experiment_dir(forked_id, project=tmp_path))
    assert forked.relaunched_from == parent_id
    assert forked.config == parent.config


def test_list_runs_route_names_the_run_s_selection_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    opened_client: TestClient,
) -> None:
    """The run listing names a launched run's selection metric and its best value.

    A launched (subprocess-delegated) run's row carries the metric the trainer itself
    stamped on its metrics-log rows, and a finite best value read back beside it.
    """
    import math

    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import launch_training
    from tests._producer_fixtures import no_tensorboard
    from tests._verified_checkpoint_fixtures import run_to_end

    no_tensorboard(monkeypatch)

    result = launch_training(
        tmp_path, _regression_config(tmp_path), actor=None
    )
    assert "error" not in result, result
    run_to_end(tmp_path, result["experiment_id"])

    from tcip_web.routes.training import list_runs_route

    by_id = {
        r["experiment_id"]: r for r in list_runs_route().model_dump()["runs"]
    }
    row = by_id[result["experiment_id"]]
    # Regression selects on its loss by default; there is no evaluation.selection_metric override
    # in this config.
    assert row["best_metric_name"] == "loss"
    assert isinstance(row["best_metric"], (int, float)) and math.isfinite(
        row["best_metric"]
    )


def test_a_sweep_trial_is_listed_read_canceled_and_streamed_by_its_id(
    opened_client: TestClient,
    opened_project: Path,
    real_hpo_base_config: dict,
) -> None:
    """A sweep trial is listed, read, canceled and streamed by its own id.

    A trial is a run directory under its sweep: the run routes reach it by its own id, and the
    listing groups it under its sweep.
    """
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
    assert [
        (t["experiment_id"], t["sweep"], t["trial_params"])
        for t in sweep["trials"]
    ] == [(trial.name, sweep_id, {"optimizer.head_lr": 0.001})]

    read = opened_client.get(f"/api/training/runs/{trial.name}")
    assert (read.status_code, read.json()["run"]["sweep"]) == (200, sweep_id)

    canceled = opened_client.post(
        f"/api/training/runs/{trial.name}/cancel", json={"user": "tester"}
    )
    assert canceled.status_code == 200, canceled.text
    assert canceled.json()["cancel_requested"] is True

    from tcip_mcp.experiments import write_final_status

    write_final_status(
        trial, "canceled", "canceled by request", checkpoint=None
    )
    with opened_client.websocket_connect(
        f"ws://127.0.0.1/api/training/runs/{trial.name}/stream"
    ) as ws:
        frame = ws.receive_json()
    assert frame["type"] == "status"
    assert (frame["status"]["state"], frame["status"]["sweep"]) == (
        "canceled",
        sweep_id,
    )


def test_a_sweep_and_its_trial_each_have_one_board_whichever_door_launches_it(
    opened_client: TestClient,
    opened_project: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
    tb_launches: list[Path],
) -> None:
    """TensorBoard launches for a sweep and its trial name the sweep's and the trial's directories.

    The launches name the sweep's own directory twice, from its body and from the run route,
    and the trial's own directory once; the sweep's board is its own directory.
    """
    import tcip_mcp.tools.training_tools as tt
    from tcip_mcp.experiments import board_of, find_sweep
    from tcip_store import canonical_path

    trials: list[Path] = []

    def one_trial_search(
        **kw: Callable[[dict, Callable[[float], None]], object]
        | dict
        | str
        | int
        | Path
        | None,
    ) -> None:
        trial = tt.open_trial(
            kw["sweep_dir"], "t0", {"optimizer.head_lr": 0.1}
        )
        board_of(trial).mkdir()
        (board_of(trial) / "events.out.tfevents.1.host").write_bytes(b"")
        trials.append(trial)

    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search", one_trial_search
    )
    sweep_id = tt.run_hyperparameter_search(
        opened_project,
        base_config=real_hpo_base_config,
        param_space=sweep_space(),
        n_trials=1,
        **fifo_search(),
        search_seed=0,
    )["sweep"]["sweep_id"]
    for name in (sweep_id, trials[0].name):
        resp = opened_client.post(
            f"/api/training/runs/{name}/tensorboard", json={}
        )
        assert resp.status_code == 200, resp.text

    sweep_dir = find_sweep(sweep_id, project=opened_project)
    assert sweep_dir is not None and board_of(sweep_dir) == sweep_dir
    assert [canonical_path(directory) for directory in tb_launches] == [
        canonical_path(sweep_dir),
        canonical_path(sweep_dir),
        canonical_path(trials[0]),
    ]


def test_a_recorded_sweep_relaunches_through_the_run_launch_door(
    opened_client: TestClient,
    opened_project: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recorded sweep relaunches through the run launch door.

    A sweep relaunches from its own recorded input through the one launch door: a new sweep
    is opened naming its source, its audit line names the person, and the one captured spawn
    request ends in ``--run-dir`` and the new sweep's directory. A relaunch also naming a
    selection directory is refused 422.
    """
    import tcip_store as ts
    from tcip_mcp.audit import audit_log_key
    from tcip_mcp.experiments import find_sweep, observe_sweep
    from tests._producer_fixtures import fake_popen
    from tests._verified_checkpoint_fixtures import opened_sweep

    started: list[list[str]] = []
    fake_popen(monkeypatch, started)
    source = opened_sweep(opened_project, real_hpo_base_config).name
    before = len(ts.read_log(audit_log_key(opened_project)).records)

    resp = opened_client.post(
        "/api/training/runs",
        json={"relaunched_from": source, "user": "tester"},
    )

    assert resp.status_code == 200, resp.text
    minted = find_sweep(resp.json()["sweep_id"], project=opened_project)
    assert minted is not None
    assert observe_sweep(minted).record.input.relaunched_from == source
    (line,) = ts.read_log(audit_log_key(opened_project)).records[before:]
    assert (line["tool"], line["actor"]) == ("open_sweep", "user:tester")
    assert [argv[-2:] for argv in started] == [["--run-dir", str(minted)]]

    refused = opened_client.post(
        "/api/training/runs",
        json={
            "relaunched_from": source,
            "selection_dir": str(opened_project),
            "user": "tester",
        },
    )
    assert refused.status_code == 422
