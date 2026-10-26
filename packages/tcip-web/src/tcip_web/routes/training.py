"""Training routes: launch or relaunch a run or sweep, list runs and sweeps, live metrics stream."""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Literal

import httpx
from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.types import Receive, Scope, Send

from tcip_mcp.experiments import (
    RunRow,
    TrainingDetail,
    TrainingListing,
    training_listing,
)
from tcip_mcp.identity import actor
from tcip_mcp.pipelines.training.tensorboard_manager import (
    BOARD_PATH,
    TensorboardLaunch,
)
from tcip_mcp.tools.training_tools import Launched
from tcip_store import FiniteJsonObject
from tcip_store.errors import BadKeyError
from tcip_web.routes._body_common import EmptyBodyPayload, PersonPayload
from tcip_web.state import NoProjectOpenError, store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/training", tags=["training"])


def _refused(result: dict) -> HTTPException:
    """Return a tool's error dict ``result`` as a 422 whose detail carries its fields.

    The ``error`` reason is named ``message``.
    """
    return HTTPException(
        422, {("message" if k == "error" else k): v for k, v in result.items()}
    )


@router.get("/runs/{experiment_id}/splits")
def list_split_choices_route(experiment_id: str) -> dict:
    """List every data choice a relaunch of the run ``experiment_id`` names can take.

    The choices are the data section its launch stated, and every selection directory this
    project's own bound runs or the dataset's own splits directory hold, each
    compatibility-checked.
    """
    from tcip_mcp.tools.training_tools import list_split_choices

    result = list_split_choices(store.held().root, experiment_id)
    if result.get("error"):
        raise HTTPException(404, result["error"])
    return result


class RelaunchPayload(BaseModel):
    """A relaunch: the run or sweep it starts from, an optional selection and the person."""

    relaunched_from: str
    selection_dir: str | None = None
    user: str


@router.post("/runs")
def relaunch_route(payload: RelaunchPayload) -> Launched:
    """Start a new run or sweep from what the run or sweep ``relaunched_from`` names recorded.

    The source is one of this project's, and the launch is by the person ``user`` names. A run
    relaunches through ``launch_training`` with its launch config; a sweep through
    ``training_tools.launch_sweep``, off the request thread in its own worker process, the way a
    run's body runs.

    An optional ``selection_dir`` names a partition to use instead of a run's own "As recorded"
    data section: the launch config then carries ``data.split`` replaced wholesale by
    ``{"selection_dir": chosen}``. Answers the launch's ``training_tools.RunLaunched`` or
    ``SweepLaunched``. A source naming nothing answers 404 (:func:`_named`), a sweep source with
    a ``selection_dir`` 422, a refusal 422, and a launch whose audit line could not be written
    409.
    """
    from tcip_mcp.audit import AuditEntryNotWrittenError
    from tcip_mcp.experiments import SWEEP_FILE, observe_run
    from tcip_mcp.tools.training_tools import (
        data_with_selection,
        launch_sweep,
        launch_training,
    )
    from tcip_web.routes.audit_gap import audit_gap_409

    person = actor(payload.user)
    project = store.held().root
    source = _named(payload.relaunched_from)
    result: Launched | dict
    try:
        if (source / SWEEP_FILE).is_file():
            if payload.selection_dir:
                raise HTTPException(
                    422, "a sweep relaunches from its own recorded input"
                )
            result = launch_sweep(project, source, actor=person)
        else:
            spec = observe_run(source).record.spec
            if payload.selection_dir:
                data = data_with_selection(
                    spec.data, payload.selection_dir, project
                )
                spec = spec.model_copy(update={"data": data})
            result = launch_training(
                project,
                spec.record(),
                relaunched_from=source.name,
                actor=person,
            )
    except AuditEntryNotWrittenError as exc:
        raise audit_gap_409(exc, exc.arguments) from exc
    if isinstance(result, dict):
        raise _refused(result)
    return result


@router.get("/runs")
def list_runs_route() -> TrainingListing:
    """List every run and sweep of the open project (``experiments.training_listing``)."""
    return training_listing(store.held().root)


@router.get("/runs/{experiment_id}")
def get_run(experiment_id: str) -> TrainingDetail:
    """Return one run, trial or sweep of the open project (``training_tools.training_detail``).

    404 for an id naming none (:func:`_named`).
    """
    from tcip_mcp.tools.training_tools import training_detail

    return training_detail(store.held().root, _named(experiment_id))


def _named(experiment_id: str) -> Path:
    """Return the open project's run, trial or sweep directory ``experiment_id`` names.

    404 when it names none (``experiments.named_directory``).
    """
    from tcip_mcp.experiments import named_directory

    directory = named_directory(experiment_id, project=store.held().root)
    if directory is None:
        raise HTTPException(404, f"Run not found: {experiment_id}")
    return directory


_BOARD_ROUTE = BOARD_PATH.removeprefix(router.prefix)


@router.post(_BOARD_ROUTE)
def launch_run_tensorboard(
    experiment_id: str, payload: EmptyBodyPayload
) -> TensorboardLaunch:
    """Start (or reuse) a TensorBoard over the board of the run, trial or sweep ``experiment_id``.

    The launch is ``tensorboard_manager.launch_tensorboard``'s over the directory
    ``experiment_id`` names, its board ``experiments.board_of``'s.

    A board holding no event file refuses with ``no_logs: True``, carrying the final status's
    error when one is written; a board with events whose run or sweep ended in error refuses
    with that error alone.
    """
    from tcip_mcp.experiments import board_of, observe
    from tcip_mcp.pipelines.training.tensorboard_manager import (
        launch_tensorboard,
    )

    directory = _named(experiment_id)
    error = observe(directory).status_error
    if not any(board_of(directory).rglob("events.out.tfevents*")):
        raise HTTPException(
            404,
            {
                "message": error or f"run produced no logs: {experiment_id}",
                "no_logs": True,
            },
        )
    if error:
        raise HTTPException(404, error)
    return launch_tensorboard(directory)


@router.api_route(_BOARD_ROUTE + "/{path:path}", methods=["GET", "POST"])
async def forward_to_board(
    experiment_id: str, path: str, request: Request
) -> StreamingResponse:
    """Forward a request under the board path of the run, trial or sweep ``experiment_id``.

    It goes to the TensorBoard this process runs over that directory's board
    (``tensorboard_manager.running``), with its method, its whole path, its query, its
    end-to-end headers and its body as received; the answer streams back with its status,
    end-to-end header fields and bytes as received (:class:`_Forwarded`). A send that raises
    closes its client. An id naming nothing in the open project, or a board no TensorBoard of
    this process serves, answers 404.
    """
    from tcip_mcp.pipelines.training.tensorboard_manager import running
    from tcip_mcp.web_client import LOOPBACK_HOST

    serving = running(await asyncio.to_thread(_named, experiment_id))
    if serving is None:
        raise HTTPException(404, f"no TensorBoard serves {experiment_id}")
    client = httpx.AsyncClient()
    try:
        upstream = await client.send(
            client.build_request(
                request.method,
                str(
                    request.url.replace(
                        hostname=LOOPBACK_HOST, port=serving.port
                    )
                ),
                headers=_end_to_end(request.headers.raw),
                content=await request.body(),
            ),
            stream=True,
        )
    except BaseException:
        await client.aclose()
        raise
    return _Forwarded(upstream, client)


class _Forwarded(StreamingResponse):
    """An upstream answer streamed on raw, with its end-to-end header fields as received.

    The upstream answer and its client close once the answer is sent, abandoned or failed.
    """

    def __init__(
        self, upstream: httpx.Response, client: httpx.AsyncClient
    ) -> None:
        """Stream ``upstream``'s raw bytes; ``client`` is the one that sent its request."""
        super().__init__(
            upstream.aiter_raw(), status_code=upstream.status_code
        )
        self.raw_headers = _end_to_end(upstream.headers.raw)
        self.upstream = upstream
        self.client = client

    async def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        """Send the answer, then close the upstream answer and its client on every path.

        Each close runs whether or not the other raised.
        """
        async with AsyncExitStack() as closing:
            closing.push_async_callback(self.client.aclose)
            closing.push_async_callback(self.upstream.aclose)
            await super().__call__(scope, receive, send)


def _end_to_end(
    fields: list[tuple[bytes, bytes]],
) -> list[tuple[bytes, bytes]]:
    """Return the header ``fields`` a forwarding passes on, named in lower case as ASGI names them.

    The hop-by-hop ones are left out: werkzeug's hop-by-hop set, ``Proxy-Connection`` and every
    field ``Connection`` names (RFC 9110 section 7.6.1).
    """
    from werkzeug.http import is_hop_by_hop_header, parse_list_header

    named = {"proxy-connection"} | {
        token.lower()
        for name, value in fields
        if name.lower() == b"connection"
        for token in parse_list_header(value.decode("latin-1"))
    }
    return [
        (name.lower(), value)
        for name, value in fields
        if not is_hop_by_hop_header(name.decode("latin-1"))
        and name.decode("latin-1").lower() not in named
    ]


@router.post("/runs/{experiment_id}/cancel")
def cancel_run_route(experiment_id: str, payload: PersonPayload) -> dict:
    """Request graceful cancellation of the run, trial or sweep ``experiment_id`` names.

    The request goes through ``cancel_training``, by the person ``user`` names; its refusal
    answers 404.
    """
    from tcip_mcp.tools.training_tools import cancel_training

    result = cancel_training(
        store.held().root, experiment_id, actor=actor(payload.user)
    )
    if result.get("error"):
        raise HTTPException(404, result["error"])
    return result


class ExperimentComparePayload(BaseModel):
    """The ids of the runs to compare."""

    experiment_ids: list[str]


@router.post("/compare")
def compare_runs_route(payload: ExperimentComparePayload) -> dict:
    """Compare the open project's runs ``experiment_ids`` names (``compare_experiments``)."""
    from tcip_mcp.experiments import compare_experiments

    return compare_experiments(
        payload.experiment_ids, project=store.held().root
    )


class CompareBestPayload(BaseModel):
    """A ranking request: the runs compared, the metric, its direction and unverified inclusion."""

    experiment_ids: list[str]
    metric: str
    higher_is_better: bool | None = None
    include_unverified: bool = False


@router.post("/compare/best")
def compare_best_route(payload: CompareBestPayload) -> dict:
    """Rank the marked comparison's own registered checkpoints by one metric.

    The ranking is :func:`~tcip_mcp.tools.model_tools.ranked_registered_model`'s.

    A registry index that will not decode answers 409 naming why; the ranking's own error dicts
    map to 422 (``_refused``). The answer is projected to name, experiment id,
    stamped metrics, source, the metric ranked by, the direction used and its source, and the
    exclusions.
    """
    from tcip_mcp.tools.model_tools import ranked_registered_model
    from tcip_store import DecodeError

    try:
        result = ranked_registered_model(
            store.held().root,
            payload.metric,
            higher_is_better=payload.higher_is_better,
            include_unverified=payload.include_unverified,
            experiment_ids=payload.experiment_ids,
            tag=None,
        )
    except DecodeError as exc:
        raise HTTPException(409, f"registry unreadable: {exc}") from exc
    if "error" in result:
        raise _refused(result)

    return {
        "name": result["name"],
        "experiment_id": result["experiment_id"],
        "metrics": result["metrics"],
        "metrics_source": result["metrics_source"],
        "ranking_basis": result["ranking_basis"],
        "higher_is_better": result["higher_is_better"],
        "direction_source": result["direction_source"],
        "excluded_unverified": result["excluded_unverified"],
    }


@router.get("/metric-directions")
def metric_directions_route() -> dict:
    """Return every metric's declared ranking direction, as ``schemas`` declares it."""
    from tcip_mcp.pipelines.schemas import HIGHER_IS_BETTER_BY_METRIC

    return {"higher_is_better": dict(HIGHER_IS_BETTER_BY_METRIC)}


# ── WebSocket live metrics ──────────────────────────────────────────────


class TrainingRowFrame(BaseModel):
    """One metrics row, by its kind (``experiments.partition_rows``).

    ``metric`` is an epoch's whole row (``experiments.epoch_rows``), pushed each time a row of
    that epoch is appended; ``batch`` is the latest per-batch row a read of the run's log found.
    """

    type: Literal["metric", "batch"]
    experiment_id: str
    row: FiniteJsonObject


class TrainingStatusFrame(BaseModel):
    """The terminal frame, carrying the run's row or an error.

    ``status`` carries the run's row for a run this process can still identify, ``error`` is set
    instead when it cannot.
    """

    type: Literal["status"]
    experiment_id: str
    status: RunRow | None
    error: str | None


async def _stream_metrics(
    ws: WebSocket, project: Path, experiment_id: str, poll_seconds: float = 1.0
) -> None:
    """Push a run's metrics, a sweep's trial included, over ``ws`` as they are appended.

    Each epoch a new epoch row touches is sent again whole (``experiments.epoch_rows`` over
    every row read), then the latest per-batch row the read found, if any
    (``experiments.partition_rows``).

    Each tick observes the run once (``experiments.observe_run``) and reads its log from a
    byte-offset cursor (``experiments.read_rows``), so it reads only what was appended since the
    last tick and a row still being written is replayed once it is complete. Rows are read after the
    observation, so a run the observation found in a terminal state has every row sent before its
    one status frame (``experiments.run_summary`` over the rows sent), which ends the stream.
    An id naming no run directory under ``project`` ends the stream with one status frame
    naming it, and one that is not a single directory name raises ``BadKeyError``. Every read
    runs off the event loop.
    """
    from tcip_mcp.experiments import (
        EPOCH_KEY,
        TERMINAL_STATES,
        epoch_rows,
        find_observation,
        launch_declarations,
        observe_run,
        partition_rows,
        read_rows,
        run_name,
        run_summary,
    )

    observation = await asyncio.to_thread(
        find_observation, run_name(experiment_id), project=project
    )
    if observation is None:
        await ws.send_json(
            TrainingStatusFrame(
                type="status",
                experiment_id=experiment_id,
                status=None,
                error=f"Run not found: {experiment_id}",
            ).model_dump()
        )
        return
    cursor = 0
    sent: list[dict] = []

    while True:
        rows, cursor = await asyncio.to_thread(
            read_rows, observation.metrics_log, after=cursor
        )
        epochs, batches = partition_rows(rows)
        sent += epochs
        touched = [row.get(EPOCH_KEY) for row in epochs]
        for row in epoch_rows(sent):
            if row.get(EPOCH_KEY) in touched:
                frame = TrainingRowFrame(
                    type="metric", experiment_id=experiment_id, row=row
                )
                await ws.send_json(frame.model_dump())
        if batches:
            await ws.send_json(
                TrainingRowFrame(
                    type="batch", experiment_id=experiment_id, row=batches[-1]
                ).model_dump()
            )
        if observation.state in TERMINAL_STATES:
            launches = await asyncio.to_thread(launch_declarations, project)
            status = await asyncio.to_thread(
                run_summary, observation, sent, launches.get(experiment_id)
            )
            await ws.send_json(
                TrainingStatusFrame(
                    type="status",
                    experiment_id=experiment_id,
                    status=status,
                    error=None,
                ).model_dump()
            )
            break
        await asyncio.sleep(poll_seconds)
        observation = await asyncio.to_thread(
            observe_run, observation.directory
        )


@router.websocket("/runs/{experiment_id}/stream")
async def training_stream_ws(websocket: WebSocket, experiment_id: str) -> None:
    """Tail the metrics log of the open project's run ``experiment_id`` and push new rows.

    The rows go over the socket; closes with 1008 while no project is open.
    """
    try:
        project = store.held().root
    except NoProjectOpenError as exc:
        await websocket.close(code=1008, reason=str(exc))
        return
    await websocket.accept()
    try:
        await _stream_metrics(websocket, project, experiment_id)
    except BadKeyError as exc:
        await websocket.close(code=1008, reason=str(exc))
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("training stream failed")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
