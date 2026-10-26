"""FastAPI application: the GUI state, the panel-event hub, the built frontend and health probe.

The GUI state is served as a snapshot and over its WebSocket; every domain route is mounted from
``tcip_web.routes``.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.websockets import WebSocketState
from pydantic import BaseModel, JsonValue

from tcip_mcp.audit import now_iso
from tcip_mcp.buckets import NotABucketError
from tcip_mcp.identity import NoActorError
from tcip_mcp.web_client import PANEL_EVENT_ANNOTATE_FOCUS, Panel
from tcip_store import bind
from tcip_web.routes import register_all as _register_routes
from tcip_web.state import (
    GuiMutationInvalidError,
    NoProjectOpenError,
    PanelEnvelope,
    ProjectNotOpenError,
    SnapshotProject,
    StateNotification,
    StateSnapshotMessage,
    store as _gui_store,
)
from tcip_web.trust_boundary import TrustBoundaryMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Size GDAL's block cache and warm the agent terminal before the app serves.

    The backend starts with no project open.
    """
    # GDAL's block cache is sized once per process, at its entry point.
    from tcip_mcp.pipelines.raster_source import configure_gdal_cache

    configure_gdal_cache()
    try:
        from tcip_web import terminal

        terminal.prewarm()
    except Exception:  # pragma: no cover - prewarm is best-effort
        logger.exception("agent terminal prewarm failed to start")
    yield
    # Off the loop: terminate can block for seconds, breaking in-flight WebSocket close handshakes.
    try:
        from tcip_web.routes import terminal as terminal_routes

        await asyncio.to_thread(terminal_routes.shutdown_all)
    except Exception:  # pragma: no cover - shutdown cleanup is best-effort
        logger.exception("agent terminal shutdown failed")


# Bound at import, so a route exercised without the app's lifespan running still has a store.
bind()


app = FastAPI(title="TCIP Pipeline", version="0.1.0", lifespan=_lifespan)

app.add_middleware(TrustBoundaryMiddleware)

# A label document's JSON grows with its polygon count, to megabytes on a dense image.
app.add_middleware(GZipMiddleware, minimum_size=1000)

_register_routes(app)

_state_watchers: set[WebSocket] = set()


async def _bad_request_handler(
    _request: Request, exc: Exception
) -> JSONResponse:
    """Answer 400 with the reason.

    For an invalid GUI mutation, a directory read as a bucket it is not, or an act naming no
    person.
    """
    return JSONResponse(status_code=400, content={"detail": str(exc)})


app.add_exception_handler(GuiMutationInvalidError, _bad_request_handler)
app.add_exception_handler(NotABucketError, _bad_request_handler)
app.add_exception_handler(NoActorError, _bad_request_handler)


@app.exception_handler(NoProjectOpenError)
async def _no_project_open_handler(
    _request: Request, exc: NoProjectOpenError
) -> JSONResponse:
    """Answer 409 with the no-project exception's detail."""
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.exception_handler(ProjectNotOpenError)
async def _project_not_open_handler(
    _request: Request, exc: ProjectNotOpenError
) -> JSONResponse:
    """Answer 409 naming the open project to a request built for another one."""
    return JSONResponse(
        status_code=409,
        content={
            "detail": {
                "message": str(exc),
                "open_project_id": exc.open_project_id,
            }
        },
    )


# Minted per process, so a client's wsVersion guard admits a restarted backend's first snapshot.
SERVER_EPOCH = uuid.uuid4().hex


def state_snapshot_message(
    notification: StateNotification,
) -> StateSnapshotMessage:
    """Return ``notification``'s state-snapshot envelope."""
    opened = _gui_store.opened
    return StateSnapshotMessage(
        type="state_snapshot",
        state=notification.state,
        version=notification.version,
        project=None if opened is None else SnapshotProject.of(opened),
        epoch=SERVER_EPOCH,
    )


async def _send_to_each(message: BaseModel, sockets: set[WebSocket]) -> None:
    """Send ``message`` as JSON to every connected socket in ``sockets``.

    A socket that is disconnected, already or by this send, is removed; any other fault raises.
    """
    text = message.model_dump_json()
    for ws in list(sockets):
        if ws.application_state is WebSocketState.CONNECTED:
            with suppress(WebSocketDisconnect):
                await ws.send_text(text)
        if ws.application_state is WebSocketState.DISCONNECTED:
            sockets.discard(ws)


async def _broadcast_state_snapshot(notification: StateNotification) -> None:
    """Push the new state, version and open project to every state socket."""
    await _send_to_each(state_snapshot_message(notification), _state_watchers)


_gui_store.subscribe(_broadcast_state_snapshot)


@app.get("/api/state")
def get_state() -> dict:
    """Return the GUI state store's current snapshot."""
    return _gui_store.snapshot()


class ActiveTabPayload(BaseModel):
    """The active tab to record."""

    active_tab: str


@app.post("/api/state/tab")
async def set_active_tab(payload: ActiveTabPayload) -> dict:
    """Record ``active_tab`` in the GUI state and answer it."""
    await _gui_store.mutate(
        {"active_tab": payload.active_tab}, project=_gui_store.opened
    )
    return {"status": "ok", "active_tab": payload.active_tab}


@app.websocket("/ws/state")
async def state_ws(websocket: WebSocket) -> None:
    """Push live GuiState snapshots over the socket; replays the current snapshot on connect.

    One-directional: an inbound frame is read and discarded, only to detect disconnect.
    """
    await websocket.accept()
    _state_watchers.add(websocket)
    try:
        await websocket.send_text(
            state_snapshot_message(_gui_store.notification()).model_dump_json()
        )
        async for _ in websocket.iter_text():
            pass
    except WebSocketDisconnect:
        pass
    finally:
        _state_watchers.discard(websocket)


# An event id's timestamp keeps it distinct across a restart, which resets this counter.
_event_counter = itertools.count(1)
_panel_subscribers: dict[Panel, set[WebSocket]] = defaultdict(set)


class PanelEvent(BaseModel):
    """An event pushed to a panel: its type, its data and the project the sender acts on."""

    event_type: str
    data: dict[str, JsonValue] = {}
    project_id: str


STATIC_DIR = Path(__file__).parent / "static"
"""The frontend build's output, inside the package in a checkout, an editable install or a wheel."""

ASSETS_DIR = STATIC_DIR / "assets"
"""The bundle's assets, which its page references absolutely under Vite's default base ``/``."""

if ASSETS_DIR.exists():
    app.mount(f"/{ASSETS_DIR.name}", StaticFiles(directory=str(ASSETS_DIR)))


FRONTEND_SERVING_PATHS = frozenset({"/", "/health"})
"""The paths this app serves for itself, the built frontend's page and the health probe; neither
is an API route."""


@app.get("/health")
def health() -> dict:
    """Answer the health probe with ``{"status": "ok"}``."""
    return {"status": "ok"}


@app.get("/")
def index() -> Response:
    """Serve the built frontend's ``index.html``, or a 503 saying the bundle is missing."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse(
        status_code=503,
        content={
            "error": "GUI not built",
            "detail": (
                "This installation carries no frontend bundle; the repository "
                "README's Setup section states how the bundle is built and installed."
            ),
            "api_docs": "/docs",
        },
    )


@app.post("/api/events/{panel}")
async def post_panel_event(
    panel: Panel, event: PanelEvent, request: Request
) -> dict:
    """Accept a pushed event and broadcast it to subscribers.

    Payload shape: ``{event_type, data, project_id}``. A path naming no panel answers 422. An
    event naming any project but the open one answers 409 (``StateStore.admit``), naming the
    open one's ``open_project_id`` when another is open and the none-open message when none is,
    and is neither retained nor broadcast. The broadcast and replay payload, not this route's
    response, carries every agent identity field the sender declared in its headers
    (``agent_identity.HEADERS``, each ``None`` when not sent). Declared, not verified: any
    sender can set the headers.
    """
    from tcip_mcp import agent_identity

    project = _gui_store.admit(event.project_id)
    envelope = PanelEnvelope(
        panel=panel,
        event_type=event.event_type,
        data=event.data,
        event_id=f"{now_iso()}#{next(_event_counter)}",
        **agent_identity.fields_from_headers(request.headers),
    )
    _gui_store.retain_event(envelope)
    # The browser applies a focus event locally and never syncs it back, so the snapshot moves here.
    if event.event_type == PANEL_EVENT_ANNOTATE_FOCUS:
        mutation: dict[str, Any] = {"active_tab": "annotate"}
        if "mode" in event.data:
            mutation["mode"] = event.data["mode"]
        if "active_subject" in event.data:
            mutation["active_subject"] = event.data["active_subject"]
        await _gui_store.mutate(mutation, project=project)
    await _send_to_each(envelope, _panel_subscribers[panel])
    return {"status": "ok", "panel": panel, "event_type": event.event_type}


@app.websocket("/ws/panel/{panel}")
async def panel_ws(websocket: WebSocket, panel: Panel) -> None:
    """Stream ``panel``'s events over the socket, the open project's retained events first.

    A path naming no panel is closed with 1008 before it is accepted. An inbound frame is read
    and discarded, only to detect disconnect.
    """
    await websocket.accept()
    _panel_subscribers[panel].add(websocket)
    try:
        for event in _gui_store.retained_events(panel):
            await websocket.send_text(event.model_dump_json())
        async for _ in websocket.iter_text():
            pass
    except WebSocketDisconnect:
        pass
    finally:
        _panel_subscribers[panel].discard(websocket)
