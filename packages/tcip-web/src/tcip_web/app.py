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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, JsonValue

from tcip_mcp.audit import now_iso
from tcip_mcp.buckets import NotABucketError
from tcip_mcp.identity import NoActorError
from tcip_mcp.web_client import PANEL_EVENT_ANNOTATE_FOCUS, VALID_PANELS
from tcip_store import bind
from tcip_web.trust_boundary import TrustBoundaryMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Size GDAL's block cache and warm the agent terminal before the app serves.

    The backend starts with no project open.
    """
    # Size GDAL's block cache once per process, at the entry point, never at source construction.
    from tcip_mcp.pipelines.raster_source import configure_gdal_cache

    configure_gdal_cache()
    # Warm the cold first-spawn cost (tcip_mcp import + PowerShell/.NET) off the request path.
    try:
        from tcip_web import terminal

        terminal.prewarm()
    except Exception:  # pragma: no cover - prewarm is best-effort
        logger.exception("agent terminal prewarm failed to start")
    yield
    # Kill live agent terminals off the loop: terminate can block seconds, and stalling the loop
    # would break in-flight WebSocket close handshakes.
    try:
        from tcip_web.routes import terminal as terminal_routes

        await asyncio.to_thread(terminal_routes.shutdown_all)
    except Exception:  # pragma: no cover - shutdown cleanup is best-effort
        logger.exception("agent terminal shutdown failed")


# At import, not in the lifespan: a route may be exercised against this app without one running,
# and a route that reaches a store with no backend bound would refuse rather than write.
bind()


app = FastAPI(title="TCIP Pipeline", version="0.1.0", lifespan=_lifespan)

# CORS is not enabled by default: the browser hits the same origin via the Vite dev proxy.
# Serving the frontend elsewhere would add fastapi.middleware.cors.CORSMiddleware here.

# A connection must arrive through this machine and name a loopback Host; the middleware also
# applies the Origin policy before a route runs (trust_boundary).
app.add_middleware(TrustBoundaryMiddleware)

# Compress JSON/text responses above ~1KB. A label document's payload scales with polygon count
# (dense images ship high-hundreds-of-KB to multi-MB uncompressed JSON).
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ── Tab routes ──
from tcip_web.routes import register_all as _register_routes  # noqa: E402  (needs `app`)

_register_routes(app)

# ── State snapshot + WS ──
from tcip_web.state import (  # noqa: E402  (needs `app`)
    GuiMutationInvalidError,
    NoProjectOpenError,
    PanelEnvelope,
    ProjectNotOpenError,
    StateNotification,
    store as _gui_store,
)

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


# This process's launch identity, minted once at import: rides every snapshot envelope so a
# restarted backend's lower-numbered first snapshot is accepted across a client's wsVersion guard.
SERVER_EPOCH = uuid.uuid4().hex


class SnapshotProject(BaseModel):
    """The open project as a state snapshot names it: its id and its directory."""

    id: str
    path: str


class StateSnapshotMessage(StateNotification):
    """The state-snapshot envelope: a notification with the open project and this epoch.

    ``project`` is ``None`` when no project is open.
    """

    type: Literal["state_snapshot"] = "state_snapshot"
    project: SnapshotProject | None
    epoch: str


def state_snapshot_message(
    notification: StateNotification,
) -> StateSnapshotMessage:
    """Return ``notification``'s state-snapshot envelope."""
    opened = _gui_store.opened
    return StateSnapshotMessage(
        state=notification.state,
        version=notification.version,
        project=None
        if opened is None
        else SnapshotProject(id=opened.id, path=str(opened.root)),
        epoch=SERVER_EPOCH,
    )


async def _broadcast_state_snapshot(notification: StateNotification) -> None:
    """Push the new state, version and open project to every state socket, dropping dead ones."""
    msg = state_snapshot_message(notification).model_dump(mode="json")
    dead: list[WebSocket] = []
    for ws in list(_state_watchers):
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _state_watchers.discard(ws)


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
        await websocket.send_json(
            state_snapshot_message(_gui_store.notification()).model_dump(
                mode="json"
            )
        )
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _state_watchers.discard(websocket)


# ── Panel event hub ──

# Per-event identity; the timestamp keeps ids distinct across a restart, where the counter resets
# while browsers keep their dismissals.
_event_counter = itertools.count(1)
# Open WebSocket subscribers per panel.
_panel_subscribers: dict[str, set[WebSocket]] = defaultdict(set)


class PanelEvent(BaseModel):
    """An event pushed to a panel: its type, its data and the project the sender acts on."""

    panel: str | None = None
    event_type: str
    data: dict[str, JsonValue] = {}
    # The id of the project the sender acts on.
    project_id: str


async def _broadcast_to_panel(event: PanelEnvelope) -> None:
    """Fan out ``event`` to every WebSocket currently subscribed to its panel."""
    message = event.model_dump(mode="json")
    dead: list[WebSocket] = []
    for ws in list(_panel_subscribers.get(event.panel, ())):
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _panel_subscribers[event.panel].discard(ws)


def _static_dir_candidates() -> list[Path]:
    """Return the built frontend's install-layout candidates, in preference order.

    The packaged copy inside an installed wheel, then the src-layout checkout.
    """
    return [
        Path(__file__).parent / "static",
        Path(__file__).parent.parent.parent / "static",
    ]


def _find_static_dir() -> Path:
    """Locate the built frontend across install layouts.

    Prefers a copy packaged inside the installed package (``tcip_web/static/``) and falls back
    to the src-layout checkout (``packages/tcip-web/static/``). Returns the src-layout path if
    neither is built yet.
    """
    candidates = _static_dir_candidates()
    for c in candidates:
        if (c / "index.html").exists():
            return c
    return candidates[-1]


# Serve static files (web frontend)
STATIC_DIR = _find_static_dir()
if STATIC_DIR.exists():
    # The built frontend references absolute /assets/... paths (Vite's default base="/"), so
    # mount that subdirectory at /assets; /static stays for ad-hoc static resources.
    assets_dir = STATIC_DIR / "assets"
    if assets_dir.exists():
        app.mount(
            "/assets", StaticFiles(directory=str(assets_dir)), name="assets"
        )
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# The paths this app serves the built frontend and its health probe through, so the generated
# dev proxy never claims them and Vite keeps serving its own root, modules and HMR.
FRONTEND_SERVING_PATHS = frozenset({"/", "/health"})


# ── Health ──


@app.get("/health")
def health() -> dict:
    """Answer the health probe with ``{"status": "ok"}``."""
    return {"status": "ok"}


# ── Frontend ──


@app.get("/")
def index() -> Response:
    """Serve the built frontend's ``index.html``, or a 503 naming how to build it when absent."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    # Fail loudly instead of silently serving a stub: the API works, but the GUI bundle
    # hasn't been built. Tell the operator exactly how to build it.
    return JSONResponse(
        status_code=503,
        content={
            "error": "GUI not built",
            "detail": (
                "The frontend bundle is missing. Build it, then reload: "
                "cd packages/tcip-web/frontend && npm install && npm run build"
            ),
            "api_docs": "/docs",
        },
    )


# ── Panel events: POST endpoint (MCP tools) + WS subscription (browsers) ──


@app.post("/api/events/{panel}")
async def post_panel_event(
    panel: str, event: PanelEvent, request: Request
) -> dict:
    """Accept a pushed event and broadcast it to subscribers.

    Payload shape: ``{panel, event_type, data, project_id}``. An event naming any project but the
    open one answers 409 (``StateStore.admit``), naming the open one's ``open_project_id`` when
    another is open and the none-open message when none is, and is neither retained nor
    broadcast. The broadcast
    and replay payload, not this route's response, carries every agent identity field the sender
    declared in its headers (``agent_identity.HEADERS``, each ``None`` when not sent). Declared,
    not verified: any sender can set the headers.
    """
    from tcip_mcp import agent_identity

    if panel not in VALID_PANELS:
        return {
            "error": f"unknown panel: {panel}",
            "valid": sorted(VALID_PANELS),
        }
    project = _gui_store.admit(event.project_id)
    envelope = PanelEnvelope(
        panel=panel,
        event_type=event.event_type,
        data=event.data,
        event_id=f"{now_iso()}#{next(_event_counter)}",
        **agent_identity.fields_from_headers(request.headers),
    )
    _gui_store.retain_event(envelope)
    # Agent focus events also update the advisory GuiState slice, so the snapshot shows where the
    # agent pointed the human: the browser applies the event locally and never syncs these back.
    if event.event_type == PANEL_EVENT_ANNOTATE_FOCUS:
        mutation: dict[str, Any] = {"active_tab": "annotate"}
        if "mode" in event.data:
            mutation["mode"] = event.data["mode"]
        if "active_subject" in event.data:
            mutation["active_subject"] = event.data["active_subject"]
        await _gui_store.mutate(mutation, project=project)
    await _broadcast_to_panel(envelope)
    return {"status": "ok", "panel": panel, "event_type": event.event_type}


@app.websocket("/ws/panel/{panel}")
async def panel_ws(websocket: WebSocket, panel: str) -> None:
    """Stream ``panel``'s events over the socket, the open project's retained events first.

    An unknown panel closes the socket with 1008.
    """
    if panel not in VALID_PANELS:
        await websocket.close(code=1008, reason=f"unknown panel: {panel}")
        return
    await websocket.accept()
    _panel_subscribers[panel].add(websocket)
    # Replay the open project's retained events so a late-joining browser sees the current state.
    for event in _gui_store.retained_events(panel):
        try:
            await websocket.send_json(event.model_dump(mode="json"))
        except Exception:
            break
    try:
        while True:
            # Keep the socket open; clients are read-only subscribers here.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _panel_subscribers[panel].discard(websocket)
