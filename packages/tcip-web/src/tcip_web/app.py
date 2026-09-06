"""FastAPI application: REST API for MCP tools + WebSocket for GUI state sync.

This backend is the single source of truth for live GUI state across the
TCIP tabs (Annotate / Review / Training / Tuning / Inference / Results).
Claude agent and browser clients both connect through here.

Domain endpoints live in ``tcip_web.routes`` (mounted via ``register_all``). This module
keeps only the app-level surface: GUI state snapshot + WS, the agent panel-event hub,
static-file serving, and health/index.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import sys
import threading
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from tcip_store.binding import bind_default

from tcip_mcp.web_client import (
    PANEL_EVENT_ACTIVE_PROJECT_CHANGED,
    PANEL_EVENT_ANNOTATE_FOCUS,
    PANEL_EVENT_REVIEW_FOCUS,
    VALID_PANELS,
)
from tcip_mcp.workspace import configured_workspace
from tcip_web.trust_boundary import TrustBoundaryMiddleware, log_exposure_opt_in

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Rehydrate persisted job/sweep/run registries so GUI history survives a restart.

    Their worker threads don't survive, so rehydrated non-terminal entries surface as
    'interrupted' (a record, not a resumable job; see ``jobstore``).
    """
    log_exposure_opt_in()
    bind_startup_root()
    # The canvas-open binding record outlives this process's restart; read it once now so the
    # first connect-time replay answers from the durable record rather than a fresh None.
    await asyncio.to_thread(_gui_store.refresh_binding_generation_from_record)
    # Size GDAL's block cache once per process, at the entry point, never at source construction.
    from tcip_mcp.pipelines.raster_source import configure_gdal_cache

    configure_gdal_cache()
    try:
        from tcip_web import jobstore
        from tcip_web.routes import inference, review, tuning

        # One try per registry, so a refused rehydrate never skips the other two.
        for registry_name, rehydrate in (
            (jobstore.INFERENCE_JOBS, inference.rehydrate_for_current_root),
            (jobstore.HPO_SWEEPS, tuning.rehydrate_for_current_root),
            (jobstore.REVIEW_PRIORITY_JOBS, review.rehydrate_for_current_root),
        ):
            try:
                rehydrate()
            except Exception as exc:  # pragma: no cover - rehydrate is best-effort
                logger.exception("%s registry rehydrate refused", registry_name)
                jobstore.record_startup_refusal(registry_name, str(exc))
        # Training runs aren't rehydrated from a state file: the training list route
        # reconstructs past runs on demand from the immutable .tcip/experiments/ records.
    except Exception:  # pragma: no cover - rehydrate is best-effort
        logger.exception("job registry rehydrate failed")
    # Warm the cold first-spawn cost (tcip_mcp import + PowerShell/.NET) off the request path.
    try:
        from tcip_web import terminal

        terminal.prewarm()
    except Exception:  # pragma: no cover - prewarm is best-effort
        logger.exception("agent terminal prewarm failed to start")
    yield
    # Kill any live agent terminals so no Claude Code process orphans the backend.
    # Off-loop: terminate can block seconds (taskkill / SIGTERM grace), and stalling the
    # event loop here would break the in-flight WebSocket close handshakes.
    try:
        from tcip_web.routes import terminal as terminal_routes

        await asyncio.to_thread(terminal_routes.shutdown_all)
    except Exception:  # pragma: no cover - shutdown cleanup is best-effort
        logger.exception("agent terminal shutdown failed")


# At import, not in the lifespan: a route may be exercised against this app without one running,
# and a route that reaches a store with no backend bound would refuse rather than write.
bind_default()


class WorkspaceUnsetUnderTest(RuntimeError):
    """Raised in place of pinning a platform-state root, when this app is served under a test
    with no ``TCIP_WORKSPACE`` bound.

    Left unrefused, :func:`bind_startup_root` would resolve the default workspace
    (``tcip_mcp.workspace.workspace_root``, ``~/tcip-projects``) and pin, then write into,
    whichever project the operator's real active-project marker names: a pytest process, a
    process that has loaded an in-process test client, or a request arriving from one has no
    business touching that project.
    """


def _running_under_pytest() -> bool:
    """True once a pytest process has begun collection, for the whole run.

    pytest inserts itself into ``sys.modules`` at startup and never removes itself.
    """
    return "pytest" in sys.modules


def _in_process_test_client_loaded() -> bool:
    """True once this process has imported starlette's ``TestClient`` module, for the whole run.

    Entering ``with TestClient(app):`` runs the lifespan with no request and so no ASGI scope,
    which :func:`bind_startup_root` calls before either check below can see one; this is the
    signal that catches that case. Verified in a fresh subprocess: importing ``tcip_web.app``
    alone leaves ``starlette.testclient`` out of ``sys.modules``, while importing
    ``fastapi.testclient`` (which re-exports ``starlette.testclient.TestClient``) or
    ``starlette.testclient`` directly brings it in, and neither module removes itself once
    imported.
    """
    return "starlette.testclient" in sys.modules


_IN_PROCESS_TEST_TRANSPORT_CLIENTS = frozenset({("testclient", 50000), ("127.0.0.1", 123)})
"""The client addresses an in-process ASGI test transport stamps on the scope it hands the app.
``starlette.testclient.TestClient.__init__`` defaults its ``client`` argument to
``("testclient", 50000)``, and ``httpx.ASGITransport.__init__`` defaults its own ``client``
argument to ``("127.0.0.1", 123)`` (both verified against the installed packages); real network
traffic never arrives with either identity, source port 123 least of all, a privileged port."""


def _scope_is_in_process_test_client(scope: dict[str, Any]) -> bool:
    """True when an ASGI scope's client address matches an in-process test transport's default
    identity (:data:`_IN_PROCESS_TEST_TRANSPORT_CLIENTS`), naming both starlette's ``TestClient``
    and httpx's ``ASGITransport`` rather than starlette's alone."""
    client = scope.get("client")
    return client is not None and tuple(client) in _IN_PROCESS_TEST_TRANSPORT_CLIENTS


def raise_if_workspace_unset_under_test(scope: dict[str, Any] | None = None) -> None:
    """Refuse to pin a platform-state root under a test that never set ``TCIP_WORKSPACE``.

    Runs ahead of every marker read this app performs: at the top of
    :func:`bind_startup_root` with no scope available (the pytest-process and test-client-import
    signals) and in :class:`_BindStartupRootMiddleware` with the request's own scope (all three
    signals). Passes silently once ``TCIP_WORKSPACE`` is configured
    (:func:`tcip_mcp.workspace.configured_workspace`) or nothing signals a test is running, so a
    served app (``python -m tcip_web``) keeps its default workspace untouched.
    """
    if configured_workspace() is not None:
        return
    process_signal = _running_under_pytest() or _in_process_test_client_loaded()
    scope_signal = scope is not None and _scope_is_in_process_test_client(scope)
    if not (process_signal or scope_signal):
        return
    raise WorkspaceUnsetUnderTest(
        "TCIP_WORKSPACE is unset. This served app would otherwise pin the workspace's "
        "active project and write into it. Set TCIP_WORKSPACE and TCIP_STATE_ROOT to "
        "scratch directories before starting a test client against it."
    )


def bind_startup_root() -> None:
    """Pin this process's platform-state root once, a served app's own responsibility rather
    than an importer's, and only when nothing has bound one yet.

    Reached from :class:`_BindStartupRootMiddleware` (ahead of every route, for a request
    served before the lifespan has run) and from the lifespan's own startup (ahead of its
    rehydrate), so every way this app is served (``python -m tcip_web``, bare uvicorn,
    ``--lifespan off``, the reloader's child) pins a root before anything resolves one,
    while a process that only imports this module (the test suite at collection, a repo
    script) never calls this and pins nothing.

    Checks :func:`tcip_mcp.project_paths.root_binding` rather than a flag of its own: a
    ``activate_project`` repin that lands before the first request already leaves a
    binding in place, and this must not replace it with a fresh marker read.

    Raises :class:`WorkspaceUnsetUnderTest` first, before either check, when this process is
    a pytest run or has loaded an in-process test client with no ``TCIP_WORKSPACE`` bound; the
    scope-carried signals of that rail live in :class:`_BindStartupRootMiddleware`, which alone
    sees the request scope.
    """
    raise_if_workspace_unset_under_test()

    from tcip_mcp.project_paths import pin_platform_root, root_binding

    if root_binding() is not None:
        return
    pin_platform_root(from_marker=True)


_bind_startup_root_lock = threading.Lock()


def _bind_startup_root_serialized() -> None:
    """:func:`bind_startup_root` under the module lock, the body the middleware runs in a
    worker thread.

    Concurrent first requests serialize onto one marker read here: the second waits, then
    finds the root already bound and returns. The lock is a thread lock taken inside the
    worker thread, never an asyncio lock, since concurrent requests may arrive on several
    event loops (one ``TestClient`` per thread), which a lock bound to one loop cannot serve.
    """
    with _bind_startup_root_lock:
        bind_startup_root()


class _BindStartupRootMiddleware:
    """ASGI middleware that calls :func:`bind_startup_root` ahead of every request.

    Covers a request served with the lifespan disabled or never started (``--lifespan off``,
    a ``TestClient`` used outside its context manager), where the lifespan's own call never
    runs. The marker read :func:`bind_startup_root` may perform is a store read bounded by a
    file-lock timeout, so it runs off the event loop in a worker thread, under the module
    lock above; once a root is bound the check inside :func:`bind_startup_root` is cheap, so
    later requests still pay the thread hop but no further store read.

    Checks :func:`raise_if_workspace_unset_under_test` with this request's own scope before
    that thread hop: the scope carries the client-address signal :func:`bind_startup_root`
    cannot see on its own, so a request from a bare ``TestClient`` or an ``httpx.ASGITransport``
    client with no ``TCIP_WORKSPACE`` bound is refused here even outside a pytest process.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] in ("http", "websocket"):
            raise_if_workspace_unset_under_test(scope)
            await asyncio.to_thread(_bind_startup_root_serialized)
        await self.app(scope, receive, send)


app = FastAPI(title="TCIP Pipeline", version="0.1.0", lifespan=_lifespan)

# CORS is not enabled by default: the browser hits the same origin via the Vite dev proxy.
# Serving the frontend elsewhere would add fastapi.middleware.cors.CORSMiddleware here.

# Exposure is decided per connection from its arrival address and the Host must name this
# backend; the middleware also applies the Origin policy before a route runs (trust_boundary).
app.add_middleware(TrustBoundaryMiddleware)

# Compress JSON/text responses above ~1KB. The /api/review/matches payload scales with
# polygon count (dense images ship high-hundreds-of-KB to multi-MB uncompressed JSON).
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Added last: Starlette's add_middleware makes the most recently added the outermost, so this
# pin resolves ahead of every other middleware and route.
app.add_middleware(_BindStartupRootMiddleware)

# ── Tab routes ──
from tcip_web.routes import register_all as _register_routes  # noqa: E402  (needs `app`)
_register_routes(app)

# ── State snapshot + WS ──
from tcip_web.state import GuiMutationInvalid, store as _gui_store  # noqa: E402  (needs `app`)
_state_watchers: set[WebSocket] = set()


@app.exception_handler(GuiMutationInvalid)
async def _gui_mutation_invalid_handler(_request: Request, exc: GuiMutationInvalid) -> JSONResponse:
    """Every route that mutates GUI state answers an invalid mutation with 400 and the reason,
    rather than the 500 an unhandled ``ValueError`` would otherwise produce."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# This process's launch identity, minted once at import: rides every snapshot envelope so a
# restarted backend's lower-numbered first snapshot is accepted across a client's wsVersion guard.
SERVER_EPOCH = uuid.uuid4().hex


def state_snapshot_message(state: dict[str, Any], version: int) -> dict[str, Any]:
    """The one envelope shape both the broadcast and the connect-time replay send.

    ``generation`` is read off ``StateStore`` rather than the binding store, so a broadcast
    never costs a store read on the event loop.
    """
    return {
        "type": "state_snapshot",
        "state": state,
        "version": version,
        "generation": _gui_store.binding_generation,
        "epoch": SERVER_EPOCH,
    }


async def _broadcast_state_snapshot(payload: dict[str, Any]) -> None:
    """Push the new state, version and binding generation to every connected browser."""
    msg = state_snapshot_message(payload["state"], payload["version"])
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
    return _gui_store.snapshot()


class ActiveTabPayload(BaseModel):
    active_tab: str


@app.post("/api/state/tab")
async def set_active_tab(payload: ActiveTabPayload) -> dict:
    """Record which tab the browser is actually showing, so ``gui.json`` (and everything
    that reads it, like ``view_gui_state``) tracks what the human sees."""
    await _gui_store.mutate({"active_tab": payload.active_tab})
    return {"status": "ok", "active_tab": payload.active_tab}


@app.websocket("/ws/state")
async def state_ws(websocket: WebSocket) -> None:
    """Push live GuiState snapshots to the browser; replays the current snapshot on connect.
    One-directional: the client never sends a payload over this socket, and an inbound frame,
    if one ever arrived, is read and discarded, only to detect disconnect."""
    await websocket.accept()
    _state_watchers.add(websocket)
    try:
        # Connects are rare: re-read the binding record before replaying, off the event loop.
        await asyncio.to_thread(_gui_store.refresh_binding_generation_from_record)
        await websocket.send_json(
            state_snapshot_message(_gui_store.snapshot(), _gui_store.version)
        )
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _state_watchers.discard(websocket)


# ── Panel event hub (replaces .tcip/events/ file bridge) ──

# Recent events per panel, kept in memory for replay on reconnect.
_recent_events: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=64))
# Per-event identity, so a browser can tell an event it has already acted on (dismissed a
# banner, say) from a new one with the same text. The timestamp keeps ids distinct across a
# backend restart, which the counter alone would not: it restarts at 0 while browsers keep
# their dismissals.
_event_counter = itertools.count(1)
# Open WebSocket subscribers per panel.
_panel_subscribers: dict[str, set[WebSocket]] = defaultdict(set)


class PanelEvent(BaseModel):
    panel: str | None = None
    event_type: str
    data: dict[str, Any] = {}


async def _broadcast_to_panel(panel: str, event: dict[str, Any]) -> None:
    """Fan out an event to every WebSocket currently subscribed to a panel."""
    dead: list[WebSocket] = []
    for ws in list(_panel_subscribers.get(panel, ())):
        try:
            await ws.send_json(event)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _panel_subscribers[panel].discard(ws)

def _static_dir_candidates() -> list[Path]:
    """The install-layout candidates ``_find_static_dir`` checks, in preference order: the
    packaged copy inside an installed wheel, then the src-layout checkout."""
    return [
        Path(__file__).parent / "static",
        Path(__file__).parent.parent.parent / "static",
    ]


def _find_static_dir() -> Path:
    """Locate the built frontend across install layouts.

    Prefers a copy packaged inside the installed package (``tcip_web/static/``, how a
    wheel should ship it) and falls back to the src-layout checkout
    (``packages/tcip-web/static/``). Returns the src-layout path if neither is built yet,
    so ``/`` can render build instructions rather than 404.
    """
    candidates = _static_dir_candidates()
    for c in candidates:
        if (c / "index.html").exists():
            return c
    return candidates[-1]


# Serve static files (web frontend)
STATIC_DIR = _find_static_dir()
if STATIC_DIR.exists():
    # The built frontend references absolute /assets/... paths (Vite's default
    # base="/"), so mount that subdirectory at /assets. Keep /static available
    # for any ad-hoc static resources.
    assets_dir = STATIC_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# The paths this app serves the built frontend and its health probe through, so the generated
# dev proxy never claims them and Vite keeps serving its own root, modules and HMR.
FRONTEND_SERVING_PATHS = frozenset({"/", "/health"})


# ── Health ──


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# ── Frontend ──


@app.get("/")
def index():
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


def _repin_from_active_project_event(sent_name: Any) -> dict[str, Any]:
    """Re-read the workspace's active-project marker and repin this process to it.

    Never trusts ``sent_name``: an unauthenticated event is a signal to re-read the marker
    the breeder controls, not a name to act on. Returns the fields the event route's response
    carries: ``platform_root`` on a repin, or ``platform_root_problem`` naming why the marker
    could not be used (a store refusal, a lock timeout, or a marker naming a project that is
    not adoptable), never both. A repin's ``platform_root_disagreement`` says when the event's
    own name differed from what the marker actually named.

    Runs on the calling thread; the route awaits this in a worker thread so the marker read
    and the rehydrates below never block the event loop.
    """
    from tcip_mcp import workspace
    from tcip_mcp.project_paths import repin_platform_root

    try:
        found = workspace.active_project_if_present(create=False)
        if found is None:
            problem = workspace.marker_problem(create=False)
            return {"platform_root_problem": problem} if problem else {}
        marker_name, marker_root = found
    except Exception as exc:  # noqa: BLE001 - reported in the response, never raised
        return {"platform_root_problem": str(exc)}

    repin_platform_root(marker_root)
    try:
        from tcip_web.routes import inference, review, tuning

        inference.rehydrate_for_current_root()
        tuning.rehydrate_for_current_root()
        review.rehydrate_for_current_root()
    except Exception:  # pragma: no cover - rehydrate is best-effort, same as at startup
        logger.exception("job registry rehydrate failed after a platform-root repin")

    result: dict[str, Any] = {"platform_root": str(marker_root)}
    if sent_name is not None and sent_name != marker_name:
        result["platform_root_disagreement"] = {"event_name": sent_name, "marker_name": marker_name}
    return result


@app.post("/api/events/{panel}")
async def post_panel_event(panel: str, event: PanelEvent, request: Request):
    """Accept an event pushed from an MCP tool and broadcast to subscribers.

    Payload shape: ``{panel, event_type, data}``. The broadcast and replay payload, not this
    route's response, carries every agent identity field the sender declared in its headers
    (``agent_identity.HEADERS``, each ``None`` when not sent), so a browser and the replay can say
    which harness steered the GUI. Declared, not verified: any sender can set the headers.
    """
    from tcip_mcp import agent_identity

    if panel not in VALID_PANELS:
        return {"error": f"unknown panel: {panel}", "valid": sorted(VALID_PANELS)}
    payload = {
        "panel": panel,
        "event_type": event.event_type,
        "data": event.data,
        "event_id": f"{datetime.now(timezone.utc).isoformat()}#{next(_event_counter)}",
        **agent_identity.fields_from_headers(request.headers),
    }
    _recent_events[panel].append(payload)
    # Agent focus events also update the advisory GuiState slice, so gui.json (what the
    # agent reads back via view_gui_state) reflects where it pointed the human: the
    # browser applies the event locally and never syncs these fields back itself.
    if event.event_type == PANEL_EVENT_REVIEW_FOCUS:
        review = _gui_store.state.review.model_copy(
            update={
                k: event.data[k]
                for k in ("filter_type", "iou_threshold", "conf_threshold", "detection_idx")
                if k in event.data
            }
        )
        await _gui_store.mutate({"active_tab": "review", "review": review})
    elif event.event_type == PANEL_EVENT_ANNOTATE_FOCUS:
        mutation: dict[str, Any] = {"active_tab": "annotate"}
        if "mode" in event.data:
            mutation["mode"] = event.data["mode"]
        if "active_subject" in event.data:
            mutation["active_subject"] = event.data["active_subject"]
        await _gui_store.mutate(mutation)
    root_fields: dict[str, Any] = {}
    if event.event_type == PANEL_EVENT_ACTIVE_PROJECT_CHANGED:
        import asyncio

        root_fields = await asyncio.to_thread(
            _repin_from_active_project_event, event.data.get("name")
        )
    await _broadcast_to_panel(panel, payload)
    return {"status": "ok", "panel": panel, "event_type": event.event_type, **root_fields}


@app.websocket("/ws/panel/{panel}")
async def panel_ws(websocket: WebSocket, panel: str):
    """Stream panel events to a browser client."""
    if panel not in VALID_PANELS:
        await websocket.close(code=1008, reason=f"unknown panel: {panel}")
        return
    await websocket.accept()
    _panel_subscribers[panel].add(websocket)
    # Replay recent events so late-joining browsers see the current state
    for event in list(_recent_events.get(panel, ())):
        try:
            await websocket.send_json(event)
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
