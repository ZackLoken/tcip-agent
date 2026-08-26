"""FastAPI application: REST API for MCP tools + WebSocket for GUI state sync.

This backend is the single source of truth for live GUI state across the
TCIP tabs (Annotate / Review / Training / Tuning / Inference / Results).
Claude agent and browser clients both connect through here.

Domain endpoints live in ``tcip_web.routes`` (mounted via ``register_all``). This module
keeps only the app-level surface: GUI state snapshot + WS, the agent panel-event hub,
static-file serving, and health/index.
"""

from __future__ import annotations

import itertools
import logging
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
from tcip_web.trust_boundary import TrustBoundaryMiddleware, log_exposure_opt_in, origin_allowed

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Rehydrate persisted job/sweep/run registries so GUI history survives a restart.

    Their worker threads don't survive, so rehydrated non-terminal entries surface as
    'interrupted' (a record, not a resumable job; see ``jobstore``).
    """
    log_exposure_opt_in()
    # Size GDAL's block cache once per process, at the entry point, never at source construction.
    from tcip_mcp.pipelines.raster_source import configure_gdal_cache

    configure_gdal_cache()
    try:
        from tcip_web.routes import inference, tuning

        inference.rehydrate()
        tuning.rehydrate()
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
        import asyncio

        from tcip_web.routes import terminal as terminal_routes

        await asyncio.to_thread(terminal_routes.shutdown_all)
    except Exception:  # pragma: no cover - shutdown cleanup is best-effort
        logger.exception("agent terminal shutdown failed")


# At import, not in the lifespan: a route may be exercised against this app without one running,
# and a route that reaches a store with no backend bound would refuse rather than write.
bind_default()
# At import too, ahead of the lifespan's rehydrate: every way this app is served (python -m
# tcip_web, bare uvicorn, --lifespan off, the reloader's child) binds the same root this way.
from tcip_mcp.project_paths import pin_project_root  # noqa: E402

pin_project_root(from_marker=True)

app = FastAPI(title="TCIP Pipeline", version="0.1.0", lifespan=_lifespan)

# CORS is not enabled by default: browser is expected to hit the same origin
# via the Vite dev proxy. If we ever serve the frontend elsewhere, add
# fastapi.middleware.cors.CORSMiddleware here.

# Exposure is decided per connection from its arrival address and the Host must name this
# backend (tcip_web.trust_boundary); the WebSocket routes apply the Origin policy before accept.
app.add_middleware(TrustBoundaryMiddleware)

# Compress JSON/text responses above ~1KB. The /api/review/matches payload scales with
# polygon count (dense images ship high-hundreds-of-KB to multi-MB uncompressed JSON).
app.add_middleware(GZipMiddleware, minimum_size=1000)

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


def state_snapshot_message(state: dict[str, Any], version: int) -> dict[str, Any]:
    """The one envelope shape both the broadcast and the connect-time replay send."""
    return {"type": "state_snapshot", "state": state, "version": version}


async def _broadcast_state_snapshot(payload: dict[str, Any]) -> None:
    """Push the new state (with its version) to every connected browser."""
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
    """Receive live GuiState deltas. Replays the current snapshot on connect."""
    if not origin_allowed(websocket.headers.get("origin"), websocket.scope):
        await websocket.close(code=1008, reason="origin not allowed")
        return
    await websocket.accept()
    _state_watchers.add(websocket)
    try:
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
    """
    from tcip_mcp import workspace
    from tcip_mcp.project_paths import repin_platform_root

    try:
        found = workspace.active_project_if_present(create=False)
        if found is None:
            name = workspace.read_active_project(create=False)
            if name:
                workspace.adoptable_project_root(name)  # raises, naming why it is not adoptable
            return {}
        marker_name, marker_root = found
    except Exception as exc:  # noqa: BLE001 - reported in the response, never raised
        return {"platform_root_problem": str(exc)}

    repin_platform_root(marker_root)
    # The per-registry rehydrate_for_current_root() calls land here; for now this repin can
    # leave a registry rehydrated from whichever root emptied it first (see rehydrate() below).
    try:
        from tcip_web.routes import inference, tuning

        inference.rehydrate()
        tuning.rehydrate()
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
        root_fields = _repin_from_active_project_event(event.data.get("name"))
    await _broadcast_to_panel(panel, payload)
    return {"status": "ok", "panel": panel, "event_type": event.event_type, **root_fields}


@app.get("/api/events/{panel}/recent")
def get_recent_panel_events(panel: str, limit: int = 16):
    """Return the last N events for a panel (useful on browser reconnect)."""
    if panel not in VALID_PANELS:
        return {"error": f"unknown panel: {panel}", "valid": sorted(VALID_PANELS)}
    events = list(_recent_events.get(panel, ()))
    return {"panel": panel, "events": events[-limit:]}


@app.websocket("/ws/panel/{panel}")
async def panel_ws(websocket: WebSocket, panel: str):
    """Stream panel events to a browser client."""
    if not origin_allowed(websocket.headers.get("origin"), websocket.scope):
        await websocket.close(code=1008, reason="origin not allowed")
        return
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
