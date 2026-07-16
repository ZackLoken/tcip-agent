"""FastAPI application — REST API for MCP tools + WebSocket for GUI state sync.

This backend is the single source of truth for live GUI state across the
TCIP tabs (Annotate / Review / Training / Tuning / Inference / Results).
Claude agent and browser clients both connect through here.

Domain endpoints live in ``tcip_web.routes`` (mounted via ``register_all``). This module
keeps only the app-level surface: GUI state snapshot + WS, the agent panel-event hub,
static-file serving, and health/index.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

from tcip_web.paths import is_loopback_host, origin_allowed

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Rehydrate persisted job/sweep/run registries so GUI history survives a restart.

    Their worker threads don't survive, so rehydrated non-terminal entries surface as
    'interrupted' — a record, not a resumable job (see ``jobstore``).
    """
    try:
        from tcip_web.routes import inference, tuning

        inference.rehydrate()
        tuning.rehydrate()
        # Training runs aren't rehydrated from a state file — the training list route
        # reconstructs past runs on demand from the immutable .tcip/experiments/ records.
    except Exception:  # pragma: no cover - rehydrate is best-effort
        logger.exception("job registry rehydrate failed")
    yield
    # Kill any live agent terminals so no Claude Code process orphans the backend.
    # Off-loop: terminate can block seconds (taskkill / SIGTERM grace), and stalling the
    # event loop here would break the in-flight WebSocket close handshakes.
    try:
        import asyncio

        from tcip_web.routes import terminal

        await asyncio.to_thread(terminal.shutdown_all)
    except Exception:  # pragma: no cover - shutdown cleanup is best-effort
        logger.exception("agent terminal shutdown failed")


app = FastAPI(title="TCIP Pipeline", version="0.1.0", lifespan=_lifespan)

# CORS is not enabled by default — browser is expected to hit the same origin
# via the Vite dev proxy. If we ever serve the frontend elsewhere, add
# fastapi.middleware.cors.CORSMiddleware here.

# ── Trust boundary ──
# Keep local single-user use frictionless (loopback bind → no auth) while closing the two
# browser-facing holes: cross-site WebSocket reads of GUI state (Origin check below) and
# DNS-rebinding (Host allow-list). A deliberately network-exposed bind is gated further in
# __main__ (refuses to bind non-loopback without an explicit opt-in).
_BIND_HOST = os.environ.get("TCIP_WEB_HOST", "127.0.0.1")
_EXPOSED = not is_loopback_host(_BIND_HOST)

# ``testserver`` is Starlette's TestClient default Host; a real deployment never sees it.
_TRUSTED_HOSTS = ["*"] if _EXPOSED else ["localhost", "127.0.0.1", "testserver"]
app.add_middleware(TrustedHostMiddleware, allowed_hosts=_TRUSTED_HOSTS)

# Compress JSON/text responses above ~1KB. The /api/review/matches payload scales with
# polygon count (dense images ship high-hundreds-of-KB to multi-MB uncompressed JSON).
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ── Tab routes ──
from tcip_web.routes import register_all as _register_routes  # noqa: E402  (needs `app`)
_register_routes(app)

# ── State snapshot + WS ──
from tcip_web.state import store as _gui_store  # noqa: E402  (needs `app`)
_state_watchers: set[WebSocket] = set()


async def _broadcast_state_snapshot(payload: dict[str, Any]) -> None:
    """Push the new state (with its version) to every connected browser."""
    msg = {"type": "state_snapshot", "state": payload["state"], "version": payload["version"]}
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


@app.websocket("/ws/state")
async def state_ws(websocket: WebSocket) -> None:
    """Receive live GuiState deltas. Replays the current snapshot on connect."""
    if not origin_allowed(websocket.headers.get("origin")):
        await websocket.close(code=1008, reason="origin not allowed")
        return
    await websocket.accept()
    _state_watchers.add(websocket)
    try:
        await websocket.send_json(
            {"type": "state_snapshot", "state": _gui_store.snapshot(), "version": _gui_store.version}
        )
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _state_watchers.discard(websocket)


# ── Panel event hub (replaces .tcip/events/ file bridge) ──

# "app" is the app-level channel the agent uses to steer the GUI ("look here"): set the
# active project, navigate a tab, select an image. The tab panels stay data-scoped.
VALID_PANELS = {"app", "annotate", "review", "training", "tuning", "inference", "results"}

# Recent events per panel, kept in memory for replay on reconnect.
_recent_events: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=64))
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

def _find_static_dir() -> Path:
    """Locate the built frontend across install layouts.

    Prefers a copy packaged inside the installed package (``tcip_web/static/`` — how a
    wheel should ship it) and falls back to the src-layout checkout
    (``packages/tcip-web/static/``). Returns the src-layout path if neither is built yet,
    so ``/`` can render build instructions rather than 404.
    """
    candidates = [
        Path(__file__).parent / "static",  # packaged in a wheel
        Path(__file__).parent.parent.parent / "static",  # src-layout checkout
    ]
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


@app.post("/api/events/{panel}")
async def post_panel_event(panel: str, event: PanelEvent):
    """Accept an event pushed from an MCP tool and broadcast to subscribers.

    Replaces the legacy ``.tcip/events/`` file-bridge. The payload schema
    matches the old file format: ``{panel, event_type, data}``.
    """
    if panel not in VALID_PANELS:
        return {"error": f"unknown panel: {panel}", "valid": sorted(VALID_PANELS)}
    payload = {
        "panel": panel,
        "event_type": event.event_type,
        "data": event.data,
    }
    _recent_events[panel].append(payload)
    # Agent focus events also update the advisory GuiState slice, so gui.json (what the
    # agent reads back via get_active_context) reflects where it pointed the human — the
    # browser applies the event locally and never syncs these fields back itself.
    if event.event_type == "review_focus":
        review = _gui_store.state.review.model_copy(
            update={
                k: event.data[k]
                for k in ("filter_type", "iou_threshold", "conf_threshold", "detection_idx")
                if k in event.data
            }
        )
        await _gui_store.mutate({"active_tab": "review", "review": review})
    elif event.event_type == "annotate_focus":
        mutation: dict[str, Any] = {"active_tab": "annotate"}
        if "mode" in event.data:
            mutation["mode"] = event.data["mode"]
        if "active_class" in event.data:
            mutation["active_class"] = event.data["active_class"]
        await _gui_store.mutate(mutation)
    await _broadcast_to_panel(panel, payload)
    return {"status": "ok", "panel": panel, "event_type": event.event_type}


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
    if not origin_allowed(websocket.headers.get("origin")):
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
