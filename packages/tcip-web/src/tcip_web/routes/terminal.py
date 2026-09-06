"""Agent terminal routes: the HTTP/WS surface over :mod:`tcip_web.terminal`.

One live terminal session (the co-pilot rail) is the norm; the API is session-plural so
multiples need no redesign. The WebSocket carries raw PTY output as text frames
(server → browser) and JSON control messages (browser → server), validated as
``TerminalInputFrame``/``TerminalResizeFrame``:

    {"type": "input",  "data": "<keystrokes>"}
    {"type": "resize", "rows": 34, "cols": 96}

Delivery model: the PTY reader thread appends output to a capped scrollback and pushes
it to one queue per connected WebSocket via ``loop.call_soon_threadsafe`` (FIFO), and a
single pump task per socket drains that queue: byte order is load-bearing for a
terminal stream, so exactly one writer task per socket. On (re)connect the scrollback
snapshot and queue registration happen under the writer's lock, so the replay is
gap-free and duplicate-free. All endpoints sit behind the loopback + Origin trust
boundary; the terminal must never be network-exposed without adding auth (it is
keyboard access to Claude Code).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import threading
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError

from tcip_web import terminal as pty_host

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/terminal", tags=["terminal"])

# Scrollback cap (chars). Enough to re-render a long session's tail without unbounded
# memory; the TUI repaints itself on the next output anyway.
SCROLLBACK_MAX_CHARS = 400_000

MAX_DIM = 500  # sanity bound on client-supplied rows/cols

# Per-subscriber delivery queue cap. A stalled browser (frozen tab, suspended laptop)
# stops draining while the TUI keeps painting; past this we drop the backlog and close
# that socket: the client reconnects and repaints from the scrollback replay.
QUEUE_MAX_CHUNKS = 2048

_EXIT_NOTE = "\r\n\x1b[2m[Claude Code exited, use Restart in the rail header]\x1b[22m\r\n"


def _offer(queue: asyncio.Queue, data: str) -> None:
    """Enqueue output for one subscriber (runs on that subscriber's loop).

    On overflow, drop the backlog and leave a ``None`` sentinel: the pump closes the
    socket, and the reconnect replay is a coherent repaint (unlike dropping chunks
    mid-stream, which would tear ANSI sequences).
    """
    try:
        queue.put_nowait(data)
    except asyncio.QueueFull:
        try:
            while True:
                queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        queue.put_nowait(None)


def _record_start(session_id: str, launched: dict) -> None:
    """One platform audit line per launch, naming the session id and the program it launched.

    The MCP server the agent starts reads this id from its environment and stamps it on its own
    lines as a declared correlation (any launcher can set that variable, so those lines say what
    the process claimed); this line says what the backend itself launched under the id. Never
    fails the request.
    """
    from tcip_mcp.audit import record_event

    record_event("agent_terminal_started", {"session_id": session_id, **launched})


class TerminalSession:
    """One PTY-attached Claude Code process and its subscriber queues."""

    def __init__(self, session_id: str):
        self.id = session_id
        # What the last start launched (executable and declared version), None before a start.
        self.launched: Optional[dict] = None
        self._pty = None
        self._lock = threading.Lock()
        self._scrollback: list[str] = []
        self._scrollback_len = 0
        # ws-id → (queue, that websocket's event loop)
        self._subs: dict[int, tuple[asyncio.Queue, asyncio.AbstractEventLoop]] = {}
        self._next_sub = 0
        # PTY generation: bumped on every (re)start so a stale reader thread (the old
        # process draining after a restart) can't inject output or its exit note into
        # the fresh session's scrollback/stream.
        self._gen = 0

    # ── lifecycle ───────────────────────────────────────────────────────

    def start(self, rows: int, cols: int) -> Optional[str]:
        """Spawn the CLI in a PTY. Returns an error reason, or None on success."""
        with self._lock:
            if self._pty is not None and self._pty.isalive():
                return None
            argv = pty_host.resolve_terminal_command()
            if argv is None:
                return pty_host.terminal_status().get("reason")
            launched = pty_host.launched_program(argv)
            try:
                pty = pty_host.spawn_pty(
                    argv, pty_host.terminal_cwd(), rows, cols, pty_host.spawn_env(self.id)
                )
            except OSError as exc:
                self._pty = None
                return f"could not start the agent terminal: {exc}"
            self._pty = pty
            self.launched = launched
            self._gen += 1
            gen = self._gen
        pty_host.start_reader(
            pty,
            lambda data: self._on_output(data, gen),
            lambda: self._on_exit(gen),
            name=f"term-{self.id}-g{gen}",
        )
        _record_start(self.id, launched)
        return None

    def restart(self, rows: int, cols: int) -> Optional[str]:
        self.terminate()
        with self._lock:
            self._scrollback = []
            self._scrollback_len = 0
        return self.start(rows, cols)

    def terminate(self) -> None:
        with self._lock:
            pty, self._pty = self._pty, None
        if pty is not None:
            try:
                pty.terminate()
            except Exception:  # pragma: no cover - best-effort cleanup
                logger.debug("terminal terminate failed", exc_info=True)

    def alive(self) -> bool:
        pty = self._pty
        return bool(pty is not None and pty.isalive())

    # ── I/O ─────────────────────────────────────────────────────────────

    def write(self, data: str) -> bool:
        pty = self._pty
        if pty is None or not pty.isalive():
            return False
        try:
            pty.write(data)
            return True
        except Exception:
            # pywinpty raises EOFError / WinptyError (not OSError) when the process
            # dies under the write; any failure here means the same thing: not sent.
            return False

    def resize(self, rows: int, cols: int) -> None:
        pty = self._pty
        if pty is None:
            return
        try:
            pty.resize(rows, cols)
        except Exception:
            # Same non-OSError zoo as write(); a failed resize on a dead/dying PTY must
            # never crash the WebSocket handler (the rail resizes on every reconnect).
            pass

    # ── output pump (called from the reader thread) ─────────────────────

    def _on_output(self, data: str, gen: Optional[int] = None) -> None:
        with self._lock:
            if gen is not None and gen != self._gen:
                return  # stale reader from a restarted PTY: drop, don't pollute
            self._scrollback.append(data)
            self._scrollback_len += len(data)
            while self._scrollback_len > SCROLLBACK_MAX_CHARS and len(self._scrollback) > 1:
                dropped = self._scrollback.pop(0)
                self._scrollback_len -= len(dropped)
            dead: list[int] = []
            for sub_id, (queue, loop) in self._subs.items():
                if loop.is_closed():
                    dead.append(sub_id)
                    continue
                try:
                    # FIFO across calls → the pump drains in byte order.
                    loop.call_soon_threadsafe(_offer, queue, data)
                except RuntimeError:
                    dead.append(sub_id)
            for sub_id in dead:
                self._subs.pop(sub_id, None)

    def _on_exit(self, gen: Optional[int] = None) -> None:
        # Visible in the terminal itself: the surface must never just go quiet.
        self._on_output(_EXIT_NOTE, gen)

    # ── subscriber registration (called from each websocket's loop) ─────

    def attach(self) -> tuple[int, asyncio.Queue, str]:
        """Register a subscriber; returns ``(sub_id, queue, replay_snapshot)``.

        Snapshot + registration are atomic w.r.t. the writer, so output can be neither
        lost (arrived after snapshot, before registration) nor duplicated.
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX_CHUNKS)
        loop = asyncio.get_running_loop()
        with self._lock:
            replay = "".join(self._scrollback)
            sub_id = self._next_sub
            self._next_sub += 1
            self._subs[sub_id] = (queue, loop)
        return sub_id, queue, replay

    def detach(self, sub_id: int) -> None:
        with self._lock:
            self._subs.pop(sub_id, None)

    def scrollback_snapshot(self) -> str:
        """The current scrollback text (diagnostics / smoke checks)."""
        with self._lock:
            return "".join(self._scrollback)


_SESSIONS: dict[str, TerminalSession] = {}
# Serializes attach-or-spawn and restart across the request threadpool: without it,
# two concurrent POSTs each spawn a Claude Code process and one runs orphaned.
_SESSIONS_LOCK = threading.Lock()


def shutdown_all() -> None:
    """Kill every live agent terminal (called from the app lifespan on shutdown)."""
    for s in list(_SESSIONS.values()):
        s.terminate()
    _SESSIONS.clear()


# ── HTTP surface ────────────────────────────────────────────────────────


@router.get("/status")
def get_status() -> dict:
    return pty_host.terminal_status()


class TerminalInputFrame(BaseModel):
    """Keystrokes typed into the rail, forwarded to the PTY verbatim."""

    type: Literal["input"]
    data: str


class TerminalResizeFrame(BaseModel):
    """The rail's terminal dimensions, applied to the PTY's window size."""

    type: Literal["resize"]
    rows: int
    cols: int


class CreateSessionRequest(BaseModel):
    rows: int = pty_host.DEFAULT_ROWS
    cols: int = pty_host.DEFAULT_COLS


def _clamp(v: int) -> int:
    return max(2, min(MAX_DIM, int(v)))


@router.post("/sessions")
def create_session(req: CreateSessionRequest) -> dict:
    """Return the live session (attach semantics, like tmux) or spawn a fresh one."""
    with _SESSIONS_LOCK:
        for s in _SESSIONS.values():
            if s.alive():
                return {"session_id": s.id, "existing": True, "launched": s.launched}
        session_id = "term_" + os.urandom(6).hex()
        session = TerminalSession(session_id)
        err = session.start(_clamp(req.rows), _clamp(req.cols))
        if err:
            raise HTTPException(503, err)
        _SESSIONS[session_id] = session
        return {"session_id": session_id, "existing": False, "launched": session.launched}


def _require(session_id: str) -> TerminalSession:
    session = _SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, f"no such terminal session: {session_id}")
    return session


@router.post("/sessions/{session_id}/restart")
def restart_session(session_id: str, req: CreateSessionRequest) -> dict:
    session = _require(session_id)
    with _SESSIONS_LOCK:
        # One live agent at a time: restarting a stale session while a different one is
        # live would silently run two Claude Code processes.
        for other in _SESSIONS.values():
            if other.id != session_id and other.alive():
                raise HTTPException(
                    409, f"another agent session is live ({other.id}); attach to it instead"
                )
        err = session.restart(_clamp(req.rows), _clamp(req.cols))
    if err:
        raise HTTPException(503, err)
    return {"session_id": session_id, "alive": True, "launched": session.launched}


async def _pump(queue: asyncio.Queue, websocket: WebSocket) -> None:
    """Single writer per socket: drain the queue in order."""
    while True:
        text = await queue.get()
        if text is None:
            # Overflow sentinel (stalled client): close; the reconnect replay repaints.
            await websocket.close(code=1013, reason="output backlog dropped; reconnect")
            return
        await websocket.send_text(text)


@router.websocket("/ws/{session_id}")
async def terminal_ws(websocket: WebSocket, session_id: str) -> None:
    """Raw terminal bridge. Origin-checked like every other WS route."""
    session = _SESSIONS.get(session_id)
    if session is None:
        await websocket.close(code=1008, reason="unknown session")
        return
    await websocket.accept()

    sub_id, queue, replay = session.attach()
    pump_task: Optional[asyncio.Task] = None
    try:
        if replay:
            await websocket.send_text(replay)
        pump_task = asyncio.create_task(_pump(queue, websocket))
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            mtype = msg.get("type") if isinstance(msg, dict) else None
            if mtype == "input":
                try:
                    input_frame = TerminalInputFrame.model_validate(msg)
                except ValidationError:
                    continue
                if input_frame.data:
                    session.write(input_frame.data)
            elif mtype == "resize":
                try:
                    resize_frame = TerminalResizeFrame.model_validate(msg)
                except ValidationError:
                    continue
                try:
                    session.resize(_clamp(resize_frame.rows), _clamp(resize_frame.cols))
                except (TypeError, ValueError):
                    pass
    except WebSocketDisconnect:
        pass
    finally:
        session.detach(sub_id)
        if pump_task is not None:
            pump_task.cancel()
            # Retrieve the task's outcome so a send that failed at the same moment
            # doesn't log "Task exception was never retrieved". CancelledError must be
            # listed explicitly (BaseException) or it propagates and cancels this
            # endpoint's own task.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await pump_task
