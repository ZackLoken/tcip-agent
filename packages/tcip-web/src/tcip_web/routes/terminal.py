"""Agent terminal routes: provider status, session launch, restart and submitted requests over
HTTP, and per session a WebSocket carrying raw PTY output as text frames out and
``TerminalInputFrame``/``TerminalResizeFrame`` JSON messages in. A reconnecting socket receives
the scrollback first, then live output, with nothing lost or repeated between them.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError

from tcip_mcp.identity import actor
from tcip_web import terminal as pty_host
from tcip_web.state import OpenProject, store

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

_EXIT_NOTE = "\r\n\x1b[2m[the agent exited, use Restart in the rail header]\x1b[22m\r\n"


def _offer(queue: asyncio.Queue, data: str) -> None:
    """Enqueue output for one subscriber (runs on that subscriber's loop).

    On overflow, drop the backlog and leave a ``None`` sentinel: the pump closes the socket, and
    the reconnect replay repaints.
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


class LaunchedProgram(BaseModel):
    """What a launch ran: the provider row's id, the rendered argv, the version its executable
    (the first element) declares, what the row states its harness's own enforcement restricts
    and leaves open (``None`` for a ``TCIP_TERMINAL_CMD`` override, which runs no row's
    arguments), why delivery to it rests on a
    composer sequence recorded on another version (``None`` when the versions agree, and for an
    override), and the steps the row's preparation took before the launch, empty when it has
    none."""

    provider: str
    argv: list[str]
    version: Optional[str]
    confinement: Optional[str]
    delivery_unverified: Optional[str]
    prepared: list[str]


class TerminalLaunch(BaseModel):
    """A started session: its id, whether it was already running when asked for, what it
    launched, and the session-start ritual that launch was given."""

    session_id: str
    existing: bool
    launched: LaunchedProgram
    ritual: str


class ProviderStatus(BaseModel):
    """One provider row and why it cannot launch here, ``None`` when it can."""

    id: str
    name: str
    unavailable_reason: Optional[str]


class TerminalStatus(BaseModel):
    """Every provider row, in table order."""

    providers: list[ProviderStatus]


def _record_start(session_id: str, launched: LaunchedProgram, project: Optional[OpenProject],
                  actor: str) -> None:
    """One audit line by ``actor`` in ``project``'s log per launch for it, naming the session id
    and the provider and program it launched; a launch for no project has no log to land in and
    records nothing. A failed append raises ``AuditEntryNotWrittenError``.
    """
    from tcip_mcp.audit import record_event_or_raise

    if project is not None:
        record_event_or_raise("agent_terminal_started",
                              {"session_id": session_id, **launched.model_dump()}, actor=actor,
                              scope=project.root)


@dataclass
class _Launch:
    """One launch of a session's agent: its generation (a stale reader's output carries an older
    one and is dropped), its session-start ritual once known, whether that ritual was delivered,
    whether the agent has bracketed paste on, the sequence one version of its row's harness was
    recorded writing when its composer appeared and whether the agent has written it, the
    requests waiting behind the ritual, and the output tail an escape sequence split across reads
    continues from."""

    gen: int
    ritual: Optional[str] = None
    ritual_sent: bool = False
    ready: bool = False
    composer_ready: Optional[str] = None
    composer_live: bool = False
    queued: list[str] = field(default_factory=list)
    tail: str = ""


_TAIL_CHARS = 32


class TerminalSession:
    """One PTY-attached agent process, its current launch and its subscriber queues."""

    launch: TerminalLaunch
    """What the last start launched; set by :meth:`start`."""

    def __init__(self, session_id: str):
        self.id = session_id
        self._pty = None
        self._lock = threading.Lock()
        self._scrollback: list[str] = []
        self._scrollback_len = 0
        # ws-id → (queue, that websocket's event loop)
        self._subs: dict[int, tuple[asyncio.Queue, asyncio.AbstractEventLoop]] = {}
        self._next_sub = 0
        self._launch = _Launch(gen=0)

    # ── lifecycle ───────────────────────────────────────────────────────

    def start(self, rows: int, cols: int, provider: pty_host.Provider, actor: str) -> Optional[str]:
        """Spawn ``provider``'s harness in a PTY for the open project as the current launch by
        ``actor``, or a new one when the current launch already spawned, its ritual built before
        the spawn from the project's record as it reads now. Returns an error reason (a record
        that will not read names why), or None on success."""
        with self._lock:
            if self._pty is not None and self._pty.isalive():
                return None
            if self._launch.ritual is not None:
                self._launch = _Launch(gen=self._launch.gen + 1)
            project = store.opened
            launch = pty_host.project_launch(project)
            try:
                ritual = pty_host.session_ritual(launch)
            except ValueError as exc:
                return str(exc)
            command = pty_host.resolve_terminal_command(provider)
            if command is None:
                return provider.unavailable_reason
            prepared: list[str] = []
            if not command[1]:
                prepared, problem = pty_host.prepare_launch(command[0][0], provider, launch)
                if problem is not None:
                    return problem
            env = pty_host.spawn_env(self.id)
            argv = pty_host.render_argv(command[0], launch, env)
            program = pty_host.launched_program(argv, command[1])
            launched = LaunchedProgram(
                provider=provider.id, prepared=prepared,
                confinement=None if command[1] else provider.confinement,
                delivery_unverified=(None if command[1]
                                     else provider.delivery_unverified(program["version"])),
                **program)
            self._launch.composer_ready = provider.composer_ready
            try:
                pty = pty_host.spawn_pty(argv, pty_host.terminal_cwd(), rows, cols, env)
            except OSError as exc:
                self._pty = None
                return f"could not start the agent terminal: {exc}"
            self._pty = pty
            self._launch.ritual = ritual
            self.launch = TerminalLaunch(session_id=self.id, existing=False, launched=launched,
                                         ritual=self._launch.ritual)
            gen = self._launch.gen
        pty_host.start_reader(
            pty,
            lambda data: self._on_output(data, gen),
            lambda: self._on_exit(gen),
            name=f"term-{self.id}-g{gen}",
        )
        from tcip_mcp.audit import AuditEntryNotWrittenError

        try:
            _record_start(self.id, launched, project, actor)
        except AuditEntryNotWrittenError as exc:
            stopped = self.terminate()
            reason = str(exc)
            if not stopped:
                reason += (
                    " The spawned process could not be stopped and stays attached to this "
                    "session."
                )
            return reason
        return None

    def restart(self, rows: int, cols: int, provider: pty_host.Provider,
                actor: str) -> Optional[str]:
        """Open the next launch, holding the current launch's undelivered requests and every
        request submitted from here on behind its ritual, end the current process and start
        ``provider`` as that launch by ``actor``. Returns an error reason, or None on success."""
        with self._lock:
            self._launch = _Launch(gen=self._launch.gen + 1, queued=self._launch.queued)
        # A survivor here stays attached: start() below would find self._pty alive and report
        # success with no process spawned and no line written.
        if not self.terminate():
            return (
                "the previous agent process could not be stopped and stays attached to this "
                "session."
            )
        with self._lock:
            self._scrollback = []
            self._scrollback_len = 0
        return self.start(rows, cols, provider, actor)

    def submit(self, text: str) -> None:
        """Queue ``text`` for the current launch's agent, delivered after its ritual."""
        with self._lock:
            self._launch.queued.append(text)
            self._deliver()

    def terminate(self) -> bool:
        """Kill the PTY. Returns True when no process remains afterward, False when it survives:
        ``self._pty`` is then restored so ``alive()``/I/O still see it.
        """
        with self._lock:
            pty, self._pty = self._pty, None
        if pty is None:
            return True
        try:
            pty.terminate()
        except Exception:  # pragma: no cover - best-effort cleanup
            logger.debug("terminal terminate failed", exc_info=True)
        try:
            survived = pty.isalive()
        except Exception:  # pragma: no cover - best-effort cleanup
            survived = False
        if survived:
            with self._lock:
                self._pty = pty
            return False
        return True

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

    def _deliver(self) -> None:
        """Paste the current launch's ritual together with every request queued by then as one
        message, then each later request on its own, once its agent has bracketed paste on and
        has written its row's ``composer_ready``; each leaves the launch only once written. Called
        under the lock."""
        launch = self._launch
        if not (launch.ready and launch.composer_live) or launch.ritual is None:
            return
        if not launch.ritual_sent:
            if not self.write(pty_host.paste("\n\n".join([launch.ritual, *launch.queued]))):
                return
            launch.ritual_sent = True
            launch.queued.clear()
        while launch.queued and self.write(pty_host.paste(launch.queued[0])):
            launch.queued.pop(0)

    def _on_output(self, data: str, gen: Optional[int] = None) -> None:
        with self._lock:
            launch = self._launch
            if gen is not None and gen != launch.gen:
                return  # stale reader from a restarted PTY: drop, don't pollute
            seen = launch.tail + data
            launch.ready = pty_host.bracketed_paste(seen, launch.ready)
            launch.composer_live = launch.composer_live or (
                launch.composer_ready is not None and launch.composer_ready in seen)
            launch.tail = seen[-_TAIL_CHARS:]
            self._deliver()
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
        """The current scrollback text."""
        with self._lock:
            return "".join(self._scrollback)


_SESSIONS: dict[str, TerminalSession] = {}
# Serializes attach-or-spawn and restart across the request threadpool: without it,
# two concurrent POSTs each spawn an agent process and one runs orphaned.
_SESSIONS_LOCK = threading.Lock()


def shutdown_all() -> None:
    """Kill every live agent terminal and forget the sessions."""
    for s in list(_SESSIONS.values()):
        if not s.terminate():
            logger.warning("terminal session %s survived shutdown termination", s.id)
    _SESSIONS.clear()


# ── HTTP surface ────────────────────────────────────────────────────────


@router.get("/status")
def get_status() -> TerminalStatus:
    """Every provider row with the reason it cannot launch here."""
    return TerminalStatus(providers=[
        ProviderStatus(id=row.id, name=row.name, unavailable_reason=pty_host.launch_problem(row))
        for row in pty_host.PROVIDERS
    ])


class TerminalInputFrame(BaseModel):
    """Keystrokes typed into the rail, forwarded to the PTY verbatim."""

    type: Literal["input"]
    data: str


class TerminalResizeFrame(BaseModel):
    """Terminal dimensions, applied to the PTY's window size."""

    type: Literal["resize"]
    rows: int
    cols: int


class CreateSessionRequest(BaseModel):
    """A launch: the id of the :data:`~tcip_web.terminal.PROVIDERS` row to run, the terminal
    dimensions and the person launching it."""

    provider: str
    rows: int = pty_host.DEFAULT_ROWS
    cols: int = pty_host.DEFAULT_COLS
    user: str


def _clamp(v: int) -> int:
    return max(2, min(MAX_DIM, v))


def _provider(provider_id: str) -> pty_host.Provider:
    """The table row whose id is exactly ``provider_id``; refuses with 422 naming the ids the
    table lists."""
    for row in pty_host.PROVIDERS:
        if row.id == provider_id:
            return row
    raise HTTPException(422, f"no agent provider {provider_id!r}; the provider table lists "
                             f"{[row.id for row in pty_host.PROVIDERS]}")


@router.post("/sessions")
def create_session(req: CreateSessionRequest) -> TerminalLaunch:
    """Return the live session (attach semantics, like tmux) or spawn a fresh one running the
    requested provider by the person ``req.user`` names."""
    person = actor(req.user)
    provider = _provider(req.provider)
    with _SESSIONS_LOCK:
        for s in _SESSIONS.values():
            if s.alive():
                return s.launch.model_copy(update={"existing": True})
        session_id = "term_" + os.urandom(6).hex()
        session = TerminalSession(session_id)
        err = session.start(_clamp(req.rows), _clamp(req.cols), provider, person)
        if err:
            # A survivor of the failed start's own termination attempt stays reachable, so a
            # retry attaches to it instead of spawning a second process beside it.
            if session.alive():
                _SESSIONS[session_id] = session
            raise HTTPException(503, err)
        _SESSIONS[session_id] = session
        return session.launch


def _require(session_id: str) -> TerminalSession:
    session = _SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, f"no such terminal session: {session_id}")
    return session


@router.post("/sessions/{session_id}/restart")
def restart_session(session_id: str, req: CreateSessionRequest) -> TerminalLaunch:
    """End the session's process and launch the requested provider in its place, by the person
    ``req.user`` names."""
    person = actor(req.user)
    session = _require(session_id)
    provider = _provider(req.provider)
    with _SESSIONS_LOCK:
        # One live agent at a time: restarting a stale session while a different one is
        # live would silently run two agent processes.
        for other in _SESSIONS.values():
            if other.id != session_id and other.alive():
                raise HTTPException(
                    409, f"another agent session is live ({other.id}); attach to it instead"
                )
        err = session.restart(_clamp(req.rows), _clamp(req.cols), provider, person)
    if err:
        raise HTTPException(503, err)
    return session.launch


class SubmitRequest(BaseModel):
    """A request for the agent: text it receives as one pasted message."""

    text: str


@router.post("/sessions/{session_id}/submit")
def submit_to_session(session_id: str, req: SubmitRequest) -> dict:
    """Queue ``req.text`` for the session's current launch (see :meth:`TerminalSession.submit`);
    404 for an unknown session."""
    _require(session_id).submit(req.text)
    return {}


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
    """Raw terminal bridge: PTY output out as text frames, input and resize frames in."""
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
