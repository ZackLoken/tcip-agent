"""Agent terminal routes: provider status, session launch and end, requests and a PTY WebSocket.

Status, launch, end and submitted requests go over HTTP; per session a WebSocket carries
raw PTY output as text frames out and ``TerminalInputFrame``/``TerminalResizeFrame`` JSON
messages in. A reconnecting socket receives the scrollback first, then live output, with nothing
lost or repeated between them.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import os
import threading
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from tcip_mcp.audit import AuditEntryNotWrittenError, record_event_or_raise
from tcip_mcp.identity import actor
from tcip_web import terminal as pty_host
from tcip_web.routes.audit_gap import audit_gap_409
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

# RFC 6455 leaves the close codes 4000-4999 to the application.
ENDED_CLOSE_CODE = 4000
"""The close code of a socket whose session's agent ended; a reconnect cannot reach it again."""

_ENDED_CLOSE = (ENDED_CLOSE_CODE, "the agent ended")
_BACKLOG_CLOSE = (1013, "output backlog dropped; reconnect")
_SURVIVED = "The agent process could not be stopped and stays attached to this session."


def _offer(queue: asyncio.Queue, item: str | tuple[int, str]) -> None:
    """Enqueue output, or a ``(code, reason)`` close, for one subscriber (on its loop).

    On overflow, drop the backlog and leave the close ``item`` is, or else the backlog close.
    """
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:
        try:
            while True:
                queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        queue.put_nowait(item if isinstance(item, tuple) else _BACKLOG_CLOSE)


class LaunchedProgram(BaseModel):
    """What a launch ran: its provider row, argv, version, confinement and preparation.

    It carries the provider row's id, the rendered argv, the version its executable (the first
    element) declares, what the row states its harness's own enforcement restricts and leaves
    open (``None`` for a ``TCIP_TERMINAL_CMD`` override, which runs no row's arguments), and
    the steps the row's preparation took before the launch, empty when it has none.
    """

    provider: str
    argv: list[str]
    version: str | None
    confinement: str | None
    prepared: list[str]


class TerminalLaunch(BaseModel):
    """A started session: its id, what it launched and the session-start ritual it was given."""

    session_id: str
    launched: LaunchedProgram
    ritual: str


class ProviderStatus(BaseModel):
    """One provider row and why it cannot launch here, ``None`` when it can."""

    id: str
    name: str
    unavailable_reason: str | None


class TerminalStatus(BaseModel):
    """Every provider row, in table order."""

    providers: list[ProviderStatus]


class TerminalEnd(BaseModel):
    """An ended session: the requests it took and its agent was never given, oldest first."""

    undelivered: list[str]


def _record(
    event: str,
    arguments: dict,
    project: OpenProject | None,
    actor: str | None,
) -> None:
    """Write one ``event`` line with ``arguments`` by ``actor`` in ``project``'s log.

    ``actor`` is ``None`` for an act no person performed. A session for no project has no log to
    land in and records nothing. A failed append raises ``AuditEntryNotWrittenError``.
    """
    if project is not None:
        record_event_or_raise(
            event, arguments, actor=actor, scope=project.root
        )


_TAIL_CHARS = 32


class TerminalSession:
    """One PTY-attached agent process, its delivery state and its subscriber queues.

    The delivery state is the agent's session-start ritual and whether it was delivered, whether
    the agent has bracketed paste on, whether it has written its row's ``composer_ready``, the
    requests waiting behind the ritual, and the output tail an escape sequence split across reads
    continues from.
    """

    launched: LaunchedProgram
    """What :meth:`start` launched."""
    ritual: str
    """The session-start ritual :meth:`start` built for the agent."""
    _pty: pty_host.Pty | None

    def __init__(self, session_id: str) -> None:
        """Take the session's id; no process runs until :meth:`start`."""
        self.id = session_id
        self._pty = None
        # The project whose log records the session's lines, set once its start line is written.
        self._project: OpenProject | None = None
        self._lock = threading.Lock()
        self._scrollback: list[str] = []
        self._scrollback_len = 0
        # ws-id → (queue, that websocket's event loop)
        self._subs: dict[
            int, tuple[asyncio.Queue, asyncio.AbstractEventLoop]
        ] = {}
        self._sub_ids = itertools.count()
        self._ritual_sent = False
        self._ready = False
        self._composer_live = False
        self._queued: list[str] = []
        self._tail = ""
        # Held by the one delivery writing to the PTY, so writes keep their order; taken after the
        # sessions lock and before the session lock, which is never held across a write.
        self._writing = threading.Lock()

    # ── lifecycle ───────────────────────────────────────────────────────

    def start(
        self, rows: int, cols: int, provider: pty_host.Provider, actor: str
    ) -> str | None:
        """Spawn ``provider``'s harness in a PTY for the open project, by ``actor``.

        The ritual is built before the spawn from the project's record as it reads now. Returns
        an error reason (a record that will not read names why), or None on success.
        """
        project = store.opened
        launch = pty_host.project_launch(project)
        try:
            ritual = pty_host.session_ritual(launch)
        except ValueError as exc:
            return str(exc)
        command = pty_host.resolve_terminal_command(provider)
        if command is None:
            return provider.unavailable_reason
        prepared = (
            []
            if command[1]
            else pty_host.prepare_launch(command[0][0], provider, launch)
        )
        if isinstance(prepared, str):
            return prepared
        env = pty_host.spawn_env(self.id)
        argv = pty_host.render_argv(command[0], launch, env)
        program = pty_host.launched_program(argv, command[1])
        launched = LaunchedProgram(
            provider=provider.id,
            prepared=prepared,
            confinement=None if command[1] else provider.confinement,
            **program,
        )
        try:
            pty = pty_host.spawn_pty(
                argv, str(store.workspace), rows, cols, env
            )
        except OSError as exc:
            return f"could not start the agent terminal: {exc}"
        self._pty = pty
        self.launched = launched
        self.ritual = ritual
        try:
            _record(
                "agent_terminal_started",
                {"session_id": self.id, **launched.model_dump()},
                project,
                actor,
            )
        except AuditEntryNotWrittenError as exc:
            if self.terminate():
                pty.close()
                return str(exc)
            return f"{exc} {_SURVIVED}"
        self._project = project
        # Started only once the start line is written, so the process's own end follows it.
        pty_host.start_reader(
            pty,
            self._on_output,
            lambda: self.end_unattended("the agent's process exited"),
            name=f"term-{self.id}",
        )
        return None

    def end(self, cause: str, actor: str | None) -> bool:
        """End the session's agent for ``cause``, by ``actor`` (``None`` when no person did).

        Under the sessions lock, so no create runs and no submit is admitted meanwhile, and in
        this order: terminates the process; waits for a delivery under way to settle, so each
        request is either written and gone from the queue or still queued, the queue it leaves
        being what its agent was never given, which nothing changes after; writes one
        ``agent_terminal_ended`` line naming the session, its provider row, ``cause`` and how many
        requests that is in the log its start line was written to, or writes the failure to the
        terminal stream and the error log; then, under the session's own lock, closes every socket
        of it with :data:`ENDED_CLOSE_CODE`, drops them, so no later output reaches one, and
        forgets the session, which is what marks it ended. Returns False when the process
        survives termination, the session still held, and True once the session ended, by this
        call or an earlier one. A failed append is raised as ``AuditEntryNotWrittenError``.
        """
        with _SESSIONS_LOCK:
            if self.id not in _SESSIONS:
                return True
            if not self.terminate():
                return False
            with self._writing, self._lock:
                undelivered_count = len(self._queued)
            try:
                _record(
                    "agent_terminal_ended",
                    {
                        "session_id": self.id,
                        "provider": self.launched.provider,
                        "cause": cause,
                        "undelivered_count": undelivered_count,
                    },
                    self._project,
                    actor,
                )
            except AuditEntryNotWrittenError as exc:
                with self._lock:
                    self._show(f"\r\n{exc}\r\n")
                logger.exception(
                    "terminal session %s ended unrecorded", self.id
                )
                raise
            finally:
                with self._lock:
                    self._broadcast(_ENDED_CLOSE)
                    self._subs.clear()
                    del _SESSIONS[self.id]
        return True

    def answer(self) -> TerminalLaunch:
        """Return the session's id, what it launched and the ritual it was given."""
        return TerminalLaunch(
            session_id=self.id, launched=self.launched, ritual=self.ritual
        )

    def submit(self, text: str) -> None:
        """Queue ``text`` for the session's agent, behind its ritual; :meth:`deliver` writes it."""
        with self._lock:
            self._queued.append(text)

    def deliver(self) -> None:
        """Paste the agent's ritual with its queued requests, then each later request.

        The ritual goes together with every request queued by then as one message, then each
        later request on its own, once its agent has bracketed paste on and has written its row's
        ``composer_ready``; each leaves the queue only once written. One delivery writes at a
        time; the session lock is held to read and update the queue, never across a write.
        """
        with self._writing:
            while True:
                with self._lock:
                    if not (self._ready and self._composer_live):
                        return
                    if not self._ritual_sent:
                        text = "\n\n".join([self.ritual, *self._queued])
                        taken = len(self._queued)
                    elif self._queued:
                        text, taken = self._queued[0], 1
                    else:
                        return
                if not self.write(pty_host.paste(text)):
                    return
                with self._lock:
                    self._ritual_sent = True
                    del self._queued[:taken]

    def terminate(self) -> bool:
        """Kill the PTY and return True when no process remains afterward.

        Returns False when it survives: ``self._pty`` is then restored.
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
        """Return whether the session's PTY process is running."""
        pty = self._pty
        return bool(pty is not None and pty.isalive())

    # ── I/O ─────────────────────────────────────────────────────────────

    def write(self, data: str) -> bool:
        """Write ``data`` to the PTY and return whether it was sent."""
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
        """Set the PTY's window size; a failed resize or no PTY is a no-op."""
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

    def _on_output(self, data: str) -> None:
        """Take the agent's output: note its paste readiness, show it, then :meth:`deliver`."""
        with self._lock:
            seen = self._tail + data
            self._ready = pty_host.bracketed_paste(seen, self._ready)
            self._composer_live = (
                self._composer_live
                or _provider(self.launched.provider).composer_ready in seen
            )
            self._tail = seen[-_TAIL_CHARS:]
            self._show(data)
        self.deliver()

    def _show(self, data: str) -> None:
        """Keep ``data`` in the scrollback and offer it to every subscriber; under the lock."""
        self._scrollback.append(data)
        self._scrollback_len += len(data)
        while (
            self._scrollback_len > SCROLLBACK_MAX_CHARS
            and len(self._scrollback) > 1
        ):
            dropped = self._scrollback.pop(0)
            self._scrollback_len -= len(dropped)
        self._broadcast(data)

    def _broadcast(self, item: str | tuple[int, str]) -> None:
        """Offer ``item`` to every subscriber, dropping one whose loop is gone; under the lock."""
        dead: list[int] = []
        for sub_id, (queue, loop) in self._subs.items():
            if loop.is_closed():
                dead.append(sub_id)
                continue
            try:
                # FIFO across calls → the pump drains in byte order.
                loop.call_soon_threadsafe(_offer, queue, item)
            except RuntimeError:
                dead.append(sub_id)
        for sub_id in dead:
            self._subs.pop(sub_id, None)

    def end_unattended(self, cause: str) -> bool:
        """:meth:`end` the session for ``cause``, an act no person performed, where no caller waits.

        A failed append, which :meth:`end` has logged, is not raised. Returns what :meth:`end`
        returns.
        """
        with contextlib.suppress(AuditEntryNotWrittenError):
            return self.end(cause, None)
        return True

    # ── subscriber registration (called from each websocket's loop) ─────

    def attach(self) -> tuple[int, asyncio.Queue, str]:
        """Register a subscriber; returns ``(sub_id, queue, replay_snapshot)``.

        Snapshot + registration are atomic w.r.t. the writer, so output can be neither
        lost (arrived after snapshot, before registration) nor duplicated. Once the session
        ended (the backend no longer holds it), the queue holds the ended close and nothing is
        registered.
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX_CHUNKS)
        loop = asyncio.get_running_loop()
        with self._lock:
            replay = "".join(self._scrollback)
            sub_id = next(self._sub_ids)
            if self.id not in _SESSIONS:
                queue.put_nowait(_ENDED_CLOSE)
            else:
                self._subs[sub_id] = (queue, loop)
        return sub_id, queue, replay

    def detach(self, sub_id: int) -> None:
        """Unregister the subscriber ``sub_id``."""
        with self._lock:
            self._subs.pop(sub_id, None)

    def scrollback_snapshot(self) -> str:
        """Return the current scrollback text."""
        with self._lock:
            return "".join(self._scrollback)


_SESSIONS: dict[str, TerminalSession] = {}
# Serializes a create, a submit's admission and a whole end across the request threadpool and
# the reader threads: without it, a create beside an end could run two agent processes.
_SESSIONS_LOCK = threading.Lock()


def shutdown_all() -> None:
    """End every agent terminal for the backend's shutdown.

    A survivor of termination and an end whose line fails to append are each logged.
    """
    for s in list(_SESSIONS.values()):
        if not s.end_unattended("the backend shut down"):
            logger.warning(
                "terminal session %s survived shutdown termination", s.id
            )


# ── HTTP surface ────────────────────────────────────────────────────────


@router.get("/status")
def get_status() -> TerminalStatus:
    """Return every provider row with the reason it cannot launch here."""
    return TerminalStatus(
        providers=[
            ProviderStatus(
                id=row.id,
                name=row.name,
                unavailable_reason=pty_host.launch_problem(row),
            )
            for row in pty_host.PROVIDERS
        ]
    )


class TerminalInputFrame(BaseModel):
    """Keystrokes typed into the rail, forwarded to the PTY verbatim."""

    type: Literal["input"]
    data: str


class TerminalResizeFrame(BaseModel):
    """Terminal dimensions, applied to the PTY's window size."""

    type: Literal["resize"]
    rows: int
    cols: int


TerminalFrame = Annotated[
    TerminalInputFrame | TerminalResizeFrame, Field(discriminator="type")
]
"""A frame the rail sends over a session's socket, told apart by its ``type``."""

_FRAME: TypeAdapter[TerminalFrame] = TypeAdapter(TerminalFrame)


class CreateSessionRequest(BaseModel):
    """A launch: the provider row to run, the terminal dimensions and the person launching it.

    The row is named by its id in :data:`~tcip_web.terminal.PROVIDERS`.
    """

    provider: str
    rows: int = pty_host.DEFAULT_ROWS
    cols: int = pty_host.DEFAULT_COLS
    user: str


def _clamp(v: int) -> int:
    return max(2, min(MAX_DIM, v))


def _provider(provider_id: str) -> pty_host.Provider:
    """Return the table row whose id is exactly ``provider_id``.

    Refuses with 422 naming the ids the table lists.
    """
    for row in pty_host.PROVIDERS:
        if row.id == provider_id:
            return row
    raise HTTPException(
        422,
        f"no agent provider {provider_id!r}; the provider table lists "
        f"{[row.id for row in pty_host.PROVIDERS]}",
    )


@router.post("/sessions")
def create_session(req: CreateSessionRequest) -> TerminalLaunch:
    """Return the live session (attach semantics, like tmux) or spawn a fresh one.

    A fresh one runs the requested provider by the person ``req.user`` names.
    """
    person = actor(req.user)
    provider = _provider(req.provider)
    with _SESSIONS_LOCK:
        for s in _SESSIONS.values():
            if s.alive():
                return s.answer()
        session_id = "term_" + os.urandom(6).hex()
        session = TerminalSession(session_id)
        err = session.start(
            _clamp(req.rows), _clamp(req.cols), provider, person
        )
        if err:
            # A survivor of the failed start's own termination attempt stays reachable, so a
            # retry attaches to it instead of spawning a second process beside it.
            if session.alive():
                _SESSIONS[session_id] = session
            raise HTTPException(503, err)
        _SESSIONS[session_id] = session
        return session.answer()


def _require(session_id: str) -> TerminalSession:
    session = _SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, f"no such terminal session: {session_id}")
    return session


class EndSessionRequest(BaseModel):
    """An end of a session's agent and its conversation, by the person ``user`` names."""

    user: str


@router.post("/sessions/{session_id}/end")
def end_session(session_id: str, req: EndSessionRequest) -> TerminalEnd:
    """End the session's agent by the person ``req.user`` names (:meth:`TerminalSession.end`).

    Answers the requests its agent was never given. 404 for an unknown session; 503 when the
    process survives termination, the session kept reachable; the audit-gap 409 carrying that
    answer when the end's line fails to append.
    """
    person = actor(req.user)
    session = _require(session_id)
    unrecorded: AuditEntryNotWrittenError | None = None
    try:
        stopped = session.end("the person ended it", person)
    except AuditEntryNotWrittenError as exc:
        stopped, unrecorded = True, exc
    if not stopped:
        raise HTTPException(503, _SURVIVED)
    answer = TerminalEnd(undelivered=list(session._queued))
    if unrecorded is not None:
        raise audit_gap_409(unrecorded, answer) from unrecorded
    return answer


class SubmitRequest(BaseModel):
    """A request for the agent: text it receives as one pasted message."""

    text: str


@router.post("/sessions/{session_id}/submit")
def submit_to_session(session_id: str, req: SubmitRequest) -> dict:
    """Queue ``req.text`` for the session's agent, delivered after its ritual.

    404 for a session the backend does not hold, ended or never started. The request is queued
    under the sessions lock, so an end after it answers it among the undelivered, and written
    once that lock is released.
    """
    with _SESSIONS_LOCK:
        session = _require(session_id)
        session.submit(req.text)
    session.deliver()
    return {}


async def _pump(queue: asyncio.Queue, websocket: WebSocket) -> None:
    """Drain the queue in order as the socket's single writer."""
    while True:
        item = await queue.get()
        if isinstance(item, tuple):
            code, reason = item
            await websocket.close(code=code, reason=reason)
            return
        await websocket.send_text(item)


@router.websocket("/ws/{session_id}")
async def terminal_ws(websocket: WebSocket, session_id: str) -> None:
    """Bridge PTY output to text frames and accept input and resize frames.

    A session the backend does not hold, ended or never started, is closed with
    :data:`ENDED_CLOSE_CODE` once accepted, so a reconnecting client learns its agent ended.
    """
    await websocket.accept()
    session = _SESSIONS.get(session_id)
    if session is None:
        await websocket.close(*_ENDED_CLOSE)
        return

    sub_id, queue, replay = session.attach()
    pump_task: asyncio.Task | None = None
    try:
        if replay:
            await websocket.send_text(replay)
        pump_task = asyncio.create_task(_pump(queue, websocket))
        while True:
            try:
                frame = _FRAME.validate_json(await websocket.receive_text())
            except ValidationError:
                continue
            if isinstance(frame, TerminalResizeFrame):
                session.resize(_clamp(frame.rows), _clamp(frame.cols))
            elif frame.data:
                session.write(frame.data)
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
