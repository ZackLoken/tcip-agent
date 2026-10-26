"""Agent terminal routes: provider status, session launch, hand-off and end, requests and a socket.

Status, launch, the session start a session's hook reports and the orientation answered for it,
a hand-off before a switch of harness, the end and submitted requests go over HTTP; per session a
WebSocket carries raw PTY
output as text frames out and ``TerminalInputFrame``/``TerminalResizeFrame`` JSON messages in. A
reconnecting socket receives the scrollback first, then live output, with nothing lost or
repeated between them. A launch resumes its row's last conversation in the open project when
that conversation's end left it resumable within the row's prompt-cache lifetime, and starts
fresh otherwise.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import itertools
import logging
import os
import threading
from collections.abc import Callable
from datetime import datetime
from queue import Queue, ShutDown
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import (
    AwareDatetime,
    BaseModel,
    Field,
    TypeAdapter,
    ValidationError,
)

from tcip_mcp.audit import (
    AuditEntryNotWrittenError,
    acts_of,
    audit_entries,
    now,
    record_event_or_raise,
)
from tcip_mcp.identity import actor
from tcip_web import terminal as pty_host
from tcip_web.routes.audit_gap import audit_gap_409
from tcip_web.state import OpenProject, store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/terminal", tags=["terminal"])

# The scrollback keeps the newest this many characters: enough to re-render a long session's
# tail without unbounded memory; the TUI repaints itself on the next output anyway.
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
    """What a launch ran: its provider row, argv, version, confinement, preparation and resume.

    It carries the provider row's id, the rendered argv, the version its executable (the first
    element) declares, what the row states its harness's own enforcement restricts and leaves
    open (``None`` for a ``TCIP_TERMINAL_CMD`` override, which runs no row's arguments), the
    steps the row's preparation took before the launch, empty when it has none, and the
    conversation it resumed, ``None`` for a fresh one.
    """

    provider: str
    argv: list[str]
    version: str | None
    confinement: str | None
    prepared: list[str]
    resumed: str | None


class TerminalLaunch(BaseModel):
    """A started session: its id and what it launched."""

    session_id: str
    launched: LaunchedProgram


class ProviderStatus(BaseModel):
    """One provider row and why it cannot launch here, ``None`` when it can."""

    id: str
    name: str
    unavailable_reason: str | None


class TerminalStatus(BaseModel):
    """Every provider row, in table order."""

    providers: list[ProviderStatus]


class TerminalEnd(BaseModel):
    """An ended session's answer: the requests its agent was never given, oldest first."""

    undelivered: list[str]


_STARTED = "agent_terminal_started"
_ENDED = "agent_terminal_ended"


class TerminalEnded(TerminalEnd):
    """What an ``agent_terminal_ended`` line records about the session it ends.

    Beside the :class:`TerminalEnd` an end answers, it holds the session, its provider row, why
    it ended in words for the person, whether its conversation may be resumed (not after the
    person ended it with no switch under way), the conversation its harness last reported,
    ``None`` when none reported one, the time of the last output the session saw from its agent,
    and, for an end during a switch of harness, whether the agent's hand-off had landed,
    ``None`` for any other end.
    """

    session_id: str
    provider: str
    cause: str
    resumable: bool
    conversation: str | None
    last_output_at: AwareDatetime | None
    handed_off: bool | None


class SessionStartReport(BaseModel):
    """A session start as its hook reported it: the harness's conversation and how it started.

    The conversation is the one the harness declared it is in.
    """

    conversation: str = Field(min_length=1)
    source: pty_host.SessionStartSource


class SessionOrientation(BaseModel):
    """What the agent of a session start is told, ``None`` for nothing."""

    text: str | None


HAND_OFF_REQUEST = (
    "[TCIP hand-off] The breeder is switching this terminal to another agent harness. Write "
    "your hand-off now with write_retrospective, as the self-improvement skill's hand-off "
    "describes; this session ends once it is recorded."
)
"""The request a switch of harness delivers to the agent it ends."""


UNRECORDED_END = (AuditEntryNotWrittenError, ValueError)
"""What leaves an end with no end line: the line failing to append, or the session's log failing
to read for its hand-off."""


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


@dataclasses.dataclass(frozen=True)
class _HandOff:
    """A switch of harness under way: who asked for it, and the log's cursor when they did."""

    by: str
    after: str


class TerminalSession:
    """One PTY-attached agent process, its delivery state, its input writer and its subscribers.

    The delivery state is the session's own text still to paste ahead of the queued requests
    (a hand-off request once a switch asks for one), whether the agent has bracketed paste on,
    whether it has written its row's ``composer_ready``, the requests a person submitted that
    are still queued, whether a paste of them is with the writer, and the output tail an escape
    sequence split across reads continues from. Every input reaches the PTY through the one
    writer (:meth:`write`).
    """

    launched: LaunchedProgram
    """What :meth:`start` launched."""
    conversation: str | None
    """The conversation the agent's harness last reported it is in, ``None`` until one does."""
    ended: TerminalEnded
    """The session's one end, set by :meth:`end` once it has stopped the process and read the
    hand-off."""
    _pty: pty_host.Pty | None

    def __init__(self, session_id: str) -> None:
        """Take the session's id; no process runs until :meth:`start`."""
        self.id = session_id
        self._pty = None
        self.conversation = None
        # The project whose log records the session's lines, set once its start line is written.
        self._project: OpenProject | None = None
        self._lock = threading.Lock()
        self._scrollback = ""
        self._inputs: Queue[pty_host.Input] = Queue()
        # ws-id → (queue, that websocket's event loop)
        self._subs: dict[
            int, tuple[asyncio.Queue, asyncio.AbstractEventLoop]
        ] = {}
        self._sub_ids = itertools.count()
        self._ahead: str | None = None
        self._ready = False
        self._composer_live = False
        self._queued: list[str] = []
        self._pasting = False
        self._tail = ""
        self._last_output_at: datetime | None = None
        self._hand_off: _HandOff | None = None

    # ── lifecycle ───────────────────────────────────────────────────────

    def start(
        self, rows: int, cols: int, provider: pty_host.Provider, actor: str
    ) -> str | None:
        """Spawn ``provider``'s harness in a PTY for the open project, by ``actor``.

        The launch resumes :func:`_conversation_to_resume`'s conversation or is fresh; an
        override resumes none. Returns an error reason (a log that will not read, or a session
        hook its arguments cannot spell, names why), or None on success.
        """
        project = store.opened
        command = pty_host.resolve_terminal_command(provider)
        if command is None:
            return provider.unavailable_reason
        unrendered, override = command
        env = pty_host.spawn_env(self.id)
        try:
            resumed = (
                None
                if override
                else _conversation_to_resume(provider, project)
            )
            argv = pty_host.render_argv(unrendered, project, env, resumed)
        except ValueError as exc:
            return str(exc)
        prepared = (
            []
            if override
            else pty_host.prepare_launch(argv[0], provider, project, self.id)
        )
        if isinstance(prepared, str):
            return prepared
        program = pty_host.launched_program(argv, override)
        launched = LaunchedProgram(
            provider=provider.id,
            prepared=prepared,
            confinement=None if override else provider.confinement,
            resumed=resumed,
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
        pty_host.start_writer(pty, self._inputs, name=f"term-{self.id}-input")
        try:
            _record(
                _STARTED,
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

    def end(
        self,
        cause: str,
        actor: str | None,
        hand_off: Callable[[], bool | None],
    ) -> bool:
        """End the session's agent for ``cause``, by ``actor`` (``None`` when no person did).

        ``hand_off`` answers :meth:`hand_off_landed` for the end, read once the process stopped.
        Under the sessions lock, so no create runs and no submit is admitted meanwhile, and in
        this order: terminates the process, which shuts the session's writer down without waiting
        for a write under way, the writer starting no write once it reads the shutdown and
        nothing it finishes taking a request off the queue; holds the session's
        :class:`TerminalEnded` as :attr:`ended`, the requests still queued, one in a paste written
        in part among them, being what its agent was never given, and writes it as one line in
        the log its start
        line was written to, its conversation resumable unless a person ended it with no switch
        under way, or, when the hand-off cannot be read or the line fails to append (an
        :data:`UNRECORDED_END`), writes that the agent ended and no end line was recorded, and
        why, to the terminal stream and the error log; then, under the session's own lock, closes
        every socket of it with :data:`ENDED_CLOSE_CODE`, drops them, so no later output reaches
        one, and forgets the session, which is what marks it ended. Returns False when the
        process survives termination, the session still held, and True once the session ended,
        by this call or an earlier one. An unrecorded end is raised as its own failure.
        """
        with _SESSIONS_LOCK:
            if self.id not in _SESSIONS:
                return True
            if not self.terminate():
                return False
            try:
                handed_off = hand_off()
                with self._lock:
                    self.ended = TerminalEnded(
                        session_id=self.id,
                        provider=self.launched.provider,
                        cause=cause,
                        undelivered=list(self._queued),
                        resumable=actor is None or self._hand_off is not None,
                        conversation=self.conversation,
                        last_output_at=self._last_output_at,
                        handed_off=handed_off,
                    )
                _record(
                    _ENDED,
                    self.ended.model_dump(mode="json"),
                    self._project,
                    actor,
                )
            except UNRECORDED_END as exc:
                with self._lock:
                    self._show(
                        f"\r\nThe agent ended and no end line was recorded: {exc}\r\n"
                    )
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
        """Return the session's id and what it launched."""
        return TerminalLaunch(session_id=self.id, launched=self.launched)

    def submit(self, text: str) -> None:
        """Queue ``text`` for the session's agent; :meth:`deliver` pastes it."""
        with self._lock:
            self._queued.append(text)

    def hand_off(self, by: str) -> bool:
        """Ask the agent for its hand-off before a switch of harness asked for ``by`` a person.

        :data:`HAND_OFF_REQUEST` goes ahead of the queued requests, and the session ends by
        ``by`` once the agent's hand-off lands (:meth:`_follow_hand_off`). Asked again, it
        changes nothing. Returns False for a session for no project, which has no log to see a
        hand-off land in, and asks for nothing.
        """
        with self._lock:
            if self._project is None:
                return False
            if self._hand_off is None:
                _, cursor = audit_entries(self._project.root)
                self._hand_off = _HandOff(by, cursor)
                self._ahead = HAND_OFF_REQUEST
            return True

    def deliver(self) -> None:
        """Hand the writer the next paste, unless one is with it already.

        The session's own text (:attr:`_ahead`) goes together with every request queued by then
        as one message, otherwise the oldest request on its own, once the agent has bracketed
        paste on and has written its row's ``composer_ready``. The requests a paste carried leave
        the queue only once it is written whole, and the next paste is handed over then.
        """
        with self._lock:
            if self._pasting or not (self._ready and self._composer_live):
                return
            if self._ahead is not None:
                text = "\n\n".join([self._ahead, *self._queued])
                taken, self._ahead = len(self._queued), None
            elif self._queued:
                text, taken = self._queued[0], 1
            else:
                return
            self._pasting = self.write(
                pty_host.paste(text), lambda: self._pasted(taken)
            )

    def _pasted(self, taken: int) -> None:
        """Take the ``taken`` requests a written paste carried off the queue, then deliver on.

        Once the writer is shut down the queue is left as the session's end found it.
        """
        with self._lock:
            if self._inputs.is_shutdown:
                return
            del self._queued[:taken]
            self._pasting = False
        self.deliver()

    def terminate(self) -> bool:
        """Kill the PTY and return True when no process remains afterward.

        A process it stops has its writer stopped too, never waited for. Returns False when it
        survives: ``self._pty`` is then restored.
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
        with self._lock:
            self._inputs.shutdown(immediate=True)
        return True

    def alive(self) -> bool:
        """Return whether the session's PTY process is running."""
        pty = self._pty
        return bool(pty is not None and pty.isalive())

    # ── I/O ─────────────────────────────────────────────────────────────

    def write(
        self, data: str, written: Callable[[], object] = lambda: None
    ) -> bool:
        """Queue ``data`` for the agent's input behind every input accepted before it.

        The session's writer writes it whole, then calls ``written``. Returns False, ``data``
        never written, once the writer has stopped.
        """
        try:
            self._inputs.put((data, written))
        except ShutDown:
            return False
        return True

    def resize(self, rows: int, cols: int) -> None:
        """Set the PTY's window size; a failed resize or no PTY is a no-op."""
        pty = self._pty
        if pty is None:
            return
        try:
            pty.resize(rows, cols)
        except Exception:
            # pywinpty raises EOFError or WinptyError, not OSError, on a dying PTY; a failed
            # resize must never crash the WebSocket handler (the rail resizes on every reconnect).
            pass

    # ── output pump (called from the reader thread) ─────────────────────

    def _on_output(self, data: str) -> None:
        """Take the agent's output: note when and its paste readiness, show it, then deliver.

        After :meth:`deliver`, :meth:`_follow_hand_off` looks for a hand-off asked for.
        """
        with self._lock:
            self._last_output_at = now()
            seen = self._tail + data
            self._ready = pty_host.bracketed_paste(seen, self._ready)
            self._composer_live = (
                self._composer_live
                or _provider(self.launched.provider).composer_ready in seen
            )
            self._tail = seen[-_TAIL_CHARS:]
            self._show(data)
        self.deliver()
        self._follow_hand_off()

    def hand_off_landed(self) -> bool | None:
        """Return whether the hand-off asked for has landed, ``None`` when none was asked for.

        It has once a ``write_retrospective`` line the agent's MCP server stamped with this
        session's id is in the session's log past the cursor the hand-off was asked at.
        """
        from tcip_mcp.tools.meta_tools import write_retrospective

        hand_off = self._hand_off
        if hand_off is None or self._project is None:
            return None
        lines, _ = acts_of(
            self._project.root,
            (write_retrospective.__name__,),
            after=hand_off.after,
        )
        return any(line.get("terminal_session") == self.id for line in lines)

    def _follow_hand_off(self) -> None:
        """End the session by the person who asked for its hand-off, once that hand-off landed.

        A termination that fails leaves the session held, and the next output or End tries again.
        """
        hand_off = self._hand_off
        if hand_off is not None and self.hand_off_landed():
            self.end(
                "the agent handed off for a switch of harness",
                hand_off.by,
                lambda: True,
            )

    def _show(self, data: str) -> None:
        """Keep ``data`` in the scrollback and offer it to every subscriber; under the lock."""
        self._scrollback = (self._scrollback + data)[-SCROLLBACK_MAX_CHARS:]
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

        Its hand-off is read by :meth:`end` once the process stopped, so a log that will not read
        still leaves the session stopped and forgotten. An :data:`UNRECORDED_END`, which
        :meth:`end` has disclosed, is not raised. Returns what :meth:`end` returns.
        """
        with contextlib.suppress(*UNRECORDED_END):
            return self.end(cause, None, self.hand_off_landed)
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
            replay = self._scrollback
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
            return self._scrollback


_SESSIONS: dict[str, TerminalSession] = {}
# Serializes a create, a submit's admission and a whole end across the request threadpool and
# the reader threads: without it, a create beside an end could run two agent processes.
_SESSIONS_LOCK = threading.Lock()


def shutdown_all() -> None:
    """End every agent terminal for the backend's shutdown.

    A survivor of termination, and an end whose line fails to append or whose hand-off cannot be
    read, are each logged, and every other session's end is still attempted.
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


def _conversation_to_resume(
    provider: pty_host.Provider, project: OpenProject | None
) -> str | None:
    """Return the conversation a launch of ``provider`` in ``project`` resumes, ``None`` for none.

    It is the conversation the newest ``agent_terminal_ended`` line of the row in the project's
    log names, when that end left it resumable and the line's last output is within the row's
    :attr:`~tcip_web.terminal.Resume.cache_lifetime` of now. A row with no
    :class:`~tcip_web.terminal.Resume` and a launch for no project resume none. Raises
    ``ValueError`` for a log or an end line that will not read.
    """
    resume = provider.resume
    if resume is None or project is None:
        return None
    lines, _ = acts_of(project.root, (_ENDED,))
    ends = [TerminalEnded.model_validate(line["arguments"]) for line in lines]
    newest = next(
        (e for e in reversed(ends) if e.provider == provider.id), None
    )
    if (
        newest is None
        or not newest.resumable
        or newest.last_output_at is None
        or now() - newest.last_output_at > resume.cache_lifetime
    ):
        return None
    return newest.conversation


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
    """Return the live session (attach semantics, like tmux) or start a new one.

    A new one runs the requested provider by the person ``req.user`` names
    (:meth:`TerminalSession.start`).
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
    """An end of a session's agent, at once or after its hand-off, by the person ``user`` names."""

    user: str


def _end_by(session: TerminalSession, person: str) -> TerminalEnd:
    """End ``session``'s agent by ``person`` at once (:meth:`TerminalSession.end`).

    Its hand-off is read first (:meth:`TerminalSession.hand_off_landed`), so a log that will not
    read raises with the agent still running. Answers the requests its agent was never given.
    503 when the process survives termination, the session kept reachable; the audit-gap 409
    carrying that answer when the end's line fails to append.
    """
    handed_off = session.hand_off_landed()
    unrecorded: AuditEntryNotWrittenError | None = None
    try:
        stopped = session.end(
            "the person ended it", person, lambda: handed_off
        )
    except AuditEntryNotWrittenError as exc:
        stopped, unrecorded = True, exc
    if not stopped:
        raise HTTPException(503, _SURVIVED)
    answer = TerminalEnd(undelivered=session.ended.undelivered)
    if unrecorded is not None:
        raise audit_gap_409(unrecorded, answer) from unrecorded
    return answer


@router.post("/sessions/{session_id}/end")
def end_session(session_id: str, req: EndSessionRequest) -> TerminalEnd:
    """End the session's agent at once by the person ``req.user`` names (:func:`_end_by`).

    404 for an unknown session.
    """
    person = actor(req.user)
    return _end_by(_require(session_id), person)


@router.post("/sessions/{session_id}/hand-off")
def hand_off_session(
    session_id: str, req: EndSessionRequest
) -> TerminalEnd | None:
    """Ask the session's agent for its hand-off before a switch of harness, by ``req.user``.

    The hand-off request is delivered ahead of the queued requests and the session ends once
    the hand-off lands (:meth:`TerminalSession.hand_off`), answering ``None``; a session for no
    project ends at once instead, answered as :func:`_end_by` answers. 404 for an unknown
    session.
    """
    person = actor(req.user)
    with _SESSIONS_LOCK:
        session = _require(session_id)
        handing_off = session.hand_off(person)
    if not handing_off:
        return _end_by(session, person)
    session.deliver()
    return None


@router.post("/sessions/{session_id}/session-start")
def session_start(
    session_id: str, req: SessionStartReport
) -> SessionOrientation:
    """Hold ``req.conversation`` as the session agent's conversation and answer its orientation.

    A later report replaces the conversation; the session's end line records the last. The
    orientation is :func:`~tcip_web.terminal.orientation` for the project the session was
    launched for and ``req.source``. 404 for a session the backend does not hold.
    """
    with _SESSIONS_LOCK:
        session = _require(session_id)
        session.conversation = req.conversation
    return SessionOrientation(
        text=pty_host.orientation(session._project, req.source)
    )


class SubmitRequest(BaseModel):
    """A request for the agent: text it receives as one pasted message."""

    text: str


@router.post("/sessions/{session_id}/submit")
def submit_to_session(session_id: str, req: SubmitRequest) -> dict:
    """Queue ``req.text`` for the session's agent (:meth:`TerminalSession.deliver`).

    404 for a session the backend does not hold, ended or never started. The request is queued
    under the sessions lock, so an end after it answers it among the undelivered, and handed to
    the writer once that lock is released.
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
        async for text in websocket.iter_text():
            try:
                frame = _FRAME.validate_json(text)
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
