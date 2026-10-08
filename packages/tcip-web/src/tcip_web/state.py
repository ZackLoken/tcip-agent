"""The web backend's own state: the workspace it serves, the project it has open, that project's
live :class:`~tcip_mcp.web_client.GuiState` (persisted to the project's GUI snapshot record on
every change made for it) and the panel events it retains for a browser that connects late.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional, TypeVar

from pydantic import BaseModel, ValidationError

from tcip_mcp.web_client import GuiState, read_gui_snapshot, require_whole, write_gui_snapshot

logger = logging.getLogger(__name__)

T = TypeVar("T")

RETAINED_EVENTS_PER_PANEL = 64
"""How many of the open project's latest panel events each panel replays to a browser that
connects late; chosen here, so a reconnecting tab catches up on a session's recent steering."""


class GuiMutationInvalidError(ValueError):
    """A :meth:`StateStore.mutate` call whose merged result does not validate as ``GuiState``."""


class NoProjectOpenError(RuntimeError):
    """A request that acts on the open project arrived while the backend has none open."""


class ProjectNotOpenError(NoProjectOpenError):
    """A request built for one project arrived while the backend has another open."""

    def __init__(self, open_project_id: str) -> None:
        super().__init__("the backend does not have this project open")
        self.open_project_id = open_project_id


class OpenProject(NamedTuple):
    """The project the backend has open: its resolved directory and the id its record holds."""

    root: Path
    id: str


class Held(NamedTuple):
    """The open project (``None`` when none is) and its live GUI state, published as one value."""

    project: Optional[OpenProject]
    state: GuiState


class StateStore:
    """Holds the workspace the backend serves, its open project and that project's live
    :class:`GuiState` as one value (:class:`Held`), which every change made for the project
    persists before it is broadcast; a change made while no project is open is held and
    persisted nowhere."""

    def __init__(self) -> None:
        self._now = Held(None, GuiState())
        self._version = 0
        self._lock = asyncio.Lock()
        self._subscribers: list = []  # list[Callable[[dict], Awaitable[None]]]
        self._workspace: Optional[Path] = None
        self._image_roots: tuple[Path, ...] = ()
        self._retained: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=RETAINED_EVENTS_PER_PANEL))

    def configure(self, workspace: Path, image_roots: tuple[Path, ...]) -> None:
        """Set the workspace this backend serves and the additive image roots it admits, each
        resolved once where the backend starts."""
        self._workspace, self._image_roots = workspace, image_roots

    @property
    def workspace(self) -> Path:
        """The workspace this backend serves; raises ``RuntimeError`` when it started without
        one."""
        if self._workspace is None:
            raise RuntimeError("the backend was started without a workspace; start it through "
                               "python -m tcip_web")
        return self._workspace

    @property
    def image_roots(self) -> tuple[Path, ...]:
        """The additive image roots this backend was started with."""
        return self._image_roots

    @property
    def opened(self) -> Optional[OpenProject]:
        """The open project, its directory and id together, or ``None`` when none is open: one
        value, so a reader that wants both reads them in one step."""
        return self._now.project

    def held(self) -> OpenProject:
        """The open project; raises :class:`NoProjectOpenError` when none is open."""
        return self.held_state()[0]

    def held_state(self) -> tuple[OpenProject, GuiState]:
        """The open project and its GUI state, read as one value, so a reader that wants both
        never pairs one project's identity with another's state; raises
        :class:`NoProjectOpenError` when none is open."""
        now = self._now
        if now.project is None:
            raise NoProjectOpenError("no project is open; open one from the project list first")
        return now.project, now.state

    def admit(self, project_id: str) -> OpenProject:
        """The open project (:meth:`held`), when ``project_id`` names it; raises
        :class:`ProjectNotOpenError` naming the open project's id when another is open, and
        what :meth:`held` raises when none is."""
        opened = self.held()
        if project_id != opened.id:
            raise ProjectNotOpenError(opened.id)
        return opened

    async def open_project(self, project: OpenProject) -> None:
        """Make ``project`` the open project (:meth:`_switch`), holding its persisted state (a
        fresh state when it has none). Raises what :func:`tcip_mcp.web_client.read_gui_snapshot`
        raises, leaving the open project unchanged."""
        state = await asyncio.to_thread(read_gui_snapshot, project.root) or GuiState()
        await self._switch(project, state)

    async def close_project(self, project_id: str) -> None:
        """Depart from the open project (:meth:`_switch`) when ``project_id`` names it
        (:meth:`admit`), the comparison and the close one step under the lock; a no-op when it
        does not, or none is open."""
        await self._switch(None, GuiState(), closing=project_id)

    async def _switch(self, following: Optional[OpenProject], state: GuiState,
                      closing: Optional[str] = None) -> None:
        """Under the lock: when ``closing`` is given and does not name the open project, nothing.
        Otherwise, when ``following`` replaces another project, end every unended session of the
        departed one (:func:`tcip_web.routes.sessions.end_sessions`), then, once that commit has
        returned, publish ``following`` with ``state`` as one value, its retained events cleared,
        and notify subscribers; a departure that raises publishes nothing. The departure's
        ``session_ended`` lines (:func:`tcip_web.routes.sessions.record_ended`) follow the
        publication, and a line that cannot be written raises then."""
        from tcip_web.routes.sessions import end_sessions, record_ended

        async with self._lock:
            if closing is not None:
                try:
                    self.admit(closing)
                except NoProjectOpenError:
                    return
            departed = self._now.project
            ended: list[dict] = []
            if departed is not None and (following is None or following.id != departed.id):
                ended = await asyncio.to_thread(end_sessions, departed.root, None)
            await self._hold(following, state)
        if departed is not None and ended:
            await asyncio.to_thread(record_ended, departed.root, ended)

    async def admitted(self, project_id: str, act: Callable[[Path], T]) -> T:
        """``act`` run on a worker thread with the open project's directory, ``project_id``
        admitted against it (:meth:`admit`, raising what it raises) and the act done under the
        lock, so no switch publishes between the admission and the act."""
        async with self._lock:
            root = self.admit(project_id).root
            return await asyncio.to_thread(act, root)

    def retain_event(self, panel: str, event: dict[str, Any]) -> None:
        """Keep ``event``, admitted for the open project by its caller in the same synchronous
        step, for :meth:`retained_events`; a switch publishes its project and clears the
        retained events in one synchronous step too, so the event is never kept for another."""
        self._retained[panel].append(event)

    def retained_events(self, panel: str) -> list[dict[str, Any]]:
        """The open project's latest events on ``panel``, oldest first."""
        return list(self._retained.get(panel, ()))

    async def _hold(self, project: Optional[OpenProject], state: GuiState) -> None:
        """Hold ``project`` with ``state`` as the next version, the retained events cleared when
        the project changes, and notify every subscriber of it; called under the lock."""
        if project != self._now.project:
            self._retained.clear()
        self._now = Held(project, state)
        self._version += 1
        payload = {"state": self.snapshot(), "version": self._version}
        for cb in list(self._subscribers):
            try:
                await cb(payload)
            except Exception:
                logger.exception("state subscriber failed")

    def subscribe(self, callback) -> None:
        """Register a coroutine called with each mutation payload ``{state, version}``."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback) -> None:
        try:
            self._subscribers.remove(callback)
        except ValueError:
            pass

    @property
    def state(self) -> GuiState:
        """The held GUI state (:meth:`held_state` for it with its project)."""
        return self._now.state

    @property
    def version(self) -> int:
        """Monotonic version, bumped on every state change.

        Broadcast alongside each snapshot so a browser can drop a stale replay
        (e.g. a reconnecting socket resending an older snapshot after newer local
        state has been applied).
        """
        return self._version

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot of the current state."""
        return self._now.state.model_dump(mode="json")

    async def mutate(self, mutation: dict[str, Any], *, project: Optional[OpenProject]) -> None:
        """Apply a partial mutation, validated, persist it to the open project, and hold it.

        ``project`` is the project the mutation was made for, admitted under the lock
        (:meth:`admit`; :class:`ProjectNotOpenError` when it is no longer the open one,
        :class:`NoProjectOpenError` when none is, nothing held either way), or ``None`` for a
        mutation made while no project was open, which is held and persisted nowhere.
        ``mutation`` is a shallow dict of top-level field names; a nested field is set by
        passing a fully built model instance or a complete dict for it.
        After the admission the merged result is validated through :class:`GuiState` and
        :func:`~tcip_mcp.web_client.require_whole`: a mutation that does not validate, or states
        a nested field partly, raises :class:`GuiMutationInvalidError` naming the field. A state
        that cannot be persisted raises the store's own error, and nothing is held.
        """
        async with self._lock:
            opened = None if project is None else self.admit(project.id)
            dumped = {
                k: v.model_dump(mode="json") if isinstance(v, BaseModel) else v
                for k, v in mutation.items()
            }
            merged = {**self._now.state.model_dump(mode="json"), **dumped}
            try:
                new_state = GuiState.model_validate(merged)
                require_whole(new_state, "the mutated state")
            except (ValidationError, ValueError) as exc:
                raise GuiMutationInvalidError(str(exc)) from exc
            if opened is not None:
                await asyncio.to_thread(write_gui_snapshot, opened.root, new_state)
            await self._hold(self._now.project, new_state)


# Module-level singleton; the FastAPI app imports this.
store = StateStore()
