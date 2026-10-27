"""The web backend's own state: its workspace, open project, GUI state and retained events.

The workspace is the one it serves; the open project's live
:class:`~tcip_mcp.web_client.GuiState` is persisted to the project's GUI snapshot record on
every change made for it; the panel events are retained.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal, NamedTuple, TypeVar

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from tcip_mcp.web_client import (
    GUI_STATE_DEFAULTS,
    DatasetSelection,
    GuiState,
    Panel,
    read_gui_snapshot,
    write_gui_snapshot,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

RETAINED_EVENTS_PER_PANEL = 64
"""How many of the open project's latest panel events each panel retains and replays on connect;
chosen here."""


class StateNotification(BaseModel):
    """A held GUI state as its subscribers are told of it: the state and its version."""

    state: GuiState
    version: int


class SnapshotProject(BaseModel):
    """The open project as the backend's answers name it: its id and its directory."""

    id: str
    path: str

    @classmethod
    def of(cls, project: OpenProject) -> SnapshotProject:
        """Return ``project``'s id and its directory as text."""
        return cls(id=project.id, path=str(project.root))


class StateSnapshotMessage(StateNotification):
    """The state-snapshot envelope: a notification with the open project and the backend's epoch.

    ``project`` is ``None`` when no project is open.
    """

    type: Literal["state_snapshot"]
    project: SnapshotProject | None
    epoch: str


type Subscriber = Callable[[StateNotification], Awaitable[None]]
"""A coroutine called with each held state's notification."""


class PanelEnvelope(BaseModel):
    """A panel event as retained and broadcast: its panel, type, data and id.

    The agent identity its sender declared
    (:func:`tcip_mcp.agent_identity.fields_from_headers`) rides beside them as further fields.
    """

    model_config = ConfigDict(extra="allow")

    panel: Panel
    event_type: str
    data: dict[str, JsonValue]
    event_id: str


class GuiMutationInvalidError(ValueError):
    """A :meth:`StateStore.mutate` call whose merged result does not validate as ``GuiState``."""


class NoSelectionError(ValueError):
    """A request that acts on the selected capture arrived while none is selected."""


def selected(state: GuiState) -> DatasetSelection:
    """Return ``state``'s capture selection.

    Raises :class:`NoSelectionError` naming the select route when none is selected.
    """
    if state.dataset is None:
        raise NoSelectionError(
            "no capture is selected; select one through /api/dataset/select"
        )
    return state.dataset


class NoProjectOpenError(RuntimeError):
    """A request that acts on the open project arrived while the backend has none open."""


class ProjectNotOpenError(NoProjectOpenError):
    """A request built for one project arrived while the backend has another open."""

    def __init__(self, open_project_id: str) -> None:
        """Take the id of the project the backend has open."""
        super().__init__("the backend does not have this project open")
        self.open_project_id = open_project_id


class OpenProject(NamedTuple):
    """The project the backend has open: its resolved directory and the id its record holds."""

    root: Path
    id: str


class Held(NamedTuple):
    """The open project (``None`` when none is) and its live GUI state, published as one value."""

    project: OpenProject | None
    state: GuiState


class StateStore:
    """The workspace the backend serves, its open project and that project's live GUI state.

    The project and its :class:`GuiState` are held as one value (:class:`Held`), which every
    change made for the project persists before it is broadcast; a change made while no project
    is open is held and persisted nowhere.
    """

    def __init__(self) -> None:
        """Start with no workspace, no open project and a fresh state at version 0."""
        self._now = Held(None, GUI_STATE_DEFAULTS)
        self._version = 0
        self._lock = asyncio.Lock()
        self._subscribers: list[Subscriber] = []
        self._workspace: Path | None = None
        self._image_roots: tuple[Path, ...] = ()
        self._retained: dict[Panel, deque[PanelEnvelope]] = defaultdict(
            lambda: deque(maxlen=RETAINED_EVENTS_PER_PANEL)
        )

    def configure(
        self, workspace: Path, image_roots: tuple[Path, ...]
    ) -> None:
        """Set the workspace this backend serves and the additive image roots it admits."""
        self._workspace, self._image_roots = workspace, image_roots

    @property
    def workspace(self) -> Path:
        """The workspace this backend serves.

        Raises ``RuntimeError`` when it started without one.
        """
        if self._workspace is None:
            raise RuntimeError(
                "the backend was started without a workspace; start it through "
                "python -m tcip_web"
            )
        return self._workspace

    @property
    def image_roots(self) -> tuple[Path, ...]:
        """The additive image roots this backend was started with."""
        return self._image_roots

    @property
    def opened(self) -> OpenProject | None:
        """The open project, its directory and id together, or ``None`` when none is open."""
        return self._now.project

    def held(self) -> OpenProject:
        """Return the open project; raises :class:`NoProjectOpenError` when none is open."""
        return self.held_state()[0]

    def held_state(self) -> tuple[OpenProject, GuiState]:
        """Return the open project and its GUI state, read as one value.

        Raises :class:`NoProjectOpenError` when none is open.
        """
        now = self._now
        if now.project is None:
            raise NoProjectOpenError(
                "no project is open; open one from the project list first"
            )
        return now.project, now.state

    def admit(self, project_id: str) -> OpenProject:
        """Return the open project (:meth:`held`) when ``project_id`` names it.

        Raises :class:`ProjectNotOpenError` naming the open project's id when another is open,
        and what :meth:`held` raises when none is.
        """
        opened = self.held()
        if project_id != opened.id:
            raise ProjectNotOpenError(opened.id)
        return opened

    async def open_project(self, project: OpenProject) -> None:
        """Make ``project`` the open project (:meth:`_switch`), holding its persisted state.

        A fresh state is held when it has none. Raises what
        :func:`tcip_mcp.web_client.read_gui_snapshot` raises, leaving the open project unchanged.
        """
        state = (
            await asyncio.to_thread(read_gui_snapshot, project.root)
            or GUI_STATE_DEFAULTS
        )
        await self._switch(project, state)

    async def close_project(self, project_id: str) -> None:
        """Depart from the open project (:meth:`_switch`) when ``project_id`` names it.

        The name is compared through :meth:`admit`, the comparison and the close one step under
        the lock; a no-op when it does not name it, or none is open.
        """
        await self._switch(None, GUI_STATE_DEFAULTS, closing=project_id)

    async def _switch(
        self,
        following: OpenProject | None,
        state: GuiState,
        closing: str | None = None,
    ) -> None:
        """Publish ``following`` with ``state`` under the lock, ending the departed sessions first.

        When ``closing`` is given and does not name the open project, nothing. Otherwise, when
        ``following`` replaces another project, end every unended session of the departed one
        (:func:`tcip_web.routes.sessions.end_sessions`), then, once that commit has returned,
        publish ``following`` with ``state`` as one value, its retained events cleared, and
        notify subscribers; a departure that raises publishes nothing. The departure's
        ``session_ended`` lines (:func:`tcip_web.routes.sessions.record_ended`) follow the
        publication, and a line that cannot be written raises then.
        """
        from tcip_web.routes.sessions import end_sessions, record_ended

        async with self._lock:
            if closing is not None:
                try:
                    self.admit(closing)
                except NoProjectOpenError:
                    return
            departed = self._now.project
            ended = (
                await asyncio.to_thread(end_sessions, departed.root, None)
                if departed is not None
                and (following is None or following.id != departed.id)
                else []
            )
            await self._hold(following, state)
        if departed is not None and ended:
            await asyncio.to_thread(record_ended, departed.root, ended)

    async def admitted(self, project_id: str, act: Callable[[Path], T]) -> T:
        """Run ``act`` on a worker thread with the open project's directory and return its result.

        ``project_id`` is admitted against it (:meth:`admit`, raising what it raises) and the act
        done under the lock, so no switch publishes between the admission and the act.
        """
        async with self._lock:
            root = self.admit(project_id).root
            return await asyncio.to_thread(act, root)

    def retain_event(self, event: PanelEnvelope) -> None:
        """Append ``event`` to its panel's retained events."""
        self._retained[event.panel].append(event)

    def retained_events(self, panel: Panel) -> list[PanelEnvelope]:
        """Return the open project's latest events on ``panel``, oldest first."""
        return list(self._retained.get(panel, ()))

    async def _hold(
        self, project: OpenProject | None, state: GuiState
    ) -> None:
        """Hold ``project`` with ``state`` as the next version and notify every subscriber of it.

        The retained events are cleared when the project changes. Called under the lock.
        """
        if project != self._now.project:
            self._retained.clear()
        self._now = Held(project, state)
        self._version += 1
        notification = self.notification()
        for cb in list(self._subscribers):
            try:
                await cb(notification)
            except Exception:
                logger.exception("state subscriber failed")

    def subscribe(self, callback: Subscriber) -> None:
        """Register a coroutine called with each held state's notification."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Subscriber) -> None:
        """Remove a registered ``callback``; a no-op when it is not registered."""
        try:
            self._subscribers.remove(callback)
        except ValueError:
            pass

    @property
    def state(self) -> GuiState:
        """The held GUI state."""
        return self._now.state

    @property
    def version(self) -> int:
        """Monotonic version, bumped on every state change."""
        return self._version

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot of the current state."""
        return self._now.state.model_dump(mode="json")

    def notification(self) -> StateNotification:
        """Return the held state with its version."""
        return StateNotification(state=self._now.state, version=self._version)

    async def mutate(
        self,
        mutation: Callable[[GuiState], dict[str, Any]],
        *,
        project: OpenProject | None,
    ) -> None:
        """Apply a partial mutation, validated, persist it to the open project, and hold it.

        ``project`` is the project the mutation was made for, admitted under the lock
        (:meth:`admit`; :class:`ProjectNotOpenError` when it is no longer the open one,
        :class:`NoProjectOpenError` when none is, nothing held either way), or ``None`` for a
        mutation made while no project was open, which is held and persisted nowhere.
        ``mutation`` returns, from the state held once the project is admitted, a shallow dict of
        top-level field names; its exception propagates with nothing held, and a nested field is
        set by a fully built model instance or a complete dict for it. The merged result is
        validated through :class:`GuiState`: a mutation that does not validate, or states a
        nested field partly, raises :class:`GuiMutationInvalidError` naming the field. A state
        that cannot be persisted raises the store's own error, and nothing is held.
        """
        async with self._lock:
            opened = None if project is None else self.admit(project.id)
            dumped = {
                k: v.model_dump(mode="json", round_trip=True)
                if isinstance(v, BaseModel)
                else v
                for k, v in mutation(self._now.state).items()
            }
            merged = {
                **self._now.state.model_dump(mode="json", round_trip=True),
                **dumped,
            }
            try:
                new_state = GuiState.model_validate(merged)
            except ValidationError as exc:
                raise GuiMutationInvalidError(str(exc)) from exc
            if opened is not None:
                await asyncio.to_thread(
                    write_gui_snapshot, opened.root, new_state
                )
            await self._hold(self._now.project, new_state)


# Module-level singleton; the FastAPI app imports this.
store = StateStore()
