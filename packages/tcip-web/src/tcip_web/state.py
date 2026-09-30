"""The web backend's own state: the workspace it serves, the project it has open, that project's
live :class:`~tcip_mcp.web_client.GuiState` (persisted to the project's ``.tcip/state/gui.json`` on
every change) and the panel events it retains for a browser that connects late.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ValidationError

from tcip_mcp.web_client import GuiState, read_gui_snapshot, require_whole, write_gui_snapshot

logger = logging.getLogger(__name__)

RETAINED_EVENTS_PER_PANEL = 64
"""How many of the open project's latest panel events each panel replays to a browser that
connects late; chosen here, so a reconnecting tab catches up on a session's recent steering."""


class GuiMutationInvalid(ValueError):
    """A :meth:`StateStore.mutate` call whose merged result does not validate as ``GuiState``."""


class NoProjectOpen(RuntimeError):
    """A request that acts on the open project arrived while the backend has none open."""


class ProjectNotOpen(RuntimeError):
    """A request built for one project arrived while the backend has another open, or none."""

    def __init__(self, open_project_id: Optional[str]) -> None:
        super().__init__("the backend does not have this project open")
        self.open_project_id = open_project_id


class StateStore:
    """Holds the workspace the backend serves, its open project and that project's live
    :class:`GuiState`, which every change persists before it is broadcast."""

    def __init__(self) -> None:
        self._state = GuiState()
        self._version = 0
        self._lock = asyncio.Lock()
        self._subscribers: list = []  # list[Callable[[dict], Awaitable[None]]]
        self._project: Optional[Path] = None
        self._project_id: Optional[str] = None
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
    def project_root(self) -> Optional[Path]:
        """The open project's resolved directory, or ``None`` when no project is open."""
        return self._project

    @property
    def project_id(self) -> Optional[str]:
        """The open project's id, read from its record when it was opened; ``None`` when no
        project is open."""
        return self._project_id

    def open_root(self) -> Path:
        """The open project's directory; raises :class:`NoProjectOpen` when none is open."""
        project = self.project_root
        if project is None:
            raise NoProjectOpen("no project is open; open one from the project list first")
        return project

    def admit(self, project_id: str) -> Path:
        """The open project's directory, when ``project_id`` names it; raises
        :class:`ProjectNotOpen` naming the open project's id otherwise."""
        if project_id != self.project_id:
            raise ProjectNotOpen(self.project_id)
        return self.open_root()

    async def open_project(self, project: Path) -> None:
        """Make ``project`` the open project, holding its persisted state (a fresh state when it
        has none), and notify subscribers. Raises what
        :func:`tcip_mcp.project_record.read_record` and
        :func:`tcip_mcp.web_client.read_gui_snapshot` raise, leaving the open project unchanged."""
        from tcip_mcp.project_record import read_record

        project_id = read_record(project)["id"]
        state = await asyncio.to_thread(read_gui_snapshot, project) or GuiState()
        async with self._lock:
            self._project, self._project_id = project, project_id
            self._retained.clear()
            await self._hold(state)

    async def close_project(self) -> None:
        """Forget the open project and the panel events it retained, and notify subscribers."""
        async with self._lock:
            self._project = self._project_id = None
            self._retained.clear()
            await self._hold(GuiState())

    def retain_event(self, panel: str, event: dict[str, Any]) -> None:
        """Keep ``event``, admitted for the open project, for :meth:`retained_events`."""
        self._retained[panel].append(event)

    def retained_events(self, panel: str) -> list[dict[str, Any]]:
        """The open project's latest events on ``panel``, oldest first."""
        return list(self._retained.get(panel, ()))

    async def _hold(self, state: GuiState) -> None:
        """Hold ``state`` as the next version and notify every subscriber of it; called under
        the lock."""
        self._state = state
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
        return self._state

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
        return self._state.model_dump(mode="json")

    async def mutate(self, mutation: dict[str, Any]) -> None:
        """Apply a partial mutation, validated, persist it to the open project, and hold it.

        ``mutation`` is a shallow dict of top-level field names; a nested field is set by
        passing a fully built model instance or a complete dict for it. The merged result is
        validated through :class:`GuiState` and :func:`~tcip_mcp.web_client.require_whole` before
        anything else: a mutation that does not validate, or states a nested field partly, raises
        :class:`GuiMutationInvalid` naming the field. A state that cannot be persisted raises the
        store's own error, and nothing is held.
        """
        async with self._lock:
            dumped = {
                k: v.model_dump(mode="json") if isinstance(v, BaseModel) else v
                for k, v in mutation.items()
            }
            merged = {**self._state.model_dump(mode="json"), **dumped}
            try:
                new_state = GuiState.model_validate(merged)
                require_whole(new_state, "the mutated state")
            except (ValidationError, ValueError) as exc:
                raise GuiMutationInvalid(str(exc)) from exc
            if self._project is not None:
                await asyncio.to_thread(write_gui_snapshot, self._project, new_state)
            await self._hold(new_state)


# Module-level singleton; the FastAPI app imports this.
store = StateStore()
