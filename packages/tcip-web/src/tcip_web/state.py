"""In-memory GUI state + debounced persistence to ``.tcip/state/gui.json``.

This is the single source of truth for live browser state. Both user actions
(from the browser) and agent actions (from MCP tools) mutate this store;
changes fan out to connected browsers via WebSocket.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from tcip_store import StoreError, read, replace

from tcip_mcp.web_client import ActiveTab, AnnotateMode, canvas_open_binding_key, gui_snapshot_key
from tcip_mcp.web_client import TAB_NAMES as TAB_NAMES

logger = logging.getLogger(__name__)

PERSIST_DEBOUNCE_SECONDS = 0.5


class GuiMutationInvalid(ValueError):
    """A :meth:`StateStore.mutate` call whose merged result does not validate as ``GuiState``."""


class GuiVocabulary(BaseModel):
    """``active_tab`` and ``mode``, held together as one small model rather than the whole of
    :class:`GuiState`, for ``tools/generate_frontend_types.py`` to project into
    ``frontend/src/api/types.generated.ts``. ``store/types.ts``'s ``Mode`` takes the ``mode``
    field's type from the generated module instead of restating the literal union by hand.
    """

    active_tab: ActiveTab
    mode: AnnotateMode


# ── Pydantic state model ────────────────────────────────────────────────


class DatasetSelection(BaseModel):
    """Which dataset the GUI is currently looking at."""

    model_config = ConfigDict(extra="forbid")

    project_root: Optional[str] = None
    dataset_root: Optional[str] = None
    subject: Optional[str] = None          # e.g. "bush"
    date: Optional[str] = None             # e.g. "2-11-26"
    image_list: list[str] = Field(default_factory=list)
    current_image_index: int = 0

    # Image/label/prediction dirs resolved through dataset_layout, never composed by the browser.
    # One file per image holds every subject, so no path carries a subject or task segment.
    images_dir: Optional[str] = None
    annotations_dir: Optional[str] = None
    predictions_dir: Optional[str] = None


class PredictionReference(BaseModel):
    """Shape of the ``pred_reference`` field below: part of ``gui_snapshot``'s frozen version 1
    schema, held to it for every already-persisted ``gui.json``. It has no producer since the
    Review-to-Annotate hand-off route was removed; the working shape for a reviewer's edit is
    the in-place Edit flow inside Review itself."""

    model_config = ConfigDict(extra="forbid")

    type: str  # "box" | "polygon"
    coords: list[float] | list[list[float]]
    subject: str = ""
    confidence: Optional[float] = None


class ViewState(BaseModel):
    """Pan/zoom state shared between Annotate and Review tabs."""

    model_config = ConfigDict(extra="forbid")

    scale: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0


class ReviewFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iou_threshold: float = 0.5
    conf_threshold: float = 0.25
    filter_type: str = "all"      # all|tp|fp|fn
    filter_class: str | int = "all"
    detection_idx: int = 0


class GuiState(BaseModel):
    """Complete GUI state, persisted to gui.json and broadcast to browsers.

    Only the *backend-authoritative* slice (``dataset``) meaningfully round-trips:
    the browser owns navigation / view / mode / class / review-filter state and
    keeps its own copy (the FE merges snapshots rather than replacing), so those
    fields here are advisory. Training-run / inference-job / class-registry state
    lives with the corresponding tabs, not here.
    """

    model_config = ConfigDict(extra="forbid")

    active_tab: ActiveTab = "annotate"
    dataset: DatasetSelection = Field(default_factory=DatasetSelection)
    view: ViewState = Field(default_factory=ViewState)
    mode: AnnotateMode = "box"
    active_subject: str = ""
    review: ReviewFilters = Field(default_factory=ReviewFilters)
    # No producer: kept only because gui_snapshot is frozen at version 1 and every persisted
    # gui.json already carries this field. See PredictionReference's own docstring.
    pred_reference: Optional[PredictionReference] = None


# ── Store with debounced persistence ────────────────────────────────────


class StateStore:
    """Holds the live :class:`GuiState` and persists it with a 500 ms debounce.

    Safe to call :meth:`mutate` from inside FastAPI request handlers or
    background tasks; a single pending-save coroutine is reused until it
    flushes.
    """

    def __init__(self) -> None:
        self._state = GuiState()
        self._version = 0
        self._save_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._subscribers: list = []  # list[Callable[[dict], Awaitable[None]]]
        self._project_root: Optional[Path] = None
        self._binding_generation: Optional[int] = None

    @property
    def project_root(self) -> Optional[Path]:
        """The open project's guarded, resolved root, or None when no project is open.

        Persistence resolves from this, never from ``dataset.project_root`` as deserialized from
        gui.json, so a snapshot edited on disk cannot redirect the next flush.
        """
        return self._project_root

    @property
    def binding_generation(self) -> Optional[int]:
        """The ``canvas_open_binding`` generation the last ``/dataset/select`` recorded, or
        ``None`` before any select has run this process.

        Held here rather than re-read from the binding store on every broadcast: the envelope
        (:func:`tcip_web.app.state_snapshot_message`) reads this in-process value on the event
        loop, and :meth:`set_binding_generation` is the only writer, called by the same select
        that wrote the binding record.
        """
        return self._binding_generation

    def set_binding_generation(self, generation: int) -> None:
        self._binding_generation = generation

    def refresh_binding_generation_from_record(self) -> None:
        """Re-read ``canvas_open_binding`` and adopt whatever generation it currently names.

        The in-process value :meth:`set_binding_generation` writes is memory only, so nothing
        moves it when the record changes for a reason outside this process's own select: a
        deleted record, an export/adopt pass, or a second web process's own bump. Call this at
        startup and before every WS connect-time replay (never per broadcast, which stays
        memory-fed) so a rendezvous with a browser always answers from the durable record rather
        than an in-memory value that a change like that would otherwise leave stale forever. A
        read failure degrades to ``None``, the same answer an absent record gives, since either
        way this process has nothing current to report.
        """
        try:
            binding = read(canvas_open_binding_key(create=False), default=None)
        except (OSError, StoreError):
            logger.exception("Could not read the canvas-open binding record")
            binding = None
        self._binding_generation = binding.get("generation") if binding is not None else None

    def open_project(self, project_root: Path) -> bool:
        """Make ``project_root`` the open project and load its persisted snapshot.

        ``project_root`` is the path the selection route's guard returned. Returns whether a
        snapshot was loaded (:meth:`load_from_disk`).
        """
        self._project_root = project_root
        return self.load_from_disk(project_root)

    def close_project(self) -> None:
        """Forget the open project; nothing persists until another is opened."""
        self._project_root = None

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

    async def replace(self, new_state: GuiState) -> None:
        """Replace the entire state (used on project-load)."""
        async with self._lock:
            self._state = new_state
            self._version += 1
            self._schedule_save()

    async def mutate(self, mutation: dict[str, Any]) -> None:
        """Apply a partial mutation, validated, and schedule persistence.

        ``mutation`` is a shallow dict of top-level field names; a nested field is set by
        passing a fully built model instance or a complete dict for it, never a partial one
        (a partial nested dict is refused by the nested field's own validation). The merged
        result is validated through :class:`GuiState` before it is held: a mutation that does
        not validate raises :class:`GuiMutationInvalid` naming the field, and nothing is held
        or scheduled.
        """
        async with self._lock:
            dumped = {
                k: v.model_dump(mode="json") if isinstance(v, BaseModel) else v
                for k, v in mutation.items()
            }
            merged = {**self._state.model_dump(mode="json"), **dumped}
            try:
                new_state = GuiState.model_validate(merged)
            except ValidationError as exc:
                raise GuiMutationInvalid(str(exc)) from exc
            self._state = new_state
            self._version += 1
            payload = {"state": self.snapshot(), "version": self._version}
            self._schedule_save()
        # Notify subscribers outside the lock so they can do I/O.
        for cb in list(self._subscribers):
            try:
                await cb(payload)
            except Exception:
                logger.exception("state subscriber failed")

    def _schedule_save(self) -> None:
        if self._project_root is None:
            return  # No project open → nothing to persist
        if self._save_task and not self._save_task.done():
            return  # Debounce: coalesce into the pending save
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Not in an event loop (e.g. called from a sync test): flush now
            self._flush_sync()
            return
        self._save_task = loop.create_task(self._save_after_debounce())

    async def _save_after_debounce(self) -> None:
        try:
            await asyncio.sleep(PERSIST_DEBOUNCE_SECONDS)
            await asyncio.to_thread(self._flush_sync)
        except Exception:
            logger.exception("Failed to persist GUI state")

    def _flush_sync(self) -> None:
        # Read at flush time, not schedule time: a project switch inside the debounce window must
        # not write the new snapshot into the previous project's gui.json.
        root = self._project_root
        if root is None:
            return
        try:
            replace(gui_snapshot_key(str(root)), self.snapshot())
        except (OSError, StoreError):
            # A snapshot that cannot be written must not fail the mutation that scheduled it,
            # so this is logged with the project it belongs to rather than raised.
            logger.exception("Could not write the GUI snapshot for %s", root)

    def load_from_disk(self, project_root: Path) -> bool:
        """Load a previous snapshot from ``<project_root>/.tcip/state/gui.json``.

        Returns ``True`` if a snapshot was loaded, ``False`` otherwise. Called when
        a project becomes known (dataset select) so backend state survives a restart.
        An absent snapshot is the ordinary first-open answer and returns quietly; one that
        exists and will not decode is logged, because that is a project losing state it had.
        A snapshot that will not decode also drops the state held in memory back to
        ``GuiState()`` defaults: without this, the previously open project's state would stay
        live under the new project's root and the next mutation would flush it into the new
        project's own gui.json.
        """
        try:
            data = read(gui_snapshot_key(project_root), default=None)
            if data is None:
                return False
            self._state = GuiState.model_validate(data)
            self._version += 1
            return True
        except Exception:
            logger.exception("Could not load the GUI snapshot for %s", project_root)
            self._state = GuiState()
            return False


# Module-level singleton; the FastAPI app imports this.
store = StateStore()
