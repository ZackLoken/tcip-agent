"""In-memory GUI state + debounced persistence to ``.tcip/state/gui.json``.

This is the single source of truth for live browser state. Both user actions
(from the browser) and agent actions (from MCP tools) mutate this store;
changes fan out to connected browsers via WebSocket.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

STATE_FILENAME = "gui.json"
PERSIST_DEBOUNCE_SECONDS = 0.5


# ── Pydantic state model ────────────────────────────────────────────────


class DatasetSelection(BaseModel):
    """Which dataset the GUI is currently looking at."""

    project_root: Optional[str] = None
    dataset_root: Optional[str] = None
    annotation_type: Optional[str] = None  # e.g. "catkin"
    date: Optional[str] = None             # e.g. "2-11-26"
    image_list: list[str] = Field(default_factory=list)
    current_image_index: int = 0

    # Paths resolved relative to dataset_root (images/{date}/<stem>.JPG):
    annotations_detect_dir: Optional[str] = None
    annotations_segment_dir: Optional[str] = None
    predictions_detect_dir: Optional[str] = None
    predictions_segment_dir: Optional[str] = None


class PredictionReference(BaseModel):
    """Overlay shown in the Annotate tab when user drew via Edit-from-Review."""

    type: str  # "box" | "polygon"
    coords: list[float] | list[list[float]]
    class_id: int = 0
    confidence: Optional[float] = None


class ViewState(BaseModel):
    """Pan/zoom state shared between Annotate and Review tabs."""

    scale: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0


class ReviewFilters(BaseModel):
    iou_threshold: float = 0.5
    conf_threshold: float = 0.25
    filter_type: str = "all"      # all|tp|fp|fn
    filter_class: str | int = "all"
    detection_idx: int = 0


class GuiState(BaseModel):
    """Complete GUI state — persisted to gui.json and broadcast to browsers.

    Only the *backend-authoritative* slice (``dataset``) meaningfully round-trips:
    the browser owns navigation / view / mode / class / review-filter state and
    keeps its own copy (the FE merges snapshots rather than replacing), so those
    fields here are advisory. Training-run / inference-job / class-registry state
    that used to live here was removed — the corresponding tabs own it directly.
    """

    active_tab: str = "annotate"  # annotate|review|training|tuning|inference|results|meta
    dataset: DatasetSelection = Field(default_factory=DatasetSelection)
    view: ViewState = Field(default_factory=ViewState)
    mode: str = "box"  # box|polygon
    active_class: int = 0
    review: ReviewFilters = Field(default_factory=ReviewFilters)
    pred_reference: Optional[PredictionReference] = None

    def state_dir(self) -> Optional[Path]:
        if not self.dataset.project_root:
            return None
        return Path(self.dataset.project_root) / ".tcip" / "state"


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
        """Apply a partial mutation and schedule persistence.

        ``mutation`` is a shallow dict of top-level field names; nested
        fields can be set by passing a nested dict (handled via
        :meth:`BaseModel.model_copy(update=...)`).
        """
        async with self._lock:
            self._state = self._state.model_copy(update=mutation)
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
        if self._state.state_dir() is None:
            return  # No project root → nothing to persist
        if self._save_task and not self._save_task.done():
            return  # Debounce: coalesce into the pending save
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Not in an event loop (e.g. called from a sync test) — flush now
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
        # Resolve the destination at flush time, not schedule time: if project_root
        # changed during the debounce window, the new project's snapshot must not be
        # written into the previous project's gui.json.
        state_dir = self._state.state_dir()
        if state_dir is None:
            return
        try:
            from tcip_mcp.utils.atomic_io import atomic_write_json

            state_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(state_dir / STATE_FILENAME, self.snapshot())
        except OSError:
            logger.exception("Could not write %s", state_dir / STATE_FILENAME)

    def load_from_disk(self, project_root: Path) -> bool:
        """Load a previous snapshot from ``<project_root>/.tcip/state/gui.json``.

        Returns ``True`` if a snapshot was loaded, ``False`` otherwise. Called when
        a project becomes known (dataset select) so backend state survives a restart.
        """
        path = Path(project_root) / ".tcip" / "state" / STATE_FILENAME
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._state = GuiState.model_validate(data)
            self._version += 1
            return True
        except Exception:
            logger.exception("Could not load GUI state from %s", path)
            return False


# Module-level singleton; the FastAPI app imports this.
store = StateStore()
