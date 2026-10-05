"""HTTP client for MCP tools to push state to the tcip-web backend (``post_panel_event``), and the
declarations of the stores, the GUI state shape and the tab vocabulary (``ActiveTab``) the web
package owns.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal, Optional, get_args

import tcip_store
from pydantic import BaseModel, ConfigDict, Field
from tcip_store import Key

logger = logging.getLogger(__name__)

LOOPBACK_HOST = "127.0.0.1"
"""The loopback address every server the platform starts binds and is reached at: the backend,
TensorBoard and Ray's dashboard."""


def free_port(requested: int) -> int:
    """``requested`` if it is free on :data:`LOOPBACK_HOST`, else a free port the OS assigns."""
    import socket

    for candidate in (requested, 0):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((LOOPBACK_HOST, candidate))
                return s.getsockname()[1]
        except OSError:
            continue
    raise OSError(f"no port could be bound on {LOOPBACK_HOST}")

BACKEND_PORT_STORE = "backend_port"
_PORT_PARTS = ("web_port",)


def backend_port_key(workspace: Path) -> Key:
    """The port the backend serving ``workspace`` bound, as a JSON string."""
    return Key(BACKEND_PORT_STORE, str(Path(workspace).resolve()), _PORT_PARTS)


GUI_SNAPSHOT_STORE = "gui_snapshot"
_SNAPSHOT_PARTS = ("gui",)


def gui_snapshot_key(project: str | Path) -> Key:
    """This project's persisted GUI snapshot."""
    return Key(GUI_SNAPSHOT_STORE, str(project), _SNAPSHOT_PARTS)


CANVAS_META_STORE = "canvas_meta"
CANVAS_GEOMETRY_STORE = "canvas_geometry"
_META_PARTS = ("canvas_live",)
_GEOMETRY_PARTS = ("canvas_shapes",)


def canvas_meta_key(project: str) -> Key:
    """The canvas meta document a push writes."""
    return Key(CANVAS_META_STORE, project, _META_PARTS)


def canvas_geometry_key(project: str) -> Key:
    """The display-resolved geometry a full push writes."""
    return Key(CANVAS_GEOMETRY_STORE, project, _GEOMETRY_PARTS)


ANNOTATION_STATS_STORE = "annotation_stats"
_ANNOTATION_STATS_PARTS = ("annotation_stats",)


def annotation_stats_key(project: str) -> Key:
    """The project's per-image annotation timings and session rollups, written in one
    transaction."""
    return Key(ANNOTATION_STATS_STORE, project, _ANNOTATION_STATS_PARTS)


ActiveTab = Literal["setup", "annotate", "training", "inference", "results", "meta"]
"""The GUI's tabs: the vocabulary ``GuiState.active_tab`` holds and ``POST /api/state/tab`` and
the canvas push validate against."""

TAB_NAMES = get_args(ActiveTab)

AnnotateMode = Literal["box", "polygon", "point"]
"""The Annotate canvas's drawing modes: the vocabulary :attr:`GuiState.mode` holds."""

ANNOTATE_MODES = get_args(AnnotateMode)

_TRANSPORT = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)
"""The GUI state models' config: no field beyond the declared ones, and every field present in
what they serialize to."""


class DatasetSelection(BaseModel):
    """Which dataset the GUI is looking at inside the open project: the capture, its image list,
    and the name of the bucket under the dataset root whose proposals the canvas shows."""

    model_config = _TRANSPORT

    dataset_root: Optional[str] = None
    subject: Optional[str] = None
    date: Optional[str] = None
    image_list: list[str] = Field(default_factory=list)
    current_image_index: int = 0
    images_dir: Optional[str] = None
    bucket: Optional[str] = None


class ViewState(BaseModel):
    """The Annotate canvas's pan/zoom state."""

    model_config = _TRANSPORT

    scale: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0


class _GuiFields(BaseModel):
    """The GUI state's fields other than its dataset, the same in the broadcast and the
    persisted form."""

    active_tab: ActiveTab = "annotate"
    view: ViewState = Field(default_factory=ViewState)
    mode: AnnotateMode = "box"
    active_subject: Optional[str] = None


class GuiState(_GuiFields):
    """The GUI state the backend holds for its open project and broadcasts to browsers; only
    ``dataset`` is the backend's own, the rest advisory."""

    model_config = _TRANSPORT

    dataset: DatasetSelection = Field(default_factory=DatasetSelection)


class _DatasetChoice(BaseModel):
    """The chosen part of a selection, the only part the GUI snapshot holds; ``dataset_root``
    spelled by :func:`tcip_mcp.registry_paths.stored_path` against the project."""

    model_config = ConfigDict(extra="forbid")

    dataset_root: str
    subject: Optional[str]
    date: str
    bucket: Optional[str]
    current_image_index: int


class _PersistedGuiState(_GuiFields):
    """The GUI snapshot's whole shape: :class:`GuiState` with its dataset held as the choice."""

    model_config = ConfigDict(extra="forbid")

    dataset: Optional[_DatasetChoice]


def require_whole(model: BaseModel, where: str) -> None:
    """Refuse (``ValueError``, naming it under ``where``) a field of ``model``, or of a model
    nested in it, that the input it was validated from did not state: a GUI record decodes as its
    producer's whole shape, never with a default standing in for a missing field."""
    missing = sorted(set(type(model).model_fields) - model.model_fields_set)
    if missing:
        raise ValueError(f"{where} states no {missing}")
    for name in type(model).model_fields:
        value = getattr(model, name)
        if isinstance(value, BaseModel):
            require_whole(value, f"{where}.{name}")


def selection_for(dataset_root: Path, subject: Optional[str], date: str,
                  bucket: Optional[str], current_image_index: int) -> DatasetSelection:
    """The selection a choice names: the image list of capture ``date`` and the name of the
    bucket under ``dataset_root`` whose proposals the canvas shows. ``current_image_index`` is
    clamped to the list. Raises ``AmbiguousImageStem`` for a capture holding two images of one
    stem, and ``ValueError`` for a ``date`` that is no capture name."""
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.pipelines.image_utils import logical_images_by_name

    images_dir = image_dir(dataset_root, date)
    image_list = list(logical_images_by_name(images_dir))
    index = max(0, min(current_image_index, len(image_list) - 1)) if image_list else 0
    return DatasetSelection(
        dataset_root=str(dataset_root),
        subject=subject,
        date=date,
        image_list=image_list,
        current_image_index=index,
        images_dir=str(images_dir),
        bucket=bucket or None,
    )


def current_image(selection: DatasetSelection) -> Optional[str]:
    """The image the selection's index names, under its images directory, or ``None`` when it
    names none."""
    if selection.images_dir is None or not selection.image_list:
        return None
    return str(Path(selection.images_dir) / selection.image_list[selection.current_image_index])


def write_gui_snapshot(project: Path, state: GuiState) -> None:
    """Persist ``state`` as ``project``'s GUI snapshot, its dataset held as the choice."""
    from tcip_mcp.registry_paths import stored_path

    dataset = state.dataset
    choice = None if dataset.dataset_root is None or dataset.date is None else _DatasetChoice(
        dataset_root=stored_path(dataset.dataset_root, project), subject=dataset.subject,
        date=dataset.date, current_image_index=dataset.current_image_index,
        bucket=dataset.bucket)
    document = _PersistedGuiState(**{**state.model_dump(exclude={"dataset"}), "dataset": choice})
    tcip_store.replace(gui_snapshot_key(project), document.model_dump(mode="json"))


def read_gui_snapshot(project: Path) -> Optional[GuiState]:
    """``project``'s persisted GUI state, its selection rebuilt through :func:`selection_for`, or
    ``None`` when it has none. A snapshot that does not decode as its whole shape raises
    (``ValidationError``, ``ValueError`` from :func:`require_whole`, or the store's own error), as
    does a selection that will not build."""
    from tcip_mcp.registry_paths import resolved_registry_path

    raw = tcip_store.read(gui_snapshot_key(project), default=None)
    if raw is None:
        return None
    persisted = _PersistedGuiState.model_validate(raw)
    require_whole(persisted, "the GUI snapshot")
    choice = persisted.dataset
    dataset = DatasetSelection() if choice is None else selection_for(
        resolved_registry_path(project, choice.dataset_root), choice.subject, choice.date,
        choice.bucket, choice.current_image_index)
    return GuiState(**{**persisted.model_dump(exclude={"dataset"}), "dataset": dataset})

# One panel per GUI tab, plus "app" for steering the GUI itself (open a project, focus a tab).
# The pusher and the receiver both validate against this one set, so neither drifts apart.
VALID_PANELS = frozenset(TAB_NAMES) | {"app"}

# The event types the platform's own tool-driven emitters send; ``push_panel_event`` accepts any
# caller-supplied type beyond this set, so this is not the full panel-event vocabulary.
PANEL_EVENT_LABELS_WRITTEN = "labels_written"
PANEL_EVENT_ANNOTATE_FOCUS = "annotate_focus"
PANEL_EVENT_CANVAS_STATE_REQUEST = "canvas_state_request"

PLATFORM_PANEL_EVENTS = (
    PANEL_EVENT_LABELS_WRITTEN,
    PANEL_EVENT_ANNOTATE_FOCUS,
    PANEL_EVENT_CANVAS_STATE_REQUEST,
)


class NoBackendPort(LookupError):
    """No backend has recorded the port it serves a workspace on."""


def resolve_web_port(workspace: Path) -> int:
    """The port the backend serving ``workspace`` recorded under it. Refuses
    (:class:`NoBackendPort`) when none is recorded, and (``ValueError``) a recorded port that is
    not an integer."""
    recorded = tcip_store.read(backend_port_key(workspace), default=None)
    if recorded is None:
        raise NoBackendPort(f"no backend has recorded a port under {workspace}; "
                            "`python -m tcip_web` serving it records one")
    try:
        return int(recorded)
    except ValueError as exc:
        raise ValueError(f"the backend's port under {workspace} does not read as a port: "
                         f"{exc}") from exc


def backend_url(workspace: Path, path: str) -> str:
    """Build a full URL to the tcip-web backend serving ``workspace`` for the given path."""
    port = resolve_web_port(workspace)
    if not path.startswith("/"):
        path = "/" + path
    return f"http://{LOOPBACK_HOST}:{port}{path}"


def post_panel_event(
    project: Path,
    workspace: Path,
    panel: str,
    event_type: str,
    data: dict[str, Any],
    *,
    timeout: float = 2.0,
) -> dict[str, Any]:
    """POST a panel event for ``project`` to the tcip-web backend serving ``workspace``, delivered
    only when that project is the one it has open. Every answer carries ``delivered``: on a 2xx,
    ``status`` ``"ok"`` and ``response`` (the JSON body, ``None`` when it does not decode); with
    the backend down, ``status`` ``"no_subscribers"``; otherwise ``error`` naming why, with
    ``open_project_id`` when another project, or none, is open."""
    import json
    import urllib.error
    import urllib.request

    # Hermetic under pytest: focus/web tests must never steer a live GUI session to ephemeral
    # fixture paths (the browser then 404s on deleted tmp dirs). Tests that exercise real
    # delivery opt back in with TCIP_ALLOW_PANEL_EVENTS=1.
    if os.environ.get("PYTEST_CURRENT_TEST") and not os.environ.get("TCIP_ALLOW_PANEL_EVENTS"):
        return {"status": "suppressed_under_pytest", "delivered": False, "url": ""}

    from tcip_mcp import agent_identity
    from tcip_mcp.project_record import read_record

    try:
        url = backend_url(workspace, f"/api/events/{panel}")
    except NoBackendPort as exc:
        return {"status": "no_subscribers", "delivered": False, "url": "", "error": str(exc)}
    payload = json.dumps({"panel": panel, "event_type": event_type, "data": data,
                          "project_id": read_record(project)["id"]}).encode("utf-8")

    # The pushing harness and session, as headers, so the backend can say who steered the GUI.
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", **agent_identity.http_headers()},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = getattr(resp, "status", 200)
            body = resp.read()
            if 200 <= code < 300:
                try:
                    parsed = json.loads(body)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    parsed = None
                return {"status": "ok", "delivered": True, "url": url, "response": parsed}
            return {"error": f"backend returned HTTP {code}", "delivered": False, "url": url}
    except urllib.error.HTTPError as exc:
        # The backend's own answer, e.g. the project it has open when that is not this one.
        try:
            detail = json.loads(exc.read()).get("detail")
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
            detail = None
        if isinstance(detail, dict):
            return {**detail, "delivered": False, "url": url}
        return {"error": f"backend returned HTTP {exc.code}", "delivered": False, "url": url}
    except urllib.error.URLError as exc:
        # ConnectionRefusedError or similar -> backend not running
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, (ConnectionRefusedError, OSError)):
            return {"status": "no_subscribers", "delivered": False, "url": url}
        return {"error": f"URL error: {reason}", "delivered": False, "url": url}
    except Exception as exc:  # pragma: no cover
        logger.exception("post_panel_event failed")
        return {"error": str(exc), "delivered": False, "url": url}
