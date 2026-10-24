"""HTTP client pushing state to the tcip-web backend, and the web package's shared declarations.

A push is an ``httpx.post`` to a ``backend_url``, the panel event's through ``post_panel_event``;
the declarations are the stores, the port the backend records, the GUI state shape and the tab
vocabulary (``ActiveTab``) the web package owns.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, JsonValue, PositiveFloat

import tcip_store
from tcip_annotation.viz import CanvasShape
from tcip_store import Key

if TYPE_CHECKING:
    from tcip_mcp.workspace import BoundProject

logger = logging.getLogger(__name__)

LOOPBACK_HOST = "127.0.0.1"
"""The IPv4 loopback address."""


def free_port(requested: int) -> int:
    """Return ``requested`` if free on :data:`LOOPBACK_HOST`, else a free port the OS assigns."""
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
    """Return the key of the port the backend serving ``workspace`` bound, as a JSON string."""
    return Key(BACKEND_PORT_STORE, str(workspace), _PORT_PARTS)


def record_backend_port(workspace: Path, port: int) -> None:
    """Record ``port`` as the one the backend serving ``workspace`` listens on."""
    tcip_store.replace(backend_port_key(workspace), str(port))


GUI_SNAPSHOT_STORE = "gui_snapshot"
_SNAPSHOT_PARTS = ("gui",)


def gui_snapshot_key(project: str | Path) -> Key:
    """Return the key of this project's persisted GUI snapshot."""
    return Key(GUI_SNAPSHOT_STORE, str(project), _SNAPSHOT_PARTS)


CANVAS_META_STORE = "canvas_meta"
CANVAS_GEOMETRY_STORE = "canvas_geometry"
_META_PARTS = ("canvas_live",)
_GEOMETRY_PARTS = ("canvas_shapes",)


def canvas_meta_key(project: str) -> Key:
    """Return the key of the project's canvas meta document."""
    return Key(CANVAS_META_STORE, project, _META_PARTS)


def canvas_geometry_key(project: str) -> Key:
    """Return the key of the project's display-resolved canvas geometry."""
    return Key(CANVAS_GEOMETRY_STORE, project, _GEOMETRY_PARTS)


ANNOTATION_STATS_STORE = "annotation_stats"
_ANNOTATION_STATS_PARTS = ("annotation_stats",)


def annotation_stats_key(project: str) -> Key:
    """Return the key of the project's per-image annotation timings and session rollups."""
    return Key(ANNOTATION_STATS_STORE, project, _ANNOTATION_STATS_PARTS)


ActiveTab = Literal[
    "setup", "annotate", "training", "inference", "results", "meta"
]
"""The GUI's tabs: the vocabulary ``GuiState.active_tab`` holds."""

TAB_NAMES = get_args(ActiveTab)

AnnotateMode = Literal["box", "polygon", "point"]
"""The Annotate canvas's drawing modes: the vocabulary :attr:`GuiState.mode` holds."""

ANNOTATE_MODES = get_args(AnnotateMode)


class Viewport(BaseModel):
    """The image region the GUI's canvas shows, in image coordinates, and its zoom ``scale``."""

    model_config = ConfigDict(extra="forbid")

    x: float
    y: float
    w: PositiveFloat
    h: PositiveFloat
    scale: float


class CanvasView(BaseModel):
    """What the GUI's canvas showed: tab, image, size, viewport, mode, person, classes, counts.

    ``viewport`` is ``None`` while the canvas shows no region, ``active_subject`` while no subject
    is active, and ``cut_armed`` is whether the Annotate cut tool is armed.
    """

    tab: ActiveTab
    image_path: str
    image: str
    img_width: int
    img_height: int
    viewport: Viewport | None
    mode: AnnotateMode
    active_subject: str | None
    cut_armed: bool
    dirty: bool
    user: str
    classes: list[dict[str, JsonValue]]  # [{name, color}]
    counts: dict[str, int]


class CanvasMeta(CanvasView):
    """The canvas meta record: the latest push's view, stamped with the instant it arrived."""

    model_config = ConfigDict(extra="forbid")

    received_at: str


class CanvasGeometry(BaseModel):
    """The canvas geometry record: a full push's display-resolved shapes and their identity.

    The identity is the ``image_path`` and ``tab`` the shapes were drawn on.
    """

    model_config = ConfigDict(extra="forbid")

    image_path: str
    tab: ActiveTab
    shapes: list[CanvasShape]
    received_at: str


def read_canvas_meta(project: str) -> CanvasMeta | None:
    """Return ``project``'s canvas meta record, or ``None`` when no push has landed.

    A record that does not decode as its shape raises ``ValidationError``.
    """
    raw = tcip_store.read(canvas_meta_key(project), default=None)
    return None if raw is None else CanvasMeta.model_validate(raw)


def read_canvas_geometry(project: str) -> CanvasGeometry | None:
    """Return ``project``'s canvas geometry record, or ``None`` when no full push has landed.

    A record that does not decode as its shape raises ``ValidationError``.
    """
    raw = tcip_store.read(canvas_geometry_key(project), default=None)
    return None if raw is None else CanvasGeometry.model_validate(raw)


class DatasetSelection(BaseModel):
    """Which dataset the GUI is looking at inside the open project.

    It holds the capture, its image list, and the name of a bucket under the dataset root.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_root: str | None
    subject: str | None
    date: str | None
    image_list: list[str]
    current_image_index: int
    images_dir: str | None
    bucket: str | None


class ViewState(BaseModel):
    """The Annotate canvas's pan/zoom state."""

    model_config = ConfigDict(extra="forbid")

    scale: float
    offset_x: float
    offset_y: float


class _GuiFields(BaseModel):
    """The GUI state's fields other than its dataset."""

    model_config = ConfigDict(extra="forbid")

    active_tab: ActiveTab
    view: ViewState
    mode: AnnotateMode
    active_subject: str | None


class GuiState(_GuiFields):
    """The GUI state the backend holds for its open project.

    Only ``dataset`` is the backend's own, the rest advisory.
    """

    dataset: DatasetSelection


GUI_STATE_DEFAULTS = GuiState(
    active_tab="annotate",
    view=ViewState(scale=1.0, offset_x=0.0, offset_y=0.0),
    mode="box",
    active_subject=None,
    dataset=DatasetSelection(
        dataset_root=None,
        subject=None,
        date=None,
        image_list=[],
        current_image_index=0,
        images_dir=None,
        bucket=None,
    ),
)
"""The GUI state a project holds before it has any of its own: the Annotate tab in box mode at
unit zoom, no subject active and no dataset chosen."""


class _DatasetChoice(BaseModel):
    """The chosen part of a selection, the only part the GUI snapshot holds.

    ``dataset_root`` is spelled by :func:`tcip_mcp.registry_paths.stored_path` against the
    project.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_root: str
    subject: str | None
    date: str
    bucket: str | None
    current_image_index: int


class _PersistedGuiState(_GuiFields):
    """The GUI snapshot's whole shape: :class:`GuiState` with its dataset held as the choice."""

    dataset: _DatasetChoice | None


def selection_for(
    dataset_root: Path,
    subject: str | None,
    date: str,
    bucket: str | None,
    current_image_index: int,
) -> DatasetSelection:
    """Return the selection a choice names.

    It holds the image list of capture ``date`` and the name of a bucket under ``dataset_root``.
    ``current_image_index`` is clamped to the list. Raises
    ``AmbiguousImageStemError`` for a capture holding two images of one stem, and ``ValueError``
    for a ``date`` that is no capture name.
    """
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.pipelines.image_utils import logical_images_by_name

    images_dir = image_dir(dataset_root, date)
    image_list = list(logical_images_by_name(images_dir))
    index = (
        max(0, min(current_image_index, len(image_list) - 1))
        if image_list
        else 0
    )
    return DatasetSelection(
        dataset_root=str(dataset_root),
        subject=subject,
        date=date,
        image_list=image_list,
        current_image_index=index,
        images_dir=str(images_dir),
        bucket=bucket or None,
    )


def current_image(selection: DatasetSelection) -> str | None:
    """Return the image the selection's index names, under its images directory, or ``None``.

    ``None`` when the selection has no images directory or no images.
    """
    if selection.images_dir is None or not selection.image_list:
        return None
    return str(
        Path(selection.images_dir)
        / selection.image_list[selection.current_image_index]
    )


def write_gui_snapshot(project: Path, state: GuiState) -> None:
    """Persist ``state`` as ``project``'s GUI snapshot, its dataset held as the choice."""
    from tcip_mcp.registry_paths import stored_path

    dataset = state.dataset
    choice = (
        None
        if dataset.dataset_root is None or dataset.date is None
        else _DatasetChoice(
            dataset_root=stored_path(dataset.dataset_root, project),
            subject=dataset.subject,
            date=dataset.date,
            current_image_index=dataset.current_image_index,
            bucket=dataset.bucket,
        )
    )
    document = _PersistedGuiState(
        **{**state.model_dump(exclude={"dataset"}), "dataset": choice}
    )
    tcip_store.replace(
        gui_snapshot_key(project), document.model_dump(mode="json")
    )


def read_gui_snapshot(project: Path) -> GuiState | None:
    """Return ``project``'s persisted GUI state, or ``None`` when it has none.

    Its selection is rebuilt through :func:`selection_for`. A snapshot that does not decode as its
    shape raises (``ValidationError``, or the store's own error), as does a selection that will
    not build.
    """
    from tcip_mcp.registry_paths import resolved_registry_path

    raw = tcip_store.read(gui_snapshot_key(project), default=None)
    if raw is None:
        return None
    persisted = _PersistedGuiState.model_validate(raw)
    choice = persisted.dataset
    dataset = (
        GUI_STATE_DEFAULTS.dataset
        if choice is None
        else selection_for(
            resolved_registry_path(project, choice.dataset_root),
            choice.subject,
            choice.date,
            choice.bucket,
            choice.current_image_index,
        )
    )
    return GuiState(
        **{**persisted.model_dump(exclude={"dataset"}), "dataset": dataset}
    )


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


class NoBackendPortError(LookupError):
    """No backend has recorded the port it serves a workspace on."""


def resolve_web_port(workspace: Path) -> int:
    """Return the port the backend serving ``workspace`` recorded under it.

    Refuses (:class:`NoBackendPortError`) when none is recorded, and
    (``pydantic.ValidationError``) a recorded port that does not validate as an integer.
    """
    from pydantic import TypeAdapter

    recorded = tcip_store.read(backend_port_key(workspace), default=None)
    if recorded is None:
        raise NoBackendPortError(
            f"no backend has recorded a port under {workspace}; "
            "`python -m tcip_web` serving it records one"
        )
    return TypeAdapter(int).validate_python(recorded)


def backend_url(workspace: Path, path: str) -> str:
    """Build a full URL to the tcip-web backend serving ``workspace`` for the given path."""
    port = resolve_web_port(workspace)
    if not path.startswith("/"):
        path = "/" + path
    return f"http://{LOOPBACK_HOST}:{port}{path}"


def post_panel_event(
    bound: BoundProject,
    panel: str,
    event_type: str,
    data: dict[str, Any],
    *,
    timeout: float = 2.0,
) -> dict[str, Any]:
    """POST a panel event for the project ``bound`` names to the backend serving its workspace.

    The event is delivered only when that project is the one the backend has open. Every answer
    carries ``delivered``: on a 2xx, ``status`` ``"ok"`` and ``response`` (the JSON body,
    ``None`` when it does not decode); with no port recorded, the connection refused or the
    request timed out, ``status`` ``"no_subscribers"``;
    otherwise ``error`` naming why: with ``open_project_id`` when another project is open, the
    backend's none-open message when none is.
    """
    import httpx

    # Hermetic under pytest: focus/web tests must never steer a live GUI session to ephemeral
    # fixture paths (the browser then 404s on deleted tmp dirs). Tests that exercise real
    # delivery opt back in with TCIP_ALLOW_PANEL_EVENTS=1.
    if os.environ.get("PYTEST_CURRENT_TEST") and not os.environ.get(
        "TCIP_ALLOW_PANEL_EVENTS"
    ):
        return {
            "status": "suppressed_under_pytest",
            "delivered": False,
            "url": "",
        }

    from tcip_mcp import agent_identity

    try:
        url = backend_url(bound.workspace, f"/api/events/{panel}")
    except NoBackendPortError as exc:
        return {
            "status": "no_subscribers",
            "delivered": False,
            "url": "",
            "error": str(exc),
        }
    try:
        # The pushing harness and session, as headers, so the backend can say who steered the GUI.
        answer = httpx.post(
            url,
            json={
                "panel": panel,
                "event_type": event_type,
                "data": data,
                "project_id": bound.id,
            },
            headers=agent_identity.http_headers(),
            timeout=timeout,
        )
    except (httpx.ConnectError, httpx.TimeoutException):
        # Refused: nothing listens on the recorded port. Timed out: nothing answered in time,
        # which does not show that no backend serves. Any other failure is an error below.
        return {"status": "no_subscribers", "delivered": False, "url": url}
    except Exception as exc:
        logger.exception("post_panel_event failed")
        return {"error": str(exc), "delivered": False, "url": url}
    try:
        body = answer.json()
    except ValueError:
        body = None
    if answer.is_success:
        return {
            "status": "ok",
            "delivered": True,
            "url": url,
            "response": body,
        }
    # The backend's own answer, e.g. the project it has open when that is not this one.
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict):
        fields = {k: v for k, v in detail.items() if k != "message"}
        return {
            **fields,
            "error": detail.get("message"),
            "delivered": False,
            "url": url,
        }
    error = (
        detail
        if isinstance(detail, str)
        else f"backend returned HTTP {answer.status_code}"
    )
    return {"error": error, "delivered": False, "url": url}
