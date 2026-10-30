"""HTTP client pushing state to the tcip-web backend, and the web package's shared declarations.

A push is an ``httpx.post`` to a ``backend_url``, the panel event's through ``post_panel_event``;
the declarations are the stores, the port the backend records, the GUI state shape and the tab
vocabulary (``ActiveTab``) the web package owns.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, get_args

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    PositiveFloat,
    TypeAdapter,
    computed_field,
)

import tcip_store
from tcip_annotation.flags import ProposalTarget
from tcip_annotation.viz import CanvasShape
from tcip_mcp.dataset_layout import image_dir
from tcip_mcp.pipelines.image_utils import logical_image_name
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
    scale: PositiveFloat


class CanvasClass(BaseModel):
    """One subject the canvas draws and the color it draws it in, the GUI's own."""

    model_config = ConfigDict(extra="forbid")

    name: str
    color: str


class CanvasView(BaseModel):
    """What the GUI's canvas showed: tab, image, viewport, mode, person, classes, counts.

    ``viewport`` is ``None`` while the canvas shows no region, ``active_subject`` while no subject
    is active, and ``cut_armed`` is whether the Annotate cut tool is armed.
    """

    tab: ActiveTab
    image_path: str
    viewport: Viewport | None
    mode: AnnotateMode
    active_subject: str | None
    cut_armed: bool
    dirty: bool
    user: str
    classes: list[CanvasClass]
    counts: dict[str, int]


class CanvasMeta(CanvasView):
    """The canvas meta record: the latest push's view, stamped with the instant it arrived."""

    model_config = ConfigDict(extra="forbid")

    received_at: AwareDatetime


class CanvasGeometry(BaseModel):
    """The canvas geometry record: a full push's display-resolved shapes and their identity.

    The identity is the ``image_path`` and ``tab`` the shapes were drawn on.
    """

    model_config = ConfigDict(extra="forbid")

    image_path: str
    tab: ActiveTab
    shapes: list[CanvasShape]
    received_at: AwareDatetime


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


class CaptureChoice(BaseModel):
    """A capture chosen to look at: its dataset root, subject, capture and the bucket reviewed.

    ``subject`` and ``bucket`` are ``None`` when none is chosen.
    """

    dataset_root: str
    subject: str | None = None
    # The capture: a date folder, a literal bucket or ``UNDATED_BUCKET``.
    date: str
    # The name of the bucket to review, published under dataset_root.
    bucket: str | None = None


class DatasetChoice(CaptureChoice):
    """The chosen part of a selection, the only part the GUI snapshot holds.

    It is the capture chosen and the position of the image in view within it.
    """

    model_config = ConfigDict(extra="forbid")

    current_image_index: int


class AnnotateFocus(CaptureChoice):
    """Where the agent drives the Annotate tab: a capture, the image, the mode and the proposal.

    It is the ``annotate_focus`` panel event. ``subject`` is the one made active; ``proposal`` is
    the proposal focused, by its bucket and its index in that bucket's document for the image, or
    ``None``.
    """

    model_config = ConfigDict(extra="forbid")

    event_type: Literal["annotate_focus"] = "annotate_focus"
    subject: str
    image_index: int
    mode: AnnotateMode
    proposal: ProposalTarget | None


class CaptureImage(BaseModel):
    """One image of a selected capture: the path its bytes are read from, and its name."""

    model_config = ConfigDict(extra="forbid")

    path: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def name(self) -> str:
        """The image's name (:func:`~tcip_mcp.pipelines.image_utils.logical_image_name`)."""
        return logical_image_name(self.path)


class DatasetSelection(DatasetChoice):
    """Which capture the GUI is looking at inside the open project, its images in name order."""

    images: list[CaptureImage]


class ViewState(BaseModel):
    """The Annotate canvas's pan/zoom state."""

    model_config = ConfigDict(extra="forbid")

    scale: PositiveFloat
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

    Only ``dataset`` is the backend's own, the rest advisory; it is ``None`` while no capture is
    chosen.
    """

    dataset: DatasetSelection | None


GUI_STATE_DEFAULTS = GuiState(
    active_tab="annotate",
    view=ViewState(scale=1.0, offset_x=0.0, offset_y=0.0),
    mode="box",
    active_subject=None,
    dataset=None,
)
"""The GUI state a project holds before it has any of its own: the Annotate tab in box mode at
unit zoom, no subject active and no dataset chosen."""


class _PersistedGuiState(_GuiFields):
    """The GUI snapshot's whole shape: :class:`GuiState` with its dataset held as the choice.

    The choice's ``dataset_root`` is spelled by :func:`tcip_mcp.registry_paths.stored_path`
    against the project.
    """

    dataset: DatasetChoice | None


def selection_for(choice: DatasetChoice) -> DatasetSelection:
    """Return the selection ``choice`` names, its images the capture's in name order.

    Each image is placed at the path its scan found it at
    (:func:`~tcip_mcp.pipelines.image_utils.source_path_of`); the index is clamped to them. Raises
    ``AmbiguousImageStemError`` for a capture holding two images of one stem, and ``ValueError``
    for a ``date`` that is no capture name.
    """
    from tcip_mcp.pipelines.image_utils import (
        logical_images_by_name,
        source_path_of,
    )

    found = logical_images_by_name(image_dir(choice.dataset_root, choice.date))
    images = [
        CaptureImage(path=source_path_of(source)) for source in found.values()
    ]
    index = max(0, min(choice.current_image_index, len(images) - 1))
    return DatasetSelection(
        **dict(choice) | {"current_image_index": index}, images=images
    )


def write_gui_snapshot(project: Path, state: GuiState) -> None:
    """Persist ``state`` as ``project``'s GUI snapshot, its dataset held as the choice."""
    from tcip_mcp.registry_paths import stored_path

    selection = state.dataset
    document = _PersistedGuiState(
        **{name: getattr(state, name) for name in _GuiFields.model_fields},
        dataset=None
        if selection is None
        else DatasetChoice(
            **{name: getattr(selection, name) for name in DatasetChoice.model_fields}
            | {"dataset_root": stored_path(selection.dataset_root, project)}
        ),
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
    return GuiState(
        **{name: getattr(persisted, name) for name in _GuiFields.model_fields},
        dataset=None
        if choice is None
        else selection_for(
            choice.model_copy(
                update={
                    "dataset_root": str(
                        resolved_registry_path(project, choice.dataset_root)
                    )
                }
            )
        ),
    )


Panel = Literal[ActiveTab, "app"]
"""A GUI panel: one per tab, plus ``"app"`` for steering the GUI itself."""


class LabelsWritten(BaseModel):
    """The ``labels_written`` panel event: a process wrote the labels of the image at a path."""

    event_type: Literal["labels_written"] = "labels_written"
    image_path: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def name(self) -> str:
        """The image's name (:func:`~tcip_mcp.pipelines.image_utils.logical_image_name`)."""
        return logical_image_name(self.image_path)


class CanvasStateRequest(BaseModel):
    """The ``canvas_state_request`` panel event: the GUI pushes its canvas state now."""

    event_type: Literal["canvas_state_request"] = "canvas_state_request"


class Banner(BaseModel):
    """The ``banner`` panel event: a note ``text`` the GUI shows above the panel's tab."""

    event_type: Literal["banner"] = "banner"
    text: str


PanelEvent = Annotated[
    LabelsWritten | AnnotateFocus | CanvasStateRequest | Banner,
    Field(discriminator="event_type"),
]
"""One event pushed to a GUI panel, its payload told apart by its ``event_type``."""


class PanelPost(BaseModel):
    """A panel event as posted to the backend: the event and the project its sender acts on."""

    event: PanelEvent
    project_id: str


class ProjectNotOpenRefusal(BaseModel):
    """The backend refusing a request built for a project other than the one it has open.

    ``open_project_id`` names the project it has open.
    """

    kind: Literal["project_not_open"] = "project_not_open"
    message: str
    open_project_id: str


ProjectNotOpen = Annotated[ProjectNotOpenRefusal, Field(discriminator="kind")]
"""A :class:`ProjectNotOpenRefusal` read only from a detail whose ``kind`` names it."""

PanelEventReason = Annotated[
    ProjectNotOpen | JsonValue, Field(union_mode="left_to_right")
]
"""Why a panel event was not taken: the backend's refusal by its ``kind``, else the detail as the
backend sent it or a sentence naming the failure."""


class PanelEventAnswer(BaseModel):
    """What posting a panel event came to: the backend's URL it went to and why it was not taken.

    ``url`` is ``None`` when no URL was resolved; ``reason`` is ``None`` once the backend accepted
    the event (never that a browser showed it), else why not.
    """

    url: str | None
    reason: PanelEventReason | None


class NoBackendPortError(LookupError):
    """No backend has recorded the port it serves a workspace on."""


def resolve_web_port(workspace: Path) -> int:
    """Return the port the backend serving ``workspace`` recorded under it.

    Refuses (:class:`NoBackendPortError`) when none is recorded, and
    (``pydantic.ValidationError``) a recorded port that does not validate as an integer.
    """
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
    panel: Panel,
    event: PanelEvent,
    *,
    timeout: float = 2.0,
) -> PanelEventAnswer:
    """POST ``event`` to ``panel`` for the project ``bound`` names, to the backend serving it.

    The event is taken only when that project is the one the backend has open. Otherwise the
    answer's ``reason`` says why not: no port recorded, nothing answering at the recorded port,
    the backend's refusal detail as it sent it (:data:`PanelEventReason`), or the failure itself.
    """
    import httpx

    # Hermetic under pytest, so a test never steers a live GUI to its fixture paths; a test
    # exercising real delivery opts back in with TCIP_ALLOW_PANEL_EVENTS=1.
    if os.environ.get("PYTEST_CURRENT_TEST") and not os.environ.get(
        "TCIP_ALLOW_PANEL_EVENTS"
    ):
        return PanelEventAnswer(
            url=None,
            reason="panel events are not posted under pytest; TCIP_ALLOW_PANEL_EVENTS=1 "
            "posts them",
        )

    from tcip_mcp import agent_identity

    try:
        url = backend_url(bound.workspace, f"/api/events/{panel}")
    except NoBackendPortError as exc:
        return PanelEventAnswer(url=None, reason=str(exc))
    try:
        # The pushing harness and session, as headers, so the backend can say who steered the GUI.
        answer = httpx.post(
            url,
            json=PanelPost(event=event, project_id=bound.id).model_dump(
                mode="json"
            ),
            headers=agent_identity.http_headers(),
            timeout=timeout,
        )
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        # Refused: nothing listens on the recorded port. Timed out: nothing answered in time,
        # which does not show that no backend serves. Any other failure is a failure below.
        return PanelEventAnswer(
            url=url, reason=f"no backend answered at the recorded port: {exc}"
        )
    except Exception as exc:
        logger.exception("post_panel_event failed")
        return PanelEventAnswer(url=url, reason=str(exc))
    if answer.is_success:
        return PanelEventAnswer(url=url, reason=None)
    try:
        body = answer.json()
    except ValueError:
        body = answer.text
    return PanelEventAnswer(
        url=url,
        reason=body.get("detail", body) if isinstance(body, dict) else body,
    )
