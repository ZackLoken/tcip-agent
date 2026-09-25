"""HTTP client for MCP tools to push state to the tcip-web backend.

MCP tools call ``post_panel_event`` to ship a panel event to the running FastAPI GUI over HTTP,
never through a file on disk.

Every store the web package owns is declared here rather than in the web package: the backend's
port handoff, the GUI snapshot, the live-canvas pair, the canvas-open binding, the SessionEnd
learning-capture log, the async job registry, and the per-project annotation-timing stats. The tab
vocabulary (``ActiveTab``/``TAB_NAMES``) is declared here too.

Port discovery order:
  1. The port record under the workspace root: the port actually bound, so a substituted port
     (the requested one was taken) is still the one found.
  2. ``TCIP_WEB_PORT`` environment variable: a request, read only when no record parses.
  3. Default: 8765.

Host discovery:
  1. ``TCIP_WEB_HOST`` environment variable.
  2. Default: 127.0.0.1.

Connection failures are treated as soft errors: the MCP tool returns ``{"status":
"no_subscribers"}`` rather than raising.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal, get_args

import tcip_store
from tcip_store import LOG_JSON, RECORD_JSON, Key, StoreDescriptor, register_store, text_codec
from tcip_store.file_backend import RootedFileLocator

from tcip_mcp.project_paths import platform_state_root

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

_PORT_DOC = RootedFileLocator(prefix=(".tcip", "state"), suffix=".txt")
"""The backend's port handoff, one document under the workspace root."""

BACKEND_PORT_STORE = "backend_port"
_PORT_PARTS = ("web_port",)
register_store(
    StoreDescriptor(
        name=BACKEND_PORT_STORE,
        kind="record",
        key_fields=("document",),
        frozen=False,
        codec=text_codec(),
        concurrency="last_writer_wins",
        locator=_PORT_DOC,
    )
)


def backend_port_key(root: Path | str | None = None) -> Key:
    """Where the backend publishes the port it bound, for MCP tools in other processes.

    ``last_writer_wins``: one backend writes the whole value once per start and reads nothing
        first. ``root`` defaults to the workspace root, which every process on this machine
        resolves identically.
    """
    if root is None:
        from tcip_mcp import workspace

        root = workspace.workspace_root(create=False)
    return Key(BACKEND_PORT_STORE, str(Path(root).resolve()), _PORT_PARTS)


_SNAPSHOT_DOC = RootedFileLocator(prefix=(".tcip", "state"), suffix=".json")
"""The GUI snapshot, one document per project."""

GUI_SNAPSHOT_STORE = "gui_snapshot"
_SNAPSHOT_PARTS = ("gui",)
register_store(
    StoreDescriptor(
        name=GUI_SNAPSHOT_STORE,
        kind="record",
        key_fields=("document",),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="last_writer_wins",
        durable=False,
        locator=_SNAPSHOT_DOC,
    )
)


def gui_snapshot_key(project_root: str | Path) -> Key:
    """This project's persisted GUI snapshot.

    ``last_writer_wins``: the backend holds the live state in memory and writes the whole snapshot
        from it. ``durable=False``: the snapshot is rewritten on a debounce cycle.
    """
    return Key(GUI_SNAPSHOT_STORE, str(project_root), _SNAPSHOT_PARTS)


_CANVAS_DOC = RootedFileLocator(prefix=(".tcip", "state"), suffix=".json")
"""The live-canvas documents, one pair per project."""

CANVAS_META_STORE = "canvas_meta"
CANVAS_GEOMETRY_STORE = "canvas_geometry"
_META_PARTS = ("canvas_live",)
_GEOMETRY_PARTS = ("canvas_shapes",)


for _canvas_store in (CANVAS_META_STORE, CANVAS_GEOMETRY_STORE):
    # declared here rather than through a helper: a call in a function body is not an import
    register_store(
        StoreDescriptor(
            name=_canvas_store,
            kind="record",
            key_fields=("document",),
            frozen=False,
            codec=RECORD_JSON,
            concurrency="last_writer_wins",
            durable=False,
            locator=_CANVAS_DOC,
        )
    )


def canvas_meta_key(project_root: str) -> Key:
    """The small meta document every push overwrites.

    ``last_writer_wins``: each push writes the document whole from the payload it was given and
        reads nothing first. ``durable=False``: the next push repaints a lost one.
    """
    return Key(CANVAS_META_STORE, project_root, _META_PARTS)


def canvas_geometry_key(project_root: str) -> Key:
    """The display-resolved geometry a full push writes, on the same terms as the meta
    document, and written before it so a reader pairing new meta with old geometry sees an
    identity mismatch rather than a false match."""
    return Key(CANVAS_GEOMETRY_STORE, project_root, _GEOMETRY_PARTS)


_BINDING_DOC = RootedFileLocator(prefix=(".tcip", "state"), suffix=".json")
"""The canvas-open binding, one record per workspace."""

CANVAS_OPEN_BINDING_STORE = "canvas_open_binding"
_BINDING_PARTS = ("canvas_open_binding",)
register_store(
    StoreDescriptor(
        name=CANVAS_OPEN_BINDING_STORE,
        kind="record",
        key_fields=("document",),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="cas",
        locator=_BINDING_DOC,
    )
)


def canvas_open_binding_key(*, create: bool = True) -> Key:
    """Which root the GUI currently has open: ``{generation, root, project_name, issued_at,
    released}``.

    Workspace-scoped and written compare-and-set: a select bumps ``generation`` when ``root``
    changed or the current record was released, before writing a fresh, unreleased record. A
    release rewrites a record naming the project being released with ``generation + 1`` and
    ``released: True``, never deleting it. ``released`` is additive and optional, absent from every
    record a select writes. ``root`` is the server's own resolved open root, never a client string;
    ``project_name`` is the workspace project name when ``root`` is one, else ``None``. ``create``
    threads through to the workspace root.
    """
    from tcip_mcp import workspace

    return Key(CANVAS_OPEN_BINDING_STORE, str(workspace.workspace_root(create=create)), _BINDING_PARTS)


_CAPTURE_LOG = RootedFileLocator(prefix=(".tcip",), suffix=".jsonl")
"""The capture log under a root's own ``.tcip/``."""

LEARNING_CAPTURE_STORE = "learning_capture"
_CAPTURE_PARTS = ("learning_capture",)
register_store(
    StoreDescriptor(
        name=LEARNING_CAPTURE_STORE,
        kind="log",
        key_fields=("document",),
        frozen=True,
        codec=LOG_JSON,
        locator=_CAPTURE_LOG,
    )
)


def learning_capture_key(root: str | Path) -> Key:
    """The session-boundary log under ``root``."""
    return Key(LEARNING_CAPTURE_STORE, str(Path(root).resolve()), _CAPTURE_PARTS)


_REGISTRY_DOC = RootedFileLocator(prefix=(".tcip", "state"), suffix=".json")
"""One registry document per job kind, one per platform root."""

INFERENCE_JOBS = "inference_jobs"
REVIEW_PRIORITY_JOBS = "review_priority_jobs"
HPO_SWEEPS = "hpo_sweeps"

JOB_REGISTRY_DOCUMENTS: tuple[str, ...] = (INFERENCE_JOBS, REVIEW_PRIORITY_JOBS, HPO_SWEEPS)
"""Every document name a job registry persists under ``.tcip/state/<name>.json``.

The one spelling of each name: routes/inference.py, routes/review.py and routes/tuning.py each
hold their own registry constant from here rather than typing the string again. tcip-store
cannot import tcip-mcp, so the ``job_registry`` claim in ``tcip_store.layout_claims`` cannot
enumerate this tuple itself; a test asserts every name here matches one of that claim's own
templates, holding the agreement from this side.
"""

JOB_REGISTRY_STORE = "job_registry"
register_store(
    StoreDescriptor(
        name=JOB_REGISTRY_STORE,
        kind="record",
        key_fields=("registry",),
        frozen=True,
        cannot_carry_field="a top-level JSON array of entries, with no object to hold the field; "
                            "a future bump wraps this into {schema_version, entries}",
        codec=RECORD_JSON,
        concurrency="last_writer_wins",
        locator=_REGISTRY_DOC,
    )
)


def current_root() -> str:
    """This process's platform-state root, resolved: the value a job's own ``platform_root``
    field carries and :func:`job_registry_key`'s default group."""
    return str(platform_state_root().resolve())


def job_registry_key(name: str, *, root: str | Path | None = None) -> Key:
    """One job registry's persisted summaries, under ``root`` (default: :func:`current_root`).

    ``last_writer_wins``: one root's own group of a registry is written whole from the live jobs
        that carry it.
    """
    resolved = str(Path(root).resolve()) if root is not None else current_root()
    return Key(JOB_REGISTRY_STORE, resolved, (name,))


ANNOTATION_STATS_STORE = "annotation_stats"
_ANNOTATION_STATS_DOC = RootedFileLocator(prefix=(".tcip", "state"), suffix=".json")
_ANNOTATION_STATS_PARTS = ("annotation_stats",)

register_store(
    StoreDescriptor(
        name=ANNOTATION_STATS_STORE,
        kind="record",
        key_fields=("document",),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="cas",
        locator=_ANNOTATION_STATS_DOC,
    )
)


def annotation_stats_key(project_root: str) -> Key:
    """The project's per-image annotation timings and session rollups, written compare-and-swap."""
    return Key(ANNOTATION_STATS_STORE, project_root, _ANNOTATION_STATS_PARTS)


class GuiBindingUnreadable(RuntimeError):
    """The canvas-open binding record could not be read, or read as something that does not carry
    the ``root`` field a comparison needs: a store error, an OS-level failure, or a record shape it
    cannot trust.
    """


def read_canvas_binding() -> dict[str, Any] | None:
    """Read the canvas-open binding record as-is, or ``None`` when none exists yet. Raises
    :class:`GuiBindingUnreadable` when the record cannot be read.
    """
    try:
        return tcip_store.read(canvas_open_binding_key(create=False), default=None)
    except (tcip_store.StoreError, OSError) as exc:
        raise GuiBindingUnreadable(f"Could not read the canvas-open binding: {exc}") from exc


def binding_released_or_absent(binding: dict[str, Any] | None) -> bool:
    """Whether ``binding`` means nothing is open: absent, or marked ``released`` by
    :func:`tcip_mcp.project_removal.release_project_binding`.
    """
    return binding is None or bool(binding.get("released"))


def gui_binding_matches(root: str | Path) -> tuple[bool, dict[str, Any] | None]:
    """Whether the GUI's currently open project is ``root``, and the binding compared against.

    Returns ``(False, None)`` when no binding record exists at all; ``(False, binding)`` before
    ``root`` is even compared when the binding was released (:func:`binding_released_or_absent`).
    Otherwise ``(matches, binding)``. Raises :class:`GuiBindingUnreadable` when the record cannot
    be read, or reads as a mapping with no ``root`` field.
    """
    binding = read_canvas_binding()
    if binding is None:
        return False, None
    if binding_released_or_absent(binding):
        return False, binding
    try:
        bound_root = binding["root"]
    except KeyError as exc:
        raise GuiBindingUnreadable(
            f"Canvas-open binding record carries no 'root' field: {binding!r}"
        ) from exc
    matches = tcip_store.canonical_path(bound_root) == tcip_store.canonical_path(str(root))
    return matches, binding


def binding_divergence(binding: dict[str, Any] | None, own_root: str) -> dict[str, Any]:
    """Name both sides of a binding disagreement (a foreign project, an unnamed root, or no binding
    at all) and the step that converges them.

    A binding on a non-workspace root (a registered dataset or a ``TCIP_IMAGE_ROOTS`` entry), or no
    binding at all, converges only through the GUI's own (re)selection. A released binding reports
    the same as no binding at all.
    """
    from tcip_mcp import workspace

    nothing_open = binding_released_or_absent(binding)
    bound_root = binding.get("root") if binding is not None and not nothing_open else None
    bound_name = binding.get("project_name") if binding is not None and not nothing_open else None
    own_name = workspace.workspace_project_name(Path(own_root))
    if bound_name:
        converge = (
            f"activate_project({bound_name!r}) repins this process to the GUI's open project "
            "and steers the GUI through the panel-event chain"
        )
    elif bound_root:
        converge = (
            f"the GUI's open root ({bound_root}) has no workspace name for activate_project "
            "to adopt; reselect this project in the GUI instead"
        )
    else:
        converge = "nothing is open in the GUI; opening a project there creates a binding"
    return {
        "bound_project": bound_name,
        "bound_root": bound_root,
        "pinned_project": own_name,
        "pinned_root": own_root,
        "converge": converge,
    }


ActiveTab = Literal["annotate", "review", "training", "tuning", "inference", "results", "meta"]
"""The GUI's tabs: the vocabulary ``GuiState.active_tab`` holds and ``POST /api/state/tab``
validates against."""

TAB_NAMES = get_args(ActiveTab)

AnnotateMode = Literal["box", "polygon", "point", "map"]
"""The Annotate canvas's tool modes: the vocabulary ``tcip_web.state.GuiState.mode`` holds. The
first three draw; ``map`` navigates the coverage lattice (a click opens a cell's tile) and
authors nothing. Declared here, not in ``tcip_web``, for the same reason as ``ActiveTab``: the
agent's own ``focus_human_attention`` tool validates a caller-supplied mode against this vocabulary and cannot
import ``tcip_web``, so the vocabulary is the protocol's, and ``tcip_web.state`` imports it."""

ANNOTATE_MODES = get_args(AnnotateMode)

# One panel per GUI tab, plus "app" for steering the GUI itself (open a project, focus a tab).
# The pusher and the receiver both validate against this one set, so neither drifts apart.
VALID_PANELS = frozenset(TAB_NAMES) | {"app"}

# The event types the platform's own tool-driven emitters send; ``push_panel_event`` accepts any
# caller-supplied type beyond this set, so this is not the full panel-event vocabulary.
PANEL_EVENT_LABELS_WRITTEN = "labels_written"
PANEL_EVENT_ANNOTATE_FOCUS = "annotate_focus"
PANEL_EVENT_REVIEW_FOCUS = "review_focus"
PANEL_EVENT_ACTIVE_PROJECT_CHANGED = "active_project_changed"
PANEL_EVENT_CANVAS_STATE_REQUEST = "canvas_state_request"

PLATFORM_PANEL_EVENTS = (
    PANEL_EVENT_LABELS_WRITTEN,
    PANEL_EVENT_ANNOTATE_FOCUS,
    PANEL_EVENT_REVIEW_FOCUS,
    PANEL_EVENT_ACTIVE_PROJECT_CHANGED,
    PANEL_EVENT_CANVAS_STATE_REQUEST,
)


def resolve_web_host() -> str:
    return os.environ.get("TCIP_WEB_HOST", DEFAULT_HOST)


def resolve_web_port() -> int:
    """Return the port the FastAPI backend is listening on, in the order the module docstring
    gives: the record under the workspace root, then ``TCIP_WEB_PORT``, then the default. An absent
    record and an unparseable one both fall through rather than raising.
    """
    from tcip_mcp import workspace

    recorded = tcip_store.read(backend_port_key(workspace.workspace_root(create=False)), default=None)
    if recorded is not None:
        try:
            return int(recorded.strip())
        except ValueError:
            logger.warning("Cannot parse recorded port %r; using default", recorded)

    env = os.environ.get("TCIP_WEB_PORT")
    if env:
        try:
            return int(env)
        except ValueError:
            logger.warning("TCIP_WEB_PORT=%r is not an integer; falling back", env)

    return DEFAULT_PORT


def backend_url(path: str) -> str:
    """Build a full URL to the tcip-web backend for the given path."""
    host = resolve_web_host()
    port = resolve_web_port()
    if not path.startswith("/"):
        path = "/" + path
    return f"http://{host}:{port}{path}"


def post_panel_event(
    panel: str,
    event_type: str,
    data: dict[str, Any],
    *,
    timeout: float = 2.0,
) -> dict[str, Any]:
    """POST a panel event to the running tcip-web backend.

    Every return carries a ``delivered`` bool so callers don't mistake "backend down"
    for success. Returns one of:
      * ``{"status": "ok", "delivered": True, "response": ..., ...}`` on 2xx response, where
        ``response`` is the parsed JSON body (``None`` for a body that does not decode as JSON).
      * ``{"status": "no_subscribers", "delivered": False, ...}`` if the backend is down.
      * ``{"error": ..., "delivered": False, ...}`` on any HTTP/serialization failure.
    """
    import json
    import urllib.error
    import urllib.request

    # Hermetic under pytest: focus/web tests must never steer a live GUI session to ephemeral
    # fixture paths (the browser then 404s on deleted tmp dirs). Tests that exercise real
    # delivery opt back in with TCIP_ALLOW_PANEL_EVENTS=1.
    if os.environ.get("PYTEST_CURRENT_TEST") and not os.environ.get("TCIP_ALLOW_PANEL_EVENTS"):
        return {"status": "suppressed_under_pytest", "delivered": False, "url": ""}

    url = backend_url(f"/api/events/{panel}")
    payload = json.dumps({"panel": panel, "event_type": event_type, "data": data}).encode("utf-8")
    from tcip_mcp import agent_identity

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
    except urllib.error.URLError as exc:
        # ConnectionRefusedError or similar -> backend not running
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, (ConnectionRefusedError, OSError)):
            return {"status": "no_subscribers", "delivered": False, "url": url}
        return {"error": f"URL error: {reason}", "delivered": False, "url": url}
    except Exception as exc:  # pragma: no cover
        logger.exception("post_panel_event failed")
        return {"error": str(exc), "delivered": False, "url": url}
