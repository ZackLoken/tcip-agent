"""HTTP client for MCP tools to push state to the tcip-web backend.

MCP tools call ``post_panel_event`` to ship a panel event to the running FastAPI GUI over HTTP,
never through a file on disk.

The stores the two packages share are declared here rather than in the web package: the
backend's port handoff, the GUI snapshot, and the live-canvas pair. MCP tools read all of them
and cannot import ``tcip_web``, so the web side imports the declarations from here, which is the
legal dependency direction and the same one ``VALID_PANELS`` already takes. A declaration on
each side would be two stores wearing one name, and whichever imported first would decide where
the documents land.

The tab vocabulary (``ActiveTab``/``TAB_NAMES``) lives here for the same reason: the agent's own
``focus`` tool takes a tab name over the wire, so the vocabulary is the protocol's, and
``tcip_web.state`` imports it rather than declaring its own.

Port discovery order:
  1. The port record under the workspace root: the port actually bound, so a substituted port
     (the requested one was taken) is still the one found. The workspace root, not the
     platform-state root, because it is the one location every process on this machine
     resolves identically; the platform root moves whenever a process adopts a project, which
     would otherwise strand a reader pinned to a different one.
  2. ``TCIP_WEB_PORT`` environment variable: a request, read only when no record parses. The
     record can fail to exist (a failed publication, or a backend started as bare ``uvicorn
     tcip_web.app:app --port N``, which never writes one), and the launcher keeps serving either
     way, so with no record the request is the best information there is.
  3. Default: 8765.

Host discovery:
  1. ``TCIP_WEB_HOST`` environment variable.
  2. Default: 127.0.0.1.

Connection failures are treated as soft errors: the MCP tool returns
``{"status": "no_subscribers"}`` rather than raising. This keeps agent
workflows working when the GUI is closed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal, get_args

import tcip_store
from tcip_store import RECORD_JSON, Key, StoreDescriptor, register_store, text_codec
from tcip_store.file_backend import RootedFileLocator

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
        codec=text_codec(),
        concurrency="last_writer_wins",
        locator=_PORT_DOC,
    )
)


def backend_port_key(root: Path | str | None = None) -> Key:
    """Where the backend publishes the port it bound, for MCP tools in other processes.

    ``last_writer_wins``: one backend writes the whole value once per start and reads nothing
    first. ``root`` defaults to the workspace root: the one location every process on this
    machine resolves identically, unlike the platform-state root, which moves whenever a
    process adopts a project. The writer and the reader both pass it explicitly.
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
        codec=RECORD_JSON,
        concurrency="last_writer_wins",
        durable=False,
        locator=_SNAPSHOT_DOC,
    )
)


def gui_snapshot_key(project_root: str | Path) -> Key:
    """This project's persisted GUI snapshot.

    ``last_writer_wins``: the backend holds the live state in memory and writes the whole
    snapshot from it, so the document is one process's view rather than one writers merge
    into. ``durable=False``: the snapshot is rewritten on a debounce cycle and a crash losing
    the last one costs a re-selection, not history.
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
            codec=RECORD_JSON,
            concurrency="last_writer_wins",
            durable=False,
            locator=_CANVAS_DOC,
        )
    )


def canvas_meta_key(project_root: str) -> Key:
    """The small meta document every push overwrites.

    ``last_writer_wins``: each push writes the document whole from the payload it was given
    and reads nothing first; the reader pairs meta with geometry by identity rather than by
    mutual exclusion. ``durable=False`` carries the canvas route's own stated property, that a
    crash losing the last push costs nothing because the next push repaints it.
    """
    return Key(CANVAS_META_STORE, project_root, _META_PARTS)


def canvas_geometry_key(project_root: str) -> Key:
    """The display-resolved geometry a full push writes, on the same terms as the meta
    document, and written before it so a reader pairing new meta with old geometry sees an
    identity mismatch rather than a false match."""
    return Key(CANVAS_GEOMETRY_STORE, project_root, _GEOMETRY_PARTS)


ActiveTab = Literal["annotate", "review", "training", "tuning", "inference", "results", "meta"]
"""The GUI's tabs: the vocabulary ``GuiState.active_tab`` holds and ``POST /api/state/tab``
validates against."""

TAB_NAMES = get_args(ActiveTab)

# One panel per GUI tab, plus "app" for steering the GUI itself (open a project, focus a tab).
# The pusher and the receiver both validate against this one set, so neither drifts apart.
VALID_PANELS = frozenset(TAB_NAMES) | {"app"}

# The event types the platform's own tool-driven emitters send; ``push_panel_data`` accepts any
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
    """Return the port the FastAPI backend is listening on.

    Reads the record under the workspace root, the same place the web backend writes it,
    regardless of this process's own platform-state root. The record is the answer: it names
    the port the backend actually bound, which can differ from any request when the requested
    one was taken and a free one substituted. ``TCIP_WEB_PORT`` is only ever a request, read
    when no record parses: the record can be absent (a failed publication, or a backend
    started as bare ``uvicorn tcip_web.app:app --port N``, which writes none), and the
    launcher serves either way, so with no record the request is the best information there
    is. An absent record and an unparseable one both fall through to the env var, then to the
    default, rather than raising: this runs before the backend is known to be up, so a missing
    handoff is an ordinary state, not a failure.
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
