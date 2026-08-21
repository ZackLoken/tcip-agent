"""HTTP client for MCP tools to push state to the tcip-web backend.

MCP tools call ``post_panel_event`` to ship a panel event to the running FastAPI GUI over HTTP,
never through a file on disk.

The stores the two packages share are declared here rather than in the web package: the
backend's port handoff, the GUI snapshot, and the live-canvas pair. MCP tools read all of them
and cannot import ``tcip_web``, so the web side imports the declarations from here, which is the
legal dependency direction and the same one ``VALID_PANELS`` already takes. A declaration on
each side would be two stores wearing one name, and whichever imported first would decide where
the documents land.

Port discovery order:
  1. ``TCIP_WEB_PORT`` environment variable.
  2. The port record under the pinned platform root.
  3. The same record under the repo root: the backend writes it at its startup root (the repo,
     pre-adoption), so after ``set_active_project`` repins this process's root to a project the
     pinned location no longer holds the record; without this fallback the ping silently degraded
     to the default port whenever the backend ran on a non-default one.
  4. Default: 8765.

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
from typing import Any, Optional

import tcip_store
from tcip_store import RECORD_JSON, Key, StoreDescriptor, register_store, text_codec
from tcip_store.file_backend import RootedFileLocator

from tcip_mcp.project_paths import project_root as _platform_root
from tcip_mcp.project_paths import repo_root_from_here as _repo_root

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

_PORT_DOC = RootedFileLocator(prefix=(".tcip", "state"), suffix=".txt")
"""The backend's port handoff, one document under the platform state root."""

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
    first. ``root`` defaults to the platform state root, so a process launched from another
    directory reads the same handoff rather than a cwd-local one; the reader passes a root
    explicitly when it walks its candidate roots.
    """
    resolved = Path(root) if root is not None else _platform_root()
    return Key(BACKEND_PORT_STORE, str(resolved.resolve()), _PORT_PARTS)


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


# Every panel a pushed event may target: one per GUI tab, plus "app", the app-level channel for
# steering the GUI itself (open a project, focus a tab). The MCP tool that pushes and the web
# backend that receives both validate against this one set, so neither can drift into accepting
# a panel the other rejects.
VALID_PANELS = frozenset(
    {"app", "annotate", "review", "training", "tuning", "inference", "results", "meta"}
)


def resolve_web_host() -> str:
    return os.environ.get("TCIP_WEB_HOST", DEFAULT_HOST)


def resolve_web_port(project_root: Optional[Path] = None) -> int:
    """Return the port the FastAPI backend is listening on.

    Parameters
    ----------
    project_root : Path, optional
        Root the port record hangs off. Defaults to the pinned platform state root
        (``$TCIP_PROJECT_ROOT`` or cwd), the same place the web backend writes it, so the port
        is found regardless of the reader's cwd.

    An absent record and an unparseable one both fall through to the next candidate root and
    then to the default, rather than raising: this runs before the backend is known to be up,
    so a missing handoff is an ordinary state, not a failure.
    """
    env = os.environ.get("TCIP_WEB_PORT")
    if env:
        try:
            return int(env)
        except ValueError:
            logger.warning("TCIP_WEB_PORT=%r is not an integer; falling back", env)

    roots = [Path(project_root)] if project_root else [_platform_root(), _repo_root()]
    for root in roots:
        recorded = tcip_store.read(backend_port_key(root), default=None)
        if recorded is None:
            continue
        try:
            return int(recorded.strip())
        except ValueError:
            logger.warning("Cannot parse port recorded under %s; using default", root)

    return DEFAULT_PORT


def backend_url(path: str, project_root: Optional[Path] = None) -> str:
    """Build a full URL to the tcip-web backend for the given path."""
    host = resolve_web_host()
    port = resolve_web_port(project_root)
    if not path.startswith("/"):
        path = "/" + path
    return f"http://{host}:{port}{path}"


def post_panel_event(
    panel: str,
    event_type: str,
    data: dict[str, Any],
    *,
    project_root: Optional[Path] = None,
    timeout: float = 2.0,
) -> dict[str, Any]:
    """POST a panel event to the running tcip-web backend.

    Every return carries a ``delivered`` bool so callers don't mistake "backend down"
    for success. Returns one of:
      * ``{"status": "ok", "delivered": True, ...}`` on 2xx response.
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

    url = backend_url(f"/api/events/{panel}", project_root=project_root)
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
            if 200 <= code < 300:
                return {"status": "ok", "delivered": True, "url": url}
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
