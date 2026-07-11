"""HTTP client for MCP tools to push state to the tcip-web backend.

Replaces the legacy ``.tcip/events/`` file-bridge. MCP tools call
``post_panel_event`` to ship a panel event to the running FastAPI GUI.

Port discovery order:
  1. ``TCIP_WEB_PORT`` environment variable.
  2. ``.tcip/state/web_port.txt`` (written by the FastAPI backend on startup).
  3. Default: 8765.

Host discovery:
  1. ``TCIP_WEB_HOST`` environment variable.
  2. Default: 127.0.0.1.

Connection failures are treated as soft errors — the MCP tool returns
``{"status": "no_subscribers"}`` rather than raising. This keeps agent
workflows working when the GUI is closed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from tcip_mcp.project_paths import project_root as _platform_root

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
PORT_FILE_RELATIVE = Path(".tcip") / "state" / "web_port.txt"


def resolve_web_host() -> str:
    return os.environ.get("TCIP_WEB_HOST", DEFAULT_HOST)


def resolve_web_port(project_root: Optional[Path] = None) -> int:
    """Return the port the FastAPI backend is listening on.

    Parameters
    ----------
    project_root : Path, optional
        Directory that contains ``.tcip/state/web_port.txt``. Defaults to the pinned platform
        state root (``$TCIP_PROJECT_ROOT`` or cwd) — the same place the web backend writes it —
        so the port is found regardless of the reader's cwd.
    """
    env = os.environ.get("TCIP_WEB_PORT")
    if env:
        try:
            return int(env)
        except ValueError:
            logger.warning("TCIP_WEB_PORT=%r is not an integer; falling back", env)

    root = Path(project_root) if project_root else _platform_root()
    port_file = root / PORT_FILE_RELATIVE
    if port_file.exists():
        try:
            return int(port_file.read_text(encoding="utf-8").strip())
        except ValueError:
            logger.warning("Cannot parse port from %s; using default", port_file)

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

    url = backend_url(f"/api/events/{panel}", project_root=project_root)
    payload = json.dumps({"panel": panel, "event_type": event_type, "data": data}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
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
