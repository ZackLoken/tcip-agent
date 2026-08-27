"""Entry point: ``python -m tcip_web``.

Reads ``TCIP_WEB_HOST`` / ``TCIP_WEB_PORT`` for network binding (default
127.0.0.1:8765) and writes the chosen port to ``.tcip/state/web_port.txt``
so MCP tools in other processes can discover the backend.
"""

from __future__ import annotations

import logging
import os
import socket

import uvicorn
from tcip_store import replace
from tcip_store.binding import bind_default

from tcip_mcp.web_client import backend_port_key

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def _pick_port(host: str, requested: int) -> int:
    """Return ``requested`` if free on ``host``, else an OS-assigned free port.

    Probes the *actual* bind host: a port free on 127.0.0.1 can be taken on the interface
    we're about to bind (and vice-versa), so probing anything else gives the wrong answer for
    a non-loopback ``TCIP_WEB_HOST``.
    """
    for candidate in (requested, 0):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, candidate))
                return s.getsockname()[1]
        except OSError:
            continue
    return requested


def _write_port_file(port: int) -> None:
    """Publish the bound port for MCP tools in other processes.

    A failure here is non-fatal: those tools fall back to ``TCIP_WEB_PORT`` or the default, so
    refusing to serve would be worse than serving on a port they have to be told. It is logged
    rather than swallowed, because that fallback silently misses a port picked by the OS.
    """
    from tcip_mcp import workspace

    try:
        replace(backend_port_key(workspace.workspace_root(create=False)), str(port))
    except Exception:
        logger.exception(
            "Could not publish port %s: MCP tools will fall back to TCIP_WEB_PORT or %s",
            port, DEFAULT_PORT,
        )


def main() -> None:
    # The port handoff is written here, before uvicorn imports the app (which binds its own
    # storage backend at import; the served app pins the platform-state root later).
    bind_default()
    # A non-loopback host binds, and the app's trust boundary then serves this machine's own
    # connections and refuses network ones until the operator opts in (tcip_web.trust_boundary).
    host = os.environ.get("TCIP_WEB_HOST", DEFAULT_HOST)
    requested = int(os.environ.get("TCIP_WEB_PORT", str(DEFAULT_PORT)))
    port = _pick_port(host, requested)
    _write_port_file(port)
    reload = os.environ.get("TCIP_WEB_RELOAD", "0") == "1"
    uvicorn.run("tcip_web.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
