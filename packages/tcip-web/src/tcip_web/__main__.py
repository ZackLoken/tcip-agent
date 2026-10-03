"""Entry point: ``python -m tcip_web``.

Resolves the workspace ``TCIP_WORKSPACE`` names once, binds the loopback address at
``TCIP_WEB_PORT`` (default 8765), records the port it bound under the workspace, then serves the
app configured with that workspace.
"""

from __future__ import annotations

import os
import socket

import uvicorn
from tcip_store import replace
from tcip_store.binding import bind_default

from tcip_mcp.web_client import BACKEND_HOST, backend_port_key

DEFAULT_PORT = 8765
"""The port asked for when ``TCIP_WEB_PORT`` is unset."""


def _pick_port(host: str, requested: int) -> int:
    """Return ``requested`` if free on ``host`` (probed on that host), else an OS-assigned free
    port.
    """
    for candidate in (requested, 0):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, candidate))
                return s.getsockname()[1]
        except OSError:
            continue
    return requested


def main() -> None:
    """Resolve the workspace once, publish the port under it, and serve the app configured with
    it and the image roots. Refuses what :func:`~tcip_mcp.workspace.workspace_from_environment`
    refuses."""
    from tcip_mcp.workspace import workspace_from_environment

    workspace = workspace_from_environment()
    instance = bind_default()
    requested = int(os.environ.get("TCIP_WEB_PORT", str(DEFAULT_PORT)))
    port = _pick_port(BACKEND_HOST, requested)
    replace(backend_port_key(workspace), str(port))
    # The app's import below binds its own instance; this one holds only the port write's
    # connection and is closed so the process ends up holding one backend's connections.
    instance.close()
    from tcip_web.app import app
    from tcip_web.paths import image_roots_from_environment
    from tcip_web.state import store

    store.configure(workspace, image_roots_from_environment())
    uvicorn.run(app, host=BACKEND_HOST, port=port)


if __name__ == "__main__":
    main()
