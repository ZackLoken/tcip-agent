"""Entry point: ``python -m tcip_web``.

Reads ``TCIP_WEB_HOST`` / ``TCIP_WEB_PORT`` for network binding (default
127.0.0.1:8765) and writes the chosen port to ``.tcip/state/web_port.txt``
so MCP tools in other processes can discover the backend.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

import uvicorn

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
PORT_FILE = Path(".tcip") / "state" / "web_port.txt"


def _pick_port(requested: int) -> int:
    """Return the requested port, or find a free one if it is bound."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((DEFAULT_HOST, requested))
            return requested
    except OSError:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((DEFAULT_HOST, 0))
            return s.getsockname()[1]


def _write_port_file(port: int) -> None:
    try:
        PORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        PORT_FILE.write_text(str(port), encoding="utf-8")
    except OSError:
        # Non-fatal: MCP tools fall back to TCIP_WEB_PORT / default
        pass


def main() -> None:
    host = os.environ.get("TCIP_WEB_HOST", DEFAULT_HOST)
    requested = int(os.environ.get("TCIP_WEB_PORT", str(DEFAULT_PORT)))
    port = _pick_port(requested)
    _write_port_file(port)
    reload = os.environ.get("TCIP_WEB_RELOAD", "0") == "1"
    uvicorn.run("tcip_web.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
