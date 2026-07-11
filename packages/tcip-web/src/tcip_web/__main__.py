"""Entry point: ``python -m tcip_web``.

Reads ``TCIP_WEB_HOST`` / ``TCIP_WEB_PORT`` for network binding (default
127.0.0.1:8765) and writes the chosen port to ``.tcip/state/web_port.txt``
so MCP tools in other processes can discover the backend.
"""

from __future__ import annotations

import os
import socket

import uvicorn

from tcip_mcp.project_paths import project_root
from tcip_web.paths import is_loopback_host

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def _pick_port(host: str, requested: int) -> int:
    """Return ``requested`` if free on ``host``, else an OS-assigned free port.

    Probes the *actual* bind host: a port free on 127.0.0.1 can be taken on the interface
    we're about to bind (and vice-versa), so the old always-127.0.0.1 probe was wrong for
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
    # Anchored to the platform state root (pinned below) so MCP tools in other processes read
    # the SAME web_port.txt regardless of cwd, instead of a fragmented cwd-relative one.
    port_file = project_root() / ".tcip" / "state" / "web_port.txt"
    try:
        port_file.parent.mkdir(parents=True, exist_ok=True)
        port_file.write_text(str(port), encoding="utf-8")
    except OSError:
        # Non-fatal: MCP tools fall back to TCIP_WEB_PORT / default
        pass


def _refuse_insecure_bind(host: str) -> None:
    """Refuse to expose the GUI network-wide with no auth unless explicitly allowed.

    A non-loopback bind makes filesystem browsing + file writes reachable by anyone on the
    network. Require ``TCIP_WEB_ALLOW_INSECURE=1`` as an explicit, trusted-network
    acknowledgement (token auth for exposed deployments is a planned follow-on).
    """
    if is_loopback_host(host):
        return
    if os.environ.get("TCIP_WEB_ALLOW_INSECURE") == "1":
        return
    raise SystemExit(
        f"Refusing to bind {host!r} (non-loopback): this exposes the GUI — including "
        "filesystem browsing and file writes — to the whole network with no login.\n"
        "Set TCIP_WEB_ALLOW_INSECURE=1 to override (only on a trusted network)."
    )


def main() -> None:
    # Pin the platform state root before importing the app (which resolves EXPERIMENTS_DIR)
    # and before writing the port file, so this backend agrees with the MCP server on one
    # .tcip/ even when launched from a different directory. Inherited by uvicorn's reloader.
    from tcip_mcp.project_paths import pin_project_root

    pin_project_root()
    host = os.environ.get("TCIP_WEB_HOST", DEFAULT_HOST)
    _refuse_insecure_bind(host)
    requested = int(os.environ.get("TCIP_WEB_PORT", str(DEFAULT_PORT)))
    port = _pick_port(host, requested)
    _write_port_file(port)
    reload = os.environ.get("TCIP_WEB_RELOAD", "0") == "1"
    uvicorn.run("tcip_web.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
