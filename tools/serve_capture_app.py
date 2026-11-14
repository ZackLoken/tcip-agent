"""Start or stop the served web app under a scratch environment.

``start`` launches ``python -m tcip_web`` with ``TCIP_WORKSPACE`` pointed at the ``workspace``
directory of a caller-named harness root, on the port named or one the operating system assigns.
It is ready when the port the backend recorded under that workspace answers for that same
directory while the process it started still runs; it then records pid, port and workspace in
``server_info.json`` under the root. A start that fails anywhere after the process exists ends
that process tree before it refuses. ``stop`` kills the recorded process tree and removes the
record, and keeps the record and refuses when the kill fails.

The server is kept to the scratch tree only so far: the caller's ``TCIP_IMAGE_ROOTS`` is not
passed on, and a workspace or project directory resolving outside the root is refused. A dataset
registered by an absolute path outside the root is still a root the served app admits, so data
goes into a scratch project by copy.

Both actions are Windows-only (``taskkill /T``) and refuse on any other host.

    python tools/serve_capture_app.py start <root> [--port 8799]
    python tools/serve_capture_app.py stop <root>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import tcip_store
from tcip_mcp.web_client import (
    LOOPBACK_HOST,
    NoBackendPortError,
    backend_port_key,
    resolve_web_port,
)
from tcip_mcp.workspace import project_dirs

REPO_ROOT = Path(__file__).resolve().parent.parent

WORKSPACE_DIRNAME = "workspace"
"""The harness root's child the scratch server serves as its workspace."""

READY_TIMEOUT_S = 120.0
"""How long a start waits for its server to answer, the value this tool has always waited."""


def refuse_unstoppable_host() -> None:
    """Refuse (``SystemExit``) a host other than Windows, where ``taskkill /T`` cannot run."""
    if sys.platform != "win32":
        raise SystemExit(
            "this tool is Windows-only (it ends the server's process tree with taskkill /T); "
            f"this host is {sys.platform}, which this tool does not support"
        )


def _existing_workspace_conflict(root: Path) -> str | None:
    """Return why ``root`` is unsafe to seed a fresh harness workspace under, or ``None``.

    It is unsafe when ``root`` itself already carries a ``.tcip`` directory (it is already a
    project root), or an immediate child of it other than its own :data:`WORKSPACE_DIRNAME` does
    (a project already lives directly under ``root``). Either means real state already sits where
    a scratch workspace would be built.
    """
    if (root / ".tcip").is_dir():
        return f"{root} already carries its own .tcip state"
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if child.name == WORKSPACE_DIRNAME:
                continue
            if child.is_dir() and (child / ".tcip").is_dir():
                return f"{child} already carries its own .tcip state"
    return None


def _refuse_unsafe_root(root: Path) -> None:
    """Refuse (``SystemExit``) a harness root that would point a scratch server at real state.

    A harness root is never inside this repository. When this process is itself bound to a
    ``TCIP_WORKSPACE``, the root must alias it in neither direction (under it, or containing it);
    when no ``TCIP_WORKSPACE`` is bound, the root must not already look like a workspace, by
    :func:`_existing_workspace_conflict`.
    """
    resolved = root.resolve()
    if resolved.is_relative_to(REPO_ROOT):
        raise SystemExit(
            f"{root} is under the repository ({REPO_ROOT}); name a root outside it"
        )
    current_workspace = os.environ.get("TCIP_WORKSPACE")
    if current_workspace:
        workspace_path = Path(current_workspace).resolve()
        if resolved.is_relative_to(
            workspace_path
        ) or workspace_path.is_relative_to(resolved):
            raise SystemExit(
                f"{root} and this process's own TCIP_WORKSPACE ({current_workspace}) alias one "
                "another (one contains the other, or they are the same directory); a harness "
                "root must never alias the caller's active workspace in either direction"
            )
        return
    conflict = _existing_workspace_conflict(root)
    if conflict:
        raise SystemExit(
            f"{root} already looks like a workspace ({conflict}); name a fresh root"
        )


def _refuse_links_outside(root: Path) -> None:
    """Refuse (``SystemExit``) a workspace, or a project listed in it, resolving outside ``root``.

    ``root`` is resolved; the projects are those ``project_dirs`` lists.
    """
    workspace = root / WORKSPACE_DIRNAME
    if not workspace.is_dir():
        return
    for directory in (workspace, *project_dirs(workspace)):
        if not directory.resolve().is_relative_to(root):
            raise SystemExit(
                f"{directory} resolves outside the harness root {root} "
                f"(to {directory.resolve()}); a scratch server serves the scratch tree alone"
            )


def build_environ(root: Path, port: int | None) -> dict[str, str]:
    """Return the environment the web app, ``python -m tcip_web``, is launched under.

    It names the scratch workspace beneath the resolved ``root``, the port to bind (``0``, one
    the operating system assigns, for ``None``), unbuffered output, and the repository root
    prefixed onto ``PYTHONPATH``, and drops ``TCIP_IMAGE_ROOTS``. Refuses an unsafe root and a
    workspace or project linked outside it before building anything.
    """
    _refuse_unsafe_root(root)
    resolved = root.resolve()
    _refuse_links_outside(resolved)
    env = dict(os.environ)
    env.pop("TCIP_IMAGE_ROOTS", None)
    env["TCIP_WORKSPACE"] = str(resolved / WORKSPACE_DIRNAME)
    env["TCIP_WEB_PORT"] = str(0 if port is None else port)
    env["PYTHONUNBUFFERED"] = "1"
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + existing if existing else ""
    )
    return env


def answers_for(port: int, workspace: Path) -> bool:
    """Return whether the backend on ``port`` answers its projects route for ``workspace``.

    Returns ``False`` while nothing answers there. Refuses (``SystemExit``) an answer naming
    another directory, compared by ``os.path.samefile``.
    """
    url = f"http://{LOOPBACK_HOST}:{port}/api/projects"
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            answered = json.load(resp)["workspace"]
    except OSError:
        return False
    if not os.path.samefile(answered, workspace):
        raise SystemExit(
            f"the backend on port {port} answers for {answered}, not {workspace}"
        )
    return True


def _await_own_server(
    proc: subprocess.Popen, workspace: Path, log_path: Path, timeout_s: float
) -> int:
    """Return the port ``proc`` serves ``workspace`` on, read from the port it recorded there.

    Refuses (``SystemExit``) a process that has exited, an answer for another workspace
    (:func:`answers_for`), and no answer within ``timeout_s`` seconds.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(
                f"server exited early with {proc.returncode}; see {log_path}"
            )
        try:
            port = resolve_web_port(workspace)
        except NoBackendPortError:
            port = None
        if (
            port is not None
            and answers_for(port, workspace)
            and proc.poll() is None
        ):
            return port
        time.sleep(0.5)
    raise SystemExit(
        f"server {proc.pid} did not answer for {workspace} within {timeout_s} s; "
        f"see {log_path}"
    )


def _kill_tree(pid: int) -> subprocess.CompletedProcess[str]:
    """Kill process ``pid`` and its descendants with ``taskkill /T /F`` and return the result."""
    return subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        check=False,
        capture_output=True,
        text=True,
    )


def _end(proc: subprocess.Popen) -> None:
    """End ``proc``'s tree if it still runs; refuse (``SystemExit``) when the kill fails."""
    if proc.poll() is not None:
        return
    killed = _kill_tree(proc.pid)
    if killed.returncode != 0:
        raise SystemExit(
            f"the started server, process {proc.pid}, could not be ended: "
            f"{(killed.stderr or killed.stdout).strip()}"
        )
    proc.wait()


def start(
    root: Path, port: int | None = None, timeout_s: float = READY_TIMEOUT_S
) -> dict:
    """Launch the web app under ``root``, wait for it, and record its pid, port and workspace.

    ``port`` is the one asked for, ``None`` for one the operating system assigns; the port
    recorded is the one the backend bound. Returns the record. Refuses a host other than Windows,
    an existing ``server_info.json`` under ``root``, what :func:`build_environ` refuses, and what
    :func:`_await_own_server` refuses, the last after ending the process tree it started.
    """
    refuse_unstoppable_host()
    root = root.resolve()
    info_path = root / "server_info.json"
    if info_path.exists():
        raise SystemExit(f"{info_path} exists; stop first")
    env = build_environ(root, port)
    workspace = Path(env["TCIP_WORKSPACE"])
    workspace.mkdir(parents=True, exist_ok=True)
    workspace = workspace.resolve()
    port_key = backend_port_key(workspace)
    if tcip_store.exists(port_key):
        tcip_store.delete(port_key)
    log_path = root / "server.log"
    with open(log_path, "a", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "tcip_web"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(root),
        )
    try:
        bound = _await_own_server(proc, workspace, log_path, timeout_s)
        info = {"pid": proc.pid, "port": bound, "workspace": str(workspace)}
        info_path.write_text(json.dumps(info), encoding="utf-8")
    except BaseException:
        _end(proc)
        raise
    finally:
        tcip_store.release_root(workspace)
    return info


def stop(root: Path) -> None:
    """Kill the recorded process tree with ``taskkill /T`` and remove its record.

    Refuses on a host other than Windows, with no record, and, keeping the record, when the kill
    fails, naming the process.
    """
    refuse_unstoppable_host()
    info_path = root / "server_info.json"
    if not info_path.is_file():
        raise SystemExit(f"no server_info.json under {root}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    killed = _kill_tree(info["pid"])
    if killed.returncode != 0:
        raise SystemExit(
            f"taskkill could not end process {info['pid']}: "
            f"{(killed.stderr or killed.stdout).strip()}; {info_path} is kept"
        )
    info_path.unlink()
    print(json.dumps({"stopped": info["pid"]}))


def main() -> int:
    """Run the ``start`` or ``stop`` action the command line names and return 0."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    p_start = sub.add_parser(
        "start", help="launch python -m tcip_web under the harness root"
    )
    p_start.add_argument("root", type=Path)
    p_start.add_argument(
        "--port",
        type=int,
        default=None,
        help="the port to ask for; without it the operating system assigns one",
    )

    p_stop = sub.add_parser(
        "stop", help="kill the recorded server and remove its record"
    )
    p_stop.add_argument("root", type=Path)

    args = parser.parse_args()
    _refuse_unsafe_root(args.root)
    if args.action == "start":
        tcip_store.bind()
        print(json.dumps(start(args.root, args.port)))
    else:
        stop(args.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
