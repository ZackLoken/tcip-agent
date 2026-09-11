"""Start or stop the served web app under a scratch environment, for a GUI capture harness.

``start`` launches ``python -m tcip_web`` with ``TCIP_WORKSPACE`` and ``TCIP_STATE_ROOT`` pointed
at scratch directories under a caller-named harness root, waits until the projects route answers,
and records pid and port in ``server_info.json`` under that root; ``stop`` kills the recorded
process tree and removes the record. Modeled on the day-3 capture harness's own
``harness_env.py``/``server_ctl.py`` pair, generalized to any harness root instead of one baked
in. No project fixture, seed data, or crop name lives here: seeding a project into the scratch
workspace before ``start``, or against the running server after it, is the capture script's own
job, never this tool's.

    python tools/serve_capture_app.py start <root> --state-project my_project --port 8799
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

REPO_ROOT = Path(__file__).resolve().parent.parent


def _is_under(path: Path, ancestor: Path) -> bool:
    path, ancestor = path.resolve(), ancestor.resolve()
    return path == ancestor or ancestor in path.parents


def _existing_workspace_conflict(root: Path) -> str | None:
    """``None`` when ``root`` looks safe to seed a fresh harness workspace under; otherwise the
    reason it does not: ``root`` itself already carries a ``.tcip`` directory (it is already a
    project root), or an immediate child of it does (a project already lives directly under
    ``root``). Either means real state already sits where a scratch workspace would be built."""
    if (root / ".tcip").is_dir():
        return f"{root} already carries its own .tcip state"
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / ".tcip").is_dir():
                return f"{child} already carries its own .tcip state"
    return None


def _refuse_unsafe_root(root: Path) -> None:
    """A harness root is never inside this repository. When this process is itself bound to a
    ``TCIP_WORKSPACE``, the root must alias it in neither direction (under it, or containing it);
    when no ``TCIP_WORKSPACE`` is bound, the root must not already look like a workspace, by
    :func:`_existing_workspace_conflict`. Any of these would point a scratch server at real
    state."""
    if _is_under(root, REPO_ROOT):
        raise SystemExit(f"{root} is under the repository ({REPO_ROOT}); name a root outside it")
    current_workspace = os.environ.get("TCIP_WORKSPACE")
    if current_workspace:
        workspace_path = Path(current_workspace)
        if _is_under(root, workspace_path) or _is_under(workspace_path, root):
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


def build_environ(root: Path, state_project: str, port: int) -> dict[str, str]:
    """The environment ``start`` launches ``python -m tcip_web`` under: a scratch workspace and
    state root beneath ``root``, the port to bind, unbuffered output, and the repository root
    prefixed onto ``PYTHONPATH`` (as the day-3 harness did) so a seeded run's subprocess can
    import a fixture module (a tiny trainer, say) by dotted name against the repository, the
    same way the capture script's own seeding step does. Refuses an unsafe root before building
    anything.
    """
    _refuse_unsafe_root(root)
    workspace = root / "workspace"
    env = dict(os.environ)
    env["TCIP_WORKSPACE"] = str(workspace)
    env["TCIP_STATE_ROOT"] = str(workspace / state_project)
    env["TCIP_WEB_PORT"] = str(port)
    env["PYTHONUNBUFFERED"] = "1"
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + existing if existing else "")
    return env


def start(root: Path, state_project: str, port: int) -> None:
    info_path = root / "server_info.json"
    if info_path.exists():
        raise SystemExit(f"{info_path} exists; stop first")
    env = build_environ(root, state_project, port)
    Path(env["TCIP_STATE_ROOT"]).mkdir(parents=True, exist_ok=True)
    log_path = root / "server.log"
    log = open(log_path, "a", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "tcip_web"], env=env, stdout=log, stderr=subprocess.STDOUT,
        cwd=str(root),
    )
    url = f"http://127.0.0.1:{port}/api/projects"
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"server exited early with {proc.returncode}; see {log_path}")
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    break
        except OSError:
            time.sleep(0.5)
    else:
        raise SystemExit("server did not answer in time")
    info_path.write_text(json.dumps({"pid": proc.pid, "port": port}), encoding="utf-8")
    print(json.dumps({"pid": proc.pid, "port": port}))


def stop(root: Path) -> None:
    info_path = root / "server_info.json"
    if not info_path.is_file():
        raise SystemExit(f"no server_info.json under {root}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    subprocess.run(
        ["taskkill", "/PID", str(info["pid"]), "/T", "/F"], check=False, capture_output=True
    )
    info_path.unlink()
    print(json.dumps({"stopped": info["pid"]}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    p_start = sub.add_parser("start", help="launch python -m tcip_web under the harness root")
    p_start.add_argument("root", type=Path)
    p_start.add_argument("--state-project", required=True,
                         help="project directory name created under <root>/workspace")
    p_start.add_argument("--port", type=int, required=True)

    p_stop = sub.add_parser("stop", help="kill the recorded server and remove its record")
    p_stop.add_argument("root", type=Path)

    args = parser.parse_args()
    _refuse_unsafe_root(args.root)
    if args.action == "start":
        start(args.root, args.state_project, args.port)
    else:
        stop(args.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
