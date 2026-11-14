"""Serve the built web app over a scratch workspace and have one cold reader walk it toward a goal.

    python -m tools.cold_walkthrough --goal <file> --root <harness root> --out <folder>

The server is ``tools/serve_capture_app.py``'s, over ``<root>/workspace``; the reader is
``tools/cross_family_ask.py``'s ``browser-only`` condition, run once from a new empty folder made
under the system's temporary directory, apart from ``<out>``, and prompted with the goal file's
text followed by the page's address. Its restricted mode loads no settings file, instruction
file or memory, but the account's email address and the working directory still reach the
reader, and a temporary directory inside a repository would hand it that repository's git
status. Each walk writes its own folder under ``<out>``, holding the reader's run and
``walkthrough.json``: the goal file's
path and hash, the port, the workspace served, the stale-bundle sentence the server sent, the
reader run's fault and whether the server was stopped. The command exits non-zero when the
reader's run has a fault or the stop failed.

The scratch project is the caller's to build under ``<root>/workspace`` with the platform's own
commands, never a link to a person's project. Images go in by ``ingest_images`` with its default,
which copies them. A plant file is copied inside the scratch project before it is registered,
since a registry stores the registered file's own path, and a dataset registered by an absolute
path outside the root is still one the served app admits.

Refuses a host the server cannot be stopped on, before starting it, and, after stopping it, a
root whose app has no built page.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from websockets.sync.client import connect

import tcip_store
from tcip_mcp.web_client import LOOPBACK_HOST
from tools import cross_family_ask, serve_capture_app

CONDITION = "browser-only"
"""The runner's condition the reader runs under."""


def _page_status(address: str) -> int:
    """Return the HTTP status the served page at ``address`` answers with."""
    try:
        with urllib.request.urlopen(
            address, timeout=serve_capture_app.PROBE_TIMEOUT_S
        ) as resp:
            return int(resp.status)
    except urllib.error.HTTPError as exc:
        return exc.code


def _stale_bundle(port: int) -> str | None:
    """Return the stale-bundle sentence of the first state snapshot the server on ``port`` sends."""
    wait_s = serve_capture_app.PROBE_TIMEOUT_S
    with connect(
        f"ws://{LOOPBACK_HOST}:{port}/ws/state", open_timeout=wait_s
    ) as ws:
        snapshot = json.loads(ws.recv(timeout=wait_s))
    return snapshot["bundle"]


def _stop(root: Path) -> str | None:
    """Stop the server under ``root``; return why the stop refused, or ``None`` when it stopped."""
    try:
        serve_capture_app.stop(root)
    except SystemExit as exc:
        return str(exc)
    return None


def walk(goal: Path, root: Path, out: Path) -> int:
    """Serve ``root``'s workspace, run one cold reader toward ``goal``, and record the walk.

    The walk's folder is a new child of ``out``; the reader's working directory is a new empty
    directory under the system's temporary directory. Returns 1 when the reader's run has a
    fault or the stop failed, else 0. Refuses what ``serve_capture_app.start`` refuses, and a
    root whose app answers its page with anything but 200, after stopping its server. When the
    walk raises, a stop that refuses raises its refusal chained onto that failure.
    """
    serve_capture_app.refuse_unstoppable_host()
    goal_bytes = goal.read_bytes()
    walk_id = datetime.datetime.now().strftime("walk-%Y%m%dT%H%M%S%f")
    folder = out / walk_id
    folder.mkdir(parents=True)
    reader_cwd = Path(tempfile.mkdtemp(prefix="cold-reader-"))
    server = serve_capture_app.start(root)
    address = f"http://{LOOPBACK_HOST}:{server['port']}/"
    try:
        status = _page_status(address)
        if status == 200:
            bundle = _stale_bundle(server["port"])
            meta = cross_family_ask.run_one(
                walk_id,
                CONDITION,
                f"{goal_bytes.decode('utf-8').rstrip()}\n\n{address}\n",
                reader_cwd,
                out,
                cross_family_ask.DEFAULT_TIMEOUT_S,
                cross_family_ask.resolve_launch("claude", None, None),
                prompt_source=str(goal),
            )
    except BaseException as failure:
        try:
            serve_capture_app.stop(root)
        except SystemExit as refused:
            raise refused from failure
        raise
    refusal = _stop(root)
    if status != 200:
        raise SystemExit(
            f"{address} answered {status}, not the built page; a 503 there is a frontend "
            "with no built page, made by its build"
            + (f" (and the stop refused: {refusal})" if refusal else "")
        )
    record = {
        "goal": str(goal),
        "goal_sha256": hashlib.sha256(goal_bytes).hexdigest(),
        "port": server["port"],
        "workspace": server["workspace"],
        "bundle": bundle,
        "reader_fault": meta["fault"],
        "stopped": refusal is None,
        "stop_refusal": refusal,
    }
    (folder / "walkthrough.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, indent=2))
    return 1 if meta["fault"] or refusal else 0


def main() -> int:
    """Run one walk from the command line and return its exit code."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--goal", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tcip_store.bind()
    return walk(args.goal, args.root, args.out)


if __name__ == "__main__":
    sys.exit(main())
