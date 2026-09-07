"""Keep one child tied to the life of the process that launched it, on platforms with no
job-object equivalent (Linux, macOS).

Run as its own process: ``python -m tcip_mcp.pipelines.training.tensorboard_guardian --parent
<pid> -- <argv...>``. Starts ``<argv...>`` as its own child: while the launching process
(identified by ``pid``, recognized only as long as ``os.getppid()`` still returns it) and the
child are both alive, it waits, checking the child every tenth of a second (so a child that exits
immediately is reflected in the guardian's own exit well inside the manager's half-second startup
grace) and the parent's liveness once a second (its death needs no faster detection than that).
Once the parent is gone, it terminates the child (``SIGTERM``, then ``SIGKILL`` if it has not
exited five seconds later) and exits with the child's return code; if the child exits on its own
first, the guardian exits with that same code; a ``SIGTERM`` delivered to the guardian itself is
forwarded to the child the same way. The one limit this buys: a parent's death is detected within
a second of it happening, never instantly.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

_CHILD_POLL_SECONDS = 0.1
_PARENT_POLL_SECONDS = 1.0
_TERM_GRACE_SECONDS = 5.0


def _terminate_and_wait(child: subprocess.Popen) -> int:
    """Ask the child to stop, escalate if it ignores that, and return its exit code."""
    child.terminate()
    try:
        return child.wait(timeout=_TERM_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        child.kill()
        return child.wait()


def _run(parent_pid: int, argv: list[str]) -> int:
    child = subprocess.Popen(argv)

    def _forward_sigterm(signum: int, frame: object) -> None:
        code = _terminate_and_wait(child)
        sys.exit(code)

    signal.signal(signal.SIGTERM, _forward_sigterm)

    last_parent_check = time.monotonic()
    while True:
        code = child.poll()
        if code is not None:
            return code
        now = time.monotonic()
        if now - last_parent_check >= _PARENT_POLL_SECONDS:
            last_parent_check = now
            if os.getppid() != parent_pid:
                return _terminate_and_wait(child)
        time.sleep(_CHILD_POLL_SECONDS)


def _parse_args(argv: list[str]) -> tuple[int, list[str]]:
    if len(argv) < 2 or argv[0] != "--parent":
        raise SystemExit("usage: tensorboard_guardian --parent <pid> -- <argv...>")
    parent_pid = int(argv[1])
    rest = argv[2:]
    if rest[:1] == ["--"]:
        rest = rest[1:]
    if not rest:
        raise SystemExit("usage: tensorboard_guardian --parent <pid> -- <argv...>")
    return parent_pid, rest


def main() -> int:
    parent_pid, argv = _parse_args(sys.argv[1:])
    return _run(parent_pid, argv)


if __name__ == "__main__":
    sys.exit(main())
