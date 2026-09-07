"""Keep one child tied to the life of the process that launched it, on platforms with no
job-object equivalent (Linux, macOS).

Run as its own process: ``python -m tcip_mcp.pipelines.training.tensorboard_guardian --parent
<pid> -- <argv...>``, under the launcher's own interpreter (``sys.executable``), which is why
``tcip_mcp`` must be importable there for ``-m`` to find this module at all. Starts ``<argv...>``
as its own child: while the launching process (identified by ``pid``, recognized only as long as
``os.getppid()`` still returns it) and the child are both alive, it waits, checking the child
every tenth of a second (so a child that exits immediately is reflected in the guardian's own
exit well inside the manager's half-second startup grace) and the parent's liveness once a second
(its death needs no faster detection than that). Once the parent is gone, it terminates the child
(``SIGTERM``, then ``SIGKILL`` if it has not exited within the two-second escalation grace) and
exits with the child's return code, 128 plus the signal number when a signal ended it (the shell
convention; a raw negative code does not survive ``sys.exit``); if the child exits on its own
first, the guardian exits with that same code; a ``SIGTERM`` delivered to the guardian itself is
forwarded to the child the same way. The escalation grace stays well under the five seconds
``stop_tensorboard`` waits for this process to end, so this process's own kill of a stubborn
child lands, and this process exits, before that wait gives up and reports stopped with
TensorBoard still alive underneath it; on the parent-death path a stubborn child is gone within
about three seconds of the parent dying (one second to detect it, plus the grace). The limits: a
parent's death is detected within a second of it happening, never instantly; a guardian the
kernel kills outright, rather than exiting through this code, leaves its child running with
nothing left watching it; and this mechanism covers macOS by the same code path as Linux,
exercised by no CI leg.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

_CHILD_POLL_SECONDS = 0.1
_PARENT_POLL_SECONDS = 1.0
# Kept well under the five seconds tensorboard_manager.stop_tensorboard waits for this
# process to end, so a stubborn child's kill lands here before that wait gives up on it.
_TERM_GRACE_SECONDS = 2.0


def _terminate_and_wait(child: subprocess.Popen) -> int:
    """Ask the child to stop, escalate if it ignores that, and return its raw exit code
    (negative when a signal ended it, as ``subprocess`` reports it)."""
    child.terminate()
    try:
        return child.wait(timeout=_TERM_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        child.kill()
        return child.wait()


def _exit_code(returncode: int) -> int:
    """The shell's own reading of a raw Popen return code: 128 plus the signal number when a
    signal ended it (a negative ``returncode``), the code itself otherwise."""
    return 128 - returncode if returncode < 0 else returncode


def _run(parent_pid: int, argv: list[str]) -> int:
    child = subprocess.Popen(argv)

    def _forward_sigterm(signum: int, frame: object) -> None:
        sys.exit(_exit_code(_terminate_and_wait(child)))

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
    return _exit_code(_run(parent_pid, argv))


if __name__ == "__main__":
    sys.exit(main())
