"""A parent process that launches a TensorBoard stand-in through the manager and prints, on one
line, the pid ``launch_tensorboard`` returned and the stand-in's own pid; then sleeps until killed
or exits, by its ``mode`` argument. ``--no-tie`` disables the platform lifetime tie before the
launch; ``--thread`` launches from a background thread that has finished before the pids print."""

from __future__ import annotations

import sys
import threading
import time


def _sleep_argv(logdir: str, port: int) -> list[str]:
    return [sys.executable, "-c", "import time; time.sleep(120)"]


def _standin_pid(returned_pid: int, guardian_expected: bool) -> int:
    """The TensorBoard stand-in's own pid.

    When a guardian is expected (POSIX with the tie enabled) the returned pid is the guardian's
    and its one child is the stand-in; this waits up to five seconds for that child to appear,
    since the guardian may not have spawned it yet, and exits naming the condition if it never
    settles on exactly one. A guardian that has already died is not a separate case: it stays a
    zombie, still resolvable and reporting zero children, until reaped by its own caller, which
    has neither polled nor waited on it since the launch's own startup check, so that outcome
    also reaches the five-second diagnosis below. Otherwise (Windows, or the tie disabled under
    ``--no-tie``) nothing sits in front of the stand-in, so the returned pid is used directly and
    children are never consulted. This is called both from the standalone script's own ``main``
    and directly by a pytest test; the ``SystemExit`` it raises on failure is accepted in the
    latter case too, since it fails the calling test the same as any exception.
    """
    if not guardian_expected:
        return returned_pid
    import psutil

    deadline = time.monotonic() + 5.0
    children: list = []
    while time.monotonic() < deadline:
        children = psutil.Process(returned_pid).children()
        if len(children) == 1:
            return children[0].pid
        time.sleep(0.1)
    status = psutil.Process(returned_pid).status()
    sys.exit(
        f"expected exactly one guardian child of pid {returned_pid} within 5 seconds, found "
        f"{len(children)} (guardian status: {status}); either it has not spawned its child yet "
        f"or it has died"
    )


def main() -> None:
    logdir, mode = sys.argv[1], sys.argv[2]
    flags = set(sys.argv[3:])

    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    tb._tensorboard_argv = _sleep_argv
    if "--no-tie" in flags:
        tb._guardian_argv = lambda argv: argv
        tb._assign_to_win_job = lambda proc: "disabled for test"

    launched: dict = {}

    def _launch() -> None:
        launched["info"] = tb.launch_tensorboard(logdir)

    if "--thread" in flags:
        thread = threading.Thread(target=_launch)
        thread.start()
        thread.join()
    else:
        _launch()

    returned_pid = launched["info"]["pid"]
    guardian_expected = sys.platform != "win32" and "--no-tie" not in flags
    print(returned_pid, _standin_pid(returned_pid, guardian_expected), flush=True)

    if mode == "sleep":
        time.sleep(120)


if __name__ == "__main__":
    main()
