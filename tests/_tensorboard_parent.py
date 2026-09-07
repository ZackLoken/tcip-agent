"""Standalone parent process for the TensorBoard lifetime tests.

Launches a stand-in child through the manager, then prints two pids on one line: the pid
``launch_tensorboard`` returned, and the TensorBoard stand-in's own pid (its guardian-spawned
child on POSIX with the tie enabled, the same pid on Windows or under ``--no-tie`` where no
guardian exists). Then either sleeps until killed or exits, depending on the ``mode`` argument, so
the test can watch what becomes of each process on each path. ``--no-tie`` disables the platform
lifetime tie before launching, so a test can prove the ``atexit`` hook in isolation; ``--thread``
launches from a background thread that has already finished before the pids are printed, since
every real launch site runs on a worker thread rather than the main one.
"""

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
    settles on exactly one, including the guardian itself having already died. Otherwise
    (Windows, or the tie disabled under ``--no-tie``) nothing sits in front of the stand-in, so
    the returned pid is used directly and children are never consulted.
    """
    if not guardian_expected:
        return returned_pid
    import psutil

    deadline = time.monotonic() + 5.0
    children: list = []
    while time.monotonic() < deadline:
        try:
            children = psutil.Process(returned_pid).children()
        except psutil.NoSuchProcess:
            sys.exit(f"guardian pid {returned_pid} died before spawning its child")
        if len(children) == 1:
            return children[0].pid
        time.sleep(0.1)
    sys.exit(
        f"expected exactly one guardian child of pid {returned_pid} within 5 seconds, found "
        f"{len(children)}"
    )


def main() -> None:
    logdir, mode = sys.argv[1], sys.argv[2]
    flags = set(sys.argv[3:])

    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    tb._tensorboard_argv = _sleep_argv
    if "--no-tie" in flags:
        tb._DISABLE_LIFETIME_TIE = True

    launched: dict = {}

    def _launch() -> None:
        launched["info"] = tb.launch_tensorboard(logdir, run_id="lifetime-parent")

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
