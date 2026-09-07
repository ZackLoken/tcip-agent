"""Standalone parent process for the TensorBoard lifetime tests.

Launches a stand-in child through the manager, then prints two pids on one line: the pid
``launch_tensorboard`` returned, and the TensorBoard stand-in's own pid (its guardian-spawned
child on POSIX, the same pid on Windows or under ``--no-tie`` where no guardian sits in front of
it). Then either sleeps until killed or exits, depending on the ``mode`` argument, so the test can
watch what becomes of each process on each path. ``--no-tie`` disables the platform lifetime tie
before launching, so a test can prove the ``atexit`` hook in isolation; ``--thread`` launches from
a background thread that has already finished before the pids are printed, since every real
launch site runs on a worker thread rather than the main one.
"""

from __future__ import annotations

import sys
import threading
import time


def _sleep_argv(logdir: str, port: int) -> list[str]:
    return [sys.executable, "-c", "import time; time.sleep(120)"]


def _standin_pid(returned_pid: int) -> int:
    """The TensorBoard stand-in's own pid: the returned pid on Windows or when the tie is
    disabled (nothing sits in front of it there), or its one guardian-spawned child on POSIX."""
    if sys.platform == "win32":
        return returned_pid
    import psutil

    children = psutil.Process(returned_pid).children()
    return children[0].pid if children else returned_pid


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
    print(returned_pid, _standin_pid(returned_pid), flush=True)

    if mode == "sleep":
        time.sleep(120)


if __name__ == "__main__":
    main()
