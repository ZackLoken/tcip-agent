"""Standalone parent process for the TensorBoard lifetime tests.

Launches a stand-in child through the manager, prints the child's pid, then either sleeps
until killed or exits, depending on the ``mode`` argument, so the test can watch what becomes of
the child on each path. ``--no-tie`` disables the platform lifetime tie before launching, so a
test can prove the ``atexit`` hook in isolation; ``--thread`` launches from a background thread
that has already finished before the pid is printed, since every real launch site runs on a
worker thread rather than the main one.
"""

from __future__ import annotations

import sys
import threading
import time


def _sleep_argv(logdir: str, port: int) -> list[str]:
    return [sys.executable, "-c", "import time; time.sleep(120)"]


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

    print(launched["info"]["pid"], flush=True)

    if mode == "sleep":
        time.sleep(120)


if __name__ == "__main__":
    main()
