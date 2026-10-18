"""A parent process that launches a TensorBoard stand-in through the manager.

It prints, on one line, the pid ``launch_tensorboard`` returned and the stand-in's own pid; then
sleeps until killed or exits, by its ``mode`` argument. ``--no-tie`` disables the platform
lifetime tie before the launch; ``--thread`` launches from a background thread that has finished
before the pids print.
"""

from __future__ import annotations

import sys
import threading
import time


def _sleep_argv(logdir: str, port: int) -> list[str]:
    return [sys.executable, "-c", "import time; time.sleep(120)"]


def _standin_pid(returned_pid: int, guardian_expected: bool) -> int:
    """Return the TensorBoard stand-in's own pid.

    When ``guardian_expected``, ``returned_pid`` is the guardian's and the stand-in is its one
    child: the guardian's children are polled for up to five seconds and the one child's pid is
    returned, or ``SystemExit`` is raised naming the child count and the guardian's status.
    Otherwise ``returned_pid`` is returned as it is.
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
    print(
        returned_pid, _standin_pid(returned_pid, guardian_expected), flush=True
    )

    if mode == "sleep":
        time.sleep(120)


if __name__ == "__main__":
    main()
