"""A parent process that launches a TensorBoard stand-in through the manager.

It prints, on one line, the pid of the child the manager tracks and the stand-in's own pid; then
sleeps until killed or exits, by its ``mode`` argument. ``--no-tie`` disables the platform
lifetime tie before the launch; ``--thread`` launches from a background thread that has finished
before the pids print.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path


def listening_argv(
    port: int,
    *,
    delay: float = 0.0,
    seconds: float = 120.0,
    ignore_term: bool = False,
) -> list[str]:
    """Return a TensorBoard stand-in's argv: after ``delay`` it accepts on ``port``, then sleeps.

    It sleeps ``seconds`` once listening, ignoring SIGTERM when ``ignore_term``.
    """
    script = (
        "import signal, socket, time\n"
        + (
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            if ignore_term
            else ""
        )
        + f"time.sleep({delay})\n"
        + "s = socket.socket()\n"
        + f"s.bind(('127.0.0.1', {port}))\n"
        + "s.listen()\n"
        + f"time.sleep({seconds})\n"
    )
    return [sys.executable, "-c", script]


def _sleep_argv(logdir: str, port: int, prefix: str) -> list[str]:
    return listening_argv(port)


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
    directory, mode = Path(sys.argv[1]), sys.argv[2]
    flags = set(sys.argv[3:])

    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    tb._tensorboard_argv = _sleep_argv
    if "--no-tie" in flags:
        tb._guardian_argv = lambda argv: argv
        tb._assign_to_win_job = lambda proc: None

    launched: dict[str, tb.TensorboardLaunch] = {}

    def _launch() -> None:
        launched["info"] = tb.launch_tensorboard(directory)

    if "--thread" in flags:
        thread = threading.Thread(target=_launch)
        thread.start()
        thread.join()
    else:
        _launch()

    info = launched["info"]
    if not isinstance(info, tb.TensorboardServing):
        sys.exit(f"launch_tensorboard did not serve: {info}")
    returned_pid = tb._TB_PROCESSES[tb._board_key(directory)].proc.pid
    guardian_expected = sys.platform != "win32" and "--no-tie" not in flags
    print(
        returned_pid, _standin_pid(returned_pid, guardian_expected), flush=True
    )

    if mode == "sleep":
        time.sleep(120)


if __name__ == "__main__":
    main()
