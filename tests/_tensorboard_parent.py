"""Standalone parent process for the TensorBoard lifetime tests.

Launches a stand-in child through the manager, prints the child's pid, then either sleeps
until killed or exits, depending on ``sys.argv[2]``, so the test can watch what happens to the
child on each path.
"""

from __future__ import annotations

import sys
import time


def _sleep_argv(logdir: str, port: int) -> list[str]:
    return [sys.executable, "-c", "import time; time.sleep(120)"]


def main() -> None:
    logdir, mode = sys.argv[1], sys.argv[2]

    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    tb._tensorboard_argv = _sleep_argv
    info = tb.launch_tensorboard(logdir, run_id="lifetime-parent")
    print(info["pid"], flush=True)

    if mode == "sleep":
        time.sleep(120)


if __name__ == "__main__":
    main()
