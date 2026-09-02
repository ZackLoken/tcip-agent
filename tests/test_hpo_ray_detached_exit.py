"""A real Ray cluster's exit from a console-free process, the scenario the fake-Ray lifecycle
tests in ``test_hpo_ray_lifecycle.py`` cannot reach: on Windows, ``ray.shutdown()`` ends each
daemon by signalling ``CTRL_BREAK_EVENT`` through its console, which raises ``OSError: [WinError
6] The handle is invalid`` on a daemon started without one (a server launched under
``DETACHED_PROCESS``, as the GUI capture harness does). This drives ``tune_search`` inside a
subprocess created with ``DETACHED_PROCESS`` and checks both that the call returns normally and
that every daemon Ray started is actually gone afterward.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="the console-signal exit path this test drives is Windows-only")

DETACHED_PROCESS = 0x00000008

_SUBPROCESS_SCRIPT = textwrap.dedent(
    """
    import sys
    import threading
    import time


    def _capture_daemon_pids(pids_holder, ready):
        import ray

        node = None
        while node is None:
            if ray.is_initialized():
                node = ray._private.worker.global_worker.node
            else:
                time.sleep(0.05)
        while not node.all_processes:
            time.sleep(0.05)
        pids = sorted(
            {
                process_info.process.pid
                for infos in node.all_processes.values()
                for process_info in infos
            }
        )
        pids_holder.extend(pids)
        ready.set()


    def main():
        from tcip_mcp.pipelines.training.hpo import tune_search

        pids = []
        ready = threading.Event()
        watcher = threading.Thread(target=_capture_daemon_pids, args=(pids, ready), daemon=True)
        watcher.start()

        def objective_fn(config, report):
            report(1.0)

        storage_path = sys.argv[1]
        result = tune_search(
            objective_fn=objective_fn,
            param_space={"lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2}},
            num_samples=1,
            search_alg="random",
            scheduler=None,
            resources_per_trial={"cpu": 1},
            storage_path=storage_path,
        )

        if not ready.wait(timeout=60):
            print("EXIT_FAIL: never captured Ray's daemon pids", flush=True)
            raise SystemExit(3)

        print("PIDS " + " ".join(str(pid) for pid in pids), flush=True)
        print("n_trials=" + str(result["n_trials"]), flush=True)
        print("EXIT_OK", flush=True)


    if __name__ == "__main__":
        main()
    """
)


def test_a_detached_console_free_sweep_exits_cleanly_and_leaves_no_ray_daemon_behind(tmp_path):
    """Reproduced by hand against the unfixed code before this test was written: the same
    script run attached to a console exits 0, and run detached (``DETACHED_PROCESS``, no
    console) raises ``OSError: [WinError 6] The handle is invalid`` out of ``tune_search``. With
    the fix, both runs exit 0 and leave no Ray daemon running."""
    import psutil

    script_path = tmp_path / "run_detached_sweep.py"
    script_path.write_text(_SUBPROCESS_SCRIPT, encoding="utf-8", newline="\n")
    storage_path = tmp_path / "hpo"
    storage_path.mkdir()
    output_path = tmp_path / "sweep_output.txt"

    with open(output_path, "w") as output_file:
        proc = subprocess.Popen(
            [sys.executable, str(script_path), str(storage_path)],
            creationflags=DETACHED_PROCESS,
            stdout=output_file,
            stderr=subprocess.STDOUT,
            env={**os.environ, "TCIP_STATE_ROOT": str(tmp_path), "PYTHONUNBUFFERED": "1"},
        )
        proc.wait(timeout=240)

    output = output_path.read_text(encoding="utf-8", errors="replace")
    assert proc.returncode == 0, f"detached sweep exited {proc.returncode}:\n{output}"
    assert "n_trials=1" in output, output

    pid_line = next(line for line in output.splitlines() if line.startswith("PIDS "))
    daemon_pids = [int(token) for token in pid_line.removeprefix("PIDS ").split()]
    assert daemon_pids, "the subprocess never reported the daemon pids it read from Ray's node"

    deadline = time.monotonic() + 30
    survivors = [pid for pid in daemon_pids if psutil.pid_exists(pid)]
    while survivors and time.monotonic() < deadline:
        time.sleep(1)
        survivors = [pid for pid in daemon_pids if psutil.pid_exists(pid)]
    assert not survivors, f"Ray daemon(s) {survivors} still running after the detached sweep exited"
