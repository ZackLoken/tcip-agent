"""A real Ray cluster's exit from a console-free process, the scenario the fake-Ray lifecycle
tests in ``test_hpo_ray_lifecycle.py`` cannot reach: on Windows, ``ray.shutdown()`` ends each
daemon by signalling ``CTRL_BREAK_EVENT`` through its console, which raises ``OSError: [WinError
6] The handle is invalid`` on a daemon started without one (a server launched under
``DETACHED_PROCESS``, as the GUI capture harness does). This drives ``tune_search`` inside a
subprocess created with ``DETACHED_PROCESS`` and checks both that the call returns normally and
that every daemon Ray started, and every process those daemons spawned, is actually gone
afterward.
"""

from __future__ import annotations

import contextlib
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


    def _capture_daemon_pids(pids_holder, descendants_holder, ready):
        import psutil
        import ray

        node = None
        while node is None:
            if ray.is_initialized():
                node = ray._private.worker.global_worker.node
            else:
                time.sleep(0.05)
        while not node.all_processes:
            time.sleep(0.05)
        table_pids = sorted(
            {
                process_info.process.pid
                for infos in node.all_processes.values()
                for process_info in infos
            }
        )
        pids_holder.extend(table_pids)

        raylet_process = node.all_processes["raylet"][0].process
        seen_descendants = set()
        while raylet_process.poll() is None:
            for pid in table_pids:
                try:
                    seen_descendants.update(
                        child.pid for child in psutil.Process(pid).children(recursive=True)
                    )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            time.sleep(0.02)
        descendants_holder.extend(sorted(seen_descendants))
        ready.set()


    def main():
        from tcip_mcp.pipelines.training.hpo import tune_search

        pids = []
        descendants = []
        ready = threading.Event()
        watcher = threading.Thread(
            target=_capture_daemon_pids, args=(pids, descendants, ready), daemon=True
        )
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
        print("DESCENDANTS " + " ".join(str(pid) for pid in descendants), flush=True)
        print("n_trials=" + str(result["n_trials"]), flush=True)
        print("EXIT_OK", flush=True)


    if __name__ == "__main__":
        main()
    """
)


def test_a_detached_console_free_sweep_exits_cleanly_and_leaves_no_ray_daemon_behind(tmp_path):
    """Asserts the subprocess exits 0 and completed its one trial, and that every process Ray
    started for the sweep, the table daemons the subprocess captured from Ray's node and every
    descendant those daemons spawned (the trial worker, the dashboard agent, the runtime-env
    agent), is gone once the subprocess exits."""
    import psutil

    script_path = tmp_path / "run_detached_sweep.py"
    script_path.write_text(_SUBPROCESS_SCRIPT, encoding="utf-8", newline="\n")
    storage_path = tmp_path / "hpo"
    storage_path.mkdir()
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    ray_tmp_path = tmp_path / "ray_tmp"
    ray_tmp_path.mkdir()
    output_path = tmp_path / "sweep_output.txt"

    env = {
        **os.environ,
        "TCIP_STATE_ROOT": str(tmp_path),
        "TCIP_WORKSPACE": str(workspace_path),
        "RAY_TMPDIR": str(ray_tmp_path),
        "PYTHONUNBUFFERED": "1",
    }

    with open(output_path, "w") as output_file:
        proc = subprocess.Popen(
            [sys.executable, str(script_path), str(storage_path)],
            creationflags=DETACHED_PROCESS,
            stdout=output_file,
            stderr=subprocess.STDOUT,
            env=env,
        )

    def read_reported_pids() -> tuple[list[int], list[int]]:
        text = output_path.read_text(encoding="utf-8", errors="replace")
        pid_line = next((line for line in text.splitlines() if line.startswith("PIDS ")), None)
        descendants_line = next(
            (line for line in text.splitlines() if line.startswith("DESCENDANTS ")), None
        )
        daemons = [int(t) for t in pid_line.removeprefix("PIDS ").split()] if pid_line else []
        descendants = (
            [int(t) for t in descendants_line.removeprefix("DESCENDANTS ").split()]
            if descendants_line else []
        )
        return daemons, descendants

    try:
        proc.wait(timeout=240)
        output = output_path.read_text(encoding="utf-8", errors="replace")
        assert proc.returncode == 0, f"detached sweep exited {proc.returncode}:\n{output}"
        assert "n_trials=1" in output, output

        daemon_pids, descendant_pids = read_reported_pids()
        assert daemon_pids, "the subprocess never reported the daemon pids it read from Ray's node"

        deadline = time.monotonic() + 30
        survivors = [pid for pid in daemon_pids + descendant_pids if psutil.pid_exists(pid)]
        while survivors and time.monotonic() < deadline:
            time.sleep(1)
            survivors = [pid for pid in daemon_pids + descendant_pids if psutil.pid_exists(pid)]
        assert not survivors, f"Ray process(es) {survivors} still running after the detached sweep exited"
    finally:
        if proc.poll() is None:
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                for child in psutil.Process(proc.pid).children(recursive=True):
                    with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                        child.kill()
            proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=10)
        for pid in {*read_reported_pids()[0], *read_reported_pids()[1]}:
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                psutil.Process(pid).kill()
