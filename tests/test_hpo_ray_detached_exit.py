"""A real Ray cluster's exit from a console-free process: ``tune_search`` inside a subprocess
created with ``DETACHED_PROCESS`` returns normally, and every daemon Ray started, and every process
those daemons spawned, is gone afterward."""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import textwrap
import time

import pytest

pytestmark = [
    pytest.mark.skipif(
        sys.platform != "win32",
        reason="the console-signal exit path this test drives is Windows-only",
    ),
    pytest.mark.ray_cluster,
]

DETACHED_PROCESS = 0x00000008

_SUBPROCESS_SCRIPT = textwrap.dedent(
    """
    import sys
    import threading
    import time
    from pathlib import Path


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

        tune_search(
            objective_fn=objective_fn,
            param_space={"optimizer.head_lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2}},
            metric="objective", mode="min", num_samples=1, search_alg="random",
            scheduler={"name": "fifo"}, max_concurrent=1,
            baseline_params=None, resources_per_trial={"cpu": 1}, split_draws=1,
            sweep_dir=Path(sys.argv[1]), seed=0
        )

        if not ready.wait(timeout=60):
            print("EXIT_FAIL: never captured Ray's daemon pids", flush=True)
            raise SystemExit(3)

        print("PIDS " + " ".join(str(pid) for pid in pids), flush=True)
        print("DESCENDANTS " + " ".join(str(pid) for pid in descendants), flush=True)
        print("EXIT_OK", flush=True)


    if __name__ == "__main__":
        main()
    """
)


def test_a_detached_console_free_sweep_exits_cleanly_and_leaves_no_ray_daemon_behind(tmp_path):
    """Asserts the subprocess exits 0 having finished its sweep, and that every process Ray
    started for the sweep, the table daemons the subprocess captured from Ray's node and every
    descendant those daemons spawned (the trial worker, the dashboard agent, the runtime-env
    agent), is gone once the subprocess exits."""
    import psutil

    script_path = tmp_path / "run_detached_sweep.py"
    script_path.write_text(_SUBPROCESS_SCRIPT, encoding="utf-8", newline="\n")
    storage_path = tmp_path / "hpo"
    storage_path.mkdir()
    ray_tmp_path = tmp_path / "ray_tmp"
    ray_tmp_path.mkdir()
    output_path = tmp_path / "sweep_output.txt"

    env = {
        **os.environ,
        "TCIP_WORKSPACE": str(tmp_path.parent),
        "RAY_TMPDIR": str(ray_tmp_path),
        "PYTHONUNBUFFERED": "1",
    }

    with open(output_path, "w") as output_file:
        proc = subprocess.Popen(
            [sys.executable, str(script_path), str(storage_path / "sweep")],
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
        assert (storage_path / "sweep").is_dir(), output

        daemon_pids, descendant_pids = read_reported_pids()
        assert daemon_pids, "the subprocess never reported the daemon pids it read from Ray's node"

        deadline = time.monotonic() + 30
        survivors = [pid for pid in daemon_pids + descendant_pids if psutil.pid_exists(pid)]
        while survivors and time.monotonic() < deadline:
            time.sleep(1)
            survivors = [pid for pid in daemon_pids + descendant_pids if psutil.pid_exists(pid)]
        assert not survivors, (
            f"Ray process(es) {survivors} still running after the detached sweep exited"
        )
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
