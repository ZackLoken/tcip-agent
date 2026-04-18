"""TensorBoard process management for training and HPO runs."""

from __future__ import annotations

import logging
import socket
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_TB_PROCESSES: dict[str, subprocess.Popen] = {}


def _find_free_port(start: int = 6006, end: int = 6099) -> int:
    """Find a free TCP port in the given range."""
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start}-{end}")


def launch_tensorboard(logdir: str, run_id: str | None = None) -> dict:
    """Launch a TensorBoard process for the given log directory.

    Returns dict with 'url', 'port', 'pid', 'logdir'.
    If TensorBoard is already running for this logdir, returns existing info.
    """
    logdir = str(Path(logdir).resolve())
    key = run_id or logdir

    # Check if already running
    if key in _TB_PROCESSES:
        proc = _TB_PROCESSES[key]
        if proc.poll() is None:  # still alive
            # Recover port from stored info
            return {"url": f"http://localhost:{proc._tb_port}", "port": proc._tb_port, "pid": proc.pid, "logdir": logdir}
        else:
            del _TB_PROCESSES[key]

    port = _find_free_port()

    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "tensorboard", "--logdir", logdir,
             "--port", str(port), "--host", "127.0.0.1", "--reload_interval", "5"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        proc._tb_port = port  # type: ignore[attr-defined]
        _TB_PROCESSES[key] = proc
        logger.info("TensorBoard started: http://localhost:%d (pid=%d, logdir=%s)", port, proc.pid, logdir)
        return {
            "url": f"http://localhost:{port}",
            "port": port,
            "pid": proc.pid,
            "logdir": logdir,
        }
    except Exception as e:
        logger.warning("Failed to launch TensorBoard: %s", e)
        return {"error": str(e), "logdir": logdir}


def stop_tensorboard(run_id: str | None = None, logdir: str | None = None) -> dict:
    """Stop a running TensorBoard process."""
    key = run_id or (str(Path(logdir).resolve()) if logdir else None)
    if not key or key not in _TB_PROCESSES:
        return {"status": "not_running"}

    proc = _TB_PROCESSES.pop(key)
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    return {"status": "stopped", "pid": proc.pid}


def list_tensorboard() -> list[dict]:
    """List all running TensorBoard instances."""
    result = []
    for key, proc in list(_TB_PROCESSES.items()):
        alive = proc.poll() is None
        if not alive:
            del _TB_PROCESSES[key]
            continue
        result.append({
            "key": key,
            "url": f"http://localhost:{proc._tb_port}",
            "port": proc._tb_port,
            "pid": proc.pid,
        })
    return result
