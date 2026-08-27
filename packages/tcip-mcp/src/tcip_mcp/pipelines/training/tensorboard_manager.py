"""TensorBoard process management for training and HPO runs."""

from __future__ import annotations

import logging
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO

logger = logging.getLogger(__name__)


@dataclass
class _Launched:
    """A TensorBoard child this process is tracking: its port and output capture live here,
    not stuffed onto the ``Popen`` object, since an attribute added to it at runtime is invisible
    to callers that only know the process's declared type."""

    proc: subprocess.Popen
    port: int
    output: IO[bytes]


_TB_PROCESSES: dict[str, _Launched] = {}

# How long to let the child prove it survived before reporting a URL. A bad logdir, a
# taken port, or a missing tensorboard install all fail within this window; anything
# slower is caught later by the poll in ``list_tensorboard``.
_STARTUP_GRACE_SECONDS = 0.5


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


def _collect_output(handle) -> str:
    """Read back (and close) what a finished child wrote to its capture file."""
    try:
        handle.seek(0)
        return handle.read().decode("utf-8", errors="replace").strip()
    except Exception:
        return ""
    finally:
        handle.close()


def _release_output(entry: _Launched) -> None:
    """Close the capture file a process that is no longer tracked was writing to."""
    try:
        entry.output.close()
    except Exception:
        pass


def launch_tensorboard(logdir: str, run_id: str | None = None) -> dict:
    """Launch a TensorBoard process for the given log directory.

    Returns dict with 'url', 'port', 'pid', 'logdir', or ``{'error': ..., 'output': ...}``
    when the process died during startup, so a caller never advertises a URL nothing is
    serving. If TensorBoard is already running for this logdir, returns existing info.
    """
    logdir = str(Path(logdir).resolve())
    key = run_id or logdir

    # Check if already running
    if key in _TB_PROCESSES:
        entry = _TB_PROCESSES[key]
        if entry.proc.poll() is None:  # still alive
            # Recover port from stored info
            return {"url": f"http://localhost:{entry.port}", "port": entry.port,
                    "pid": entry.proc.pid, "logdir": logdir}
        else:
            _release_output(_TB_PROCESSES.pop(key))

    port = _find_free_port()

    # The child outlives this call and nothing reads its streams, so they go to a temp file:
    # an undrained pipe blocks the writer as soon as the OS buffer fills.
    output = tempfile.TemporaryFile()
    try:
        proc = subprocess.Popen(
            # The installed tensorboard package has no __main__.py, so "-m tensorboard" fails
            # with "cannot be directly executed"; tensorboard.main is the module that defines
            # run_main() under an `if __name__ == "__main__"` guard.
            [sys.executable, "-m", "tensorboard.main", "--logdir", logdir,
             "--port", str(port), "--host", "127.0.0.1", "--reload_interval", "5"],
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    except Exception as e:
        output.close()
        logger.warning("Failed to launch TensorBoard: %s", e)
        return {"error": str(e), "logdir": logdir}

    time.sleep(_STARTUP_GRACE_SECONDS)
    if proc.poll() is not None:
        detail = _collect_output(output)
        logger.warning("TensorBoard exited during startup (code=%s): %s", proc.returncode, detail)
        return {
            "error": f"tensorboard exited during startup with code {proc.returncode}",
            "output": detail,
            "logdir": logdir,
        }

    _TB_PROCESSES[key] = _Launched(proc=proc, port=port, output=output)
    logger.info("TensorBoard started: http://localhost:%d (pid=%d, logdir=%s)", port, proc.pid, logdir)
    return {
        "url": f"http://localhost:{port}",
        "port": port,
        "pid": proc.pid,
        "logdir": logdir,
    }


def stop_tensorboard(run_id: str | None = None, logdir: str | None = None) -> dict:
    """Stop a running TensorBoard process."""
    key = run_id or (str(Path(logdir).resolve()) if logdir else None)
    if not key or key not in _TB_PROCESSES:
        return {"status": "not_running"}

    entry = _TB_PROCESSES.pop(key)
    if entry.proc.poll() is None:
        entry.proc.terminate()
        try:
            entry.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            entry.proc.kill()
    _release_output(entry)
    return {"status": "stopped", "pid": entry.proc.pid}


def list_tensorboard() -> list[dict]:
    """List all running TensorBoard instances."""
    result = []
    for key, entry in list(_TB_PROCESSES.items()):
        alive = entry.proc.poll() is None
        if not alive:
            _release_output(_TB_PROCESSES.pop(key))
            continue
        result.append({
            "key": key,
            "url": f"http://localhost:{entry.port}",
            "port": entry.port,
            "pid": entry.proc.pid,
        })
    return result
