"""TensorBoard process management for training and HPO runs.

A launched child is tied to the life of the process that launched it, since nothing else stops
it once the launcher is gone. On Windows every child is assigned to one job object created with
``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``, so the kernel kills every assigned child the moment this
process's last handle to the job closes, however this process ends. On Linux each child asks the
kernel for ``PR_SET_PDEATHSIG`` against this process, so a parent death by any signal, including
an unblockable one, ends the child. macOS has neither mechanism; there, and as a second line of
defense everywhere, a normal interpreter exit runs an ``atexit`` hook that stops every tracked
child, which does not run when this process is killed rather than exiting on its own.
"""

from __future__ import annotations

import atexit
import ctypes
import logging
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, cast

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

_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_PR_SET_PDEATHSIG = 1

_atexit_registered = False
_win_kernel32: ctypes.WinDLL | None = None
_win_job_handle: int | None = None


class _JobObjectBasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _JobObjectExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JobObjectBasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _get_win_kernel32() -> ctypes.WinDLL:
    """The kernel32 handle for the job-object calls, with pointer-sized signatures set once so a
    64-bit handle is never truncated by ctypes' default 32-bit return type."""
    global _win_kernel32
    if _win_kernel32 is None:
        dll = ctypes.WinDLL("kernel32", use_last_error=True)
        dll.CreateJobObjectW.restype = ctypes.c_void_p
        dll.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        dll.SetInformationJobObject.restype = ctypes.c_int
        dll.SetInformationJobObject.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32,
        ]
        dll.AssignProcessToJobObject.restype = ctypes.c_int
        dll.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        dll.CloseHandle.restype = ctypes.c_int
        dll.CloseHandle.argtypes = [ctypes.c_void_p]
        _win_kernel32 = dll
    return _win_kernel32


def _ensure_win_job() -> int | None:
    """Create, once, the job object every child is assigned to; ``None`` if that failed. The
    kernel closes it, and every child still assigned, when this process's handles all close."""
    global _win_job_handle
    if _win_job_handle is not None:
        return _win_job_handle
    kernel32 = _get_win_kernel32()
    handle = kernel32.CreateJobObjectW(None, None)
    if not handle:
        logger.warning("CreateJobObjectW failed: %s", ctypes.WinError(ctypes.get_last_error()))
        return None
    info = _JobObjectExtendedLimitInformation()
    info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(
        handle, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info)
    ):
        logger.warning(
            "SetInformationJobObject failed: %s", ctypes.WinError(ctypes.get_last_error())
        )
        kernel32.CloseHandle(handle)
        return None
    _win_job_handle = handle
    return handle


def _assign_to_win_job(proc: subprocess.Popen) -> None:
    """Tie ``proc`` to the job object so it dies with this process; a failed assignment is
    logged and the child is left tracked exactly as it would be without a job at all."""
    job = _ensure_win_job()
    if job is None:
        return
    kernel32 = _get_win_kernel32()
    # subprocess.Popen keeps the raw process handle here on Windows; nothing else exposes it.
    handle = cast(int, getattr(proc, "_handle"))
    if not kernel32.AssignProcessToJobObject(job, handle):
        logger.warning(
            "AssignProcessToJobObject failed for pid %d: %s",
            proc.pid, ctypes.WinError(ctypes.get_last_error()),
        )


def _set_pdeathsig_on_parent_exit() -> None:
    """Run in the child, right after ``fork`` and before ``exec``: ask the kernel to deliver
    ``SIGTERM`` if this process's parent ends by any means, including an unblockable signal."""
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)


def _stop_all_tracked() -> None:
    for key in list(_TB_PROCESSES):
        stop_tensorboard(run_id=key)


def _register_atexit_once() -> None:
    global _atexit_registered
    if _atexit_registered:
        return
    _atexit_registered = True
    atexit.register(_stop_all_tracked)


def _tensorboard_argv(logdir: str, port: int) -> list[str]:
    """The child's command line, as its own function so a test can run a stand-in process."""
    # tensorboard has no __main__.py ("-m tensorboard" fails with "cannot be directly executed");
    # tensorboard.main defines run_main() under an `if __name__ == "__main__"` guard.
    return [
        sys.executable, "-m", "tensorboard.main", "--logdir", logdir,
        "--port", str(port), "--host", "127.0.0.1", "--reload_interval", "5",
    ]


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
    _register_atexit_once()

    # The child outlives this call and nothing reads its streams, so they go to a temp file:
    # an undrained pipe blocks the writer as soon as the OS buffer fills.
    output = tempfile.TemporaryFile()
    preexec_fn = _set_pdeathsig_on_parent_exit if sys.platform == "linux" else None
    try:
        proc = subprocess.Popen(
            _tensorboard_argv(logdir, port),
            stdout=output,
            stderr=subprocess.STDOUT,
            preexec_fn=preexec_fn,
        )
    except Exception as e:
        output.close()
        logger.warning("Failed to launch TensorBoard: %s", e)
        return {"error": str(e), "logdir": logdir}

    if sys.platform == "win32":
        _assign_to_win_job(proc)

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
