"""TensorBoard process management for training and HPO runs.

A launched child is tied to the life of the process that launched it, since nothing else stops
it once the launcher is gone. On Windows every child is assigned to one job object created with
``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``, so the kernel kills every assigned child the moment this
process's last handle to the job closes, on any parent death. On Linux and macOS each child runs
under a guardian process (``tensorboard_guardian``) that watches this process by pid and, on this
process's death, ends the child within about three seconds (one second to detect the death, plus
the guardian's own escalation grace for a child that ignores its terminate signal). Everywhere,
and as a second line of defense beside the platform tie, a normal interpreter exit runs an
``atexit`` hook that stops every tracked child, which does not run when this process is killed
rather than exiting on its own.

``stop_tensorboard`` sends the guardian a terminate signal and waits five seconds for it to end
before force-killing it; the guardian's own escalation grace against a stubborn TensorBoard stays
well under that five seconds, so the guardian's kill lands, and the guardian itself exits, before
this process's own wait gives up and reports stopped with TensorBoard still alive underneath it.
The half-second startup grace below now also covers the guardian's own start (about 0.15 s
measured), leaving roughly 0.35 s of it as the margin left for TensorBoard's own failure to
surface in time.
"""

from __future__ import annotations

import atexit
import logging
import os
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, cast

if sys.platform == "win32":
    import ctypes

logger = logging.getLogger(__name__)


@dataclass
class _Launched:
    """A TensorBoard child this process is tracking: its port, output capture, and lifetime tie
    live here, not stuffed onto the ``Popen`` object, since an attribute added to it at runtime
    is invisible to callers that only know the process's declared type."""

    proc: subprocess.Popen
    port: int
    output: IO[bytes]
    lifetime_tie: str


_TB_PROCESSES: dict[str, _Launched] = {}

# How long to let the child prove it survived before reporting a URL; anything slower is
# caught later by the poll in ``list_tensorboard``.
_STARTUP_GRACE_SECONDS = 0.5

# Test seam: the lifetime test helper sets this to launch bare, with no platform tie at all,
# so a test can prove the atexit hook in isolation from the job object or the guardian.
_DISABLE_LIFETIME_TIE = False

_atexit_registered = False

if sys.platform == "win32":
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9

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
        """The kernel32 handle for the job-object calls, with pointer-sized signatures set once
        so a 64-bit handle is never truncated by ctypes' default 32-bit return type."""
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

    def _ensure_win_job() -> tuple[int | None, str | None]:
        """Create, once, the job object every child is assigned to.

        Returns the handle and ``None`` on success, or ``None`` and the failure reason. The
        kernel closes it, and every child still assigned, when this process's handles all close.
        """
        global _win_job_handle
        if _win_job_handle is not None:
            return _win_job_handle, None
        kernel32 = _get_win_kernel32()
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            reason = f"CreateJobObjectW failed: {ctypes.WinError(ctypes.get_last_error())}"
            logger.warning(reason)
            return None, reason
        info = _JobObjectExtendedLimitInformation()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            handle, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info)
        ):
            reason = f"SetInformationJobObject failed: {ctypes.WinError(ctypes.get_last_error())}"
            logger.warning(reason)
            kernel32.CloseHandle(handle)
            return None, reason
        _win_job_handle = handle
        return handle, None

    def _assign_to_win_job(proc: subprocess.Popen) -> str | None:
        """Tie ``proc`` to the job object so it dies with this process.

        Returns ``None`` on success, or the failure reason; a failed assignment leaves the
        child tracked exactly as it would be without a job at all.
        """
        job, reason = _ensure_win_job()
        if job is None:
            return reason
        kernel32 = _get_win_kernel32()
        # subprocess.Popen keeps the raw process handle here on Windows; nothing else exposes it.
        handle = cast(int, getattr(proc, "_handle"))
        if not kernel32.AssignProcessToJobObject(job, handle):
            reason = (
                f"AssignProcessToJobObject failed for pid {proc.pid}: "
                f"{ctypes.WinError(ctypes.get_last_error())}"
            )
            logger.warning(reason)
            return reason
        return None


def _stop_all_tracked() -> None:
    """Stop every child this process still has tracked, for the atexit hook to call."""
    for key in list(_TB_PROCESSES):
        stop_tensorboard(run_id=key)


def _register_atexit_once() -> None:
    """Install the atexit hook that stops every tracked child, at most once per process."""
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


def _guardian_argv(argv: list[str]) -> list[str]:
    """Wrap ``argv`` to run under the POSIX lifetime guardian instead of bare."""
    return [
        sys.executable, "-m", "tcip_mcp.pipelines.training.tensorboard_guardian",
        "--parent", str(os.getpid()), "--", *argv,
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

    Returns dict with 'url', 'port', 'pid', 'logdir', 'lifetime_tie', or
    ``{'error': ..., 'output': ...}`` when the process died during startup, so a caller never
    advertises a URL nothing is serving. If TensorBoard is already running for this logdir,
    returns existing info. ``lifetime_tie`` is ``"job"``, ``"guardian"``, or ``"none: <reason>"``,
    a fact recorded for whichever caller wants it; no route reads it today. Two things can go
    wrong here: an exception during the launch or the tie assignment is a failed launch, the
    child killed and waited if one was started, reported back as ``error``; a platform tie call
    returning falsy for failure (the Windows job API's own convention) is not an exception, the
    launch still succeeds and ``lifetime_tie`` becomes ``"none: <reason>"`` instead.
    """
    logdir = str(Path(logdir).resolve())
    key = run_id or logdir

    # Check if already running
    if key in _TB_PROCESSES:
        entry = _TB_PROCESSES[key]
        if entry.proc.poll() is None:  # still alive
            # Recover port from stored info
            return {"url": f"http://localhost:{entry.port}", "port": entry.port,
                    "pid": entry.proc.pid, "logdir": logdir, "lifetime_tie": entry.lifetime_tie}
        else:
            _release_output(_TB_PROCESSES.pop(key))

    port = _find_free_port()
    _register_atexit_once()

    # The child outlives this call and nothing reads its streams, so they go to a temp file:
    # an undrained pipe blocks the writer as soon as the OS buffer fills.
    output = tempfile.TemporaryFile()
    argv = _tensorboard_argv(logdir, port)
    launch_argv = argv if (_DISABLE_LIFETIME_TIE or sys.platform == "win32") else _guardian_argv(argv)

    proc: subprocess.Popen | None = None
    try:
        proc = subprocess.Popen(launch_argv, stdout=output, stderr=subprocess.STDOUT)
        if _DISABLE_LIFETIME_TIE:
            lifetime_tie = "none: disabled for test"
        elif sys.platform == "win32":
            failure = _assign_to_win_job(proc)
            lifetime_tie = "job" if failure is None else f"none: {failure}"
        else:
            lifetime_tie = "guardian"
    except Exception as e:
        output.close()
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
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

    _TB_PROCESSES[key] = _Launched(proc=proc, port=port, output=output, lifetime_tie=lifetime_tie)
    logger.info("TensorBoard started: http://localhost:%d (pid=%d, logdir=%s)", port, proc.pid, logdir)
    return {
        "url": f"http://localhost:{port}",
        "port": port,
        "pid": proc.pid,
        "logdir": logdir,
        "lifetime_tie": lifetime_tie,
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
            # Five seconds gives the guardian, whose own escalation grace stays well under
            # this, time to finish killing a stubborn TensorBoard and exit before this gives up.
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
