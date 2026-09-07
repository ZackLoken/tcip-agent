"""TensorBoard process management for training and HPO runs.

A launched child is tied to the life of the process that launched it, since nothing else stops
it once the launcher is gone. On Windows every child is assigned to one job object created with
``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``, so the kernel kills every assigned child the moment this
process's last handle to the job closes, on any parent death. On Linux and macOS each child runs
under a guardian process (``tensorboard_guardian``) that watches this process by pid and, on this
process's death, ends the child within the guardian's own ``_PARENT_POLL_SECONDS`` (about a
second) to detect the death, plus ``_GUARDIAN_TERM_GRACE_SECONDS`` for a child that ignores its
terminate signal, about three seconds together at today's values. Everywhere, and as a second
line of defense beside the platform tie, a normal interpreter exit runs an ``atexit`` hook that
stops every tracked child, which does not run when this process is killed rather than exiting on
its own.

``_GUARDIAN_TERM_GRACE_SECONDS`` (passed to the guardian as ``--term-grace``, which carries no
grace of its own) must stay more than a second under ``_STOP_WAIT_SECONDS``, the wait
``stop_tensorboard`` gives the guardian to end before force-killing it; a module-level check at
import raises if it does not, so the two never drift apart silently. With the constraint held,
the guardian's kill of a stubborn TensorBoard lands, and the guardian itself exits, before this
process's own wait gives up and reports stopped with TensorBoard still alive underneath it. The
guardian failing to exit within that first wait despite the margin is the one case left
uncovered; ``stop_tensorboard``'s own docstring states what its own kill of the guardian then
does to a TensorBoard still mid-escalation. The half-second startup grace below now also covers
the guardian's own start (about 0.2 to 0.3 s measured), leaving about 0.2 to 0.3 s of it as the
margin left for TensorBoard's own failure to surface in time.
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

# The guardian's escalation grace, passed as --term-grace, must stay more than a second under
# the wait below so its kill always lands before stop_tensorboard gives up.
_STOP_WAIT_SECONDS = 5.0
_GUARDIAN_TERM_GRACE_SECONDS = 2.0

if not _GUARDIAN_TERM_GRACE_SECONDS + 1.0 < _STOP_WAIT_SECONDS:
    raise RuntimeError(
        f"_GUARDIAN_TERM_GRACE_SECONDS ({_GUARDIAN_TERM_GRACE_SECONDS}) leaves no margin under "
        f"_STOP_WAIT_SECONDS ({_STOP_WAIT_SECONDS})"
    )

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
    """Stop every child this process still has tracked, for the atexit hook to call.

    Runs serially, one stop after another, each up to twice ``_STOP_WAIT_SECONDS`` (the wait for
    a clean exit, then the same wait again for the kill to be reaped). A stop that cannot confirm
    its kill within that reports so rather than raising, so this continues past any single
    stop's outcome instead of leaving the rest of the sweep unrun; each such answer is logged
    here too, so a TensorBoard the exit sweep leaves running is visible in the log rather than
    only in a return value nothing reads.
    """
    for key in list(_TB_PROCESSES):
        result = stop_tensorboard(key=key)
        if result.get("status") == "kill_unconfirmed":
            logger.warning(
                "Exit sweep left TensorBoard key %s (pid=%s) running: its kill went unconfirmed",
                key, result.get("pid"),
            )


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
        "--parent", str(os.getpid()), "--term-grace", str(_GUARDIAN_TERM_GRACE_SECONDS),
        "--", *argv,
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


def launch_tensorboard(logdir: str, key: str | None = None) -> dict:
    """Launch a TensorBoard process for the given log directory.

    ``key`` is the tracking key this process's children are indexed by, generic since a sweep
    and a trial each pass their own unnamespaced key while a run passes none and is keyed by its
    own log directory (the default below), so a caller-chosen record id can never collide with a
    sweep's or a trial's key.

    Returns dict with 'url', 'port', 'pid', 'logdir', 'lifetime_tie', or
    ``{'error': ..., 'output': ...}`` when the process died during startup, so a caller never
    advertises a URL nothing is serving. If TensorBoard is already running for this logdir,
    returns existing info. ``lifetime_tie`` is ``"job"``, ``"guardian"``, or ``"none: <reason>"``,
    a fact recorded for whichever caller wants it; no route reads it today. Two things can go
    wrong here: an exception during the launch or the tie assignment is a failed launch, the
    child killed and waited if one was started, reported back as ``error``, a kill this cannot
    confirm within ``_STOP_WAIT_SECONDS`` folded into that same message by pid rather than
    raising a ``TimeoutExpired`` out of this function; a platform tie call returning falsy for
    failure (the Windows job API's own convention) is not an exception, the launch still
    succeeds and ``lifetime_tie`` becomes ``"none: <reason>"`` instead.
    """
    logdir = str(Path(logdir).resolve())
    key = key or logdir

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
        error = str(e)
        if proc is not None and proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=_STOP_WAIT_SECONDS)
            except subprocess.TimeoutExpired:
                error = f"{error}; kill unconfirmed for pid {proc.pid}"
        logger.warning("Failed to launch TensorBoard: %s", error)
        return {"error": error, "logdir": logdir}

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


def stop_tensorboard(key: str | None = None, logdir: str | None = None) -> dict:
    """Stop a running TensorBoard process.

    Waits up to ``_STOP_WAIT_SECONDS`` for the process to end on its own, then force-kills it
    and waits the same bound again for the kill to be reaped, so the worst case is twice the
    wait, never unbounded; the entry is dropped from tracking either way, since whatever
    platform tie the launch got still holds the child regardless of whether this function's own
    wait confirms the kill (the job object still assigned on Windows, the guardian still
    watching its own child on POSIX). A kill this cannot confirm reaped within that second wait
    is logged (the key and pid) and answered ``{"status": "kill_unconfirmed", "pid": ...}``
    rather than left to raise a ``TimeoutExpired`` out of this function. On POSIX, ``entry.proc``
    is the guardian, not TensorBoard itself: reaching the second wait means the first one gave up
    on the guardian's own graceful exit, so this function's own ``kill()`` call lands on the
    guardian directly, a signal it cannot catch or forward, and if TensorBoard had not yet been
    stopped by the guardian's own escalation that TensorBoard is now orphaned; the answer is
    then ``stopped`` (the guardian's own death is reaped by the second wait) or
    ``kill_unconfirmed`` (it is not).
    """
    key = key or (str(Path(logdir).resolve()) if logdir else None)
    if not key or key not in _TB_PROCESSES:
        return {"status": "not_running"}

    entry = _TB_PROCESSES.pop(key)
    if entry.proc.poll() is None:
        entry.proc.terminate()
        try:
            entry.proc.wait(timeout=_STOP_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            entry.proc.kill()
            try:
                entry.proc.wait(timeout=_STOP_WAIT_SECONDS)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "TensorBoard key %s (pid=%d) has a kill unconfirmed", key, entry.proc.pid
                )
                _release_output(entry)
                return {"status": "kill_unconfirmed", "pid": entry.proc.pid}
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
