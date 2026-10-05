"""TensorBoard process management for training and HPO runs.

A launched child is tied to the life of the process that launched it. On Windows every child is
assigned to one job object created with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``, so the kernel kills
every assigned child when this process's last handle to the job closes. On Linux and macOS each
child runs under a guardian process (``tensorboard_guardian``) that watches this process by pid
and, on its death, ends the child within ``_PARENT_POLL_SECONDS`` plus
``_GUARDIAN_TERM_GRACE_SECONDS``. Everywhere, a normal interpreter exit runs an ``atexit`` hook
that stops every tracked child.
"""

from __future__ import annotations

import atexit
import logging
import os
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
    """A TensorBoard child this process tracks: its port, the URL it serves at, output capture
    and lifetime tie."""

    proc: subprocess.Popen
    port: int
    url: str
    output: IO[bytes]
    lifetime_tie: str


_TB_PROCESSES: dict[str, _Launched] = {}

# How long to let the child prove it survived before reporting a URL.
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
        """The kernel32 handle for the job-object calls, with pointer-sized signatures set once."""
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
    """Stop every child this process still has tracked, serially, each up to twice
    ``_STOP_WAIT_SECONDS``; a stop that cannot confirm its kill is logged and the sweep continues.
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
    """The TensorBoard child's command line over ``logdir`` at ``port``."""
    from tcip_mcp.web_client import LOOPBACK_HOST

    # tensorboard has no __main__.py; tensorboard.main runs under its own __main__ guard.
    return [
        sys.executable, "-m", "tensorboard.main", "--logdir", logdir,
        "--port", str(port), "--host", LOOPBACK_HOST, "--reload_interval", "5",
    ]


def _guardian_argv(argv: list[str]) -> list[str]:
    """``argv`` run under the POSIX lifetime guardian."""
    return [
        sys.executable, "-m", "tcip_mcp.pipelines.training.tensorboard_guardian",
        "--parent", str(os.getpid()), "--term-grace", str(_GUARDIAN_TERM_GRACE_SECONDS),
        "--", *argv,
    ]


def _key_of(key: str | None, logdir: str | None) -> str | None:
    """The tracking key a child is indexed by: ``key``, else ``logdir`` resolved."""
    return key or (str(Path(logdir).resolve()) if logdir else None)


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

    ``key`` is the tracking key this process's children are indexed by; a run passes none and is
    keyed by its own log directory.

    Returns dict with 'url', 'port', 'pid', 'logdir', 'lifetime_tie', or ``{'error': ..., 'output':
    ...}`` when the process died during startup. If TensorBoard is already running for this logdir,
    returns existing info. ``lifetime_tie`` is ``"job"``, ``"guardian"``, or ``"none: <reason>"``.
    An exception during the launch or the tie assignment is a failed launch, the child killed and
    waited if one was started, reported back as ``error`` (a kill this cannot confirm within
    ``_STOP_WAIT_SECONDS`` named by pid in that message); a platform tie call returning falsy for
    failure still succeeds with ``lifetime_tie`` ``"none: <reason>"``.
    """
    from tcip_mcp.web_client import LOOPBACK_HOST, free_port

    logdir = str(Path(logdir).resolve())
    key = cast(str, _key_of(key, logdir))

    entry = _running(key)
    if entry is not None:
        return {"url": entry.url, "port": entry.port, "pid": entry.proc.pid, "logdir": logdir,
                "lifetime_tie": entry.lifetime_tie}

    port = free_port(6006)
    _register_atexit_once()

    # The child outlives this call and nothing reads its streams, so they go to a temp file:
    # an undrained pipe blocks the writer as soon as the OS buffer fills.
    output = tempfile.TemporaryFile()
    argv = _tensorboard_argv(logdir, port)
    launch_argv = argv if sys.platform == "win32" else _guardian_argv(argv)

    proc: subprocess.Popen | None = None
    try:
        proc = subprocess.Popen(launch_argv, stdout=output, stderr=subprocess.STDOUT)
        if sys.platform == "win32":
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

    url = f"http://{LOOPBACK_HOST}:{port}"
    _TB_PROCESSES[key] = _Launched(proc=proc, port=port, url=url, output=output,
                                   lifetime_tie=lifetime_tie)
    logger.info("TensorBoard started: %s (pid=%d, logdir=%s)", url, proc.pid, logdir)
    return {"url": url, "port": port, "pid": proc.pid, "logdir": logdir,
            "lifetime_tie": lifetime_tie}


def _running(key: str) -> _Launched | None:
    """The child tracked under ``key`` while it runs; one that has exited is dropped."""
    entry = _TB_PROCESSES.get(key)
    if entry is not None and entry.proc.poll() is not None:
        _release_output(_TB_PROCESSES.pop(key))
        return None
    return entry


def running_url(key: str | None = None, logdir: str | None = None) -> str | None:
    """The URL the TensorBoard tracked under ``key`` (else under ``logdir``) serves at, ``None``
    when none runs."""
    tracked = _key_of(key, logdir)
    entry = _running(tracked) if tracked else None
    return entry.url if entry is not None else None


def stop_tensorboard(key: str | None = None, logdir: str | None = None) -> dict:
    """Stop a running TensorBoard process.

    Waits up to ``_STOP_WAIT_SECONDS`` for the process to end on its own, then force-kills it and
    waits the same bound again for the kill to be reaped; the entry is dropped from tracking either
    way. A kill this cannot confirm reaped within that second wait is logged (the key and pid) and
    answered ``{"status": "kill_unconfirmed", "pid": ...}``.

    On a ``"guardian"`` tie, ``entry.proc`` is the guardian, and a kill that reaches it can orphan
    a TensorBoard the guardian had not yet stopped; the answer is then ``stopped`` or
    ``kill_unconfirmed`` for the guardian itself.
    """
    key = _key_of(key, logdir)
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
