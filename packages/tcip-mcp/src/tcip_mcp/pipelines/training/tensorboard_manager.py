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
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, cast

from pydantic import BaseModel

from tcip_store import canonical_path

if sys.platform == "win32":
    import ctypes

logger = logging.getLogger(__name__)

BOARD_PATH = "/api/training/runs/{experiment_id}/tensorboard"
"""The path a run's, trial's or sweep's TensorBoard serves under, its id filled in.

The TensorBoard serves it on the loopback host at its own port, and the web backend forwards the
same path to the TensorBoard it launched.
"""

RELOAD_SELECTOR = "app-header-reload button.reload-button"
"""The installed TensorBoard frontend's header reload button, an internal name no version keeps."""


class TensorboardServing(BaseModel):
    """A TensorBoard serving a board.

    ``url`` is the path it serves under (:data:`BOARD_PATH`), ending in ``/``, which the
    TensorBoard answers at ``port`` on the loopback host and the web backend forwards for the
    TensorBoards it launched.
    """

    url: str
    port: int


class TensorboardFailure(BaseModel):
    """A TensorBoard launch that did not start: why, and what its child wrote when it ran."""

    error: str
    output: str | None


TensorboardLaunch = TensorboardServing | TensorboardFailure


@dataclass
class _Launched:
    """A TensorBoard child this process tracks: its process, its output capture and its answer.

    The process is the guardian where a guardian sits in front of the TensorBoard.
    """

    proc: subprocess.Popen
    output: IO[bytes]
    serving: TensorboardServing


_TB_PROCESSES: dict[str, _Launched] = {}

# How often a launch checks whether its child accepts connections: a small fraction of the
# 3.5 s TensorBoard took to first accept one on the development machine.
_ACCEPT_POLL_SECONDS = 0.1

# The guardian's escalation grace, passed as --term-grace, must stay more than a second under
# the wait below so its kill always lands before stop_tensorboard gives up.
_STOP_WAIT_SECONDS = 5.0
_GUARDIAN_TERM_GRACE_SECONDS = 2.0

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
        """Return the kernel32 handle for the job-object calls, its signatures set once."""
        global _win_kernel32
        if _win_kernel32 is None:
            dll = ctypes.WinDLL("kernel32", use_last_error=True)
            dll.CreateJobObjectW.restype = ctypes.c_void_p
            dll.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
            dll.SetInformationJobObject.restype = ctypes.c_int
            dll.SetInformationJobObject.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_uint32,
            ]
            dll.AssignProcessToJobObject.restype = ctypes.c_int
            dll.AssignProcessToJobObject.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            dll.CloseHandle.restype = ctypes.c_int
            dll.CloseHandle.argtypes = [ctypes.c_void_p]
            _win_kernel32 = dll
        return _win_kernel32

    def _ensure_win_job() -> int | None:
        """Create, once, the job object every child is assigned to.

        Returns the handle, or ``None`` with the failure reason logged. The kernel closes it, and
        every child still assigned, when this process's handles all close.
        """
        global _win_job_handle
        if _win_job_handle is not None:
            return _win_job_handle
        kernel32 = _get_win_kernel32()
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            logger.warning(
                "CreateJobObjectW failed: %s",
                ctypes.WinError(ctypes.get_last_error()),
            )
            return None
        info = _JobObjectExtendedLimitInformation()
        info.BasicLimitInformation.LimitFlags = (
            _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        if not kernel32.SetInformationJobObject(
            handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            logger.warning(
                "SetInformationJobObject failed: %s",
                ctypes.WinError(ctypes.get_last_error()),
            )
            kernel32.CloseHandle(handle)
            return None
        _win_job_handle = handle
        return handle

    def _assign_to_win_job(proc: subprocess.Popen) -> None:
        """Tie ``proc`` to the job object so it dies with this process; a failure is logged."""
        job = _ensure_win_job()
        if job is None:
            return
        kernel32 = _get_win_kernel32()
        # subprocess.Popen keeps the raw process handle here on Windows; nothing else exposes it.
        handle = cast(int, getattr(proc, "_handle"))
        if not kernel32.AssignProcessToJobObject(job, handle):
            logger.warning(
                "AssignProcessToJobObject failed for pid %s: %s",
                proc.pid,
                ctypes.WinError(ctypes.get_last_error()),
            )


def _stop_all_tracked() -> None:
    """Stop every child this process still has tracked, serially.

    Each stop waits up to twice ``_STOP_WAIT_SECONDS``; a stop that cannot confirm its kill is
    logged and the sweep continues.
    """
    for key in list(_TB_PROCESSES):
        result = stop_tensorboard(key)
        if result.get("status") == "kill_unconfirmed":
            logger.warning(
                "Exit sweep left TensorBoard key %s (pid=%s) running: its kill went unconfirmed",
                key,
                result.get("pid"),
            )


def _register_atexit_once() -> None:
    """Install the atexit hook that stops every tracked child, at most once per process."""
    global _atexit_registered
    if _atexit_registered:
        return
    _atexit_registered = True
    atexit.register(_stop_all_tracked)


def _tensorboard_argv(logdir: str, port: int, prefix: str) -> list[str]:
    """Return the TensorBoard child's command line over ``logdir`` at ``port`` under ``prefix``."""
    from tcip_mcp.web_client import LOOPBACK_HOST

    # tensorboard has no __main__.py; tensorboard.main runs under its own __main__ guard.
    return [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        logdir,
        "--port",
        str(port),
        "--host",
        LOOPBACK_HOST,
        "--path_prefix",
        prefix,
    ]


def _guardian_argv(argv: list[str]) -> list[str]:
    """Return ``argv`` wrapped to run under the POSIX lifetime guardian."""
    return [
        sys.executable,
        "-m",
        "tcip_mcp.pipelines.training.tensorboard_guardian",
        "--parent",
        str(os.getpid()),
        "--term-grace",
        str(_GUARDIAN_TERM_GRACE_SECONDS),
        "--",
        *argv,
    ]


def _collect_output(handle: IO[bytes]) -> str:
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


def _accepts(port: int) -> bool:
    """Return whether a connection to ``port`` on the loopback host is accepted."""
    from tcip_mcp.web_client import LOOPBACK_HOST

    try:
        socket.create_connection(
            (LOOPBACK_HOST, port), timeout=_ACCEPT_POLL_SECONDS
        ).close()
    except OSError:
        return False
    return True


def launch_tensorboard(directory: Path) -> TensorboardLaunch:
    """Launch, or reuse, the TensorBoard over the board of the run, trial or sweep ``directory``.

    The board is ``experiments.board_of``'s, and the process is tracked by its one spelling
    (``tcip_store.canonical_path``), which the child is handed; the child serves under
    :data:`BOARD_PATH` filled with the directory's name. The call returns once the child accepts
    a connection at its port, or once it has exited.

    Answers the :class:`TensorboardServing` already tracked for the board, or the new one's, or a
    :class:`TensorboardFailure` when the child died during startup (with what it wrote). An
    exception while taking a port, opening the output capture, starting the child or assigning
    its lifetime tie is a failed launch, the child killed and waited if one was started,
    answered as a failure with no ``output`` (a kill this cannot confirm within
    ``_STOP_WAIT_SECONDS`` named by pid in its ``error``); a platform tie call returning falsy
    for failure is logged and the child still serves.
    """
    from tcip_mcp.web_client import free_port

    key = logdir = _board_key(directory)

    entry = _running(key)
    if entry is not None:
        return entry.serving

    _register_atexit_once()
    prefix = BOARD_PATH.format(experiment_id=directory.name)
    output: IO[bytes] | None = None
    proc: subprocess.Popen | None = None
    try:
        port = free_port(6006)
        # The child outlives this call and nothing reads its streams, so they go to a temp file:
        # an undrained pipe blocks the writer as soon as the OS buffer fills.
        output = tempfile.TemporaryFile()
        argv = _tensorboard_argv(logdir, port, prefix)
        proc = subprocess.Popen(
            argv if sys.platform == "win32" else _guardian_argv(argv),
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        if sys.platform == "win32":
            _assign_to_win_job(proc)
    except Exception as e:
        if output is not None:
            output.close()
        error = str(e)
        if proc is not None and proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=_STOP_WAIT_SECONDS)
            except subprocess.TimeoutExpired:
                error = f"{error}; kill unconfirmed for pid {proc.pid}"
        logger.warning("Failed to launch TensorBoard: %s", error)
        return TensorboardFailure(error=error, output=None)

    while proc.poll() is None and not _accepts(port):
        time.sleep(_ACCEPT_POLL_SECONDS)
    if proc.poll() is not None:
        detail = _collect_output(output)
        logger.warning(
            "TensorBoard exited during startup (code=%s): %s",
            proc.returncode,
            detail,
        )
        return TensorboardFailure(
            error=f"tensorboard exited during startup with code {proc.returncode}",
            output=detail,
        )

    serving = TensorboardServing(url=f"{prefix}/", port=port)
    _TB_PROCESSES[key] = _Launched(proc=proc, output=output, serving=serving)
    logger.info(
        "TensorBoard started on port %d under %s (pid=%d, logdir=%s)",
        port,
        prefix,
        proc.pid,
        logdir,
    )
    return serving


def _board_key(directory: Path) -> str:
    """Return the one spelling of the run, trial or sweep ``directory``'s board, its tracking key.

    The board is ``experiments.board_of``'s, spelled by ``tcip_store.canonical_path``.
    """
    from tcip_mcp.experiments import board_of

    return canonical_path(board_of(directory))


def _running(key: str) -> _Launched | None:
    """Return the child tracked under ``key`` while it runs; one that has exited is dropped."""
    entry = _TB_PROCESSES.get(key)
    if entry is not None and entry.proc.poll() is not None:
        _release_output(_TB_PROCESSES.pop(key))
        return None
    return entry


def running(directory: Path) -> TensorboardServing | None:
    """Return the TensorBoard this process runs over ``directory``'s board, or ``None``.

    ``directory`` is a run, trial or sweep directory, its board the one
    :func:`launch_tensorboard` tracks.
    """
    entry = _running(_board_key(directory))
    return entry.serving if entry is not None else None


def stop_tensorboard(logdir: str) -> dict:
    """Stop the running TensorBoard process tracked under ``logdir``.

    A process still running is terminated and waited on up to ``_STOP_WAIT_SECONDS``, then
    force-killed and waited on the same bound again for the kill to be reaped; the entry is
    dropped from tracking either way, and a key tracking nothing answers
    ``{"status": "not_running"}``. A kill this cannot confirm reaped within that second wait is
    logged (the key and pid) and answered ``{"status": "kill_unconfirmed", "pid": ...}``.

    Where a guardian sits in front of the TensorBoard, ``entry.proc`` is the guardian, and a
    kill that reaches it can orphan a TensorBoard the guardian had not yet stopped; the answer is
    then ``stopped`` or ``kill_unconfirmed`` for the guardian itself.
    """
    key = canonical_path(logdir)
    if key not in _TB_PROCESSES:
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
                    "TensorBoard key %s (pid=%d) has a kill unconfirmed",
                    key,
                    entry.proc.pid,
                )
                _release_output(entry)
                return {"status": "kill_unconfirmed", "pid": entry.proc.pid}
    _release_output(entry)
    return {"status": "stopped", "pid": entry.proc.pid}
