"""A test that hangs inside a C call holding the GIL is ended at its timeout and named, alone
and under xdist. Each case runs pytest in a child process on a one-test file written here, with
the watchdog loaded the way ``tests/conftest.py`` loads it, and bounds the child by the timeout
several times over: a child that outlives the bound is the defect the watchdog exists for.
"""

from __future__ import annotations

import textwrap
import time

import pytest

from tests import child_pytest

HANG_HOLDING_THE_GIL = textwrap.dedent(
    """
    import ctypes
    import ctypes.util
    import sys


    def test_hangs_holding_the_gil():
        # ctypes.PyDLL calls keep the GIL for their whole duration, unlike CDLL and WinDLL.
        if sys.platform == "win32":
            ctypes.PyDLL("kernel32").Sleep(60_000)
        else:
            ctypes.PyDLL(ctypes.util.find_library("c")).sleep(60)
    """
)

TIMEOUT_S = 2
BOUND_S = 10 * TIMEOUT_S


@pytest.mark.parametrize("workers", ["0", "2"])
def test_a_test_blocked_in_c_with_the_gil_is_ended_and_named(
    tmp_path, workers
):
    test_file = tmp_path / "test_gil_hang.py"
    test_file.write_text(HANG_HOLDING_THE_GIL, encoding="utf-8")
    started = time.monotonic()
    proc = child_pytest(
        tmp_path,
        test_file,
        "-n",
        workers,
        f"--timeout={TIMEOUT_S}",
        "-p",
        "tests.hung_test_watchdog",
        timeout=BOUND_S,
    )
    elapsed = time.monotonic() - started
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, output
    assert elapsed < BOUND_S, output
    assert "test_hangs_holding_the_gil" in output, output
