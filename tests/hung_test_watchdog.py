"""End a hung test from a thread that needs no interpreter lock.

pytest-timeout's ``thread`` method, its only method where SIGALRM does not exist (Windows), is a
Python ``threading.Timer``: its callback runs only once it holds the GIL, so a test blocked in a
C call that never releases the lock (Ray's driver waiting on its raylet, a ctypes call through
``PyDLL``) outlives its timeout for as long as the call does. ``faulthandler``'s watchdog is a C
thread that waits, dumps every thread's stack and exits the process without the GIL, so it ends
exactly those hangs. Wired through pytest-timeout's own ``pytest_timeout_set_timer`` hook, it
takes the timeout that plugin resolved for the item (command line, ini or marker) and claims only
the ``thread`` method; ``signal`` stays the plugin's own, since that one fails the single test and
lets the run go on. Under xdist the controller names the test the dead worker was running; alone,
the dumped stack names it. ``tests/conftest.py`` imports the hook, and a run outside the suite
loads it with ``-p tests.hung_test_watchdog``.
"""

from __future__ import annotations

import faulthandler

from _pytest.faulthandler import fault_handler_stderr_fd_key
import pytest
import pytest_timeout


@pytest.hookimpl
def pytest_timeout_set_timer(item: pytest.Item, settings: pytest_timeout.Settings) -> bool | None:
    if settings.method != "thread" or pytest_timeout.is_debugging():
        return None
    faulthandler.dump_traceback_later(
        settings.timeout, exit=True, file=item.config.stash[fault_handler_stderr_fd_key],
    )
    # pytest-timeout's own cancel hook calls this attribute when the item finishes.
    item.cancel_timeout = faulthandler.cancel_dump_traceback_later  # type: ignore[attr-defined]
    return True
