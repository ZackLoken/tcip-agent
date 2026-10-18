"""A ``faulthandler`` watchdog standing in for pytest-timeout's ``thread`` timer.

Through pytest-timeout's ``pytest_timeout_set_timer`` hook it takes the timeout that plugin
resolved for the item and, for the ``thread`` method outside a debugger, arms
``faulthandler.dump_traceback_later`` with it, which dumps every thread's stack and exits the
process when the timeout passes; the item's finish cancels it. Any other method is left to the
plugin. A run outside the suite loads it with ``-p tests.hung_test_watchdog``.
"""

from __future__ import annotations

import faulthandler

import pytest
import pytest_timeout
from _pytest.faulthandler import fault_handler_stderr_fd_key


@pytest.hookimpl
def pytest_timeout_set_timer(
    item: pytest.Item, settings: pytest_timeout.Settings
) -> bool | None:
    if settings.method != "thread" or pytest_timeout.is_debugging():
        return None
    faulthandler.dump_traceback_later(
        settings.timeout,
        exit=True,
        file=item.config.stash[fault_handler_stderr_fd_key],
    )
    # pytest-timeout's own cancel hook calls this attribute when the item finishes.
    item.cancel_timeout = faulthandler.cancel_dump_traceback_later  # type: ignore[attr-defined]
    return True
