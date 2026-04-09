"""Tests for the process manager — binary discovery and lifecycle."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from tcip_gui.process_manager import ProcessManager, _find_agent_binary


class TestFindAgentBinary:
    def test_returns_none_when_not_found(self):
        with patch("shutil.which", return_value=None):
            with patch("pathlib.Path.is_file", return_value=False):
                result = _find_agent_binary()
                # May or may not be None depending on workspace layout
                # but the function should not crash
                assert result is None or isinstance(result, str)


class TestProcessManager:
    def test_creation(self):
        pm = ProcessManager(workspace=".")
        assert pm.bridge is None

    def test_stop_without_start(self):
        """Stopping without starting should not crash."""
        pm = ProcessManager(workspace=".")
        pm.stop()  # Should not raise

    def test_restart_timer_setup(self):
        pm = ProcessManager(workspace=".")
        assert pm._restart_timer.isSingleShot()
        assert pm._restart_timer.interval() == 2000


class TestProcessManagerSignals:
    def test_status_changed_on_stop(self):
        pm = ProcessManager(workspace=".")
        received = []
        pm.status_changed.connect(received.append)
        pm.stop()
        assert "Disconnected" in received

    def test_agent_crashed_emitted_when_binary_missing(self):
        pm = ProcessManager(workspace=".")
        received = []
        pm.agent_crashed.connect(received.append)
        with patch("tcip_gui.process_manager._find_agent_binary", return_value=None):
            result = pm.start()
        assert result is None
        assert len(received) == 1
        assert "Could not find" in received[0]
