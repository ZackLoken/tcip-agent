"""launch_tensorboard reports a URL only once the child has proved it survived startup.

The command itself is substituted (a dying one, then a living one) so the process
lifecycle is exercised for real without depending on a TensorBoard install.
"""

from __future__ import annotations

import subprocess
import sys


def test_launch_reports_the_failure_when_the_process_exits_immediately(monkeypatch, tmp_path):
    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    real_popen = subprocess.Popen

    def dying_popen(cmd, **kwargs):
        proc = real_popen(
            [sys.executable, "-c",
             "import sys; sys.stderr.write('tensorboard is not installed\\n'); sys.exit(3)"],
            **kwargs,
        )
        proc.wait(timeout=30)
        return proc

    monkeypatch.setattr(tb.subprocess, "Popen", dying_popen)

    info = tb.launch_tensorboard(str(tmp_path), run_id="dead-run")

    assert "url" not in info
    assert "exited during startup" in info["error"]
    assert "tensorboard is not installed" in info["output"]
    assert all(entry["key"] != "dead-run" for entry in tb.list_tensorboard())


def test_launch_returns_a_url_for_a_process_that_stays_up(monkeypatch, tmp_path):
    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    real_popen = subprocess.Popen

    def living_popen(cmd, **kwargs):
        return real_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)

    monkeypatch.setattr(tb.subprocess, "Popen", living_popen)

    info = tb.launch_tensorboard(str(tmp_path), run_id="live-run")
    try:
        assert info["url"] == f"http://localhost:{info['port']}"
        assert any(entry["key"] == "live-run" for entry in tb.list_tensorboard())
    finally:
        tb.stop_tensorboard(run_id="live-run")
    assert all(entry["key"] != "live-run" for entry in tb.list_tensorboard())
