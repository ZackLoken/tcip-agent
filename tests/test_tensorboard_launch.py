"""launch_tensorboard reports a URL only once the child has proved it survived startup.

The command itself is substituted (a dying one, then a living one) so the process
lifecycle is exercised for real without depending on a TensorBoard install.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


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

    info = tb.launch_tensorboard(str(tmp_path), key="dead-run")

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

    info = tb.launch_tensorboard(str(tmp_path), key="live-run")
    try:
        assert info["url"] == f"http://localhost:{info['port']}"
        assert any(entry["key"] == "live-run" for entry in tb.list_tensorboard())
    finally:
        tb.stop_tensorboard(key="live-run")
    assert all(entry["key"] != "live-run" for entry in tb.list_tensorboard())


def test_launch_reports_the_platform_lifetime_tie(monkeypatch, tmp_path):
    """lifetime_tie names the mechanism a normal launch actually got, through the argv seam
    with a sleeping stand-in rather than a real TensorBoard install."""
    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    monkeypatch.setattr(
        tb, "_tensorboard_argv",
        lambda logdir, port: [sys.executable, "-c", "import time; time.sleep(30)"],
    )

    info = tb.launch_tensorboard(str(tmp_path), run_id="tie-run")
    try:
        assert info["lifetime_tie"] == ("job" if sys.platform == "win32" else "guardian")
    finally:
        tb.stop_tensorboard(run_id="tie-run")


def test_launch_reports_the_tie_disabled_under_the_test_seam(monkeypatch, tmp_path):
    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    monkeypatch.setattr(
        tb, "_tensorboard_argv",
        lambda logdir, port: [sys.executable, "-c", "import time; time.sleep(30)"],
    )
    monkeypatch.setattr(tb, "_DISABLE_LIFETIME_TIE", True)

    info = tb.launch_tensorboard(str(tmp_path), run_id="tie-disabled-run")
    try:
        assert info["lifetime_tie"] == "none: disabled for test"
    finally:
        tb.stop_tensorboard(run_id="tie-disabled-run")


def test_manager_refuses_to_import_when_the_grace_leaves_no_margin(tmp_path):
    """The module-level check holds the grace-under-wait constraint in code, not only in
    prose: a copy with the grace pushed past the margin fails at import."""
    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    source = Path(tb.__file__).read_text(encoding="utf-8")
    broken = source.replace(
        "_GUARDIAN_TERM_GRACE_SECONDS = 2.0", "_GUARDIAN_TERM_GRACE_SECONDS = 4.5"
    )
    assert broken != source
    module_path = tmp_path / "broken_tensorboard_manager.py"
    module_path.write_text(broken, encoding="utf-8")

    spec = importlib.util.spec_from_file_location("broken_tensorboard_manager", module_path)
    module = importlib.util.module_from_spec(spec)
    # the dataclass below needs its own module registered to resolve string annotations.
    sys.modules[spec.name] = module
    try:
        with pytest.raises(RuntimeError, match="_GUARDIAN_TERM_GRACE_SECONDS"):
            spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
