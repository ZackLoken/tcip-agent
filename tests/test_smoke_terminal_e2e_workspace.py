"""``tools/smoke_terminal_e2e.py`` must never bind against the machine's own workspace: it
sets ``TCIP_WORKSPACE`` to a workspace it is given before the served app's first request
resolves anything, so a run against a developer's real projects never audits into one.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "smoke_terminal_e2e.py"


def _load():
    spec = importlib.util.spec_from_file_location("smoke_terminal_e2e", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["smoke_terminal_e2e"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_binds_under_the_given_workspace_not_the_machines_live_marker(tmp_path, monkeypatch):
    """A live marker naming an adoptable project must not decide this process's root: the
    script's first request has to see the workspace it was given, not the one the machine
    already had active."""
    from tcip_mcp import audit, project_paths
    from tcip_mcp import workspace as ws_mod

    live_ws = tmp_path / "live-workspace"
    live_proj = live_ws / "elderberry_cyme_bloom"
    (live_proj / ".tcip").mkdir(parents=True)
    monkeypatch.setenv("TCIP_WORKSPACE", str(live_ws))
    monkeypatch.delenv("TCIP_STATE_ROOT", raising=False)
    ws_mod.activate_project("elderberry_cyme_bloom")
    monkeypatch.delenv("TCIP_STATE_ROOT", raising=False)
    project_paths.restore_binding(None)
    monkeypatch.setenv("TCIP_TERMINAL_CLI", "tcip-smoke-test-nonexistent-cli")

    scratch_ws = tmp_path / "scratch-workspace"
    mod = _load()
    result = mod.main(workspace=str(scratch_ws))

    assert result == 1
    binding = project_paths.root_binding()
    assert binding is not None
    assert binding.source != "marker"
    assert binding.root.resolve() != live_proj.resolve()
    assert audit.platform_audit_scope().resolve() != live_proj.resolve()
