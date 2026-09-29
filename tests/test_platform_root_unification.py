"""Adopting a project unifies platform state under one ``<project>/.tcip/``.

``activate_project`` repins ``TCIP_STATE_ROOT`` so the platform's own audit log (now the
project's, one file at one key), the experiment store, and the model registry all resolve under
the adopted project (self-contained + portable).
The conftest ``_restore_platform_root_env`` autouse fixture keeps the in-process repin from
leaking into other tests.
"""

from __future__ import annotations

from pathlib import Path


def _adopt(tmp_path, monkeypatch, name="currant_bud_valley") -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    proj = ws / name
    (proj / ".tcip").mkdir(parents=True)
    from tcip_mcp import workspace

    workspace.activate_project(name)
    return proj


def test_adoption_repins_platform_root(tmp_path, monkeypatch):
    proj = _adopt(tmp_path, monkeypatch)
    from tcip_mcp.project_paths import platform_state_root, resolve_state

    assert platform_state_root() == proj
    assert resolve_state(Path(".tcip/audit.jsonl")) == proj / ".tcip" / "audit.jsonl"
    assert resolve_state(Path(".tcip/experiments")) == proj / ".tcip" / "experiments"


def test_experiment_and_registry_co_locate_under_adopted_project(tmp_path, monkeypatch):
    proj = _adopt(tmp_path, monkeypatch)
    clean_cwd = tmp_path / "cwd"
    clean_cwd.mkdir()
    monkeypatch.chdir(clean_cwd)  # so the "nothing leaked to cwd" check is meaningful
    from tcip_mcp import experiments
    from tcip_mcp.model_registry import ModelRegistry
    from tests._verified_checkpoint_fixtures import finished_run

    # No root named: the run lands under the pinned platform root, the adopted project.
    run_dir = finished_run(None, experiment_id="exp_unify")
    assert run_dir == experiments.experiments_dir(proj) / "exp_unify"

    # The run's completion is its registration, read under the same project.
    assert "exp_unify" in {m["name"] for m in ModelRegistry(str(proj)).list_models()}
    assert list(clean_cwd.iterdir()) == []


def test_viz_mirrors_the_platform_root_env_var_name():
    """tcip_annotation must not import tcip_mcp, so it restates the platform-state-root
    variable name as its own constant; this holds the restatement equal to the declaration
    it mirrors so the two cannot drift apart silently."""
    from tcip_annotation.viz import _PLATFORM_ROOT_ENV
    from tcip_mcp.project_paths import ENV_VAR

    assert _PLATFORM_ROOT_ENV == ENV_VAR


def test_no_adoption_keeps_cwd_default(tmp_path, monkeypatch):
    # Without adoption (env unset), the platform root stays the cwd, the default
    # that keeps tests and un-pinned runs hermetic.
    monkeypatch.delenv("TCIP_STATE_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.project_paths import platform_state_root

    assert platform_state_root() == tmp_path


def test_app_import_alone_leaves_the_platform_root_at_cwd(tmp_path, monkeypatch):
    """Importing ``tcip_web.app`` is a served app's own bind, never an importer's: a fresh
    interpreter that only imports the module, with no ``TCIP_STATE_ROOT`` inherited, stays
    on its own cwd rather than silently repinning to the workspace marker's project (see
    tests/test_tcip_web_app_startup_root.py for when the pin actually happens)."""
    import subprocess
    import sys

    _adopt(tmp_path, monkeypatch)
    monkeypatch.delenv("TCIP_STATE_ROOT", raising=False)

    code = (
        "import tcip_web.app\n"
        "from tcip_mcp.project_paths import platform_state_root\n"
        "print(platform_state_root())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(tmp_path), timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(tmp_path)
