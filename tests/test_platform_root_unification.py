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
    import tcip_store as ts
    from tcip_mcp.model_registry import registry_index_key

    proj = _adopt(tmp_path, monkeypatch)
    clean_cwd = tmp_path / "cwd"
    clean_cwd.mkdir()
    monkeypatch.chdir(clean_cwd)  # so the "nothing leaked to cwd" check is meaningful
    from tcip_mcp import experiments

    experiments.create_experiment("exp_unify", {"model_source": {"builder": "x:y"}}, data_source="imgs")
    # config_key resolves against the pinned platform root with no root override, so its
    # existence proves the record landed under the adopted project, not wherever cwd is.
    assert ts.exists(experiments.config_key("exp_unify"))

    ckpt = tmp_path / "model_best.pt"
    ckpt.write_bytes(b"fake checkpoint")
    assert "error" not in experiments.complete_run("exp_unify", str(ckpt))
    # Auto-register path uses the experiments-module default (empty project_path) → platform root.
    result = experiments.register_model_from_experiment("exp_unify", str(ckpt))
    assert "error" not in result
    assert ts.exists(registry_index_key(proj))
    # A different project's registry is untouched: nothing leaked to the repo root / cwd.
    assert not ts.exists(registry_index_key(Path.cwd()))


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
