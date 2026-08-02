"""Adopting a project unifies platform state under one ``<project>/.tcip/``.

``set_active_project`` repins ``TCIP_PROJECT_ROOT`` so the audit log, the experiment store,
and the model registry all resolve under the adopted project (self-contained + portable).
The conftest ``_restore_platform_root_env`` autouse fixture keeps the in-process repin from
leaking into other tests.
"""

from __future__ import annotations

from pathlib import Path


def _adopt(tmp_path, monkeypatch, name="hazelnut_catkin_valley") -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    proj = ws / name
    proj.mkdir()
    from tcip_mcp import workspace

    workspace.set_active_project(name)
    return proj


def test_adoption_repins_platform_root(tmp_path, monkeypatch):
    proj = _adopt(tmp_path, monkeypatch)
    from tcip_mcp.project_paths import project_root, resolve_state

    assert project_root() == proj
    assert resolve_state(Path(".tcip/audit.jsonl")) == proj / ".tcip" / "audit.jsonl"
    assert resolve_state(Path(".tcip/experiments")) == proj / ".tcip" / "experiments"


def test_experiment_and_registry_co_locate_under_adopted_project(tmp_path, monkeypatch):
    proj = _adopt(tmp_path, monkeypatch)
    clean_cwd = tmp_path / "cwd"
    clean_cwd.mkdir()
    monkeypatch.chdir(clean_cwd)  # so the "nothing leaked to cwd" check is meaningful
    from tcip_mcp import experiments

    experiments.create_experiment("exp_unify", {"model_source": {"builder": "x:y"}}, data_source="imgs")
    assert (proj / ".tcip" / "experiments" / "exp_unify").is_dir()

    ckpt = tmp_path / "model_best.pt"
    ckpt.write_bytes(b"fake checkpoint")
    # Auto-register path uses the experiments-module default (empty project_path) → platform root.
    result = experiments.register_model_from_experiment("exp_unify", str(ckpt))
    assert "error" not in result
    assert (proj / ".tcip" / "models" / "registry.json").is_file()
    # A different project's registry is untouched: nothing leaked to the repo root / cwd.
    assert not (Path.cwd() / ".tcip" / "models" / "registry.json").is_file()


def test_no_adoption_keeps_cwd_default(tmp_path, monkeypatch):
    # Without adoption (env unset), the platform root stays the cwd, the default
    # that keeps tests and un-pinned runs hermetic.
    monkeypatch.delenv("TCIP_PROJECT_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.project_paths import project_root

    assert project_root() == tmp_path
