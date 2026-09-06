"""Platform-state-root pinning shared by the operator commands that call an ``@audited`` tool
function outside the MCP server: scan-dataset, inspect-compute-resources,
render-failure-cases, archive-project, and import-project, all built on
``tcip_mcp.project_paths.require_and_pin_platform_root`` so the resolve-or-refuse logic cannot drift
across five copies.

scan-dataset stands in for the shared mechanism, exercised as the real process entry point
(subprocess, matching how an operator actually runs it): the read-then-call shape, where
``folder_path`` stays what is scanned and ``--project`` decides only where the audit line and
any store the run creates land.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(args: list[str], cwd: Path, platform_root: str | None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if platform_root is None:
        env.pop("TCIP_STATE_ROOT", None)
    else:
        env["TCIP_STATE_ROOT"] = platform_root
    return subprocess.run(
        [sys.executable, "-m", "tcip_web.cli", "scan-dataset", *args],
        cwd=str(cwd), env=env, capture_output=True, text=True, timeout=60,
    )


def test_require_and_pin_platform_root_refuses_naming_both_when_neither_is_set(monkeypatch):
    monkeypatch.delenv("TCIP_STATE_ROOT", raising=False)
    from tcip_mcp.project_paths import require_and_pin_platform_root

    try:
        require_and_pin_platform_root(None)
        raised = None
    except SystemExit as exc:
        raised = exc
    assert raised is not None
    message = str(raised)
    assert "--project" in message
    assert "TCIP_STATE_ROOT" in message


def test_require_and_pin_platform_root_resolves_and_sets_the_environment_variable(monkeypatch, tmp_path):
    monkeypatch.delenv("TCIP_STATE_ROOT", raising=False)
    from tcip_mcp.project_paths import require_and_pin_platform_root

    resolved = require_and_pin_platform_root(str(tmp_path))

    assert resolved == tmp_path.resolve()
    assert os.environ["TCIP_STATE_ROOT"] == str(tmp_path.resolve())


def test_scan_dataset_refuses_from_an_unpinned_cwd_and_plants_no_store(tmp_path):
    """Neither --project nor $TCIP_STATE_ROOT names a root: the command must refuse before it
    ever resolves state, so no fresh .tcip lands under the operator's cwd."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    cwd = tmp_path / "operator_cwd"
    cwd.mkdir()

    result = _run([str(dataset)], cwd=cwd, platform_root=None)

    assert result.returncode != 0, result.stdout
    assert "TCIP_STATE_ROOT" in result.stderr
    assert not (cwd / ".tcip").exists()


def test_scan_dataset_with_project_writes_its_audit_store_under_the_project(tmp_path):
    """--project pins the root: the audit store lands under the named project, and the operator's
    cwd (a different directory entirely) gets nothing, even though the scanned folder is a third,
    unrelated directory."""
    project = tmp_path / "project"
    project.mkdir()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    cwd = tmp_path / "operator_cwd"
    cwd.mkdir()

    result = _run([str(dataset), "--project", str(project)], cwd=cwd, platform_root=None)

    assert result.returncode == 0, result.stderr
    audit_root = project / ".tcip"
    assert audit_root.is_dir() and any(audit_root.rglob("*")), (
        f"no state under {audit_root} for either store backend"
    )
    assert not (cwd / ".tcip").exists()
    assert not (dataset / ".tcip").exists()


def _resolve_platform_state_root():
    """The current resolver, found by attribute rather than ``from ... import`` so this proof
    still collects (and fails on its own assertion, not an ImportError) against a pre-rename
    baseline that only carries the old name."""
    import tcip_mcp.project_paths as pp

    return getattr(pp, "platform_state_root", None) or pp.project_root  # type: ignore[attr-defined]


def _require_and_pin_platform_root():
    import tcip_mcp.project_paths as pp

    return pp.require_and_pin_platform_root


def test_platform_state_root_ignores_the_old_env_var_name_and_commands_refuse_it(
    monkeypatch, tmp_path
):
    """Only $TCIP_PROJECT_ROOT set (the retired spelling): the resolver must not honor it
    (returns cwd, not the named path) and the operator commands' pin must refuse naming
    $TCIP_STATE_ROOT, since the rename is a clean break with no silent fallback to the old
    name."""
    monkeypatch.delenv("TCIP_STATE_ROOT", raising=False)
    # The retired variable names a directory the cwd is not, so a resolver honoring it
    # returns a path this assertion can tell apart from the honest cwd fallback.
    old_var_dir = tmp_path / "old-var-target"
    old_var_dir.mkdir()
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(old_var_dir))
    monkeypatch.chdir(tmp_path)

    resolve = _resolve_platform_state_root()
    assert resolve() == Path.cwd()
    assert resolve() != old_var_dir

    pin = _require_and_pin_platform_root()
    try:
        pin(None)
        raised = None
    except SystemExit as exc:
        raised = exc
    assert raised is not None
    assert "TCIP_STATE_ROOT" in str(raised)


def test_platform_state_root_honors_the_new_env_var_name(monkeypatch, tmp_path):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))

    from tcip_mcp.project_paths import platform_state_root

    assert platform_state_root() == tmp_path


def test_require_and_pin_platform_root_resolves_the_new_env_var_name(monkeypatch, tmp_path):
    monkeypatch.delenv("TCIP_STATE_ROOT", raising=False)
    from tcip_mcp.project_paths import require_and_pin_platform_root

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    resolved = require_and_pin_platform_root(None)

    assert resolved == tmp_path.resolve()
