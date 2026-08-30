"""Platform-state-root pinning shared by the operator scripts that call an ``@audited`` tool
function outside the MCP server: scripts/scan_dataset.py, inspect_compute_resources.py,
render_failure_cases.py, archive_project.py, and import_project.py, all built on
scripts/_script_root.pin_project_root so the resolve-or-refuse logic cannot drift across five
copies.

scan_dataset.py stands in for the shared mechanism, exercised as the real process entry point
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
SCAN_SCRIPT = REPO_ROOT / "scripts" / "scan_dataset.py"


def _run(args: list[str], cwd: Path, project_root: str | None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if project_root is None:
        env.pop("TCIP_PROJECT_ROOT", None)
    else:
        env["TCIP_PROJECT_ROOT"] = project_root
    return subprocess.run(
        [sys.executable, str(SCAN_SCRIPT), *args],
        cwd=str(cwd), env=env, capture_output=True, text=True, timeout=60,
    )


def test_pin_project_root_refuses_naming_both_when_neither_is_set(monkeypatch):
    monkeypatch.delenv("TCIP_PROJECT_ROOT", raising=False)
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from _script_root import pin_project_root

    try:
        pin_project_root(None)
        raised = None
    except SystemExit as exc:
        raised = exc
    assert raised is not None
    message = str(raised)
    assert "--project" in message
    assert "TCIP_PROJECT_ROOT" in message


def test_pin_project_root_resolves_and_sets_the_environment_variable(monkeypatch, tmp_path):
    monkeypatch.delenv("TCIP_PROJECT_ROOT", raising=False)
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from _script_root import pin_project_root

    resolved = pin_project_root(str(tmp_path))

    assert resolved == tmp_path.resolve()
    assert os.environ["TCIP_PROJECT_ROOT"] == str(tmp_path.resolve())


def test_scan_dataset_refuses_from_an_unpinned_cwd_and_plants_no_store(tmp_path):
    """Neither --project nor $TCIP_PROJECT_ROOT names a root: the script must refuse before it
    ever resolves state, so no fresh .tcip lands under the operator's cwd."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    cwd = tmp_path / "operator_cwd"
    cwd.mkdir()

    result = _run([str(dataset)], cwd=cwd, project_root=None)

    assert result.returncode != 0, result.stdout
    assert "TCIP_PROJECT_ROOT" in result.stderr
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

    result = _run([str(dataset), "--project", str(project)], cwd=cwd, project_root=None)

    assert result.returncode == 0, result.stderr
    audit_root = project / ".tcip"
    assert audit_root.is_dir() and any(audit_root.rglob("*")), (
        f"no state under {audit_root} for either store backend"
    )
    assert not (cwd / ".tcip").exists()
    assert not (dataset / ".tcip").exists()
