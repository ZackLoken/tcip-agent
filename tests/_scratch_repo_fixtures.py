"""Scratch git repositories the tool tests build, each holding a copy of the fail-before script.

The copy acts on the repository that holds it, so a test runs it there as its users do.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from tests import REPO_ROOT

SCRIPT_RELATIVE_PATH = "tools/prove_test_fails_before.py"
"""The fail-before script's path under a repository's root, this one's and each scratch copy's."""
SCRIPT = REPO_ROOT / SCRIPT_RELATIVE_PATH
SCRIPT_SOURCE = SCRIPT.read_text(encoding="utf-8")


def git(repo: Path, *args: str) -> str:
    """Return git's standard output for ``args`` run in ``repo``, asserting git exited 0."""
    proc = subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` with LF line endings."""
    path.write_text(text, encoding="utf-8", newline="\n")


def scratch_repo(tmp_path: Path) -> Path:
    """Make a git repository holding nothing but the script under test; return its root."""
    repo = tmp_path / "scratch"
    copy = repo / SCRIPT_RELATIVE_PATH
    copy.parent.mkdir(parents=True)
    (repo / "tests").mkdir()
    write(copy, SCRIPT_SOURCE)
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test")
    return repo


def commit_all(repo: Path, message: str) -> str:
    """Stage everything in ``repo``, commit it with ``message`` and return the new HEAD."""
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD").strip()
