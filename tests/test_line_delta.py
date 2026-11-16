"""Tests for tools/line_delta.py, run by its command line over a scratch repository."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tests import REPO_ROOT
from tests._scratch_repo_fixtures import commit_all, git, scratch_repo, write

LINE_DELTA = REPO_ROOT / "tools" / "line_delta.py"


def test_the_rev_dotdot_form_counts_each_area_against_rev(
    tmp_path: Path,
) -> None:
    """``<rev>..`` counts staged and unstaged changes to tracked files, one line per area."""
    repo = scratch_repo(tmp_path)
    (repo / "packages").mkdir()
    write(repo / "packages" / "gizmo.py", "a = 1\nb = 2\n")
    write(repo / "tests" / "test_gizmo.py", "x = 1\n")
    write(repo / "notes.txt", "one\ntwo\n")
    rev = commit_all(repo, "the areas before the change")
    write(repo / "packages" / "gizmo.py", "a = 1\nb = 3\nc = 4\n")
    git(repo, "add", "packages/gizmo.py")
    write(repo / "tests" / "test_gizmo.py", "x = 1\ny = 2\n")
    write(repo / "notes.txt", "one\n")

    result = subprocess.run(
        [sys.executable, str(LINE_DELTA), str(repo), f"{rev}.."],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "packages  +2 -1 net +1",
        "tests     +1 -0 net +1",
        "other     +0 -1 net -1",
    ]
