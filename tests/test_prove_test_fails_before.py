"""Tests for tools/prove_test_fails_before.py's refusal of a test_file outside both test trees.

A path outside tests/ and the frontend refuses, whether named relative or absolute, naming both
trees. They run by subprocess and need no git history.
"""

from __future__ import annotations

import subprocess
import sys

from tests import FRONTEND, REPO_ROOT

SCRIPT = REPO_ROOT / "tools" / "prove_test_fails_before.py"


def test_a_path_outside_tests_is_refused_by_name() -> None:
    outside = REPO_ROOT / "tools" / "prove_test_fails_before.py"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(outside)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 3, result.stdout + result.stderr
    assert "REFUSED" in result.stdout
    assert "prove_test_fails_before.py" in result.stdout
    assert "outside tests/" in result.stdout
    assert FRONTEND.relative_to(REPO_ROOT).as_posix() in result.stdout


def test_a_relative_and_absolute_dotdot_path_outside_tests_are_both_refused() -> (
    None
):
    """A relative and an absolute ".." path outside tests/ are both refused.

    A relative spelling that reads as inside tests/ before its ".." segments collapse
    (tests/../tools/x.py) and its absolute spelling each exit 3, REFUSED as outside tests/.
    """
    relative = "tests/../tools/prove_test_fails_before.py"
    absolute = str(
        REPO_ROOT / "tests" / ".." / "tools" / "prove_test_fails_before.py"
    )
    for spelling in (relative, absolute):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), spelling],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 3, result.stdout + result.stderr
        assert "REFUSED" in result.stdout
        assert "outside tests/" in result.stdout
