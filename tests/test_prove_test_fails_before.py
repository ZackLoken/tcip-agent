"""Tests for tools/prove_test_fails_before.py's test_file resolution, run by subprocess
against the repository's own history.

An absolute test_file path must name the file in the tree pytest actually runs in, not the
working checkout's own file: a relative and an absolute path naming the same file under tests/
must report the same verdict, and a path outside tests/ must refuse rather than be materialized
anywhere.

The two guard checks run against the real baseline commit ``BASELINE``, so the checkout they run
in must carry the repository's history: CI's python job checks out with ``fetch-depth: 0`` for
this module, and a depth-1 clone makes both report REFUSED rather than GUARDS.
"""

from __future__ import annotations

import subprocess
import sys

from tests import REPO_ROOT

SCRIPT = REPO_ROOT / "tools" / "prove_test_fails_before.py"
TEST_FILE = REPO_ROOT / "tests" / "test_admission_rule_of.py"
KNOWN_GUARD = "test_admission_rule_of_answers_none_for_no_stamp"
# A commit whose own tests/ tree holds the guard, paired with the commit before the guard's fix
# as the baseline, so every fixture the guard reaches is of the baseline's own era.
TEST_REV = "14d03827"
BASELINE = "b8ed53a4"


def _run_guard_check(test_file: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), test_file, "-k", KNOWN_GUARD, "--test-rev", TEST_REV,
         "--baseline", BASELINE],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300,
    )


def test_a_known_guard_named_by_its_absolute_path_reports_guards():
    result = _run_guard_check(str(TEST_FILE))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "GUARDS" in result.stdout


def test_the_same_guard_named_by_its_repo_relative_path_reports_guards():
    result = _run_guard_check("tests/test_admission_rule_of.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "GUARDS" in result.stdout


def test_a_path_outside_tests_is_refused_by_name():
    outside = REPO_ROOT / "tools" / "prove_test_fails_before.py"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(outside)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 3, result.stdout + result.stderr
    assert "REFUSED" in result.stdout
    assert "prove_test_fails_before.py" in result.stdout
    assert "outside tests/" in result.stdout


def test_a_relative_and_absolute_dotdot_path_outside_tests_are_both_refused():
    """A relative spelling that reads as inside tests/ before its ".." segments collapse
    (tests/../tools/x.py) must refuse the same way its absolute spelling does, not be
    materialized as if it were a real tests/ path."""
    relative = "tests/../tools/prove_test_fails_before.py"
    absolute = str(REPO_ROOT / "tests" / ".." / "tools" / "prove_test_fails_before.py")
    for spelling in (relative, absolute):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), spelling],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 3, result.stdout + result.stderr
        assert "REFUSED" in result.stdout
        assert "outside tests/" in result.stdout
