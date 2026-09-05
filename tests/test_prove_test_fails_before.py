"""Tests for scripts/prove_test_fails_before.py's test_file resolution, run by subprocess
against the repository's own history.

Before this fix an absolute test_file path named the working checkout's file regardless of the
tree pytest actually ran in, so a materialized baseline that should fail collected and ran the
fixed tree's test instead and reported VACUOUS for a real guard. A relative and an absolute path
naming the same file under tests/ must now report the same verdict, and a path outside tests/
must refuse rather than be materialized anywhere.

The two guard checks run against the real baseline commit ``BASELINE``, so the checkout they run
in must carry the repository's history: CI's python job checks out with ``fetch-depth: 0`` for
this module, and a depth-1 clone makes both report REFUSED rather than GUARDS.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "prove_test_fails_before.py"
TEST_FILE = REPO / "tests" / "test_orthomosaic_tools.py"
KNOWN_GUARD = (
    "test_deliver_orthomosaic_plant_counts_refuses_a_registry_csv_rewritten_after_registration"
)
BASELINE = "d766eaa6"


def _run_guard_check(test_file: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), test_file, "-k", KNOWN_GUARD, "--baseline", BASELINE],
        cwd=str(REPO), capture_output=True, text=True, timeout=300,
    )


def test_a_known_guard_named_by_its_absolute_path_reports_guards():
    result = _run_guard_check(str(TEST_FILE))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "GUARDS" in result.stdout


def test_the_same_guard_named_by_its_repo_relative_path_reports_guards():
    result = _run_guard_check("tests/test_orthomosaic_tools.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "GUARDS" in result.stdout


def test_a_path_outside_tests_is_refused_by_name():
    outside = REPO / "scripts" / "prove_test_fails_before.py"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(outside)],
        cwd=str(REPO), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 3, result.stdout + result.stderr
    assert "REFUSED" in result.stdout
    assert "prove_test_fails_before.py" in result.stdout
    assert "outside tests/" in result.stdout


def test_a_relative_and_absolute_dotdot_path_outside_tests_are_both_refused():
    """A relative spelling that reads as inside tests/ before its ".." segments collapse
    (tests/../scripts/x.py) must refuse the same way its absolute spelling does, not be
    materialized as if it were a real tests/ path."""
    relative = "tests/../scripts/prove_test_fails_before.py"
    absolute = str(REPO / "tests" / ".." / "scripts" / "prove_test_fails_before.py")
    for spelling in (relative, absolute):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), spelling],
            cwd=str(REPO), capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 3, result.stdout + result.stderr
        assert "REFUSED" in result.stdout
        assert "outside tests/" in result.stdout
