"""A bare pytest run from the repository's root collects the suite and nothing nested under it."""

import subprocess
import sys
from pathlib import Path

from tests import REPO_ROOT

SUITE_FILES = {
    "__init__.py": "",
    "conftest.py": "",
    "test_one.py": "def test_passes():\n    pass\n",
}


def write_suite(directory: Path) -> None:
    """Write ``SUITE_FILES`` into ``directory``, a package with a conftest and one test."""
    directory.mkdir(parents=True)
    for name, text in SUITE_FILES.items():
        (directory / name).write_text(text, encoding="utf-8", newline="\n")


def test_bare_collection_from_the_root_skips_a_nested_worktree(
    tmp_path: Path,
) -> None:
    """The checkout's own ``pytest.ini`` makes a no-argument run collect ``tests/`` alone."""
    (tmp_path / "pytest.ini").write_bytes(
        (REPO_ROOT / "pytest.ini").read_bytes()
    )
    write_suite(tmp_path / "tests")
    write_suite(tmp_path / "worktrees" / "nested" / "tests")

    collected = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert collected.returncode == 0, collected.stdout + collected.stderr
    lines = collected.stdout.splitlines()
    assert "tests/test_one.py::test_passes" in lines
    assert not [line for line in lines if line.startswith("worktrees/")]
