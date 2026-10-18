"""A test's workspace is emptied when the test passes and kept when it fails.

This holds under the ``failed`` retention policy, through the suite's own ``tmp_path`` fixture
loaded into a child session the way ``tests/conftest.py`` loads it; the emptied root stays as
an empty directory.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

from tests import child_pytest

TWO_TESTS = textwrap.dedent(
    """
    from pathlib import Path


    def _note(tmp_path, name):
        root = tmp_path.parent.parent
        (root / "payload.bin").write_bytes(b"x" * 1024)
        with open(Path(__file__).with_name("roots.txt"), "a", encoding="utf-8") as f:
            f.write(f"{name}\\t{root}\\n")


    def test_passes(tmp_path):
        _note(tmp_path, "passes")


    def test_fails(tmp_path):
        _note(tmp_path, "fails")
        assert False
    """
)


def test_a_passed_tests_workspace_is_emptied_and_a_failed_ones_kept(tmp_path):
    test_file = tmp_path / "test_two.py"
    test_file.write_text(TWO_TESTS, encoding="utf-8")
    proc = child_pytest(
        tmp_path,
        test_file,
        "-p",
        "tests.conftest",
        "-o",
        "tmp_path_retention_policy=failed",
        timeout=120,
        env={k: v for k, v in os.environ.items() if k != "TCIP_MIN_TESTS"},
    )
    output = proc.stdout + proc.stderr
    assert "1 failed, 1 passed" in output, output
    roots = dict(
        line.split("\t")
        for line in (tmp_path / "roots.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    passed_root = Path(roots["passes"])
    assert passed_root.is_dir() and not any(passed_root.iterdir()), roots
    assert (Path(roots["fails"]) / "payload.bin").exists(), roots
