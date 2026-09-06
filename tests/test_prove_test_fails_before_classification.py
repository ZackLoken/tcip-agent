"""Tests for tools/prove_test_fails_before.py's failure classification, run against a scratch
git repository the test builds itself, so a baseline whose failure never reached the code under
test is distinguished from one whose failure is the assertion the guard actually names.

Before this fix a baseline failure counted as GUARDS whenever its headline was not a missing
import, whatever raised it: a fixture constructor called with a keyword the baseline lacks, or a
setup fixture erroring outright, scored the same as the test's own assertion failing or the code
under test raising. Each case here builds a two-revision scratch repository (a baseline commit
the fault sits in, a second commit with the fix and the guard test) and runs the real script
against it, exactly as a caller would.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "tools" / "prove_test_fails_before.py"
SCRIPT_SOURCE = SCRIPT.read_text(encoding="utf-8")

EXIT = {"GUARDS": 0, "VACUOUS": 1, "INDETERMINATE": 2, "REFUSED": 3}


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def _scratch_repo(tmp_path: Path) -> Path:
    """A git repository holding nothing but the script under test, so it resolves its own
    ``REPO`` to this scratch tree rather than the real one."""
    repo = tmp_path / "scratch"
    (repo / "tools").mkdir(parents=True)
    (repo / "tests").mkdir()
    _write(repo / "tools" / "prove_test_fails_before.py", SCRIPT_SOURCE)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    return repo


def _commit_all(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD").strip()


def _run(repo: Path, test_file: str, baseline: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(repo / "tools" / "prove_test_fails_before.py"),
         test_file, "--baseline", baseline],
        cwd=str(repo), capture_output=True, text=True, timeout=120,
    )


def test_a_fixture_shaped_type_error_in_the_test_body_is_refused(tmp_path):
    """A constructor called with a keyword the baseline does not accept raises TypeError at the
    call site, inside the test file itself: the code under test was never reached."""
    repo = _scratch_repo(tmp_path)
    _write(repo / "widgets.py",
           "class Widget:\n"
           "    def __init__(self, size):\n"
           "        self.size = size\n")
    baseline = _commit_all(repo, "widget carries no color")

    _write(repo / "widgets.py",
           "class Widget:\n"
           "    def __init__(self, size, color=None):\n"
           "        self.size = size\n"
           "        self.color = color\n")
    _write(repo / "tests" / "test_widgets.py",
           "def test_widget_carries_a_color():\n"
           "    from widgets import Widget\n"
           "    w = Widget(size=3, color='red')\n"
           "    assert w.color == 'red'\n")
    _commit_all(repo, "widget carries a color")

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    assert "REFUSED" in result.stdout
    assert "fixture-shaped" in result.stdout
    assert "1 failed on an error other than the assertion" in result.stdout
    assert "[fixture]" in result.stdout


def test_a_setup_fixture_error_is_refused(tmp_path):
    """A fixture that fails during setup never lets the test body run, so its failure carries the
    same weight as any other fixture-shaped error: a run to redo, not a guard."""
    repo = _scratch_repo(tmp_path)
    _write(repo / "widgets.py",
           "class Widget:\n"
           "    def __init__(self, size):\n"
           "        self.size = size\n")
    baseline = _commit_all(repo, "widget carries no color")

    _write(repo / "widgets.py",
           "class Widget:\n"
           "    def __init__(self, size, color=None):\n"
           "        self.size = size\n"
           "        self.color = color\n")
    _write(repo / "tests" / "test_widgets.py",
           "import pytest\n"
           "\n"
           "\n"
           "@pytest.fixture\n"
           "def thing():\n"
           "    from widgets import Widget\n"
           "    return Widget(size=3, color='red')\n"
           "\n"
           "\n"
           "def test_uses_thing(thing):\n"
           "    assert thing.color == 'red'\n")
    _commit_all(repo, "widget carries a color")

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    assert "REFUSED" in result.stdout
    assert "fixture-shaped" in result.stdout


def test_an_assertion_failure_guards(tmp_path):
    """The test's own assert failing on the value the baseline actually produced is exactly the
    evidence a guard is built to carry. Admits-valid-work coverage rather than a guard for any
    defect fixed here: a call-phase ``AssertionError`` already scored GUARDS under the
    classification this file's other cases fix."""
    repo = _scratch_repo(tmp_path)
    _write(repo / "widgets.py", "def double(x):\n    return x\n")
    baseline = _commit_all(repo, "double is a no-op")

    _write(repo / "widgets.py", "def double(x):\n    return x * 2\n")
    _write(repo / "tests" / "test_widgets.py",
           "def test_double_doubles():\n"
           "    from widgets import double\n"
           "    assert double(3) == 6\n")
    _commit_all(repo, "double actually doubles")

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["GUARDS"], result.stdout + result.stderr
    assert "GUARDS" in result.stdout
    assert "[behavioral]" in result.stdout


def test_a_value_error_raised_from_code_under_test_guards(tmp_path):
    """An exception the code under test raises itself, uncaught, is the code under test doing
    the wrong thing: behavioral evidence even though the headline is not AssertionError.
    Admits-valid-work coverage rather than a guard for any defect fixed here: a crash frame
    outside tests/ already scored GUARDS under the classification this file's other cases fix."""
    repo = _scratch_repo(tmp_path)
    _write(repo / "widgets.py",
           "def normalize(value):\n"
           "    if value <= 0:\n"
           "        raise ValueError('must be positive')\n"
           "    return value\n")
    baseline = _commit_all(repo, "normalize wrongly rejects zero")

    _write(repo / "widgets.py",
           "def normalize(value):\n"
           "    if value < 0:\n"
           "        raise ValueError('must be positive')\n"
           "    return value\n")
    _write(repo / "tests" / "test_widgets.py",
           "def test_normalize_permits_zero():\n"
           "    from widgets import normalize\n"
           "    assert normalize(0) == 0\n")
    _commit_all(repo, "normalize permits zero")

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["GUARDS"], result.stdout + result.stderr
    assert "GUARDS" in result.stdout
    assert "[behavioral]" in result.stdout
    assert "ValueError" in result.stdout


def test_a_fixture_calling_package_code_that_raises_is_refused(tmp_path):
    """A fixture that builds its value by calling package code, and that call raises before the
    fixture returns, never lets the test body run: fixture-shaped even though the crash frame
    sits in the package's own file, outside tests/, which the crash-frame test alone would read
    as the code under test raising. Before this fix, ``_failure_kind`` never read the failure's
    phase and scored this GUARDS."""
    repo = _scratch_repo(tmp_path)
    _write(repo / "widgets.py",
           "def make_widget():\n"
           "    raise ValueError('widgets are not ready yet')\n")
    baseline = _commit_all(repo, "widgets are not buildable")

    _write(repo / "widgets.py",
           "class Widget:\n"
           "    pass\n"
           "\n"
           "\n"
           "def make_widget():\n"
           "    return Widget()\n")
    _write(repo / "tests" / "test_widgets.py",
           "import pytest\n"
           "\n"
           "\n"
           "@pytest.fixture\n"
           "def widget():\n"
           "    from widgets import make_widget\n"
           "    return make_widget()\n"
           "\n"
           "\n"
           "def test_widget_builds(widget):\n"
           "    assert widget is not None\n")
    _commit_all(repo, "widgets are buildable")

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    assert "REFUSED" in result.stdout
    assert "fixture-shaped" in result.stdout
    assert "[fixture]" in result.stdout


def test_a_key_error_on_a_package_result_guards(tmp_path):
    """An assertion that inspects a package result by key, where the baseline's result lacks
    that key, raises KeyError at the assert line itself, inside the test file: the code under
    test was reached and its result found wanting. Admits the call-phase residue rule (nothing
    but the call-signature-mismatch TypeError shape is fixture-shaped in the call phase); before
    this fix, a crash frame inside tests/ that was neither AssertionError/Failed nor outside
    tests/ fell through to fixture-shaped and this scored REFUSED instead."""
    repo = _scratch_repo(tmp_path)
    _write(repo / "widgets.py",
           "def describe():\n"
           "    return {'size': 3}\n")
    baseline = _commit_all(repo, "describe carries no color")

    _write(repo / "widgets.py",
           "def describe():\n"
           "    return {'size': 3, 'color': 'red'}\n")
    _write(repo / "tests" / "test_widgets.py",
           "def test_describe_carries_a_color():\n"
           "    from widgets import describe\n"
           "    assert describe()['color'] == 'red'\n")
    _commit_all(repo, "describe carries a color")

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["GUARDS"], result.stdout + result.stderr
    assert "GUARDS" in result.stdout
    assert "[behavioral]" in result.stdout
    assert "KeyError" in result.stdout
