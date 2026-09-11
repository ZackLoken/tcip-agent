"""Tests for tools/prove_test_fails_before.py's failure classification, run against a scratch
git repository the test builds itself, so a baseline whose failure never reached the code under
test is distinguished from one whose failure is the assertion the guard actually names.

A baseline failure counts as GUARDS only when its headline is the test's own assertion failing
or the code under test raising, never a missing import, a fixture constructor called with a
keyword the baseline lacks, or a setup fixture erroring outright. Each case here builds a
two-revision scratch repository (a baseline commit
the fault sits in, a second commit with the fix and the guard test) and runs the real script
against it, exactly as a caller would.
"""

from __future__ import annotations

import importlib.util
import re
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
    as the code under test raising. ``_failure_kind`` reads the failure's phase, so this scores
    REFUSED, not GUARDS."""
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


# ── --baseline-from-working-tree ────────────────────────────────────────────


def _snapshot(repo: Path, test_file: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(repo / "tools" / "prove_test_fails_before.py"),
         test_file, "--baseline-from-working-tree"],
        cwd=str(repo), capture_output=True, text=True, timeout=60,
    )


def test_baseline_from_working_tree_captures_tracked_and_untracked_changes_untouched(tmp_path):
    """A tracked modification and an untracked file both land in the printed snapshot's tree,
    and the working tree, the index, and the stash list are all unchanged afterward."""
    repo = _scratch_repo(tmp_path)
    _write(repo / "widgets.py", "def double(x):\n    return x\n")
    _commit_all(repo, "double is a no-op")

    _write(repo / "widgets.py", "def double(x):\n    return x * 2\n")
    _write(repo / "untracked.py", "NEW = True\n")
    status_before = _git(repo, "status", "--porcelain")

    proc = _snapshot(repo, "tests/test_widgets.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    match = re.search(r"snapshot ([0-9a-f]{40})", proc.stdout)
    assert match, proc.stdout
    snapshot = match.group(1)

    assert "x * 2" in _git(repo, "show", f"{snapshot}:widgets.py")
    assert "NEW = True" in _git(repo, "show", f"{snapshot}:untracked.py")
    assert _git(repo, "status", "--porcelain") == status_before
    assert _git(repo, "stash", "list").strip() == ""


def test_baseline_from_working_tree_excludes_ignored_files(tmp_path):
    repo = _scratch_repo(tmp_path)
    _write(repo / ".gitignore", "ignored.txt\n")
    _write(repo / "widgets.py", "def double(x):\n    return x\n")
    _commit_all(repo, "double is a no-op")
    _write(repo / "ignored.txt", "should not appear\n")

    proc = _snapshot(repo, "tests/test_widgets.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    snapshot = re.search(r"snapshot ([0-9a-f]{40})", proc.stdout).group(1)

    listing = _git(repo, "ls-tree", "-r", "--name-only", snapshot)
    assert "ignored.txt" not in listing


def test_baseline_from_working_tree_together_with_baseline_is_refused(tmp_path):
    repo = _scratch_repo(tmp_path)
    _write(repo / "widgets.py", "def double(x):\n    return x\n")
    baseline = _commit_all(repo, "double is a no-op")

    proc = subprocess.run(
        [sys.executable, str(repo / "tools" / "prove_test_fails_before.py"),
         "tests/test_widgets.py", "--baseline-from-working-tree", "--baseline", baseline],
        cwd=str(repo), capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode != 0


def test_the_two_step_flow_snapshot_then_fix_then_baseline_reports_guards(tmp_path):
    """The honest shape: snapshot before the fix, apply the fix, then run normally with
    --baseline <the printed hash>, exactly as an operator would."""
    repo = _scratch_repo(tmp_path)
    _write(repo / "widgets.py", "def double(x):\n    return x\n")
    _commit_all(repo, "double is a no-op")

    _write(repo / "tests" / "test_widgets.py",
           "def test_double_doubles():\n"
           "    from widgets import double\n"
           "    assert double(3) == 6\n")

    proc = _snapshot(repo, "tests/test_widgets.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    snapshot = re.search(r"snapshot ([0-9a-f]{40})", proc.stdout).group(1)

    _write(repo / "widgets.py", "def double(x):\n    return x * 2\n")

    result = _run(repo, "tests/test_widgets.py", snapshot)
    assert result.returncode == EXIT["GUARDS"], result.stdout + result.stderr
    assert "GUARDS" in result.stdout


def test_a_bare_file_not_found_error_outside_the_assertion_is_refused_not_guards(tmp_path):
    """A helper module outside tests/ opens a data file the baseline commit does not carry (the
    tests/ overlay cannot supply it, living as it does outside tests/); the crash frame sits in
    that helper, outside tests/, so the code under test was never reached and this scores
    REFUSED, never GUARDS."""
    repo = _scratch_repo(tmp_path)
    _write(repo / "widgets.py", "def double(x):\n    return x\n")
    _write(repo / "loader.py",
           "def load_expected():\n"
           "    with open('data/expected.txt') as f:\n"
           "        return int(f.read())\n")
    baseline = _commit_all(repo, "double is a no-op, no recorded value yet")

    (repo / "data").mkdir()
    _write(repo / "data" / "expected.txt", "6\n")
    _write(repo / "widgets.py", "def double(x):\n    return x * 2\n")
    _write(repo / "tests" / "test_widgets.py",
           "def test_double_matches_the_recorded_value():\n"
           "    from widgets import double\n"
           "    from loader import load_expected\n"
           "    expected = load_expected()\n"
           "    assert double(3) == expected\n")
    _commit_all(repo, "double actually doubles, checked against a recorded value")

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    assert "REFUSED" in result.stdout
    assert "[unreached]" in result.stdout


def test_a_file_not_found_error_at_the_tests_own_assertion_line_guards(tmp_path):
    """The fix under test writes a file; the guard test's own assertion reads it back directly
    (never through a helper outside tests/), so at the baseline, before the fix, that read raises
    FileNotFoundError with its crash frame inside the test's own body. That is behavioral
    evidence, not a fixture-shaped discount: the code under test was reached (it ran and simply
    did not write the file), and the test's own line is what noticed."""
    repo = _scratch_repo(tmp_path)
    _write(repo / "widgets.py", "def write_summary(path):\n    pass\n")
    baseline = _commit_all(repo, "write_summary is a no-op")

    _write(repo / "widgets.py",
           "def write_summary(path):\n"
           "    with open(path, 'w') as f:\n"
           "        f.write('6')\n")
    _write(repo / "tests" / "test_widgets.py",
           "def test_write_summary_writes_the_expected_content(tmp_path):\n"
           "    from widgets import write_summary\n"
           "    out = tmp_path / 'summary.txt'\n"
           "    write_summary(out)\n"
           "    with open(out) as f:\n"
           "        assert f.read() == '6'\n")
    _commit_all(repo, "write_summary actually writes, checked by reading it back")

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["GUARDS"], result.stdout + result.stderr
    assert "GUARDS" in result.stdout


# ── --per-test-timeout ───────────────────────────────────────────────────────


def _load_tool():
    """Import the real module under test by path, for a pure classification-logic check that
    needs no scratch repository or subprocess of its own."""
    spec = importlib.util.spec_from_file_location("prove_test_fails_before_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_signal_method_timeout_message_is_a_timeout_not_a_behavioral_pass():
    """Where SIGALRM exists, pytest-timeout raises pytest.fail(PYTEST_FAILURE_MESSAGE), a Failed
    exception that would otherwise misread as a genuine assertion failure (GUARDS)."""
    tool = _load_tool()
    entry = {"headline": "Timeout (>5.0s) from pytest-timeout.", "phase": "call",
              "exc_typename": "Failed"}
    assert tool._failure_kind(entry, Path(".")) == "timeout"


def test_a_hung_test_under_per_test_timeout_reports_indeterminate_naming_it(tmp_path):
    """On this project's harness (no SIGALRM), pytest-timeout kills the process outright; the
    tool must still say which test hung, from the logstart marker it writes as each test starts,
    rather than reporting a bare no-outcome refusal that reads the same as any other crash."""
    repo = _scratch_repo(tmp_path)
    _write(repo / "tests" / "test_widgets.py",
           "import time\n"
           "\n"
           "\n"
           "def test_hangs_forever():\n"
           "    time.sleep(30)\n")
    _commit_all(repo, "a test that hangs")

    proc = subprocess.run(
        [sys.executable, str(repo / "tools" / "prove_test_fails_before.py"),
         "tests/test_widgets.py", "--baseline", "HEAD", "--per-test-timeout", "1"],
        cwd=str(repo), capture_output=True, text=True, timeout=60,
    )

    assert proc.returncode == EXIT["INDETERMINATE"], proc.stdout + proc.stderr
    assert "INDETERMINATE" in proc.stdout
    assert "test_hangs_forever" in proc.stdout


# ── build_module_inventory.py's root resolution works inside a .git-less sandbox ──


def _load_inventory_builder():
    """Import the real build_module_inventory.py by path, module-level REPO_ROOT resolution
    included; that resolution runs against this file's own location, not a test tree."""
    script = REPO / "tools" / "build_module_inventory.py"
    spec = importlib.util.spec_from_file_location("build_module_inventory_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repo_root_resolves_by_the_packages_and_tools_marker_with_no_git_present(tmp_path):
    """A git-archive sandbox carries no .git; find_repo_root resolves by an ancestor holding both
    packages/ and tools/ instead, the marker true of such a sandbox as well as a real checkout."""
    builder = _load_inventory_builder()
    tree = tmp_path / "tree"
    (tree / "packages").mkdir(parents=True)
    (tree / "tools").mkdir()

    assert builder.find_repo_root(tree) == tree


def test_repo_root_resolution_climbs_from_a_nested_subdirectory(tmp_path):
    builder = _load_inventory_builder()
    tree = tmp_path / "tree"
    nested = tree / "packages" / "tcip-mcp" / "src" / "tcip_mcp"
    nested.mkdir(parents=True)
    (tree / "tools").mkdir()

    assert builder.find_repo_root(nested) == tree


def test_a_key_error_on_a_package_result_guards(tmp_path):
    """An assertion that inspects a package result by key, where the baseline's result lacks
    that key, raises KeyError at the assert line itself, inside the test file: the code under
    test was reached and its result found wanting. Admits the call-phase residue rule (nothing
    but the call-signature-mismatch TypeError shape is fixture-shaped in the call phase): a crash
    frame inside tests/ that is neither AssertionError/Failed nor outside tests/ still scores
    GUARDS rather than falling through to fixture-shaped."""
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
