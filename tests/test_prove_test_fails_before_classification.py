"""Tests for tools/prove_test_fails_before.py's failure classification.

A baseline failure that does not reach the code under test is told apart from one at the guard's
own assertion. Each case builds a two-revision scratch repository (a baseline commit the fault
sits in, a second commit with the fix and the guard test) and runs the real script against it.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests import REPO_ROOT
from tools.prove_test_fails_before import _failure_kind

SCRIPT = REPO_ROOT / "tools" / "prove_test_fails_before.py"
SCRIPT_SOURCE = SCRIPT.read_text(encoding="utf-8")

EXIT = {"GUARDS": 0, "VACUOUS": 1, "INDETERMINATE": 2, "REFUSED": 3}


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def _scratch_repo(tmp_path: Path) -> Path:
    """Make a git repository holding nothing but the script under test; return its root."""
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


def _run(
    repo: Path, test_file: str, baseline: str
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(repo / "tools" / "prove_test_fails_before.py"),
            test_file,
            "--baseline",
            baseline,
        ],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_a_setup_fixture_error_is_refused(tmp_path: Path) -> None:
    """A setup fixture error is refused.

    A fixture that fails during setup is reported REFUSED, fixture-shaped.
    """
    repo = _scratch_repo(tmp_path)
    _write(
        repo / "widgets.py",
        "class Widget:\n"
        "    def __init__(self, size):\n"
        "        self.size = size\n",
    )
    baseline = _commit_all(repo, "widget carries no color")

    _write(
        repo / "widgets.py",
        "class Widget:\n"
        "    def __init__(self, size, color=None):\n"
        "        self.size = size\n"
        "        self.color = color\n",
    )
    _write(
        repo / "tests" / "test_widgets.py",
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
        "    assert thing.color == 'red'\n",
    )
    _commit_all(repo, "widget carries a color")

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    assert "REFUSED" in result.stdout
    assert FIXTURE_SENTENCE in result.stdout
    assert CALL_PHASE_SENTENCE not in result.stdout
    assert CAUSE_CLAUSE not in result.stdout


def test_an_assertion_failure_guards(tmp_path: Path) -> None:
    """An assertion failure at the baseline scores GUARDS.

    The test's own assert failing on the value the baseline produced is reported GUARDS,
    behavioral.
    """
    repo = _scratch_repo(tmp_path)
    _write(repo / "widgets.py", "def double(x):\n    return x\n")
    baseline = _commit_all(repo, "double is a no-op")

    _write(repo / "widgets.py", "def double(x):\n    return x * 2\n")
    _write(
        repo / "tests" / "test_widgets.py",
        "def test_double_doubles():\n"
        "    from widgets import double\n"
        "    assert double(3) == 6\n",
    )
    _commit_all(repo, "double actually doubles")

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["GUARDS"], result.stdout + result.stderr
    assert "GUARDS" in result.stdout
    assert "[behavioral]" in result.stdout


def test_a_value_error_raised_from_code_under_test_guards(
    tmp_path: Path,
) -> None:
    """A ValueError raised from the code under test scores GUARDS.

    An uncaught ValueError the code under test raises is reported GUARDS, behavioral, naming
    ValueError.
    """
    repo = _scratch_repo(tmp_path)
    _write(
        repo / "widgets.py",
        "def normalize(value):\n"
        "    if value <= 0:\n"
        "        raise ValueError('must be positive')\n"
        "    return value\n",
    )
    baseline = _commit_all(repo, "normalize wrongly rejects zero")

    _write(
        repo / "widgets.py",
        "def normalize(value):\n"
        "    if value < 0:\n"
        "        raise ValueError('must be positive')\n"
        "    return value\n",
    )
    _write(
        repo / "tests" / "test_widgets.py",
        "def test_normalize_permits_zero():\n"
        "    from widgets import normalize\n"
        "    assert normalize(0) == 0\n",
    )
    _commit_all(repo, "normalize permits zero")

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["GUARDS"], result.stdout + result.stderr
    assert "GUARDS" in result.stdout
    assert "[behavioral]" in result.stdout
    assert "ValueError" in result.stdout


def test_a_fixture_calling_package_code_that_raises_is_refused(
    tmp_path: Path,
) -> None:
    """A fixture calling package code that raises is refused.

    A fixture whose call into package code raises before the fixture returns is reported REFUSED,
    fixture-shaped, though the crash frame sits in the package's own file, outside tests/.
    """
    repo = _scratch_repo(tmp_path)
    _write(
        repo / "widgets.py",
        "def make_widget():\n"
        "    raise ValueError('widgets are not ready yet')\n",
    )
    baseline = _commit_all(repo, "widgets are not buildable")

    _write(
        repo / "widgets.py",
        "class Widget:\n"
        "    pass\n"
        "\n"
        "\n"
        "def make_widget():\n"
        "    return Widget()\n",
    )
    _write(
        repo / "tests" / "test_widgets.py",
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
        "    assert widget is not None\n",
    )
    _commit_all(repo, "widgets are buildable")

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    assert "REFUSED" in result.stdout
    assert FIXTURE_SENTENCE in result.stdout
    assert "[fixture]" in result.stdout


# ── a test's own code raising TypeError ──────────────────────────────────────


def _fn(params: str, returns: str = "1") -> str:
    return f"def f({params}):\n    return {returns}\n"


# Each wrong call is a real call: the baseline's callable lacks what the head's has, and the test's
# call fits the head, so the TypeError is the interpreter's own message for that baseline.
WRONG_CALLS = {
    "unexpected keyword": (
        _fn("a, b"),
        _fn("a, b, color=None"),
        "f(1, 2, color='red')",
    ),
    "positional missing": (_fn("a, b"), _fn("a, b=0"), "f(1)"),
    "keyword-only missing": (_fn("a, *, key"), _fn("a, *, key=None"), "f(1)"),
    "too many positional": (_fn("a, b"), _fn("a, b, c"), "f(1, 2, 3)"),
    "too many positional with defaults": (
        _fn("a, b=0"),
        _fn("a, b=0, c=0"),
        "f(1, 2, 3)",
    ),
    "too many positional with a keyword-only given": (
        _fn("a, *, key=0"),
        _fn("a, b=0, *, key=0"),
        "f(1, 2, key=3)",
    ),
    "argument given twice": (_fn("a, b=0"), _fn("b, *, a=0"), "f(5, a=1)"),
    "positional-only given by keyword": (_fn("a, /"), _fn("a"), "f(a=1)"),
    "class without an initializer": (
        "class f:\n    pass\n",
        "class f:\n    def __init__(self, size):\n        self.size = size\n",
        "f(3)",
    ),
    "C callable given too many": ("f = len\n", _fn("*args"), "f([1], [2])"),
    "C callable given too few": ("f = len\n", _fn("*args"), "f()"),
    "C callable given a keyword": (
        "f = len\n",
        _fn("value, key=None"),
        "f([1], key=1)",
    ),
}

# TypeErrors raised in the test's own body that no call made: the baseline returned None where the
# head returns something the test then uses.
NOT_WRONG_CALLS = {
    "None subscripted": (
        _fn("", "None"),
        _fn("", "{'color': 'red'}"),
        "f()['color'] == 'red'",
    ),
    "None called": (_fn("", "None"), _fn("", "lambda: 1"), "f()() == 1"),
}

CALL_PHASE_SENTENCE = (
    "the test's own code raised a TypeError before its assertion"
)
CAUSE_CLAUSE = "A call that does not fit the baseline's code is one cause"
FIXTURE_SENTENCE = "a fixture failed during setup"
TEARDOWN_SENTENCE = "a fixture failed during teardown"
UNREACHED_SENTENCE = "a missing import or file"


def _one_call_repo(
    tmp_path: Path, baseline_source: str, head_source: str, expression: str
) -> tuple[Path, str]:
    """Build a scratch repository whose one test asserts ``expression`` against ``widgets``.

    Returns the repository at its head and the baseline revision.
    """
    repo = _scratch_repo(tmp_path)
    _write(repo / "widgets.py", baseline_source)
    baseline = _commit_all(repo, "the baseline's widgets")
    _write(repo / "widgets.py", head_source)
    _write(
        repo / "tests" / "test_widgets.py",
        "def test_the_call_fits_the_head():\n"
        "    from widgets import f\n"
        f"    assert {expression}\n",
    )
    _commit_all(repo, "the head's widgets")
    return repo, baseline


def _passes_at_head(repo: Path, test_file: str) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            test_file,
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.parametrize("kind", sorted(WRONG_CALLS))
def test_a_wrong_call_in_the_test_body_is_refused_whatever_its_wording(
    tmp_path: Path, kind: str
) -> None:
    """Each kind of wrong call the interpreter words differently is refused.

    The test's call fits the head's signature and not the baseline's, so the baseline raises the
    interpreter's own TypeError in the test body. The refusal says what was observed, names a
    misfit call as one possible cause, and neither asks for a rerun nor says the code was never
    reached.
    """
    baseline_source, head_source, call = WRONG_CALLS[kind]
    repo, baseline = _one_call_repo(
        tmp_path, baseline_source, head_source, f"{call} is not None"
    )
    _passes_at_head(repo, "tests/test_widgets.py")

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    assert "[fixture]" in result.stdout
    why = _why(result)
    assert CALL_PHASE_SENTENCE in why
    assert CAUSE_CLAUSE in why
    assert "no behavior was compared" in why
    assert "not evidence either way" in why
    assert "redo" not in why.lower()
    assert "never reached" not in why.lower()


@pytest.mark.parametrize("kind", sorted(NOT_WRONG_CALLS))
def test_a_type_error_that_no_call_made_is_refused_the_same_way(
    tmp_path: Path, kind: str
) -> None:
    """A TypeError from using what the baseline returned is refused with the call-phase sentence.

    The class and the place of the raise are all the tool reads.
    """
    baseline_source, head_source, expression = NOT_WRONG_CALLS[kind]
    repo, baseline = _one_call_repo(
        tmp_path, baseline_source, head_source, expression
    )
    _passes_at_head(repo, "tests/test_widgets.py")

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    assert CALL_PHASE_SENTENCE in result.stdout
    assert "[fixture]" in result.stdout


def test_a_valid_call_whose_package_code_raises_type_error_guards(
    tmp_path: Path,
) -> None:
    """A TypeError the code under test raises on a right call is still counted.

    The crash frame sits in the package's file, outside tests/, so the code under test was reached.
    """
    repo, baseline = _one_call_repo(
        tmp_path,
        "def f(x):\n    return x + None\n",
        "def f(x):\n    return x * 2\n",
        "f(3) == 6",
    )
    _passes_at_head(repo, "tests/test_widgets.py")

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["GUARDS"], result.stdout + result.stderr
    assert "[behavioral]" in result.stdout
    assert "TypeError" in result.stdout


def test_a_type_error_raised_inside_a_library_by_package_code_guards(
    tmp_path: Path,
) -> None:
    """A TypeError raised in a library frame under a right call is counted, not refused.

    The crash frame is the standard library's, which cannot be placed under the tests/ of the
    baseline's tree, so the run is not said to have the test's own code raising.
    """
    repo, baseline = _one_call_repo(
        tmp_path,
        "import json\n\n\ndef f():\n    return json.dumps({1})\n",
        "import json\n\n\ndef f():\n    return json.dumps([1])\n",
        'f() == "[1]"',
    )
    _passes_at_head(repo, "tests/test_widgets.py")

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["GUARDS"], result.stdout + result.stderr
    assert "[behavioral]" in result.stdout
    assert "the test's own code raised" not in result.stdout


def _why(result: subprocess.CompletedProcess) -> str:
    return next(
        line
        for line in result.stdout.splitlines()
        if line.startswith(("GUARDS:", "REFUSED:"))
    )


def test_a_run_mixing_an_assertion_failure_and_a_wrong_call_names_each_by_role(
    tmp_path: Path,
) -> None:
    """The verdict names the assertion failure it rests on and the wrong call as not counted."""
    repo = _scratch_repo(tmp_path)
    _write(
        repo / "widgets.py",
        "def double(x):\n"
        "    return x\n"
        "\n"
        "\n"
        "class Widget:\n"
        "    def __init__(self, size):\n"
        "        self.size = size\n",
    )
    baseline = _commit_all(repo, "double is a no-op, widget has no color")
    _write(
        repo / "widgets.py",
        "def double(x):\n"
        "    return x * 2\n"
        "\n"
        "\n"
        "class Widget:\n"
        "    def __init__(self, size, color=None):\n"
        "        self.size = size\n"
        "        self.color = color\n",
    )
    _write(
        repo / "tests" / "test_widgets.py",
        "def test_double_doubles():\n"
        "    from widgets import double\n"
        "    assert double(3) == 6\n"
        "\n"
        "\n"
        "def test_widget_carries_a_color():\n"
        "    from widgets import Widget\n"
        "    assert Widget(size=3, color='red').color == 'red'\n",
    )
    _commit_all(repo, "double doubles, widget carries a color")

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["GUARDS"], result.stdout + result.stderr
    counted, _, not_counted = _why(result).partition("not counted")
    assert "test_double_doubles" in counted
    assert "test_widget_carries_a_color" not in counted
    assert "test_widget_carries_a_color" in not_counted
    assert CALL_PHASE_SENTENCE in not_counted
    assert "test_double_doubles" not in not_counted


def test_a_run_mixing_a_setup_failure_and_a_wrong_call_names_each_with_its_own_sentence(
    tmp_path: Path,
) -> None:
    """A setup failure and a call-phase TypeError each get the sentence for their own phase."""
    repo = _scratch_repo(tmp_path)
    _write(
        repo / "widgets.py",
        "class Widget:\n"
        "    def __init__(self, size):\n"
        "        self.size = size\n",
    )
    baseline = _commit_all(repo, "widget has no color")
    _write(
        repo / "widgets.py",
        "class Widget:\n"
        "    def __init__(self, size, color=None):\n"
        "        self.size = size\n"
        "        self.color = color\n",
    )
    _write(
        repo / "tests" / "test_widgets.py",
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
        "    assert thing.color == 'red'\n"
        "\n"
        "\n"
        "def test_widget_carries_a_color():\n"
        "    from widgets import Widget\n"
        "    assert Widget(size=3, color='red').color == 'red'\n",
    )
    _commit_all(repo, "widget carries a color")

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    clauses = _why(result).split("; ")
    assert any(
        "test_uses_thing" in c and FIXTURE_SENTENCE in c for c in clauses
    )
    assert any(
        "test_widget_carries_a_color" in c and CALL_PHASE_SENTENCE in c
        for c in clauses
    )


def _head_adds_color_double_and_gadgets(
    tmp_path: Path, test_source: str
) -> tuple[Path, str]:
    """Build a scratch repository whose head holds ``test_source`` and what it needs.

    The baseline's ``widgets`` has a no-op ``double`` and a ``Widget`` without a color, and has no
    ``gadgets`` module; the head fixes the first two and adds the third. Returns the repository at
    its head and the baseline revision.
    """
    repo = _scratch_repo(tmp_path)
    _write(
        repo / "widgets.py",
        "def double(x):\n"
        "    return x\n"
        "\n"
        "\n"
        "class Widget:\n"
        "    def __init__(self, size):\n"
        "        self.size = size\n",
    )
    baseline = _commit_all(repo, "double is a no-op, widget has no color")
    _write(
        repo / "widgets.py",
        "def double(x):\n"
        "    return x * 2\n"
        "\n"
        "\n"
        "class Widget:\n"
        "    def __init__(self, size, color=None):\n"
        "        self.size = size\n"
        "        self.color = color\n",
    )
    _write(repo / "gadgets.py", "GADGET = 1\n")
    _write(repo / "tests" / "test_widgets.py", test_source)
    _commit_all(repo, "double doubles, widget carries a color, gadgets exists")
    return repo, baseline


def _clauses_naming(clauses: list[str], test: str, sentence: str) -> list[str]:
    return [c for c in clauses if test in c and sentence in c]


def test_a_fixture_refusal_lists_a_missing_import_failure_as_not_counted(
    tmp_path: Path,
) -> None:
    """A fixture refusal lists the failure resting on a missing import after "Also not counted"."""
    repo, baseline = _head_adds_color_double_and_gadgets(
        tmp_path,
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
        "    assert thing.color == 'red'\n"
        "\n"
        "\n"
        "def test_imports_a_gadget():\n"
        "    from gadgets import GADGET\n"
        "    assert GADGET == 1\n",
    )

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    why = _why(result)
    assert why.startswith("REFUSED:")
    _, marker, listing = why.partition(" Also not counted: ")
    assert marker, why
    assert "test_uses_thing" not in listing
    assert (
        len(
            _clauses_naming(
                listing.split("; "),
                "test_imports_a_gadget",
                UNREACHED_SENTENCE,
            )
        )
        == 1
    )


def test_a_guards_run_lists_a_missing_import_failure_as_not_counted(
    tmp_path: Path,
) -> None:
    """A GUARDS run names a missing-import failure as not counted, never among the counted."""
    repo, baseline = _head_adds_color_double_and_gadgets(
        tmp_path,
        "def test_double_doubles():\n"
        "    from widgets import double\n"
        "    assert double(3) == 6\n"
        "\n"
        "\n"
        "def test_imports_a_gadget():\n"
        "    from gadgets import GADGET\n"
        "    assert GADGET == 1\n",
    )

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["GUARDS"], result.stdout + result.stderr
    counted, marker, not_counted = _why(result).partition("not counted")
    assert marker
    assert "test_double_doubles" in counted
    assert "test_imports_a_gadget" not in counted
    assert "test_double_doubles" not in not_counted
    assert (
        len(
            _clauses_naming(
                not_counted.split("; "),
                "test_imports_a_gadget",
                UNREACHED_SENTENCE,
            )
        )
        == 1
    )


def test_a_fixture_failing_during_teardown_is_refused_with_the_teardown_sentence(
    tmp_path: Path,
) -> None:
    """A fixture raising after its yield is named as failing during teardown, without a cause."""
    repo, baseline = _head_adds_color_double_and_gadgets(
        tmp_path,
        "import pytest\n"
        "\n"
        "\n"
        "@pytest.fixture\n"
        "def thing():\n"
        "    yield 1\n"
        "    from widgets import Widget\n"
        "    Widget(size=3, color='red')\n"
        "\n"
        "\n"
        "def test_uses_thing(thing):\n"
        "    assert thing == 1\n",
    )

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    why = _why(result)
    assert (
        len(
            _clauses_naming(
                why.split("; "), "test_uses_thing", TEARDOWN_SENTENCE
            )
        )
        == 1
    )
    assert FIXTURE_SENTENCE not in why
    assert CAUSE_CLAUSE not in why


# ── --baseline-from-working-tree ────────────────────────────────────────────


def _snapshot(repo: Path, test_file: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(repo / "tools" / "prove_test_fails_before.py"),
            test_file,
            "--baseline-from-working-tree",
        ],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_baseline_from_working_tree_captures_tracked_and_untracked_changes_untouched(
    tmp_path: Path,
) -> None:
    """A working-tree baseline captures tracked and untracked changes and leaves them untouched.

    A tracked modification and an untracked file both land in the printed snapshot's tree, and the
    working tree, the index, and the stash list are all unchanged afterward.
    """
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


def test_baseline_from_working_tree_excludes_ignored_files(
    tmp_path: Path,
) -> None:
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


def test_baseline_from_working_tree_together_with_baseline_is_refused(
    tmp_path: Path,
) -> None:
    repo = _scratch_repo(tmp_path)
    _write(repo / "widgets.py", "def double(x):\n    return x\n")
    baseline = _commit_all(repo, "double is a no-op")

    proc = subprocess.run(
        [
            sys.executable,
            str(repo / "tools" / "prove_test_fails_before.py"),
            "tests/test_widgets.py",
            "--baseline-from-working-tree",
            "--baseline",
            baseline,
        ],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode != 0


def test_the_two_step_flow_snapshot_then_fix_then_baseline_reports_guards(
    tmp_path: Path,
) -> None:
    """Snapshot, then fix, then a run against the snapshot as baseline reports GUARDS.

    The snapshot is taken before the fix, the fix applied, then the tool runs normally with
    --baseline <the printed hash>.
    """
    repo = _scratch_repo(tmp_path)
    _write(repo / "widgets.py", "def double(x):\n    return x\n")
    _commit_all(repo, "double is a no-op")

    _write(
        repo / "tests" / "test_widgets.py",
        "def test_double_doubles():\n"
        "    from widgets import double\n"
        "    assert double(3) == 6\n",
    )

    proc = _snapshot(repo, "tests/test_widgets.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    snapshot = re.search(r"snapshot ([0-9a-f]{40})", proc.stdout).group(1)

    _write(repo / "widgets.py", "def double(x):\n    return x * 2\n")

    result = _run(repo, "tests/test_widgets.py", snapshot)
    assert result.returncode == EXIT["GUARDS"], result.stdout + result.stderr
    assert "GUARDS" in result.stdout


def test_a_bare_file_not_found_error_outside_the_assertion_is_refused_not_guards(
    tmp_path: Path,
) -> None:
    """A bare FileNotFoundError outside the assertion scores REFUSED as unreached, not GUARDS.

    A helper module outside tests/ opens a data file the baseline commit does not carry, and the
    crash frame sits in that helper, outside tests/; the tool reports REFUSED, unreached.
    """
    repo = _scratch_repo(tmp_path)
    _write(repo / "widgets.py", "def double(x):\n    return x\n")
    _write(
        repo / "loader.py",
        "def load_expected():\n"
        "    with open('data/expected.txt') as f:\n"
        "        return int(f.read())\n",
    )
    baseline = _commit_all(repo, "double is a no-op, no recorded value yet")

    (repo / "data").mkdir()
    _write(repo / "data" / "expected.txt", "6\n")
    _write(repo / "widgets.py", "def double(x):\n    return x * 2\n")
    _write(
        repo / "tests" / "test_widgets.py",
        "def test_double_matches_the_recorded_value():\n"
        "    from widgets import double\n"
        "    from loader import load_expected\n"
        "    expected = load_expected()\n"
        "    assert double(3) == expected\n",
    )
    _commit_all(
        repo, "double actually doubles, checked against a recorded value"
    )

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    assert "REFUSED" in result.stdout
    assert "[unreached]" in result.stdout


def test_a_file_not_found_error_at_the_tests_own_assertion_line_guards(
    tmp_path: Path,
) -> None:
    """A FileNotFoundError at the test's own assertion line scores GUARDS.

    The fix writes a file and the guard test's own assertion line reads it back, so at the
    baseline that read raises FileNotFoundError with its crash frame inside the guard test's body.
    """
    repo = _scratch_repo(tmp_path)
    _write(repo / "widgets.py", "def write_summary(path):\n    pass\n")
    baseline = _commit_all(repo, "write_summary is a no-op")

    _write(
        repo / "widgets.py",
        "def write_summary(path):\n"
        "    with open(path, 'w') as f:\n"
        "        f.write('6')\n",
    )
    _write(
        repo / "tests" / "test_widgets.py",
        "def test_write_summary_writes_the_expected_content(tmp_path):\n"
        "    from widgets import write_summary\n"
        "    out = tmp_path / 'summary.txt'\n"
        "    write_summary(out)\n"
        "    with open(out) as f:\n"
        "        assert f.read() == '6'\n",
    )
    _commit_all(
        repo, "write_summary actually writes, checked by reading it back"
    )

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["GUARDS"], result.stdout + result.stderr
    assert "GUARDS" in result.stdout


# ── --per-test-timeout ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "headline",
    [
        "Timeout (>5.0s) from pytest-timeout.",
        "Failed: Timeout (>1.0s) from pytest-timeout.",
    ],
)
def test_the_signal_method_timeout_message_is_a_timeout_not_a_behavioral_pass(
    headline: str,
) -> None:
    """The signal-method timeout message reads as a timeout, not a behavioral pass.

    Where SIGALRM exists, pytest-timeout raises pytest.fail(PYTEST_FAILURE_MESSAGE), a Failed
    exception; its headline, bare and prefixed with the exception class, classifies as a timeout.
    """
    entry = {"headline": headline, "phase": "call", "exc_typename": "Failed"}
    assert _failure_kind(entry, Path(".")) == "timeout"


def test_a_hung_test_under_per_test_timeout_reports_indeterminate_naming_it(
    tmp_path: Path,
) -> None:
    """A hung test under a per-test timeout scores INDETERMINATE and is named.

    The tool's output names INDETERMINATE and the hung test.
    """
    repo = _scratch_repo(tmp_path)
    _write(
        repo / "tests" / "test_widgets.py",
        "import time\n\n\ndef test_hangs_forever():\n    time.sleep(30)\n",
    )
    _commit_all(repo, "a test that hangs")

    proc = subprocess.run(
        [
            sys.executable,
            str(repo / "tools" / "prove_test_fails_before.py"),
            "tests/test_widgets.py",
            "--baseline",
            "HEAD",
            "--per-test-timeout",
            "1",
        ],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert "INDETERMINATE" in proc.stdout
    assert "test_hangs_forever" in proc.stdout


def test_a_key_error_on_a_package_result_guards(tmp_path: Path) -> None:
    """A KeyError on a package result at the assert line scores GUARDS.

    An assertion that inspects a package result by a key the baseline's result lacks raises
    KeyError at the assert line inside the test file, and the tool reports GUARDS, behavioral,
    naming KeyError.
    """
    repo = _scratch_repo(tmp_path)
    _write(repo / "widgets.py", "def describe():\n    return {'size': 3}\n")
    baseline = _commit_all(repo, "describe carries no color")

    _write(
        repo / "widgets.py",
        "def describe():\n    return {'size': 3, 'color': 'red'}\n",
    )
    _write(
        repo / "tests" / "test_widgets.py",
        "def test_describe_carries_a_color():\n"
        "    from widgets import describe\n"
        "    assert describe()['color'] == 'red'\n",
    )
    _commit_all(repo, "describe carries a color")

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["GUARDS"], result.stdout + result.stderr
    assert "GUARDS" in result.stdout
    assert "[behavioral]" in result.stdout
    assert "KeyError" in result.stdout
