"""Tests for tools/prove_test_fails_before.py: its verdicts, options and scratch directory.

Most cases build a two-revision scratch repository (a baseline commit the fault sits in, a second
commit with the fix and the guard test) and run the real script against it. They cover:

- the failure classification, a baseline failure that does not reach the code under test told
  apart from one at the guard's own assertion;
- ``--test-rev``, over a third revision that removes the test;
- ``--baseline-from-working-tree``, the snapshot it prints and leaves the working tree untouched,
  and its refusal beside ``--baseline``, with no commit to snapshot, or when neither form makes one;
- a command line naming no test file and a ``--test-rev`` naming no revision, each refused, and an
  exception the tool does not name, which exits with the tool's own failure code;
- the test file's place and the tree it comes from, each way the baseline is chosen or refused,
  the import proof over a tree holding a package, and each cause of a pytest run's verdict;
- ``--per-test-timeout``, one case calling ``_failure_kind`` directly on a headline;
- the run's scratch directory, where the baseline's temporary root lies and the line printed when
  the directory cannot be removed;
- a vitest file under a scratch frontend whose installed packages are this checkout's, each verdict
  and refusal it can reach, and a changed frontend test counted as the test side of a pytest run.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests import FRONTEND, REPO_ROOT
from tests._scratch_repo_fixtures import commit_all, git, scratch_repo, write
from tools.prove_test_fails_before import OUTCOME_ENV, _failure_kind

EXIT = {
    "GUARDS": 0,
    "VACUOUS": 1,
    "INDETERMINATE": 2,
    "REFUSED": 3,
    "TOOL_FAILURE": 4,
}


def _tool(repo: Path, *args: str) -> subprocess.CompletedProcess:
    """Run the script copied into ``repo`` with ``args`` as its command line, from ``repo``."""
    return subprocess.run(
        [
            sys.executable,
            str(repo / "tools" / "prove_test_fails_before.py"),
            *args,
        ],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=120,
    )


def _run(
    repo: Path, test_file: str, baseline: str, *extra: str
) -> subprocess.CompletedProcess:
    return _tool(repo, test_file, "--baseline", baseline, *extra)


DOUBLE_GUARD = (
    "def test_double_doubles():\n"
    "    from widgets import double\n"
    "    assert double(3) == 6\n"
)


def _fixed_double(
    tmp_path: Path, test_source: str = DOUBLE_GUARD
) -> tuple[Path, str]:
    """Build a scratch repository whose baseline ``double`` is a no-op and whose head fixes it.

    The head commit also holds ``test_source`` as ``tests/test_widgets.py``. Returns the
    repository at its head and the baseline revision.
    """
    repo = scratch_repo(tmp_path)
    write(repo / "widgets.py", "def double(x):\n    return x\n")
    baseline = commit_all(repo, "double is a no-op")
    write(repo / "widgets.py", "def double(x):\n    return x * 2\n")
    write(repo / "tests" / "test_widgets.py", test_source)
    commit_all(repo, "double actually doubles")
    return repo, baseline


def _redirect_temporary_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Point the tool's temporary folder at a new folder under ``tmp_path`` and return it."""
    scratch_root = tmp_path / "t"
    scratch_root.mkdir()
    for name in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(name, str(scratch_root))
    return scratch_root


def test_a_setup_fixture_error_is_refused(tmp_path: Path) -> None:
    """A setup fixture error is refused.

    A fixture that fails during setup is reported REFUSED, kind ``fixture``.
    """
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
        "    assert thing.color == 'red'\n",
    )

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
    repo, baseline = _fixed_double(tmp_path)

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["GUARDS"], result.stdout + result.stderr
    assert "GUARDS" in result.stdout
    assert "[behavioral]" in result.stdout


@pytest.mark.parametrize("spelling", ["relative", "absolute"])
def test_a_test_file_taken_from_test_rev_guards_by_either_spelling(
    tmp_path: Path, spelling: str
) -> None:
    """A test file taken from ``--test-rev`` scores GUARDS named relative or absolute.

    The scratch repository's head no longer holds the test file, so the verdict can only come
    from the file at the fix commit; the repo-relative and the absolute spelling report it alike.
    """
    repo, baseline = _fixed_double(tmp_path)
    fix = git(repo, "rev-parse", "HEAD").strip()

    test_path = repo / "tests" / "test_widgets.py"
    test_path.unlink()
    commit_all(repo, "remove the test")

    named = (
        "tests/test_widgets.py" if spelling == "relative" else str(test_path)
    )
    result = _run(repo, named, baseline, "--test-rev", fix)

    assert result.returncode == EXIT["GUARDS"], result.stdout + result.stderr
    assert "GUARDS" in result.stdout


def test_a_value_error_raised_from_code_under_test_guards(
    tmp_path: Path,
) -> None:
    """A ValueError raised from the code under test scores GUARDS.

    An uncaught ValueError the code under test raises is reported GUARDS, behavioral, naming
    ValueError.
    """
    repo = scratch_repo(tmp_path)
    write(
        repo / "widgets.py",
        "def normalize(value):\n"
        "    if value <= 0:\n"
        "        raise ValueError('must be positive')\n"
        "    return value\n",
    )
    baseline = commit_all(repo, "normalize wrongly rejects zero")

    write(
        repo / "widgets.py",
        "def normalize(value):\n"
        "    if value < 0:\n"
        "        raise ValueError('must be positive')\n"
        "    return value\n",
    )
    write(
        repo / "tests" / "test_widgets.py",
        "def test_normalize_permits_zero():\n"
        "    from widgets import normalize\n"
        "    assert normalize(0) == 0\n",
    )
    commit_all(repo, "normalize permits zero")

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
    kind ``fixture``, though the crash frame sits in the package's own file, outside tests/.
    """
    repo = scratch_repo(tmp_path)
    write(
        repo / "widgets.py",
        "def make_widget():\n"
        "    raise ValueError('widgets are not ready yet')\n",
    )
    baseline = commit_all(repo, "widgets are not buildable")

    write(
        repo / "widgets.py",
        "class Widget:\n"
        "    pass\n"
        "\n"
        "\n"
        "def make_widget():\n"
        "    return Widget()\n",
    )
    write(
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
    commit_all(repo, "widgets are buildable")

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
    repo = scratch_repo(tmp_path)
    write(repo / "widgets.py", baseline_source)
    baseline = commit_all(repo, "the baseline's widgets")
    write(repo / "widgets.py", head_source)
    write(
        repo / "tests" / "test_widgets.py",
        "def test_the_call_fits_the_head():\n"
        "    from widgets import f\n"
        f"    assert {expression}\n",
    )
    commit_all(repo, "the head's widgets")
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
    assert "[test_code]" in result.stdout
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
    assert "[test_code]" in result.stdout


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
    materialized tree, so the run is not said to have the test's own code raising.
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
    """Return the verdict line ``result`` printed, or the empty string when it printed none."""
    return next(
        (
            line
            for line in result.stdout.splitlines()
            if line.startswith(tuple(f"{verdict}:" for verdict in EXIT))
        ),
        "",
    )


def _head_adds_color_double_and_gadgets(
    tmp_path: Path, test_source: str
) -> tuple[Path, str]:
    """Build a scratch repository whose head holds ``test_source`` and what it needs.

    The baseline's ``widgets`` has a no-op ``double`` and a ``Widget`` without a color, and has no
    ``gadgets`` module; the head fixes the first two and adds the third. Returns the repository at
    its head and the baseline revision.
    """
    repo = scratch_repo(tmp_path)
    write(
        repo / "widgets.py",
        "def double(x):\n"
        "    return x\n"
        "\n"
        "\n"
        "class Widget:\n"
        "    def __init__(self, size):\n"
        "        self.size = size\n",
    )
    baseline = commit_all(repo, "double is a no-op, widget has no color")
    write(
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
    write(repo / "gadgets.py", "GADGET = 1\n")
    write(repo / "tests" / "test_widgets.py", test_source)
    commit_all(repo, "double doubles, widget carries a color, gadgets exists")
    return repo, baseline


def _assert_one_clause_names(text: str, test: str, sentence: str) -> None:
    """Assert exactly one ``"; "`` clause of ``text`` holds both ``test`` and ``sentence``."""
    clauses = [c for c in text.split("; ") if test in c and sentence in c]
    assert len(clauses) == 1, text


def test_a_run_mixing_an_assertion_failure_and_a_wrong_call_names_each_by_role(
    tmp_path: Path,
) -> None:
    """The verdict names the assertion failure it rests on and the wrong call as not counted."""
    repo, baseline = _head_adds_color_double_and_gadgets(
        tmp_path,
        "def test_double_doubles():\n"
        "    from widgets import double\n"
        "    assert double(3) == 6\n"
        "\n"
        "\n"
        "def test_widget_carries_a_color():\n"
        "    from widgets import Widget\n"
        "    assert Widget(size=3, color='red').color == 'red'\n",
    )

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["GUARDS"], result.stdout + result.stderr
    counted, _, not_counted = _why(result).partition("not counted")
    assert "test_double_doubles" in counted
    assert "test_widget_carries_a_color" not in counted
    _assert_one_clause_names(
        not_counted, "test_widget_carries_a_color", CALL_PHASE_SENTENCE
    )
    assert "test_double_doubles" not in not_counted


def test_a_run_mixing_a_setup_failure_and_a_wrong_call_names_each_with_its_own_sentence(
    tmp_path: Path,
) -> None:
    """A setup failure and a call-phase TypeError each get the sentence for their own phase."""
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
        "def test_widget_carries_a_color():\n"
        "    from widgets import Widget\n"
        "    assert Widget(size=3, color='red').color == 'red'\n",
    )

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    why = _why(result)
    _assert_one_clause_names(why, "test_uses_thing", FIXTURE_SENTENCE)
    _assert_one_clause_names(
        why, "test_widget_carries_a_color", CALL_PHASE_SENTENCE
    )


def test_the_json_record_carries_each_failures_kind(tmp_path: Path) -> None:
    """The ``--json`` record names a setup failure, a call-phase TypeError and an assertion failure.

    Their kinds are ``fixture``, ``test_code`` and ``behavioral``, and each is the tag printed for
    its test. The GUARDS sentence lists the setup failure among the not-counted.
    """
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
        "def test_widget_carries_a_color():\n"
        "    from widgets import Widget\n"
        "    assert Widget(size=3, color='red').color == 'red'\n"
        "\n"
        "\n"
        "def test_double_doubles():\n"
        "    from widgets import double\n"
        "    assert double(3) == 6\n",
    )
    record_path = tmp_path / "record.json"

    result = _run(
        repo,
        "tests/test_widgets.py",
        baseline,
        "--json",
        str(record_path),
    )

    assert result.returncode == EXIT["GUARDS"], result.stdout + result.stderr
    record = json.loads(record_path.read_text(encoding="utf-8"))
    recorded = {
        f["nodeid"].rpartition("::")[2]: f["kind"]
        for f in record["observed"]["failures"]
    }
    printed = dict(
        re.findall(r"failing\s+\S+::(\w+)\s+\[(\w+)\]", result.stdout)
    )
    expected = {
        "test_uses_thing": "fixture",
        "test_widget_carries_a_color": "test_code",
        "test_double_doubles": "behavioral",
    }
    assert recorded == expected
    assert printed == expected
    _, _, not_counted = _why(result).partition("not counted")
    _assert_one_clause_names(not_counted, "test_uses_thing", FIXTURE_SENTENCE)


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
    before, marker, listing = why.partition(" Also not counted: ")
    assert marker, why
    assert "test_imports_a_gadget" not in before
    assert "test_uses_thing" not in listing
    _assert_one_clause_names(
        listing, "test_imports_a_gadget", UNREACHED_SENTENCE
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
    _assert_one_clause_names(
        not_counted, "test_imports_a_gadget", UNREACHED_SENTENCE
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
    _assert_one_clause_names(why, "test_uses_thing", TEARDOWN_SENTENCE)
    assert FIXTURE_SENTENCE not in why
    assert CAUSE_CLAUSE not in why


# ── --baseline-from-working-tree ────────────────────────────────────────────


def _snapshot(repo: Path, test_file: str) -> subprocess.CompletedProcess:
    return _tool(repo, test_file, "--baseline-from-working-tree")


def test_baseline_from_working_tree_captures_tracked_and_untracked_changes_untouched(
    tmp_path: Path,
) -> None:
    """A working-tree baseline captures tracked and untracked changes and leaves them untouched.

    A tracked modification and an untracked file both land in the printed snapshot's tree, and the
    working tree, the index, and the stash list are all unchanged afterward.
    """
    repo = scratch_repo(tmp_path)
    write(repo / "widgets.py", "def double(x):\n    return x\n")
    commit_all(repo, "double is a no-op")

    write(repo / "widgets.py", "def double(x):\n    return x * 2\n")
    write(repo / "untracked.py", "NEW = True\n")
    status_before = git(repo, "status", "--porcelain")

    proc = _snapshot(repo, "tests/test_widgets.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    match = re.search(r"snapshot ([0-9a-f]{40})", proc.stdout)
    assert match, proc.stdout
    snapshot = match.group(1)

    assert "x * 2" in git(repo, "show", f"{snapshot}:widgets.py")
    assert "NEW = True" in git(repo, "show", f"{snapshot}:untracked.py")
    assert git(repo, "status", "--porcelain") == status_before
    assert git(repo, "stash", "list").strip() == ""


def test_baseline_from_working_tree_excludes_ignored_files(
    tmp_path: Path,
) -> None:
    repo = scratch_repo(tmp_path)
    write(repo / ".gitignore", "ignored.txt\n")
    write(repo / "widgets.py", "def double(x):\n    return x\n")
    commit_all(repo, "double is a no-op")
    write(repo / "ignored.txt", "should not appear\n")

    proc = _snapshot(repo, "tests/test_widgets.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    snapshot = re.search(r"snapshot ([0-9a-f]{40})", proc.stdout).group(1)

    listing = git(repo, "ls-tree", "-r", "--name-only", snapshot)
    assert "ignored.txt" not in listing


def test_baseline_from_working_tree_together_with_baseline_is_refused(
    tmp_path: Path,
) -> None:
    repo = scratch_repo(tmp_path)
    write(repo / "widgets.py", "def double(x):\n    return x\n")
    baseline = commit_all(repo, "double is a no-op")

    proc = _tool(
        repo,
        "tests/test_widgets.py",
        "--baseline-from-working-tree",
        "--baseline",
        baseline,
    )

    assert proc.returncode == EXIT["REFUSED"], proc.stdout + proc.stderr
    assert _why(proc).startswith("REFUSED: "), proc.stdout + proc.stderr


def test_a_snapshot_in_a_repository_with_no_commit_is_refused(
    tmp_path: Path,
) -> None:
    """A snapshot in a repository with no commit is refused on git's failure to read HEAD.

    The refusal is git's own, before either form of the snapshot is tried.
    """
    repo = scratch_repo(tmp_path)

    proc = _snapshot(repo, "tests/test_widgets.py")

    why = _why(proc)
    assert why.startswith("REFUSED: "), proc.stdout + proc.stderr
    assert "git rev-parse HEAD failed" in why
    assert "could not snapshot" not in why
    assert proc.returncode == EXIT["REFUSED"], proc.stdout + proc.stderr


def test_a_snapshot_neither_form_can_make_is_refused(tmp_path: Path) -> None:
    """A snapshot neither form makes is refused with the sentence naming both forms.

    The repository has a commit and a clean tracked tree beside an untracked folder that is a
    repository with no commit of its own, which ``git add -A`` refuses to stage, and ``git stash
    create`` makes nothing of a clean tracked tree.
    """
    repo = scratch_repo(tmp_path)
    write(repo / "widgets.py", "def double(x):\n    return x\n")
    commit_all(repo, "double is a no-op")
    inner = repo / "inner"
    inner.mkdir()
    git(inner, "init", "-q")

    proc = _snapshot(repo, "tests/test_widgets.py")

    assert proc.returncode == EXIT["REFUSED"], proc.stdout + proc.stderr
    assert "could not snapshot the working tree" in _why(proc)


def test_the_two_step_flow_snapshot_then_fix_then_baseline_reports_guards(
    tmp_path: Path,
) -> None:
    """Snapshot, then fix, then a run against the snapshot as baseline reports GUARDS.

    The snapshot is taken before the fix, the fix applied, then the tool runs normally with
    --baseline <the printed hash>.
    """
    repo = scratch_repo(tmp_path)
    write(repo / "widgets.py", "def double(x):\n    return x\n")
    commit_all(repo, "double is a no-op")

    write(repo / "tests" / "test_widgets.py", DOUBLE_GUARD)

    proc = _snapshot(repo, "tests/test_widgets.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    snapshot = re.search(r"snapshot ([0-9a-f]{40})", proc.stdout).group(1)

    write(repo / "widgets.py", "def double(x):\n    return x * 2\n")

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
    repo = scratch_repo(tmp_path)
    write(repo / "widgets.py", "def double(x):\n    return x\n")
    write(
        repo / "loader.py",
        "def load_expected():\n"
        "    with open('data/expected.txt') as f:\n"
        "        return int(f.read())\n",
    )
    baseline = commit_all(repo, "double is a no-op, no recorded value yet")

    (repo / "data").mkdir()
    write(repo / "data" / "expected.txt", "6\n")
    write(repo / "widgets.py", "def double(x):\n    return x * 2\n")
    write(
        repo / "tests" / "test_widgets.py",
        "def test_double_matches_the_recorded_value():\n"
        "    from widgets import double\n"
        "    from loader import load_expected\n"
        "    expected = load_expected()\n"
        "    assert double(3) == expected\n",
    )
    commit_all(
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
    repo = scratch_repo(tmp_path)
    write(repo / "widgets.py", "def write_summary(path):\n    pass\n")
    baseline = commit_all(repo, "write_summary is a no-op")

    write(
        repo / "widgets.py",
        "def write_summary(path):\n"
        "    with open(path, 'w') as f:\n"
        "        f.write('6')\n",
    )
    write(
        repo / "tests" / "test_widgets.py",
        "def test_write_summary_writes_the_expected_content(tmp_path):\n"
        "    from widgets import write_summary\n"
        "    out = tmp_path / 'summary.txt'\n"
        "    write_summary(out)\n"
        "    with open(out) as f:\n"
        "        assert f.read() == '6'\n",
    )
    commit_all(
        repo, "write_summary actually writes, checked by reading it back"
    )

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["GUARDS"], result.stdout + result.stderr
    assert "GUARDS" in result.stdout


# ── a command line that names no run, and the tool's own failure ─────────────


def test_a_command_line_with_no_test_file_is_refused(tmp_path: Path) -> None:
    repo = scratch_repo(tmp_path)

    result = _tool(repo)

    assert _why(result).startswith("REFUSED: "), result.stdout + result.stderr
    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr


def test_a_test_rev_that_does_not_resolve_is_refused_naming_it(
    tmp_path: Path,
) -> None:
    """A ``--test-rev`` naming no revision is refused by flag and name, with no traceback."""
    repo = scratch_repo(tmp_path)
    write(repo / "widgets.py", "def double(x):\n    return x\n")
    baseline = commit_all(repo, "double is a no-op")

    result = _run(
        repo,
        "tests/test_widgets.py",
        baseline,
        "--test-rev",
        "no_such_revision",
    )

    why = _why(result)
    assert why.startswith("REFUSED: "), result.stdout + result.stderr
    assert "--test-rev" in why
    assert "no_such_revision" in why
    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    assert "Traceback" not in result.stderr


def test_an_exception_the_tool_does_not_name_exits_with_its_own_failure_code(
    tmp_path: Path,
) -> None:
    """An exception the tool does not name exits with the tool's own code, traceback shown.

    ``--json`` names a directory, so writing the record of a run that reached its verdict raises,
    and no verdict line is printed.
    """
    repo, baseline = _fixed_double(tmp_path)
    record = tmp_path / "record"
    record.mkdir()

    result = _run(
        repo, "tests/test_widgets.py", baseline, "--json", str(record)
    )

    assert result.returncode == EXIT["TOOL_FAILURE"], (
        result.stdout + result.stderr
    )
    assert "Traceback" in result.stderr
    assert _why(result) == "", result.stdout


def test_a_refused_run_whose_record_cannot_be_written_exits_with_the_tools_own_code(
    tmp_path: Path,
) -> None:
    """A refused run given ``--json`` a directory exits with the tool's own code, no verdict line.

    The baseline names no commit, so the run is refused, and writing its record then raises.
    """
    repo, _ = _fixed_double(tmp_path)
    record = tmp_path / "record"
    record.mkdir()

    result = _run(
        repo,
        "tests/test_widgets.py",
        "no_such_revision",
        "--json",
        str(record),
    )

    assert result.returncode == EXIT["TOOL_FAILURE"], (
        result.stdout + result.stderr
    )
    assert "Traceback" in result.stderr
    assert _why(result) == "", result.stdout


# ── the test file and the tree it comes from ─────────────────────────────────


def _baseline_line(result: subprocess.CompletedProcess) -> str:
    """Return the ``baseline`` line ``result`` printed, stripped, or the empty string."""
    return next(
        (
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip().startswith("baseline ")
        ),
        "",
    )


def test_a_test_file_outside_the_repository_is_refused(tmp_path: Path) -> None:
    repo = scratch_repo(tmp_path)
    outside = tmp_path / "elsewhere" / "test_widgets.py"
    outside.parent.mkdir()
    write(outside, "def test_passes():\n    assert True\n")

    result = _tool(repo, str(outside))

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    assert "is outside this repository" in _why(result)


@pytest.mark.parametrize(
    "spelling", ["relative", "relative through ..", "absolute through .."]
)
def test_a_file_off_the_test_side_is_refused_naming_both_test_trees(
    tmp_path: Path, spelling: str
) -> None:
    """A file in the repository and off the test side is refused however it is spelled.

    A spelling that reads as inside ``tests/`` before its ``..`` collapses is refused the same
    way, and the sentence names the path it resolves to, ``tests/`` and the frontend.
    """
    repo = scratch_repo(tmp_path)
    named = {
        "relative": "tools/prove_test_fails_before.py",
        "relative through ..": "tests/../tools/prove_test_fails_before.py",
        "absolute through ..": str(
            repo / "tests" / ".." / "tools" / "prove_test_fails_before.py"
        ),
    }[spelling]

    result = _tool(repo, named)

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    why = _why(result)
    assert "resolves to tools/prove_test_fails_before.py" in why
    assert "outside tests/ and no test file under" in why
    assert FRONTEND_PATH in why


def test_test_rev_without_a_baseline_is_refused(tmp_path: Path) -> None:
    repo, _ = _fixed_double(tmp_path)

    result = _tool(repo, "tests/test_widgets.py", "--test-rev", "HEAD")

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    assert "--test-rev needs an explicit --baseline" in _why(result)


def test_a_test_file_the_test_rev_does_not_hold_is_refused(
    tmp_path: Path,
) -> None:
    """A test file present in the working tree and absent at ``--test-rev`` is refused."""
    repo, baseline = _fixed_double(tmp_path)

    result = _run(
        repo, "tests/test_widgets.py", baseline, "--test-rev", baseline
    )

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    assert f"tests/test_widgets.py does not exist at {baseline}" in _why(
        result
    )


def test_a_test_file_absent_from_the_working_tree_is_refused(
    tmp_path: Path,
) -> None:
    repo, baseline = _fixed_double(tmp_path)

    result = _run(repo, "tests/test_absent.py", baseline)

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    assert "tests/test_absent.py does not exist in the working tree" in _why(
        result
    )


# ── the baseline ─────────────────────────────────────────────────────────────


def test_a_baseline_naming_no_commit_is_refused_and_recorded_without_one(
    tmp_path: Path,
) -> None:
    repo, _ = _fixed_double(tmp_path)
    record_path = tmp_path / "record.json"

    result = _run(
        repo,
        "tests/test_widgets.py",
        "no_such_revision",
        "--json",
        str(record_path),
    )

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    assert "--baseline no_such_revision does not name a commit here" in _why(
        result
    )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["verdict"] == "REFUSED"
    assert "baseline" not in record


def test_an_uncommitted_source_change_takes_head_as_the_baseline(
    tmp_path: Path,
) -> None:
    """With no ``--baseline``, a source change not yet committed makes HEAD the baseline."""
    repo = scratch_repo(tmp_path)
    write(repo / "widgets.py", "def double(x):\n    return x\n")
    head = commit_all(repo, "double is a no-op")
    write(repo / "widgets.py", "def double(x):\n    return x * 2\n")
    write(repo / "tests" / "test_widgets.py", DOUBLE_GUARD)

    result = _tool(repo, "tests/test_widgets.py")

    assert result.returncode == EXIT["GUARDS"], result.stdout + result.stderr
    assert _baseline_line(result) == (
        f"baseline      {head[:12]} (HEAD, because the change under test is uncommitted)"
    )


def test_a_committed_change_takes_its_merge_base_with_the_integration_branch(
    tmp_path: Path,
) -> None:
    """With no ``--baseline`` and a clean tree, the merge-base with ``--integration`` is used.

    The fix is committed on a branch cut from the branch the test names, so that branch's head
    is the baseline.
    """
    repo = scratch_repo(tmp_path)
    write(repo / "widgets.py", "def double(x):\n    return x\n")
    baseline = commit_all(repo, "double is a no-op")
    git(repo, "branch", "trunk")
    git(repo, "checkout", "-q", "-b", "fix")
    write(repo / "widgets.py", "def double(x):\n    return x * 2\n")
    write(repo / "tests" / "test_widgets.py", DOUBLE_GUARD)
    commit_all(repo, "double actually doubles")

    result = _tool(repo, "tests/test_widgets.py", "--integration", "trunk")

    assert result.returncode == EXIT["GUARDS"], result.stdout + result.stderr
    assert _baseline_line(result) == (
        f"baseline      {baseline[:12]} (merge-base of HEAD and trunk)"
    )


def test_head_already_on_the_integration_branch_is_refused(
    tmp_path: Path,
) -> None:
    repo, _ = _fixed_double(tmp_path)
    git(repo, "branch", "trunk")
    record_path = tmp_path / "record.json"

    result = _tool(
        repo,
        "tests/test_widgets.py",
        "--integration",
        "trunk",
        "--json",
        str(record_path),
    )

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    why = _why(result)
    assert "HEAD is already contained in trunk" in why
    assert "Name the pre-change revision with --baseline" in why
    assert "baseline" not in json.loads(
        record_path.read_text(encoding="utf-8")
    )


def test_an_integration_branch_that_does_not_exist_is_refused(
    tmp_path: Path,
) -> None:
    repo, _ = _fixed_double(tmp_path)
    record_path = tmp_path / "record.json"

    result = _tool(
        repo,
        "tests/test_widgets.py",
        "--integration",
        "no_such_branch",
        "--json",
        str(record_path),
    )

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    assert "no merge-base of HEAD and --integration no_such_branch" in _why(
        result
    )
    assert "baseline" not in json.loads(
        record_path.read_text(encoding="utf-8")
    )


def test_a_baseline_whose_tree_cannot_be_archived_is_refused_with_gits_error(
    tmp_path: Path,
) -> None:
    """A baseline commit missing one of its file objects resolves and is refused at its archive.

    The object deleted is the baseline's ``widgets.py``, which differs at the head, so the diff
    that finds the change reads no content of it and the archive is the first read that needs it.
    """
    repo, baseline = _fixed_double(tmp_path)
    blob = git(repo, "rev-parse", f"{baseline}:widgets.py").strip()
    loose = repo / ".git" / "objects" / blob[:2] / blob[2:]
    loose.chmod(stat.S_IWRITE)
    loose.unlink()

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    why = _why(result)
    assert f"git archive {baseline} failed" in why
    assert blob in why


# ── the import proof ─────────────────────────────────────────────────────────

GIZMO_GUARD = (
    "def test_gizmo_doubles():\n"
    "    from gizmo import double\n"
    "    assert double(3) == 6\n"
)


def _gizmo_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, baseline_source: str
) -> tuple[Path, str, Path]:
    """Build a scratch repository holding the package ``packages/gizmo/src/gizmo``.

    Its baseline ``__init__.py`` is ``baseline_source`` and its head's doubles, beside
    ``GIZMO_GUARD``. The tool's temporary folder is redirected to a folder under ``tmp_path``.
    Returns the repository at its head, the baseline revision and that folder.
    """
    scratch_root = _redirect_temporary_folder(tmp_path, monkeypatch)
    repo = scratch_repo(tmp_path)
    package = repo / "packages" / "gizmo" / "src" / "gizmo"
    package.mkdir(parents=True)
    write(package / "__init__.py", baseline_source)
    baseline = commit_all(repo, "the baseline's gizmo")
    write(package / "__init__.py", "def double(x):\n    return x * 2\n")
    write(repo / "tests" / "test_gizmo.py", GIZMO_GUARD)
    commit_all(repo, "gizmo doubles")
    return repo, baseline, scratch_root


def test_a_package_in_the_tree_is_proved_imported_from_the_baseline_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A package under ``packages/*/src`` resolves inside the run's baseline tree, on the record."""
    repo, baseline, scratch_root = _gizmo_repo(
        tmp_path, monkeypatch, "def double(x):\n    return x\n"
    )
    record_path = tmp_path / "record.json"

    result = _run(
        repo, "tests/test_gizmo.py", baseline, "--json", str(record_path)
    )

    assert result.returncode == EXIT["GUARDS"], result.stdout + result.stderr
    proof = json.loads(record_path.read_text(encoding="utf-8"))[
        "harness_proof"
    ]
    assert proof["baseline_source_imported"] is True
    resolved = Path(proof["resolved"]["gizmo"]).relative_to(
        scratch_root.resolve()
    )
    assert resolved.parts[0].startswith("failbefore-")
    assert resolved.parts[1:] == (
        "tree",
        "packages",
        "gizmo",
        "src",
        "gizmo",
        "__init__.py",
    )


def test_a_tree_holding_no_package_records_no_resolved_import(
    tmp_path: Path,
) -> None:
    repo, baseline = _fixed_double(tmp_path)
    record_path = tmp_path / "record.json"

    result = _run(
        repo, "tests/test_widgets.py", baseline, "--json", str(record_path)
    )

    assert result.returncode == EXIT["GUARDS"], result.stdout + result.stderr
    proof = json.loads(record_path.read_text(encoding="utf-8"))[
        "harness_proof"
    ]
    assert proof == {"resolved": {}, "baseline_source_imported": True}


def test_a_baseline_package_that_raises_at_import_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, baseline, _ = _gizmo_repo(
        tmp_path, monkeypatch, "raise RuntimeError('gizmo is not ready')\n"
    )
    record_path = tmp_path / "record.json"

    result = _run(
        repo, "tests/test_gizmo.py", baseline, "--json", str(record_path)
    )

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    assert "the baseline tree's own source is not what got imported" in _why(
        result
    )
    proof = json.loads(record_path.read_text(encoding="utf-8"))[
        "harness_proof"
    ]
    assert proof["baseline_source_imported"] is False
    assert "IMPORT FAILED" in proof["resolved"]["gizmo"]
    assert "gizmo is not ready" in proof["resolved"]["gizmo"]


# ── the run ──────────────────────────────────────────────────────────────────


def test_a_run_past_the_runs_timeout_is_refused_naming_the_runner_and_seconds(
    tmp_path: Path,
) -> None:
    """A test that never returns ends the run at ``--timeout``, refused by runner and seconds."""
    repo, baseline = _fixed_double(
        tmp_path,
        "import time\n"
        "\n"
        "\n"
        "def test_never_returns():\n"
        "    while True:\n"
        "        time.sleep(1)\n",
    )

    result = _run(repo, "tests/test_widgets.py", baseline, "--timeout", "3")

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    assert "pytest did not finish within 3s" in _why(result)


def test_a_run_that_ends_before_its_session_finishes_is_refused(
    tmp_path: Path,
) -> None:
    """A test ending the interpreter at once leaves no outcome record, and the run is refused."""
    repo, baseline = _fixed_double(
        tmp_path,
        "import os\n\n\ndef test_ends_the_interpreter():\n    os._exit(0)\n",
    )

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    assert "pytest reported no outcome record" in _why(result)


def test_an_outcome_record_that_is_not_json_is_refused(tmp_path: Path) -> None:
    """A test writing text that is not JSON where the record goes, then ending, is refused."""
    repo, baseline = _fixed_double(
        tmp_path,
        "import os\n"
        "\n"
        "\n"
        "def test_writes_the_record_itself():\n"
        f"    with open(os.environ[{OUTCOME_ENV!r}], 'w') as record:\n"
        "        record.write('not json')\n"
        "    os._exit(0)\n",
    )

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    assert "is not JSON" in _why(result)


def test_an_internal_error_of_the_runner_is_refused(tmp_path: Path) -> None:
    """A ``conftest.py`` hook raising during collection is pytest's internal error, refused."""
    repo, baseline = _fixed_double(tmp_path)
    write(
        repo / "tests" / "conftest.py",
        "def pytest_collection_modifyitems(items):\n"
        "    raise RuntimeError('a hook that raises')\n",
    )
    commit_all(repo, "a hook that raises")

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    assert "the runner hit an internal or unhandled error" in _why(result)


def test_a_module_that_raises_at_import_at_the_baseline_is_refused(
    tmp_path: Path,
) -> None:
    """A test module raising a ``KeyError`` at import at the baseline is a collection error."""
    repo = scratch_repo(tmp_path)
    write(repo / "widgets.py", "SETTINGS = {}\n")
    baseline = commit_all(repo, "no settings")
    write(repo / "widgets.py", "SETTINGS = {'color': 'red'}\n")
    write(
        repo / "tests" / "test_widgets.py",
        "from widgets import SETTINGS\n"
        "\n"
        "COLOR = SETTINGS['color']\n"
        "\n"
        "\n"
        "def test_the_color_is_red():\n"
        "    assert COLOR == 'red'\n",
    )
    commit_all(repo, "a color setting")

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    assert "the baseline could not collect tests/test_widgets.py" in _why(
        result
    )


def test_a_k_expression_matching_no_test_is_refused_as_nothing_selected(
    tmp_path: Path,
) -> None:
    repo, baseline = _fixed_double(tmp_path)

    result = _run(
        repo, "tests/test_widgets.py", baseline, "-k", "no_such_test"
    )

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    assert "nothing was selected" in _why(result)


def test_a_module_skipped_at_import_is_refused_as_nothing_selected(
    tmp_path: Path,
) -> None:
    """A module ``pytest.importorskip`` skips at import collects no test, so nothing is selected.

    It is not refused as every test skipped: no test of it was collected to be skipped.
    """
    repo, baseline = _fixed_double(
        tmp_path,
        "import pytest\n"
        "\n"
        "pytest.importorskip('no_such_module_anywhere')\n"
        "\n"
        "\n"
        "def test_passes():\n"
        "    assert True\n",
    )

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    why = _why(result)
    assert "nothing was selected" in why
    assert "were skipped" not in why


SKIPPED_TEST = "\n\n@pytest.mark.skip(reason='not yet')\ndef test_{name}():\n    assert False\n"


def test_every_test_skipped_at_setup_is_refused(tmp_path: Path) -> None:
    repo, baseline = _fixed_double(
        tmp_path,
        "import pytest\n"
        + SKIPPED_TEST.format(name="first")
        + SKIPPED_TEST.format(name="second"),
    )

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    assert "all 2 selected tests were skipped" in _why(result)


def test_a_pass_beside_a_skip_at_a_baseline_before_the_change_is_vacuous(
    tmp_path: Path,
) -> None:
    repo, baseline = _fixed_double(
        tmp_path,
        "import pytest\n"
        "\n"
        "\n"
        "def test_passes():\n"
        "    assert True\n" + SKIPPED_TEST.format(name="skipped"),
    )

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["VACUOUS"], result.stdout + result.stderr
    assert "at a baseline shown to precede the change" in _why(result)


def test_a_missing_import_failing_every_test_is_refused_quoting_it(
    tmp_path: Path,
) -> None:
    """Every failure resting on a module the baseline lacks is refused, quoting the import."""
    repo, baseline = _head_adds_color_double_and_gadgets(
        tmp_path,
        "def test_imports_a_gadget():\n"
        "    from gadgets import GADGET\n"
        "    assert GADGET == 1\n",
    )

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    why = _why(result)
    assert "all 1 failing tests failed on a missing import or file" in why
    assert "No module named 'gadgets'" in why


def test_a_failure_at_a_baseline_no_source_change_follows_guards_with_the_caveat(
    tmp_path: Path,
) -> None:
    repo = scratch_repo(tmp_path)
    write(repo / "widgets.py", "def double(x):\n    return x\n")
    write(repo / "tests" / "test_widgets.py", DOUBLE_GUARD)
    commit_all(repo, "a test double does not pass")

    result = _run(repo, "tests/test_widgets.py", "HEAD")

    assert result.returncode == EXIT["GUARDS"], result.stdout + result.stderr
    assert "No source file differs between the baseline and the tree" in _why(
        result
    )


def test_a_k_expression_selects_the_one_test_it_names_on_the_pytest_path(
    tmp_path: Path,
) -> None:
    repo, baseline = _fixed_double(
        tmp_path,
        "def test_double_of_three():\n"
        "    from widgets import double\n"
        "    assert double(3) == 6\n"
        "\n"
        "\n"
        "def test_double_of_four():\n"
        "    from widgets import double\n"
        "    assert double(4) == 8\n",
    )

    result = _run(repo, "tests/test_widgets.py", baseline, "-k", "four")

    assert result.returncode == EXIT["GUARDS"], result.stdout + result.stderr
    why = _why(result)
    assert "1 of 1 selected tests failed on the behavior" in why
    assert "test_double_of_four" in why
    assert "test_double_of_three" not in result.stdout


def test_change_counts_a_declared_file_only_when_it_differs(
    tmp_path: Path,
) -> None:
    """Each ``--change`` path that differs is counted, and only those, on the record.

    A declared path identical at the baseline is named as not counted, and a changed path not
    declared is not counted either.
    """
    repo = scratch_repo(tmp_path)
    write(repo / "widgets.py", "def double(x):\n    return x\n")
    write(repo / "gadgets.py", "GADGET = 1\n")
    write(repo / "helpers.py", "HELPER = 1\n")
    baseline = commit_all(repo, "double is a no-op")
    write(repo / "widgets.py", "def double(x):\n    return x * 2\n")
    write(repo / "helpers.py", "HELPER = 2\n")
    write(repo / "tests" / "test_widgets.py", DOUBLE_GUARD)
    commit_all(repo, "double actually doubles, the helper changes")
    record_path = tmp_path / "record.json"

    result = _run(
        repo,
        "tests/test_widgets.py",
        baseline,
        "--change",
        "widgets.py",
        "--change",
        "gadgets.py",
        "--json",
        str(record_path),
    )

    assert result.returncode == EXIT["GUARDS"], result.stdout + result.stderr
    change = json.loads(record_path.read_text(encoding="utf-8"))[
        "change_under_test"
    ]
    assert change["source_files"] == ["widgets.py"]
    assert (
        "declared with --change and confirmed to differ"
        in change["how_determined"]
    )
    assert change["how_determined"].endswith(
        "declared but identical at the baseline, so not counted: gadgets.py"
    )


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
    repo = scratch_repo(tmp_path)
    write(
        repo / "tests" / "test_widgets.py",
        "import time\n\n\ndef test_hangs_forever():\n    time.sleep(30)\n",
    )
    commit_all(repo, "a test that hangs")

    proc = _run(
        repo, "tests/test_widgets.py", "HEAD", "--per-test-timeout", "1"
    )

    assert "INDETERMINATE" in proc.stdout
    assert "test_hangs_forever" in proc.stdout


def test_a_key_error_on_a_package_result_guards(tmp_path: Path) -> None:
    """A KeyError on a package result at the assert line scores GUARDS.

    An assertion that inspects a package result by a key the baseline's result lacks raises
    KeyError at the assert line inside the test file, and the tool reports GUARDS, behavioral,
    naming KeyError.
    """
    repo = scratch_repo(tmp_path)
    write(repo / "widgets.py", "def describe():\n    return {'size': 3}\n")
    baseline = commit_all(repo, "describe carries no color")

    write(
        repo / "widgets.py",
        "def describe():\n    return {'size': 3, 'color': 'red'}\n",
    )
    write(
        repo / "tests" / "test_widgets.py",
        "def test_describe_carries_a_color():\n"
        "    from widgets import describe\n"
        "    assert describe()['color'] == 'red'\n",
    )
    commit_all(repo, "describe carries a color")

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["GUARDS"], result.stdout + result.stderr
    assert "GUARDS" in result.stdout
    assert "[behavioral]" in result.stdout
    assert "KeyError" in result.stdout


# ── the run's scratch directory ──────────────────────────────────────────────

SCRATCH_REMOVAL_LINE = "scratch directory left in place"

SUBJECT_REFUSING_ITS_OWN_REPOSITORY = (
    "from pathlib import Path\n"
    "\n"
    "REPO = Path(__file__).resolve().parent.parent\n"
    "\n"
    "\n"
    "def accept_root(root):\n"
    "    root = Path(root).resolve()\n"
    "    if root.is_relative_to(REPO):\n"
    "        raise ValueError(f'{root} is under the repository ({REPO})')\n"
    "    return root\n"
)

TEST_OF_THE_SUBJECT = (
    "import importlib.util\n"
    "from pathlib import Path\n"
    "\n"
    "\n"
    "def _subject():\n"
    "    path = Path(__file__).resolve().parent.parent / 'tools' / 'subject.py'\n"
    "    spec = importlib.util.spec_from_file_location('subject', path)\n"
    "    module = importlib.util.module_from_spec(spec)\n"
    "    spec.loader.exec_module(module)\n"
    "    return module\n"
    "\n"
    "\n"
    "def test_a_root_outside_the_repository_is_accepted(tmp_path):\n"
    "    assert _subject().accept_root(tmp_path) == tmp_path.resolve()\n"
)

TEST_LEAVING_READ_ONLY_FILES = (
    "import stat\n"
    "\n"
    "\n"
    "def test_leaves_read_only_files(tmp_path):\n"
    "    plain = tmp_path / 'plain.txt'\n"
    "    plain.write_text('plain')\n"
    "    plain.chmod(stat.S_IREAD)\n"
    "    objects = tmp_path / 'objects' / 'ab'\n"
    "    objects.mkdir(parents=True)\n"
    "    packed = objects / 'cdef'\n"
    "    packed.write_text('packed')\n"
    "    packed.chmod(stat.S_IREAD)\n"
    "    objects.chmod(stat.S_IREAD | stat.S_IEXEC)\n"
)

# Holds a file open until the release file appears, then closes it and writes the done file. The
# wait is bounded by the 300 s per-test timeout CI runs the suite under (ci.yml, pytest --timeout).
FILE_HOLDER = (
    "import sys\n"
    "import time\n"
    "from pathlib import Path\n"
    "\n"
    "held, release, done = map(Path, sys.argv[1:4])\n"
    "deadline = time.monotonic() + 300\n"
    "with open(held, 'w') as handle:\n"
    "    print('ready', flush=True)\n"
    "    while not release.exists() and time.monotonic() < deadline:\n"
    "        time.sleep(0.05)\n"
    "done.write_text('done')\n"
)

TEST_LEAVING_A_FILE_A_PROCESS_HOLDS = (
    "import os\n"
    "import subprocess\n"
    "import sys\n"
    "from pathlib import Path\n"
    "\n"
    "\n"
    "def test_leaves_a_file_a_process_still_holds(tmp_path):\n"
    "    hold = Path(os.environ['FILE_HOLD_DIR'])\n"
    "    holder = subprocess.Popen(\n"
    "        [\n"
    "            sys.executable,\n"
    "            str(hold / 'holder.py'),\n"
    "            str(tmp_path / 'held.txt'),\n"
    "            str(hold / 'release'),\n"
    "            str(hold / 'done'),\n"
    "        ],\n"
    "        cwd=hold,\n"
    "        stdin=subprocess.DEVNULL,\n"
    "        stdout=subprocess.PIPE,\n"
    "        stderr=subprocess.DEVNULL,\n"
    "    )\n"
    "    assert holder.stdout.readline().strip() == b'ready'\n"
)


def _run_baseline_leaving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, test_source: str
) -> tuple[subprocess.CompletedProcess, Path]:
    """Run the tool over a repository whose guard test is ``test_source``.

    The run's temporary folder is redirected to a folder under ``tmp_path``, returned beside the
    result, so a scratch directory left behind is found there.
    """
    scratch_root = _redirect_temporary_folder(tmp_path, monkeypatch)
    repo, baseline = _fixed_double(tmp_path, test_source)
    return _run(repo, "tests/test_widgets.py", baseline), scratch_root


def test_the_baseline_runs_temporary_root_lies_outside_the_materialized_tree(
    tmp_path: Path,
) -> None:
    """A test given its ``tmp_path`` is never given one inside the baseline tree.

    A subject that finds its repository from its own file location and refuses a root under it
    accepts the test's ``tmp_path``, so the test passes at the baseline and the verdict is VACUOUS.
    """
    repo = scratch_repo(tmp_path)
    write(repo / "tools" / "subject.py", SUBJECT_REFUSING_ITS_OWN_REPOSITORY)
    write(repo / "widgets.py", "def double(x):\n    return x\n")
    baseline = commit_all(repo, "double is a no-op")

    write(repo / "widgets.py", "def double(x):\n    return x * 2\n")
    write(repo / "tests" / "test_subject.py", TEST_OF_THE_SUBJECT)
    commit_all(repo, "double actually doubles")

    result = _run(repo, "tests/test_subject.py", baseline)

    assert result.returncode == EXIT["VACUOUS"], result.stdout + result.stderr
    assert "VACUOUS" in result.stdout
    assert SCRATCH_REMOVAL_LINE not in result.stderr


def test_a_scratch_directory_holding_read_only_files_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scratch directory holding read-only files is removed, silently.

    The baseline run's test leaves a read-only file and a read-only file in a directory set read
    and execute only, which a plain removal cannot delete; no scratch directory remains and no
    removal line is printed.
    """
    result, scratch_root = _run_baseline_leaving(
        tmp_path, monkeypatch, TEST_LEAVING_READ_ONLY_FILES
    )

    assert result.returncode == EXIT["VACUOUS"], result.stdout + result.stderr
    assert list(scratch_root.glob("failbefore-*")) == []
    assert SCRATCH_REMOVAL_LINE not in result.stderr, result.stderr


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="only Windows refuses to delete a file another process holds open",
)
def test_a_scratch_directory_that_cannot_be_removed_is_named_on_standard_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scratch directory the tool cannot remove is named on standard error, verdict unchanged.

    The baseline run's test starts a process that holds a file under the scratch directory open
    past the tool's removal. This test releases the process and removes the directory left behind.
    """
    hold = tmp_path / "hold"
    hold.mkdir()
    write(hold / "holder.py", FILE_HOLDER)
    monkeypatch.setenv("FILE_HOLD_DIR", str(hold))

    try:
        result, scratch_root = _run_baseline_leaving(
            tmp_path, monkeypatch, TEST_LEAVING_A_FILE_A_PROCESS_HOLDS
        )
        left = sorted(scratch_root.glob("failbefore-*"))

        assert result.returncode == EXIT["VACUOUS"], (
            result.stdout + result.stderr
        )
        assert len(left) == 1, result.stdout + result.stderr
        assert SCRATCH_REMOVAL_LINE in result.stderr, (
            result.stdout + result.stderr
        )
        assert str(left[0]) in result.stderr
        assert "held.txt" in result.stderr
    finally:
        write(hold / "release", "")
        # The holder closes its file within one 0.05 s poll of the release; 60 s only bounds a hang.
        deadline = time.monotonic() + 60
        while not (hold / "done").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        for leftover in (tmp_path / "t").glob("failbefore-*"):
            shutil.rmtree(leftover)


# ── a vitest file ────────────────────────────────────────────────────────────

FRONTEND_PATH = FRONTEND.relative_to(REPO_ROOT).as_posix()
INSTALL = FRONTEND / "node_modules"
WIDGET_TEST = f"{FRONTEND_PATH}/src/widget.test.ts"


def _scratch_frontend(repo: Path) -> Path:
    """Give ``repo`` a frontend at the real frontend's path; return its directory.

    It holds the real frontend's scripts, a ``tsconfig.json`` including ``src``, an empty vitest
    configuration and an empty ``src``, and git ignores its ``node_modules``, so what is made there
    is never committed nor counted as a change.
    """
    frontend = repo / FRONTEND_PATH
    shutil.copytree(FRONTEND / "scripts", frontend / "scripts")
    write(frontend / "tsconfig.json", '{"include": ["src"]}\n')
    write(frontend / "package.json", '{"type": "module"}\n')
    write(frontend / "vitest.config.mjs", "export default {};\n")
    (frontend / "src").mkdir()
    write(repo / ".git" / "info" / "exclude", "node_modules\n")
    return frontend


@pytest.fixture
def linked_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Yield a scratch repository whose frontend's install links each package of this checkout's.

    The scratch frontend's ``node_modules`` is a directory of its own holding one link per entry
    of this checkout's install not named with a leading dot, so whatever a run writes into the
    install it links to lands there, apart from every other test. The tool's temporary folder is
    redirected under ``tmp_path``. Teardown removes the link a run left in its tree and then each
    package link, before anything removes a directory holding one.
    """
    scratch_root = _redirect_temporary_folder(tmp_path, monkeypatch)
    repo = scratch_repo(tmp_path)
    install = _scratch_frontend(repo) / "node_modules"
    install.mkdir()
    packages = sorted(INSTALL.iterdir()) if INSTALL.is_dir() else []
    try:
        for package in packages:
            if not package.name.startswith("."):
                (install / package.name).symlink_to(
                    package, target_is_directory=package.is_dir()
                )
        yield repo
    finally:
        try:
            for run in scratch_root.glob("failbefore-*"):
                left = run / "tree" / FRONTEND_PATH / "node_modules"
                if left.is_symlink():
                    left.unlink()
        finally:
            for entry in install.iterdir():
                if entry.is_symlink():
                    entry.unlink()


def _widget_repo(
    repo: Path, baseline_source: str, head_source: str, test_source: str
) -> str:
    """Commit ``baseline_source`` as the frontend's widget, then the head's widget and its test.

    Returns the baseline revision; the head commit leaves the working tree clean.
    """
    src = repo / FRONTEND_PATH / "src"
    write(src / "widget.ts", baseline_source)
    baseline = commit_all(repo, "the baseline's widget")
    write(src / "widget.ts", head_source)
    write(src / "widget.test.ts", test_source)
    commit_all(repo, "the head's widget and its test")
    assert git(repo, "status", "--porcelain") == ""
    return baseline


DOUBLE_TEST = (
    'import { expect, test } from "vitest";\n'
    "\n"
    'import { double } from "./widget";\n'
    "\n"
    'test("double doubles", () => {\n'
    "  expect(double(3)).toBe(6);\n"
    "});\n"
)
NO_OP_DOUBLE = "export function double(x: number): number {\n  return x;\n}\n"
DOUBLE = "export function double(x: number): number {\n  return x * 2;\n}\n"


def test_a_vitest_assertion_failing_at_the_baseline_guards_and_leaves_the_install_alone(
    linked_repo: Path,
) -> None:
    """A vitest assertion failing at the baseline scores GUARDS, behavioral, exit 0.

    The install the baseline tree links to holds the same entries afterward, vitest among them, so
    the run neither removed anything nor wrote a cache or a bundled configuration through the link.
    """
    baseline = _widget_repo(linked_repo, NO_OP_DOUBLE, DOUBLE, DOUBLE_TEST)
    install = linked_repo / FRONTEND_PATH / "node_modules"
    entries_before = sorted(entry.name for entry in install.iterdir())

    result = _run(linked_repo, WIDGET_TEST, baseline, "-k", "double doubles")

    assert result.returncode == EXIT["GUARDS"], result.stdout + result.stderr
    assert "GUARDS" in result.stdout
    assert "[behavioral]" in result.stdout
    assert (install / "vitest" / "package.json").is_file()
    assert sorted(entry.name for entry in install.iterdir()) == entries_before


def test_a_vitest_test_passing_at_a_baseline_before_a_source_change_is_vacuous(
    linked_repo: Path,
) -> None:
    """A vitest test that passes at a baseline a source change follows scores VACUOUS, exit 1."""
    baseline = _widget_repo(
        linked_repo,
        DOUBLE,
        DOUBLE + "\nexport const TRIPLE = 3;\n",
        DOUBLE_TEST,
    )

    result = _run(linked_repo, WIDGET_TEST, baseline)

    assert result.returncode == EXIT["VACUOUS"], result.stdout + result.stderr
    assert "VACUOUS" in result.stdout


def test_a_vitest_test_timing_out_at_the_baseline_is_indeterminate(
    linked_repo: Path,
) -> None:
    """A vitest test that does not finish within its own timeout scores INDETERMINATE, exit 2."""
    baseline = _widget_repo(
        linked_repo,
        "export function settle(): Promise<number> {\n  return new Promise(() => {});\n}\n",
        "export function settle(): Promise<number> {\n  return Promise.resolve(1);\n}\n",
        'import { expect, test } from "vitest";\n'
        "\n"
        'import { settle } from "./widget";\n'
        "\n"
        'test("settle settles", async () => {\n'
        "  expect(await settle()).toBe(1);\n"
        "}, 200);\n",
    )

    result = _run(linked_repo, WIDGET_TEST, baseline)

    assert result.returncode == EXIT["INDETERMINATE"], (
        result.stdout + result.stderr
    )
    assert "settle settles" in _why(result)


def test_a_type_error_raised_in_a_vitest_test_is_refused_as_not_compared(
    linked_repo: Path,
) -> None:
    """A ``TypeError`` raised in a vitest test's own file at the baseline is refused, test_code."""
    baseline = _widget_repo(
        linked_repo,
        "export function paint(): { color: string } | null {\n  return null;\n}\n",
        'export function paint(): { color: string } | null {\n  return { color: "red" };\n}\n',
        'import { expect, test } from "vitest";\n'
        "\n"
        'import { paint } from "./widget";\n'
        "\n"
        'test("paint carries a color", () => {\n'
        '  expect(paint()!.color).toBe("red");\n'
        "});\n",
    )

    result = _run(linked_repo, WIDGET_TEST, baseline)

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    assert "[test_code]" in result.stdout
    assert CALL_PHASE_SENTENCE in _why(result)


@pytest.mark.parametrize("hook", ["beforeEach", "beforeAll"])
def test_a_vitest_hook_raising_at_the_baseline_is_refused_as_a_fixture(
    linked_repo: Path, hook: str
) -> None:
    """A vitest hook that raises at the baseline is refused as not compared, kind ``fixture``."""
    baseline = _widget_repo(
        linked_repo,
        'export function prepare(): void {\n  throw new Error("not ready");\n}\n',
        "export function prepare(): void {}\n",
        f'import {{ {hook}, expect, test }} from "vitest";\n'
        "\n"
        'import { prepare } from "./widget";\n'
        "\n"
        f"{hook}(() => {{\n"
        "  prepare();\n"
        "});\n"
        "\n"
        'test("prepared", () => {\n'
        "  expect(1).toBe(1);\n"
        "});\n",
    )

    result = _run(linked_repo, WIDGET_TEST, baseline)

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    assert "[fixture]" in result.stdout
    assert f"a fixture failed during {hook}" in _why(result)


def test_a_vitest_file_with_no_install_is_refused_naming_the_install_command(
    tmp_path: Path,
) -> None:
    """A frontend with no installed packages is refused, naming the command that installs them."""
    repo = scratch_repo(tmp_path)
    _scratch_frontend(repo)
    baseline = _widget_repo(repo, NO_OP_DOUBLE, DOUBLE, DOUBLE_TEST)

    result = _run(repo, WIDGET_TEST, baseline)

    assert result.returncode == EXIT["REFUSED"], result.stdout + result.stderr
    assert f"npm ci` in {FRONTEND_PATH}" in _why(result)


def test_a_changed_frontend_test_alone_leaves_a_pytest_pass_indeterminate(
    tmp_path: Path,
) -> None:
    """A frontend test is the test side, so a pytest pass with no other change is INDETERMINATE.

    The only difference between the baseline and the head is a frontend test file, so no source
    file differs and the baseline is not shown to precede a change.
    """
    repo = scratch_repo(tmp_path)
    src = _scratch_frontend(repo) / "src"
    write(src / "widget.ts", DOUBLE)
    write(src / "widget.test.ts", DOUBLE_TEST)
    write(
        repo / "tests" / "test_widgets.py",
        "def test_passes():\n    assert True\n",
    )
    baseline = commit_all(repo, "a passing pytest test and a frontend test")
    write(src / "widget.test.ts", DOUBLE_TEST.replace("toBe(6)", "toEqual(6)"))
    commit_all(repo, "the frontend test alone changes")

    result = _run(repo, "tests/test_widgets.py", baseline)

    assert result.returncode == EXIT["INDETERMINATE"], (
        result.stdout + result.stderr
    )
    assert "no source file differs" in _why(result)


# Starts the holder over a file beside the test, in a folder a bottom-up removal reaches before the
# frontend folder holding the link, and returns once the holder has the file open.
VITEST_TEST_LEAVING_A_FILE_A_PROCESS_HOLDS = (
    'import { spawn } from "node:child_process";\n'
    'import { join } from "node:path";\n'
    'import { fileURLToPath } from "node:url";\n'
    "\n"
    'import { expect, test } from "vitest";\n'
    "\n"
    'test("leaves a file a process still holds", async () => {\n'
    "  const hold = process.env.FILE_HOLD_DIR as string;\n"
    '  const held = fileURLToPath(new URL("held.txt", import.meta.url));\n'
    "  const holder = spawn(\n"
    "    process.env.HOLDER_PYTHON as string,\n"
    '    [join(hold, "holder.py"), held, join(hold, "release"), join(hold, "done")],\n'
    '    { cwd: hold, detached: true, stdio: ["ignore", "pipe", "ignore"] },\n'
    "  );\n"
    "  const line: string = await new Promise((resolve) =>\n"
    '    holder.stdout.once("data", (data) => resolve(String(data))),\n'
    "  );\n"
    "  holder.stdout.destroy();\n"
    "  holder.unref();\n"
    '  expect(line.trim()).toBe("ready");\n'
    "});\n"
)


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="only Windows refuses to delete a file another process holds open",
)
def test_a_scratch_directory_left_in_place_holds_no_link_to_the_install(
    linked_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A vitest run's scratch directory the tool cannot remove is left holding no link.

    The vitest test starts a process that holds a file beside it open past the tool's removal, so
    the removal stops before it reaches the frontend folder holding the link; the link into the
    install is gone all the same, since the tool removes it first. This test releases the process
    and removes the directory left behind.
    """
    hold = linked_repo.parent / "hold"
    hold.mkdir()
    write(hold / "holder.py", FILE_HOLDER)
    monkeypatch.setenv("FILE_HOLD_DIR", str(hold))
    monkeypatch.setenv("HOLDER_PYTHON", sys.executable)
    baseline = _widget_repo(
        linked_repo,
        DOUBLE,
        DOUBLE + "\nexport const TRIPLE = 3;\n",
        VITEST_TEST_LEAVING_A_FILE_A_PROCESS_HOLDS,
    )
    scratch_root = linked_repo.parent / "t"

    try:
        result = _run(linked_repo, WIDGET_TEST, baseline)
        left = sorted(scratch_root.glob("failbefore-*"))

        assert result.returncode == EXIT["VACUOUS"], (
            result.stdout + result.stderr
        )
        assert len(left) == 1, result.stdout + result.stderr
        assert SCRATCH_REMOVAL_LINE in result.stderr, (
            result.stdout + result.stderr
        )
        assert "held.txt" in result.stderr
        assert not os.path.lexists(
            left[0] / "tree" / FRONTEND_PATH / "node_modules"
        )
    finally:
        write(hold / "release", "")
        # The holder closes its file within one 0.05 s poll of the release; 60 s only bounds a hang.
        deadline = time.monotonic() + 60
        while not (hold / "done").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        for leftover in scratch_root.glob("failbefore-*"):
            link = leftover / "tree" / FRONTEND_PATH / "node_modules"
            if link.is_symlink():
                link.unlink()
            shutil.rmtree(leftover)
