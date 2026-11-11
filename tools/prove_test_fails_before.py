"""Run a test against a baseline tree without the change under test; report whether it fails.

The baseline is materialized with ``git archive`` (the working tree is never touched), the current
``tests/`` directory is overlaid on it, the baseline's own source is proved to be what gets
imported, and pytest's per-test outcome is read rather than its exit code.

    python tools/prove_test_fails_before.py tests/test_foo.py
    python tools/prove_test_fails_before.py tests/test_foo.py -k "new_behavior"
    python tools/prove_test_fails_before.py tests/test_foo.py --baseline 196eedf1~1
    python tools/prove_test_fails_before.py tests/test_foo.py --json out.json
    python tools/prove_test_fails_before.py tests/test_foo.py --test-rev 8b09bd17
        --baseline ae3dbbb8
    python tools/prove_test_fails_before.py tests/test_foo.py --per-test-timeout 30

Four verdicts, each its own exit code:

    GUARDS (0)         at least one selected test failed at the baseline in the call phase, on
                       the assertion it names (the raised exception's own class is
                       ``AssertionError`` or ``Failed``), on an exception raised from outside
                       ``tests/`` (the code under test raised), or on any other exception raised
                       in the test's own code except a ``TypeError``; the failing assertion is
                       printed and recorded, each failure not counted is named beside it
    VACUOUS (1)        every selected test passed at a baseline shown to precede the change
    INDETERMINATE (2)  every selected test passed, but the baseline is not shown to precede the
                       change, so passing says nothing about the test; or, under
                       ``--per-test-timeout``, a selected test did not finish within it, named in
                       the verdict, so a hung baseline is told apart from a merely slow one
    REFUSED (3)        no verdict is available: the test file is outside ``tests/`` or does not
                       exist, ``--test-rev`` was given without ``--baseline``, the baseline does
                       not resolve or no tree here precedes the change, nothing was selected,
                       everything was skipped, the baseline could not collect the file, the
                       baseline's own source was not what got imported, pytest did not finish
                       within ``--timeout``, hit an internal error or never reported an outcome,
                       or no failure is behavioral (each rests on a missing import or file or is
                       fixture-shaped), so no behavior was compared. A setup or teardown
                       failure that is neither a missing import or file nor a pytest-timeout
                       report is fixture-shaped, whatever raised it; a call-phase failure is
                       fixture-shaped when it is a ``TypeError`` whose crash frame is placed
                       under ``tests/`` of the baseline's tree (a frame that cannot be placed
                       there, a library's, is behavioral), whatever its message. This is a
                       policy: the class and the place of the raise do not establish that a call
                       failed to fit, so no wording is read, and a
                       ``TypeError`` from handling what the code under test returned (a ``None``
                       subscripted) is refused the same way. Such a test becomes a guard by
                       asserting on what was returned before using it.

Not closed, each still counted as behavioral: an exception of another class raised in the test's
own code before the behavior is reached; a wrong call made in a helper outside ``tests/`` or
inside package code (the second rightly, since the package's own behavior was reached); an
assertion that fails before the call under test.

The baseline is ``--baseline`` when given, else ``HEAD`` for uncommitted work, else the
merge-base against the integration branch; the baseline used, how it was chosen and the source
files the change touches are recorded with the verdict.

``--test-rev`` takes the test tree from a revision instead of the working tree, and requires
``--baseline``.

``--baseline-from-working-tree`` snapshots the working tree (tracked modifications and untracked
files, ignored files excluded) into a commit object and prints its hash, touching neither the
working tree, the index nor the stash list, and computes no verdict.

``--per-test-timeout`` passes ``--timeout`` through to pytest-timeout; a selected test that does
not finish within it is named and the verdict is INDETERMINATE.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEST_TREE = "tests"
PLUGIN_MODULE = "_fail_before_outcome"
OUTCOME_ENV = "FAIL_BEFORE_OUTCOME_JSON"
CURRENT_TEST_ENV = "FAIL_BEFORE_CURRENT_TEST"

GUARDS, VACUOUS, INDETERMINATE, REFUSED = (
    "GUARDS",
    "VACUOUS",
    "INDETERMINATE",
    "REFUSED",
)
EXIT = {GUARDS: 0, VACUOUS: 1, INDETERMINATE: 2, REFUSED: 3}

PLUGIN_SOURCE = '''\
"""Record what pytest observed, so a caller reads outcomes instead of an exit code."""
import json
import os

_state = {"collected": None, "collect_errors": [], "tests": [], "internal_error": None}
_exc_types = {}  # (nodeid, when) -> the raised exception's own class name


def pytest_runtest_logstart(nodeid, location):
    """Name the test that is about to run, flushed to disk immediately.

    A thread-method pytest-timeout kill (os._exit, no SIGALRM) never lets
    pytest_sessionfinish write the outcome record; this file, written before the test body
    runs and fsynced, is the only way the caller learns which test was running when the
    process died.
    """
    path = os.environ.get("FAIL_BEFORE_CURRENT_TEST")
    if not path:
        return
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(nodeid)
        fh.flush()
        os.fsync(fh.fileno())


def _headline(text):
    """The first E-prefixed line, which is the exception. Later ones are its continuation."""
    marked = [line.strip() for line in text.splitlines() if line.strip().startswith("E ")]
    if marked:
        return marked[0][2:].strip()
    return text.strip().splitlines()[-1] if text.strip() else ""


def pytest_collection_finish(session):
    _state["collected"] = len(session.items)


def pytest_collectreport(report):
    if report.failed:
        _state["collect_errors"].append(
            {"nodeid": report.nodeid, "detail": str(report.longrepr)[-4000:]}
        )


def _crash_location(report):
    """The crash frame's file and line, from ``longrepr.reprcrash`` when the report carries one.

    Absent for a report whose ``longrepr`` is a plain string (a skip, an internal collection
    shortcut) rather than an exception repr; the caller treats a missing location as unknown
    rather than as evidence of anything.
    """
    crash = getattr(getattr(report, "longrepr", None), "reprcrash", None)
    path = getattr(crash, "path", None) if crash is not None else None
    lineno = getattr(crash, "lineno", None) if crash is not None else None
    return path, lineno


def pytest_exception_interact(node, call, report):
    """Stash the raised exception's own class name, keyed by (nodeid, phase).

    The headline text pytest's assertion rewriting prints omits the class name for a bare
    comparison (``assert 3 == 6`` prints only ``assert 3 == 6``, never ``AssertionError: ...``),
    while keeping it for ``in``/``not in`` and for any other exception, so headline text alone
    cannot tell an assertion failure apart from another exception rewritten the same way. The
    exception object itself always carries its own type, whatever the message looks like.
    """
    if call.excinfo is not None:
        _exc_types[(report.nodeid, call.when)] = call.excinfo.typename


def pytest_runtest_logreport(report):
    if report.when == "call":
        entry = {"nodeid": report.nodeid, "phase": "call", "outcome": report.outcome}
    elif report.when == "setup" and report.skipped:
        entry = {"nodeid": report.nodeid, "phase": "setup", "outcome": "skipped"}
    elif report.failed:
        entry = {"nodeid": report.nodeid, "phase": report.when, "outcome": "error"}
    else:
        return
    if report.failed:
        entry["detail"] = report.longreprtext[-4000:]
        entry["headline"] = _headline(report.longreprtext)
        entry["crash_path"], entry["crash_lineno"] = _crash_location(report)
    _state["tests"].append(entry)


def pytest_internalerror(excrepr):
    _state["internal_error"] = str(excrepr)[-4000:]


def pytest_sessionfinish(session, exitstatus):
    # pytest_exception_interact fires after pytest_runtest_logreport for the same phase, so the
    # exception type is looked up here rather than at entry-construction time.
    for entry in _state["tests"]:
        if entry.get("outcome") in ("failed", "error"):
            entry["exc_typename"] = _exc_types.get((entry["nodeid"], entry["phase"]))
    _state["exitstatus"] = int(exitstatus)
    with open(os.environ["FAIL_BEFORE_OUTCOME_JSON"], "w", encoding="utf-8") as fh:
        json.dump(_state, fh)
'''


def _test_file_relative_to_repo(raw: str) -> tuple[str | None, str]:
    """Return ``raw`` normalized to a repo-relative, forward-slash path under ``tests/``.

    Accepts a relative path as always, or an absolute one naming a file inside this repository;
    either way the return is a tree-relative path. A relative path is resolved against the
    repository root, never the caller's ``cwd``, and either spelling's ``..`` segments are
    collapsed before the ``tests/`` check, so a path that only reads as inside ``tests/`` before
    normalizing (``tests/../tools/x.py``) is refused rather than admitted. Returns
    ``(None, reason)`` when ``raw`` resolves outside ``tests/``.
    """
    path = Path(raw)
    anchored = path if path.is_absolute() else REPO / path
    try:
        rel = anchored.resolve().relative_to(REPO.resolve())
    except ValueError:
        return None, f"{raw} is outside this repository ({REPO})."
    rel_str = str(rel).replace("\\", "/")
    if not rel_str.startswith(f"{TEST_TREE}/"):
        return None, f"{raw} resolves to {rel_str}, outside {TEST_TREE}/."
    return rel_str, ""


def git_output(*args: str) -> str:
    """Return git's stdout for ``args`` run in the repository; ``RuntimeError`` on failure."""
    proc = subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed: {proc.stderr.strip()}"
        )
    return proc.stdout


def _is_test_side(path: str) -> bool:
    """Return whether ``path`` is one the tool overlays from the current tree.

    A diff in such a path is not the change under test.
    """
    return path.replace("\\", "/").startswith(f"{TEST_TREE}/")


def _uncommitted_paths() -> list[str]:
    out = git_output("status", "--porcelain", "--untracked-files=all")
    paths = []
    for line in out.splitlines():
        if not line.strip():
            continue
        entry = line[3:].strip().strip('"')
        paths.append(entry.split(" -> ")[-1] if " -> " in entry else entry)
    return paths


def _resolve_baseline(
    named: str | None, integration: str
) -> tuple[str | None, str, str]:
    """Return (revision, how it was chosen, why it is unusable if it is)."""
    if named:
        try:
            sha = git_output(
                "rev-parse", "--verify", f"{named}^{{commit}}"
            ).strip()
        except RuntimeError as exc:
            return None, f"named by the caller as {named}", str(exc)
        return sha, f"named by the caller as {named}", ""

    source_side = [p for p in _uncommitted_paths() if not _is_test_side(p)]
    if source_side:
        return (
            git_output("rev-parse", "HEAD").strip(),
            "HEAD, because the change under test is uncommitted",
            "",
        )

    try:
        head = git_output("rev-parse", "HEAD").strip()
        base = git_output("merge-base", "HEAD", integration).strip()
    except RuntimeError as exc:
        return None, f"merge-base against {integration}", str(exc)
    if base == head:
        return (
            None,
            f"merge-base against {integration}",
            (
                "the working tree carries no uncommitted source change and HEAD is already "
                f"contained in {integration}, so no tree here precedes a change. Name the "
                "pre-change revision with --baseline."
            ),
        )
    return base, f"merge-base of HEAD and {integration}", ""


def _changed_source_files(
    baseline: str, declared: list[str], test_rev: str | None
) -> tuple[list[str], str]:
    """Return the source-side files that differ from the baseline, and how they were found.

    The comparison is against the tree the test comes from; a ``declared`` path counts only when
    it differs too.
    """
    if test_rev:
        tracked = git_output(
            "diff", "--name-only", baseline, test_rev, "--"
        ).splitlines()
        untracked: list[str] = []
        how = f"git diff {baseline[:8]}..{test_rev[:8]}, test tree excluded"
    else:
        tracked = git_output(
            "diff", "--name-only", baseline, "--"
        ).splitlines()
        untracked = git_output(
            "ls-files", "--others", "--exclude-standard"
        ).splitlines()
        how = f"git diff against {baseline[:8]} plus untracked files, test tree excluded"
    changed = sorted(
        {
            p.strip()
            for p in [*tracked, *untracked]
            if p.strip() and not _is_test_side(p.strip())
        }
    )
    if declared:
        wanted = {p.replace("\\", "/") for p in declared}
        kept = sorted(wanted & set(changed))
        ignored = sorted(wanted - set(changed))
        note = f"declared with --change and confirmed to differ ({how})"
        if ignored:
            note += (
                "; declared but identical at the baseline, so not counted: "
                f"{', '.join(ignored)}"
            )
        return kept, note
    return changed, how


def _snapshot_via_temporary_index(head: str, index_path: Path) -> str | None:
    """Build the snapshot commit through a scratch index; ``None`` if this form fails here."""
    env = {**os.environ, "GIT_INDEX_FILE": str(index_path)}
    read = subprocess.run(
        ["git", "read-tree", head],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    if read.returncode != 0:
        return None
    added = subprocess.run(
        ["git", "add", "-A"], cwd=REPO, env=env, capture_output=True, text=True
    )
    if added.returncode != 0:
        return None
    tree = subprocess.run(
        ["git", "write-tree"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    if tree.returncode != 0 or not tree.stdout.strip():
        return None
    commit = subprocess.run(
        [
            "git",
            "commit-tree",
            tree.stdout.strip(),
            "-p",
            head,
            "-m",
            "prove_test_fails_before.py working-tree snapshot",
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    if commit.returncode != 0 or not commit.stdout.strip():
        return None
    return commit.stdout.strip()


def _snapshot_via_stash_create() -> str | None:
    """Snapshot through ``git stash create``; ``None`` when it produces nothing.

    The fallback when the temporary-index form fails here: covers tracked changes only, and
    touches neither the tree, the index, nor the stash list.
    """
    out = subprocess.run(
        [
            "git",
            "stash",
            "create",
            "prove_test_fails_before.py working-tree snapshot",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    hashed = out.stdout.strip()
    return hashed or None


def snapshot_working_tree() -> tuple[str, str]:
    """Make a commit object for the working tree exactly as it stands now.

    It holds tracked modifications and untracked files, ignored files excluded. Returns
    ``(commit_hash, method)``. Touches neither the working tree, the real index, nor the stash
    list.
    """
    with tempfile.TemporaryDirectory(prefix="failbefore-index-") as scratch:
        head = git_output("rev-parse", "HEAD").strip()
        commit = _snapshot_via_temporary_index(head, Path(scratch) / "index")
        if commit:
            return commit, "temporary index (git add -A, git write-tree)"
    commit = _snapshot_via_stash_create()
    if commit:
        return commit, "git stash create (tracked changes only)"
    raise RuntimeError(
        "could not snapshot the working tree: the temporary-index form failed here and git "
        "stash create produced nothing, most likely because there are no tracked changes to stash"
    )


def materialize(rev: str, dest: Path) -> None:
    """Extract a revision's whole tree into `dest`. The working tree is never touched."""
    dest.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "archive", rev], cwd=REPO, check=True, stdout=subprocess.PIPE
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r|") as tar:
        tar.extractall(dest, filter="data")


def _overlay_test_tree(dest: Path, test_rev: str | None) -> int:
    """Copy the whole test tree, from ``test_rev`` or the working tree, into ``dest``.

    Returns the number of files copied.
    """
    if test_rev:
        archive = subprocess.run(
            ["git", "archive", test_rev, "--", TEST_TREE],
            cwd=REPO,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r") as tar:
            files = [m for m in tar.getmembers() if m.isfile()]
            tar.extractall(dest, filter="data")
        return len(files)

    copied = 0
    for path in (REPO / TEST_TREE).rglob("*"):
        if "__pycache__" in path.parts or path.is_dir():
            continue
        target = dest / path.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    return copied


def child_env(
    tree: Path, outcome_json: Path, current_test_path: Path
) -> dict[str, str]:
    """Return the environment a pytest child runs under in the materialized ``tree``.

    ``PYTHONPATH`` names the tree's package sources and the tree itself, bytecode writing is off,
    the outcome and current-test paths are set, and ``TCIP_MIN_TESTS`` is removed.
    """
    srcs = [str(p) for p in sorted(tree.glob("packages/*/src"))]
    sep = ";" if sys.platform == "win32" else ":"
    env = {
        **os.environ,
        "PYTHONPATH": sep.join([*srcs, str(tree)]),
        "PYTHONDONTWRITEBYTECODE": "1",
        OUTCOME_ENV: str(outcome_json),
        CURRENT_TEST_ENV: str(current_test_path),
    }
    env.pop("TCIP_MIN_TESTS", None)
    return env


def prove_tree_imports(
    tree: Path, env: dict[str, str]
) -> tuple[bool, dict[str, str]]:
    """Confirm each package resolves inside the baseline tree, not the editable-installed checkout.

    Returns whether every package under ``tree``'s ``packages/*/src`` imported from inside
    ``tree`` under ``env``, and each package's resolved path or import failure.
    """
    names = sorted(
        {p.parent.name for p in tree.glob("packages/*/src/*/__init__.py")}
    )
    if not names:
        return True, {}
    probe = (
        "import importlib, json, sys\n"
        f"names = {names!r}\n"
        "out = {}\n"
        "for n in names:\n"
        "    try:\n"
        "        out[n] = getattr(importlib.import_module(n), '__file__', '') or ''\n"
        "    except Exception as exc:\n"
        "        out[n] = 'IMPORT FAILED: ' + repr(exc)\n"
        "sys.stdout.write(json.dumps(out))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tree,
        env=env,
        capture_output=True,
        text=True,
    )
    try:
        resolved = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return False, {
            "probe": f"probe produced no JSON: {proc.stdout!r} {proc.stderr[-500:]!r}"
        }
    root = str(tree.resolve()).lower()
    ok = all(
        str(Path(p).resolve()).lower().startswith(root)
        for p in resolved.values()
        if p and "IMPORT FAILED" not in p
    )
    ok = ok and not any("IMPORT FAILED" in p for p in resolved.values())
    return ok, resolved


def install_outcome_plugin(tree: Path) -> None:
    """Drop the outcome-recording plugin into a materialized tree, where pytest will import it."""
    (tree / f"{PLUGIN_MODULE}.py").write_text(PLUGIN_SOURCE, encoding="utf-8")


def run_capturing_outcome(
    tree: Path,
    targets: list[str],
    expr: str,
    env: dict[str, str],
    outcome_json: Path,
    timeout: int,
    per_test_timeout: float | None = None,
) -> tuple[subprocess.CompletedProcess[str], dict | None]:
    """Run pytest in `tree` and return its process plus what it observed.

    The observation is None if it reported none.
    """
    # The child session keeps its temporary root inside this run's own tree: pytest deletes all
    # but the last few roots under the shared one, including another session's, at every start.
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *targets,
        "-q",
        "-p",
        "no:cacheprovider",
        "-p",
        PLUGIN_MODULE,
        "--rootdir",
        str(tree),
        "--basetemp",
        str(tree / ".pytest-basetemp"),
    ]
    if expr:
        cmd += ["-k", expr]
    if per_test_timeout:
        cmd += ["--timeout", str(per_test_timeout)]
    proc = subprocess.run(
        cmd, cwd=tree, env=env, capture_output=True, text=True, timeout=timeout
    )
    observed = None
    if outcome_json.is_file():
        observed = json.loads(outcome_json.read_text(encoding="utf-8"))
        outcome_json.unlink()
    return proc, observed


_TIMEOUT_HEADLINE = re.compile(
    r"^(?:[A-Za-z_][\w.]*: )?Timeout \(>[\d.]+s\) from pytest-timeout\.$"
)


def _is_timeout(headline: str) -> bool:
    """Return whether this failure is pytest-timeout's own signal-method report.

    That report is its literal ``PYTEST_FAILURE_MESSAGE``; a hang the thread method reports
    through the ``FAIL_BEFORE_CURRENT_TEST`` marker is not one.

    The leading exception class is optional because pytest writes the headline both ways: bare
    where the timeout is the whole report, and prefixed ``Failed: `` where pytest-timeout's
    signal method raises it as an ordinary ``Failed`` inside the test, which only hosts carrying
    ``SIGALRM`` do.
    """
    return bool(_TIMEOUT_HEADLINE.match(headline.strip()))


def _is_unreached(headline: str, crash_path: str | None, tree: Path) -> bool:
    """Return whether a failure never reached the code under test.

    A missing module or import always counts. A missing file counts only when its crash frame sits
    outside ``tests/`` (:func:`_crash_outside_test_tree`).
    """
    if headline.startswith(("ModuleNotFoundError", "ImportError")):
        return True
    if headline.startswith("FileNotFoundError"):
        return _crash_outside_test_tree(crash_path, tree)
    return False


def _crash_relative_to_tree(crash_path: str | None, tree: Path) -> str | None:
    """Return the crash frame's forward-slash path relative to the materialized tree.

    ``None`` for a missing path (a report with no exception repr) and for one that cannot be
    placed under the tree, such as a frame in the standard library or site-packages.
    """
    if not crash_path:
        return None
    try:
        rel = Path(crash_path).resolve().relative_to(tree.resolve())
    except (OSError, ValueError):
        return None
    return str(rel).replace("\\", "/")


def _crash_under_test_tree(crash_path: str | None, tree: Path) -> bool:
    """Return whether the crash frame is placed positively under ``tests/`` in the tree.

    A frame under ``tests/`` is the test's own body or a fixture. A missing or unplaceable path
    is not under it.
    """
    rel = _crash_relative_to_tree(crash_path, tree)
    return rel is not None and rel.startswith(f"{TEST_TREE}/")


def _crash_outside_test_tree(crash_path: str | None, tree: Path) -> bool:
    """Return whether the crash frame is placed in the tree and outside ``tests/``.

    One outside it is the code under test raising. A missing or unplaceable path is never outside.
    """
    rel = _crash_relative_to_tree(crash_path, tree)
    return rel is not None and not rel.startswith(f"{TEST_TREE}/")


def _failure_kind(entry: dict, tree: Path) -> str:
    """Classify one failed test: ``unreached``, ``timeout``, ``behavioral`` or ``fixture``.

    An errored test is classified the same way.

    ``unreached``: the import never resolved, always; or a needed file was missing and the crash
    frame sits outside ``tests/`` (:func:`_is_unreached`). A missing file whose crash frame sits
    inside the test's own body is left to the rest of this function instead.

    ``timeout``: pytest-timeout's own signal-method report (:func:`_is_timeout`), checked before
    the phase split below, so a timeout during setup is ``timeout`` too.

    After those two checks, a setup or teardown failure (``entry["phase"]`` is not ``"call"``) is
    ``fixture``, whatever raised it, wherever the crash frame sits. The rest applies only to a
    call-phase failure:

    ``behavioral``: the test's own ``assert`` or ``pytest.fail``/``pytest.raises`` failed (the
    raised exception's own class is ``AssertionError`` or ``Failed``, read from
    ``exc_typename``), or the crash frame sits outside ``tests/``, meaning the code under test
    raised. ``fixture``: a ``TypeError`` whose crash frame is placed under ``tests/``
    (:func:`_crash_under_test_tree`), whatever its message. The class and the place of the raise
    do not establish that a call failed to fit, so the tool reads no wording and claims no cause.
    Any other call-phase exception, and a ``TypeError`` whose crash frame cannot be placed under
    the tree (a library's), is ``behavioral``.
    """
    headline = entry.get("headline", "")
    if _is_unreached(headline, entry.get("crash_path"), tree):
        return "unreached"
    if _is_timeout(headline):
        return "timeout"
    if entry.get("phase") != "call":
        return "fixture"
    if entry.get("exc_typename") in ("AssertionError", "Failed"):
        return "behavioral"
    if _crash_outside_test_tree(entry.get("crash_path"), tree):
        return "behavioral"
    if entry.get("exc_typename") == "TypeError" and _crash_under_test_tree(
        entry.get("crash_path"), tree
    ):
        return "fixture"
    return "behavioral"


def _not_counted(entry: dict, kind: str) -> str:
    """Name a failure that is not counted: its test, what was observed and in which phase."""
    where = entry["nodeid"]
    headline = entry.get("headline", "")
    if kind == "unreached":
        return f"{where}: a missing import or file ({headline})"
    if entry["phase"] == "call":
        raised = entry["exc_typename"]
        return f"{where}: the test's own code raised a {raised} before its assertion ({headline})"
    return f"{where}: a fixture failed during {entry['phase']} ({headline})"


def _classify(
    observed: dict, baseline_precedes: bool, tree: Path
) -> tuple[str, str]:
    if observed.get("internal_error"):
        return (
            REFUSED,
            "pytest hit an internal error, so nothing it reported can be trusted.",
        )
    if observed["collect_errors"]:
        first = observed["collect_errors"][0]["nodeid"]
        return REFUSED, (
            f"the baseline could not collect {first}. Zero assertions were evaluated, so this is "
            "not evidence either way. The test most likely references something only the change "
            "introduces, which means the baseline predates more than the change under test."
        )
    if not observed["collected"]:
        return (
            REFUSED,
            "nothing was selected, so nothing was evaluated. Check the -k expression.",
        )

    outcomes = [t["outcome"] for t in observed["tests"]]
    if outcomes and all(o == "skipped" for o in outcomes):
        return REFUSED, (
            f"all {len(outcomes)} selected tests were skipped, so nothing was evaluated. A "
            "module-level importorskip at the baseline will do this."
        )
    failed = [
        t for t in observed["tests"] if t["outcome"] in ("failed", "error")
    ]
    kinds = [_failure_kind(t, tree) for t in failed]
    unreached = [t for t, k in zip(failed, kinds) if k == "unreached"]
    timed_out = [t for t, k in zip(failed, kinds) if k == "timeout"]
    fixture = [t for t, k in zip(failed, kinds) if k == "fixture"]
    behavioral = [t for t, k in zip(failed, kinds) if k == "behavioral"]
    if timed_out:
        names = ", ".join(t["nodeid"] for t in timed_out)
        return INDETERMINATE, (
            f"{len(timed_out)} of {observed['collected']} selected test(s) did not finish within "
            f"--per-test-timeout ({names}), so passing or failing says nothing here: a hung "
            "baseline is a different fact from a slow pass or a real failure."
        )
    if failed and not behavioral and fixture:
        cause = (
            " A call that does not fit the baseline's code is one cause of this."
            if any(t["phase"] == "call" for t in fixture)
            else ""
        )
        also = (
            " Also not counted: "
            + "; ".join(_not_counted(t, "unreached") for t in unreached)
            + "."
            if unreached
            else ""
        )
        return REFUSED, (
            f"{len(fixture)} of {len(failed)} failing test(s) failed before reaching a comparison "
            "of behavior, so no behavior was compared and the run is not evidence either way: "
            + "; ".join(_not_counted(t, "fixture") for t in fixture)
            + f".{cause}{also}"
        )
    if failed and not behavioral:
        return REFUSED, (
            f"all {len(failed)} failing tests failed on a missing import or file rather than on "
            f"an assertion ({unreached[0]['headline']}), so the code under test was never reached "
            "and no behavior was compared. A baseline whose source will not import, or whose tree "
            "lacks a file the test needs, in the current environment cannot produce evidence."
        )
    if behavioral:
        others = [
            *(_not_counted(t, "fixture") for t in fixture),
            *(_not_counted(t, "unreached") for t in unreached),
        ]
        discount = (
            f"; not counted, as not evidence either way: {'; '.join(others)}"
            if others
            else ""
        )
        caveat = (
            ""
            if baseline_precedes
            else (
                " No source file differs between the baseline and the tree this test comes from, "
                "so this is a fact about the baseline rather than proof that a change here is "
                "guarded."
            )
        )
        return GUARDS, (
            f"{len(behavioral)} of {observed['collected']} selected tests failed on the behavior "
            f"under test at the baseline ({', '.join(t['nodeid'] for t in behavioral)})"
            f"{discount}.{caveat} The failures are recorded below."
        )
    if baseline_precedes:
        return VACUOUS, (
            "every selected test passed at a baseline shown to precede the change. It guards "
            "nothing; assert on what only the change produces."
        )
    return INDETERMINATE, (
        "every selected test passed, but no source file differs between the baseline and the "
        "working tree, so the baseline is not shown to precede the change and passing there says "
        "nothing about the test. Name the pre-change revision with --baseline."
    )


def main() -> int:
    """Run the command line's test at its baseline, print the verdict, return its exit code."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "test_file", help="path, repo-relative, e.g. tests/test_foo.py"
    )
    ap.add_argument("-k", dest="expr", default="", help="pytest -k expression")
    ap.add_argument(
        "--baseline",
        default=None,
        help="pre-change revision; default is HEAD for uncommitted work, else the "
        "merge-base against the integration branch",
    )
    ap.add_argument(
        "--baseline-from-working-tree",
        action="store_true",
        help="snapshot the current working tree into a commit and print its hash; "
        "computes no verdict. Take this before applying the fix, then re-run "
        "normally with --baseline <the printed hash>.",
    )
    ap.add_argument(
        "--integration",
        default="main",
        help="branch to take a merge-base against (default main)",
    )
    ap.add_argument(
        "--test-rev",
        default=None,
        help="take the test tree from this revision instead of the working tree, to "
        "check a guard claim already in the history; requires --baseline",
    )
    ap.add_argument(
        "--change",
        action="append",
        default=[],
        help="a source file the change touches; repeatable. Overrides the computed set",
    )
    ap.add_argument(
        "--json",
        dest="json_out",
        default=None,
        help="write the full record here",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="seconds to allow pytest (default 900)",
    )
    ap.add_argument(
        "--per-test-timeout",
        type=float,
        default=None,
        help="seconds pytest-timeout allows one test (passed through as --timeout); "
        "a test that does not finish within it scores INDETERMINATE by name, "
        "told apart from a hang that says nothing and a real pass or failure",
    )
    args = ap.parse_args()

    if args.baseline_from_working_tree:
        if args.baseline:
            ap.error(
                "--baseline-from-working-tree and --baseline name the baseline two ways at once"
            )
        commit, method = snapshot_working_tree()
        expr_flag = f"-k {args.expr!r} " if args.expr else ""
        print(f"snapshot {commit} ({method})")
        print(
            "computes no verdict by itself. This is the working tree as it stands right now: a "
            "snapshot that already holds the fix is a baseline the change does not precede, and a "
            "passing run against it is the same INDETERMINATE case as any other baseline already "
            "containing the change, never evidence the test guards anything. Apply the fix now, "
            "then run:\n"
            f"  python tools/prove_test_fails_before.py {args.test_file} {expr_flag}"
            f"--baseline {commit}"
        )
        return 0

    record: dict = {
        "test_file": args.test_file,
        "k": args.expr,
        "verdict": None,
        "test_tree_from": args.test_rev or "the working tree",
    }

    test_file, refusal = _test_file_relative_to_repo(args.test_file)
    if test_file is None:
        record.update(verdict=REFUSED, why=refusal)
        return _report(record, args.json_out)
    record["test_file"] = test_file

    if args.test_rev and not args.baseline:
        record.update(
            verdict=REFUSED,
            why=(
                "--test-rev needs an explicit --baseline. A change spans one commit per concern, "
                "so the commit before a test can be inside it, and guessing one would decide the "
                "verdict."
            ),
        )
        return _report(record, args.json_out)

    if args.test_rev:
        listed = git_output(
            "ls-tree", "--name-only", "-r", args.test_rev, "--", test_file
        ).strip()
        if not listed:
            record.update(
                verdict=REFUSED,
                why=f"{test_file} does not exist at {args.test_rev}.",
            )
            return _report(record, args.json_out)
    elif not (REPO / test_file).is_file():
        record.update(
            verdict=REFUSED,
            why=f"{test_file} does not exist in the working tree.",
        )
        return _report(record, args.json_out)

    rev, how, unusable = _resolve_baseline(args.baseline, args.integration)
    record["baseline"] = {"revision": rev, "how_chosen": how}
    if rev is None:
        record.update(verdict=REFUSED, why=unusable)
        return _report(record, args.json_out)

    changed, how_changed = _changed_source_files(
        rev, args.change, args.test_rev
    )
    baseline_precedes = bool(changed)
    record["change_under_test"] = {
        "source_files": changed,
        "how_determined": how_changed,
        "baseline_precedes_change": baseline_precedes,
    }

    tmp = Path(tempfile.mkdtemp(prefix="failbefore-"))
    tree = tmp / "tree"
    outcome_json = tmp / "outcome.json"
    current_test_path = tmp / "current_test.txt"
    try:
        materialize(rev, tree)
        record["test_tree_overlay"] = {
            "from": f"{TEST_TREE}/ at {args.test_rev}"
            if args.test_rev
            else f"{TEST_TREE}/ in the working tree",
            "files": _overlay_test_tree(tree, args.test_rev),
        }
        install_outcome_plugin(tree)

        env = child_env(tree, outcome_json, current_test_path)
        imports_ok, resolved = prove_tree_imports(tree, env)
        record["harness_proof"] = {
            "resolved": resolved,
            "baseline_source_imported": imports_ok,
        }
        if not imports_ok:
            record.update(
                verdict=REFUSED,
                why=(
                    "the baseline tree's own source is not what got imported, so any verdict would "
                    "describe the working checkout instead. Resolved module paths are recorded."
                ),
            )
            return _report(record, args.json_out)

        try:
            proc, observed = run_capturing_outcome(
                tree,
                [test_file],
                args.expr,
                env,
                outcome_json,
                args.timeout,
                per_test_timeout=args.per_test_timeout,
            )
        except subprocess.TimeoutExpired:
            record.update(
                verdict=REFUSED,
                why=f"pytest did not finish within {args.timeout}s.",
            )
            return _report(record, args.json_out)

        record["pytest"] = {
            "exit_code": proc.returncode,
            "tail": proc.stdout.strip().splitlines()[-15:],
        }
        if observed is None:
            timed_out_nodeid = (
                current_test_path.read_text(encoding="utf-8").strip()
                if args.per_test_timeout and current_test_path.is_file()
                else ""
            )
            if timed_out_nodeid:
                record.update(
                    verdict=INDETERMINATE,
                    why=(
                        f"{timed_out_nodeid} was running when the process ended without "
                        f"reporting an outcome, under --per-test-timeout "
                        f"({args.per_test_timeout}s). The marker names which test did not "
                        "finish, not why: a hang killed outright (the thread method "
                        "pytest-timeout uses where SIGALRM is unavailable) and a crash read "
                        "the same here, and either is a different fact from a slow pass or a "
                        "real failure."
                    ),
                )
                return _report(record, args.json_out)
            record.update(
                verdict=REFUSED,
                why=(
                    "pytest reported no outcome record, which happens on a usage error before the "
                    "session starts. Its output is recorded under pytest.tail."
                ),
            )
            record["pytest"]["stderr_tail"] = proc.stderr.strip().splitlines()[
                -15:
            ]
            return _report(record, args.json_out)

        # _failure_kind and _classify resolve crash_path against tree, so they run before the
        # materialized tree is removed below.
        record["observed"] = {
            "collected": observed["collected"],
            "counts": {
                o: sum(1 for t in observed["tests"] if t["outcome"] == o)
                for o in sorted({t["outcome"] for t in observed["tests"]})
            },
            "collect_errors": observed["collect_errors"],
            "failures": [
                {
                    "nodeid": t["nodeid"],
                    "headline": t.get("headline", ""),
                    "kind": _failure_kind(t, tree),
                    "crash_path": t.get("crash_path"),
                    "crash_lineno": t.get("crash_lineno"),
                }
                for t in observed["tests"]
                if t["outcome"] in ("failed", "error")
            ],
        }
        verdict, why = _classify(observed, baseline_precedes, tree)
        record.update(verdict=verdict, why=why)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return _report(record, args.json_out)


def _report(record: dict, json_out: str | None) -> int:
    verdict = record["verdict"]
    baseline = record.get("baseline", {})
    print()
    print(f"{verdict}: {record['why']}")
    if baseline.get("revision"):
        print(
            f"  baseline      {baseline['revision'][:12]} ({baseline['how_chosen']})"
        )
    change = record.get("change_under_test", {})
    if change:
        files = change["source_files"]
        shown = ", ".join(files[:4]) + (
            f" and {len(files) - 4} more" if len(files) > 4 else ""
        )
        print(
            f"  change        {len(files)} source file(s){': ' + shown if files else ''}"
        )
    observed = record.get("observed", {})
    if observed:
        print(
            f"  observed      {observed['collected']} selected, {observed['counts']}"
        )
    for failure in observed.get("failures", [])[:10]:
        print(f"  failing       {failure['nodeid']}  [{failure['kind']}]")
        print(f"                {failure['headline']}")
        if failure.get("crash_path"):
            print(
                f"                {failure['crash_path']}:{failure.get('crash_lineno')}"
            )
    if json_out:
        Path(json_out).write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        print(f"  record        {json_out}")
    return EXIT[verdict]


if __name__ == "__main__":
    sys.exit(main())
