"""Prove a test actually fails against the code it was written to catch.

A test added alongside a fix is only a guard if it fails *before* the fix. Reasoning that it would
is not evidence: a test can pass everywhere because an unrelated path produces the same outcome,
and a vacuous test passes in the normal gate too.

This materializes a baseline tree with ``git archive`` (the working tree is never touched), overlays
the current ``tests/`` directory so the test's own conftest and helpers travel with it, proves the
baseline's own source is what gets imported, runs pytest there, and reads pytest's per-test outcome
rather than its exit code.

    python tools/prove_test_fails_before.py tests/test_foo.py
    python tools/prove_test_fails_before.py tests/test_foo.py -k "new_behaviour"
    python tools/prove_test_fails_before.py tests/test_foo.py --baseline 196eedf1~1
    python tools/prove_test_fails_before.py tests/test_foo.py --json out.json
    python tools/prove_test_fails_before.py tests/test_foo.py --test-rev 8b09bd17 --baseline ae3dbbb8

Four verdicts, four exit codes, because an exit code alone cannot carry this:

    GUARDS (0)         at least one selected test failed at the baseline in the call phase, on
                       the assertion it names (the raised exception's own class is
                       ``AssertionError`` or ``Failed``), on an exception raised from outside
                       ``tests/`` (the code under test raised), or on any other exception a
                       wrongly-constructed call raises except the call-signature-mismatch shape a
                       fixture-shaped failure takes; the failing assertion is printed and recorded
    VACUOUS (1)        every selected test passed at a baseline shown to precede the change
    INDETERMINATE (2)  every selected test passed, but the baseline is not shown to precede the
                       change, so passing says nothing about the test
    REFUSED (3)        no verdict is available: nothing was selected, everything was skipped, the
                       baseline could not collect the file, the baseline's own source was not what
                       got imported, pytest never reported an outcome, or every failure is
                       fixture-shaped so the code under test was never reached. A setup or
                       teardown failure is always fixture-shaped, whatever raised it; a call-phase
                       failure is fixture-shaped only when it is a ``TypeError`` raised inside
                       ``tests/`` whose message is CPython's own wording for a call-signature
                       mismatch (an unexpected keyword, a missing required argument, too many
                       positional arguments), the shape a fixture or test calling something with
                       an argument the baseline lacks takes. A call-phase
                       failure that constructs its inputs wrongly in any other way, whatever the
                       exception class, is behavioral: the code under test was reached

A baseline is only usable if the change under test is absent from it. With uncommitted work that is
``HEAD``. In a history where one commit carries one file, the commit before the test file is inside
the change rather than before it, so the previous commit is not a baseline: pass the pre-change
revision with ``--baseline``, or let the tool take the merge-base against the integration branch.
Which baseline was used, how it was chosen, and which source files the change touches are all
recorded, so a later reader can check the verdict rather than trust it.

``--test-rev`` takes the test tree from a revision instead of the working tree, which is how a guard
claim already in the history gets checked. It requires an explicit baseline, since the revision
before a test commit is the tree a one-file-per-commit history makes untrustworthy.
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

GUARDS, VACUOUS, INDETERMINATE, REFUSED = "GUARDS", "VACUOUS", "INDETERMINATE", "REFUSED"
EXIT = {GUARDS: 0, VACUOUS: 1, INDETERMINATE: 2, REFUSED: 3}

PLUGIN_SOURCE = '''\
"""Record what pytest observed, so a caller reads outcomes instead of an exit code."""
import json
import os

_state = {"collected": None, "collect_errors": [], "tests": [], "internal_error": None}
_exc_types = {}  # (nodeid, when) -> the raised exception's own class name


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
    """``raw`` normalized to a repo-relative, forward-slash path under ``tests/``.

    Accepts a relative path as always, or an absolute one naming a file inside this repository;
    either way the return is a tree-relative path, so pytest collects the target inside whichever
    tree gets materialized rather than an absolute path that names the working checkout's file
    regardless of ``cwd``. A relative path is resolved against the repository root, never the
    caller's ``cwd``, and either spelling's ``..`` segments are collapsed before the ``tests/``
    check, so a path that only reads as inside ``tests/`` before normalizing (``tests/../tools/
    x.py``) is refused rather than admitted. Returns ``(None, reason)`` when ``raw`` resolves
    outside ``tests/``.
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
    proc = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def _is_test_side(path: str) -> bool:
    """Paths the tool overlays from the current tree, so a diff in them is not the change under test."""
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


def _resolve_baseline(named: str | None, integration: str) -> tuple[str | None, str, str]:
    """Return (revision, how it was chosen, why it is unusable if it is)."""
    if named:
        try:
            sha = git_output("rev-parse", "--verify", f"{named}^{{commit}}").strip()
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
        return None, f"merge-base against {integration}", (
            f"the working tree carries no uncommitted source change and HEAD is already contained "
            f"in {integration}, so no tree here precedes a change. Name the pre-change revision "
            f"with --baseline."
        )
    return base, f"merge-base of HEAD and {integration}", ""


def _changed_source_files(baseline: str, declared: list[str], test_rev: str | None) -> tuple[list[str], str]:
    """The source-side files that differ between the baseline and the tree the test comes from.

    A caller-declared path still has to differ. Trusting the declaration would let the caller decide
    the verdict, which is the thing this tool exists to stop.
    """
    if test_rev:
        tracked = git_output("diff", "--name-only", baseline, test_rev, "--").splitlines()
        untracked: list[str] = []
        how = f"git diff {baseline[:8]}..{test_rev[:8]}, test tree excluded"
    else:
        tracked = git_output("diff", "--name-only", baseline, "--").splitlines()
        untracked = git_output("ls-files", "--others", "--exclude-standard").splitlines()
        how = f"git diff against {baseline[:8]} plus untracked files, test tree excluded"
    changed = sorted({p.strip() for p in [*tracked, *untracked] if p.strip() and not _is_test_side(p.strip())})
    if declared:
        wanted = {p.replace("\\", "/") for p in declared}
        kept = sorted(wanted & set(changed))
        ignored = sorted(wanted - set(changed))
        note = f"declared with --change and confirmed to differ ({how})"
        if ignored:
            note += f"; declared but identical at the baseline, so not counted: {', '.join(ignored)}"
        return kept, note
    return changed, how


def materialize(rev: str, dest: Path) -> None:
    """Extract a revision's whole tree into `dest`. The working tree is never touched."""
    dest.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "archive", rev], cwd=REPO, check=True, stdout=subprocess.PIPE
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r|") as tar:
        tar.extractall(dest, filter="data")


def _overlay_test_tree(dest: Path, test_rev: str | None) -> int:
    """Bring one whole test tree across, so conftest and helper modules match the test.

    The point of overlaying the tree rather than the single file is that a test's support files are
    part of the test: a helper added alongside it is missing from the baseline, and a baseline that
    cannot import the test yields a collection error rather than a verdict.
    """
    if test_rev:
        archive = subprocess.run(
            ["git", "archive", test_rev, "--", TEST_TREE], cwd=REPO, check=True,
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


def child_env(tree: Path, outcome_json: Path) -> dict[str, str]:
    srcs = [str(p) for p in sorted(tree.glob("packages/*/src"))]
    sep = ";" if sys.platform == "win32" else ":"
    env = {
        **os.environ,
        "PYTHONPATH": sep.join([*srcs, str(tree)]),
        "PYTHONDONTWRITEBYTECODE": "1",
        OUTCOME_ENV: str(outcome_json),
    }
    env.pop("TCIP_MIN_TESTS", None)
    return env


def prove_tree_imports(tree: Path, env: dict[str, str]) -> tuple[bool, dict[str, str]]:
    """Confirm each package resolves inside the baseline tree, not the editable-installed checkout.

    The three packages are installed editable pointing at the working checkout, so a run that
    resolves them there measures the current source while reporting a baseline verdict.
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
        [sys.executable, "-c", probe], cwd=tree, env=env, capture_output=True, text=True
    )
    try:
        resolved = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return False, {"probe": f"probe produced no JSON: {proc.stdout!r} {proc.stderr[-500:]!r}"}
    root = str(tree.resolve()).lower()
    ok = all(str(Path(p).resolve()).lower().startswith(root) for p in resolved.values() if p and "IMPORT FAILED" not in p)
    ok = ok and not any("IMPORT FAILED" in p for p in resolved.values())
    return ok, resolved


def install_outcome_plugin(tree: Path) -> None:
    """Drop the outcome-recording plugin into a materialized tree, on the path pytest will import."""
    (tree / f"{PLUGIN_MODULE}.py").write_text(PLUGIN_SOURCE, encoding="utf-8")


def run_capturing_outcome(tree: Path, targets: list[str], expr: str, env: dict[str, str],
                          outcome_json: Path, timeout: int):
    """Run pytest in `tree` and return its process plus what it observed, or None if it reported none.

    The observed record, not the exit code, is what a caller judges on: an exit code cannot tell a
    real failure apart from an empty selection, a collection error or a usage error.
    """
    cmd = [
        sys.executable, "-m", "pytest", *targets,
        "-q", "-p", "no:cacheprovider", "-p", PLUGIN_MODULE, "--rootdir", str(tree),
    ]
    if expr:
        cmd += ["-k", expr]
    proc = subprocess.run(cmd, cwd=tree, env=env, capture_output=True, text=True, timeout=timeout)
    observed = None
    if outcome_json.is_file():
        observed = json.loads(outcome_json.read_text(encoding="utf-8"))
        outcome_json.unlink()
    return proc, observed


def _is_unreached(headline: str) -> bool:
    """A failure that never reached the code under test carries the same weight as a collection error."""
    return headline.startswith(("ModuleNotFoundError", "ImportError"))


def _crash_outside_test_tree(crash_path: str | None, tree: Path) -> bool:
    """Whether the failure's crash frame sits outside ``tests/`` inside the materialized tree.

    A frame under ``tests/`` is the test's own body or a fixture; one outside it is the code
    under test raising. An unresolvable or missing path (a report with no exception repr) is
    never treated as outside, since that would count silence as evidence.
    """
    if not crash_path:
        return False
    try:
        rel = Path(crash_path).resolve().relative_to(tree.resolve())
    except (OSError, ValueError):
        return False
    return not str(rel).replace("\\", "/").startswith(f"{TEST_TREE}/")


_CALL_SIGNATURE_MISMATCH_PATTERNS = (
    re.compile(r"got an unexpected keyword argument"),
    re.compile(r"missing \d+ required (?:positional|keyword-only) arguments?"),
    re.compile(r"takes \d+ positional arguments? but \d+ (?:was|were) given"),
)


def _is_call_signature_mismatch(headline: str) -> bool:
    """Whether ``headline`` is CPython's own wording for a call supplying the wrong arguments to
    a callable's signature: an unexpected keyword, a missing required argument (positional or
    keyword-only), or too many positional arguments. This is the shape a fixture or test calling
    a constructor with an argument the baseline lacks takes, never wording the code under test
    composes on its own.
    """
    return any(p.search(headline) for p in _CALL_SIGNATURE_MISMATCH_PATTERNS)


def _failure_kind(entry: dict, tree: Path) -> str:
    """One of ``unreached``, ``behavioral``, ``fixture``, for one failed or errored test.

    ``unreached``: the import never resolved, the same weight as a collection error.

    A setup or teardown failure (``entry["phase"]`` is not ``"call"``) is always ``fixture``: the
    test body never ran, whatever raised it, wherever the crash frame sits. The rest applies only
    to a call-phase failure:

    ``behavioral``: the test's own ``assert`` or ``pytest.fail``/``pytest.raises`` failed (the
    raised exception's own class is ``AssertionError`` or ``Failed``, read from ``exc_typename``
    rather than the headline text: pytest's assertion rewriting omits the class name from the
    headline for a bare comparison, so text alone cannot tell an assertion apart from another
    exception formatted the same way), or the crash frame sits outside ``tests/``, meaning the
    code under test raised. ``fixture``: a ``TypeError`` raised from inside ``tests/`` whose
    message is a call-signature mismatch (:func:`_is_call_signature_mismatch`), the shape a
    fixture constructor called with an argument the baseline lacks takes; the code under test was
    never reached. Everything else in the call phase, whatever exception a wrongly-constructed
    call raises, is ``behavioral``: the code under test was reached and its result inspected,
    whatever went wrong from there.
    """
    headline = entry.get("headline", "")
    if _is_unreached(headline):
        return "unreached"
    if entry.get("phase") != "call":
        return "fixture"
    if entry.get("exc_typename") in ("AssertionError", "Failed"):
        return "behavioral"
    if _crash_outside_test_tree(entry.get("crash_path"), tree):
        return "behavioral"
    if entry.get("exc_typename") == "TypeError" and _is_call_signature_mismatch(headline):
        return "fixture"
    return "behavioral"


def _classify(observed: dict, baseline_precedes: bool, tree: Path) -> tuple[str, str]:
    if observed.get("internal_error"):
        return REFUSED, "pytest hit an internal error, so nothing it reported can be trusted."
    if observed["collect_errors"]:
        first = observed["collect_errors"][0]["nodeid"]
        return REFUSED, (
            f"the baseline could not collect {first}. Zero assertions were evaluated, so this is "
            "not evidence either way. The test most likely references something only the change "
            "introduces, which means the baseline predates more than the change under test."
        )
    if not observed["collected"]:
        return REFUSED, "nothing was selected, so nothing was evaluated. Check the -k expression."

    outcomes = [t["outcome"] for t in observed["tests"]]
    if outcomes and all(o == "skipped" for o in outcomes):
        return REFUSED, (
            f"all {len(outcomes)} selected tests were skipped, so nothing was evaluated. A "
            "module-level importorskip at the baseline will do this."
        )
    failed = [t for t in observed["tests"] if t["outcome"] in ("failed", "error")]
    kinds = [_failure_kind(t, tree) for t in failed]
    unreached = [t for t, k in zip(failed, kinds) if k == "unreached"]
    fixture = [t for t, k in zip(failed, kinds) if k == "fixture"]
    behavioral = [t for t, k in zip(failed, kinds) if k == "behavioral"]
    if failed and not behavioral and fixture:
        discount = (f" {len(unreached)} further failure(s) rest on a missing import."
                    if unreached else "")
        return REFUSED, (
            f"{len(fixture)} of {len(failed)} failing test(s) are fixture-shaped: {len(fixture)} "
            f"failed on an error other than the assertion the test names "
            f"({fixture[0].get('headline', '')}), so the code under test was never reached and "
            f"no behavior was compared. This is a run to redo, not evidence either way.{discount}"
        )
    if failed and not behavioral:
        return REFUSED, (
            f"all {len(failed)} failing tests failed on a missing import rather than on an "
            f"assertion ({unreached[0]['headline']}), so the code under test was never reached and "
            "no behavior was compared. A baseline whose source will not import in the current "
            "environment cannot produce evidence."
        )
    if behavioral:
        discount_parts = []
        if unreached:
            discount_parts.append(f"{len(unreached)} further failure(s) rest on a missing import")
        if fixture:
            discount_parts.append(f"{len(fixture)} further failure(s) are fixture-shaped")
        discount = f" {' and '.join(discount_parts)}, not evidence either way." if discount_parts else ""
        caveat = "" if baseline_precedes else (
            " No source file differs between the baseline and the tree this test comes from, so this "
            "is a fact about the baseline rather than proof that a change here is guarded."
        )
        return GUARDS, (
            f"{len(behavioral)} of {observed['collected']} selected tests failed on the behavior "
            f"under test at the baseline.{discount}{caveat} The failures are recorded below."
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
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("test_file", help="path, repo-relative, e.g. tests/test_foo.py")
    ap.add_argument("-k", dest="expr", default="", help="pytest -k expression")
    ap.add_argument("--baseline", default=None,
                    help="pre-change revision; default is HEAD for uncommitted work, else the "
                         "merge-base against the integration branch")
    ap.add_argument("--integration", default="main",
                    help="branch to take a merge-base against (default main)")
    ap.add_argument("--test-rev", default=None,
                    help="take the test tree from this revision instead of the working tree, to "
                         "check a guard claim already in the history; requires --baseline")
    ap.add_argument("--change", action="append", default=[],
                    help="a source file the change touches; repeatable. Overrides the computed set")
    ap.add_argument("--json", dest="json_out", default=None, help="write the full record here")
    ap.add_argument("--timeout", type=int, default=900, help="seconds to allow pytest (default 900)")
    args = ap.parse_args()

    record: dict = {"test_file": args.test_file, "k": args.expr, "verdict": None,
                    "test_tree_from": args.test_rev or "the working tree"}

    test_file, refusal = _test_file_relative_to_repo(args.test_file)
    if test_file is None:
        record.update(verdict=REFUSED, why=refusal)
        return _report(record, args.json_out)
    record["test_file"] = test_file

    if args.test_rev and not args.baseline:
        record.update(verdict=REFUSED, why=(
            "--test-rev needs an explicit --baseline. The commit before a test is inside the change "
            "in a one-file-per-commit history, so guessing one would decide the verdict."
        ))
        return _report(record, args.json_out)

    if args.test_rev:
        listed = git_output("ls-tree", "--name-only", "-r", args.test_rev, "--", test_file).strip()
        if not listed:
            record.update(verdict=REFUSED, why=f"{test_file} does not exist at {args.test_rev}.")
            return _report(record, args.json_out)
    elif not (REPO / test_file).is_file():
        record.update(verdict=REFUSED, why=f"{test_file} does not exist in the working tree.")
        return _report(record, args.json_out)

    rev, how, unusable = _resolve_baseline(args.baseline, args.integration)
    record["baseline"] = {"revision": rev, "how_chosen": how}
    if rev is None:
        record.update(verdict=REFUSED, why=unusable)
        return _report(record, args.json_out)

    changed, how_changed = _changed_source_files(rev, args.change, args.test_rev)
    baseline_precedes = bool(changed)
    record["change_under_test"] = {"source_files": changed, "how_determined": how_changed,
                                   "baseline_precedes_change": baseline_precedes}

    tmp = Path(tempfile.mkdtemp(prefix="failbefore-"))
    tree = tmp / "tree"
    outcome_json = tmp / "outcome.json"
    try:
        materialize(rev, tree)
        record["test_tree_overlay"] = {
            "from": f"{TEST_TREE}/ at {args.test_rev}" if args.test_rev
            else f"{TEST_TREE}/ in the working tree",
            "files": _overlay_test_tree(tree, args.test_rev),
        }
        install_outcome_plugin(tree)

        env = child_env(tree, outcome_json)
        imports_ok, resolved = prove_tree_imports(tree, env)
        record["harness_proof"] = {"resolved": resolved, "baseline_source_imported": imports_ok}
        if not imports_ok:
            record.update(verdict=REFUSED, why=(
                "the baseline tree's own source is not what got imported, so any verdict would "
                "describe the working checkout instead. Resolved module paths are recorded."
            ))
            return _report(record, args.json_out)

        try:
            proc, observed = run_capturing_outcome(
                tree, [test_file], args.expr, env, outcome_json, args.timeout)
        except subprocess.TimeoutExpired:
            record.update(verdict=REFUSED, why=f"pytest did not finish within {args.timeout}s.")
            return _report(record, args.json_out)

        record["pytest"] = {"exit_code": proc.returncode, "tail": proc.stdout.strip().splitlines()[-15:]}
        if observed is None:
            record.update(verdict=REFUSED, why=(
                "pytest reported no outcome record, which happens on a usage error before the "
                "session starts. Its output is recorded under pytest.tail."
            ))
            record["pytest"]["stderr_tail"] = proc.stderr.strip().splitlines()[-15:]
            return _report(record, args.json_out)

        # _failure_kind and _classify resolve crash_path against tree, so they run before the
        # materialized tree is removed below.
        record["observed"] = {
            "collected": observed["collected"],
            "counts": {o: sum(1 for t in observed["tests"] if t["outcome"] == o)
                       for o in sorted({t["outcome"] for t in observed["tests"]})},
            "collect_errors": observed["collect_errors"],
            "failures": [{"nodeid": t["nodeid"], "headline": t.get("headline", ""),
                          "kind": _failure_kind(t, tree), "crash_path": t.get("crash_path"),
                          "crash_lineno": t.get("crash_lineno")}
                         for t in observed["tests"] if t["outcome"] in ("failed", "error")],
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
        print(f"  baseline      {baseline['revision'][:12]} ({baseline['how_chosen']})")
    change = record.get("change_under_test", {})
    if change:
        files = change["source_files"]
        shown = ", ".join(files[:4]) + (f" and {len(files) - 4} more" if len(files) > 4 else "")
        print(f"  change        {len(files)} source file(s){': ' + shown if files else ''}")
    observed = record.get("observed", {})
    if observed:
        print(f"  observed      {observed['collected']} selected, {observed['counts']}")
    for failure in observed.get("failures", [])[:10]:
        print(f"  failing       {failure['nodeid']}  [{failure['kind']}]")
        print(f"                {failure['headline']}")
        if failure.get("crash_path"):
            print(f"                {failure['crash_path']}:{failure.get('crash_lineno')}")
    if json_out:
        Path(json_out).write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"  record        {json_out}")
    return EXIT[verdict]


if __name__ == "__main__":
    sys.exit(main())
