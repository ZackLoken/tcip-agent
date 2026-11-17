"""Run a test against a baseline tree without the change under test; report whether it fails.

The baseline is materialized with ``git archive`` (the working tree is never touched), the current
test side is overlaid on it, and the test runner's per-test outcome is read rather than its exit
code. The test side is every file under ``tests/`` and every test file under
``packages/tcip-web/frontend/``, as the frontend's own rule (``scripts/testFiles.mjs``, asked
through ``node``) names one; ``node`` is started only when a path under the frontend is to be
judged.

A file under ``tests/`` runs under pytest, and the baseline's own Python source is proved to be
what gets imported. A test file under the frontend runs under the vitest of this checkout's
frontend install, the baseline tree's frontend linked to that install for the run, vitest's results
cache off and its configuration loaded without a bundled copy, so nothing is written through the
link, and ``-k`` passed as vitest's ``-t`` test name pattern; the proof is weaker, that every
test module vitest ran lies in the baseline tree, and it says nothing of where those modules'
imports resolve. The link is removed before the scratch directory is.

    python tools/prove_test_fails_before.py tests/test_foo.py
    python tools/prove_test_fails_before.py tests/test_foo.py -k "new_behavior"
    python tools/prove_test_fails_before.py tests/test_foo.py --baseline 48970001~1
    python tools/prove_test_fails_before.py tests/test_foo.py --json out.json
    python tools/prove_test_fails_before.py tests/test_foo.py --test-rev e2f9d837
        --baseline 90d974bd
    python tools/prove_test_fails_before.py tests/test_foo.py --per-test-timeout 30
    python tools/prove_test_fails_before.py packages/tcip-web/frontend/src/lib/x.test.ts -k "name"

Four verdicts, each its own exit code:

    GUARDS (0)         at least one selected test failed at the baseline in the call phase, on
                       the assertion it names (the raised exception's own class is
                       ``AssertionError`` or ``Failed``), on an exception raised from outside
                       the test side (the code under test raised), or on any other exception
                       raised in the test's own code except a ``TypeError``; the failing assertion
                       is printed and recorded, each failure not counted is named beside it. At a
                       baseline not shown to precede the change the verdict says so, as a fact
                       about the baseline rather than proof the change is guarded
    VACUOUS (1)        no selected test failed and not every one was skipped, at a baseline shown
                       to precede the change
    INDETERMINATE (2)  no selected test failed and not every one was skipped, but the baseline is
                       not shown to precede the change, so passing says nothing about the test; or
                       a selected test did not finish within its timeout (pytest-timeout's report,
                       vitest's own "Test timed out in" message, or the test running when the
                       process ended under ``--per-test-timeout``), named in the verdict, so a
                       hung baseline is told apart from a merely slow one
    REFUSED (3)        no verdict is available: the command line is not usable, the test file is
                       not on the test side or does not exist, ``--test-rev`` was given without
                       ``--baseline`` or does not resolve, the baseline does not resolve or no tree
                       here precedes the change, a git command failed, the working-tree snapshot
                       could not be made, nothing was selected, everything was skipped, the
                       baseline could not collect the file, the baseline's own source was not what
                       got imported (for vitest, a test module ran from outside the tree), the
                       runner did not finish within ``--timeout``, hit an internal or unhandled
                       error, never reported an outcome or wrote one that is not JSON, ``node`` is
                       not on the path where a frontend path is to be judged, the frontend's test
                       file rule failed or answered with something that is not JSON or not a list
                       of one boolean for each path it was asked about, the frontend has no
                       install, an installed vitest whose manifest cannot be read or decoded or
                       has no ``bin`` entry for vitest, the link to the install cannot be made, or
                       no failure is behavioral (each is ``unreached``, ``fixture`` or
                       ``test_code``), so no behavior was compared. A setup or teardown failure
                       that is neither a missing import or file nor a timeout is ``fixture``,
                       whatever raised it;
                       under vitest that is a failure following a hook that started and did not
                       end, or an error a suite or a module with tests carries of its own. A
                       call-phase failure is ``test_code``, raised in the test's own code, when it
                       is a ``TypeError`` whose crash frame is placed on the test side of the
                       materialized tree (a frame that cannot be placed there, a library's, is
                       behavioral), whatever its message. This is a policy: the class and the
                       place of the raise do not establish that a call failed to fit, so no
                       wording is read, and a ``TypeError`` from handling what the code under test
                       returned (a ``None`` subscripted) is refused the same way. Such a test
                       becomes a guard by asserting on what was returned before using it.

Not closed, each still counted as behavioral: an exception of another class raised in the test's
own code before the behavior is reached; a wrong call made in a helper off the test side or
inside package code (the second rightly, since the package's own behavior was reached); an
assertion that fails before the call under test.

An exception the tool does not name above is its own failure: its traceback goes to standard error
and the exit code is ``TOOL_FAILURE``, which no verdict uses.

The baseline is ``--baseline`` when given, else ``HEAD`` for uncommitted work, else the
merge-base against the integration branch; the baseline used, how it was chosen and the source
files the change touches are recorded with the verdict.

``--test-rev`` takes the test tree from a revision instead of the working tree, and requires
``--baseline``.

``--baseline-from-working-tree`` snapshots the working tree (tracked modifications and untracked
files, ignored files excluded) into a commit object and prints its hash, touching neither the
working tree, the index nor the stash list, and computes no verdict. It exits 0 when the snapshot
is made and REFUSED when it cannot be.

``--per-test-timeout`` passes ``--timeout`` through to pytest-timeout; a selected test that does
not finish within it is named and the verdict is INDETERMINATE. A vitest run takes each test's
own timeout.
"""

from __future__ import annotations

import argparse
import functools
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import traceback
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, NoReturn, NotRequired, TypedDict

REPO = Path(__file__).resolve().parent.parent
TEST_TREE = "tests"
FRONTEND = "packages/tcip-web/frontend"
TEST_FILE_RULE = f"{FRONTEND}/scripts/testFiles.mjs"
INSTALL = REPO / FRONTEND / "node_modules"
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
TOOL_FAILURE = max(EXIT.values()) + 1

FailureKind = Literal[
    "unreached", "timeout", "behavioral", "test_code", "fixture"
]


class ObservedTest(TypedDict):
    """One test's entry of the outcome record the pytest plugin or the vitest reporter writes.

    Each writes the failure keys only for a failed or errored test, and the tool adds ``kind`` to
    those.
    """

    nodeid: str
    phase: str
    outcome: str
    headline: NotRequired[str]
    crash_path: NotRequired[str | None]
    crash_lineno: NotRequired[int | None]
    exc_typename: NotRequired[str | None]
    kind: NotRequired[FailureKind]


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
    path = os.environ.get(CURRENT_TEST_ENV)
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
    with open(os.environ[OUTCOME_ENV], "w", encoding="utf-8") as fh:
        json.dump(_state, fh)
'''

REPORTER_SOURCE = r"""/** Record what vitest observed, in the pytest plugin's outcome record. */
import { writeFileSync } from "node:fs";
import { relative, sep } from "node:path";
import { fileURLToPath } from "node:url";

const FRAME = /^\s*at (?:.+ \()?(.+?):(\d+):\d+\)?$/;

/** The first frame of the error's stack, as its file and line; nulls when it names none. */
function crash(error) {
  const line = (error.stack ?? "").split("\n").find((text) => /^\s*at /.test(text));
  const frame = line?.match(FRAME);
  if (!frame) return [null, null];
  const file = frame[1].startsWith("file://") ? fileURLToPath(frame[1]) : frame[1];
  return [file, Number(frame[2])];
}

/** `entry` carrying the failure keys of `error`. */
function failure(entry, error) {
  const [path, line] = crash(error);
  return {
    ...entry,
    headline: `${error.name}: ${error.message.split("\n")[0]}`,
    crash_path: path,
    crash_lineno: line,
    exc_typename: error.name,
  };
}

export default class FailBeforeOutcome {
  state = { collected: 0, collect_errors: [], tests: [], internal_error: null, modules: [] };
  started = new Map();

  onInit(vitest) {
    this.root = vitest.config.root;
  }

  nodeid(module, name) {
    const path = relative(this.root, module.moduleId).split(sep).join("/");
    return name === undefined ? path : `${path}::${name}`;
  }

  onHookStart(hook) {
    this.started.set(hook.entity.id, [...(this.started.get(hook.entity.id) ?? []), hook.name]);
  }

  onHookEnd(hook) {
    const names = this.started.get(hook.entity.id) ?? [];
    this.started.set(hook.entity.id, names.filter((name) => name !== hook.name));
  }

  onTestCaseResult(test) {
    if (test.options.mode === "skip" || test.options.mode === "todo") return;
    this.state.collected += 1;
    const result = test.result();
    const entry = {
      nodeid: this.nodeid(test.module, test.fullName),
      phase: this.started.get(test.id)?.[0] ?? "call",
      outcome: result.state,
    };
    this.state.tests.push(result.state === "failed" ? failure(entry, result.errors[0]) : entry);
  }

  ownErrors(entity, nodeid) {
    const phase = this.started.get(entity.id)?.[0] ?? "suite";
    for (const error of entity.errors()) {
      this.state.tests.push(failure({ nodeid, phase, outcome: "error" }, error));
    }
  }

  onTestSuiteResult(suite) {
    this.ownErrors(suite, this.nodeid(suite.module, suite.fullName));
  }

  onTestModuleEnd(module) {
    this.state.modules.push(module.moduleId);
    if ([...module.children.allTests()].length > 0) {
      this.ownErrors(module, this.nodeid(module));
      return;
    }
    for (const error of module.errors()) {
      this.state.collect_errors.push({ nodeid: this.nodeid(module), detail: error.message });
    }
  }

  onTestRunEnd(modules, unhandledErrors) {
    if (unhandledErrors.length > 0) {
      this.state.internal_error = unhandledErrors.map((error) => error.message).join("\n");
    }
    writeFileSync(process.env[OUTCOME_ENV], JSON.stringify(this.state));
  }
}
"""


class RefusalError(Exception):
    """No verdict is available; the message says why and what to do."""


class _Parser(argparse.ArgumentParser):
    """An argument parser whose usage error raises :class:`RefusalError` with argparse's message."""

    def error(self, message: str) -> NoReturn:
        raise RefusalError(f"{message}; run with --help for the usage.")


def _test_file_relative_to_repo(raw: str) -> str:
    """Return ``raw`` normalized to a repo-relative, forward-slash path on the test side.

    Accepts a relative path as always, or an absolute one naming a file inside this repository;
    either way the return is a tree-relative path. A relative path is resolved against the
    repository root, never the caller's ``cwd``, and either spelling's ``..`` segments are
    collapsed before the test-side check, so a path that only reads as inside ``tests/`` before
    normalizing (``tests/../tools/x.py``) is refused rather than admitted. :class:`RefusalError`
    when ``raw`` resolves outside the repository or off the test side.
    """
    path = Path(raw)
    anchored = path if path.is_absolute() else REPO / path
    try:
        rel = anchored.resolve().relative_to(REPO.resolve())
    except ValueError as error:
        raise RefusalError(
            f"{raw} is outside this repository ({REPO})."
        ) from error
    rel_str = str(rel).replace("\\", "/")
    if not _is_test_side(rel_str):
        raise RefusalError(
            f"{raw} resolves to {rel_str}, outside {TEST_TREE}/ and no test file under "
            f"{FRONTEND}/."
        )
    return rel_str


def _git(*args: str) -> bytes:
    """Return git's standard output for ``args`` run in the repository, as bytes.

    :class:`RefusalError` naming the command and git's own error when git exits non-zero.
    """
    proc = subprocess.run(["git", *args], cwd=REPO, capture_output=True)
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace").strip()
        raise RefusalError(f"git {' '.join(args)} failed: {stderr}")
    return proc.stdout


def git_output(*args: str) -> str:
    """Return git's standard output for ``args`` as text; :class:`RefusalError` as :func:`_git`."""
    return _git(*args).decode()


def _commit(flag: str, named: str) -> str:
    """Return the commit ``named`` resolves to; :class:`RefusalError` naming ``flag`` when none."""
    try:
        return git_output(
            "rev-parse", "--verify", f"{named}^{{commit}}"
        ).strip()
    except RefusalError as refusal:
        raise RefusalError(
            f"{flag} {named} does not name a commit here: {refusal}"
        ) from refusal


def _node() -> str:
    """Return the ``node`` executable on the path; :class:`RefusalError` when there is none."""
    node = shutil.which("node")
    if node is None:
        raise RefusalError(
            f"no node on the path, which judging a path under {FRONTEND}/ and running vitest "
            "need. Put Node.js on the path."
        )
    return node


def _test_side(paths: Iterable[str]) -> set[str]:
    """Return those of the repo-relative, forward-slash ``paths`` that lie on the test side.

    The test side is every path under ``tests/`` and every path under the frontend that its own
    rule, ``TEST_FILE_RULE`` run under ``node``, calls a test file. The tool overlays the test side
    from the tree the test comes from, so a diff in it is not the change under test. ``node`` is
    started only when a path lies under the frontend; :class:`RefusalError` when the rule fails,
    its answer is not JSON, or its answer is not a list of one boolean for each frontend path.
    """
    wanted = set(paths)
    side = {p for p in wanted if p.startswith(f"{TEST_TREE}/")}
    frontend = sorted(p for p in wanted if p.startswith(f"{FRONTEND}/"))
    if not frontend:
        return side
    proc = subprocess.run(
        [_node(), str(REPO / TEST_FILE_RULE)],
        input=json.dumps([p.removeprefix(f"{FRONTEND}/") for p in frontend]),
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RefusalError(
            f"the frontend's test file rule {TEST_FILE_RULE} failed: {proc.stderr.strip()}"
        )
    try:
        answer = json.loads(proc.stdout)
    except json.JSONDecodeError as error:
        raise RefusalError(
            f"the frontend's test file rule {TEST_FILE_RULE} answered with no JSON: {error}"
        ) from error
    if (
        not isinstance(answer, list)
        or len(answer) != len(frontend)
        or not all(isinstance(is_test, bool) for is_test in answer)
    ):
        raise RefusalError(
            f"the frontend's test file rule {TEST_FILE_RULE} answered {json.dumps(answer)}, not "
            f"a list of one true or false for each path it was asked about ({len(frontend)})"
        )
    return side | {
        path for path, is_test in zip(frontend, answer, strict=True) if is_test
    }


@functools.cache
def _is_test_side(path: str) -> bool:
    """Return whether the repo-relative, forward-slash ``path`` lies on the test side."""
    return path in _test_side([path])


def _uncommitted_paths() -> list[str]:
    out = git_output("status", "--porcelain", "--untracked-files=all")
    paths = []
    for line in out.splitlines():
        if not line.strip():
            continue
        entry = line[3:].strip().strip('"')
        paths.append(entry.split(" -> ")[-1] if " -> " in entry else entry)
    return paths


def _resolve_baseline(named: str | None, integration: str) -> tuple[str, str]:
    """Return the baseline revision and how it was chosen.

    :class:`RefusalError` when ``named`` is no commit, when the merge-base against
    ``integration`` cannot be read, or when that merge-base is HEAD itself.
    """
    if named:
        return _commit("--baseline", named), f"named by the caller as {named}"

    uncommitted = _uncommitted_paths()
    if set(uncommitted) - _test_side(uncommitted):
        return (
            git_output("rev-parse", "HEAD").strip(),
            "HEAD, because the change under test is uncommitted",
        )

    try:
        head = git_output("rev-parse", "HEAD").strip()
        base = git_output("merge-base", "HEAD", integration).strip()
    except RefusalError as refusal:
        raise RefusalError(
            f"no merge-base of HEAD and --integration {integration} here: {refusal}"
        ) from refusal
    if base == head:
        raise RefusalError(
            "the working tree carries no uncommitted source change and HEAD is already "
            f"contained in {integration}, so no tree here precedes a change. Name the "
            "pre-change revision with --baseline."
        )
    return base, f"merge-base of HEAD and {integration}"


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
        how = f"git diff {baseline[:8]}..{test_rev[:8]}, test side excluded"
    else:
        tracked = git_output(
            "diff", "--name-only", baseline, "--"
        ).splitlines()
        untracked = git_output(
            "ls-files", "--others", "--exclude-standard"
        ).splitlines()
        how = f"git diff against {baseline[:8]} plus untracked files, test side excluded"
    differing = {p.strip() for p in [*tracked, *untracked] if p.strip()}
    changed = sorted(differing - _test_side(differing))
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
    list. :class:`RefusalError` when HEAD cannot be read or neither form makes a commit.
    """
    with tempfile.TemporaryDirectory(prefix="failbefore-index-") as scratch:
        head = git_output("rev-parse", "HEAD").strip()
        commit = _snapshot_via_temporary_index(head, Path(scratch) / "index")
        if commit:
            return commit, "temporary index (git add -A, git write-tree)"
    commit = _snapshot_via_stash_create()
    if commit:
        return commit, "git stash create (tracked changes only)"
    raise RefusalError(
        "could not snapshot the working tree: the temporary-index form failed here and git "
        "stash create produced nothing, most likely because there are no tracked changes to stash"
    )


def materialize(rev: str, dest: Path) -> None:
    """Extract a revision's whole tree into `dest`. The working tree is never touched."""
    dest.mkdir(parents=True, exist_ok=True)
    archive = _git("archive", rev)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r|") as tar:
        tar.extractall(dest, filter="data")


def _overlay_test_tree(dest: Path, test_rev: str | None) -> int:
    """Copy the whole test side, from ``test_rev`` or the working tree, into ``dest``.

    The working tree's test side is its tracked and untracked files git does not ignore. Returns
    the number of files copied.
    """
    if test_rev:
        archive = _git("archive", test_rev)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r") as tar:
            files = [m for m in tar.getmembers() if m.isfile()]
            side = _test_side(m.name for m in files)
            chosen = [m for m in files if m.name in side]
            tar.extractall(dest, members=chosen, filter="data")
        return len(chosen)

    listed = git_output(
        "ls-files", "--cached", "--others", "--exclude-standard"
    ).splitlines()
    present = [p for p in listed if (REPO / p).is_file()]
    side = _test_side(present)
    for path in present:
        if path in side:
            target = dest / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / path, target)
    return len(side)


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
    """Drop the outcome-recording plugin into a materialized tree, where pytest will import it.

    The environment names it reads are this tool's constants, assigned at the module's end.
    """
    names = f"OUTCOME_ENV = {OUTCOME_ENV!r}\nCURRENT_TEST_ENV = {CURRENT_TEST_ENV!r}\n"
    (tree / f"{PLUGIN_MODULE}.py").write_text(
        f"{PLUGIN_SOURCE}\n\n{names}", encoding="utf-8"
    )


def run_capturing_outcome(
    tree: Path,
    targets: list[str],
    expr: str,
    env: dict[str, str],
    outcome_json: Path,
    basetemp: Path,
    timeout: int,
    per_test_timeout: float | None = None,
) -> tuple[subprocess.CompletedProcess[str], dict | None]:
    """Run pytest in `tree` and return its process plus what it observed.

    The child's temporary root is `basetemp`, which the caller places outside `tree` so a test's
    `tmp_path` never lies under the baseline's own repository. The observation is None if it
    reported none.
    """
    # basetemp is this run's own folder: pytest deletes all but the last few roots under the
    # shared one, including another session's, at every start.
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
        str(basetemp),
    ]
    if expr:
        cmd += ["-k", expr]
    if per_test_timeout:
        cmd += ["--timeout", str(per_test_timeout)]
    proc = subprocess.run(
        cmd, cwd=tree, env=env, capture_output=True, text=True, timeout=timeout
    )
    return proc, _read_outcome(outcome_json)


def _read_outcome(outcome_json: Path) -> dict | None:
    """Return the outcome record at ``outcome_json`` and remove the file; None if there is none.

    :class:`RefusalError` when the record is not JSON.
    """
    if not outcome_json.is_file():
        return None
    try:
        observed = json.loads(outcome_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RefusalError(
            f"the runner's outcome record {outcome_json} is not JSON: {error}"
        ) from error
    outcome_json.unlink()
    return observed


def require_installed_vitest() -> None:
    """Return when ``INSTALL`` holds vitest.

    :class:`RefusalError` otherwise, naming the command that installs it.
    """
    if not (INSTALL / "vitest" / "package.json").is_file():
        raise RefusalError(
            f"{FRONTEND} has no installed vitest; run `npm ci` in {FRONTEND}."
        )


def link_install(tree: Path) -> Path:
    """Link ``tree``'s frontend ``node_modules`` to ``INSTALL``; return the link.

    :class:`RefusalError` when that install holds no vitest, naming the command that installs it, or
    when the link cannot be made, with the system's error.
    """
    require_installed_vitest()
    link = tree / INSTALL.relative_to(REPO)
    try:
        os.symlink(INSTALL, link, target_is_directory=True)
    except OSError as error:
        raise RefusalError(
            f"the baseline tree's link to {INSTALL} could not be made: {error}"
        ) from error
    return link


def _vitest_entry(manifest: Path) -> str:
    """Return the ``bin`` entry for vitest in the package manifest at ``manifest``.

    :class:`RefusalError` naming ``manifest`` when it cannot be read or decoded, or when it is not
    an object with a ``bin`` object holding a string for ``vitest``.
    """
    try:
        decoded = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RefusalError(
            f"the installed vitest's manifest {manifest} could not be read or decoded: {error}"
        ) from error
    bins = decoded.get("bin") if isinstance(decoded, dict) else None
    entry = bins.get("vitest") if isinstance(bins, dict) else None
    if not isinstance(entry, str):
        raise RefusalError(
            f"the installed vitest's manifest {manifest} holds no bin entry for vitest"
        )
    return entry


def run_vitest_capturing_outcome(
    tree: Path,
    test_file: str,
    expr: str,
    outcome_json: Path,
    timeout: int,
) -> tuple[subprocess.CompletedProcess[str], dict | None]:
    """Run vitest over ``test_file`` in ``tree``'s frontend and return its process and observation.

    vitest is the one ``INSTALL`` holds, run with its root at the tree's frontend, the outcome
    reporter written into ``tree`` with the ``OUTCOME_ENV`` it reads assigned at its end, and
    ``expr`` as its test name pattern. Its results cache is off and its configuration is loaded
    without a bundled copy, since each would be written under the tree's ``node_modules``, the
    link into the install. The observation is None if it reported none. :class:`RefusalError` when
    the installed vitest's manifest cannot be read or decoded or has no ``bin`` entry for vitest.
    """
    package = INSTALL / "vitest"
    entry = _vitest_entry(package / "package.json")
    reporter = tree / f"{PLUGIN_MODULE}.mjs"
    reporter.write_text(
        f"{REPORTER_SOURCE}\nconst OUTCOME_ENV = {json.dumps(OUTCOME_ENV)};\n",
        encoding="utf-8",
    )
    cmd = [
        _node(),
        str(package / entry),
        "run",
        test_file.removeprefix(f"{FRONTEND}/"),
        "--root",
        str(tree / FRONTEND),
        "--reporter",
        str(reporter),
        "--no-cache",
        "--configLoader",
        "runner",
    ]
    if expr:
        cmd += ["-t", expr]
    proc = subprocess.run(
        cmd,
        cwd=tree / FRONTEND,
        env={**os.environ, OUTCOME_ENV: str(outcome_json)},
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    return proc, _read_outcome(outcome_json)


_TIMEOUT_HEADLINE = re.compile(
    r"^(?:(?:[A-Za-z_][\w.]*: )?Timeout \(>[\d.]+s\) from pytest-timeout\."
    r"|Error: Test timed out in \d+ms\.)$"
)


def _is_timeout(headline: str) -> bool:
    """Return whether this failure is a test runner's own report of a test timing out.

    That report is pytest-timeout's signal-method ``PYTEST_FAILURE_MESSAGE`` or vitest's ``Test
    timed out in <n>ms.``; a hang pytest-timeout's thread method reports through the
    ``CURRENT_TEST_ENV`` marker is not one.

    pytest-timeout's leading exception class is optional because pytest writes the headline both
    ways: bare where the timeout is the whole report, and prefixed ``Failed: `` where
    pytest-timeout's signal method raises it as an ordinary ``Failed`` inside the test, which only
    hosts carrying ``SIGALRM`` do.
    """
    return bool(_TIMEOUT_HEADLINE.match(headline.strip()))


def _is_unreached(headline: str, crash_path: str | None, tree: Path) -> bool:
    """Return whether a failure never reached the code under test.

    A missing module or import always counts, vitest's ``Cannot find module`` among them. A missing
    file counts only when its crash frame sits off the test side
    (:func:`_crash_outside_test_tree`).
    """
    if headline.startswith(
        ("ModuleNotFoundError", "ImportError", "Error: Cannot find module")
    ):
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
    """Return whether the crash frame is placed positively on the test side of the tree.

    A frame there is the test's own body or a fixture. A missing or unplaceable path is not.
    """
    rel = _crash_relative_to_tree(crash_path, tree)
    return rel is not None and _is_test_side(rel)


def _crash_outside_test_tree(crash_path: str | None, tree: Path) -> bool:
    """Return whether the crash frame is placed in the tree and off its test side.

    One off it is the code under test raising. A missing or unplaceable path is never off it.
    """
    rel = _crash_relative_to_tree(crash_path, tree)
    return rel is not None and not _is_test_side(rel)


def _failure_kind(entry: ObservedTest, tree: Path) -> FailureKind:
    """Return the kind of one failed test.

    An errored test is classified the same way as a failed one.

    ``unreached``: the import never resolved, always; or a needed file was missing and the crash
    frame sits off the test side (:func:`_is_unreached`). A missing file whose crash frame sits
    inside the test's own body is left to the rest of this function instead.

    ``timeout``: the runner's own report of a test timing out (:func:`_is_timeout`), checked
    before the phase split below, so a timeout during setup is ``timeout`` too.

    After those two checks, a setup or teardown failure (``entry["phase"]`` is not ``"call"``, a
    pytest phase or a vitest hook's name) is ``fixture``, whatever raised it, wherever the crash
    frame sits. The rest applies only to a call-phase failure:

    ``behavioral``: the test's own ``assert`` or ``pytest.fail``/``pytest.raises`` failed (the
    raised exception's own class is ``AssertionError`` or ``Failed``, read from
    ``exc_typename``), or the crash frame sits off the test side, meaning the code under test
    raised. ``test_code``: a ``TypeError`` whose crash frame is placed on the test side
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
        return "test_code"
    return "behavioral"


def _not_counted(entry: ObservedTest) -> str:
    """Name a failure that is not counted, by its ``kind``.

    The kind is ``unreached``, ``test_code`` or ``fixture``. The sentence names the test, what was
    observed and, for a fixture, the phase it failed in.
    """
    where = entry["nodeid"]
    headline = entry.get("headline", "")
    if entry["kind"] == "unreached":
        return f"{where}: a missing import or file ({headline})"
    if entry["kind"] == "test_code":
        raised = entry["exc_typename"]
        return f"{where}: the test's own code raised a {raised} before its assertion ({headline})"
    return f"{where}: a fixture failed during {entry['phase']} ({headline})"


def _record_failure_kinds(observed: dict, tree: Path) -> None:
    """Write :func:`_failure_kind` under ``kind`` onto each failed or errored test's entry."""
    for t in _failures(observed):
        t["kind"] = _failure_kind(t, tree)


def _failures(observed: dict) -> list[ObservedTest]:
    """Return the entries of ``observed["tests"]`` whose outcome is ``failed`` or ``error``."""
    tests: list[ObservedTest] = observed["tests"]
    return [t for t in tests if t["outcome"] in ("failed", "error")]


def _classify(observed: dict, baseline_precedes: bool) -> tuple[str, str]:
    """Return the verdict and its sentence for ``observed``."""
    failed = _failures(observed)
    if observed.get("internal_error"):
        return (
            REFUSED,
            "the runner hit an internal or unhandled error, so nothing it reported can be "
            "trusted.",
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
            f"all {len(outcomes)} selected tests were skipped, so nothing was evaluated."
        )
    unreached = [t for t in failed if t["kind"] == "unreached"]
    timed_out = [t for t in failed if t["kind"] == "timeout"]
    not_compared_kinds: tuple[FailureKind, ...] = ("fixture", "test_code")
    not_compared = [t for t in failed if t["kind"] in not_compared_kinds]
    behavioral = [t for t in failed if t["kind"] == "behavioral"]
    if timed_out:
        names = ", ".join(t["nodeid"] for t in timed_out)
        return INDETERMINATE, (
            f"{len(timed_out)} of {observed['collected']} selected test(s) did not finish within "
            f"their timeout ({names}), so passing or failing says nothing here: a hung "
            "baseline is a different fact from a slow pass or a real failure."
        )
    if failed and not behavioral and not_compared:
        cause = (
            " A call that does not fit the baseline's code is one cause of this."
            if any(t["kind"] == "test_code" for t in not_compared)
            else ""
        )
        also = (
            " Also not counted: "
            + "; ".join(_not_counted(t) for t in unreached)
            + "."
            if unreached
            else ""
        )
        return REFUSED, (
            f"{len(not_compared)} of {len(failed)} failing test(s) failed before reaching a "
            "comparison of behavior, so no behavior was compared and the run is not evidence "
            "either way: "
            + "; ".join(_not_counted(t) for t in not_compared)
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
            *(_not_counted(t) for t in not_compared),
            *(_not_counted(t) for t in unreached),
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
            "no selected test failed and not every one was skipped, at a baseline shown to "
            "precede the change. It guards nothing; assert on what only the change produces."
        )
    return INDETERMINATE, (
        "no selected test failed and not every one was skipped, but no source file differs "
        "between the baseline and the tree this test comes from, so the baseline is not shown to "
        "precede the change and passing there says nothing about the test. Name the pre-change "
        "revision with --baseline."
    )


def main() -> int:
    """Run the command line, print what it found, and return the exit code.

    A verdict returns its code from ``EXIT``, a snapshot that is made returns 0, and any exception
    other than a :class:`RefusalError` prints its traceback to standard error and returns
    ``TOOL_FAILURE``, which no verdict uses.
    """
    try:
        return _run_command_line()
    except Exception:
        traceback.print_exc()
        return TOOL_FAILURE


def _run_command_line() -> int:
    """Run the command line's test at its baseline, or its snapshot; return the exit code.

    A :class:`RefusalError` raised anywhere, by a usage error or by the run, is reported as the
    verdict REFUSED with its message as the reason; the record is written to ``--json`` unless the
    refusal is a usage error.
    """
    ap = _Parser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "test_file",
        help="path, repo-relative, e.g. tests/test_foo.py or a test file under "
        f"{FRONTEND}/",
    )
    ap.add_argument(
        "-k",
        dest="expr",
        default="",
        help="pytest -k expression, or vitest's -t test name pattern for a frontend test",
    )
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
        help="seconds to allow the test runner (default 900)",
    )
    ap.add_argument(
        "--per-test-timeout",
        type=float,
        default=None,
        help="seconds pytest-timeout allows one test (passed through as --timeout); "
        "a test that does not finish within it scores INDETERMINATE by name, "
        "told apart from a hang that says nothing and a real pass or failure",
    )
    record: dict = {}
    json_out: str | None = None
    try:
        args = ap.parse_args()
        if args.baseline_from_working_tree and args.baseline:
            ap.error(
                "--baseline-from-working-tree and --baseline name the baseline two ways at once"
            )
        json_out = args.json_out
        if args.baseline_from_working_tree:
            commit, method = snapshot_working_tree()
            expr_flag = f"-k {args.expr!r} " if args.expr else ""
            print(f"snapshot {commit} ({method})")
            print(
                "computes no verdict by itself. This is the working tree as it stands right now: "
                "a snapshot that already holds the fix is a baseline the change does not precede, "
                "and a passing run against it is the same INDETERMINATE case as any other "
                "baseline already containing the change, never evidence the test guards "
                "anything. Apply the fix now, then run:\n"
                f"  python tools/prove_test_fails_before.py {args.test_file} {expr_flag}"
                f"--baseline {commit}"
            )
            return 0
        record.update(
            test_file=args.test_file,
            k=args.expr,
            verdict=None,
            test_tree_from=args.test_rev or "the working tree",
        )
        _prove(args, record)
    except RefusalError as refusal:
        record.update(verdict=REFUSED, why=str(refusal))
    return _report(record, json_out)


def _prove(args: argparse.Namespace, record: dict) -> None:
    """Run the test ``args`` names at its baseline; write its findings and verdict to ``record``.

    :class:`RefusalError` when no verdict is available, raised once the scratch directory is
    removed.
    """
    test_file = _test_file_relative_to_repo(args.test_file)
    record["test_file"] = test_file

    if args.test_rev and not args.baseline:
        raise RefusalError(
            "--test-rev needs an explicit --baseline. A change spans one commit per concern, so "
            "the commit before a test can be inside it, and guessing one would decide the verdict."
        )

    test_rev = _commit("--test-rev", args.test_rev) if args.test_rev else None
    if test_rev:
        listed = git_output(
            "ls-tree", "--name-only", "-r", test_rev, "--", test_file
        ).strip()
        if not listed:
            raise RefusalError(
                f"{test_file} does not exist at {args.test_rev}."
            )
    elif not (REPO / test_file).is_file():
        raise RefusalError(f"{test_file} does not exist in the working tree.")

    rev, how = _resolve_baseline(args.baseline, args.integration)
    record["baseline"] = {"revision": rev, "how_chosen": how}

    changed, how_changed = _changed_source_files(rev, args.change, test_rev)
    baseline_precedes = bool(changed)
    record["change_under_test"] = {
        "source_files": changed,
        "how_determined": how_changed,
        "baseline_precedes_change": baseline_precedes,
    }

    runner = "vitest" if test_file.startswith(f"{FRONTEND}/") else "pytest"
    scratch = tempfile.TemporaryDirectory(prefix="failbefore-", delete=False)
    tmp = Path(scratch.name)
    tree = tmp / "tree"
    outcome_json = tmp / "outcome.json"
    current_test_path = tmp / "current_test.txt"
    basetemp = tmp / "pytest-basetemp"
    link: Path | None = None
    try:
        materialize(rev, tree)
        record["test_tree_overlay"] = {
            "from": f"the test side at {args.test_rev}"
            if args.test_rev
            else "the test side in the working tree",
            "files": _overlay_test_tree(tree, test_rev),
        }
        if runner == "vitest":
            link = link_install(tree)
            run = functools.partial(
                run_vitest_capturing_outcome,
                tree,
                test_file,
                args.expr,
                outcome_json,
                args.timeout,
            )
        else:
            install_outcome_plugin(tree)
            env = child_env(tree, outcome_json, current_test_path)
            imports_ok, resolved = prove_tree_imports(tree, env)
            record["harness_proof"] = {
                "resolved": resolved,
                "baseline_source_imported": imports_ok,
            }
            if not imports_ok:
                raise RefusalError(
                    "the baseline tree's own source is not what got imported, so any verdict "
                    "would describe the working checkout instead. Resolved module paths are "
                    "recorded."
                )
            run = functools.partial(
                run_capturing_outcome,
                tree,
                [test_file],
                args.expr,
                env,
                outcome_json,
                basetemp,
                args.timeout,
                per_test_timeout=args.per_test_timeout,
            )

        try:
            proc, observed = run()
        except subprocess.TimeoutExpired as expired:
            raise RefusalError(
                f"{runner} did not finish within {args.timeout}s."
            ) from expired

        record[runner] = {
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
                return
            record[runner]["stderr_tail"] = proc.stderr.strip().splitlines()[
                -15:
            ]
            raise RefusalError(
                f"{runner} reported no outcome record, which happens on a usage error before "
                f"the run starts. Its output is recorded under {runner}.tail."
            )

        if runner == "vitest":
            modules = observed["modules"]
            inside = all(
                Path(m).resolve().is_relative_to(tree.resolve())
                for m in modules
            )
            record["harness_proof"] = {
                "test_modules": modules,
                "test_modules_in_baseline_tree": inside,
            }
            if not inside:
                raise RefusalError(
                    "a test module vitest ran lies outside the baseline tree, so any verdict "
                    "would describe another checkout. The modules are recorded."
                )

        # _failure_kind resolves crash_path against tree, so it runs before the materialized tree
        # is removed below.
        _record_failure_kinds(observed, tree)
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
                    "kind": t["kind"],
                    "crash_path": t.get("crash_path"),
                    "crash_lineno": t.get("crash_lineno"),
                }
                for t in _failures(observed)
            ],
        }
        verdict, why = _classify(observed, baseline_precedes)
        record.update(verdict=verdict, why=why)
    finally:
        _remove_scratch_directory(scratch, link)


def _remove_scratch_directory(
    scratch: tempfile.TemporaryDirectory[str], link: Path | None
) -> None:
    """Remove ``link``, the one :func:`link_install` made if any, then the directory `scratch`.

    The directory goes with everything under it; read-only paths are made writable and retried by
    the library's own removal. When the link or anything else cannot be removed, the directory is
    left in place, so no removal reaches through a link still standing, and one line on standard
    error names the directory and the error, which names the path. Nothing is raised, so the
    verdict already reached stands.
    """
    try:
        if link is not None:
            link.unlink()
        scratch.cleanup()
    except OSError as error:
        print(
            f"scratch directory left in place: {scratch.name}; not removed: {error}",
            file=sys.stderr,
        )


def _report(record: dict, json_out: str | None) -> int:
    verdict = record["verdict"]
    baseline = record.get("baseline", {})
    if json_out:
        Path(json_out).write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
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
        print(f"  record        {json_out}")
    return EXIT[verdict]


if __name__ == "__main__":
    sys.exit(main())
