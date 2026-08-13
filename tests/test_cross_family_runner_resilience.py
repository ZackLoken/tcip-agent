"""One harness returning nothing must not destroy the other families' answers.

The cross-family runner poses one question to several agent harnesses concurrently and writes each
one's transcript. A harness that produces no stdout used to raise inside the worker, and the
exception escaped the result-collection loop, so the sibling families' completed runs were lost with
it. That is the expensive failure: the runs that succeeded are the artifact, and a harness that
returns nothing is a normal outcome rather than a programming error.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "cross_family_ask.py"


def _load():
    """Import the runner by path: it is a script, not an installed module."""
    spec = importlib.util.spec_from_file_location("cross_family_ask", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cross_family_ask"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def runner():
    return _load()


def _stub_run(stdout, stderr=""):
    """A subprocess.run replacement returning the given streams verbatim."""
    def _run(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=["stub"], returncode=0,
                                           stdout=stdout, stderr=stderr)
    return _run


@pytest.mark.parametrize("stdout", [None, b"bytes not str"])
def test_a_harness_returning_no_usable_stdout_still_records_its_run(runner, tmp_path, monkeypatch,
                                                                   stdout):
    """A None or bytes stdout is written as text rather than raising TypeError."""
    monkeypatch.setattr(runner.subprocess, "run", _stub_run(stdout))
    monkeypatch.setattr(runner, "harness_version", lambda *a, **k: "stub-version")
    monkeypatch.setitem(runner.BUILDERS, "codex",
                        lambda *a, **k: (["stub", "argv"], None))
    (tmp_path / "q.txt").write_text("question", encoding="utf-8")

    meta = runner.run_one("codex", "qid", "as-shipped", "question",
                          tmp_path, tmp_path / "out", 5, None, None)

    run_dir = tmp_path / "out" / "qid" / "as-shipped" / "codex"
    assert (run_dir / "stdout.txt").is_file()
    assert (run_dir / "meta.json").is_file()
    assert meta["family"] == "codex"


def test_one_familys_failure_leaves_the_others_results_intact(runner, tmp_path, monkeypatch):
    """The collection loop records a failed run and keeps every sibling's result."""
    def _explode(*_args, **_kwargs):
        raise RuntimeError("harness blew up")

    real_run_one = runner.run_one

    def _dispatch(family, *args, **kwargs):
        if family == "codex":
            _explode()
        return real_run_one(family, *args, **kwargs)

    monkeypatch.setattr(runner.subprocess, "run", _stub_run("an answer"))
    monkeypatch.setattr(runner, "harness_version", lambda *a, **k: "stub-version")
    for fam in ("codex", "antigravity"):
        monkeypatch.setitem(runner.BUILDERS, fam, lambda *a, **k: (["stub", "argv"], None))
    monkeypatch.setattr(runner, "run_one", _dispatch)
    (tmp_path / "q.txt").write_text("question", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [
        "cross_family_ask.py", "--question-id", "qid",
        "--prompt-file", str(tmp_path / "q.txt"),
        "--families", "codex,antigravity",
        "--cwd", str(tmp_path), "--out", str(tmp_path / "out"), "--timeout", "5",
    ])

    # Non-zero, because a family that failed is a failed invocation and must not read as clean.
    assert runner.main() != 0

    surviving = tmp_path / "out" / "qid" / "as-shipped" / "antigravity" / "response.md"
    assert surviving.is_file(), "the family that answered lost its transcript to the one that failed"
    assert surviving.read_text(encoding="utf-8").strip() == "an answer"

    summary = json.loads(
        (tmp_path / "out" / "qid" / "as-shipped" / "summary.json").read_text(encoding="utf-8"))
    rows = {r["family"]: r for r in (summary if isinstance(summary, list) else summary["runs"])}
    assert rows["antigravity"]["exit_code"] == 0
    assert rows["codex"]["response_source"] == "runner_error"
    assert "harness blew up" in rows["codex"]["runner_error"]
