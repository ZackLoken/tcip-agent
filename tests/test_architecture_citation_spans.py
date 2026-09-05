"""check_architecture_citations resolves backtick spans with no length cap, reporting an
oversized one as its own finding, and anchors a fragment across a wrapped, indented line."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_architecture_citations.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_architecture_citations", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_architecture_citations"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_source(tmp_path, relpath, lines):
    path = tmp_path / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_an_oversized_span_is_reported_and_does_not_desync_a_later_citation(tmp_path):
    _write_source(tmp_path, "scripts/example.py", ["def something():", "    pass"])
    long_literal = "x" * 320
    doc = tmp_path / "ARCHITECTURE.md"
    doc.write_text(
        f"A big literal `{long_literal}` sits here.\n\n"
        "- `def something(` (`scripts/example.py:1`)\n",
        encoding="utf-8",
    )

    checker = _load()
    findings, _unanchored = checker.check(doc, tmp_path)

    oversized = [f for f in findings if f["status"] == "oversized_span"]
    verified = [f for f in findings if f["status"] == "verified"]
    assert len(oversized) == 1
    assert oversized[0]["length"] == 320
    assert len(verified) == 1
    assert verified[0]["key"] == "def something("


def test_a_comma_ended_line_anchors_a_citation_indented_on_the_next_line(tmp_path):
    _write_source(tmp_path, "scripts/example.py", ["def create_something():", "    pass"])
    doc = tmp_path / "ARCHITECTURE.md"
    doc.write_text(
        "Written by `create_something`,\n"
        "  `scripts/example.py:1`.\n",
        encoding="utf-8",
    )

    checker = _load()
    findings, unanchored = checker.check(doc, tmp_path)

    assert unanchored == 0
    assert len(findings) == 1
    assert findings[0]["status"] == "verified"
    assert findings[0]["key"] == "create_something"


def test_a_wide_gap_with_real_prose_between_stays_unanchored(tmp_path):
    """The widened gap admits whitespace and connector punctuation only, so a fragment and a
    citation separated by an unrelated sentence still count as unanchored."""
    _write_source(tmp_path, "scripts/example.py", ["def create_something():", "    pass"])
    doc = tmp_path / "ARCHITECTURE.md"
    doc.write_text(
        "Written by `create_something`. A different sentence follows.\n"
        "`scripts/example.py:1` is unrelated to it.\n",
        encoding="utf-8",
    )

    checker = _load()
    findings, unanchored = checker.check(doc, tmp_path)

    assert unanchored == 1
    assert findings == []
