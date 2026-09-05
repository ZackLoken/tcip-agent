"""Verify ARCHITECTURE.md's file:line citations against the code they quote, for CI.

ARCHITECTURE.md cites the tree constantly: a seam side, a format's writer, a public-surface
row. Almost every such citation pairs a location with a fragment of what stands there, and it
is that pairing this check holds to the tree, so a refactor that moves code cannot leave the
document quietly pointing at the wrong line.

A citation is checked when a backticked `path:line` (or `path:a,b` / `path:a-b`) sits next to a
backticked fragment: either the fragment opens a parenthetical directly after the citation, the
seam ledger's form, or the fragment ends within four characters before it, the form the format
sections and the surface tables use (a comma-ended wrapped line plus a two-space indent is the
longest such gap seen so far). A citation with no such neighbour quotes nothing to check, so it
is counted and reported as unanchored rather than assumed correct; citations into ``tests/`` and
``docs/`` are the bulk of those, pointing at evidence rather than quoting a line.

A citation may name only a file's basename rather than its full repo-relative path; this
resolves so long as exactly one tracked file carries that basename. When the tree carries the
basename twice, the short form fails as ambiguous (`verdicts.py` under both `tcip_annotation` and
`pipelines/feedback` is the standing example): write the fuller path to resolve it.

The check is quote-tolerant, because a moved line is not a wrong statement:

- verified: the fragment is on a line the citation names.
- re-anchorable: the fragment is gone from those lines but still present elsewhere in the file,
  so only the number is stale. ``--fix`` rewrites the number in place.
- failed: the fragment is nowhere in the file, or the path resolves to no file or to several.
  Only this needs a human, because the statement itself, not its line, has gone stale.
- oversized span: a backtick-delimited span past ``SPAN_LENGTH_CAP`` characters, reported as its
  own finding since one usually means an inline literal that should be split rather than quoted
  whole.

Exit 0 when nothing failed, 1 otherwise (an oversized span counts as a failure). Re-anchorable
citations do not fail the run on their own; they are reported, and ``--strict`` promotes them to
failures for a pass that means to leave the document exactly anchored. Defaults resolve
ARCHITECTURE.md and the repo root relative to this script, so CI invokes it as
`python scripts/check_architecture_citations.py`.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: Directories that hold no citable source: build output, caches, vendored packages.
SKIP_DIRS = frozenset(
    {".git", "node_modules", "__pycache__", ".venv", "static", "dist",
     ".mypy_cache", ".ruff_cache", ".pytest_cache"}
)

#: Code spans in document order, consecutive backticks always pairing so no repetition cap; a
#: capped one once let a long span find no closing backtick and desynchronize every pair after.
SPAN_RE = re.compile(r"`([^`]*)`", re.DOTALL)
SPAN_LENGTH_CAP = 300
CITATION_RE = re.compile(
    r"^(?P<path>[\w./@-]+\.(?:tsx|ts|py|json|ya?ml|cmd|css))"
    r":(?P<nums>\d+(?:\s*[,-]\s*\d+)*)$"
)
#: Between a fragment and the citation it anchors: a comma, a table pipe, a newline, plus up to
#: two more characters, so a comma-ended line still anchors a citation two spaces into the next.
LEFT_GAP_RE = re.compile(r"^[\s,;(|]{0,4}$")
OPENS_PARENTHETICAL_RE = re.compile(r"^\s*\(\s*$")
DEF_RE = re.compile(r"^(?:async\s+)?(?:def|class)\s+(\w+)")
ASSIGN_RE = re.compile(r"^([\w.]+)\s*[:=][^=]")
DOTTED_RE = re.compile(r"^[\w.]+\(?$")


def index_repo(repo_root: Path) -> dict[str, Path]:
    """Map every citable repo-relative path to its file, for suffix resolution.

    Resolution runs over the tracked tree (git ls-files), never a filesystem walk of the
    checkout: local-only trees such as design records and installed dependencies carry
    shadow copies of source files that would make unique basenames ambiguous. The walk
    remains only as a fallback for a tree that is not a git checkout.
    """
    try:
        listed = subprocess.run(
            ["git", "ls-files"], cwd=repo_root, capture_output=True, text=True, check=True
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        listed = []
    if listed:
        return {rel: repo_root / rel for rel in listed if (repo_root / rel).is_file()}
    index = {}
    for path in repo_root.rglob("*"):
        if path.is_file() and not any(part in SKIP_DIRS for part in path.parts):
            index[path.relative_to(repo_root).as_posix()] = path
    return index


def resolve_path(cited: str, index: dict[str, Path]) -> tuple[Path | None, str]:
    """Resolve a cited path, which may be written in full or as a trailing fragment."""
    if cited in index:
        return index[cited], ""
    matches = [p for rel, p in index.items() if rel.endswith("/" + cited)]
    if len(matches) == 1:
        return matches[0], ""
    if not matches:
        return None, "names no file in the tree"
    return None, f"names {len(matches)} files in the tree, so it is ambiguous"


def line_numbers(nums: str) -> list[int]:
    """Expand a citation's line part: a number, a comma list, or an inclusive range."""
    if "-" in nums:
        lo, hi = (int(n) for n in nums.split("-", 1))
        return list(range(lo, hi + 1))
    return [int(n) for n in nums.split(",")]


def candidates(fragment: str) -> list[str]:
    """Progressively looser keys to look for, most faithful to the quote first.

    Every key stays specific enough to mean something: a bare tail like ``get`` off a method
    call would match half a module, so a dotted tail is taken only when the whole fragment is a
    dotted name, and an assignment yields its target rather than the call on its right.
    """
    out = [fragment]
    if "\n" in fragment:
        out.append(" ".join(fragment.split()))
        out.extend(part.strip() for part in fragment.splitlines() if part.strip())
        fragment = fragment.splitlines()[0].strip()
        out.append(fragment)
    stripped = fragment.rstrip("(,: ")
    if stripped != fragment:
        out.append(stripped)
    m = DEF_RE.match(fragment)
    if m:
        out.append(f"def {m.group(1)}(")
        out.append(f"class {m.group(1)}")
        out.append(m.group(1))
    m = ASSIGN_RE.match(fragment)
    if m:
        out.append(m.group(1))
    if DOTTED_RE.match(fragment):
        head = fragment.rstrip("(")
        out.append(head)
        if "." in head:
            out.append(head.rsplit(".", 1)[-1])
    seen, uniq = set(), []
    for c in out:
        c = c.strip()
        if len(c) >= 4 and c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def defines(source_line: str, key: str) -> bool:
    """Whether this line is where ``key`` is defined, rather than one of its use sites.

    A symbol's citation belongs at its definition, so re-anchoring prefers one over a call.
    """
    if not key.replace(".", "").replace("_", "").isalnum():
        return False
    escaped = re.escape(key)
    return bool(
        re.match(rf"\s*(?:async\s+)?(?:def|class)\s+{escaped}\b", source_line)
        or re.match(rf"\s*{escaped}\s*(?::[^=]*)?=[^=]", source_line)
    )


def find_anchor(spans: list[tuple[int, int, str]], i: int, text: str) -> str | None:
    """The fragment a citation quotes: a parenthetical after it, else the token before it."""
    start, end, _ = spans[i]
    if i + 1 < len(spans):
        nxt = spans[i + 1]
        between = text[end:nxt[0]]
        if OPENS_PARENTHETICAL_RE.match(between) and not CITATION_RE.match(nxt[2]):
            return nxt[2]
    if i > 0:
        prev = spans[i - 1]
        between = text[prev[1]:start]
        if LEFT_GAP_RE.match(between) and not CITATION_RE.match(prev[2]):
            return prev[2]
    return None


def check(doc_path: Path, repo_root: Path) -> tuple[list[dict], int]:
    """Classify every citation in the document; return the findings and the unanchored count."""
    text = doc_path.read_text(encoding="utf-8")
    line_starts = [0] + [m.end() for m in re.finditer(r"\n", text)]

    def doc_line_of(offset: int) -> int:
        lo, hi = 0, len(line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_starts[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    index = index_repo(repo_root)
    spans = [(m.start(), m.end(), m.group(1)) for m in SPAN_RE.finditer(text)]
    findings: list[dict] = []
    unanchored = 0
    file_lines: dict[Path, list[str]] = {}

    for start, _end, span_text in spans:
        if len(span_text) > SPAN_LENGTH_CAP:
            findings.append({
                "status": "oversized_span", "doc_line": doc_line_of(start), "length": len(span_text),
            })

    for i, (start, _end, span_text) in enumerate(spans):
        m = CITATION_RE.match(span_text)
        if not m:
            continue
        fragment = find_anchor(spans, i, text)
        if fragment is None:
            unanchored += 1
            continue
        cited_path, why = resolve_path(m.group("path"), index)
        doc_line = doc_line_of(start)
        if cited_path is None:
            findings.append({"status": "failed", "doc_line": doc_line, "cite": span_text,
                             "fragment": fragment, "detail": why})
            continue
        if cited_path not in file_lines:
            file_lines[cited_path] = cited_path.read_text(
                encoding="utf-8", errors="replace").splitlines()
        source = file_lines[cited_path]
        cited_lines = [n for n in line_numbers(m.group("nums"))]
        rel = cited_path.relative_to(repo_root).as_posix()
        hit = None
        for key in candidates(fragment):
            for n in cited_lines:
                if 1 <= n <= len(source) and key in source[n - 1]:
                    hit = (key, n)
                    break
            if hit:
                break
        if hit:
            findings.append({"status": "verified", "doc_line": doc_line, "cite": span_text,
                             "fragment": fragment, "path": rel, "key": hit[0], "line": hit[1]})
            continue
        elsewhere = None
        for key in candidates(fragment):
            found = [n for n, s in enumerate(source, 1) if key in s]
            if not found:
                continue
            defined = [n for n in found if defines(source[n - 1], key)]
            pool = defined or found
            elsewhere = (key, min(pool, key=lambda n: (abs(n - cited_lines[0]), n)), len(found))
            break
        if elsewhere:
            findings.append({"status": "re-anchorable", "doc_line": doc_line, "cite": span_text,
                             "fragment": fragment, "path": rel, "key": elsewhere[0],
                             "line": elsewhere[1], "hits": elsewhere[2],
                             "was": m.group("nums"), "cited_path": m.group("path")})
        else:
            findings.append({"status": "failed", "doc_line": doc_line, "cite": span_text,
                             "fragment": fragment, "path": rel,
                             "detail": f"nothing matching {fragment!r} is anywhere in the file"})
    return findings, unanchored


def apply_fix(doc_path: Path, findings: list[dict]) -> int:
    """Rewrite each re-anchorable citation's line number to where its fragment now stands."""
    lines = doc_path.read_text(encoding="utf-8").splitlines(keepends=True)
    fixed = 0
    for f in findings:
        if f["status"] != "re-anchorable":
            continue
        i = f["doc_line"] - 1
        old = f"`{f['cited_path']}:{f['was']}`"
        new = f"`{f['cited_path']}:{f['line']}`"
        if old in lines[i]:
            lines[i] = lines[i].replace(old, new, 1)
            fixed += 1
    doc_path.write_text("".join(lines), encoding="utf-8", newline="")
    return fixed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("architecture_md", nargs="?", default=str(_REPO_ROOT / "ARCHITECTURE.md"))
    ap.add_argument("repo_root", nargs="?", default=str(_REPO_ROOT))
    ap.add_argument("--fix", action="store_true",
                    help="rewrite every re-anchorable citation's line number in place")
    ap.add_argument("--strict", action="store_true",
                    help="treat a re-anchorable citation as a failure too")
    args = ap.parse_args()

    doc_path = Path(args.architecture_md)
    repo_root = Path(args.repo_root)
    findings, unanchored = check(doc_path, repo_root)

    oversized = [f for f in findings if f["status"] == "oversized_span"]
    verified = [f for f in findings if f["status"] == "verified"]
    moved = [f for f in findings if f["status"] == "re-anchorable"]
    failed = [f for f in findings if f["status"] == "failed"]

    print(f"checked {len(verified) + len(moved) + len(failed)} anchored citations "
          f"({unanchored} carry no quoted fragment and are not checked)")
    for f in oversized:
        print(f"OVERSIZED SPAN ARCHITECTURE.md:{f['doc_line']}  {f['length']} characters "
              f"(over the {SPAN_LENGTH_CAP}-character cap; split it)")
    for f in moved:
        print(f"RE-ANCHORABLE ARCHITECTURE.md:{f['doc_line']}  {f['cite']}")
        print(f"  {f['fragment']!r} is not on the cited line; found at "
              f"{f['path']}:{f['line']} ({f['hits']} line(s) match {f['key']!r})")
    for f in failed:
        print(f"FAILED ARCHITECTURE.md:{f['doc_line']}  {f['cite']}")
        print(f"  {f['detail']}")

    if args.fix and moved:
        print(f"\n--fix rewrote {apply_fix(doc_path, findings)} citation(s)")

    print(f"verified {len(verified)}, re-anchorable {len(moved)}, failed {len(failed)}, "
          f"oversized spans {len(oversized)}")
    total = len(failed) + len(oversized) + (len(moved) if args.strict else 0)
    print(f"{'FAIL' if total else 'PASS'}: {total} problem(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
