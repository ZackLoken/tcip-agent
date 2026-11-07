"""Check the line cites of a work order's premise half before the premise checker reads it.

    python tools/check_premise_cites.py <issue> [--checkout <label>=<path>]...

The issue's body and comments are read with ``gh issue view``. The premise half opens at the line
``Premises, read at <commit>:``, whose header names one or more commits, each optionally preceded
by a repository label, and continues as the contiguous bullet list that follows it. A premise's
repository is its ``[<label>]`` prefix, or the only one the header names. The unlabeled
repository is the checkout the script runs from; a labeled one is the path its ``--checkout``
names, and a label with none is a failing premise.

Each cite in the writer's form, ``at <path>:<a>`` or ``at <path>:<a>-<b>``, is read with
``git show <commit>:<path>``. It fails when the path is absent at the commit or the range is past
the file. A premise that backticks identifiers fails when none of them appears in the union of
its cited ranges. A premise with no cite in the writer's form fails. Prints PASS, or each failing
premise and FAIL, and exits 1 on FAIL or when no premise half can be read.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

HEADER = re.compile(
    r"^Premises, read at\s+(?P<refs>[^:\n]*?)(?::|\.(?:\s|$))",
    re.IGNORECASE | re.MULTILINE,
)
REPOSITORY_COMMIT = re.compile(
    r"(?:\b(?!and\b)(?P<label>[A-Za-z][\w-]*)\s+)?\b(?P<commit>[0-9a-f]{7,40})\b"
)
PREMISE_LABEL = re.compile(r"^[-*]\s+\[(?P<label>[^\]]+)\]")
CITE = re.compile(
    r"\bat\s+`?(?P<path>[\w./-]+):(?P<first>\d+)(?:-(?P<last>\d+))?"
)
BACKTICKED = re.compile(r"`([^`]+)`")
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")


class PremiseHalfError(ValueError):
    """The issue text holds no premise half that can be read."""


class CiteError(ValueError):
    """A cite cannot be resolved to the text it names."""


@dataclass(frozen=True)
class Cite:
    """One ``at <path>:<first>[-<last>]`` cite; ``last`` equals ``first`` for a single line."""

    path: str
    first: int
    last: int


@dataclass(frozen=True)
class PremiseHalf:
    """The commits the premises were read at, by repository label, and the premise lines.

    A commit whose repository the header does not label is keyed None.
    """

    commits: dict[str | None, str]
    premises: list[str]


def fetch_issue(issue: int) -> str:
    """Return the issue's body and every comment's body, joined, as ``gh`` serves them."""
    proc = subprocess.run(
        ["gh", "issue", "view", str(issue), "--json", "body,comments"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    issue_json = json.loads(proc.stdout)
    return "\n\n".join(
        [issue_json["body"], *(c["body"] for c in issue_json["comments"])]
    )


def parse_premise_half(text: str) -> PremiseHalf:
    """Return the commits of the first ``Premises, read at`` header in ``text`` and its bullet list.

    Raises PremiseHalfError when there is no header, no commit, two commits for one repository, or
    no premise line.
    """
    header = HEADER.search(text)
    if header is None:
        raise PremiseHalfError("no 'Premises, read at <commit>:' header")
    commits: dict[str | None, str] = {}
    for match in REPOSITORY_COMMIT.finditer(header["refs"]):
        if match["label"] in commits:
            raise PremiseHalfError(
                f"the header names two commits for repository {match['label']!r}"
            )
        commits[match["label"]] = match["commit"]
    if not commits:
        raise PremiseHalfError("the header names no commit")
    premises = []
    for line in text[header.end() :].splitlines()[1:]:
        if line.startswith(("- ", "* ")):
            premises.append(line)
        elif line.strip():
            break
    if not premises:
        raise PremiseHalfError("no premise line follows the header")
    return PremiseHalf(commits, premises)


def parse_cites(line: str) -> list[Cite]:
    """Return each ``at <path>:<a>[-<b>]`` cite in ``line``.

    A bare ``:12`` or a basename without ``at`` is not one.
    """
    return [
        Cite(m["path"], int(m["first"]), int(m["last"] or m["first"]))
        for m in CITE.finditer(line)
    ]


def backticked_identifiers(line: str) -> list[str]:
    """Return the backticked tokens of ``line`` that are one identifier, dots allowed."""
    return [t for t in BACKTICKED.findall(line) if IDENTIFIER.fullmatch(t)]


def cited_text(checkout: Path, commit: str, cite: Cite) -> str:
    """Return the lines ``cite`` names in its file at ``commit``.

    Raises CiteError when ``git show`` cannot read the path at the commit or the range is not
    inside the file.
    """
    proc = subprocess.run(
        ["git", "-C", str(checkout), "show", f"{commit}:{cite.path}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode:
        raise CiteError(
            f"{cite.path} unreadable at {commit[:12]}: {proc.stderr.strip()}"
        )
    lines = proc.stdout.splitlines()
    if not 1 <= cite.first <= cite.last <= len(lines):
        raise CiteError(
            f"{cite.path}:{cite.first}-{cite.last} is outside the file ({len(lines)} lines)"
        )
    return "\n".join(lines[cite.first - 1 : cite.last])


def repository_of(
    line: str, half: PremiseHalf, checkouts: dict[str, Path], home: Path
) -> tuple[str, Path]:
    """Return the commit and checkout the premise ``line`` is read against.

    Raises CiteError when the line's label is not in the header, an unlabeled line meets a header
    naming several labeled repositories, or a labeled repository has no checkout.
    """
    prefix = PREMISE_LABEL.match(line)
    label = prefix["label"] if prefix else None
    if label is None and len(half.commits) == 1:
        label = next(iter(half.commits))
    if label not in half.commits:
        raise CiteError(f"repository {label!r} is not one the header names")
    if label is not None and label in checkouts:
        return half.commits[label], checkouts[label]
    if label is None or len(half.commits) == 1:
        return half.commits[label], home
    raise CiteError(f"no --checkout for repository {label!r}")


def check_premise(
    line: str, half: PremiseHalf, checkouts: dict[str, Path], home: Path
) -> list[str]:
    """Return one message per way the premise ``line`` fails; empty when its cites hold."""
    cites = parse_cites(line)
    if not cites:
        return [f"not in the writer's form: {line}"]
    try:
        commit, checkout = repository_of(line, half, checkouts, home)
        texts = [cited_text(checkout, commit, cite) for cite in cites]
    except CiteError as err:
        return [f"{err}: {line}"]
    names = backticked_identifiers(line)
    ranges = ", ".join(f"{c.path}:{c.first}-{c.last}" for c in cites)
    if names and not any(
        re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(n)}(?![A-Za-z0-9_])",
            "\n".join(texts),
        )
        for n in names
    ):
        return [f"none of {', '.join(names)} appears in {ranges}: {line}"]
    return []


def _checkout_argument(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not (label and separator and path):
        raise argparse.ArgumentTypeError(f"{value!r} is not <label>=<path>")
    return label, Path(path)


def main(
    argv: Sequence[str] | None = None,
    *,
    fetch: Callable[[int], str] = fetch_issue,
) -> int:
    """Print PASS, or each failing premise and FAIL; return 0 on PASS and 1 otherwise."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("issue", type=int)
    parser.add_argument(
        "--checkout",
        action="append",
        default=[],
        type=_checkout_argument,
        metavar="LABEL=PATH",
    )
    args = parser.parse_args(argv)
    checkouts = dict(args.checkout)
    try:
        half = parse_premise_half(fetch(args.issue))
    except PremiseHalfError as err:
        print(f"{err}\nFAIL")
        return 1
    failures = [
        message
        for line in half.premises
        for message in check_premise(line, half, checkouts, Path.cwd())
    ]
    print("\n".join([*failures, "FAIL" if failures else "PASS"]))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
