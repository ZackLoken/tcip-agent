"""Check the line cites of every premise block of a work order before the premise checker reads it.

    python tools/check_premise_cites.py <issue> [--checkout <label>=<path>]...

The issue's body and comments are read with ``gh issue view``. A premise block (a premise half, a
supplement, a half read again) opens at a line ``Premises, read at <commit>:``, whose header names
one or more commits, each optionally preceded by a repository label, and ends at its colon or at a
period followed by whitespace. The block continues as the contiguous bullet list that begins on
the line after the header's own line, an indented line continuing the bullet before it. Each block
is checked against the commits its own header names. A premise's repository is its ``[<label>]``
prefix, or the only one the header names. The unlabeled repository is the checkout the script runs
from; a labeled one is the path its ``--checkout`` names, and a label with none is a failing
premise.

Each cite in the writer's form, ``at <path>:<a>`` or ``at <path>:<a>-<b>``, is read with
``git show <commit>:<path>``. It fails when the path is absent at the commit or the range is past
the file. Lines are counted on newline bytes alone, as git counts them. A premise whose backticked
spans hold identifier tokens fails when none of them appears in the union of its cited ranges. A
premise with no cite in the writer's form fails. Prints PASS with the number of blocks checked and
the rule it applied, or each failing premise and FAIL, and exits 1 on FAIL or when any premise
block cannot be read. A failing premise and a block's refusal are each printed after the block's
name: its place among the issue's headers, their count and the header's words.
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
    r"^Premises, read at\s+(?P<refs>[^:\n]*?)(?::|\.(?=\s|$))",
    re.IGNORECASE | re.MULTILINE,
)
REPOSITORY_COMMIT = re.compile(
    r"(?:\b(?!and\b)(?P<label>[A-Za-z][\w-]*)\s+)?\b(?P<commit>[0-9a-f]{7,40})\b"
)
PREMISE_LABEL = re.compile(r"^[-*]\s+\[(?P<label>[^\]]+)\]")
CITE = re.compile(
    r"\bat\s+`?(?P<path>[\w./-]+):(?P<first>\d+)(?:-(?P<last>\d+))?"
)
PASS_LINE = (
    "PASS: {blocks} checked; each premise has one or more cites in the writer's form and every "
    "one resolves at its commit, and a premise that backticks any name has one or more of them "
    "on its cited lines."
)
BACKTICKED = re.compile(r"`([^`]+)`")
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")


class PremiseBlockError(ValueError):
    """A premise block in the issue text cannot be read, or the text holds none."""


class CiteError(ValueError):
    """A cite cannot be resolved to the text it names."""


@dataclass(frozen=True)
class Cite:
    """One ``at <path>:<first>[-<last>]`` cite; ``last`` equals ``first`` for a single line."""

    path: str
    first: int
    last: int


@dataclass(frozen=True)
class PremiseBlock:
    """One premise block: its commits by repository label, its premise lines and its header's words.

    A commit whose repository the header does not label is keyed None.
    """

    commits: dict[str | None, str]
    premises: list[str]
    header_words: str


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


def _header_words(header: re.Match[str]) -> str:
    """Return the header's text from ``Premises, read at`` through its ``refs`` group, on one line.

    The ``refs`` group is whatever stands between ``read at`` and the terminating colon or period:
    the commits and any words around them, or nothing that is a commit. Each run of whitespace, a
    line break included, is one space; the terminator and any prose after it are left out.
    """
    return " ".join(header.string[header.start() : header.end("refs")].split())


def block_name(place: int, count: int, header_words: str) -> str:
    """Return the prefix that names the block at ``place`` of ``count`` headers, from one."""
    return f'block {place} of {count}, "{header_words}": '


def _premise_block(text: str, header: re.Match[str]) -> PremiseBlock:
    """Return the block that ``header``, a match of HEADER in ``text``, opens.

    Raises PremiseBlockError when the header names no commit or two commits for one repository, or
    no premise line follows it.
    """
    commits: dict[str | None, str] = {}
    for match in REPOSITORY_COMMIT.finditer(header["refs"]):
        if match["label"] in commits:
            raise PremiseBlockError(
                f"the header names two commits for repository {match['label']!r}"
            )
        commits[match["label"]] = match["commit"]
    if not commits:
        raise PremiseBlockError("the header names no commit")
    premises: list[str] = []
    for raw in text[header.end() :].split("\n")[1:]:
        line = raw.rstrip("\r")
        if not line.strip():
            continue
        if line.startswith(("- ", "* ")):
            premises.append(line)
        elif premises and line[0] in " \t":
            premises[-1] += " " + line.strip()
        else:
            break
    if not premises:
        raise PremiseBlockError("no premise line follows the header")
    return PremiseBlock(commits, premises, _header_words(header))


def parse_premise_blocks(text: str) -> list[PremiseBlock]:
    """Return the block of every ``Premises, read at`` header in ``text``, in order.

    A block ends at the first line that is neither a bullet nor indented, so the next header
    begins the next block. Raises PremiseBlockError when there is no header, or when a block
    cannot be read, its sentence then led by the block's name.
    """
    headers = list(HEADER.finditer(text))
    if not headers:
        raise PremiseBlockError("no 'Premises, read at <commit>:' header")
    blocks: list[PremiseBlock] = []
    for place, header in enumerate(headers, 1):
        try:
            blocks.append(_premise_block(text, header))
        except PremiseBlockError as err:
            name = block_name(place, len(headers), _header_words(header))
            raise PremiseBlockError(f"{name}{err}") from err
    return blocks


def parse_cites(line: str) -> list[Cite]:
    """Return each ``at <path>:<a>[-<b>]`` cite in ``line``.

    A bare ``:12`` or a basename without ``at`` is not one.
    """
    return [
        Cite(m["path"], int(m["first"]), int(m["last"] or m["first"]))
        for m in CITE.finditer(line)
    ]


def backticked_identifiers(line: str) -> list[str]:
    """Return every identifier token inside the backticked spans of ``line``, dots allowed.

    A span such as ``f(x, default=None)`` contributes ``f``, ``x``, ``default`` and ``None``.
    """
    tokens = (
        token
        for span in BACKTICKED.findall(line)
        for token in IDENTIFIER.findall(span)
    )
    return list(dict.fromkeys(tokens))


def cited_text(checkout: Path, commit: str, cite: Cite) -> str:
    """Return the lines ``cite`` names in its file at ``commit``.

    Raises CiteError when ``git show`` cannot read the path at the commit or the range is not
    inside the file.
    """
    proc = subprocess.run(
        ["git", "-C", str(checkout), "show", f"{commit}:{cite.path}"],
        capture_output=True,
    )
    if proc.returncode:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise CiteError(f"{cite.path} unreadable at {commit[:12]}: {stderr}")
    lines = proc.stdout.split(b"\n")
    if lines[-1] == b"":
        lines.pop()
    if not 1 <= cite.first <= cite.last <= len(lines):
        raise CiteError(
            f"{cite.path}:{cite.first}-{cite.last} is outside the file ({len(lines)} lines)"
        )
    return b"\n".join(lines[cite.first - 1 : cite.last]).decode(
        "utf-8", errors="replace"
    )


def repository_of(
    line: str, block: PremiseBlock, checkouts: dict[str, Path], home: Path
) -> tuple[str, Path]:
    """Return the commit and checkout the premise ``line`` is read against.

    Raises CiteError when the line's label is not in the header, an unlabeled line meets a header
    naming several labeled repositories, or a labeled repository has no checkout.
    """
    prefix = PREMISE_LABEL.match(line)
    label = prefix["label"] if prefix else None
    if label is None and len(block.commits) == 1:
        label = next(iter(block.commits))
    if label not in block.commits:
        raise CiteError(f"repository {label!r} is not one the header names")
    if label is not None and label in checkouts:
        return block.commits[label], checkouts[label]
    if label is None:
        return block.commits[label], home
    raise CiteError(f"no --checkout for repository {label!r}")


def check_premise(
    line: str, block: PremiseBlock, checkouts: dict[str, Path], home: Path
) -> list[str]:
    """Return one message per way the premise ``line`` fails; empty when its cites hold."""
    cites = parse_cites(line)
    if not cites:
        return [f"not in the writer's form: {line}"]
    try:
        commit, checkout = repository_of(line, block, checkouts, home)
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
    """Return 0 when no premise of any of the issue's premise blocks fails, and 1 otherwise."""
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
        blocks = parse_premise_blocks(fetch(args.issue))
    except PremiseBlockError as err:
        print(f"{err}\nFAIL")
        return 1
    failures = [
        f"{block_name(place, len(blocks), block.header_words)}{message}"
        for place, block in enumerate(blocks, 1)
        for line in block.premises
        for message in check_premise(line, block, checkouts, Path.cwd())
    ]
    count = f"{len(blocks)} premise block{'s' if len(blocks) != 1 else ''}"
    print(
        "\n".join(
            [*failures, "FAIL" if failures else PASS_LINE.format(blocks=count)]
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
