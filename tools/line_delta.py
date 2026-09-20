"""Sum a change's insertions and deletions by area: package code, tests, everything else.

    python tools/line_delta.py <repo> <rev | --cached | <rev1>..<rev2>>

The one measurement a contraction change reports before it is landed (CLAUDE.md, "Working a
change"): package code that grew names a mechanism that survived beside its replacement. Reads
`git diff --numstat` or `git show --numstat` for the target and prints one line per area.
"""

from __future__ import annotations

import subprocess
import sys


def area_of(path: str) -> str:
    """The area a changed path counts under: ``packages``, ``tests`` or ``other``."""
    if path.startswith("packages/"):
        return "packages"
    if path.startswith("tests/"):
        return "tests"
    return "other"


def line_delta(repo: str, target: str) -> dict[str, tuple[int, int]]:
    """``{area: (insertions, deletions)}`` for ``target`` in ``repo``.

    ``target`` is a commit, a ``rev1..rev2`` range, or ``--cached`` for the staged change. Binary
    files, which numstat reports with dashes, are not counted.
    """
    if target == "--cached":
        args = ["git", "diff", "--cached", "--numstat"]
    elif ".." in target:
        args = ["git", "diff", "--numstat", target]
    else:
        args = ["git", "show", "--numstat", "--format=", target]
    out = subprocess.run(args, cwd=repo, capture_output=True, text=True, check=True).stdout
    totals: dict[str, list[int]] = {"packages": [0, 0], "tests": [0, 0], "other": [0, 0]}
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 3 or parts[0] == "-":
            continue
        added, removed, path = int(parts[0]), int(parts[1]), parts[2]
        totals[area_of(path)][0] += added
        totals[area_of(path)][1] += removed
    return {area: (added, removed) for area, (added, removed) in totals.items()}


def main() -> int:
    repo, target = sys.argv[1], sys.argv[2]
    for area, (added, removed) in line_delta(repo, target).items():
        print(f"{area:9} +{added} -{removed} net {added - removed:+d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
