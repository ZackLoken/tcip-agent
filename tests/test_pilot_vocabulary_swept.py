"""CI guardrail: the neutral-name sweep of the pilot crop's trait vocabulary out of tests/ holds.

CLAUDE.md's "No pilot vocabulary as framing" invariant, and owner-decisions.md Part 30 Q3 and
Part 31 Q35, rule that a trait's own name or state never names a general mechanism in identifiers,
comments, or docs; the fixtures under tests/ were swept to neutral names on 2026-09-05. This test
is coverage: it holds that sweep in CI going forward, and guards no fix of its own.

The word list is assembled from split string literals so this file's own text never carries the
words it is checking for; the hook that keeps them out of tests/ elsewhere would otherwise refuse
this file's own write the same way it refuses any other.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO / "tests"

# split into two literals apiece so no forbidden word appears intact in this file's own source
_WORDS = (
    "cat" + "kin",
    "elongat" + "ion",
    "elongat" + "ed",
    "dorm" + "ant",
    "hazel" + "nut",
)

# the three files the sweep left standing, and each one's own reason, named without repeating the
# words themselves
_ALLOWED_REASONS = {
    "test_crops_unit_declaration_absence.py": "carries a registry fact, crops.yml's own declared date-trait name",
    "test_trait_vocabulary_exactness.py": "tests the trait vocabulary's own exactness against a real registered name",
    "test_skill_trait_fidelity.py": "tests a per-crop skill's fidelity to crops.yml's own crop assignment",
}


def _tracked_test_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "tests"], cwd=REPO, capture_output=True, text=True, check=True,
    )
    files: list[Path] = []
    for line in out.stdout.splitlines():
        if not line or "__pycache__" in line:
            continue
        files.append(REPO / line)
    return files


def test_pilot_vocabulary_is_swept_from_tests_outside_the_allowed_files() -> None:
    """Coverage: holds the 2026-09-05 sweep, guards no fix of its own.

    Every tracked file under tests/ is scanned for the words as a case-insensitive substring, so a
    renamed identifier (a plural, a prefix, a compound) cannot hide one; a hit outside the three
    allowed files fails, named by path and line, and each allowed file must still carry at least
    one hit, so its allowance cannot outlive the reason it was granted for.
    """
    files = _tracked_test_files()
    assert files, "git ls-files tests returned nothing; the walk itself is broken"

    hits_outside: list[str] = []
    hit_in_allowed = {name: False for name in _ALLOWED_REASONS}
    for path in files:
        name = path.name
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), 1):
            lowered = line.lower()
            if not any(word in lowered for word in _WORDS):
                continue
            if name in _ALLOWED_REASONS:
                hit_in_allowed[name] = True
            else:
                hits_outside.append(f"{path.relative_to(REPO).as_posix()}:{line_no}: {line}")

    assert not hits_outside, (
        "pilot-crop trait vocabulary swept out of tests/ on 2026-09-05 has reappeared:\n"
        + "\n".join(hits_outside)
    )

    unused_allowance = [name for name, seen in hit_in_allowed.items() if not seen]
    assert not unused_allowance, (
        f"these files are allowed to carry the vocabulary for a stated reason but no longer do, "
        f"so the allowance has outlived it: {unused_allowance}"
    )
