"""Coverage: no tracked file names the deleted visual-analysis skill.

The skill's rendering-convention facts and capture semantics moved into the vision tool
docstrings (`vision_tools.py`); its visual-QA-after-staging line moved into the annotation skill;
every other reference was re-pointed or dropped. This walk is the proof the merge left nothing
behind: every tracked file's content is read straight from disk (`git ls-files`, never a copied
manifest of what should be there), so a stray reference anywhere in the tree fails it.

A positive control matters here because a broken walk (a wrong root, an empty file list, a
silently-skipped read) reports the same "found nothing" verdict as a clean merge: the assertion
below, over a reference known to exist, proves the walk actually reads content, so the main
assertion's "found nothing" means the merge landed, not that the walk never ran.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()
DELETED_SKILL = ".github/skills/visual-analysis"
LIVE_SKILL_REFERENCE = ".github/skills/annotation"

# Skipped even in the no-git fallback below, so it never inflates the walk.
_EXCLUDE_DIR_NAMES = {".git", "node_modules", "__pycache__", ".pytest_cache"}


def _tracked_files() -> list[Path]:
    # Skips this test file itself, since it necessarily names DELETED_SKILL to search for it.
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
        )
        paths = [REPO / line for line in out.stdout.splitlines() if line]
    except (subprocess.CalledProcessError, OSError):
        # No .git here (a fail-before proof's git-archive baseline): a plain walk only widens
        # the search, never narrows what the git path would have found.
        paths = [
            p for p in REPO.rglob("*")
            if p.is_file() and not _EXCLUDE_DIR_NAMES & set(p.relative_to(REPO).parts[:-1])
        ]
    return [p for p in paths if p != THIS_FILE]


def _files_containing(needle: str) -> list[Path]:
    hits = []
    for path in _tracked_files():
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if needle in text:
            hits.append(path)
    return hits


def test_the_walk_actually_finds_a_reference_known_to_exist():
    # project-setup/SKILL.md names the annotation skill by this same path shape, so the walk
    # must find it here, or the main assertion below would prove nothing.
    hits = _files_containing(LIVE_SKILL_REFERENCE)
    assert hits, f"the walk found no tracked file naming {LIVE_SKILL_REFERENCE!r}; it is broken"


def test_no_tracked_file_references_the_deleted_visual_analysis_skill():
    hits = _files_containing(DELETED_SKILL)
    assert not hits, (
        f"{len(hits)} tracked file(s) still reference the deleted skill {DELETED_SKILL!r}: "
        f"{[str(p.relative_to(REPO)) for p in hits]}"
    )
