"""Coverage: no file in the tree names the deleted visual-analysis skill by its slug.

Every file's content is read straight from disk (`git ls-files`, never a copied manifest of
what should be there), checked against both the hyphen and underscore spellings of the slug, so
a stray reference anywhere in the tree fails it. This proves only that the name is gone, not
that whatever the name once pointed to still exists somewhere else.

A positive control matters here because a broken walk (a wrong root, an empty file list, a
silently-skipped read) reports the same "found nothing" verdict as a genuinely clean tree: the
assertion below, over a reference known to exist, proves the walk actually reads content, so the
main assertion's "found nothing" means the slug is gone, not that the walk never ran.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()
LIVE_SKILL_REFERENCE = "tcip_mcp/knowledge/annotation.md"
# Both spellings a reference could plausibly use (the spaced "visual analysis" is excluded: it
# is also an ordinary description elsewhere, e.g. vision_tools.py's own module docstring).
DELETED_SKILL_NEEDLES = ("visual-analysis", "visual_analysis")

# Skipped even in the no-git fallback below, so it never inflates the walk.
_EXCLUDE_DIR_NAMES = {".git", "node_modules", "__pycache__", ".pytest_cache"}


def _tracked_files() -> tuple[list[Path], bool]:
    """Every file's path (except this one), plus whether the git-free fallback had to run.

    `git ls-files` is the paths of record: exactly what the tree commits, gitignore included.
    Without git (a fail-before proof's git-archive baseline), the fallback walks the
    filesystem directly, which necessarily also sweeps this repo's own gitignored trees (docs/
    among them), since a plain walk has no index to check paths against.
    """
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
        )
        paths = [REPO / line for line in out.stdout.splitlines() if line]
        via_git = True
    except (subprocess.CalledProcessError, OSError):
        paths = [
            p for p in REPO.rglob("*")
            if p.is_file() and not _EXCLUDE_DIR_NAMES & set(p.relative_to(REPO).parts[:-1])
        ]
        via_git = False
    return [p for p in paths if p != THIS_FILE], via_git


def _files_containing(needle: str) -> tuple[list[Path], bool]:
    paths, via_git = _tracked_files()
    hits = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if needle in text:
            hits.append(path)
    return hits, via_git


def test_the_walk_actually_finds_a_reference_known_to_exist():
    # project-setup/SKILL.md names the annotation skill by this same path shape, so the walk
    # must find it here, or the main assertion below would prove nothing.
    hits, _ = _files_containing(LIVE_SKILL_REFERENCE)
    assert hits, f"the walk found no file naming {LIVE_SKILL_REFERENCE!r}; it is broken"


def test_no_file_references_the_deleted_visual_analysis_skill():
    all_hits: dict[str, list[Path]] = {}
    via_git = True
    for needle in DELETED_SKILL_NEEDLES:
        hits, via_git = _files_containing(needle)
        if hits:
            all_hits[needle] = hits
    note = "" if via_git else " (no .git here; this fallback also sweeps gitignored trees)"
    assert not all_hits, (
        f"file(s) in the tree still reference the deleted skill's slug{note}: "
        f"{ {n: [str(p.relative_to(REPO)) for p in ps] for n, ps in all_hits.items()} }"
    )
