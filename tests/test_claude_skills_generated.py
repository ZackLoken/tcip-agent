"""The checked-in generated skills under .claude/skills/ are what the canonical knowledge
documents currently produce, not a hand-edited or stale copy.

`scripts/generate_claude_skills.py` renders one `.claude/skills/<name>/SKILL.md` per document
under `packages/tcip-mcp/src/tcip_mcp/knowledge/`; this holds every checked-in generated skill to
that projection, the same shape `tests/test_generated_frontend_types.py` holds the frontend's
generated types module to.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = REPO_ROOT / "scripts" / "generate_claude_skills.py"


def _generator():
    spec = importlib.util.spec_from_file_location("tcip_generate_claude_skills", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_generated_skill_matches_the_canonical_frontmatter():
    """Fails when a checked-in generated skill is stale: a document renamed, re-described, or
    added since the last `python scripts/generate_claude_skills.py` run would leave Claude Code
    reading a selection hint the canonical document no longer carries."""
    from tcip_mcp.knowledge import list_documents

    generator = _generator()
    documents = list_documents()
    assert documents, "no knowledge documents found; the generator has nothing to check"

    for document in documents:
        skill_path = generator.CLAUDE_SKILLS_DIR / document.name / "SKILL.md"
        assert skill_path.is_file(), f"missing generated skill: {skill_path}"
        expected = generator.render_skill(document.name, document.description, document.path)
        actual = skill_path.read_text(encoding="utf-8")
        assert actual == expected, (
            f"{skill_path.relative_to(REPO_ROOT)} is out of date; "
            "run python scripts/generate_claude_skills.py"
        )


def test_no_stray_generated_skill_directories():
    """Every directory under .claude/skills/ names a real knowledge document: a document
    deleted or renamed without regenerating would otherwise leave an orphaned skill behind."""
    from tcip_mcp.knowledge import list_documents

    generator = _generator()
    documented_names = {d.name for d in list_documents()}
    on_disk = {p.name for p in generator.CLAUDE_SKILLS_DIR.iterdir() if p.is_dir()}
    assert on_disk == documented_names


def test_no_file_sits_directly_under_claude_skills():
    """.claude/skills/ holds only per-document subdirectories: a hand-added file at its top
    level would otherwise go unchecked by the directory-name comparison above."""
    generator = _generator()
    stray_files = [p for p in generator.CLAUDE_SKILLS_DIR.iterdir() if p.is_file()]
    assert not stray_files, f"file(s) directly under .claude/skills/: {stray_files}"


def test_each_generated_skill_directory_holds_only_skill_md():
    """A hand-added resource file beside a generated SKILL.md would otherwise go unchecked:
    each skill directory holds exactly one entry, SKILL.md."""
    generator = _generator()
    for skill_dir in generator.CLAUDE_SKILLS_DIR.iterdir():
        if not skill_dir.is_dir():
            continue
        entries = sorted(p.name for p in skill_dir.iterdir())
        assert entries == ["SKILL.md"], f"{skill_dir.relative_to(REPO_ROOT)}: found {entries}"


def test_the_generator_writes_lf_line_endings():
    generator = _generator()
    for skill_dir in generator.CLAUDE_SKILLS_DIR.iterdir():
        if not skill_dir.is_dir():
            continue
        raw = (skill_dir / "SKILL.md").read_bytes()
        assert b"\r\n" not in raw, f"{skill_dir / 'SKILL.md'} carries CRLF line endings"
