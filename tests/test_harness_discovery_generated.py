"""The checked-in generated Claude Code skills, the shared Codex/Antigravity skill tree, and
`AGENTS.md`'s generated block are what the canonical knowledge documents currently produce, not
a hand-edited or stale copy.

`tools/generate_harness_discovery.py` renders one `.claude/skills/<name>/SKILL.md` and one
`.agents/skills/<name>/SKILL.md` per document under `packages/tcip-mcp/src/tcip_mcp/knowledge/`,
plus the block between markers in `AGENTS.md`; this holds every checked-in generated file to
that projection, the same shape `tests/test_generated_frontend_types.py` holds the frontend's
generated types module to.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = REPO_ROOT / "tools" / "generate_harness_discovery.py"


def _generator():
    spec = importlib.util.spec_from_file_location("tcip_generate_harness_discovery", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _documents():
    from tcip_mcp.knowledge import list_documents

    return list_documents()


def test_every_generated_claude_skill_matches_the_canonical_frontmatter():
    """Fails when a checked-in Claude Code skill is stale: a document renamed, re-described, or
    added since the last `python tools/generate_harness_discovery.py` run would leave Claude
    Code reading a selection hint the canonical document no longer carries."""
    generator = _generator()
    documents = _documents()
    assert documents, "no knowledge documents found; the generator has nothing to check"

    stale = generator.stale_skills(generator.CLAUDE_SKILLS_DIR, generator.render_skill, documents)
    assert not stale, (
        f"{stale} out of date under {generator.CLAUDE_SKILLS_DIR}; "
        "run python tools/generate_harness_discovery.py"
    )


def test_every_generated_agents_skill_matches_the_canonical_frontmatter():
    """The same staleness check as above, for the shared Codex/Antigravity skill tree under
    `.agents/skills/`."""
    generator = _generator()
    documents = _documents()

    stale = generator.stale_skills(generator.AGENTS_SKILLS_DIR, generator.render_agents_skill, documents)
    assert not stale, (
        f"{stale} out of date under {generator.AGENTS_SKILLS_DIR}; "
        "run python tools/generate_harness_discovery.py"
    )


def test_no_stray_generated_skill_directories():
    """Every directory under either skills tree names a real knowledge document: a document
    deleted or renamed without regenerating would otherwise leave an orphaned skill behind."""
    generator = _generator()
    documents = _documents()
    for skills_dir in (generator.CLAUDE_SKILLS_DIR, generator.AGENTS_SKILLS_DIR):
        stray = generator.stray_skill_directories(skills_dir, documents)
        assert not stray, f"{skills_dir}: stray directories {stray}"


def test_no_file_sits_directly_under_a_generated_skills_tree():
    """Both skills trees hold only per-document subdirectories: a hand-added file at either
    tree's top level would otherwise go unchecked by the directory-name comparison above."""
    generator = _generator()
    for skills_dir in (generator.CLAUDE_SKILLS_DIR, generator.AGENTS_SKILLS_DIR):
        stray_files = [p for p in skills_dir.iterdir() if p.is_file()]
        assert not stray_files, f"file(s) directly under {skills_dir}: {stray_files}"


def test_each_generated_skill_directory_holds_only_skill_md():
    """A hand-added resource file beside a generated SKILL.md would otherwise go unchecked:
    each skill directory, in either tree, holds exactly one entry, SKILL.md."""
    generator = _generator()
    for skills_dir in (generator.CLAUDE_SKILLS_DIR, generator.AGENTS_SKILLS_DIR):
        for skill_dir in skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            entries = sorted(p.name for p in skill_dir.iterdir())
            assert entries == ["SKILL.md"], f"{skill_dir.relative_to(REPO_ROOT)}: found {entries}"


def test_the_generator_writes_lf_line_endings():
    generator = _generator()
    for skills_dir in (generator.CLAUDE_SKILLS_DIR, generator.AGENTS_SKILLS_DIR):
        for skill_dir in skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            raw = (skill_dir / "SKILL.md").read_bytes()
            assert b"\r\n" not in raw, f"{skill_dir / 'SKILL.md'} carries CRLF line endings"


def test_agents_md_generated_block_matches_the_canonical_documents():
    """`AGENTS.md`'s generated block is what the current documents produce, the same staleness
    check as the skill trees, for the one block the generator owns in that file."""
    generator = _generator()
    documents = _documents()
    text = generator.AGENTS_MD_PATH.read_text(encoding="utf-8")
    start = text.find(generator.AGENTS_BLOCK_START)
    end = text.find(generator.AGENTS_BLOCK_END)
    assert start != -1 and end != -1, "AGENTS.md carries no generated block"
    actual_block = text[start:end + len(generator.AGENTS_BLOCK_END)] + "\n"
    expected_block = generator.render_agents_block(documents)
    assert actual_block == expected_block, (
        "AGENTS.md's generated block is out of date; run "
        "python tools/generate_harness_discovery.py"
    )


def test_agents_md_generated_block_stays_under_the_byte_budget():
    """The generator's own guard rail, exercised directly: a future description that pushes the
    block over Codex's combined-file budget must fail here, not silently ship."""
    generator = _generator()
    block = generator.render_agents_block(_documents())
    assert len(block.encode("utf-8")) <= generator.AGENTS_BLOCK_MAX_BYTES


def test_agents_md_block_round_trips_through_surrounding_text(tmp_path):
    """Regenerating the block leaves text outside the markers untouched: a maintainer may write
    around the generated block in AGENTS.md without it being clobbered."""
    generator = _generator()
    fixture = tmp_path / "AGENTS.md"
    fixture.write_text(
        "# Before\n\nHand-written text above the block.\n\n"
        + generator.render_agents_block(_documents())
        + "\nHand-written text below the block.\n",
        encoding="utf-8",
    )
    generator.write_agents_block(_documents(), path=fixture)
    text = fixture.read_text(encoding="utf-8")
    assert "Hand-written text above the block." in text
    assert "Hand-written text below the block." in text


def test_write_agents_block_is_idempotent_on_a_fixture_with_surrounding_text(tmp_path):
    """Regenerating a fixture AGENTS.md a second time produces byte-identical output to the
    first regeneration: the block does not grow or reflow the surrounding text run after run."""
    generator = _generator()
    documents = _documents()
    fixture = tmp_path / "AGENTS.md"
    fixture.write_text(
        "# Before\n\nHand-written text above the block.\n\n"
        + generator.render_agents_block(documents)
        + "\nHand-written text below the block.\n",
        encoding="utf-8",
    )
    generator.write_agents_block(documents, path=fixture)
    first = fixture.read_bytes()
    generator.write_agents_block(documents, path=fixture)
    second = fixture.read_bytes()
    assert first == second


def test_write_agents_block_creates_the_file_when_absent(tmp_path):
    generator = _generator()
    fixture = tmp_path / "AGENTS.md"
    generator.write_agents_block(_documents(), path=fixture)
    assert fixture.is_file()
    assert generator.AGENTS_BLOCK_START in fixture.read_text(encoding="utf-8")


def test_write_agents_block_refuses_an_oversized_block(tmp_path):
    """A future document whose description alone would blow the shared AGENTS.md budget is
    refused rather than silently written over Codex's combined-file limit."""
    generator = _generator()

    class _FakeDocument:
        def __init__(self, name, description, path):
            self.name = name
            self.description = description
            self.path = path

    huge_documents = [
        _FakeDocument(f"doc-{i}", "x" * 2000, generator.REPO_ROOT / "CLAUDE.md")
        for i in range(20)
    ]
    fixture = tmp_path / "AGENTS.md"
    with pytest.raises(ValueError):
        generator.write_agents_block(huge_documents, path=fixture)


def test_write_agents_block_refuses_a_lone_marker(tmp_path):
    """A start marker with no matching end marker (or the reverse) is a malformed hand-edit,
    refused by name rather than silently rewritten around."""
    generator = _generator()
    fixture = tmp_path / "AGENTS.md"
    fixture.write_text(generator.AGENTS_BLOCK_START + "\nstray text, no end marker\n", encoding="utf-8")
    with pytest.raises(ValueError, match="start marker or an end marker but not both"):
        generator.write_agents_block(_documents(), path=fixture)


def test_write_agents_block_refuses_an_end_marker_above_the_start(tmp_path):
    """Both markers present but in the wrong order is a distinct malformation from a lone
    marker, and gets its own message naming that case."""
    generator = _generator()
    fixture = tmp_path / "AGENTS.md"
    fixture.write_text(
        generator.AGENTS_BLOCK_END + "\nstray text\n" + generator.AGENTS_BLOCK_START + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="end marker appears above the start marker"):
        generator.write_agents_block(_documents(), path=fixture)


def test_a_deleted_agents_skill_is_caught_on_a_perturbed_copy(tmp_path):
    """Proves the staleness check actually fires on a missing file: delete one generated skill
    from a temporary copy of the checked-in tree and confirm `stale_skills`, the same helper the
    main staleness test calls, names exactly that document, rather than trusting the equality
    assertion above on a tree that has never held a deletion."""
    generator = _generator()
    documents = _documents()
    copy_dir = tmp_path / "agents-skills"
    shutil.copytree(generator.AGENTS_SKILLS_DIR, copy_dir)
    victim = documents[0]
    (copy_dir / victim.name / "SKILL.md").unlink()

    stale = generator.stale_skills(copy_dir, generator.render_agents_skill, documents)
    assert stale == [victim.name]


def test_an_edited_agents_skill_is_caught_on_a_perturbed_copy(tmp_path):
    """Proves the staleness check actually fires on a hand-edit: append text to one generated
    skill in a temporary copy and confirm `stale_skills` names exactly that document."""
    generator = _generator()
    documents = _documents()
    copy_dir = tmp_path / "agents-skills"
    shutil.copytree(generator.AGENTS_SKILLS_DIR, copy_dir)
    victim = documents[0]
    victim_path = copy_dir / victim.name / "SKILL.md"
    victim_path.write_text(
        victim_path.read_text(encoding="utf-8") + "\nhand-edited\n", encoding="utf-8"
    )

    stale = generator.stale_skills(copy_dir, generator.render_agents_skill, documents)
    assert stale == [victim.name]


def test_an_extra_agents_skill_directory_is_caught_on_a_perturbed_copy(tmp_path):
    """Proves the stray-directory check actually fires: add a directory naming no real document
    to a temporary copy and confirm `stray_skill_directories`, the same helper the main
    stray-directory test calls, names it."""
    generator = _generator()
    documents = _documents()
    copy_dir = tmp_path / "agents-skills"
    shutil.copytree(generator.AGENTS_SKILLS_DIR, copy_dir)
    stray_dir = copy_dir / "not-a-real-document"
    stray_dir.mkdir()
    (stray_dir / "SKILL.md").write_text("stray\n", encoding="utf-8")

    stray = generator.stray_skill_directories(copy_dir, documents)
    assert stray == {"not-a-real-document"}


def test_the_live_tree_is_unchanged_by_running_the_generator_twice():
    """Running every writer against the real checked-in tree twice in a row produces the same
    bytes both times: the promised idempotency, checked against the rendered files rather than
    by shelling out to `git status` (a checked-in tree that already matches the documents, as
    the staleness tests above confirm, means neither run should change anything)."""
    generator = _generator()
    documents = _documents()

    def _snapshot() -> dict[Path, bytes]:
        snapshot: dict[Path, bytes] = {}
        for skills_dir, render in (
            (generator.CLAUDE_SKILLS_DIR, generator.render_skill),
            (generator.AGENTS_SKILLS_DIR, generator.render_agents_skill),
        ):
            for document in documents:
                path = skills_dir / document.name / "SKILL.md"
                snapshot[path] = path.read_bytes()
        snapshot[generator.AGENTS_MD_PATH] = generator.AGENTS_MD_PATH.read_bytes()
        return snapshot

    generator.write_claude_skills(documents)
    generator.write_agents_skills(documents)
    generator.write_agents_block(documents)
    first = _snapshot()

    generator.write_claude_skills(documents)
    generator.write_agents_skills(documents)
    generator.write_agents_block(documents)
    second = _snapshot()

    assert first == second
