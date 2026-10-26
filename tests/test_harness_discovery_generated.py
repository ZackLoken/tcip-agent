"""The harness discovery files are what the current knowledge documents produce.

The checked-in files are those `tools/generate_harness_discovery.py` writes.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests import REPO_ROOT
from tools import generate_harness_discovery

if TYPE_CHECKING:
    from collections.abc import Callable

    from tcip_mcp.knowledge import DocumentInfo

GENERATOR = REPO_ROOT / "tools" / "generate_harness_discovery.py"


def _documents() -> list[DocumentInfo]:
    from tcip_mcp.knowledge import list_documents

    return list_documents()


def _trees() -> list[tuple[Path, Callable[[str, str, Path], str]]]:
    """Return each generated skill tree with the renderer that produces it."""
    return [
        (
            generate_harness_discovery.CLAUDE_SKILLS_DIR,
            generate_harness_discovery.render_skill,
        ),
        (
            generate_harness_discovery.AGENTS_SKILLS_DIR,
            generate_harness_discovery.render_agents_skill,
        ),
    ]


def test_every_generated_skill_matches_its_document() -> None:
    documents = _documents()
    assert documents, (
        "no knowledge documents found; the generator has nothing to check"
    )
    for skills_dir, render in _trees():
        stale = generate_harness_discovery.stale_skills(
            skills_dir, render, documents
        )
        assert not stale, (
            f"{stale} out of date under {skills_dir}; run {GENERATOR.name}"
        )


def test_the_claude_settings_grant_exactly_the_research_documents_hosts() -> (
    None
):
    from tcip_web.terminal import CLAUDE_SETTINGS

    assert (
        CLAUDE_SETTINGS.read_text(encoding="utf-8")
        == generate_harness_discovery.expected_claude_settings()
    ), (
        f"{CLAUDE_SETTINGS} is out of date; run python tools/generate_harness_discovery.py"
    )


def test_research_hosts_are_read_from_the_source_list_alone() -> None:
    text = (
        "## 1. Research\n\nThe allowed/preferred set:\n\n- arXiv (`arxiv.org`): preprints.\n\n"
        "Search discipline:\n\n- See `crops.yml` for an unrelated example.\n"
    )
    assert generate_harness_discovery.research_hosts(text) == ["arxiv.org"]


def test_no_stray_generated_skill_directories() -> None:
    documents = _documents()
    for skills_dir, _render in _trees():
        stray = generate_harness_discovery.stray_skill_directories(
            skills_dir, documents
        )
        assert not stray, f"{skills_dir}: stray directories {stray}"


def test_no_file_sits_directly_under_a_generated_skills_tree() -> None:
    for skills_dir, _render in _trees():
        stray_files = [p for p in skills_dir.iterdir() if p.is_file()]
        assert not stray_files, (
            f"file(s) directly under {skills_dir}: {stray_files}"
        )


def test_each_generated_skill_directory_holds_only_skill_md() -> None:
    for skills_dir, _render in _trees():
        for skill_dir in skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            entries = sorted(p.name for p in skill_dir.iterdir())
            assert entries == ["SKILL.md"], (
                f"{skill_dir.relative_to(REPO_ROOT)}: found {entries}"
            )


def test_the_generator_writes_lf_line_endings() -> None:
    for skills_dir, _render in _trees():
        for skill_dir in skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            raw = (skill_dir / "SKILL.md").read_bytes()
            assert b"\r\n" not in raw, (
                f"{skill_dir / 'SKILL.md'} carries CRLF line endings"
            )


def test_agents_md_generated_block_matches_the_canonical_documents() -> None:
    text = generate_harness_discovery.AGENTS_MD_PATH.read_text(
        encoding="utf-8"
    )
    start = text.find(generate_harness_discovery.AGENTS_BLOCK_START)
    end = text.find(generate_harness_discovery.AGENTS_BLOCK_END)
    assert start != -1 and end != -1, "AGENTS.md carries no generated block"
    actual_block = (
        text[start : end + len(generate_harness_discovery.AGENTS_BLOCK_END)]
        + "\n"
    )
    assert actual_block == generate_harness_discovery.render_agents_block(
        _documents()
    ), (
        "AGENTS.md's generated block is out of date; run "
        "python tools/generate_harness_discovery.py"
    )


def test_agents_md_generated_block_stays_under_the_byte_budget() -> None:
    block = generate_harness_discovery.render_agents_block(_documents())
    assert (
        len(block.encode("utf-8"))
        <= generate_harness_discovery.AGENTS_BLOCK_MAX_BYTES
    )


def _agents_md_between_text(tmp_path: Path) -> Path:
    """Write an `AGENTS.md` holding the generated block between hand-written text."""
    fixture = tmp_path / "AGENTS.md"
    fixture.write_text(
        "# Before\n\nHand-written text above the block.\n\n"
        + generate_harness_discovery.render_agents_block(_documents())
        + "\nHand-written text below the block.\n",
        encoding="utf-8",
    )
    return fixture


def test_regenerating_the_block_keeps_the_text_around_it_and_changes_nothing_twice(
    tmp_path: Path,
) -> None:
    fixture = _agents_md_between_text(tmp_path)

    generate_harness_discovery.write_agents_block(_documents(), path=fixture)
    first = fixture.read_bytes()
    generate_harness_discovery.write_agents_block(_documents(), path=fixture)

    assert b"Hand-written text above the block." in first
    assert b"Hand-written text below the block." in first
    assert fixture.read_bytes() == first


def test_write_agents_block_creates_the_file_when_absent(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "AGENTS.md"
    generate_harness_discovery.write_agents_block(_documents(), path=fixture)
    assert generate_harness_discovery.AGENTS_BLOCK_START in fixture.read_text(
        encoding="utf-8"
    )


def test_write_agents_block_refuses_an_oversized_block(tmp_path: Path) -> None:
    from tcip_mcp.knowledge import DocumentInfo

    huge_documents = [
        DocumentInfo(
            f"doc-{i}",
            "x" * 2000,
            generate_harness_discovery.REPO_ROOT / "CLAUDE.md",
        )
        for i in range(20)
    ]
    with pytest.raises(ValueError):
        generate_harness_discovery.write_agents_block(
            huge_documents, path=tmp_path / "AGENTS.md"
        )


def test_write_agents_block_refuses_a_lone_marker(tmp_path: Path) -> None:
    fixture = tmp_path / "AGENTS.md"
    fixture.write_text(
        generate_harness_discovery.AGENTS_BLOCK_START
        + "\nstray text, no end marker\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError, match="start marker or an end marker but not both"
    ):
        generate_harness_discovery.write_agents_block(
            _documents(), path=fixture
        )


def test_write_agents_block_refuses_an_end_marker_above_the_start(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "AGENTS.md"
    fixture.write_text(
        generate_harness_discovery.AGENTS_BLOCK_END
        + "\nstray text\n"
        + generate_harness_discovery.AGENTS_BLOCK_START
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError, match="end marker appears above the start marker"
    ):
        generate_harness_discovery.write_agents_block(
            _documents(), path=fixture
        )


def _perturbed_agents_tree(tmp_path: Path) -> Path:
    copy_dir = tmp_path / "agents-skills"
    shutil.copytree(generate_harness_discovery.AGENTS_SKILLS_DIR, copy_dir)
    return copy_dir


def test_a_deleted_skill_is_named_stale(tmp_path: Path) -> None:
    documents = _documents()
    copy_dir = _perturbed_agents_tree(tmp_path)
    (copy_dir / documents[0].name / "SKILL.md").unlink()

    assert generate_harness_discovery.stale_skills(
        copy_dir, generate_harness_discovery.render_agents_skill, documents
    ) == [documents[0].name]


def test_an_edited_skill_is_named_stale(tmp_path: Path) -> None:
    documents = _documents()
    copy_dir = _perturbed_agents_tree(tmp_path)
    victim_path = copy_dir / documents[0].name / "SKILL.md"
    victim_path.write_text(
        victim_path.read_text(encoding="utf-8") + "\nhand-edited\n",
        encoding="utf-8",
    )

    assert generate_harness_discovery.stale_skills(
        copy_dir, generate_harness_discovery.render_agents_skill, documents
    ) == [documents[0].name]


def test_an_extra_skill_directory_is_named_stray(tmp_path: Path) -> None:
    copy_dir = _perturbed_agents_tree(tmp_path)
    stray_dir = copy_dir / "not-a-real-document"
    stray_dir.mkdir()
    (stray_dir / "SKILL.md").write_text("stray\n", encoding="utf-8")

    assert generate_harness_discovery.stray_skill_directories(
        copy_dir, _documents()
    ) == {"not-a-real-document"}


def test_the_entry_point_writes_the_checked_in_files_and_the_same_bytes_twice(
    tmp_path: Path,
) -> None:
    from tcip_web.terminal import CLAUDE_SETTINGS

    written = generate_harness_discovery.generate(tmp_path)
    first = {path: path.read_bytes() for path in written}
    generate_harness_discovery.generate(tmp_path)

    assert tmp_path / CLAUDE_SETTINGS.relative_to(REPO_ROOT) in first

    for path, content in first.items():
        assert path.read_bytes() == content, f"{path} changed on a second run"
        checked_in = (REPO_ROOT / path.relative_to(tmp_path)).read_bytes()
        if path.name == "AGENTS.md":
            assert content.decode("utf-8") in checked_in.decode("utf-8")
        else:
            assert content == checked_in, (
                f"{path.relative_to(tmp_path)} is out of date"
            )
