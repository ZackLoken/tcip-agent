"""Generate the thin Claude Code skill files under .claude/skills/ from the canonical
domain-knowledge documents.

Claude Code loads a project skill from `.claude/skills/<name>/SKILL.md`; the platform's
canonical content lives under `packages/tcip-mcp/src/tcip_mcp/knowledge/` instead, the one
directory every client (Claude Code's generated skills, the `domain_knowledge` MCP tool, the
guardrails) reaches through `tcip_mcp.knowledge`. This script renders one generated skill per
knowledge document: the document's own frontmatter (`name`, `description`, its selection hint),
and a two-sentence body pointing at the canonical file rather than duplicating its content, so
there is exactly one place any of it is authored.

Run it after adding, renaming, or re-describing a knowledge document; it rewrites every
generated skill in place. `tests/test_claude_skills_generated.py` fails when a checked-in
generated skill is not what the current knowledge documents produce.

    python scripts/generate_claude_skills.py
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLAUDE_SKILLS_DIR = REPO_ROOT / ".claude" / "skills"


def render_skill(name: str, description: str, document_path: Path) -> str:
    """The generated SKILL.md text for one knowledge document: its own frontmatter, verbatim,
    plus a body that points at the canonical content instead of duplicating it."""
    import yaml

    front = yaml.safe_dump(
        {"name": name, "description": description},
        sort_keys=False, allow_unicode=True, default_flow_style=False, width=float("inf"),
    )
    relative_path = document_path.resolve().relative_to(REPO_ROOT).as_posix()
    body = (
        f"The content is at `{relative_path}`. Read it in full with Read before acting in "
        "its domain.\n"
    )
    return f"---\n{front}---\n\n{body}"


def write_skills() -> list[Path]:
    """Write every generated SKILL.md and return the paths written."""
    from tcip_mcp.knowledge import list_documents

    written = []
    for document in list_documents():
        skill_dir = CLAUDE_SKILLS_DIR / document.name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_path = skill_dir / "SKILL.md"
        skill_path.write_text(
            render_skill(document.name, document.description, document.path),
            encoding="utf-8", newline="\n",
        )
        written.append(skill_path)
    return written


def main() -> int:
    for path in write_skills():
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
