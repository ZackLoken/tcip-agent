"""Generate per-harness discovery files from the canonical domain-knowledge documents.

Three on-disk forms read the same canonical corpus under
`packages/tcip-mcp/src/tcip_mcp/knowledge/` without duplicating it: Claude Code's project
skills under `.claude/skills/`, the shared skill form Codex and Antigravity both discover under
`.agents/skills/`, and a generated block in the repository root's `AGENTS.md` naming the
documents for a harness with no skill mechanism of its own (Codex reads `AGENTS.md` directly;
any other client, and a harness with neither a skill nor an instruction-file mechanism, reaches
the same corpus through the `serve_domain_knowledge` MCP tool). This script renders all three from
`tcip_mcp.knowledge.list_documents()`, so there is exactly one place any of it is authored.

Run it after adding, renaming, or re-describing a knowledge document; it rewrites every
generated file in place. `tests/test_harness_discovery_generated.py` fails when a checked-in
generated file is not what the current knowledge documents produce.

    python tools/generate_harness_discovery.py
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLAUDE_SKILLS_DIR = REPO_ROOT / ".claude" / "skills"
AGENTS_SKILLS_DIR = REPO_ROOT / ".agents" / "skills"
AGENTS_MD_PATH = REPO_ROOT / "AGENTS.md"

AGENTS_BLOCK_START = "<!-- tcip:harness-discovery:start -->"
AGENTS_BLOCK_END = "<!-- tcip:harness-discovery:end -->"

# Codex reads up to 32 KiB combined across every AGENTS.md it concatenates; this generator owns
# one block in one file, so it is held to half that shared budget rather than the whole thing.
AGENTS_BLOCK_MAX_BYTES = 16 * 1024


def _render_skill_md(name: str, description: str, document_path: Path, read_instruction: str) -> str:
    """One knowledge document's SKILL.md text: its own frontmatter, verbatim, plus a body that
    points at the canonical content instead of duplicating it, phrased per `read_instruction`
    for the harness the tree is generated for."""
    import yaml

    front = yaml.safe_dump(
        {"name": name, "description": description},
        sort_keys=False, allow_unicode=True, default_flow_style=False, width=float("inf"),
    )
    relative_path = document_path.resolve().relative_to(REPO_ROOT).as_posix()
    body = f"The content is at `{relative_path}`. {read_instruction}\n"
    return f"---\n{front}---\n\n{body}"


def render_skill(name: str, description: str, document_path: Path) -> str:
    """The generated Claude Code SKILL.md text for one knowledge document."""
    return _render_skill_md(
        name, description, document_path, "Read it in full with Read before acting in its domain."
    )


def render_agents_skill(name: str, description: str, document_path: Path) -> str:
    """The generated Codex/Antigravity SKILL.md text for one knowledge document, phrased for a
    harness with its own file reader rather than a Claude tool."""
    return _render_skill_md(
        name, description, document_path, "Read it in full before acting in its domain."
    )


def render_agents_block(documents) -> str:
    """The generated block for `AGENTS.md`'s root: what TCIP is, the document index, and how a
    harness with no skill mechanism reaches the same corpus."""
    lines = [
        AGENTS_BLOCK_START,
        "TCIP is an agentic ML/CV platform for automated phenotyping in tree-crop breeding. "
        "CLAUDE.md at the repository root is the operating contract; read it before acting.",
        "",
        "Domain knowledge documents, as `name: description`, each with the file to read in "
        "full before acting in its domain:",
        "",
    ]
    for document in documents:
        relative_path = document.path.resolve().relative_to(REPO_ROOT).as_posix()
        lines.append(f"- {document.name}: {document.description} (`{relative_path}`)")
    lines += [
        "",
        "The `serve_domain_knowledge` MCP tool returns this same index when called without a name, "
        "and one document's full text when called with a name from it. A harness with no "
        "skill or instruction-file mechanism of its own reaches these documents through that "
        "tool alone.",
        AGENTS_BLOCK_END,
    ]
    return "\n".join(lines) + "\n"


def stale_skills(skills_dir: Path, render, documents) -> list[str]:
    """Document names whose generated `SKILL.md` under `skills_dir` is missing, or does not
    equal `render(name, description, path)`: the one staleness comparison every check on a
    skill tree, live or perturbed, runs against."""
    stale = []
    for document in documents:
        skill_path = skills_dir / document.name / "SKILL.md"
        if not skill_path.is_file():
            stale.append(document.name)
            continue
        expected = render(document.name, document.description, document.path)
        if skill_path.read_text(encoding="utf-8") != expected:
            stale.append(document.name)
    return stale


def stray_skill_directories(skills_dir: Path, documents) -> set[str]:
    """Directory names directly under `skills_dir` that name no document in `documents`: the
    one stray-directory comparison every check on a skill tree, live or perturbed, runs against."""
    documented_names = {document.name for document in documents}
    on_disk = {entry.name for entry in skills_dir.iterdir() if entry.is_dir()}
    return on_disk - documented_names


def _write_skill_tree(skills_dir: Path, render, documents) -> list[Path]:
    """Render every document into one `SKILL.md` under its own subdirectory of `skills_dir`,
    and return the paths written."""
    written = []
    for document in documents:
        skill_dir = skills_dir / document.name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_path = skill_dir / "SKILL.md"
        skill_path.write_text(
            render(document.name, document.description, document.path),
            encoding="utf-8", newline="\n",
        )
        written.append(skill_path)
    return written


def write_claude_skills(documents=None) -> list[Path]:
    """Write every generated Claude Code SKILL.md and return the paths written."""
    from tcip_mcp.knowledge import list_documents

    return _write_skill_tree(CLAUDE_SKILLS_DIR, render_skill, documents or list_documents())


def write_agents_skills(documents=None) -> list[Path]:
    """Write every generated Codex/Antigravity SKILL.md and return the paths written."""
    from tcip_mcp.knowledge import list_documents

    return _write_skill_tree(AGENTS_SKILLS_DIR, render_agents_skill, documents or list_documents())


def write_agents_block(documents=None, path: Path = AGENTS_MD_PATH) -> Path:
    """Rewrite `AGENTS.md`'s generated block in place, creating the file if it is absent and
    leaving any text outside the markers untouched.

    Raises `ValueError` when the rendered block exceeds the byte budget Codex's combined
    `AGENTS.md` reader allows, when the file carries a start marker with no matching end marker
    or the reverse, and when both markers are present but the end marker sits above the start.
    """
    from tcip_mcp.knowledge import list_documents

    block = render_agents_block(documents or list_documents())
    if len(block.encode("utf-8")) > AGENTS_BLOCK_MAX_BYTES:
        raise ValueError(
            f"the generated AGENTS.md block is over {AGENTS_BLOCK_MAX_BYTES} bytes; trim a "
            "document's description before it crowds out Codex's combined AGENTS.md budget"
        )
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    start = text.find(AGENTS_BLOCK_START)
    end = text.find(AGENTS_BLOCK_END)
    if start == -1 and end == -1:
        prefix = text.rstrip("\n")
        new_text = (prefix + "\n\n" if prefix else "") + block
    elif start != -1 and end != -1:
        if end <= start:
            raise ValueError(f"{path}: end marker appears above the start marker, malformed")
        # block carries the newline after its own end marker; drop the file's copy of that
        # same newline so repeated regeneration does not grow the file by one line each run.
        after = end + len(AGENTS_BLOCK_END)
        if text[after:after + 1] == "\n":
            after += 1
        new_text = text[:start] + block + text[after:]
    else:
        raise ValueError(f"{path}: carries a start marker or an end marker but not both, malformed")
    path.write_text(new_text, encoding="utf-8", newline="\n")
    return path


def main() -> int:
    for path in write_claude_skills():
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    for path in write_agents_skills():
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    print(f"wrote {write_agents_block().relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
