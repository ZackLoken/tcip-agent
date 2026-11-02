"""The operating contract and the knowledge skills, written where each agent harness reads them.

The contract is the :data:`CONTRACT` document's body, written as a marked block of a harness's
instruction file so the text around the block stays its owner's. Every document's skill is a
``SKILL.md`` under a harness's skills directory naming the tool that serves the document whole.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from tcip_mcp.knowledge import list_documents, read_document

if TYPE_CHECKING:
    from tcip_mcp.knowledge import DocumentInfo

CONTRACT = "operating-contract"
"""The knowledge document holding the agent's operating contract."""

BLOCK_START = "<!-- tcip:operating-contract:start -->"
BLOCK_END = "<!-- tcip:operating-contract:end -->"

BLOCK_MAX_BYTES = 16 * 1024
"""Half the 32 KiB Codex reads across every ``AGENTS.md`` it concatenates, the block owning one."""


class HarnessFiles(NamedTuple):
    """Where one harness reads its instruction file and its skills, relative to its root."""

    instructions: Path
    skills: Path


CLAUDE_CODE = HarnessFiles(Path("CLAUDE.md"), Path(".claude") / "skills")
"""Claude Code's project instruction file and project skills directory."""

AGENTS = HarnessFiles(Path("AGENTS.md"), Path(".agents") / "skills")
"""The ``AGENTS.md`` instruction file and the ``.agents/skills`` directory Codex reads."""


def write_text(path: Path, text: str) -> Path:
    """Write ``text`` to ``path`` as UTF-8 with LF line ends, making its directory; return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def render_skill(document: DocumentInfo) -> str:
    """Return ``document``'s ``SKILL.md`` text.

    Its frontmatter is the document's name and description; its body names the tool that serves
    the document whole.
    """
    import yaml

    from tcip_mcp.tools.knowledge_tools import serve_domain_knowledge

    front = yaml.safe_dump(
        {"name": document.name, "description": document.description},
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=float("inf"),
    )
    call = f"{serve_domain_knowledge.__name__}(name='{document.name}')"
    return (
        f"---\n{front}---\n\n"
        f"Read it whole through the tcip tool `{call}` before acting in its "
        "domain.\n"
    )


def write_skills(skills: Path) -> list[Path]:
    """Write every knowledge document's :func:`render_skill` as ``<skills>/<name>/SKILL.md``.

    Each replaces the file an earlier write left; nothing else under ``skills`` is touched.
    Returns the paths written.
    """
    return [
        write_text(skills / document.name / "SKILL.md", render_skill(document))
        for document in list_documents()
    ]


def contract_block() -> str:
    """Return the :data:`CONTRACT` document's body between :data:`BLOCK_START` and the end."""
    return f"{BLOCK_START}\n{read_document(CONTRACT)}{BLOCK_END}\n"


def write_block(path: Path) -> Path:
    """Write :func:`contract_block` into the file at ``path`` and return ``path``.

    The block replaces the one between the markers an earlier write left, or is appended after
    the file's text, and the file is created when absent; the text outside the markers is kept.
    Raises ``ValueError`` when the block exceeds :data:`BLOCK_MAX_BYTES`, when the file carries
    one marker without the other, or when its end marker sits above its start marker.
    """
    block = contract_block()
    if len(block.encode("utf-8")) > BLOCK_MAX_BYTES:
        raise ValueError(
            f"the operating contract's block is over {BLOCK_MAX_BYTES} bytes; trim the "
            f"{CONTRACT} document before it crowds out Codex's combined AGENTS.md budget"
        )
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    start = text.find(BLOCK_START)
    end = text.find(BLOCK_END)
    if start == -1 and end == -1:
        prefix = text.rstrip("\n")
        return write_text(path, (prefix + "\n\n" if prefix else "") + block)
    if start == -1 or end == -1:
        raise ValueError(
            f"{path}: carries a start marker or an end marker but not both, malformed"
        )
    if end <= start:
        raise ValueError(
            f"{path}: end marker appears above the start marker, malformed"
        )
    # The block carries the newline after its own end marker; the file's copy of it is dropped.
    after = end + len(BLOCK_END)
    if text[after : after + 1] == "\n":
        after += 1
    return write_text(path, text[:start] + block + text[after:])
