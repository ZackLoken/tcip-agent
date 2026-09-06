"""The ``serve_domain_knowledge`` MCP tool: the route to the platform's domain knowledge documents for
a client with no skill or instruction-file mechanism of its own.

Claude Code reaches the same documents through the generated skills under ``.claude/skills/``;
Codex and Antigravity reach them under ``.agents/skills/`` and, for Codex, the generated block in
``AGENTS.md`` too (``tools/generate_harness_discovery.py`` renders all three); any other
client, and a harness with neither a skill nor an instruction-file mechanism, reaches them here.
Unlike every other tool in this package, its client-visible description is composed at import
time from the knowledge corpus itself (``tcip_mcp.knowledge.list_documents``) rather than left
as its bare docstring, so the selection hint a client sees is never a second copy of the corpus
to fall out of step with it. Composing at import also means a document with malformed or duplicate
frontmatter fails ``import tcip_mcp.server``, every tool with it: the intended rail, caught by
the staleness and index tests in ``tests/`` before this one tool would ever be blamed alone.
"""

from __future__ import annotations

from tcip_mcp.audit import audited
from tcip_mcp.knowledge import list_documents, read_document
from tcip_mcp.project_paths import repo_root_from_here
from tcip_mcp.server import mcp


def _repo_relative_path(document) -> str:
    """A document's path relative to the repository root, for a tool-only client with no
    checkout-relative context of its own to resolve the bare filename against."""
    return document.path.resolve().relative_to(repo_root_from_here()).as_posix()


def _description() -> str:
    """A lead sentence naming what the tool is, a sentence stating its two call forms, then one
    ``name: description`` line per knowledge document, so the tool's own description is the
    selection index."""
    lines = [
        "The platform's domain knowledge: the same documents every skill-bearing harness sees "
        "as skills.",
        "Without a name it returns the index of names and descriptions below; with a name "
        "from the lines below it returns that document's content.",
    ]
    lines += [f"{document.name}: {document.description}" for document in list_documents()]
    return "\n".join(lines)


@mcp.tool(description=_description())
@audited
def serve_domain_knowledge(name: str | None = None) -> dict:
    """Read the platform's domain knowledge: trait semantics, workflow patterns, and per-crop
    biology, the same documents Claude Code loads as generated skills. A client without skills
    reaches the identical corpus here.

    Args:
        name: One document's name, from this tool's own description. Omitted or empty returns
            the index (every document's name and description) instead of a document's content.
    """
    documents = list_documents()
    if not name:
        return {
            "documents": [
                {"name": d.name, "description": d.description, "path": _repo_relative_path(d)}
                for d in documents
            ]
        }
    for document in documents:
        if document.name == name:
            return {
                "name": document.name,
                "description": document.description,
                "content": read_document(name),
            }
    return {
        "error": f"unknown document '{name}'",
        "valid_names": sorted(d.name for d in documents),
    }
