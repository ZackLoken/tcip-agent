"""The ``serve_domain_knowledge`` MCP tool, serving the platform's domain knowledge documents."""

from __future__ import annotations

from tcip_mcp.knowledge import DocumentInfo, list_documents, read_document
from tcip_mcp.project_paths import repo_root_from_here
from tcip_mcp.server import tool


def _repo_relative_path(document: DocumentInfo) -> str:
    """Return a document's path relative to the repository root, as a POSIX string."""
    return (
        document.path.resolve().relative_to(repo_root_from_here()).as_posix()
    )


def _description() -> str:
    """Compose the tool's client-visible description from the knowledge corpus.

    A lead sentence naming what the tool is, a sentence stating its two call forms, then one
    ``name: description`` line per knowledge document, so the tool's own description is the
    selection index.
    """
    lines = [
        "The platform's domain knowledge: the same documents every skill-bearing harness sees "
        "as skills.",
        "Without a name it returns the index of names and descriptions below; with a name "
        "from the lines below it returns that document's content.",
    ]
    lines += [
        f"{document.name}: {document.description}"
        for document in list_documents()
    ]
    return "\n".join(lines)


@tool(description=_description())
def serve_domain_knowledge(name: str | None = None) -> dict:
    """Read the platform's domain knowledge: trait semantics, workflow patterns, crop biology.

    Args:
        name: One document's name, from this tool's own description. Omitted or empty returns the
            index (every document's name and description) instead of a document's content.

    """
    documents = list_documents()
    if not name:
        return {
            "documents": [
                {
                    "name": d.name,
                    "description": d.description,
                    "path": _repo_relative_path(d),
                }
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
