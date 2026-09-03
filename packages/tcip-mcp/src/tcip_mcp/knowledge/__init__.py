"""The one canonical domain-knowledge directory and its one reader.

Every document a client reaches, whether through the generated Claude Code skills under
``.claude/skills/``, the shared skill tree Codex and Antigravity read under
``.agents/skills/``, the generated block in ``AGENTS.md``, or the ``serve_domain_knowledge`` MCP
tool, is a file under :data:`KNOWLEDGE_DIR`:
the domain documents beside this module and the per-crop documents plus ``crops.yml`` (the
trait authority) under ``crops/``. The two-field frontmatter (``name``,
``description``) each document carries is its selection hint, read once here rather than
re-parsed by every consumer.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).resolve().parent

_NAME_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*$")


@dataclasses.dataclass(frozen=True)
class DocumentInfo:
    """One knowledge document's selection metadata: enough to choose it without reading its
    body."""

    name: str
    description: str
    path: Path


def _iter_markdown_paths() -> list[Path]:
    return sorted(KNOWLEDGE_DIR.rglob("*.md"))


def _parse_frontmatter(path: Path) -> tuple[dict, str]:
    """A document's frontmatter mapping and its body, the text after the closing ``---``.

    Raises ``ValueError`` naming the file for anything that does not parse: a document that is
    not valid UTF-8, a missing or unclosed frontmatter block, a non-mapping result, a
    missing/blank ``name`` or ``description``, a ``name`` that is not a single safe path
    segment (letters, digits, hyphen), or a ``description`` carrying a newline (the composed
    tool description is one line per document). A document this rejects is never silently
    skipped; every reader of the knowledge corpus (the generated skills, the tool description,
    the guardrails) needs every document accounted for.
    """
    import yaml

    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: not valid UTF-8: {exc}") from exc
    raw = raw.removeprefix("﻿")
    if not raw.startswith("---"):
        raise ValueError(f"{path}: no YAML frontmatter")
    _, _, rest = raw.partition("---")
    front, sep, body = rest.partition("\n---")
    if not sep:
        raise ValueError(f"{path}: frontmatter not closed")
    try:
        data = yaml.safe_load(front)
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: frontmatter does not parse: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: frontmatter is not a mapping")
    name = data.get("name")
    description = data.get("description")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{path}: frontmatter has no non-empty 'name'")
    if not _NAME_SEGMENT.match(name):
        raise ValueError(
            f"{path}: 'name' {name!r} is not a single safe path segment "
            "(letters, digits, hyphen only)"
        )
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"{path}: frontmatter has no non-empty 'description'")
    if "\n" in description:
        raise ValueError(f"{path}: 'description' carries a newline; it must be one line")
    return {"name": name, "description": description}, body.lstrip("\n")


def list_documents() -> list[DocumentInfo]:
    """Every knowledge document's name, description and path, one entry per ``.md`` file under
    :data:`KNOWLEDGE_DIR`, never a hardcoded count.

    Raises ``ValueError`` naming the file for a document whose frontmatter is missing or
    malformed, and the same error for a document whose name duplicates an earlier one: a name
    collision would make selection ambiguous for every reader, so it is caught here rather than
    left to whichever reader happens to run first.
    """
    documents: list[DocumentInfo] = []
    seen: dict[str, Path] = {}
    for path in _iter_markdown_paths():
        front, _ = _parse_frontmatter(path)
        name = front["name"]
        if name in seen:
            raise ValueError(
                f"{path}: document name {name!r} duplicates {seen[name]}; every knowledge "
                "document name must be unique"
            )
        seen[name] = path
        documents.append(DocumentInfo(name=name, description=front["description"], path=path))
    return documents


def document_paths() -> list[Path]:
    """Every knowledge document's path, for a consumer that walks the whole corpus as files (a
    prose-surface scanner, an example-code checker) rather than selecting one by name."""
    return [document.path for document in list_documents()]


def document_path(name: str) -> Path:
    """The whole file for one named document, for a consumer that reads it as a file (a
    guardrail, a trait-fidelity check) rather than through :func:`read_document`'s
    frontmatter-stripped body.

    Raises ``KeyError`` for a name no document declares.
    """
    for document in list_documents():
        if document.name == name:
            return document.path
    raise KeyError(name)


def read_document(name: str) -> str:
    """One document's body, with its frontmatter stripped: what the ``serve_domain_knowledge`` tool
    returns for a named document.

    Raises ``KeyError`` for a name no document declares.
    """
    _, body = _parse_frontmatter(document_path(name))
    return body


def crops_yml_path() -> Path:
    """Where the crops.yml controlled vocabulary lives, the trait authority every reader (the
    runtime trait registry, the guardrails) resolves through this one function."""
    return KNOWLEDGE_DIR / "crops" / "crops.yml"
