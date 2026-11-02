"""The domain-knowledge directory and its reader.

Every document a client reaches is a file under :data:`KNOWLEDGE_DIR`: the domain documents beside
this module and the per-crop documents under ``crops/``, each selected by the two-field
frontmatter (``name``, ``description``) it carries, and :data:`CROPS`, the trait authority's
``crops.yml`` beside them, whose name and description are stated here.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).resolve().parent

_NAME_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*$")


@dataclasses.dataclass(frozen=True)
class DocumentInfo:
    """One knowledge document's selection metadata: its name, description and path."""

    name: str
    description: str
    path: Path


CROPS = DocumentInfo(
    name="crops",
    description=(
        "The trait authority, crops.yml, served whole: the name, definition, format, units and "
        "crops of every trait. Read it before asserting that a trait exists, naming one in a "
        "revision or a delivery, or assigning one to a crop."
    ),
    path=KNOWLEDGE_DIR / "crops" / "crops.yml",
)
"""The trait vocabulary, served under its own name and description with its text whole."""


def _iter_markdown_paths() -> list[Path]:
    return sorted(KNOWLEDGE_DIR.rglob("*.md"))


def _parse_frontmatter(path: Path) -> tuple[dict, str]:
    """Return a document's frontmatter mapping and its body, the text after the closing ``---``.

    Raises ``ValueError`` naming the file for anything that does not parse: a document that is not
    valid UTF-8, a missing or unclosed frontmatter block, a non-mapping result, a missing/blank
    ``name`` or ``description``, a ``name`` that is not a single safe path segment (letters,
    digits, hyphen), or a ``description`` carrying a newline.
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
        raise ValueError(
            f"{path}: 'description' carries a newline; it must be one line"
        )
    return {"name": name, "description": description}, body.lstrip("\n")


def list_documents() -> list[DocumentInfo]:
    """List every knowledge document's name, description and path.

    One entry per ``.md`` file under :data:`KNOWLEDGE_DIR`, then :data:`CROPS`. Raises
    ``ValueError`` naming the file for a document whose frontmatter is missing or malformed, and
    the same error for a document whose name duplicates an earlier one.
    """
    documents: list[DocumentInfo] = []
    seen: dict[str, Path] = {}
    for path in _iter_markdown_paths():
        front, _ = _parse_frontmatter(path)
        documents.append(
            DocumentInfo(
                name=front["name"], description=front["description"], path=path
            )
        )
    documents.append(CROPS)
    for document in documents:
        if document.name in seen:
            raise ValueError(
                f"{document.path}: document name {document.name!r} duplicates "
                f"{seen[document.name]}; every knowledge document name must be unique"
            )
        seen[document.name] = document.path
    return documents


def document_paths() -> list[Path]:
    """Return every knowledge document's path."""
    return [document.path for document in list_documents()]


def document_path(name: str) -> Path:
    """Return the path of one named document's whole file, frontmatter included.

    Raises ``KeyError`` for a name no document declares.
    """
    for document in list_documents():
        if document.name == name:
            return document.path
    raise KeyError(name)


def read_document(name: str) -> str:
    """Return one named document's body, its frontmatter stripped, or :data:`CROPS`'s text whole.

    Raises ``KeyError`` for a name no document declares.
    """
    if name == CROPS.name:
        return CROPS.path.read_bytes().decode("utf-8")
    _, body = _parse_frontmatter(document_path(name))
    return body
