"""The tcip_mcp.knowledge reader and the serve_domain_knowledge tool built on it.

These tests hold the reader's contract over `packages/tcip-mcp/src/tcip_mcp/knowledge/` (each
document accounted for, malformed or duplicate frontmatter refused) and the tool's surface (the
index, a named document, the unknown-name error, and a description composed from that corpus at
import time).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tcip_mcp.knowledge import (
    CROPS,
    KNOWLEDGE_DIR,
    document_path,
    list_documents,
    read_document,
)
from tests import REPO_ROOT

_BACKTICKED = re.compile(r"`(\w[\w./-]*)`")
_REPOSITORY_WORDS = re.compile(r"\bREADME\b|[\w/-]*SKILL\.md")


def _repository_paths(text: str) -> list[str]:
    """Return every name in ``text`` pointing into the repository instead of at a served document.

    That is the README, a skill file, and a backticked path resolving under the repository root
    but not under the ``tcip_mcp`` package the documents ship in.
    """
    package = KNOWLEDGE_DIR.parent
    return [
        *_REPOSITORY_WORDS.findall(text),
        *(
            token
            for token in _BACKTICKED.findall(text)
            if (REPO_ROOT / token).exists() and not (package / token).exists()
        ),
    ]


def test_no_served_document_points_into_the_repository() -> None:
    """Every document the knowledge tool serves names documents, never repository paths."""
    found = {
        document.name: paths
        for document in list_documents()
        if (paths := _repository_paths(read_document(document.name)))
    }

    assert found == {}


def test_list_documents_returns_one_entry_per_markdown_file_and_the_vocabulary() -> (
    None
):
    """``list_documents`` lists each .md file under KNOWLEDGE_DIR and CROPS, each named and read."""
    documents = list_documents()
    on_disk = list(KNOWLEDGE_DIR.rglob("*.md"))
    assert documents, (
        "the walk found no documents; the count comparison below would prove nothing"
    )
    assert len(documents) == len(on_disk) + 1
    assert documents[-1] == CROPS
    names = [d.name for d in documents]
    assert len(names) == len(set(names)), "document names must be unique"
    for document in documents:
        assert document.name.strip()
        assert document.description.strip()
        body = read_document(document.name)
        assert body.strip(), f"{document.name}: body is empty"
        assert not body.startswith("---"), (
            f"{document.name}: body still carries the frontmatter fence"
        )


def test_read_document_strips_the_frontmatter() -> None:
    body = read_document("delivery")
    assert not body.startswith("---")
    assert "name: delivery" not in body


def test_document_path_resolves_a_real_file() -> None:
    path = document_path("delivery")
    assert path.is_file()
    assert path.read_text(encoding="utf-8").startswith("---")


def test_document_path_raises_for_an_unknown_name() -> None:
    with pytest.raises(KeyError):
        document_path("not-a-real-document")


def test_malformed_frontmatter_raises_value_error_naming_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tcip_mcp import knowledge

    monkeypatch.setattr(knowledge, "KNOWLEDGE_DIR", tmp_path)
    bad = tmp_path / "broken.md"
    bad.write_text("no frontmatter here at all\n", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        knowledge.list_documents()
    assert str(bad) in str(excinfo.value)


def test_duplicate_document_name_raises_value_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tcip_mcp import knowledge

    monkeypatch.setattr(knowledge, "KNOWLEDGE_DIR", tmp_path)
    for stem in ("a", "b"):
        (tmp_path / f"{stem}.md").write_text(
            '---\nname: dup\ndescription: "two documents, one name"\n---\nbody\n',
            encoding="utf-8",
        )
    with pytest.raises(ValueError) as excinfo:
        knowledge.list_documents()
    assert "dup" in str(excinfo.value)


def test_non_utf8_document_raises_value_error_naming_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tcip_mcp import knowledge

    monkeypatch.setattr(knowledge, "KNOWLEDGE_DIR", tmp_path)
    bad = tmp_path / "latin1.md"
    bad.write_bytes(
        "---\nname: latin1\ndescription: caf\xe9\n---\nbody\n".encode(
            "latin-1"
        )
    )
    with pytest.raises(ValueError) as excinfo:
        knowledge.list_documents()
    assert str(bad) in str(excinfo.value)


def test_a_utf8_bom_does_not_defeat_the_frontmatter_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tcip_mcp import knowledge

    monkeypatch.setattr(knowledge, "KNOWLEDGE_DIR", tmp_path)
    doc = tmp_path / "bommed.md"
    doc.write_bytes(
        '---\nname: bommed\ndescription: "has a byte-order mark"\n---\nbody\n'.encode(
            "utf-8-sig"
        )
    )
    documents = knowledge.list_documents()
    assert [d.name for d in documents] == ["bommed", CROPS.name]
    assert knowledge.read_document("bommed").strip() == "body"


def test_a_name_carrying_a_slash_raises_value_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tcip_mcp import knowledge

    monkeypatch.setattr(knowledge, "KNOWLEDGE_DIR", tmp_path)
    bad = tmp_path / "slashed.md"
    bad.write_text(
        (
            '---\nname: "crops/currant"\n'
            'description: "a name that is not a single segment"\n---\nbody\n'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as excinfo:
        knowledge.list_documents()
    assert str(bad) in str(excinfo.value)


def test_a_description_carrying_a_newline_raises_value_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tcip_mcp import knowledge

    monkeypatch.setattr(knowledge, "KNOWLEDGE_DIR", tmp_path)
    bad = tmp_path / "multiline.md"
    bad.write_text(
        '---\nname: multiline\ndescription: "first line\\nsecond line"\n---\nbody\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as excinfo:
        knowledge.list_documents()
    assert str(bad) in str(excinfo.value)


def test_the_vocabulary_is_a_file_under_knowledge_dir() -> None:
    assert CROPS.path.is_relative_to(KNOWLEDGE_DIR)
    assert CROPS.path.is_file()


def test_the_vocabulary_still_loads_through_the_relocated_path() -> None:
    from tcip_mcp import traits

    records = traits._crops_traits()
    assert records, (
        "crops.yml must still load real trait records from the relocated path"
    )


def test_import_tcip_mcp_server_succeeds_and_registers_serve_domain_knowledge() -> (
    None
):
    from tcip_web.cli import tool_listing

    assert "serve_domain_knowledge" in {t.name for t in tool_listing()}


def test_the_registered_tool_description_names_every_document() -> None:
    from tcip_mcp.server import Session, build_server

    tools = {
        t.name: t for t in build_server(Session())._tool_manager.list_tools()
    }
    tool = tools["serve_domain_knowledge"]
    assert (
        "Without a name it returns the index of names and descriptions below; with a name "
        "from the lines below it returns that document's content."
    ) in tool.description
    for document in list_documents():
        assert document.name in tool.description
        assert document.description in tool.description


def test_serve_domain_knowledge_with_no_name_returns_the_index() -> None:
    from tcip_mcp.tools.knowledge_tools import serve_domain_knowledge

    result = serve_domain_knowledge()
    assert result["documents"] == [
        {"name": d.name, "description": d.description}
        for d in list_documents()
    ]


def test_serve_domain_knowledge_with_a_name_returns_that_documents_body() -> (
    None
):
    from tcip_mcp.tools.knowledge_tools import serve_domain_knowledge

    result = serve_domain_knowledge("delivery")
    assert result["name"] == "delivery"
    assert result["content"] == read_document("delivery")


def test_the_trait_vocabulary_is_served_whole_by_its_name() -> None:
    """``serve_domain_knowledge(name='crops')`` returns ``crops.yml`` byte for byte."""
    from tcip_mcp.tools.knowledge_tools import serve_domain_knowledge

    result = serve_domain_knowledge("crops")

    assert result["content"].encode("utf-8") == CROPS.path.read_bytes()


def test_every_document_naming_the_vocabulary_names_the_call_serving_it() -> (
    None
):
    """A document that names ``crops.yml`` tells the agent the call that serves it."""
    from tcip_mcp.tools.knowledge_tools import serve_domain_knowledge

    call = f"`{serve_domain_knowledge.__name__}(name='crops')`"
    naming = [
        document.name
        for document in list_documents()
        if "crops.yml" in (text := read_document(document.name))
        and call not in text
    ]

    assert naming == []


def test_serve_domain_knowledge_with_an_unknown_name_names_the_available_names() -> (
    None
):
    from tcip_mcp.tools.knowledge_tools import serve_domain_knowledge

    result = serve_domain_knowledge("not-a-real-document")
    assert "error" in result
    assert "delivery" in result["valid_names"]
