"""The tcip_mcp.knowledge reader and the serve_domain_knowledge tool built on it.

One canonical directory (`packages/tcip-mcp/src/tcip_mcp/knowledge/`) backs both the generated
Claude Code skills and this tool; these tests hold the reader's contract (every document
accounted for, malformed or duplicate frontmatter refused rather than skipped) and the tool's
surface (the index, a named document, the unknown-name error, and a description composed from
the same corpus at import time).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_mcp.knowledge import KNOWLEDGE_DIR, crops_yml_path, document_path, list_documents, read_document


def test_list_documents_returns_one_entry_per_markdown_file():
    """Never a hardcoded count: as many entries as there are .md files under KNOWLEDGE_DIR."""
    documents = list_documents()
    on_disk = list(KNOWLEDGE_DIR.rglob("*.md"))
    assert documents, "the walk found no documents; the count comparison below would prove nothing"
    assert len(documents) == len(on_disk)
    names = [d.name for d in documents]
    assert len(names) == len(set(names)), "document names must be unique"
    for document in documents:
        assert document.name.strip()
        assert document.description.strip()
        body = read_document(document.name)
        assert body.strip(), f"{document.name}: body is empty"
        assert not body.startswith("---"), f"{document.name}: body still carries the frontmatter fence"


def test_read_document_strips_the_frontmatter():
    body = read_document("delivery")
    assert not body.startswith("---")
    assert "name: delivery" not in body


def test_document_path_resolves_a_real_file():
    path = document_path("delivery")
    assert path.is_file()
    assert path.read_text(encoding="utf-8").startswith("---")


def test_document_path_raises_for_an_unknown_name():
    with pytest.raises(KeyError):
        document_path("not-a-real-document")


def test_malformed_frontmatter_raises_value_error_naming_the_file(tmp_path, monkeypatch):
    from tcip_mcp import knowledge

    monkeypatch.setattr(knowledge, "KNOWLEDGE_DIR", tmp_path)
    bad = tmp_path / "broken.md"
    bad.write_text("no frontmatter here at all\n", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        knowledge.list_documents()
    assert str(bad) in str(excinfo.value)


def test_duplicate_document_name_raises_value_error(tmp_path, monkeypatch):
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


def test_non_utf8_document_raises_value_error_naming_the_file(tmp_path, monkeypatch):
    from tcip_mcp import knowledge

    monkeypatch.setattr(knowledge, "KNOWLEDGE_DIR", tmp_path)
    bad = tmp_path / "latin1.md"
    bad.write_bytes("---\nname: latin1\ndescription: caf\xe9\n---\nbody\n".encode("latin-1"))
    with pytest.raises(ValueError) as excinfo:
        knowledge.list_documents()
    assert str(bad) in str(excinfo.value)


def test_a_utf8_bom_does_not_defeat_the_frontmatter_check(tmp_path, monkeypatch):
    from tcip_mcp import knowledge

    monkeypatch.setattr(knowledge, "KNOWLEDGE_DIR", tmp_path)
    doc = tmp_path / "bommed.md"
    doc.write_bytes(
        '---\nname: bommed\ndescription: "has a byte-order mark"\n---\nbody\n'.encode("utf-8-sig")
    )
    documents = knowledge.list_documents()
    assert [d.name for d in documents] == ["bommed"]
    assert knowledge.read_document("bommed").strip() == "body"


def test_a_name_carrying_a_slash_raises_value_error(tmp_path, monkeypatch):
    from tcip_mcp import knowledge

    monkeypatch.setattr(knowledge, "KNOWLEDGE_DIR", tmp_path)
    bad = tmp_path / "slashed.md"
    bad.write_text(
        '---\nname: "crops/currant"\ndescription: "a name that is not a single segment"\n---\nbody\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as excinfo:
        knowledge.list_documents()
    assert str(bad) in str(excinfo.value)


def test_a_description_carrying_a_newline_raises_value_error(tmp_path, monkeypatch):
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


def test_crops_yml_path_resolves_under_knowledge_dir():
    path = crops_yml_path()
    assert path == KNOWLEDGE_DIR / "crops" / "crops.yml"
    assert path.is_file()


def test_the_vocabulary_still_loads_through_the_relocated_path():
    from tcip_mcp import traits

    records = traits._crops_traits()
    assert records, "crops.yml must still load real trait records from the relocated path"


def test_import_tcip_mcp_server_succeeds_and_registers_serve_domain_knowledge():
    import tcip_mcp.server as server

    assert "serve_domain_knowledge" in server.list_registered_tools()


def test_the_registered_tool_description_names_every_document():
    from tcip_mcp.server import mcp

    tools = {t.name: t for t in mcp._tool_manager.list_tools()}
    tool = tools["serve_domain_knowledge"]
    assert (
        "Without a name it returns the index of names and descriptions below; with a name "
        "from the lines below it returns that document's content."
    ) in tool.description
    for document in list_documents():
        assert document.name in tool.description
        assert document.description in tool.description


def test_serve_domain_knowledge_with_no_name_returns_the_index():
    from tcip_mcp.tools.knowledge_tools import serve_domain_knowledge

    result = serve_domain_knowledge()
    names = {d["name"] for d in result["documents"]}
    assert names == {d.name for d in list_documents()}
    for entry in result["documents"]:
        assert entry["path"], f"{entry['name']}: index entry carries no path"
        assert not Path(entry["path"]).is_absolute(), f"{entry['name']}: path is not repo-relative"


def test_serve_domain_knowledge_with_a_name_returns_that_documents_body():
    from tcip_mcp.tools.knowledge_tools import serve_domain_knowledge

    result = serve_domain_knowledge("delivery")
    assert result["name"] == "delivery"
    assert result["content"] == read_document("delivery")


def test_serve_domain_knowledge_with_an_unknown_name_names_the_available_names():
    from tcip_mcp.tools.knowledge_tools import serve_domain_knowledge

    result = serve_domain_knowledge("not-a-real-document")
    assert "error" in result
    assert "delivery" in result["valid_names"]
