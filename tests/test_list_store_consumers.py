"""tools/list_store_consumers.py's writer/reader/references split, over a small fixture tree
with one store, one writer module and one reader module."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "tools" / "list_store_consumers.py"


def _load():
    spec = importlib.util.spec_from_file_location("list_store_consumers", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["list_store_consumers"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fixture_tree(tmp_path: Path) -> dict:
    """One store declared in pkg/widget_store.py, a writer, a reader, a bystander that
    references the symbol but calls no recognized seam operation, a reader that reaches the
    store only through the declaring module's own key builder (never naming the bare symbol),
    and a module that references the symbol but only calls list.append, never a store write."""
    _write(tmp_path / "pkg" / "widget_store.py",
           "from tcip_store import register_store, StoreDescriptor\n\n"
           "WIDGET_STORE = \"widgets\"\n"
           "register_store(StoreDescriptor(name=WIDGET_STORE, kind=\"record\", "
           "key_fields=(\"id\",), concurrency=\"cas\"))\n\n\n"
           "def widget_key(id):\n"
           "    return (WIDGET_STORE, id)\n")
    _write(tmp_path / "pkg" / "writer_module.py",
           "from tcip_store import replace\n"
           "from pkg.widget_store import WIDGET_STORE\n\n"
           "def save(key, value):\n"
           "    replace(key, value)\n")
    _write(tmp_path / "pkg" / "reader_module.py",
           "from tcip_store import read\n"
           "from pkg.widget_store import WIDGET_STORE\n\n"
           "def load(key):\n"
           "    return read(key)\n")
    _write(tmp_path / "pkg" / "bystander_module.py",
           "from pkg.widget_store import WIDGET_STORE\n\n"
           "NAME = WIDGET_STORE\n")
    _write(tmp_path / "pkg" / "key_builder_reader.py",
           "from tcip_store import read\n"
           "from pkg.widget_store import widget_key\n\n"
           "def load(id):\n"
           "    return read(widget_key(id))\n")
    _write(tmp_path / "pkg" / "list_append_module.py",
           "from pkg.widget_store import WIDGET_STORE\n\n"
           "findings = []\n"
           "findings.append(WIDGET_STORE)\n")
    _write(tmp_path / "pkg" / "unrelated_module.py",
           "def do_nothing():\n    return None\n")

    return {
        "python_modules": [
            {"path": "pkg/widget_store.py", "dotted_name": "pkg.widget_store",
             "imported_by": ["pkg/writer_module.py", "pkg/reader_module.py",
                              "pkg/bystander_module.py", "pkg/key_builder_reader.py",
                              "pkg/list_append_module.py"]},
            {"path": "pkg/writer_module.py", "dotted_name": "pkg.writer_module", "imported_by": []},
            {"path": "pkg/reader_module.py", "dotted_name": "pkg.reader_module", "imported_by": []},
            {"path": "pkg/bystander_module.py", "dotted_name": "pkg.bystander_module", "imported_by": []},
            {"path": "pkg/key_builder_reader.py", "dotted_name": "pkg.key_builder_reader",
             "imported_by": []},
            {"path": "pkg/list_append_module.py", "dotted_name": "pkg.list_append_module",
             "imported_by": []},
            {"path": "pkg/unrelated_module.py", "dotted_name": "pkg.unrelated_module", "imported_by": []},
        ],
    }


def test_store_consumers_splits_writer_reader_and_references(tmp_path):
    tool = _load()
    inventory = _fixture_tree(tmp_path)

    result = tool.store_consumers(inventory, tmp_path, "widgets", "pkg.widget_store")

    assert result["symbol"] == "WIDGET_STORE"
    assert result["writers"] == ["pkg.writer_module"]
    assert result["readers"] == ["pkg.key_builder_reader", "pkg.reader_module"]
    assert result["references"] == ["pkg.bystander_module", "pkg.list_append_module"]


def test_find_symbol_reads_the_name_store_convention():
    tool = _load()
    text = "OTHER = 1\nWIDGET_STORE = \"widgets\"\nMORE = 2\n"

    assert tool.find_symbol(text, "widgets") == "WIDGET_STORE"
    assert tool.find_symbol(text, "no_such_store") is None


def test_classify_module_prefers_writer_over_reader():
    tool = _load()
    assert tool.classify_module(
        "from tcip_store import replace\nreplace(key, value)\n") == "writer"
    assert tool.classify_module("from tcip_store import read\nread(key)\n") == "reader"
    assert tool.classify_module("NAME = WIDGET_STORE\n") == "references"
    assert tool.classify_module(
        "from tcip_store import read, replace\nread(key)\nreplace(key, value)\n") == "writer"


def test_classify_module_recognizes_a_qualified_store_seam_call_with_no_import(tmp_path):
    """A store-seam receiver dotted onto the operation is recognized on its own, since the
    module need not import the bare name to hold a store handle under one of these names."""
    tool = _load()
    assert tool.classify_module("store.replace(key, value)\n") == "writer"
    assert tool.classify_module("txn.read(key)\n") == "reader"


def test_classify_module_does_not_mistake_an_unrelated_bare_call_for_a_store_write():
    """A module that never imports replace/append/etc. from tcip_store, and calls a same-named
    method on an unrelated receiver (list.append), references the store without writing to it."""
    tool = _load()
    assert tool.classify_module("findings = []\nfindings.append(x)\n") == "references"


def test_key_builders_collects_only_top_level_functions_mentioning_the_symbol():
    tool = _load()
    source = (
        "WIDGET_STORE = \"widgets\"\n\n"
        "def widget_key(id):\n"
        "    return (WIDGET_STORE, id)\n\n"
        "def unrelated():\n"
        "    return 1\n"
    )

    assert tool.key_builders(source, "WIDGET_STORE") == ["widget_key"]


def test_store_consumers_answers_empty_when_the_declaring_module_is_not_in_the_inventory(tmp_path):
    tool = _load()
    result = tool.store_consumers({"python_modules": []}, tmp_path, "widgets", "pkg.widget_store")

    assert result == {"symbol": None, "writers": [], "readers": [], "references": []}


def test_build_listing_composes_one_row_per_registered_store(tmp_path, monkeypatch):
    tool = _load()
    inventory = _fixture_tree(tmp_path)

    class _FakeDescriptor:
        kind = "record"
        frozen = True
        schema_version = 2
        declared_in = "pkg.widget_store"

    fake_module = type(sys)("tcip_mcp.store_catalog")
    fake_module.bootstrapped_stores = lambda: ("widgets",)
    fake_store_module = type(sys)("tcip_store")
    fake_store_module.get_descriptor = lambda name: _FakeDescriptor()
    monkeypatch.setitem(sys.modules, "tcip_mcp.store_catalog", fake_module)
    monkeypatch.setitem(sys.modules, "tcip_store", fake_store_module)

    rows = tool.build_listing(inventory, tmp_path)

    assert rows == [{
        "store": "widgets", "kind": "record", "frozen": True, "schema_version": 2,
        "declared_in": "pkg.widget_store", "symbol": "WIDGET_STORE",
        "writers": ["pkg.writer_module"],
        "readers": ["pkg.key_builder_reader", "pkg.reader_module"],
        "references": ["pkg.bystander_module", "pkg.list_append_module"],
    }]


def test_main_json_output_is_the_built_listing(tmp_path, monkeypatch):
    tool = _load()
    canned = [{"store": "widgets", "kind": "record", "frozen": False, "schema_version": 1,
               "declared_in": "pkg.widget_store", "symbol": "WIDGET_STORE",
               "writers": [], "readers": [], "references": []}]
    monkeypatch.setattr(tool, "build_listing", lambda inventory, repo_root: canned)
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps({"python_modules": []}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "list_store_consumers.py", "--inventory-json", str(inventory_path), "--json",
    ])

    import io
    from contextlib import redirect_stdout

    out = io.StringIO()
    with redirect_stdout(out):
        code = tool.main()

    assert code == 0
    assert json.loads(out.getvalue()) == canned


def test_the_console_command_runs_end_to_end_over_the_fixture_tree(tmp_path):
    """A legitimate call still succeeds: run the real script by subprocess over the fixture."""
    inventory = _fixture_tree(tmp_path)
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--inventory-json", str(inventory_path),
         "--repo-root", str(tmp_path), "--help"],
        capture_output=True, text=True, timeout=30,
    )

    assert result.returncode == 0
    assert "--json" in result.stdout
