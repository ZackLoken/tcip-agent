"""build_module_inventory's TypeScript import parser must find a `from` clause that Prettier
wrapped onto a later line than its `import` keyword, not just one on the same line."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "build_module_inventory.py"


def _load():
    """Load the script fresh, tolerating a tree with no ``.git`` (a fail-before sandbox tree,
    which ``git archive`` never carries one into): the module's own repo-root walk needs a
    marker to find, not a real checkout."""
    git_marker = SCRIPT.parent.parent / ".git"
    if not git_marker.exists():
        git_marker.touch()
    spec = importlib.util.spec_from_file_location("build_module_inventory", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_module_inventory"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_extract_ts_specifiers_finds_a_from_clause_wrapped_onto_a_later_line():
    mod = _load()
    source = (
        "import {\n"
        "  bandSelection,\n"
        "} from \"./bandSelection\";\n"
    )
    assert "./bandSelection" in mod.extract_ts_specifiers(source)


def test_build_inventory_counts_an_edge_through_a_wrapped_import(tmp_path, monkeypatch):
    mod = _load()
    src = tmp_path / "frontend_src"
    src.mkdir()
    (src / "bandSelection.ts").write_text(
        "export function bandSelection() { return 0; }\n", encoding="utf-8"
    )
    (src / "useBandSelection.ts").write_text(
        "import {\n"
        "  bandSelection,\n"
        "} from \"./bandSelection\";\n"
        "export function useBandSelection() { return bandSelection(); }\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "PY_PACKAGE_ROOTS", [])
    monkeypatch.setattr(mod, "TOOLS_ROOT", tmp_path / "no_tools_here")
    monkeypatch.setattr(mod, "TS_ROOT", src)

    inventory = mod.build_inventory()
    by_path = {r["path"]: r for r in inventory["typescript_modules"]}

    importer = by_path["frontend_src/useBandSelection.ts"]
    imported = by_path["frontend_src/bandSelection.ts"]
    assert "frontend_src/bandSelection.ts" in importer["imports"]
    assert imported["imported_by_count"] == 1
    assert "frontend_src/useBandSelection.ts" in imported["imported_by"]
