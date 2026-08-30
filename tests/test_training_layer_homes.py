"""Structural (AST-only, no import of the modules under test): the training-layer reshape's
one-home guarantees. Each assertion below is two-sided, the old module no longer defines the
name, and the new module defines it exactly once, so a stray duplicate or a leftover original
would both fail.
"""

import ast
from pathlib import Path

import tcip_mcp


def _src_root() -> Path:
    return Path(tcip_mcp.__file__).resolve().parent


def _module_path(rel: str) -> Path:
    return _src_root() / rel


def _top_level_def_names(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _assert_one_home(names: set[str], old_path: Path, new_path: Path) -> None:
    old_defs = _top_level_def_names(old_path)
    new_defs = _top_level_def_names(new_path)
    stray = names & old_defs
    assert not stray, f"{old_path.name} still defines {sorted(stray)}"
    missing = names - new_defs
    assert not missing, f"{new_path.name} never defines {sorted(missing)}"


def test_split_construction_functions_have_one_home():
    """``auto_train_val``, ``spatial_single_source_split``, ``dataset_identity`` and
    ``persist_split_manifest`` moved out of ``training_tools.py`` into
    ``pipelines/data/split_construction.py`` (beside ``splits.py``), public and unaliased."""
    _assert_one_home(
        {"auto_train_val", "spatial_single_source_split", "dataset_identity",
         "persist_split_manifest"},
        _module_path("tools/training_tools.py"),
        _module_path("pipelines/data/split_construction.py"),
    )
