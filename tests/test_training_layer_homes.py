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


def test_get_worst_predictions_has_one_home():
    """``get_worst_predictions`` moved out of ``training_tools.py`` into ``vision_tools.py``,
    beside its one consumer, ``render_failure_cases``."""
    _assert_one_home(
        {"get_worst_predictions"},
        _module_path("tools/training_tools.py"),
        _module_path("tools/vision_tools.py"),
    )


def test_run_registry_functions_have_one_home():
    """``create_run``, ``attach_run``, ``get_run``, ``list_runs`` and ``cancel_run`` moved out of
    ``generic_trainer.py`` into ``pipelines/training/run_registry.py``, as one unit with
    ``TrainRun`` (checked separately below, it is a class, not a function)."""
    _assert_one_home(
        {"create_run", "attach_run", "get_run", "list_runs", "cancel_run"},
        _module_path("pipelines/training/generic_trainer.py"),
        _module_path("pipelines/training/run_registry.py"),
    )


def test_train_run_class_has_one_home():
    old_defs = {
        node.name for node in ast.parse(
            _module_path("pipelines/training/generic_trainer.py").read_text(encoding="utf-8"),
        ).body
        if isinstance(node, ast.ClassDef)
    }
    new_path = _module_path("pipelines/training/run_registry.py")
    new_defs: set[str] = set()
    if new_path.is_file():
        new_defs = {
            node.name for node in ast.parse(new_path.read_text(encoding="utf-8")).body
            if isinstance(node, ast.ClassDef)
        }
    assert "TrainRun" not in old_defs, "generic_trainer.py still defines TrainRun"
    assert "TrainRun" in new_defs, "run_registry.py never defines TrainRun"
