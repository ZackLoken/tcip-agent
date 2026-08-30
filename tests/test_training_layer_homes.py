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
    """``auto_train_val``, ``spatial_single_source_split``, ``dataset_identity``,
    ``persist_split_manifest``, ``checked_label_format``, ``build_full_admitted_dataset`` and
    ``spatial_split_raster_identity`` moved out of ``training_tools.py`` into
    ``pipelines/data/split_construction.py`` (beside ``splits.py``), public and unaliased."""
    _assert_one_home(
        {"auto_train_val", "spatial_single_source_split", "dataset_identity",
         "persist_split_manifest", "checked_label_format", "build_full_admitted_dataset",
         "spatial_split_raster_identity"},
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


def test_collation_functions_have_one_home():
    """``task_collate``, ``_detection_collate`` and ``_stack_collate`` moved out of
    ``generic_trainer.py`` into ``pipelines/training/collation.py``."""
    _assert_one_home(
        {"task_collate", "_detection_collate", "_stack_collate"},
        _module_path("pipelines/training/generic_trainer.py"),
        _module_path("pipelines/training/collation.py"),
    )


def test_eval_runner_functions_have_one_home():
    """``run_test_evaluation``, ``run_full_frame_evaluation`` and ``write_evaluation_result``
    moved out of ``evaluation.py`` into ``pipelines/training/eval_runners.py``, as one unit;
    ``evaluation_results_key`` (checked separately below, it names the evaluation_results store,
    not a function these three call, though it does move with them) travels too."""
    _assert_one_home(
        {"run_test_evaluation", "run_full_frame_evaluation", "write_evaluation_result"},
        _module_path("pipelines/training/evaluation.py"),
        _module_path("pipelines/training/eval_runners.py"),
    )


def test_evaluation_results_key_has_one_home():
    _assert_one_home(
        {"evaluation_results_key"},
        _module_path("pipelines/training/evaluation.py"),
        _module_path("pipelines/training/eval_runners.py"),
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


def _literal_loads(tree: ast.AST, literal: str) -> list[ast.AST]:
    return [
        node for node in ast.walk(tree)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("get", "pop", "setdefault") and node.args
            and isinstance(node.args[0], ast.Constant) and node.args[0].value == literal)
        or (isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load)
            and isinstance(node.slice, ast.Constant) and node.slice.value == literal)
        or (isinstance(node, ast.Compare)
            and any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops)
            and isinstance(node.left, ast.Constant) and node.left.value == literal)
    ]


def test_checkpoint_marker_keys_have_one_home():
    """Structural (AST-only, no import of the modules under test): every reader of the two
    checkpoint payload marker keys, the importable model reference and the weights, spells them
    only through ``model_build.MODEL_SOURCE_KEY``/``STATE_DICT_KEY``, never the bare
    ``"model_source"``/``"model_state_dict"`` literal. The literal scan covers every load shape
    the key could still hide behind: a ``.get(``/``.pop(``/``.setdefault(`` call, a subscript, or
    an ``in``/``not in`` membership test. A reader importing the constant under a local
    re-spelling instead of the real one would still pass the absence half alone, so the second
    half requires a genuine ``from tcip_mcp.pipelines.model_build import ...``, not just a
    same-named local variable or an import from anywhere else. ``model_build.py`` itself defines
    both constants rather than reading them, so it is excluded from both halves.
    """
    keys = {
        "model_source": ("MODEL_SOURCE_KEY", {
            "experiments.py": _module_path("experiments.py"),
            "pipelines/training/subprocess_worker.py":
                _module_path("pipelines/training/subprocess_worker.py"),
            "tools/training_tools.py": _module_path("tools/training_tools.py"),
            "pipelines/training/generic_trainer.py":
                _module_path("pipelines/training/generic_trainer.py"),
            "pipelines/inference/generic_predictor.py":
                _module_path("pipelines/inference/generic_predictor.py"),
            "pipelines/inference/predictor.py": _module_path("pipelines/inference/predictor.py"),
        }),
        "model_state_dict": ("STATE_DICT_KEY", {
            "pipelines/training/generic_trainer.py":
                _module_path("pipelines/training/generic_trainer.py"),
            "pipelines/training/eval_runners.py":
                _module_path("pipelines/training/eval_runners.py"),
            "pipelines/inference/generic_predictor.py":
                _module_path("pipelines/inference/generic_predictor.py"),
            "pipelines/inference/predictor.py": _module_path("pipelines/inference/predictor.py"),
        }),
    }

    for literal, (const_name, files) in keys.items():
        for name, path in files.items():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            stray = _literal_loads(tree, literal)
            assert not stray, f"{name} still reads a raw {literal!r} literal"

            loaded = any(
                isinstance(n, ast.Name) and n.id == const_name and isinstance(n.ctx, ast.Load)
                for n in ast.walk(tree)
            )
            assert loaded, f"{name} never loads {const_name}"

            imported = any(
                isinstance(n, ast.ImportFrom) and n.module == "tcip_mcp.pipelines.model_build"
                and any(a.name == const_name for a in n.names)
                for n in ast.walk(tree)
            )
            assert imported, f"{name} reads {const_name} without importing it from model_build"
