"""Structural (AST-only, no import of the modules under test): the training-layer reshape's
one-home guarantees. Each assertion is three-sided: the old module no longer defines the name,
the new module defines it exactly once, and no other module under either package's source tree
defines it either. Nested definitions count (a def tucked inside another function, a class body,
or a conditional block is not invisible), and the old/new paths themselves are asserted to exist
before anything is scanned, so a typo'd or renamed path reads as a failure rather than a vacuous
pass.
"""

import ast
from collections import Counter
from pathlib import Path

import tcip_annotation
import tcip_mcp


def _src_root() -> Path:
    return Path(tcip_mcp.__file__).resolve().parent


def _module_path(rel: str) -> Path:
    return _src_root() / rel


def _package_roots() -> tuple[Path, Path]:
    """The two packages a moved training-layer/tools name could still be hiding in: the platform
    package the reshape happened in, and the one engine package it imports from."""
    return _src_root(), Path(tcip_annotation.__file__).resolve().parent


def _def_name_counts(
    path: Path, node_types: tuple[type, ...] = (ast.FunctionDef, ast.AsyncFunctionDef),
) -> Counter:
    """Every def of ``node_types`` in ``path`` that lives in module (free-function) namespace,
    nested defs included: a def tucked inside another function, a conditional block, or a
    ``try``/``except`` is counted rather than invisible, the same as one sitting in ``tree.body``.
    A def inside a class body is excluded, method and module namespaces don't collide, so an
    unrelated method that happens to share a moved function's name (a ``ctx`` wrapper delegating
    to it by design, say) is never mistaken for a stray duplicate of the free function.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    counts: Counter = Counter()

    def visit(node: ast.AST, in_class: bool) -> None:
        for child in ast.iter_child_nodes(node):
            child_in_class = in_class or isinstance(node, ast.ClassDef)
            if isinstance(child, node_types) and not child_in_class:
                counts[child.name] += 1
            visit(child, child_in_class)

    visit(tree, False)
    return counts


def _assert_one_home(
    names: set[str], old_path: Path, new_path: Path,
    node_types: tuple[type, ...] = (ast.FunctionDef, ast.AsyncFunctionDef),
) -> None:
    assert old_path.is_file(), f"{old_path} does not exist"
    assert new_path.is_file(), f"{new_path} does not exist"

    old_counts = _def_name_counts(old_path, node_types)
    stray = names & set(old_counts)
    assert not stray, f"{old_path.name} still defines {sorted(stray)}"

    new_counts = _def_name_counts(new_path, node_types)
    missing = names - set(new_counts)
    assert not missing, f"{new_path.name} never defines {sorted(missing)}"
    duplicated = {n for n in names if new_counts[n] > 1}
    assert not duplicated, f"{new_path.name} defines {sorted(duplicated)} more than once"

    elsewhere: dict[str, set[str]] = {}
    for root in _package_roots():
        for py_file in root.rglob("*.py"):
            if py_file in (old_path, new_path):
                continue
            found = names & set(_def_name_counts(py_file, node_types))
            if found:
                elsewhere.setdefault(str(py_file), set()).update(found)
    assert not elsewhere, f"{sorted(names)} also defined outside its one home: {elsewhere}"


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
    """``run_test_evaluation``, ``run_full_frame_evaluation``, ``write_evaluation_result``,
    ``evaluation_results_path`` and ``_producer_identity`` moved out of ``evaluation.py`` into
    ``pipelines/training/eval_runners.py``, as one unit; ``evaluation_results_key`` (checked
    separately below, it names the evaluation_results store, not a function these five call,
    though it does move with them) travels too."""
    _assert_one_home(
        {"run_test_evaluation", "run_full_frame_evaluation", "write_evaluation_result",
         "evaluation_results_path", "_producer_identity"},
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
    """``TrainRun`` moved out of ``generic_trainer.py`` into ``pipelines/training/run_registry.py``,
    beside the run-registry functions (checked above) that construct and read it."""
    _assert_one_home(
        {"TrainRun"},
        _module_path("pipelines/training/generic_trainer.py"),
        _module_path("pipelines/training/run_registry.py"),
        node_types=(ast.ClassDef,),
    )


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


def test_vision_side_proposal_tools_have_one_home():
    """``propose_annotations`` and ``accept_proposals`` moved out of ``vision_tools.py`` into
    ``tools/proposal_tools.py``, beside the annotation-side pair below."""
    _assert_one_home(
        {"propose_annotations", "accept_proposals"},
        _module_path("tools/vision_tools.py"),
        _module_path("tools/proposal_tools.py"),
    )


def test_annotation_side_proposal_tools_have_one_home():
    """``segment_prompt`` and ``stage_proposals`` moved out of ``annotation_tools.py`` into
    ``tools/proposal_tools.py``, the same one-home reshape as the vision-side pair above."""
    _assert_one_home(
        {"segment_prompt", "stage_proposals"},
        _module_path("tools/annotation_tools.py"),
        _module_path("tools/proposal_tools.py"),
    )


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
