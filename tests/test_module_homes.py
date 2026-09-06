"""Structural (AST-only, no import of the modules under test): one-home guarantees for every
module reshape a move commit claims, across the training layer, the label-store/data-library
query functions, the proposal-engine tools, and the GUI-driving tools. Each assertion is
three-sided: the old module no longer defines the name, the new module defines it exactly once,
and no other module under any of the platform's source trees defines it either. Nested
definitions count (a def tucked inside another function, a class body, or a conditional block is
not invisible), and the old/new paths themselves are asserted to exist before anything is
scanned, so a typo'd or renamed path reads as a failure rather than a vacuous pass.
"""

import ast
from collections import Counter
from pathlib import Path

import tcip_annotation
import tcip_mcp
import tcip_web


def _src_root() -> Path:
    return Path(tcip_mcp.__file__).resolve().parent


def _module_path(rel: str) -> Path:
    return _src_root() / rel


def _web_src_root() -> Path:
    return Path(tcip_web.__file__).resolve().parent


def _web_module_path(rel: str) -> Path:
    return _web_src_root() / rel


def _package_roots() -> tuple[Path, Path]:
    """The two packages a moved training-layer/tools name could still be hiding in as a real
    duplicate: the platform package the reshape happened in, and the one engine package it
    imports from. Deliberately excludes the GUI package's route modules from the default
    elsewhere-scan: a route handler commonly carries the same bare REST-ish name
    (``get_run``, ``list_runs``) as an unrelated domain function without being a duplicate of
    it, so that scan only widens per-check, via ``extra_roots``, for a name where a real inline
    copy was found (``_logical_image_names``, below)."""
    return _src_root(), Path(tcip_annotation.__file__).resolve().parent


def _tcip_web_root() -> Path:
    return Path(tcip_web.__file__).resolve().parent


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
    extra_roots: tuple[Path, ...] = (),
    roots: "tuple[Path, ...] | None" = None,
) -> None:
    """``roots`` scopes the "defined nowhere else" scan; defaults to :func:`_package_roots`
    (tcip_mcp + tcip_annotation) for the training-layer moves this file started with. A move
    confined to tcip_web (the batch's own routes reshape) passes ``roots=(_web_src_root(),)``."""
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
    base = roots if roots is not None else _package_roots()
    for root in (*base, *extra_roots):
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


def test_label_query_functions_have_one_home():
    """The label-store/registry query library (``image_name_map`` through ``assemble_coco``)
    moved out of ``datasets.py`` into ``pipelines/data/label_queries.py``. A name a consumer
    outside the library reads (``datasets.py`` itself, or an outside-layer tool/pipeline module)
    lost its underscore; a helper only ``label_queries.py`` calls internally kept its private
    name."""
    _assert_one_home(
        {"image_name_map", "authored_frame", "resolved_classes_path", "resolve_registry_id_map",
         "coco_det_targets", "json_det_targets", "first_labels_json", "dir_label_format",
         "trainable_stems", "require_samples", "_label_record_state", "_raw_status_store",
         "confirmed_negative_names", "_exclude_contradicted", "confirmed_negative_records",
         "assemble_coco"},
        _module_path("pipelines/data/datasets.py"),
        _module_path("pipelines/data/label_queries.py"),
    )


def _assign_name_counts(path: Path) -> Counter:
    """Every module-level (``tree.body``) ``ast.Assign``/``ast.AnnAssign`` simple-name target in
    ``path``, for a constant's one-home check: unlike a def, a constant's home is meaningfully
    only its module-level binding, not one nested inside a function or class body."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    counts: Counter = Counter()
    for node in tree.body:
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for t in targets:
            if isinstance(t, ast.Name):
                counts[t.id] += 1
    return counts


def test_dataset_fingerprint_functions_have_one_home():
    """The fingerprint block (``dataset_fingerprint``, its four term helpers and
    ``fingerprint_formula_version``) moved out of ``resolution.py`` into
    ``pipelines/data/dataset_fingerprint.py``, the formula untouched."""
    _assert_one_home(
        {"dataset_fingerprint", "_labels_term", "_images_term", "_registry_term",
         "_confirmations_term", "fingerprint_formula_version"},
        _module_path("pipelines/resolution.py"),
        _module_path("pipelines/data/dataset_fingerprint.py"),
    )


def test_fingerprint_formula_version_constant_has_one_home():
    old_path = _module_path("pipelines/resolution.py")
    new_path = _module_path("pipelines/data/dataset_fingerprint.py")
    assert "FINGERPRINT_FORMULA_VERSION" not in _assign_name_counts(old_path)
    assert _assign_name_counts(new_path)["FINGERPRINT_FORMULA_VERSION"] == 1
    for root in _package_roots():
        for py_file in root.rglob("*.py"):
            if py_file in (old_path, new_path):
                continue
            assert "FINGERPRINT_FORMULA_VERSION" not in _assign_name_counts(py_file), py_file


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
    """``propose_annotations`` and ``stage_proposals`` (the latter renamed from
    ``accept_proposals``, later merged with the explicit-shapes door of the same final name),
    plus the staging primitives beneath them
    (``proposal_staging_key``, ``_staging_key_for``, ``_unresolvable_staging_source``,
    ``_region_rect_from_cells``, ``_write_region_crop``, ``_offset_candidates``), moved out of
    ``vision_tools.py`` into ``tools/proposal_tools.py``, beside the annotation-side pair below.
    ``StagingAddress`` (checked separately below, it is a class) moved with them."""
    _assert_one_home(
        {"propose_annotations", "stage_proposals", "proposal_staging_key",
         "_staging_key_for", "_unresolvable_staging_source", "_region_rect_from_cells",
         "_write_region_crop", "_offset_candidates"},
        _module_path("tools/vision_tools.py"),
        _module_path("tools/proposal_tools.py"),
    )


def test_staging_address_class_has_one_home():
    """``StagingAddress`` moved out of ``vision_tools.py`` into ``tools/proposal_tools.py``,
    beside the staging functions it addresses (checked above)."""
    _assert_one_home(
        {"StagingAddress"},
        _module_path("tools/vision_tools.py"),
        _module_path("tools/proposal_tools.py"),
        node_types=(ast.ClassDef,),
    )


def test_annotation_side_proposal_tools_have_one_home():
    """``segment_prompt`` and ``stage_proposals`` moved out of ``annotation_tools.py`` into
    ``tools/proposal_tools.py``, the same one-home reshape as the vision-side pair above."""
    _assert_one_home(
        {"segment_prompt", "stage_proposals"},
        _module_path("tools/annotation_tools.py"),
        _module_path("tools/proposal_tools.py"),
    )


def test_gui_driving_tools_have_one_home():
    """``push_panel_event``, ``focus_human_attention`` and their private drivers
    ``_focus_annotate``/``_focus_review`` moved out of ``annotation_tools.py`` into
    ``tools/gui_tools.py``, with the helpers only they used (``_subject_task``,
    ``_logical_image_names``). The private drivers keep their old names, since neither is the
    tool. The elsewhere-scan widens to ``tcip_web`` for this one check, not the default two
    packages: the GUI's own route modules are where a by-name reader most plausibly re-derives
    image naming inline instead of calling the shared one, the way ``routes/dataset.py`` once
    did before it was pointed at the same primitive gui_tools calls."""
    _assert_one_home(
        {"push_panel_event", "focus_human_attention", "_focus_annotate", "_focus_review",
         "_subject_task", "_logical_image_names"},
        _module_path("tools/annotation_tools.py"),
        _module_path("tools/gui_tools.py"),
        extra_roots=(_tcip_web_root(),),
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


def test_calibration_sweep_functions_have_one_home():
    """``calibrate_operating_point`` (de-underscored from ``_calibrate_operating_point``) and
    ``gate_evidence_summary`` (de-underscored from ``_sweep_summary``, then renamed off the
    calibration sense of "sweep") moved out of ``tools/inference_tools.py`` into
    ``pipelines/calibration.py``, their consumers being cross-module (``_run_inference_verified``
    and a dozen-plus test files)."""
    _assert_one_home(
        {"calibrate_operating_point", "gate_evidence_summary"},
        _module_path("tools/inference_tools.py"),
        _module_path("pipelines/calibration.py"),
    )


def test_redraw_calibration_holdout_has_one_home():
    """``redraw_calibration_holdout`` moved out of ``tools/inference_tools.py`` into
    ``tools/calibration_tools.py``, name unchanged. Decorators are not this test's concern:
    ``_assert_one_home`` reads def names only, never a decorator list."""
    _assert_one_home(
        {"redraw_calibration_holdout"},
        _module_path("tools/inference_tools.py"),
        _module_path("tools/calibration_tools.py"),
    )


def test_applied_operating_point_has_one_home():
    """``applied_operating_point`` (de-underscored from ``_applied_operating_point``) moved out of
    ``tools/inference_tools.py`` into ``pipelines/resolution.py``, its callers widening from
    ``run_inference``'s three internal resolutions (the dry-run preview, the verified body, the
    raster branch) to every direct resolver of a stated-or-default conf/NMS/max_dets, the
    full-frame evaluation runner included."""
    _assert_one_home(
        {"applied_operating_point"},
        _module_path("tools/inference_tools.py"),
        _module_path("pipelines/resolution.py"),
    )


def test_ordinal_regression_calibration_functions_have_one_home():
    """``calibrate_scalar_operating_point`` moved out of
    ``tools/phenology_tools.py`` into ``tools/calibration_tools.py``, name unchanged;
    ``_scalar_predictions`` (its sole helper, no other consumer) travels with it. Decorators
    are not this test's concern (see ``_assert_one_home``)."""
    _assert_one_home(
        {"calibrate_scalar_operating_point", "_scalar_predictions"},
        _module_path("tools/phenology_tools.py"),
        _module_path("tools/calibration_tools.py"),
    )


def test_ordinal_regression_tasks_constant_has_one_home():
    old_path = _module_path("tools/phenology_tools.py")
    new_path = _module_path("tools/calibration_tools.py")
    assert "_ORDINAL_REGRESSION_TASKS" not in _assign_name_counts(old_path)
    assert _assign_name_counts(new_path)["_ORDINAL_REGRESSION_TASKS"] == 1
    for root in _package_roots():
        for py_file in root.rglob("*.py"):
            if py_file in (old_path, new_path):
                continue
            assert "_ORDINAL_REGRESSION_TASKS" not in _assign_name_counts(py_file), py_file


def test_revise_trait_spec_has_one_home():
    """``revise_trait_spec`` moved out of ``tools/phenology_tools.py`` into
    ``tools/trait_spec_authoring_tools.py``, beside ``author_trait_spec``. Decorators are not
    this test's concern (see ``_assert_one_home``)."""
    _assert_one_home(
        {"revise_trait_spec"},
        _module_path("tools/phenology_tools.py"),
        _module_path("tools/trait_spec_authoring_tools.py"),
    )


def test_accept_proposals_is_absent_from_package_source():
    """The rename's structural half: the retired name ``accept_proposals`` is gone from every
    package's shipped source, not just from the live MCP registry the manifest test checks (a
    registry lookup only fires for an ``@mcp.tool()``-decorated name, so an undecorated
    back-compat alias or shim under the old name would pass that check unnoticed). Text search,
    not an AST identifier scan: a shim could just as easily be a string key or an alias
    assignment as a def. ``tests/`` deliberately keeps the old spelling (the manifest's removed
    set, this file's own history) and is out of scope by construction, since it sits outside
    every package's own source tree.
    """
    import tcip_store

    roots = (*_package_roots(), _tcip_web_root(), Path(tcip_store.__file__).resolve().parent)
    for root in roots:
        for py_file in root.rglob("*.py"):
            text = py_file.read_text(encoding="utf-8")
            assert "accept_proposals" not in text, f"{py_file} still names accept_proposals"


def test_review_priority_queue_no_longer_defines_its_own_dict_and_lock():
    """The priority-queue registry's dict-plus-lock state moved onto jobstore.JobRegistry
    (checked below); review.py keeps only its routes and its own job dataclass and worker."""
    path = _web_module_path("routes/review.py")
    assert path.is_file()
    stray = {"_pq_jobs", "_pq_lock"} & set(_assign_name_counts(path))
    assert not stray, f"review.py still defines {sorted(stray)}"


def test_images_overview_builds_no_longer_defines_its_own_dict_and_lock():
    """images.py's overview-build registry moved onto jobstore.JobRegistry the same way."""
    path = _web_module_path("routes/images.py")
    assert path.is_file()
    stray = {"_overview_jobs", "_overview_lock"} & set(_assign_name_counts(path))
    assert not stray, f"images.py still defines {sorted(stray)}"


def test_job_registry_class_is_the_one_home_for_the_dict_plus_lock_registry_shape():
    """jobstore.JobRegistry is the one home for the register/get/persist/rehydrate shape
    review.py's priority queue and images.py's overview builds used to restate around their own
    dict-plus-lock registry (inference.py and tuning.py adopt the same class too)."""
    jobstore_path = _web_module_path("jobstore.py")
    assert jobstore_path.is_file()
    counts = _def_name_counts(jobstore_path, node_types=(ast.ClassDef,))
    assert counts["JobRegistry"] == 1, "jobstore.py does not define JobRegistry exactly once"
    for py_file in _web_src_root().rglob("*.py"):
        if py_file == jobstore_path:
            continue
        other = _def_name_counts(py_file, node_types=(ast.ClassDef,))
        assert "JobRegistry" not in other, f"{py_file} also defines JobRegistry"


def test_inference_and_tuning_no_longer_bind_historical_dict_aliases():
    """inference.py's ``_jobs``/``_job_lock`` and tuning.py's ``_sweeps`` restated
    ``_registry.jobs``/``_registry.lock`` under their pre-adoption names for no shipped
    reader (tuning.py's ``_sweeps``) or none at all (inference.py's pair); every reaching test
    now goes through ``_registry`` directly, the same access review.py's and images.py's own
    registries never offered another name for. tuning.py's ``_lock`` stays: its own
    ``_workers`` dict, unrelated to the registry, still guards through it."""
    inference_path = _web_module_path("routes/inference.py")
    tuning_path = _web_module_path("routes/tuning.py")
    assert inference_path.is_file() and tuning_path.is_file()
    stray_inference = {"_jobs", "_job_lock"} & set(_assign_name_counts(inference_path))
    assert not stray_inference, f"inference.py still defines {sorted(stray_inference)}"
    stray_tuning = {"_sweeps"} & set(_assign_name_counts(tuning_path))
    assert not stray_tuning, f"tuning.py still defines {sorted(stray_tuning)}"
    assert "_lock" in _assign_name_counts(tuning_path), (
        "tuning.py's _workers guard still needs its own _lock alias")


def test_validate_reference_and_its_exclusive_helpers_moved_to_validation_module():
    """validate_reference and the two helpers only it used (``_dataset_root_of_all``,
    ``_recorded_prediction_digests``) moved out of review.py into routes/validation.py, public and
    unaliased; the route path is unchanged (checked live in test_review_path_confinement.py and
    test_review_validation_affordance.py, which still call POST /api/review/validate_reference).
    ``_prediction_digest``, ``_get_engine``, ``_bucket_of_dir``, ``_guard_path`` and ``_audit``
    stay in review.py: each is also used by a route that stayed (mark_complete, /action,
    /matches), so validation.py imports them rather than restating them."""
    _assert_one_home(
        {"validate_reference", "_dataset_root_of_all", "_recorded_prediction_digests"},
        _web_module_path("routes/review.py"),
        _web_module_path("routes/validation.py"),
        roots=(_web_src_root(),),
    )


def test_validate_reference_request_and_response_models_moved_to_validation_module():
    _assert_one_home(
        {"ValidateReferenceRequest", "ValidateReferenceResponse"},
        _web_module_path("routes/review.py"),
        _web_module_path("routes/validation.py"),
        node_types=(ast.ClassDef,),
        roots=(_web_src_root(),),
    )


def test_the_retired_sweep_vocabulary_is_absent_from_package_source():
    """The calibration sense of "sweep" retired its ``sweep_data``/``has_sweep`` spellings for
    ``gate_evidence``/``has_gate_evidence`` everywhere a producer or reader carries them. Text
    search over every package's own source tree, not an AST identifier scan: a shim could as
    easily be a string key as a def. ``tests/`` deliberately keeps the old spelling (fixtures for
    the carried-subrecord contract, this file's own history) and is out of scope by construction.
    """
    import tcip_store

    roots = (*_package_roots(), _tcip_web_root(), Path(tcip_store.__file__).resolve().parent)
    for root in roots:
        for py_file in root.rglob("*.py"):
            text = py_file.read_text(encoding="utf-8")
            assert "sweep_data" not in text, f"{py_file} still names sweep_data"
            assert "has_sweep" not in text, f"{py_file} still names has_sweep"
