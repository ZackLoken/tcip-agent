"""Training MCP tools, config validation, launch training, HPO, status."""

from __future__ import annotations

import itertools
import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path, PureWindowsPath
from typing import Any, Sized

from tcip_store import (
    LOG_JSON,
    RECORD_JSON,
    BadKey,
    DecodeError,
    Key,
    StoreDescriptor,
    StoreError,
    VersionConflict,
    check_json_value,
    register_store,
    store,
    stored_number,
)
from tcip_store.file_backend import RootedFileLocator

from tcip_mcp.server import mcp
from tcip_mcp.audit import audited

logger = logging.getLogger(__name__)

# Round-robins unpinned concurrent launches across available GPUs (no-op with 0-1 devices).
_gpu_round_robin = itertools.count()

# Serializes the overfit diagnostic's reseed-run-restore, since the RNG streams are process-global.
_OVERFIT_CHECK_LOCK = threading.Lock()

# A reconstructed non-terminal run reads "running" only while its heartbeat is within this
# window, past it a live process is presumed dead and it reads "interrupted".
TCIP_HEARTBEAT_STALE_SECONDS = float(os.environ.get("TCIP_HEARTBEAT_STALE_SECONDS", "600"))

_SPLIT_MANIFEST_CONFLICT_KEYS = (
    "group_by", "group_key_map", "val_ratio", "seed", "stratify_foreground",
    "test_ratio", "reserve_calibration_fraction",
)


def _split_manifest_drawn_conflicts(data_cfg: dict, split_cfg: dict) -> list[str]:
    """Every top-level key ``data.split.manifest_dir`` conflicts with beside
    ``data.val_images_dir`` (checked separately, its own refusal): a document a manifest never
    read (``coco_json``/``label_format='coco'``), or a drawn split's own parameters
    (:data:`_SPLIT_MANIFEST_CONFLICT_KEYS`). ``seed`` is admitted, not a conflict, only when
    ``data.split.redraw_within_manifest`` is true (:func:`_redraw_flag_issue` covers the flag's
    own remaining requirement, that a redraw states a seed at all): a seed left over from a
    forked or drawn config still conflicts by name otherwise, as it always has. Shared by
    ``preflight_config`` and :func:`~tcip_mcp.pipelines.data.split_construction.auto_train_val`'s
    manifest branch, so the two report the identical set for one config.
    """
    keys: tuple[str, ...] = _SPLIT_MANIFEST_CONFLICT_KEYS
    if split_cfg.get("redraw_within_manifest"):
        keys = tuple(k for k in keys if k != "seed")
    conflicts = [k for k in keys if split_cfg.get(k) is not None]
    if data_cfg.get("coco_json"):
        conflicts.append("coco_json")
    if (data_cfg.get("label_format") or "").lower() == "coco":
        conflicts.append("label_format")
    return sorted(conflicts)


def _redraw_flag_issue(split_cfg: dict) -> str | None:
    """The one objection ``data.split.redraw_within_manifest`` raises that
    :func:`_split_manifest_drawn_conflicts` cannot: the flag set true with no ``seed`` beside
    it, the redraw's own required pairing (a seed states which partition inside the manifest's
    train-plus-val members the redraw draws; the redraw states its own seed here rather than
    inheriting one, since ``run_hyperparameter_search``'s own default of 42 for a bound ``base_config``
    (:func:`_base_config_for_split_draws`) is that caller's own choice, not a fallback this
    function reaches for). ``None`` when the flag is unset, false, or paired with a seed.
    """
    if split_cfg.get("redraw_within_manifest") and split_cfg.get("seed") is None:
        return (
            "data.split.redraw_within_manifest=true requires data.split.seed: the seed the "
            "redraw draws train and val at."
        )
    return None


def _data_dir_issues(data_cfg: dict) -> list[str]:
    """Every objection ``preflight_config``'s known-loader branch raises about
    ``data.images_dir``/``data.labels_dir`` presence alone: missing, or naming a directory that
    does not exist. The one implementation ``preflight_config`` and the data picker's "As
    recorded" listing both call, so a relaunch whose recorded directories moved shows the same
    words before Start that ``launch_training`` would refuse it with. A no-op for a bespoke
    ``data.dataset_source`` config: the known-loader presence check does not apply when the
    agent's own builder owns loading.
    """
    from tcip_mcp.pipelines.model_build import DATASET_SOURCE_KEY

    if data_cfg.get(DATASET_SOURCE_KEY) is not None:
        return []
    issues: list[str] = []
    for key in ("images_dir", "labels_dir"):
        path = data_cfg.get(key)
        if not path:
            issues.append(f"Missing 'data.{key}'")
        elif not Path(path).is_dir():
            issues.append(f"Directory not found: data.{key} = '{path}'")
    return issues


def _manifest_dir_conflicts(config: dict) -> tuple[list[str], bool]:
    """Every objection a ``data.split.manifest_dir`` binding raises from ``config``'s own
    fields alone, computed without reading any manifest: the ``data.val_images_dir`` conflict,
    the drawn-split key conflicts (:data:`_SPLIT_MANIFEST_CONFLICT_KEYS`), a
    ``redraw_within_manifest`` flag with no seed beside it (:func:`_redraw_flag_issue`), and the
    task check (only detection and instance_seg admit through the ``trainable_stems`` draw a
    manifest is drawn through). ``preflight_config`` and :func:`manifest_compatibility` both
    compute these before attempting to read a manifest at all, so a moved or absent manifest
    never suppresses an objection the config alone already carries.

    Returns ``(issues, task_binds)``; when the task cannot bind a manifest at all, checking a
    manifest's subject, date or images root against it would name nothing new, so the caller
    stops there.
    """
    from tcip_mcp.pipelines.model_build import MODEL_SOURCE_KEY

    model_source = config.get(MODEL_SOURCE_KEY)
    data_cfg_raw = config.get("data")
    data_cfg: dict = data_cfg_raw if isinstance(data_cfg_raw, dict) else {}
    split_cfg_raw = data_cfg.get("split")
    split_cfg: dict = split_cfg_raw if isinstance(split_cfg_raw, dict) else {}

    issues: list[str] = []
    if data_cfg.get("val_images_dir"):
        issues.append(
            "data.split.manifest_dir conflicts with data.val_images_dir: two membership "
            "sources for one run's validation split."
        )
    conflicts = _split_manifest_drawn_conflicts(data_cfg, split_cfg)
    if conflicts:
        issues.append(
            f"data.split.manifest_dir conflicts with {conflicts}: a recorded partition and "
            "a drawn split's own parameters/source cannot both govern one run."
        )
    flag_issue = _redraw_flag_issue(split_cfg)
    if flag_issue:
        issues.append(flag_issue)

    task_for_manifest = (model_source.get("task") if isinstance(model_source, dict) else None) \
        or data_cfg.get("task", "detection")
    if task_for_manifest not in ("detection", "instance_seg"):
        issues.append(
            f"data.split.manifest_dir names a split manifest, and only detection and "
            f"instance_seg admit through the trainable_stems draw a manifest is drawn "
            f"through; task={task_for_manifest!r} cannot bind to one."
        )
        return issues, False
    return issues, True


def _manifest_dependent_issues(config: dict, manifest: dict, manifest_dir: str) -> list[str]:
    """Every objection that needs the manifest itself, read at ``manifest_dir``, to answer: the
    shared scope check every manifest consumer shares (:func:`~tcip_mcp.pipelines.data.splits.
    manifest_scope_issues`: subject/attribute agreement, the members block under the run's own
    date, its images root, the images root not having moved) plus the sides narrowed to that date
    not being empty (:func:`~tcip_mcp.pipelines.data.splits.empty_side_issue`), run first, then
    the run's own ``data.date`` disagreeing with its ``data.labels_dir``'s date, appended after
    rather than returned alone: a config wrong on both never hides the scope issue behind the
    date one. Called only once :func:`_manifest_dir_conflicts`'s task check has passed and the
    manifest has been read successfully; :func:`manifest_compatibility` composes both for a
    caller with one manifest in hand, and ``preflight_config`` calls this directly so a failed
    read never hides the other check's issues.
    """
    from tcip_mcp.dataset_layout import annotation_date
    from tcip_mcp.pipelines.data.splits import empty_side_issue, manifest_scope_issues
    from tcip_mcp.pipelines.model_build import MODEL_SOURCE_KEY

    model_source = config.get(MODEL_SOURCE_KEY)
    data_cfg_raw = config.get("data")
    data_cfg: dict = data_cfg_raw if isinstance(data_cfg_raw, dict) else {}
    task_for_manifest = (model_source.get("task") if isinstance(model_source, dict) else None) \
        or data_cfg.get("task", "detection")

    norm_src = _dataset_source_kwargs(task_for_manifest, data_cfg)
    subject, attribute = norm_src.get("subject"), norm_src.get("attribute")

    labels_dir = data_cfg.get("labels_dir", "")
    run_date = annotation_date(labels_dir)

    issues, narrowing = manifest_scope_issues(
        manifest, subject=subject, attribute=attribute, date=run_date,
        images_dir=data_cfg.get("images_dir"), label="data.images_dir",
        manifest_dir=manifest_dir,
    )
    if narrowing is not None:
        issues.extend(empty_side_issue(narrowing, run_date))

    declared_date = data_cfg.get("date")
    if declared_date is not None and declared_date != run_date:
        issues.append(
            f"data.date={declared_date!r} disagrees with the date "
            f"data.labels_dir={labels_dir!r} is under ({run_date!r}); a split "
            "manifest binds under one date, so the negative confirmations and the "
            "manifest must be read under the same one."
        )
    return issues


def _redraw_starvation_issues(config: dict, manifest: dict, manifest_dir: str) -> list[str]:
    """Whether ``data.split.redraw_within_manifest`` on ``config`` would starve a side, checked
    over ``manifest`` before any run starts: the manifest's own train-plus-val identities for
    this run's date, resolved through its recorded grouping
    (:func:`~tcip_mcp.pipelines.data.splits.manifest_redraw_universe`), named by
    :func:`~tcip_mcp.pipelines.data.splits.redraw_starved_issue` when fewer than two foreground
    groups result, over the same subject- and attribute-scoped per-stem annotation counts the
    child's own redraw counts at run time. An unrecognized ``group_by`` or an incomplete
    ``group_key_map`` surfaces here too, the redraw's own by-name refusal for that case, phrased
    under ``data.split`` like the drawn path's own group-policy check above.
    """
    from tcip_mcp.dataset_layout import annotation_date
    from tcip_mcp.pipelines.data.splits import (
        count_label_lines, manifest_redraw_universe, redraw_starved_issue,
    )

    data_cfg = config.get("data") or {}
    split_cfg = data_cfg.get("split") or {}
    run_date = annotation_date(data_cfg.get("labels_dir", ""))
    try:
        stems, group_key_fn = manifest_redraw_universe(manifest, run_date)
    except ValueError as exc:
        return [f"data.split.redraw_within_manifest: {exc}"]
    labels_dir = data_cfg.get("labels_dir", "")
    subject, attribute = data_cfg.get("subject"), data_cfg.get("attribute")
    foreground_counts = {
        s: count_label_lines(labels_dir, s, subject=subject, attribute=attribute) for s in stems
    }
    starved = redraw_starved_issue(
        stems, group_key_fn, foreground_counts=foreground_counts, manifest_dir=manifest_dir,
        date=run_date, seed=split_cfg.get("seed"), group_by=manifest.get("group_by"),
    )
    return [starved] if starved else []


def manifest_compatibility(config: dict, manifest: dict, manifest_dir: str) -> list[str]:
    """Every objection a launch binding ``config`` to ``manifest`` (read at ``manifest_dir``)
    would raise, checked ahead of that launch rather than only inside it.

    Composes :func:`_manifest_dir_conflicts` (config-only, computed before any manifest read)
    with :func:`_manifest_dependent_issues` (needs the manifest in hand): between them, the
    shared checks every other manifest consumer already carries,
    :func:`~tcip_mcp.pipelines.data.splits.refuse_if_images_root_moved` (which the training
    child, both inference entry points and calibration also call) and
    :func:`_split_manifest_drawn_conflicts`, plus the subject, attribute, date and empty-side
    comparisons ``preflight_config`` inlined until this pair existed. ``preflight_config`` calls
    both directly, in the same order, so a manifest read failure never suppresses the
    config-only issues; the data picker's :func:`list_split_choices` calls this one composed
    function per candidate manifest, before Start, over a manifest it also read itself. Never
    calls :func:`preflight_config`: that function imports the config's builder and scans its
    labels, a cost neither caller here means to pay for a compatibility read.
    """
    issues, task_binds = _manifest_dir_conflicts(config)
    if not task_binds:
        return issues
    issues.extend(_manifest_dependent_issues(config, manifest, manifest_dir))
    return issues


def candidate_config_with_manifest(config: dict, manifest_dir: str) -> dict:
    """The launch config choosing ``manifest_dir`` over ``config``'s own "As recorded" data
    section would build: ``data.split`` replaced wholesale by ``{"manifest_dir": manifest_dir}``
    with any ``data.val_images_dir`` dropped, since a chosen partition supplies its own
    validation source. The one implementation the data picker's own compatibility check
    (:func:`list_split_choices`) and the relaunch route's launch build both call, so an offer is
    exactly what would launch.
    """
    data_cfg_raw = config.get("data")
    data_cfg: dict = {**data_cfg_raw} if isinstance(data_cfg_raw, dict) else {}
    data_cfg.pop("val_images_dir", None)
    data_cfg["split"] = {"manifest_dir": manifest_dir}
    return {**config, "data": data_cfg}

# Lazy imports of heavy dependencies inside tool functions to keep server startup fast.


def preflight_config(config: dict, smoke: bool = False, overfit: bool = False) -> dict:
    """Validate a training configuration before launching.

    Not an MCP tool: run through ``tcip preflight-config``, per the admission standard
    (packages/tcip-mcp/CLAUDE.md), while staying importable for its own tests and for
    ``launch_training``, which calls this function directly before spawning the training thread.

    Config structure:
        model_source: {builder, builder_kwargs, task, in_chans}
        data: {images_dir, labels_dir, task}  # known loaders, OR a bespoke
              # {dataset_source: {builder, builder_kwargs, source_files, task}, task}
        training: {batch_size, ...}  # the full key list generic_trainer.train() reads
              # (device/seed/deterministic/mixed_precision/stages/optimizer/scheduler/
              # lr_scaling/stage_warmup_epochs/enforce_monotonic_unfreeze/
              # gradient_accumulation_steps/checkpoint_every_n_epochs/early_stopping)
              # is documented on train()'s own docstring, not repeated here, read that for the
              # canonical, always-current list.
        evaluation: {trait, selection_metric, ...}  # top level or training.evaluation, top
              # level wins; read consistently through schemas.evaluation_section.
        training_source: optional custom train(ctx) loop.

    Args:
        config: Full training configuration dict.
        smoke: When True, actually build the model and run ``check_model_contract`` (a train+eval
            forward at the resolved in_chans/num_classes/img_size). A contract failure is a
            guaranteed real-run failure, so it is appended to ``issues`` and blocks the launch,
            ``launch_training`` runs this before spawning the training thread. For a task the
            contract has no synthetic batch schema for, one real batch is built from ``data`` and
            used instead; if no batch can be built either, the boundary is unproven and that also
            blocks. Default False keeps a plain call to structural checks plus a builder import,
            no model construction and no forward pass.
        overfit: When True (with ``smoke``), also run the voluntary ``overfit_check`` diagnostic and
            report it under ``overfit_check``, never gating (a noisy-but-valid model can fail it).
            The stored report is already rendered (``model_contract.render_overfit_report``): a
            diverging model's raw losses may hold ``nan``/``inf``, which a JSON-RPC caller such as
            ``launch_training`` cannot answer with directly.
    """
    from tcip_mcp.pipelines.schemas import (
        evaluation_section, normalize_train_config, validate_train_config_schema,
    )
    from tcip_mcp.pipelines.model_build import DATASET_SOURCE_KEY, MODEL_SOURCE_KEY, TRAINING_SOURCE_KEY

    # The trainer's own view: training.* hoisted onto the top level, top-level wins.
    normalized = normalize_train_config(config)

    # Pydantic schema over the same normalized view: a model_source nested under training is
    # then typed by ModelSourceSchema (extra="forbid") exactly as a top-level one is.
    issues: list[str] = list(validate_train_config_schema(normalized))
    warnings: list[str] = []

    # model_source presence + builder importability (the one build path).
    model_source = normalized.get(MODEL_SOURCE_KEY)
    if not model_source:
        issues.append("Missing 'model_source' section")
    elif not isinstance(model_source, dict) or not model_source.get("builder"):
        issues.append("model_source must be a dict with a 'builder' (module:function)")
    else:
        from tcip_mcp.pipelines.model_build import _import_dotted
        try:
            _import_dotted(model_source["builder"])
        except Exception as exc:
            issues.append(f"model_source.builder not importable: {exc}")

    # training_source seam (mirrors model_source/dataset_source above), a bare "module:function"
    # string, not a dict.
    training_source = normalized.get(TRAINING_SOURCE_KEY)
    if training_source is not None:
        if not isinstance(training_source, str) or not training_source:
            issues.append("training_source must be a non-empty 'module:function' string")
        else:
            from tcip_mcp.pipelines.model_build import _import_dotted
            try:
                _import_dotted(training_source)
            except Exception as exc:
                issues.append(f"training_source not importable: {exc}")

    # Data config validation
    data_cfg = normalized.get("data")
    if not data_cfg:
        issues.append("Missing 'data' section")
    elif not isinstance(data_cfg, dict):
        issues.append("'data' must be a dict")
    elif data_cfg.get(DATASET_SOURCE_KEY) is not None:
        # Bespoke dataset seam (mirrors model_source): the agent's builder owns loading, so the
        # known-loader images_dir/labels_dir aren't required, only the builder must import.
        dataset_source = data_cfg[DATASET_SOURCE_KEY]
        if not isinstance(dataset_source, dict) or not dataset_source.get("builder"):
            issues.append("data.dataset_source must be a dict with a 'builder' (module:function)")
        else:
            from tcip_mcp.pipelines.model_build import _import_dotted
            try:
                _import_dotted(dataset_source["builder"])
            except Exception as exc:
                issues.append(f"data.dataset_source.builder not importable: {exc}")
    else:
        issues.extend(_data_dir_issues(data_cfg))

    # Channel firewall: probe one sample raster and check its band count against the declared
    # in_chans, so a channel-wrong train is caught here rather than deep in the training subprocess.
    # Only fires when a raster is actually readable, never a false-fail on an empty/absent dir.
    if isinstance(model_source, dict) and isinstance(data_cfg, dict) and data_cfg:
        from tcip_mcp.pipelines.model_build import declared_in_chans
        declared = declared_in_chans(model_source)
        images_dir = data_cfg.get("images_dir")
        if declared is not None and images_dir and Path(images_dir).is_dir():
            from tcip_store import SchemaVersionRefused

            from tcip_mcp.pipelines.image_utils import list_logical_images
            try:
                logical = list_logical_images(images_dir)
            except SchemaVersionRefused as exc:
                issues.append(f"a .bandgroup manifest under {images_dir} could not be read: {exc}")
                logical = {}
            sample = logical[sorted(logical)[0]] if logical else None
            if sample is not None:
                from tcip_mcp.pipelines.derivations import probe_channels
                from tcip_mcp.pipelines.resolution import (
                    ResolvedBundle, default as _resolved_default, validate_resolved_bundle,
                )
                try:
                    probed = int(probe_channels(sample))
                except Exception:
                    probed = None
                if probed is not None:
                    b = ResolvedBundle(trait="", dataset_hash=None, params={
                        "in_chans": _resolved_default("in_chans", declared)})
                    issues.extend(validate_resolved_bundle(b, probed_channels=probed))

    # Normalization provenance: per-band builder_kwargs statistics must carry which images produced them.
    image_stats_containment: str | None = None
    if isinstance(model_source, dict):
        bk = model_source.get("builder_kwargs")
        bk = bk if isinstance(bk, dict) else {}
        if bk.get("image_mean") is not None or bk.get("image_std") is not None:
            from pydantic import ValidationError

            from tcip_mcp.pipelines.schemas import ImageStatsSampling

            raw_sampling = model_source.get("image_stats_sampling")
            sampling_record: ImageStatsSampling | None = None
            if isinstance(raw_sampling, dict):
                try:
                    sampling_record = ImageStatsSampling.model_validate(raw_sampling)
                except ValidationError:
                    sampling_record = None
            if sampling_record is None or not sampling_record.windows:
                issues.append(
                    "model_source.builder_kwargs carries image_mean/image_std with no "
                    "model_source.image_stats_sampling beside it: per-band statistics must carry "
                    "a non-empty 'windows' list and a 'pixel_fraction', naming which images they "
                    "were derived from (see derivations.image_stats_provenance)."
                )
            else:
                images_dir = data_cfg.get("images_dir") if isinstance(data_cfg, dict) else None
                if images_dir and Path(images_dir).is_dir():
                    from tcip_mcp.pipelines.image_utils import list_logical_images
                    known = set()
                    for src_img in list_logical_images(images_dir).values():
                        ref_path = src_img if isinstance(src_img, Path) else src_img.manifest_path
                        known.add(str(Path(ref_path).resolve()))
                    bad = sorted({
                        label for label, _ in sampling_record.windows
                        if str(Path(label).resolve()) not in known
                    })
                    image_stats_containment = "checked"
                    if bad:
                        issues.append(
                            f"model_source.image_stats_sampling names path(s) {bad} outside "
                            f"data.images_dir={images_dir!r}."
                        )
                else:
                    # A bespoke dataset_source run legitimately has no data.images_dir to check
                    # window paths against; say so rather than silently skip or pass.
                    image_stats_containment = "not_checked"

    # Split-policy validation: mirrors the channel firewall above, only fires when a
    # grouping policy is actually declared, probes the dataset's stems the same way the channel
    # check probes a sample image, and never false-fails on an empty/absent/unreadable dir. Catches
    # an unrecognized ``group_by`` or an incomplete ``group_key_map`` here, at preflight, rather
    # than deep in ``auto_train_val`` where it would otherwise raise.
    data_cfg_dict: dict = data_cfg if isinstance(data_cfg, dict) else {}
    split_cfg = data_cfg_dict.get("split")
    split_cfg_dict: dict = split_cfg if isinstance(split_cfg, dict) else {}
    if split_cfg_dict.get("group_by") or split_cfg_dict.get("group_key_map"):
        images_dir = data_cfg_dict.get("images_dir")
        if images_dir and Path(images_dir).is_dir():
            from tcip_store import SchemaVersionRefused

            from tcip_mcp.pipelines.image_utils import list_logical_images
            try:
                stems = sorted(list_logical_images(images_dir))
            except SchemaVersionRefused as exc:
                issues.append(f"a .bandgroup manifest under {images_dir} could not be read: {exc}")
                stems = []
            if stems:
                from tcip_mcp.pipelines.data.splits import resolve_group_key_fn
                try:
                    resolve_group_key_fn(split_cfg_dict.get("group_by", "tile_prefix"), stems,
                                         group_key_map=split_cfg_dict.get("group_key_map"))
                except ValueError as exc:
                    issues.append(f"data.split: {exc}")

    # Config-only issues fire before the read, so a moved or absent manifest never hides them;
    # bind_manifest_stems itself never runs here, only what the two halves can answer without it.
    manifest_dir = split_cfg_dict.get("manifest_dir")
    if manifest_dir:
        from tcip_mcp.tools.data_tools import read_split_manifest_dir

        conflict_issues, task_binds = _manifest_dir_conflicts(normalized)
        issues.extend(conflict_issues)
        if task_binds:
            try:
                manifest = read_split_manifest_dir(manifest_dir)
            except ValueError as exc:
                issues.append(str(exc))
            else:
                issues.extend(_manifest_dependent_issues(normalized, manifest, manifest_dir))
                if split_cfg_dict.get("redraw_within_manifest"):
                    warnings.append(
                        "data.split.redraw_within_manifest=true: this run redraws train and "
                        "val inside the split manifest's own members for this date and seed; "
                        "the manifest's calibration side stays untouched."
                    )
                    issues.extend(_redraw_starvation_issues(normalized, manifest, manifest_dir))

    # Four-way spatial split feasibility (reserve_calibration_fraction, opt-in): must refuse by
    # name when infeasible, not silently degrade to no validation (see the helper's own docstring).
    reserve_cal_frac = split_cfg_dict.get("reserve_calibration_fraction")
    if reserve_cal_frac:
        issues.extend(_reserve_calibration_feasibility_issues(
            model_source, data_cfg_dict, split_cfg_dict, reserve_cal_frac, smoke=smoke))

    # Trainable-sample coverage: trainable_stems' own partition was computed
    # by DetectionDataset/InstanceSegDataset and then thrown away, a run whose label store admits
    # only a fraction of its annotated images (an unconfirmed-empty backlog, a stale-schema
    # quarantine, incomplete attribute coverage) reported "valid, no warnings" with no visibility
    # into what would silently train on far fewer images than the operator expects. Never gating,
    # a real project legitimately has unconfirmed/unannotated images, and only fires for the known
    # loaders (a dataset_source's own admission logic is the agent's to report, not this rail's).
    task_for_coverage = (model_source.get("task") if isinstance(model_source, dict) else None) \
        or (data_cfg.get("task", "detection") if isinstance(data_cfg, dict) else "detection")
    if (isinstance(data_cfg, dict) and data_cfg.get(DATASET_SOURCE_KEY) is None
            and task_for_coverage in ("detection", "instance_seg")):
        images_dir, labels_dir = data_cfg.get("images_dir"), data_cfg.get("labels_dir")
        if images_dir and labels_dir and Path(images_dir).is_dir() and Path(labels_dir).is_dir():
            contradicted_negatives: set[str] = set()
            from tcip_annotation.json_io import UnreadableLabelDocument
            try:
                from tcip_mcp.pipelines.data.label_queries import trainable_stems
                stems, sample_counts = trainable_stems(
                    labels_dir, images_dir, subject=data_cfg.get("subject"),
                    date=data_cfg.get("date"), contradicted_out=contradicted_negatives)
            except UnreadableLabelDocument as exc:
                stems, sample_counts = [], {}
                # A run over this labels_dir fails on the same file, so this blocks, not warns.
                issues.append(f"data.labels_dir: {exc}")
            except (OSError, ValueError):
                stems, sample_counts = [], {}
            if contradicted_negatives:
                warnings.append(
                    f"data: {sorted(contradicted_negatives)} are recorded negative for the "
                    "subject but their label file now holds subject annotations; the stored "
                    "negative is stale, they train on their labelled content instead, and the "
                    "confirmation needs re-review."
                )
            if sample_counts:
                dropped = {k: v for k, v in sample_counts.items()
                          if k not in ("annotated", "confirmed_negative") and v}
                total = sum(sample_counts.values())
                n_dropped = sum(dropped.values())
                if n_dropped and total:
                    warnings.append(
                        f"data: {n_dropped}/{total} candidate images ({n_dropped / total:.0%}) will "
                        f"not train, {dict(sorted(dropped.items()))}. {len(stems)} stem(s) admitted.")

        # A validation build from val_labels_dir (or labels_dir as its fallback) re-raises an
        # unreadable label instead of degrading to no validation, so it blocks here too.
        val_images_dir = data_cfg.get("val_images_dir")
        if val_images_dir:
            val_labels_dir = data_cfg.get("val_labels_dir") or labels_dir
            if val_labels_dir and Path(val_labels_dir).is_dir():
                from tcip_annotation.json_io import (
                    UnreadableLabelDocument as _ULD, prediction_documents, read_annotations,
                )
                for label_path in prediction_documents(val_labels_dir):
                    try:
                        read_annotations(str(label_path))
                    except _ULD as exc:
                        issues.append(f"data.val_labels_dir: {exc}")

    # Training config validation, read through the same top-level hoist train() reads under
    # (a top-level batch_size/stages entry wins over training.batch_size/training.stages).
    batch_size = normalized.get("batch_size", 2)
    if not isinstance(batch_size, int) or batch_size < 1:
        issues.append("'training.batch_size' must be a positive integer")

    # Per-stage 'epochs' is required; 'lr' is optional (StageSpec) and the trainer
    # reads learning rates from config['optimizer'], never from a stage. Absent
    # stages are fine, launch_training supplies its own default schedule.
    for i, stage in enumerate(normalized.get("stages") or []):
        if "epochs" not in stage:
            issues.append(f"Stage {i} missing 'epochs'")
        if "lr" in stage:
            warnings.append(
                f"training.stages[{i}].lr is set but ignored, the trainer reads learning rate "
                "only from the top-level 'optimizer' block (backbone_lr/head_lr), applied "
                "uniformly across every stage. Move the value into 'optimizer' if you meant to "
                "change it."
            )

    # Fail fast on an explicit selection_metric that is undeclared or, with a center-match trait,
    # comparability-only, at validation time rather than mid-run.
    eval_cfg = evaluation_section(config)
    if not isinstance(eval_cfg, dict):
        issues.append(
            f"'evaluation' must be a mapping (trait/selection_metric/... keys), got "
            f"{type(eval_cfg).__name__}"
        )
        eval_cfg = {}
    sel_metric = eval_cfg.get("selection_metric")
    trait_name = eval_cfg.get("trait")
    if sel_metric:
        from tcip_mcp.pipelines.training.generic_trainer import resolve_selection_metric

        task_for_check = (model_source.get("task") if isinstance(model_source, dict) else None) \
            or (data_cfg.get("task", "detection") if isinstance(data_cfg, dict) else "detection")
        try:
            resolve_selection_metric(task_for_check, trait_name, sel_metric)
        except ValueError as exc:
            issues.append(str(exc))

    result: dict = {"valid": False, "issues": issues, "warnings": warnings}
    if image_stats_containment is not None:
        result["image_stats_containment"] = image_stats_containment

    # Smoke: build the model and run the correctness contract at the resolved dims, so a broken
    # bespoke builder is caught here (before the training subprocess spawns) rather than surfacing
    # only as run.status='failed'. Only attempt once the structural checks pass, otherwise the
    # config can't build and the contract would just re-report the same failure. Overfit stays a
    # voluntary, non-gating diagnostic (a valid model can fail 20 steps on noise).
    if smoke and not issues:
        try:
            from tcip_mcp.pipelines.model_build import build_model, resolve_contract_dims
            from tcip_mcp.pipelines.model_contract import (
                check_model_contract, overfit_check, render_overfit_report,
            )

            ms = normalized.get(MODEL_SOURCE_KEY) or {}
            task = ms.get("task") or (normalized.get("data") or {}).get("task", "detection")
            dims = resolve_contract_dims(normalized, task)
            model = build_model(normalized)
            report = check_model_contract(model, task, **dims)
            batch, why_no_batch = None, None
            if report.get("not_smokeable"):
                # No synthetic batch schema for this task: smoke against a real batch from the
                # run's own dataset instead, the only reference for a task the platform doesn't enumerate.
                batch, why_no_batch = _one_real_batch(task, normalized)
                if batch is not None:
                    report = check_model_contract(model, task, sample_batch=batch, **dims)
            # ``dims`` shape the synthetic batch only, so they describe nothing once a real batch
            # is used, record which reference actually proved the contract.
            result["smoke"] = {**report, "task": task,
                               "batch_source": "dataset" if batch is not None else "synthetic",
                               "dims": None if batch is not None else dims}
            if report.get("not_smokeable"):
                issues.append(
                    f"model contract: {report['not_smokeable']} Building one from the run's data "
                    f"config failed too ({why_no_batch}), so the measurement boundary is unproven."
                )
            elif not report["ok"]:
                issues.extend(f"model contract: {msg}" for msg in report["issues"])
            if overfit:
                from tcip_mcp.pipelines.training.generic_trainer import (
                    capture_rng_state, restore_rng_state,
                )

                # Same batch the contract used, otherwise this re-synthesizes and reports a false
                # "does not learn" for exactly the bespoke tasks the real-batch path exists for.
                with _OVERFIT_CHECK_LOCK:
                    rng_state = capture_rng_state()
                    try:
                        raw_report = overfit_check(model, task, sample_batch=batch, **dims)
                    except Exception as exc:  # noqa: BLE001, becomes the report's issue only
                        raw_report = {
                            "passed": False, "losses": [], "initial": None, "final": None,
                            "issue": f"overfit check failed: {exc}",
                        }
                    finally:
                        restore_rng_state(rng_state)
                # Rendered before storage: this tool answers over JSON-RPC, which a raw non-finite
                # loss cannot cross, so a diverging model's report is sanitized here, once.
                result["overfit_check"] = render_overfit_report(raw_report)
        except Exception as exc:  # noqa: BLE001, a build/contract crash is itself a blocking issue
            issues.append(f"model smoke build failed: {exc}")

    result["valid"] = len(issues) == 0
    return result


_RUN_DOC = RootedFileLocator(suffix=".json")
"""One document in a run's own output directory."""

LAUNCH_CONFIG_STORE = "run_launch_config"
_LAUNCH_CONFIG_PARTS = ("launch_config",)
register_store(
    StoreDescriptor(
        name=LAUNCH_CONFIG_STORE,
        kind="record",
        key_fields=("document",),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="last_writer_wins",
        locator=_RUN_DOC,
    )
)


def launch_config_key(output_dir: Path | str) -> Key:
    """The bootstrap config a training subprocess reads itself out of.

    Keyed off the run's output directory, which the child is given, so neither side carries a
    path to the other's document. ``last_writer_wins``: the launching process writes the whole
    config once, before the child exists, and the child only reads it.
    """
    return Key(LAUNCH_CONFIG_STORE, str(Path(output_dir).resolve()), _LAUNCH_CONFIG_PARTS)


@mcp.tool()
@audited
def launch_training(
    config: dict, output_dir: str = "", resume_from: str = "",
    max_wall_clock_seconds: float | None = None, overfit_check: bool = False,
) -> dict:
    """Launch a training run in an isolated subprocess from a bespoke ``model_source`` builder.

    The run's actual training body (dataset build, model forward/backward, checkpointing) executes
    in a separate OS process, not this one, a bug/OOM/hang in one run can't take down this
    process or any other concurrent run's process. Use monitor_training to monitor progress;
    it reads the run's own status/metrics from disk, not shared memory.

    Watching a launched run's metrics and deciding to stop a poorly performing one is the
    launching agent's own judgment call, made through cancel_training; the platform itself only
    stops a run objectively dead (two consecutive full training passes with no finite batch
    loss) or stagnant against its own validation metric (early stopping), never one merely
    performing worse than hoped. Early stopping needs a validation loader to have anything to
    watch: a run launched with none gets divergence as its only automatic stop, and is worth
    closer agent monitoring than a run early stopping can also catch.

    Args:
        config: Full training configuration dict with model_source, data, training sections.
        output_dir: Directory for checkpoints and logs. Empty defaults to the experiment store
            (``<project>/.tcip/experiments``, the same base the experiment records use); a
            relative path resolves against the platform state root, never the server process's cwd.
        resume_from: Optional path to a ``checkpoint_epoch_*.pt`` to resume from
            (restores model + optimizer + scheduler + scaler and continues).
        max_wall_clock_seconds: Optional hard timeout. If the training process hasn't exited on its
            own by then, it is terminated and the run marked failed with that reason, no
            cooperative grace period is attempted (a hung process isn't responding to cooperative
            signals). Omit for no timeout (the default).
        overfit_check: When True, runs the voluntary ``overfit_check`` diagnostic (twenty
            optimizer steps at the training tile edge, on the CPU, inside this synchronous call)
            on the same batch the contract proved, before the subprocess spawns, and records the
            result on the run's ``model_contract`` under ``overfit_check``. Never gates: a valid
            model can fail twenty steps on noise, so only the contract itself decides ``valid``.
            Default False, since the cost is one the agent elects per launch rather than pays on
            every one.
    """
    # The caller's config is stored twice, as the launch config and as the experiment's
    # snapshot, so what it holds is checked before either write.
    check_json_value(config, path="config")
    # smoke=True: build the model and run the correctness contract before spawning the training
    # subprocess, so a broken builder returns here instead of wasting a full audited run.
    validation = preflight_config(config, smoke=True, overfit=overfit_check)
    if not validation["valid"]:
        return {"error": "Invalid config", "issues": validation["issues"]}

    # GUI schema nests stages/mixed_precision/batch_size under ``training``, but the trainer reads them top-level; without this hoist a GUI-launched run silently trains the default single stage.
    # run_hyperparameter_search normalizes separately, inside _apply_hpo_params.
    from tcip_mcp.pipelines.schemas import normalize_train_config
    config = normalize_train_config(config)

    # The top-level key, never the smoke sub-report: overfit_check runs beside the contract's build, not inside it, on the same batch.
    # Already rendered by preflight_config for storage (a raw non-finite loss cannot cross the JSON-RPC boundary), so nothing renders it again here.
    rendered_overfit_report = validation.get("overfit_check")

    # Recorded on the copy above, never the caller's own config dict, so a launch never hands
    # back an argument it silently mutated.
    smoke_report = validation.get("smoke") or {}
    model_contract_record = {
        "subject": "the model as built at launch, before any training step",
        "gating": True,
        "batch_source": smoke_report.get("batch_source"),
        "dims": smoke_report.get("dims"),
        "issues": smoke_report.get("issues", []),
        "gradient_magnitudes": smoke_report.get("gradient_magnitudes"),
        "operating_point_knobs": smoke_report.get("operating_point_knobs"),
        "overfit_check": rendered_overfit_report,
    }
    check_json_value(model_contract_record, path="model_contract")
    config["model_contract"] = model_contract_record

    from tcip_mcp.experiments import experiments_dir
    from tcip_mcp.pipelines.training.run_registry import create_run
    from tcip_mcp.project_paths import resolve_output_path

    # Training artifacts (weights, tensorboard, metrics) live with the project the run belongs
    # to, same as its experiment record; only an absolute output_dir points anywhere else.
    output_dir = str(resolve_output_path(output_dir) if output_dir else experiments_dir())

    data_cfg = config.get("data", {})

    run = create_run(config, output_dir)
    # Nest each run's artifacts under its run_id. The GUI (and typical callers) pass a
    # shared base such as ``<project>/.tcip/experiments``; without nesting, sequential
    # runs write ``model_best.pt`` / TensorBoard events to the *same* flat directory and
    # clobber each other, violating experiment immutability. Nesting also gives the subprocess
    # a single directory to write into and the cancel sentinel to live in.
    run.output_dir = str(Path(output_dir) / run.run_id)

    # Auto-create experiment if not already tracked. Experiments are immutable:
    # reusing an id that already has a run would interleave metrics histories and
    # overwrite lineage/registry entries, so such relaunches get a fresh id.
    experiment_id = config.get("experiment_id") or run.run_id
    try:
        from tcip_mcp.experiments import update_status
        from tcip_mcp.pipelines.data.split_construction import dataset_identity

        # The dataset identity this run trains on, passed to the immutable lineage record; the
        # child recomputes it independently for split.json ("recompute-on-read" is its authority).
        ds_id, ds_fp = dataset_identity(data_cfg)
        experiment_id = _ensure_experiment(
            experiment_id, config, data_cfg.get("images_dir"), resume_from, run.run_id,
            output_dir=run.output_dir, dataset_id=ds_id, dataset_fingerprint=ds_fp,
        )
        # Thread the resolved id into the live config so the child's checkpoints carry it (the
        # envelope's ctx.save_checkpoint stamps it explicitly).
        config["experiment_id"] = experiment_id
        run.experiment_id = experiment_id
        update_status(experiment_id, "running")
    except Exception as exc:  # Experiment tracking is best-effort, but failures must be visible.
        logger.warning("Experiment tracking failed for %s: %s", experiment_id, exc)
        run.experiment_error = str(exc)

    # The child reads its own bootstrap config from here, independent of whether experiment
    # tracking above succeeded, a filesystem hiccup in .tcip/experiments degrades tracking (as it
    # always has) without also preventing the run from training at all. experiment_id is never
    # read from here (see subprocess_worker.py), only passed as the explicit CLI arg below,
    # because this file is written before config["experiment_id"] is guaranteed resolved in the
    # fresh-id-relaunch branch.
    store.replace(launch_config_key(run.output_dir), config)

    # Captured once, beside the child's environment snapshot: the watchdog below writes about
    # this run under the root it launched under, even if this process later adopts another.
    from tcip_mcp.project_paths import platform_state_root

    launch_root = platform_state_root()
    child_env = _child_env_for_launch(config)

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "tcip_mcp.pipelines.training.subprocess_worker",
            "--run-id", run.run_id,
            "--experiment-id", experiment_id,
            "--output-dir", run.output_dir,
            "--resume-from", resume_from,
        ],
        env=child_env,
    )
    run.pid = proc.pid

    if max_wall_clock_seconds is not None:
        _watch_wall_clock(proc, run, experiment_id, max_wall_clock_seconds, root=launch_root)

    # Launch TensorBoard for live monitoring
    tb_info = {}
    try:
        from tcip_mcp.pipelines.training.tensorboard_manager import launch_tensorboard
        tb_dir = str(Path(run.output_dir) / "tensorboard")
        tb_info = launch_tensorboard(tb_dir, run_id=run.run_id)
    except Exception:
        pass  # TensorBoard launch is best-effort

    return {
        "run_id": run.run_id,
        "experiment_id": experiment_id,
        "status": "launched",
        "output_dir": run.output_dir,
        "tensorboard": tb_info,
        "pid": proc.pid,
        "overfit_check": rendered_overfit_report,
    }


def _child_env_for_launch(config: dict) -> dict[str, str]:
    """Subprocess env for a launch: round-robin GPU pinning when the config names no
    explicit device, left untouched when it does. ``CUDA_VISIBLE_DEVICES`` remaps device
    *indices* inside the child, pinning it would ask an explicit ``device: "cuda:1"`` config for
    an ordinal invalid in the child's own remapped view, so pinning applies only to the unpinned
    case it's meant to spread out.

    Also propagates this process's own import search path via ``PYTHONPATH``, the child is a
    fresh interpreter with only sys.path's own defaults, not whatever got this process's bespoke
    ``model_source``/``training_source``/``dataset_source`` module importable in the first place
    (an editable install's extra path entries, a test runner's rootdir insertion, an agent's own
    working-directory convention). Without this, a bespoke module importable to the caller can
    become unimportable to the child purely because of the process boundary, a correctness gap,
    not just a convenience.
    """
    import os

    from tcip_mcp.pipelines.model_build import child_pythonpath

    env = dict(os.environ)
    env["PYTHONPATH"] = child_pythonpath()

    if config.get("device") or config.get("training", {}).get("device"):
        return env

    try:
        import torch
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        count = 0
    if count > 1:
        from tcip_mcp.pipelines.training.run_registry import _RUNS_LOCK
        with _RUNS_LOCK:
            idx = next(_gpu_round_robin) % count
        env["CUDA_VISIBLE_DEVICES"] = str(idx)
    return env


def _watch_wall_clock(proc: subprocess.Popen, run: Any, experiment_id: str,
                      timeout_seconds: float, *, root: Path | str) -> None:
    """Daemon watcher: hard-terminates ``proc`` if it outlives ``timeout_seconds`` and
    records the reason through the same status channel every other terminal state uses, never an
    in-memory-only mark, since ``monitor_training`` always defers to disk for a pid-bearing
    run and would otherwise never surface it. No cooperative grace period: a hung process isn't
    responding to cooperative signals, so this is a hard kill, not the cancel path.

    ``root`` is the platform root this run launched under, captured once at launch: this
    process may have since adopted a different project, and the write belongs to the run's
    own root regardless.
    """
    def _watch() -> None:
        try:
            proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            proc.terminate()
            reason = f"exceeded max_wall_clock_seconds ({timeout_seconds})"
            run.status = "failed"
            run.error = reason
            try:
                from tcip_mcp.experiments import update_status
                update_status(experiment_id, "failed", error=reason, root=root)
            except Exception:
                logger.warning("wall-clock timeout status update failed for %s",
                               experiment_id, exc_info=True)

    threading.Thread(target=_watch, daemon=True).start()


@mcp.tool()
@audited
def monitor_training(run_id: str | None = None, sweep_id: str | None = None) -> dict:
    """Check the status of a training run, or of a hyperparameter sweep.

    Exactly one of ``run_id`` and ``sweep_id`` names what to check; both or neither refuses by
    name. The two return different shapes.

    ``run_id``: reads the run's own status/metrics from disk whenever its training body runs in
    a subprocess, the in-memory record for a subprocess-delegated run is a launch-time
    placeholder only, since the subprocess mutates its own separate copy in its own process
    memory, or when this process never held the run in memory at all (a different process
    launched it). Returns ``{"run_id", "status", "epoch", "best_metric", "output_dir", "error",
    "tensorboard_url"}``.

    ``sweep_id``: reads the sweep's own manifest and trial directories from disk under this
    process's own pinned platform root, through :func:`read_sweep_from_disk`, the same reader
    ``routes.tuning``'s disk-only paths call, then layers the study result's own fields onto a
    completed sweep through :func:`enrich_with_study_result`, the same rule that route applies:
    an agent on a host with no browser open answers the same "how is this sweep doing" question
    the Tuning tab reads. This is a disk read only, so a sweep a live web session just launched
    over HTTP but has not yet written a manifest for reads as not found. Returns
    ``read_sweep_from_disk``'s own shape, enriched (``{"sweep_id", "status", "error", "result",
    "manifest", "relaunched_from", "has_manifest", "trials"}``), or ``{"error": ...}`` when no
    manifest exists or ``sweep_id`` would address a record outside the HPO store.

    Args:
        run_id: Training run identifier. Exactly one of ``run_id``/``sweep_id`` is required.
        sweep_id: Hyperparameter sweep identifier. Exactly one of ``run_id``/``sweep_id``.
    """
    if sweep_id is not None:
        if run_id is not None:
            return {"error": "exactly one of run_id or sweep_id is required, got "
                              f"run_id={run_id!r} sweep_id={sweep_id!r}"}
        try:
            disk_sweep = read_sweep_from_disk(sweep_id)
        except BadKey:
            return {"error": f"invalid sweep_id: {sweep_id}"}
        if disk_sweep is None:
            return {"error": f"sweep not found: {sweep_id}"}
        return enrich_with_study_result(disk_sweep, sweep_id)
    if run_id is None:
        return {"error": "exactly one of run_id or sweep_id is required, got "
                          "run_id=None sweep_id=None"}

    from tcip_mcp.pipelines.training.run_registry import get_run
    run = get_run(run_id)

    result: dict[str, Any] | None = None
    if run is None or run.pid is not None:
        from tcip_mcp.experiments import reconstruct_run_status
        disk = reconstruct_run_status(run_id, stale_seconds=TCIP_HEARTBEAT_STALE_SECONDS)
        if disk is not None:
            result = {
                "run_id": disk["run_id"],
                "status": disk["status"],
                "epoch": disk["current_epoch"],
                "best_metric": disk["best_metric"],
                "output_dir": disk["output_dir"],
                "error": disk.get("error"),
            }

    if result is None:
        if run is None:
            return {"error": f"Run not found: {run_id}"}
        result = {
            "run_id": run.run_id,
            "status": run.status,
            "epoch": run.current_epoch,
            "best_metric": run.best_metric,
            "output_dir": run.output_dir,
            "error": run.error or None,
        }

    # Check for running TensorBoard
    tb_url = None
    try:
        from tcip_mcp.pipelines.training.tensorboard_manager import _TB_PROCESSES
        entry = _TB_PROCESSES.get(run_id)
        if entry is not None and entry.proc.poll() is None:
            tb_url = f"http://localhost:{entry.port}"
    except Exception:
        pass
    result["tensorboard_url"] = tb_url
    return result


def _launched_training_runs(*, read_progress: bool) -> list[dict[str, Any]]:
    """Every launched training run this store holds a record for, reconstructed from disk.

    A record is a launched run when its config carries ``model_source`` and
    :func:`~tcip_mcp.experiments.is_launched` says so: a stamped ``run_id``, a state other than
    ``"created"``, or the ``metrics_logged`` marker, so a launch whose best-effort stamp or status
    write was lost still lists, and a pre-created experiment that never launched does not; the
    same predicate :func:`~tcip_mcp.experiments.compare_experiments` consults before deriving a
    heartbeat state at all. Rows come back sorted by
    experiment id (``experiment_ids_with_status``'s own order), each carrying ``external: True``.
    ``read_progress`` governs whether ``current_epoch`` costs a metrics-log read per record.
    Cost: one status read and one config read per experiment record on disk, plus, when
    ``read_progress`` is true, one metrics-log read per launched record.
    """
    from tcip_store import DecodeError

    from tcip_mcp.experiments import (
        config_key, experiment_ids_with_status, is_launched, reconstruct_from_status, status_key,
    )
    from tcip_mcp.pipelines.model_build import MODEL_SOURCE_KEY

    rows: list[dict[str, Any]] = []
    for experiment_id in experiment_ids_with_status():
        try:
            config = store.read(config_key(experiment_id), default={})
            status = store.read(status_key(experiment_id), default={})
        except DecodeError:
            # One unreadable record must not cost the caller the whole run listing.
            logger.warning("experiment %s has a member that does not decode", experiment_id,
                           exc_info=True)
            continue
        if not isinstance(config, dict) or not config.get(MODEL_SOURCE_KEY):
            continue  # not a training experiment (e.g. review-feedback lineage)
        if not is_launched(status):
            continue
        row = reconstruct_from_status(experiment_id, status, stale_seconds=TCIP_HEARTBEAT_STALE_SECONDS,
                                      read_progress=read_progress)
        row["external"] = True
        rows.append(row)
    return rows


def _all_training_runs(*, read_progress: bool) -> list[dict[str, Any]]:
    """This process's in-memory registry merged with every launched run's own disk record: the
    one implementation :func:`tcip_mcp.tools.experiment_tools.list_experiments` (with
    ``launched_only=True``) and :func:`inspect_compute_resources` both build on, so a
    subprocess-delegated run's real status is visible to both and neither reimplements the merge.

    A live in-memory entry (HPO trials excluded) wins by ``run_id`` over its own disk row: a
    ``pid``-bearing one takes the disk overlay for ``status``/``current_epoch``/``error`` and
    ``best_metric``/``best_metric_name`` (a subprocess-delegated run mutates its own separate
    copy on disk, so the parent-side in-memory record, ``best_metric`` included, is a stale
    launch-time placeholder past that point); a ``pid``-less one (every synchronous run) is
    reported from its own in-memory record, untouched. Both carry
    ``external: False`` and an ``experiment_id``: the row's own resolved field
    (``TrainRun.experiment_id``, set by ``launch_training`` once ``_ensure_experiment`` resolves
    it) when it has one, the disk overlay's own id only as a fallback for a row that has none.
    Rows: this process's own, in registry order, then the disk-only rows, sorted by experiment id.
    """
    from tcip_mcp.pipelines.training.run_registry import list_runs

    live = list_runs()
    disk = _launched_training_runs(read_progress=read_progress)
    disk_by_run_id = {r["run_id"]: r for r in disk}

    merged: list[dict[str, Any]] = []
    for r in live:
        row = dict(r)
        row["external"] = False
        overlay = disk_by_run_id.get(row["run_id"]) if row.get("pid") is not None else None
        if overlay is not None:
            row["status"] = overlay["status"]
            if overlay["current_epoch"] is not None:
                row["current_epoch"] = overlay["current_epoch"]
            if overlay.get("error"):
                row["error"] = overlay["error"]
            if overlay.get("best_metric_name") is not None:
                row["best_metric"] = overlay["best_metric"]
                row["best_metric_name"] = overlay["best_metric_name"]
            if not row.get("experiment_id"):
                row["experiment_id"] = overlay["experiment_id"]
        merged.append(row)

    live_run_ids = {r["run_id"] for r in live}
    disk_only = [r for r in disk if r["run_id"] not in live_run_ids]
    return merged + disk_only


def list_launchable_configs() -> list[dict]:
    """Every experiment in this project with a model source, as a row the config picker can
    start a run from: the id, the builder and task, the data it names, the subject, its derived
    state and its parent when it has one.

    Not agent-facing (not an ``@mcp.tool()``): a GUI-picker-specific projection over records the
    agent already has ``list_experiments``/``get_experiment`` for, never ``get_experiment``
    (which also reads the whole metrics log). Cost: ``list_experiments()``'s own one status
    read plus one config read per experiment record, and this function's own further read of
    that same config, plus one status read and one lineage read per experiment that carries a
    model source.

    State is ``derived_state``, gated by ``is_launched`` the identical way ``compare_experiments``
    and this platform's runs list already gate it, so a pristine never-launched config reads its
    recorded ``"created"`` rather than a heartbeat-derived ``"interrupted"`` implying a crash
    that never happened, and a crashed run reads the same way here as it does in the runs list.
    """
    from tcip_mcp.experiments import (
        config_key, derived_state, is_launched, lineage_key, list_experiments, read_member,
        status_key,
    )
    from tcip_mcp.pipelines.model_build import MODEL_SOURCE_KEY

    rows = []
    for exp in list_experiments():
        if not exp["has_model_source"]:
            continue
        experiment_id = exp["experiment_id"]
        config = read_member(config_key(experiment_id), {})
        lineage = read_member(lineage_key(experiment_id), {})
        status = read_member(status_key(experiment_id), {})
        model_source = config.get(MODEL_SOURCE_KEY) if isinstance(config, dict) else None
        data_cfg = config.get("data") if isinstance(config, dict) else None
        rows.append({
            "experiment_id": experiment_id,
            "builder": (model_source or {}).get("builder"),
            "task": (model_source or {}).get("task"),
            "images_dir": (data_cfg or {}).get("images_dir"),
            "subject": (data_cfg or {}).get("subject"),
            "created": exp["created"],
            "state": derived_state(status, TCIP_HEARTBEAT_STALE_SECONDS) if is_launched(status)
                     else status.get("state", exp["state"]),
            "parent_experiment": (lineage or {}).get("parent_experiment"),
        })
    return rows


def split_dir_identity(path: str) -> str:
    """The normalized identity of a split-manifest directory: case-folded, symlinks resolved
    (``os.path.normcase(str(Path(path).resolve()))``), so two differently spelled, cased or
    symlinked paths to the same directory compare equal. Shared by ``list_split_choices``
    (excluding a config's own binding from its candidates, deduping the rest) and the relaunch
    route (checking a client-picked spelling against what was actually offered), so a directory
    is never refused, or offered twice, over a spelling difference alone.
    """
    return os.path.normcase(str(Path(path).resolve()))


def list_split_choices(experiment_id: str) -> dict:
    """Every choice this config's own "Data" control offers a relaunch of ``experiment_id``: its
    stored data section as recorded, and every split manifest directory this project's own bound
    runs or the dataset's own ``splits`` directory hold, each compatibility-checked the identical
    way a launch itself would check it.

    Not agent-facing (not an ``@mcp.tool()``): a plain reader, importable by the agent's own
    scripts, wrapped by ``GET /api/training/configs/{experiment_id}/splits``. Never calls
    ``preflight_config`` (a demoted function that imports the picked config's builder and scans
    its labels): every check here is :func:`manifest_compatibility` over a manifest this reader read
    itself, through :func:`~tcip_mcp.tools.data_tools.read_split_manifest_dir_checked`, no second
    presence test anywhere. A candidate manifest is checked against the config
    :func:`candidate_config_with_manifest` builds (``data.split`` replaced wholesale,
    ``val_images_dir`` dropped), the identical shape the launch route builds, so an offer is
    exactly what would launch; "As recorded" is checked against the stored config unchanged
    (plus the directory-presence issues :func:`preflight_config` would raise, so a snapshot
    whose recorded directories moved shows the same words before Start that a launch would
    refuse it with).

    The listing is thin: the manifest directories other enumerable experiment configs in this
    project bound to (the picked config's own excluded, since it is "As recorded"), plus, when
    ``dataset_root_of(data.images_dir)`` resolves, that root's ``splits`` directory (offered only
    when something is actually recorded there directly, the ``draw_splits`` default materializes
    it, it does not always exist) and every directory one level under it holding a manifest
    (where ``freeze_split_manifest`` writes a frozen run's own drawn partition). The own-binding
    exclusion and the candidate dedupe compare each directory by :func:`split_dir_identity`
    (folding both case and symlinks), so a differently spelled, cased or symlinked path to the
    identical directory is never offered as if it were a second one.

    Returns ``{"error": ...}`` for an unknown ``experiment_id`` (the route's own 404). Otherwise:
    ``{"as_recorded": {"case": "bound"|"drawn", "line": str, "compatible": bool,
    "reason": str | None}, "manifests": [{"manifest_dir": str, "enabled": bool,
    "reason": str | None, "seed": int | None, "group_by": str | None, "train": int, "val": int,
    "calibration": int, "other_dates": int, "replaced_split_keys": list[str],
    "origin": dict | None}, ...]}``. ``origin`` is the manifest's own ``{"experiment_id",
    "frozen_at"}`` for a frozen manifest, ``None`` for a drawn one.
    ``replaced_split_keys`` names every recorded ``data.split`` key other than ``manifest_dir``
    choosing any offered partition drops, exactly what :func:`candidate_config_with_manifest`
    replaces wholesale, read from the stored config once and carried on every entry: the same
    set for every candidate, since it describes what the config's own recorded policy holds, not
    what a given candidate carries. The config a candidate launches never carries a seed
    (:func:`candidate_config_with_manifest` replaces ``data.split`` outright), so this listing
    says nothing about a redraw either; that is the launched config's own concern.
    A config naming no ``data.images_dir`` or ``data.labels_dir`` answers only ``as_recorded``
    (``manifests`` always empty; there is nothing to narrow a candidate manifest's counts to).
    """
    from tcip_mcp.dataset_layout import annotation_date, dataset_root_of
    from tcip_mcp.experiments import config_key, experiment_exists, experiment_ids_with_status, read_member
    from tcip_mcp.pipelines.data.splits import narrow_manifest_to_date
    from tcip_mcp.tools.data_tools import read_split_manifest_dir_checked

    if not experiment_exists(experiment_id):
        return {"error": f"Experiment not found: {experiment_id}"}
    config = read_member(config_key(experiment_id), {})
    config = config if isinstance(config, dict) else {}
    data_cfg = config.get("data")
    data_cfg = data_cfg if isinstance(data_cfg, dict) else {}
    split_cfg = data_cfg.get("split")
    split_cfg = split_cfg if isinstance(split_cfg, dict) else {}
    own_manifest_dir = split_cfg.get("manifest_dir")
    replaced_split_keys = sorted(
        k for k, v in split_cfg.items() if k != "manifest_dir" and v is not None
    )

    if own_manifest_dir:
        as_recorded = {"case": "bound", "line": "on the partition it bound",
                       "compatible": True, "reason": None}
        own_manifest, own_error = read_split_manifest_dir_checked(own_manifest_dir)
        if own_manifest is None:
            as_recorded["compatible"] = False
            as_recorded["reason"] = own_error or (
                f"no split manifest recorded under {own_manifest_dir}; run draw_splits first."
            )
        else:
            own_issues = manifest_compatibility(config, own_manifest, own_manifest_dir)
            if own_issues:
                as_recorded["compatible"] = False
                as_recorded["reason"] = "; ".join(own_issues)
    else:
        seed = split_cfg.get("seed", 42)
        as_recorded = {
            "case": "drawn",
            "line": f"draws its split again with seed {seed} over the labels as they are now",
            "compatible": True, "reason": None,
        }

    dir_issues = _data_dir_issues(data_cfg)
    if dir_issues:
        as_recorded["compatible"] = False
        combined_reason = list(dir_issues)
        prior_reason = as_recorded["reason"]
        if prior_reason:
            combined_reason.insert(0, str(prior_reason))
        as_recorded["reason"] = "; ".join(combined_reason)

    images_dir, labels_dir = data_cfg.get("images_dir"), data_cfg.get("labels_dir")
    if not images_dir or not labels_dir:
        return {"as_recorded": as_recorded, "manifests": []}

    own_norm = split_dir_identity(own_manifest_dir) if own_manifest_dir else None
    candidate_dirs: list[str] = []
    seen: set[str] = set()
    for other_id in experiment_ids_with_status():
        if other_id == experiment_id:
            # Its own manifest_dir is already own_manifest_dir, read once above; re-reading its
            # config here to derive the identical fact a second time is work this listing skips.
            continue
        other_config = read_member(config_key(other_id), {})
        if not isinstance(other_config, dict):
            continue
        other_data = other_config.get("data")
        other_data = other_data if isinstance(other_data, dict) else {}
        other_split = other_data.get("split")
        other_split = other_split if isinstance(other_split, dict) else {}
        candidate = other_split.get("manifest_dir")
        if not candidate:
            continue
        candidate_norm = split_dir_identity(candidate)
        if candidate_norm == own_norm or candidate_norm in seen:
            continue
        seen.add(candidate_norm)
        candidate_dirs.append(candidate)
    dataset_root = dataset_root_of(images_dir)
    if dataset_root is not None:
        default_dir = str(dataset_root / "splits")
        default_norm = split_dir_identity(default_dir)
        if default_norm != own_norm and default_norm not in seen:
            seen.add(default_norm)
            candidate_dirs.append(default_dir)
        # One level down: where freeze_split_manifest writes a frozen run's own partition.
        splits_dir = Path(default_dir)
        if splits_dir.is_dir():
            for sub in sorted(p for p in splits_dir.iterdir() if p.is_dir()):
                sub_norm = split_dir_identity(str(sub))
                if sub_norm == own_norm or sub_norm in seen:
                    continue
                seen.add(sub_norm)
                candidate_dirs.append(str(sub))

    run_date = annotation_date(labels_dir)
    manifests: list[dict] = []
    for candidate_dir in candidate_dirs:
        manifest, error_text = read_split_manifest_dir_checked(candidate_dir)
        if manifest is None:
            if error_text is None:
                continue  # nothing recorded there; not a real candidate
            manifests.append({
                "manifest_dir": candidate_dir, "enabled": False, "reason": error_text,
                "seed": None, "group_by": None, "train": 0, "val": 0, "calibration": 0,
                "other_dates": 0, "replaced_split_keys": replaced_split_keys, "origin": None,
            })
            continue
        candidate_config = candidate_config_with_manifest(config, candidate_dir)
        issues = manifest_compatibility(candidate_config, manifest, candidate_dir)
        narrowing = narrow_manifest_to_date(manifest, run_date)
        entry: dict = {
            "manifest_dir": candidate_dir, "seed": manifest.get("seed"),
            "group_by": manifest.get("group_by"), "train": len(narrowing.train_ids),
            "val": len(narrowing.val_ids), "calibration": len(narrowing.calibration_ids),
            "other_dates": narrowing.other_dates, "replaced_split_keys": replaced_split_keys,
            "origin": manifest.get("origin"),
        }
        if issues:
            entry["enabled"] = False
            entry["reason"] = "; ".join(issues)
        else:
            entry["enabled"] = True
            entry["reason"] = None
        manifests.append(entry)

    return {"as_recorded": as_recorded, "manifests": manifests}


@mcp.tool()
@audited
def cancel_training(run_id: str) -> dict:
    """Request graceful cancellation of a running training run.

    The trainer stops at the next batch/epoch boundary, still saves ``model_final.pt``
    (so partial progress is recoverable), and sets the run + its experiment to
    'cancelled'. Status updates asynchronously, so the returned status may still read
    'running' immediately after the request. A run whose divergence verdict lands first (two
    consecutive full training passes with no finite batch loss, checked ahead of cancellation
    at the same boundary) ends 'failed' instead, with no ``model_final.pt``.

    Args:
        run_id: Training run identifier (from launch_training).
    """
    from tcip_mcp.pipelines.training.run_registry import cancel_run, get_run
    if not cancel_run(run_id):
        return {"error": f"Run not found: {run_id}"}
    run = get_run(run_id)
    if run is not None:
        status = run.status
    else:
        # Cancelled via the disk fallback, this process never held the run locally, so
        # there's no in-memory status to read; reflect the same disk record cancel_run itself
        # resolved to write the sentinel, if it's still discoverable.
        from tcip_mcp.experiments import reconstruct_run_status
        disk = reconstruct_run_status(run_id, stale_seconds=TCIP_HEARTBEAT_STALE_SECONDS)
        status = disk["status"] if disk is not None else "running"
    return {"run_id": run_id, "status": status, "cancel_requested": True}


@audited
def inspect_compute_resources() -> dict:
    """Report the host's current compute headroom, a fact to reason with before launching
    another concurrent training/HPO run, not an enforced cap.

    Not an MCP tool: run through ``tcip inspect-compute-resources``, per the admission
    standard (packages/tcip-mcp/CLAUDE.md), while staying importable for its own tests. This
    platform doesn't cap memory/CPU per run (no portable, non-pinned way to do that across
    POSIX/Windows without guessing a number that's wrong on the next host); it gives you the
    real numbers and trusts you to judge whether another candidate run fits, the same way you'd
    judge any other CV-scientist tradeoff.

    Returns:
        ``cpu``: ``{logical_count, percent_used}``, ``percent_used`` is ``None`` without
            ``psutil`` installed.
        ``memory``: ``{total_bytes, available_bytes}``, both ``None`` without ``psutil``.
        ``gpus``: ``[{index, free_bytes, total_bytes}, ...]``, always populated when CUDA is
            available (``torch.cuda.mem_get_info``, no extra dependency); ``[]`` otherwise.
        ``active_training_runs``: count of every run whose derived state is ``"running"``, a
            heartbeat fresher than ``TCIP_HEARTBEAT_STALE_SECONDS`` (600s by default, so a live
            run whose epoch outlasts the window reads ``"interrupted"`` and is not counted),
            through :func:`_all_training_runs` with progress reads off: this process's own
            in-memory registry (no live process ever reads as stale) merged with every launched
            record on disk, one status read and one config read per disk record, no metrics-log
            read.
    """
    cpu: dict[str, Any] = {"logical_count": os.cpu_count(), "percent_used": None}
    memory: dict[str, Any] = {"total_bytes": None, "available_bytes": None}
    try:
        import psutil
        cpu["percent_used"] = psutil.cpu_percent(interval=0.1)
        vm = psutil.virtual_memory()
        memory["total_bytes"] = vm.total
        memory["available_bytes"] = vm.available
    except Exception:
        logger.info("psutil unavailable or failed; cpu/memory visibility degraded to None",
                   exc_info=True)

    gpus: list[dict[str, Any]] = []
    try:
        import torch
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                free_b, total_b = torch.cuda.mem_get_info(idx)
                gpus.append({"index": idx, "free_bytes": free_b, "total_bytes": total_b})
    except Exception:
        logger.info("GPU visibility unavailable", exc_info=True)

    active = sum(1 for r in _all_training_runs(read_progress=False) if r.get("status") == "running")

    return {"cpu": cpu, "memory": memory, "gpus": gpus, "active_training_runs": active}


# Keys _apply_hpo_params gives purpose-built handling (routed into nested optimizer/
# training structures train() reads unconditionally), tracking them by their own top-level
# name would be meaningless. Only the "else"-routed passthrough keys (the actual risk case:
# a swept axis that lands somewhere no consumer reads) are checked against consumption.
_HPO_KNOWN_KEYS = {"lr", "batch_size", "weight_decay"}


class _AccessTrackingConfig(dict):
    """Dict subclass recording which dotted paths are ever read via ``__getitem__``/``get``/
    ``__contains__``, installed on ``run.config`` for one HPO trial's dispatch, so
    ``unconsumed_params`` reflects genuine runtime access (did anything read this key during
    this trial), not a static comparison against ``train()``'s known key list, which would
    falsely flag a bespoke ``training_source``'s own legitimate custom sweep key.

    A nested dict value returned by ``__getitem__``/``get`` is itself wrapped the same way,
    sharing this instance's own ``accessed`` set under its own dotted prefix, so
    ``ctx.config["model_source"]["builder_kwargs"]["width"]`` records ``"model_source"``,
    ``"model_source.builder_kwargs"`` and ``"model_source.builder_kwargs.width"`` in one read, and
    a misspelled leaf under an otherwise-read block is reported by its own dotted name rather
    than being hidden behind the block it lives in.

    Real, stated limitations (never gates the run, warn-only, so a false positive costs a log
    line, not a failed trial): ``dict(cfg)``/``**cfg`` copies bypass the overrides entirely
    (CPython copies at the C level, and the copy is a plain dict with no wrapping of its own);
    whole-dict iteration (``.items()``/``.values()``/``.keys()``) isn't tracked per-key. A nested
    value ``_wrap`` returns is itself a freshly constructed ``_AccessTrackingConfig``, its own
    copy of that nested dict's items, never a live view over the original: a write through it
    (``cfg["model_source"]["builder_kwargs"]["width"] = 8``) lands on that throwaway copy and is
    never visible on ``cfg``. Only a write straight onto a config already held (``cfg["width"] =
    8``, or a reference to a nested block kept before further indexing) lands. ``dict`` offers no
    way to make a subclass instance alias another dict's own storage, so this only avoids
    surprise by being read, not fixed without replacing the class with a proxy that stops being a
    plain ``dict`` (breaking every consumer that relies on it being one).
    """

    def __init__(self, *args: Any, _prefix: str = "", _accessed: set[str] | None = None,
                **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._prefix = _prefix
        self.accessed: set[str] = set() if _accessed is None else _accessed

    def _dotted(self, key: Any) -> str:
        return f"{self._prefix}.{key}" if self._prefix else str(key)

    def _wrap(self, key: Any, value: Any) -> Any:
        if isinstance(value, dict) and not isinstance(value, _AccessTrackingConfig):
            return _AccessTrackingConfig(value, _prefix=self._dotted(key), _accessed=self.accessed)
        return value

    def __getitem__(self, key: Any) -> Any:
        self.accessed.add(self._dotted(key))
        return self._wrap(key, super().__getitem__(key))

    def get(self, key: Any, default: Any = None) -> Any:
        self.accessed.add(self._dotted(key))
        if not dict.__contains__(self, key):
            return default
        return self._wrap(key, super().__getitem__(key))

    def __contains__(self, key: Any) -> bool:
        self.accessed.add(self._dotted(key))
        return super().__contains__(key)


def hpo_root(output_dir: str = "", *, root: Path | str | None = None) -> Path:
    """Where HPO sweeps live: ``output_dir`` when the caller named one, else ``.tcip/hpo``
    under ``root`` (default: the platform state root). A relative ``output_dir`` resolves
    against the platform state root, never the server process's cwd.

    ``root`` lets a caller that already knows which project a sweep belongs to (its own
    registry entry's launch root) resolve its directory there, rather than under whatever
    root this process currently has pinned.

    The one resolver for that decision. Anything that has to find a sweep on disk (the
    Tuning routes included) calls this rather than rebuilding the same default.
    """
    from tcip_mcp.project_paths import platform_state_root, resolve_output_path

    if output_dir:
        return resolve_output_path(output_dir)
    base = Path(root) if root is not None else platform_state_root()
    return base / ".tcip" / "hpo"


def sweep_dir(study_name: str, output_dir: str = "", *, root: Path | str | None = None) -> Path:
    """One sweep's own directory: its manifest, its ``trial_<id>`` dirs, and (because Ray
    is handed ``storage_path=hpo_root`` and ``name=study_name``) Ray's experiment store."""
    return hpo_root(output_dir, root=root) / study_name


SWEEP_CANCEL_SENTINEL = ".sweep_cancel_requested"
"""The sweep-level cooperative-cancel stop file's name, written at a sweep's own root by
:func:`cancel_hyperparameter_search` and polled by ``run_hyperparameter_search``, ``_run_hpo_trial`` and the sweep
:class:`~tcip_mcp.pipelines.training.hpo.Stopper`. Distinct from the run-level
``run_registry.CANCEL_SENTINEL`` (written per trial directory): one name per protocol, since a
sweep-wide stop and one run's own stop answer different questions."""

_CANCEL_BEFORE_START_REASON = "cancelled before the sweep's first trial started"
_CANCEL_DURING_RUN_REASON = "the sweep was cancelled by request before it could finish"

_TRIAL_DIR_PREFIX = "trial_"

def sweep_heartbeat_seconds() -> float:
    """How often ``run_hyperparameter_search``'s driver thread restamps the sweep manifest's ``heartbeat`` while
    ``tune_search`` runs: :data:`TCIP_HEARTBEAT_STALE_SECONDS` read fresh on every call (not
    frozen at import) and divided by ten, so a live driver stamps ten times within one
    staleness window whatever that window is set to while the heartbeat loop is running.
    """
    return TCIP_HEARTBEAT_STALE_SECONDS / 10

_LAUNCHING_SWEEPS: dict[str, Path] = {}
_LAUNCHING_SWEEPS_LOCK = threading.Lock()
"""Every sweep a caller has minted a ``study_name`` for and is about to hand to ``run_hyperparameter_search``,
before that call's own manifest exists: the pre-manifest window in which a cancel request
would otherwise find nothing on disk and nothing in ``run_registry._RUNS`` to act on."""


def mark_sweep_launching(study_name: str, output_dir: str = "", *, root: Path | str | None = None) -> None:
    """Record that ``study_name`` is about to become a sweep at this resolved root, closing
    the window between a caller minting the id and ``run_hyperparameter_search`` writing its first manifest.

    ``run_hyperparameter_search`` itself discards the mark, in a ``finally`` around everything from its own
    entry through its first manifest write, so the mark is live for exactly that window; a
    caller that marks a study and never actually calls ``run_hyperparameter_search`` for it (a request that
    errored before starting the worker) leaves an entry this module cannot itself clean up,
    which is why the web relaunch route's worker also discards it, in its own ``finally``, on
    every exit that never reached ``run_hyperparameter_search`` (see :func:`discard_sweep_launching`).
    """
    resolved = sweep_dir(study_name, output_dir, root=root).resolve()
    with _LAUNCHING_SWEEPS_LOCK:
        _LAUNCHING_SWEEPS[study_name] = resolved


def discard_sweep_launching(study_name: str | None) -> None:
    """Drop ``study_name``'s pre-manifest mark, if it holds one. A no-op for ``None`` (no
    caller-supplied name) or a name nothing marked (an agent-launched sweep)."""
    if study_name is None:
        return
    with _LAUNCHING_SWEEPS_LOCK:
        _LAUNCHING_SWEEPS.pop(study_name, None)


def _sweep_launching(study_name: str, resolved_root: Path) -> bool:
    """Whether ``study_name`` is in its pre-manifest window right now, at ``resolved_root``.

    A mark recorded under a different resolved root names a study ``run_hyperparameter_search`` will never look
    for at this location, so it does not count as found here: the caller's own resolved sweep
    root is what must match the mark's, not the study name alone.
    """
    with _LAUNCHING_SWEEPS_LOCK:
        marked = _LAUNCHING_SWEEPS.get(study_name)
    return marked is not None and marked == resolved_root


def sweep_state(manifest: dict, *, stale_seconds: float, driver_live: bool = False) -> str:
    """The sweep's derived liveness, the one rule every Tuning listing row reports as its
    ``status`` instead of a manifest's own recorded value verbatim.

    ``driver_live`` is true only where a process can vouch for the driver directly (the
    Tuning route's own worker thread, still alive, for a sweep it launched): that beats the
    manifest's heartbeat outright, since a live thread proves the driver is running even in
    the instant before its next heartbeat write lands. Every other case reads through
    :func:`tcip_mcp.experiments.derived_state`, the training run's own heartbeat-freshness
    rule, over ``{"state": manifest["status"], "heartbeat": manifest["heartbeat"]}``: a
    sweep's four statuses (``running``/``completed``/``failed``/``cancelled``) are the same
    words a training run uses, so the one rule applies unchanged rather than a second,
    silently-drifting copy of it living here.
    """
    from tcip_mcp.experiments import _RECORDED_AS_DONE, derived_state

    status = manifest.get("status", "unknown") if isinstance(manifest, dict) else "unknown"
    if driver_live and status not in _RECORDED_AS_DONE:
        return "running"
    heartbeat = manifest.get("heartbeat") if isinstance(manifest, dict) else None
    return derived_state({"state": status, "heartbeat": heartbeat}, stale_seconds)


def _running_trial_dirs(sweep_root: Path) -> list[Path]:
    """Every ``trial_<id>`` directory under ``sweep_root`` that has not yet written its
    resolved-config record: ``_run_hpo_trial``'s own ``finally`` block writes that record
    exactly once, at the very end of the trial, success or failure, so its absence means the
    trial has not reached that point yet.

    Used by :func:`cancel_hyperparameter_search` to decide which trial directories still need the run-level
    cancel sentinel written into them; a sentinel dropped into a trial that already wrote its
    resolved config is inert. The sweep :class:`~tcip_mcp.pipelines.training.hpo.Stopper`'s own
    ``stop_all`` no longer reads this: a trial Ray killed outright may never reach its own
    ``finally``, so the stopper tracks Ray's own live-trial reports instead (see
    ``_build_sweep_stopper``).
    """
    if not sweep_root.is_dir():
        return []
    running = []
    for d in sorted(sweep_root.iterdir()):
        if not d.is_dir() or not d.name.startswith(_TRIAL_DIR_PREFIX):
            continue
        try:
            wrote_resolved_config = store.read(trial_config_key(sweep_root, d.name), default=None) is not None
        except DecodeError:
            # The record exists but will not decode: the trial wrote it, so it is not running.
            wrote_resolved_config = True
        if not wrote_resolved_config:
            running.append(d)
    return running


SWEEP_MANIFEST_STORE = "hpo_sweep_manifest"
register_store(
    StoreDescriptor(
        name=SWEEP_MANIFEST_STORE,
        kind="record",
        key_fields=("study_name", "document"),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="last_writer_wins",
        locator=RootedFileLocator(suffix=".json"),
    )
)

STUDY_RESULT_STORE = "hpo_study_result"
register_store(
    StoreDescriptor(
        name=STUDY_RESULT_STORE,
        kind="record",
        key_fields=("study_name",),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="last_writer_wins",
        locator=RootedFileLocator(suffix=".json"),
    )
)

TRIAL_CONFIG_STORE = "hpo_trial_config"
register_store(
    StoreDescriptor(
        name=TRIAL_CONFIG_STORE,
        kind="record",
        key_fields=("trial", "document"),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="last_writer_wins",
        locator=RootedFileLocator(suffix=".json"),
    )
)

TRIAL_METRICS_STORE = "hpo_trial_metrics"
register_store(
    StoreDescriptor(
        name=TRIAL_METRICS_STORE,
        kind="log",
        key_fields=("trial", "document"),
        frozen=True,
        codec=LOG_JSON,
        locator=RootedFileLocator(suffix=".jsonl"),
    )
)


def _sweep_name(study_name: str) -> str:
    """``study_name`` once it is known to name one sweep and not a path through the store.

    A sweep name reaches this from an HTTP path segment, so a separator, a drive letter or a
    parent reference is refused here rather than resolved into a record somewhere else.
    """
    if PureWindowsPath(study_name).name != study_name or study_name == "..":
        raise BadKey(
            f"sweep name {study_name!r} is not a single name: a name carrying a path "
            "separator, a drive or a parent reference would address a record outside the "
            "HPO store"
        )
    return study_name


def sweep_manifest_key(
    study_name: str, output_dir: str = "", *, root: Path | str | None = None
) -> Key:
    """The manifest a sweep is listed and read back from.

    Keyed off the HPO root, the scope every sweep-level record hangs off, and declared here
    because ``hpo_root``/``sweep_dir`` already answer where a sweep lives.

    Two writers share this record. ``run_hyperparameter_search`` holds the manifest in memory for the whole sweep
    and replaces the whole document at each state change; it re-derives ``cancel_requested`` from
    the sweep's own stop file on every one of those writes, so it never overwrites a cancel
    ``cancel_hyperparameter_search`` recorded in between. ``cancel_hyperparameter_search`` itself, reached from a different request
    (a cancel this process is not running the sweep for), read-modify-writes only
    ``cancel_requested`` through the store's compare-and-set (``read_versioned`` plus
    ``replace(..., expect=version)``) and never over a manifest already in a terminal status, so a
    ``run_hyperparameter_search`` write racing ahead of it is never reverted back to a stale ``"running"``.
    ``concurrency="last_writer_wins"`` names the store's own policy; the two writers avoid
    clobbering each other by these rules, not by the store enforcing one.
    """
    return Key(SWEEP_MANIFEST_STORE, str(hpo_root(output_dir, root=root).resolve()),
               (_sweep_name(study_name), "manifest"))


STUDY_RESULT_FIELDS = ("all_trials", "search_alg", "scheduler", "warm_start", "baseline_params")
"""The study result's own fields, absent from the manifest's completion projection."""


def study_result_key(
    study_name: str, output_dir: str = "", *, root: Path | str | None = None
) -> Key:
    """A finished sweep's result document, beside the sweep's own directory.

    ``root`` mirrors :func:`sweep_manifest_key`'s: a caller that already knows which root a
    sweep launched under resolves the record there rather than under whatever root this
    process currently has pinned.

    ``last_writer_wins``: written once, when the sweep ends, from the result it returns.
    """
    return Key(STUDY_RESULT_STORE, str(hpo_root(output_dir, root=root).resolve()),
               (_sweep_name(study_name),))


def _trial_name(trial_dir_name: str) -> str:
    """``trial_dir_name`` once it is known to name one trial and not a path through the sweep.

    A trial name reaches this from an HTTP path segment, so a separator, a drive letter or a
    parent reference is refused here rather than resolved into a record somewhere else.
    """
    if PureWindowsPath(trial_dir_name).name != trial_dir_name or trial_dir_name == "..":
        raise BadKey(
            f"trial name {trial_dir_name!r} is not a single name: a name carrying a path "
            "separator, a drive or a parent reference would address a record outside the sweep"
        )
    return trial_dir_name


def trial_config_key(sweep_root: Path | str, trial_dir_name: str) -> Key:
    """The point one trial actually trained at: its merged config plus the sampled params.

    Scoped to the sweep rather than to the HPO root, because a trial belongs to its sweep and
    not to the store the sweeps sit in. ``last_writer_wins``: one trial process writes its own
    document once, when the trial finishes.
    """
    return Key(TRIAL_CONFIG_STORE, str(Path(sweep_root).resolve()),
               (_trial_name(trial_dir_name), "resolved_config"))


def trial_metrics_key(sweep_root: Path | str, trial_dir_name: str) -> Key:
    """One trial's epoch-by-epoch metrics, one entry per row, append only.

    A trial has no experiment record, so its rows belong to the sweep rather than to the
    experiment metrics log: same scope as :func:`trial_config_key`, which is the directory the
    Tuning view reads a trial back from.
    """
    return Key(TRIAL_METRICS_STORE, str(Path(sweep_root).resolve()),
               (_trial_name(trial_dir_name), "metrics"))


def trial_metrics_key_for_dir(trial_dir: Path | str) -> Key:
    """The metrics log of the trial that writes into ``trial_dir``.

    What a trainer holds is its own output directory, not the sweep it belongs to, so the
    split into (sweep root, trial name) is made here rather than at each caller.
    """
    path = Path(trial_dir).resolve()
    return trial_metrics_key(path.parent, path.name)


def log_holds_anything(page: Any) -> bool:
    """Whether a metrics log holds anything at all: rows, a torn tail, undecodable bytes, or
    entries at a schema_version this reader does not accept.

    One predicate: ``read_sweep_from_disk``'s per-trial ``has_metrics`` flag and
    ``routes.tuning.get_trial_metrics``'s own ``exists`` answer share this, rather than each
    restating what a log with nothing readable back looks like.
    """
    return bool(page.records or page.torn_tail or page.corrupt or page.version_refused)


def enrich_with_study_result(
    response: dict[str, Any], sweep_id: str, *, root: Path | str | None = None
) -> dict[str, Any]:
    """Layer the study result's own fields onto ``response["result"]`` for a completed sweep,
    read through the store and never fabricated: a sweep whose study result is absent (or
    already carries these fields, the common case for a sweep this process just ran) is served
    exactly as it already was.

    The one place this rule is written: ``read_sweep_from_disk`` answers the manifest-and-trials
    question only, and both ``routes.tuning.get_sweep``'s disk branch and
    ``monitor_training(sweep_id=)`` call this afterward on a completed sweep rather than each
    layering the study result on their own.
    """
    if response.get("status") != "completed":
        return response
    result = response.get("result") or {}
    if "all_trials" in result:
        return response
    try:
        key = study_result_key(sweep_id, root=root)
    except BadKey:
        return response
    try:
        study_result = store.read(key, default=None)
    except DecodeError:
        logger.warning("the study result for sweep %s does not decode", sweep_id, exc_info=True)
        study_result = None
    if not isinstance(study_result, dict):
        return response
    response["result"] = {
        **result,
        **{k: study_result[k] for k in STUDY_RESULT_FIELDS if k in study_result},
    }
    return response


def read_sweep_from_disk(sweep_id: str, *, root: Path | str | None = None) -> dict[str, Any] | None:
    """One sweep's manifest-derived summary plus every trial directory it has produced, read
    from the sweep's own store records alone, with no in-memory job registry consulted.

    Returns ``None`` when no manifest exists under ``root`` (the current platform root when
    ``root`` is ``None``): the "sweep not found" case ``routes.tuning.get_sweep``'s disk branch
    and ``routes.tuning.list_trials`` already answer this way, and ``monitor_training``'s own
    caller answers the same. Otherwise: ``{"sweep_id", "status", "error", "result", "manifest",
    "relaunched_from", "has_manifest": True, "trials"}``. ``status`` is the derived liveness
    (:func:`sweep_state`, ``driver_live=False``: no process reading from disk alone can vouch
    for a sweep's driver). ``result`` is the manifest's own, exactly as written: layering the
    study result's own fields onto a completed sweep's result is :func:`enrich_with_study_result`'s
    own separate question, so a caller that only wants the manifest and trials never pays for a
    study-result read it does not need. ``trials`` is one entry per ``trial_<id>`` directory
    under the sweep's own root: its resolved params, unconsumed params, and whether it has
    logged any metrics yet (:func:`log_holds_anything`).

    This is the one reader ``routes.tuning``'s disk-only paths and ``monitor_training(sweep_id=)``
    both call rather than each re-implementing it: the web route's own live/jobstore branch (an
    in-memory job this process is still running) is a different question this function does not
    answer, and stays the route's own code.
    """
    from tcip_store import read_log

    # BadKey from an invalid sweep_id propagates uncaught, matching cancel_hyperparameter_search.
    manifest_key = sweep_manifest_key(sweep_id, root=root)
    try:
        manifest = store.read(manifest_key, default=None)
    except DecodeError:
        logger.warning("the manifest for sweep %s does not decode", sweep_id, exc_info=True)
        manifest = None
    if not isinstance(manifest, dict):
        return None

    result = manifest.get("result") or {}
    status = sweep_state(manifest, stale_seconds=TCIP_HEARTBEAT_STALE_SECONDS, driver_live=False)

    trials: list[dict[str, Any]] = []
    sweep_directory = sweep_dir(sweep_id, root=root)
    if sweep_directory.is_dir():
        for d in sorted(sweep_directory.iterdir()):
            if not d.is_dir() or not d.name.startswith(_TRIAL_DIR_PREFIX):
                continue
            try:
                resolved = store.read(trial_config_key(sweep_directory, d.name), default={})
            except DecodeError:
                logger.warning("the resolved config for %s does not decode", d.name, exc_info=True)
                resolved = {}
            if not isinstance(resolved, dict):
                resolved = {}
            page = read_log(trial_metrics_key(sweep_directory, d.name))
            trials.append({
                "trial_id": d.name[len(_TRIAL_DIR_PREFIX):],
                "has_metrics": log_holds_anything(page),
                "params": resolved.get("trial_params") or {},
                "unconsumed_params": resolved.get("unconsumed_params") or [],
            })

    return {
        "sweep_id": manifest.get("study_name", sweep_id),
        "status": status,
        "error": manifest.get("error"),
        "result": result,
        "manifest": manifest,
        "relaunched_from": manifest.get("relaunched_from"),
        "has_manifest": True,
        "trials": trials,
    }


def _run_hpo_trial(config: dict, report, base_config: dict, trial_dir: str) -> None:
    """Train one HPO trial and ``report`` its resolved selection metric, in whatever direction
    that metric's own declaration says is better (``evaluation.HIGHER_IS_BETTER_BY_METRIC``, via
    :func:`~tcip_mcp.pipelines.training.generic_trainer.resolve_selection_metric`), never a fixed
    minimize convention.

    ``report(value)`` feeds the Ray Tune searcher/scheduler; call it each epoch (so a scheduler
    can prune) and once at the end with the best value this trial actually reached (Tune's default
    ``get_best_result`` scope reads only the last value each trial reported, so the run's best
    epoch would otherwise be lost behind a worse later one). A trial that never reports a real
    value, before training starts or on any failure, and a trial whose run ended ``"failed"`` or
    ``"cancelled"`` (even one that reported a real value from an epoch before it ended), both
    report the losing side of its own direction as that final value instead of a real number, so
    neither a trial with nothing to say, nor a config that killed its own run, nor a run a
    sweep-wide cancel cut short mid-training, can outrank a config that merely scored worse.
    Trials train under the final run's
    regime, same augmentation, imbalance handling, and dispatch: a ``training_source`` in
    ``base_config`` actually runs under that loop here too, not always the stock trainer, or the
    selected hyperparameters won't transfer.
    """
    merged = _apply_hpo_params(base_config, config)

    from tcip_mcp.pipelines.training.envelope import TrainContext, dispatch_train_body
    from tcip_mcp.pipelines.training.evaluation import HIGHER_IS_BETTER_BY_METRIC
    from tcip_mcp.pipelines.training.generic_trainer import (
        _improves, resolve_selection_metric, seeded_loader_kwargs,
        stamp_effective_data_geometry,
    )
    from tcip_mcp.pipelines.training.collation import task_collate
    from tcip_mcp.pipelines.training.run_registry import create_run
    from tcip_mcp.pipelines.data.samplers import build_sampler
    from tcip_mcp.pipelines.data.split_construction import auto_train_val
    from tcip_mcp.pipelines.model_build import MODEL_SOURCE_KEY
    from tcip_mcp.pipelines.schemas import evaluation_section
    from torch.utils.data import DataLoader

    model_source = merged.get(MODEL_SOURCE_KEY)
    # setdefault, not get: the geometry stamp below mutates this dict and must land in the
    # resolved-config snapshot written from merged.
    data_cfg = merged.setdefault("data", {})
    train_cfg = merged.get("training", {})
    task = (model_source.get("task") if model_source else None) or data_cfg.get("task", "detection")
    eval_cfg = evaluation_section(merged)
    try:
        higher_is_better = HIGHER_IS_BETTER_BY_METRIC[resolve_selection_metric(
            task, eval_cfg.get("trait"), eval_cfg.get("selection_metric"))]
    except Exception:
        # Undeclared direction, an unregistered trait, or any other resolution failure; the
        # trial fails below either way, this only decides which sentinel that failure reports.
        higher_is_better = False
    losing_side = float("-inf") if higher_is_better else float("inf")

    # A sweep-wide cancel already requested: report the losing side without training, so every
    # trial Ray still schedules after the request ends at once.
    if (Path(trial_dir).parent / SWEEP_CANCEL_SENTINEL).exists():
        report(losing_side)
        return

    if not model_source:
        report(losing_side)
        return

    # Track which top-level keys the trial reads, so an unconsumed swept param is caught by
    # observation, not a whitelist that would forbid a bespoke training_source's own axes.
    tracked_config = _AccessTrackingConfig(merged)
    run = create_run(tracked_config, trial_dir, origin="hpo_trial")  # kept off the Training tab

    # The best value this trial has actually reported, in the resolved direction; call_report is
    # what every reporting path below goes through, so this is the one place that tracks it.
    best = {"value": losing_side}

    def call_report(value: float) -> None:
        value = float(value)
        if _improves(value, best["value"], higher_is_better=higher_is_better):
            best["value"] = value
        report(value)

    try:
        transforms = None
        aug_cfg = merged.get("augmentation", {})
        if aug_cfg:
            from tcip_mcp.pipelines.data.augmentations import build_augmentation
            transforms = build_augmentation(aug_cfg)

        # Auto-val gives the val_loader that the composite objective / the scheduler need.
        train_ds, val_ds, _label_digests = auto_train_val(task, data_cfg, transforms)
        # Stamped before training so a pruned/failed trial's resolved-config snapshot still
        # records the geometry the trial actually trained on.
        stamp_effective_data_geometry(data_cfg, train_ds)
        batch_size = train_cfg.get("batch_size", config.get("batch_size", 4))
        num_workers = train_cfg.get("num_workers", 0)
        # Built after the loader context is known: a sampler whose read order depends on the
        # worker regime and batching consumes both.
        sampler = build_sampler(merged.get("sampler", "random"), train_ds,
                                num_workers=num_workers, batch_size=batch_size)
        # run.config's seed is create_run-resolved (auto-drawn if base_config left it unset),
        # read it off run.config, not merged, so the loader is seeded with the value actually used.
        loader_kwargs = seeded_loader_kwargs(run.config.get("seed"), num_workers=num_workers)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=(sampler is None),
            sampler=sampler, collate_fn=task_collate(task), num_workers=num_workers,
            **loader_kwargs,
        )
        val_loader = None
        if val_ds is not None:
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False,
                collate_fn=task_collate(task), num_workers=num_workers,
                **loader_kwargs,
            )

        def epoch_cb(epoch: int, metrics: dict) -> None:
            # resolve_selection_metric governs which key decides checkpoint choice once
            # evaluation.trait/selection_metric are set; prefer it over the raw composite.
            value = metrics.get("selection", metrics.get("val_objective", metrics.get("val_loss")))
            if value is not None:
                call_report(value)

        # Same training_source-or-default_train dispatch the full envelope uses; experiment_id=None
        # isolates a trial from the registry, and trial_report feeds a bespoke loop's own progress.
        ctx = TrainContext(run=run, train_loader=train_loader, val_loader=val_loader, task=task,
                           experiment_id=None, epoch_hook=epoch_cb, trial_report=call_report)
        dispatch_train_body(ctx)
        # A diverged or cancelled run reports the losing side, never what it reported before ending.
        if run.status in ("failed", "cancelled"):
            report(losing_side)
        else:
            report(best["value"])  # the trial's best reported value, or the losing side if none
    except Exception as e:
        logger.warning("HPO trial failed: %s", e)
        report(losing_side)
    finally:
        # Surface any swept param no consumer touched (warn-only); a dotted key is consumed only
        # by its own full dotted path being read, never by an ancestor block being read.
        swept = set(config.keys()) - _HPO_KNOWN_KEYS
        unconsumed = sorted(key for key in swept if key not in tracked_config.accessed)
        try:
            # trial_params is the sampled point itself, the only record of which axes this
            # sweep actually varied (the merged config cannot say that).
            trial_path = Path(trial_dir)
            # merged never gets create_run's drawn/pinned seed (tracked_config is a separate
            # dict); read it back the same way create_run/train() resolve it.
            seed = run.config.get("seed", run.config.get("training", {}).get("seed"))
            store.replace(trial_config_key(trial_path.parent, trial_path.name),
                          {**merged, "trial_params": dict(config),
                           "unconsumed_params": unconsumed, "seed": seed})
        except (OSError, StoreError):
            logger.warning("could not persist the resolved config for %s", trial_dir, exc_info=True)
        if unconsumed:
            logger.warning(
                "HPO trial %s: swept params %s were never read by the training body, check "
                "_apply_hpo_params' routing, or (for a bespoke training_source) confirm the "
                "loop actually reads them from ctx.config.", trial_dir, unconsumed)


@mcp.tool()
@audited
def run_hyperparameter_search(
    base_config: dict,
    param_space: dict | None = None,
    n_trials: int = 5,
    output_dir: str = "",
    search_alg: str = "random",
    scheduler: str = "asha",
    grace_period: int = 5,
    reduction_factor: int = 3,
    warm_start: bool = False,
    baseline_params: dict | None = None,
    max_concurrent: int = 1,
    resources_per_trial: dict | None = None,
    study_name: str | None = None,
    auto_tensorboard: bool = True,
    split_draws: int = 1,
    split_draw_seeds: list[int] | None = None,
    *,
    relaunched_from: str | None = None,
) -> dict:
    """Run hyperparameter optimization on Ray Tune, training each trial for real.

    The search *algorithm* and trial *scheduler* are yours to choose per task/data, pick
    from what is installed on this machine (call the ``hpo`` module's ``available_search_algs``
    / ``available_schedulers`` for the live list); the defaults below are a sane starting
    point, not a recipe:
      - ``search_alg``: ``random``/``grid`` (native), or a backend, ``optuna``, ``bayesopt``,
        ``hyperopt``, ``nevergrad``, ``ax``.
      - ``scheduler``: ``asha`` (async HyperBand), ``hyperband``, ``pbt``, ``median``, or
        ``none`` to run every trial to completion.

    Trials optimize ``base_config``'s own resolved selection metric, in whatever direction that
    metric's declaration says is better (``evaluation.HIGHER_IS_BETTER_BY_METRIC``, resolved once
    for the whole sweep via ``resolve_selection_metric``), not a fixed minimize convention; each
    trains under the base config's regime so the chosen hyperparameters transfer to
    ``launch_training``.

    Everything one sweep writes lands under ``<output_dir or .tcip/hpo>/<study_name>/``: a
    ``manifest.json`` stamped ``running`` before the first trial starts (so a sweep is
    visible while it runs, not only once it ends) and updated when the sweep ends, one
    ``trial_<id>/`` directory per trial, and Ray's own experiment store (also the
    TensorBoard logdir). The full result is written alongside as ``<study_name>.json``. The
    manifest also carries every argument this call resolved (``base_config`` and the resolved
    ``param_space`` included), so a relaunch can replay it exactly from the manifest alone.
    The manifest's ``heartbeat`` is restamped every :func:`sweep_heartbeat_seconds` from a
    daemon thread for as long as the search runs, stopped and joined before any terminal
    write, so a listing reading the manifest mid-sweep can tell a live driver from a dead one
    (see ``sweep_state``); a restamp a store or OS error interrupts costs one beat, not the
    rest of the sweep. A caller that calls :func:`mark_sweep_launching` for ``study_name``
    before this call keeps a cancel reachable in the window before the first manifest write
    (see :func:`cancel_hyperparameter_search`); this call discards that mark itself once it no longer needs it.

    Refuses (``{"error": ..., "issues": [...]}``, nothing minted) an unimportable builder or
    training source, or a config with no ``data`` section, at every point the search space
    could resolve a trial's config to (``base_config`` with a sampled point applied, not only
    the first). Also refuses a ``param_space`` axis whose sampled points would resolve to a
    different selection metric or ranking direction than ``base_config``'s own resolution,
    Ray's Tuner taking only one fixed metric/mode for the whole sweep: this catches a dotted or
    nested-dict axis naming ``selection_metric`` directly, and an axis (``model_source.task``,
    in particular) that changes the metric's own task-derived default with no
    ``selection_metric`` key in sight. ``cancel_hyperparameter_search`` requested against this study before or
    during the run instead ends the sweep ``{"status": "cancelled", ...}``, the manifest
    recording the same, rather than a completed result.

    Args:
        base_config: Base training config each trial modifies.
        param_space: Param-space dict (see ``hpo.get_default_space``); default when omitted.
            Every axis is checked against ``base_config``'s own resolved selection metric and
            direction (see above); an axis that would disagree at any sampled point is refused
            rather than minted.
        n_trials: Number of trials.
        output_dir: Base output directory for trial results (defaults under ``.tcip/hpo``).
        search_alg: Search algorithm, see the list above; call ``hpo.available_search_algs()``
            for what's actually installed on this box.
        scheduler: Trial scheduler, see the list above.
        grace_period: Minimum epochs before a halving scheduler (``asha``/``hyperband``) can
            stop a trial early.
        reduction_factor: Halving factor for ``asha``/``hyperband`` (fraction of trials kept
            at each rung).
        warm_start: Seed the search with ``baseline_params`` as a known-good starting point.
        baseline_params: Hyperparameter values to seed the search with when ``warm_start=True``.
        max_concurrent: Trials to run at once (default 1, safe for single-GPU training).
        resources_per_trial: Ray resource request per trial, omit to derive one from the
            host's real GPU count and ``max_concurrent`` (see ``hpo._default_trial_resources``);
            an explicit value always wins over the derivation.
        study_name: The sweep's id, for a caller (the Tuning route's launch) that already
            minted one and must have its own registry entry, manifest and every sweep route
            agree on it; omitted mints one the way this always has.
        auto_tensorboard: Launch a TensorBoard over the sweep root once it finishes. The
            Tuning route's launch passes ``False``: it serves its own per-sweep TensorBoard
            view on demand, so leaving this on there would run a second, unaddressable
            TensorBoard process over the same trials.
        relaunched_from: The sweep this one replays, for a caller (the Tuning route's relaunch)
            that started it from another sweep's own recorded manifest; recorded on this
            sweep's manifest so a listing can show the fork, ``None`` when this sweep was not
            a relaunch. Refused when it names no sweep manifest under this resolved root.
        split_draws: Above 1, adds ``data.split.seed`` to the search space as a grid over
            ``split_draw_seeds`` (default: the base config's own ``data.split.seed``, else 42,
            plus the draw index), paired with every sampled point through Ray's own
            ``BasicVariantGenerator(constant_grid_search=True)`` so each point trains once per
            seed, a blocked comparison of the split's own sensitivity. ``base_config`` bound to
            a split manifest is admitted, not refused: its own copy gains
            ``data.split.redraw_within_manifest: true``, defaulting ``data.split.seed`` to 42
            when the bound config carries none, the same default an unset-seed drawn config
            uses, so every trial redraws train and val inside the manifest's own train-plus-val
            members at its own seed, calibration untouched, rather than training every trial on
            the manifest's one recorded partition; refused before minting when the manifest's
            own train-plus-val members for the base config's date resolve to fewer than two
            foreground groups (a redraw could only starve a side). Otherwise refused, before
            minting the sweep, when
            ``base_config`` names ``data.val_images_dir`` (nothing to redraw), ``data.auto_val``
            is off or ``task`` sits outside the drawn path's own tasks, ``search_alg`` is not a
            native one (``random``/``grid``/``variant_generator``: only the native generator
            pairs a grid axis), ``scheduler`` is not ``none`` (a pruned draw is not comparable
            with a completed one), ``split_draw_seeds`` is given at a length other than
            ``split_draws``, ``warm_start``'s ``baseline_params`` names ``data.split.seed``
            (Ray's preset-variant pinning would pin every draw to one seed instead of pairing
            the grid), ``param_space`` already sweeps ``data.split.seed`` itself, or
            ``param_space`` sweeps any other ``data.*`` axis (a second data axis would change
            what a point admits or how it draws, so draw ``k`` would no longer be the same
            partition for every point). The result groups trials by point (params minus the
            seed) and chooses the best by mean over each point's draws; see
            ``result["best_value_spread"]``, and every point's own block at
            ``result["split_sensitivity"]`` beside ``result["n_points"]`` (planned points) and
            ``result["split_draws"]``, both mirrored onto the sweep manifest's own ``result``.
            1 (the default) changes nothing.
        split_draw_seeds: The seeds ``split_draws`` pairs with every sampled point, one per
            draw; omit for the derived default (see ``split_draws``).
    """
    from tcip_mcp.pipelines.training.hpo import SPLIT_DRAW_SEED_KEY, tune_search, get_default_space

    if param_space is None:
        param_space = get_default_space()

    # Everything through the first manifest write sits in this try/finally, so a caller's
    # mark_sweep_launching entry for study_name is discarded on every exit, refusal included.
    try:
        # Both reach a stored record: the space into the sweep manifest, the base config into
        # every trial's resolved config once a sampled point is applied to it.
        check_json_value(param_space, path="param_space")
        check_json_value(base_config, path="base_config")

        if relaunched_from is not None:
            try:
                source_exists = store.read(
                    sweep_manifest_key(relaunched_from, output_dir), default=None) is not None
            except BadKey:
                source_exists = False
            if not source_exists:
                return {"error": f"relaunched_from names no sweep manifest under this root: "
                                  f"{relaunched_from!r}", "issues": []}

        # A bound base_config admitted to split_draws redraws inside its manifest from here on.
        base_config = _base_config_for_split_draws(base_config, split_draws)

        # Checked ahead of preflight, so its own reason is what a bound/val_images_dir/task
        # refusal reads as, not whatever preflight would have hit first.
        hpo_task = _resolved_task(base_config)
        draws_refusal = _split_draws_refusal(
            base_config, param_space, hpo_task, search_alg, scheduler,
            split_draws, split_draw_seeds, warm_start, baseline_params)
        if draws_refusal is not None:
            return {"error": draws_refusal, "issues": []}

        from tcip_mcp.pipelines.training.evaluation import HIGHER_IS_BETTER_BY_METRIC
        from tcip_mcp.pipelines.training.generic_trainer import resolve_selection_metric
        from tcip_mcp.pipelines.schemas import evaluation_section

        # Ray forbids setting metric/mode anywhere but the Tuner, so the direction is resolved
        # once here, from base_config; every point below is checked against it the same way.
        hpo_eval_cfg = evaluation_section(base_config)
        try:
            hpo_metric = resolve_selection_metric(
                hpo_task, hpo_eval_cfg.get("trait"), hpo_eval_cfg.get("selection_metric"))
        except ValueError as exc:
            return {"error": str(exc), "issues": []}
        hpo_mode = "max" if HIGHER_IS_BETTER_BY_METRIC[hpo_metric] else "min"

        axis_conflict = _selection_metric_axis_conflict(
            base_config, param_space, hpo_task, hpo_metric, hpo_mode)
        if axis_conflict is not None:
            return {"error": axis_conflict, "issues": []}

        # Structural preflight over every point the search space could resolve a trial's
        # builder or data section to, not only the first sampled corner.
        for label, point in _preflight_points(param_space):
            try:
                preflight_cfg = _apply_hpo_params(base_config, point)
            except ValueError as exc:
                return {"error": f"the sweep's base config fails preflight at {label}: {exc}",
                        "issues": []}
            preflight = preflight_config(preflight_cfg)
            if not preflight["valid"]:
                return {"error": f"the sweep's base config fails preflight at {label}",
                        "issues": preflight["issues"]}

        search_param_space = param_space
        resolved_draw_seeds: list[int] | None = None
        if split_draws > 1:
            base_seed = int(((base_config.get("data") or {}).get("split") or {}).get("seed", 42))
            resolved_draw_seeds = (
                list(split_draw_seeds) if split_draw_seeds is not None
                else [base_seed + i for i in range(split_draws)]
            )
            search_param_space = {
                **param_space,
                SPLIT_DRAW_SEED_KEY: {"type": "categorical", "choices": resolved_draw_seeds},
            }

        import uuid
        from datetime import datetime, timezone

        hpo_dir = hpo_root(output_dir)
        hpo_dir.mkdir(parents=True, exist_ok=True)
        study_name = study_name or f"hpo_{uuid.uuid4().hex[:8]}"
        sweep_root = sweep_dir(study_name, output_dir)
        sweep_root.mkdir(parents=True, exist_ok=True)
        cancel_path = sweep_root / SWEEP_CANCEL_SENTINEL

        manifest = {
            "study_name": study_name,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "n_trials": n_trials,
            "search_alg": search_alg,
            "scheduler": scheduler,
            "grace_period": grace_period,
            "reduction_factor": reduction_factor,
            "max_concurrent": max_concurrent,
            "warm_start": warm_start,
            "baseline_params": baseline_params,
            "resources_per_trial": resources_per_trial,
            "param_space": param_space,
            "base_config": base_config,
            "sweep_dir": str(sweep_root),
            "relaunched_from": relaunched_from,
            "split_draws": split_draws,
            "split_draw_seeds": resolved_draw_seeds,
        }
        manifest_key = sweep_manifest_key(study_name, output_dir)
        manifest_lock = threading.Lock()

        def _write_manifest() -> None:
            # Every write, the heartbeat thread's included, goes through this call under one
            # lock, restamping cancel_requested and heartbeat fresh so no write lands on another's back.
            with manifest_lock:
                manifest["cancel_requested"] = cancel_path.exists()
                manifest["heartbeat"] = datetime.now(timezone.utc).isoformat()
                store.replace(manifest_key, manifest)

        # A cancel already requested (the study_name was minted and registered before this
        # call reached the manifest write) records a cancelled manifest rather than refusing.
        if cancel_path.exists():
            manifest.update(status="cancelled", error=_CANCEL_BEFORE_START_REASON,
                            finished_at=datetime.now(timezone.utc).isoformat())
            _write_manifest()
            return {"status": "cancelled", "study_name": study_name, "error": _CANCEL_BEFORE_START_REASON}

        _write_manifest()
    finally:
        discard_sweep_launching(study_name)

    heartbeat_stop = threading.Event()

    def _heartbeat_loop() -> None:
        while not heartbeat_stop.wait(sweep_heartbeat_seconds()):
            try:
                _write_manifest()
            except (OSError, StoreError):
                logger.warning(
                    "could not restamp the heartbeat for the sweep %s; will retry at the "
                    "next interval", study_name, exc_info=True)

    heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
    heartbeat_thread.start()

    def objective_fn(config: dict, report) -> None:
        try:
            from ray import tune as _tune
            tid = _tune.get_context().get_trial_id()
        except Exception:
            tid = uuid.uuid4().hex[:8]
        _run_hpo_trial(config, report, base_config, str(sweep_root / f"{_TRIAL_DIR_PREFIX}{tid}"))

    try:
        result = tune_search(
            objective_fn=objective_fn,
            param_space=search_param_space,
            metric="objective",
            mode=hpo_mode,
            num_samples=n_trials,
            search_alg=search_alg,
            scheduler=scheduler,
            grace_period=grace_period,
            reduction_factor=reduction_factor,
            warm_start=warm_start,
            baseline_params=baseline_params,
            max_concurrent=max_concurrent,
            storage_path=str(hpo_dir),
            study_name=study_name,
            resources_per_trial=resources_per_trial,
            stop_all_when=lambda: cancel_path.exists(),
            split_draws=split_draws,
        )
    except Exception as exc:
        # Stopped and joined before any terminal write, so the write below is always the
        # last word, never raced by one more heartbeat restamp landing after it.
        heartbeat_stop.set()
        heartbeat_thread.join()
        if cancel_path.exists():
            manifest.update(status="cancelled", error=_CANCEL_DURING_RUN_REASON,
                            finished_at=datetime.now(timezone.utc).isoformat())
            _write_manifest()
            return {"status": "cancelled", "study_name": study_name, "error": _CANCEL_DURING_RUN_REASON}
        manifest.update(status="failed", error=str(exc),
                        finished_at=datetime.now(timezone.utc).isoformat())
        _write_manifest()
        raise

    heartbeat_stop.set()
    heartbeat_thread.join()

    if cancel_path.exists():
        manifest.update(status="cancelled", error=_CANCEL_DURING_RUN_REASON,
                        finished_at=datetime.now(timezone.utc).isoformat())
        _write_manifest()
        return {"status": "cancelled", "study_name": study_name, "error": _CANCEL_DURING_RUN_REASON}

    # Auto-launch TensorBoard on the sweep root: Ray's per-trial event files and each
    # trial's own tensorboard dir both sit under it.
    tb_info: dict = {}
    tb_logdir = result.get("tensorboard_logdir")
    if tb_logdir and auto_tensorboard:
        try:
            from tcip_mcp.pipelines.training.tensorboard_manager import launch_tensorboard
            tb_info = launch_tensorboard(tb_logdir, run_id=f"hpo_{study_name}")
        except Exception:
            pass

    result["tensorboard"] = tb_info

    if split_draws > 1:
        # The best is run_hyperparameter_search's own choice by mean over each point's draws (all_trials),
        # never Ray's get_best_result; split_sensitivity keeps every point's own block.
        groups = group_split_draws(result.get("all_trials") or [], resolved_draw_seeds or [])
        eligible = [g for g in groups if g["eligible"]]
        result.pop("best_value_state", None)
        result["split_sensitivity"] = groups
        result["n_points"] = n_trials
        result["split_draws"] = split_draws
        if eligible:
            pick = (max if hpo_mode == "max" else min)(eligible, key=lambda g: g["block"]["mean"])
            result["best_params"] = pick["point"]
            result.update(stored_number("best_value", pick["block"]["mean"]))
            result["best_value_spread"] = pick["block"]
        else:
            result["best_params"] = None
            result["best_value"] = None
            result["best_value_state"] = (
                "no eligible point: every drawn point had an errored or never-answered draw"
            )
            result["best_value_spread"] = None

    # best_value_state (stored_number's sibling for a non-finite best_value) rides along whenever the search produced one.
    manifest_result = {k: result.get(k) for k in ("best_params", "best_value", "n_trials")}
    if "best_value_state" in result:
        manifest_result["best_value_state"] = result["best_value_state"]
    for key in ("best_value_spread", "split_sensitivity", "n_points", "split_draws"):
        if key in result:
            manifest_result[key] = result[key]
    manifest.update(
        status="completed",
        finished_at=datetime.now(timezone.utc).isoformat(),
        result=manifest_result,
    )
    # Durable result records (best-effort, a write hiccup must not sink a completed sweep).
    try:
        _write_manifest()
        store.replace(study_result_key(study_name, output_dir), result)
    except (OSError, StoreError):
        logger.warning("could not persist the hpo result for %s", study_name, exc_info=True)
    return result


def _path_under(path: Path, root: Path) -> bool:
    """Whether ``path`` (resolved) is ``root`` itself or somewhere beneath it (also resolved)."""
    try:
        path.resolve().relative_to(root)
        return True
    except ValueError:
        return False


@mcp.tool()
@audited
def cancel_hyperparameter_search(study_name: str, output_dir: str = "", *, root: str | None = None) -> dict:
    """Request cooperative cancellation of a running HPO sweep.

    Writes the sweep's own stop file (``SWEEP_CANCEL_SENTINEL``) at the sweep's root: ``run_hyperparameter_search``
    checks it after preflight and before minting the manifest, ``_run_hpo_trial`` checks it at
    the start of every trial, and the sweep's own Tune ``Stopper`` polls it to end each trial's
    report and, once no trial directory still looks unfinished (or the heartbeat stale window has
    passed since the file was written), the whole experiment. Also writes the run-level sentinel
    (``run_registry.CANCEL_SENTINEL``) into every trial directory that has not yet written its
    resolved config, so a trial mid-epoch sees the request at its very next batch boundary the
    same way a standalone training run does, without waiting on the driver's own poll.

    Refuses when the study names no sweep this process can find: no manifest under the resolved
    root, no live trial of this study registered in this process's own run registry, and no
    ``mark_sweep_launching`` entry for it at this same resolved root either (the last covers
    the narrow window between a caller minting the id and ``run_hyperparameter_search`` writing its first
    manifest; a marked study with no manifest yet answers ``"running"`` with
    ``cancel_requested`` set. A mark recorded under a different root does not count: that study
    is one ``run_hyperparameter_search`` will never look for here).

    The manifest's own ``cancel_requested`` (read by the Tuning listing) is written through the
    store's compare-and-set, and never over a manifest already in a terminal status: ``run_hyperparameter_search``
    is the other writer of this record, and a terminal write of its own must never be reverted
    back to ``"running"`` by a cancel that read the manifest just before it (see
    :func:`sweep_manifest_key`). The sentinel files below are the authoritative signal either
    way; the manifest field is a best-effort mirror of them for the listing to read.

    Args:
        study_name: The sweep to cancel.
        output_dir: Where the sweep's own directory lives, as given to ``run_hyperparameter_search``; empty
            resolves the same ``.tcip/hpo`` default, under ``root``.
        root: The platform root this sweep launched under, for a caller (the Tuning cancel
            route) that already knows it; omitted resolves under this process's own root.
    """
    from tcip_mcp.pipelines.training.run_registry import CANCEL_SENTINEL, _RUNS, _RUNS_LOCK

    sweep_root = sweep_dir(study_name, output_dir, root=root)
    manifest_key = sweep_manifest_key(study_name, output_dir, root=root)
    versioned = store.read_versioned(manifest_key, default=None)
    manifest = versioned.value
    has_manifest = isinstance(manifest, dict)

    resolved_root = sweep_root.resolve()
    with _RUNS_LOCK:
        live_trial = any(
            r.origin == "hpo_trial" and r.output_dir and _path_under(Path(r.output_dir), resolved_root)
            for r in _RUNS.values()
        )
    if not has_manifest and not live_trial and not _sweep_launching(study_name, resolved_root):
        return {"error": f"no sweep named {study_name!r}: no manifest, no live trial and no "
                          "pre-manifest launch mark for it"}

    sweep_root.mkdir(parents=True, exist_ok=True)
    (sweep_root / SWEEP_CANCEL_SENTINEL).touch()
    for trial_dir in _running_trial_dirs(sweep_root):
        (trial_dir / CANCEL_SENTINEL).touch()

    if not has_manifest:
        # A live trial or a pre-manifest launch mark, either way nothing on disk yet to judge
        # a heartbeat against: this is a sweep actively starting, not a stale disk record.
        return {"study_name": study_name, "status": "running", "cancel_requested": True}

    state_manifest = manifest
    status = manifest.get("status", "running")
    if status not in ("completed", "failed", "cancelled"):
        working = {**manifest, "cancel_requested": True}
        try:
            store.replace(manifest_key, working, expect=versioned.version)
        except VersionConflict:
            # Another of run_hyperparameter_search's own writes (start, a heartbeat restamp, or terminal)
            # landed first; losing this costs nothing, since each re-derives cancel_requested.
            refreshed = store.read(manifest_key, default=manifest)
            state_manifest = refreshed if isinstance(refreshed, dict) else manifest
        else:
            state_manifest = working
    derived = sweep_state(state_manifest, stale_seconds=TCIP_HEARTBEAT_STALE_SECONDS, driver_live=False)
    return {"study_name": study_name, "status": derived, "cancel_requested": True}


def _apply_hpo_params(base_config: dict, params: dict) -> dict:
    """Apply flat HPO params onto a deep copy of ``base_config``, where ``train()`` reads them.

    Architecture is owned by the bespoke ``model_source`` builder (unknown to the sweep), so
    only optimizer/batch axes get purpose-built handling here; ``base_config``'s own progressive-
    unfreeze schedule is left untouched, preserving whatever schedule the agent configured:

      - ``lr``           -> ``optimizer["head_lr"]``, plus ``optimizer["backbone_lr"]`` scaled by
                            whatever backbone/head ratio ``base_config`` already expressed
                            (derived, not pinned, a frozen ``lr*0.1`` would discard an agent's
                            own deliberate ratio)
      - ``weight_decay`` -> ``optimizer["weight_decay"]``
      - ``batch_size``   -> ``training["batch_size"]``
      - anything else    -> the top level of ``cfg`` (not nested under ``training``, which runs
                            after ``normalize_train_config``'s hoist and would never reach
                            ``train()``'s top-level config reads), free for a bespoke
                            ``training_source`` to sweep its own axes; no whitelist, no reject.
    """
    import copy

    from tcip_mcp.pipelines.schemas import normalize_train_config

    cfg = normalize_train_config(copy.deepcopy(base_config))
    training = cfg.setdefault("training", {})

    # The ratio the agent already configured, read from base_config's own optimizer block
    # before this loop overwrites head_lr. Default to 1.0 (not a frozen 0.1) only when the agent
    # expressed no explicit backbone/head split at all.
    base_optimizer = base_config.get("optimizer") or {}
    base_backbone_lr = base_optimizer.get("backbone_lr")
    base_head_lr = base_optimizer.get("head_lr")
    backbone_head_ratio = (
        base_backbone_lr / base_head_lr if (base_head_lr and base_backbone_lr) else 1.0
    )

    for key, value in params.items():
        if key == "lr":
            lr = float(value)
            optimizer = cfg.setdefault("optimizer", {})
            optimizer["head_lr"] = lr
            optimizer["backbone_lr"] = lr * backbone_head_ratio
        elif key == "batch_size":
            training["batch_size"] = value
        elif key == "weight_decay":
            cfg.setdefault("optimizer", {})["weight_decay"] = value
        elif "." in key:
            # A dotted path (e.g. "model_source.builder") reaches the nested field it names,
            # rather than landing as a literal top-level key nothing reads.
            *path, leaf = key.split(".")
            node = cfg
            for i, part in enumerate(path):
                if not isinstance(node, dict):
                    raise ValueError(
                        f"hpo param {key!r} cannot reach {'.'.join(path[:i])!r}: base_config "
                        f"holds a {type(node).__name__} there, not a mapping to walk into"
                    )
                node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(
                    f"hpo param {key!r} cannot be set: base_config holds a "
                    f"{type(node).__name__} at {'.'.join(path)!r}, not a mapping"
                )
            node[leaf] = value
        else:
            cfg[key] = value
    return cfg


def _resolved_task(config: dict) -> str:
    """The task ``run_hyperparameter_search`` resolves a config to: ``model_source.task``, else ``data.task``,
    else ``"detection"``. The one definition both the base config's own resolution and a
    param_space point's resolution share, so an axis that changes task is judged by the same
    rule everywhere it is read.
    """
    from tcip_mcp.pipelines.model_build import MODEL_SOURCE_KEY

    return (config.get(MODEL_SOURCE_KEY) or {}).get("task") \
        or (config.get("data") or {}).get("task", "detection")


def _selection_metric_axis_conflict(
    base_config: dict, param_space: dict, hpo_task: str, hpo_metric: str, hpo_mode: str,
) -> str | None:
    """The refusal reason, if any, when some point ``param_space`` could resolve a trial to
    picks a different selection metric or ranking direction than ``(hpo_metric, hpo_mode)``,
    the pair ``run_hyperparameter_search`` already resolved from ``base_config`` and fixes once on the Tuner (Ray
    forbids setting metric/mode anywhere else).

    Resolves every :func:`_preflight_points` point through the same
    ``evaluation_section``/``resolve_selection_metric``/``HIGHER_IS_BETTER_BY_METRIC`` path the
    sweep itself uses, rather than enumerating the key shapes that could reach
    ``selection_metric``: this also catches an axis that changes the metric's own task-derived
    default (``model_source.task``, in particular) with no ``selection_metric`` key in sight.
    A point whose params fail to apply is left for the structural preflight loop to report.
    ``None`` when every point agrees with ``base_config``.
    """
    from tcip_mcp.pipelines.training.evaluation import HIGHER_IS_BETTER_BY_METRIC
    from tcip_mcp.pipelines.training.generic_trainer import resolve_selection_metric
    from tcip_mcp.pipelines.schemas import evaluation_section

    axes = sorted(param_space)
    for label, point in _preflight_points(param_space):
        try:
            point_cfg = _apply_hpo_params(base_config, point)
        except ValueError:
            continue
        point_task = _resolved_task(point_cfg)
        point_eval_cfg = evaluation_section(point_cfg)
        try:
            point_metric = resolve_selection_metric(
                point_task, point_eval_cfg.get("trait"), point_eval_cfg.get("selection_metric"))
        except ValueError as exc:
            return (f"param_space (axes {axes}) disagrees with base_config's selection metric "
                    f"at {label}: {exc}")
        point_mode = "max" if HIGHER_IS_BETTER_BY_METRIC[point_metric] else "min"
        if (point_metric, point_mode) != (hpo_metric, hpo_mode):
            return (f"param_space (axes {axes}) sweeps a selection metric or its direction at "
                    f"{label}: base_config resolves to {hpo_metric!r} ({hpo_mode}), this point "
                    f"to {point_metric!r} ({point_mode}); the sweep's selection metric and "
                    "direction are fixed once from base_config, not the param space; move it "
                    "into base_config.")
    return None


def _base_config_for_split_draws(base_config: dict, split_draws: int) -> dict:
    """``base_config`` as ``run_hyperparameter_search`` mints the sweep from: unchanged unless ``split_draws`` is
    above 1 and the config is bound to a split manifest, in which case a copy carries
    ``data.split.redraw_within_manifest: true`` (defaulting ``data.split.seed`` to 42 when
    absent, the same default every unset-seed config draws from), so every trial redraws train
    and val inside the manifest's own members instead of running on its one recorded partition.
    A caller who already set the flag (and a seed) by hand gets the same copy back in
    substance: an already-true flag or an already-set seed is left as it is.
    """
    if split_draws <= 1:
        return base_config
    data_cfg = base_config.get("data") or {}
    split_cfg = data_cfg.get("split") or {}
    if not split_cfg.get("manifest_dir"):
        return base_config
    new_split = {**split_cfg, "redraw_within_manifest": True}
    new_split.setdefault("seed", 42)
    return {**base_config, "data": {**data_cfg, "split": new_split}}


def _split_draws_refusal(
    base_config: dict, param_space: dict | None, task: str, search_alg: str, scheduler: str,
    split_draws: int, split_draw_seeds: list[int] | None, warm_start: bool,
    baseline_params: dict | None,
) -> str | None:
    """Every reason ``run_hyperparameter_search`` refuses ``split_draws`` above 1, checked before minting the
    sweep. ``None`` when nothing here objects; every call at ``split_draws=1`` is one of them,
    since split_draws itself governs nothing at its default.

    ``base_config`` bound to a split manifest is admitted rather than refused: ``run_hyperparameter_search`` has
    already set ``data.split.redraw_within_manifest`` on its own copy
    (:func:`_base_config_for_split_draws`) before this call, so every trial redraws train and
    val inside the manifest's own train-plus-val members instead of running on its one recorded
    partition. A bound config reads neither ``val_images_dir`` nor ``auto_val`` (the manifest
    branch binds ahead of both), so those two legs are skipped for it; in their place, the
    manifest's own distinct-groups check runs once here so a sweep whose every trial would
    starve a side is refused before minting rather than after every trial fails the same way.
    """
    if split_draws <= 1:
        return None
    from tcip_mcp.pipelines.data.split_construction import STEM_TASKS
    from tcip_mcp.pipelines.training.hpo import SPLIT_DRAW_SEED_KEY, _NATIVE_SEARCH, _NO_SCHEDULER

    data_cfg = base_config.get("data") or {}
    split_cfg = data_cfg.get("split") or {}
    bound = bool(split_cfg.get("manifest_dir"))
    if not bound:
        if data_cfg.get("val_images_dir"):
            return ("split_draws redraws the split, and base_config names data.val_images_dir: "
                    "an explicit validation source draws nothing.")
        if not data_cfg.get("auto_val", True):
            return ("split_draws needs a drawn validation split, and base_config sets "
                     "data.auto_val=False.")
    if task not in STEM_TASKS:
        return (f"split_draws needs a drawn validation split, and task={task!r} sits outside "
                f"the drawn path's own tasks ({sorted(STEM_TASKS)}).")
    if (search_alg or "").lower() not in _NATIVE_SEARCH:
        native = sorted(x for x in _NATIVE_SEARCH if isinstance(x, str) and x)
        return (f"split_draws pairs a grid axis through Ray's own BasicVariantGenerator, which "
                f"only a native search_alg ({native}) builds; search_alg={search_alg!r} does not.")
    if (scheduler or "none").lower() not in _NO_SCHEDULER:
        return (f"split_draws makes each draw a blocked comparison, and a pruning scheduler "
                f"({scheduler!r}) could end one draw before another completes; pass "
                "scheduler='none' with split_draws.")
    if split_draw_seeds is not None and len(split_draw_seeds) != split_draws:
        return (f"split_draw_seeds has {len(split_draw_seeds)} seed(s) but split_draws="
                f"{split_draws}: one seed per draw.")
    if warm_start and baseline_params and SPLIT_DRAW_SEED_KEY in baseline_params:
        return (f"warm_start's baseline_params names {SPLIT_DRAW_SEED_KEY!r}: Ray's own "
                "preset-variant pinning would pin every draw to that one seed instead of "
                f"pairing the grid; drop {SPLIT_DRAW_SEED_KEY!r} from baseline_params.")
    if SPLIT_DRAW_SEED_KEY in (param_space or {}):
        return (f"param_space already sweeps {SPLIT_DRAW_SEED_KEY}, the same axis split_draws "
                "adds as a paired grid; state the crossing through split_draws/"
                "split_draw_seeds, not a second data.split.seed axis.")
    other_data_axes = sorted(
        k for k in (param_space or {}) if k.startswith("data.") and k != SPLIT_DRAW_SEED_KEY)
    if other_data_axes:
        return (f"split_draws pairs the identical partition with every sampled point at each "
                f"draw, and param_space sweeps {other_data_axes} beside it: a data.* axis other "
                f"than {SPLIT_DRAW_SEED_KEY} changes what a point admits or how it draws, so "
                "draw k would no longer be the same partition for every point.")
    if bound:
        return _bound_redraw_starvation_issue(data_cfg, split_cfg)
    return None


def _bound_redraw_starvation_issue(data_cfg: dict, split_cfg: dict) -> str | None:
    """Whether ``run_hyperparameter_search``'s ``split_draws`` minting a redraw sweep over a bound
    ``base_config``'s manifest would starve every trial the identical way: the manifest's own
    foreground-groups check (:func:`~tcip_mcp.pipelines.data.splits.manifest_redraw_universe`,
    :func:`~tcip_mcp.pipelines.data.splits.redraw_starved_issue`), the same one
    ``preflight_config`` runs for one config, run once here over the sweep's shared manifest and
    date. ``None`` when the manifest reads and holds enough foreground groups.
    """
    from tcip_mcp.dataset_layout import annotation_date
    from tcip_mcp.pipelines.data.splits import (
        count_label_lines, manifest_redraw_universe, redraw_starved_issue,
    )
    from tcip_mcp.tools.data_tools import read_split_manifest_dir

    manifest_dir = split_cfg["manifest_dir"]
    try:
        manifest = read_split_manifest_dir(manifest_dir)
    except ValueError as exc:
        return f"split_draws: {exc}"
    date = annotation_date(data_cfg.get("labels_dir", ""))
    try:
        stems, group_key_fn = manifest_redraw_universe(manifest, date)
    except ValueError as exc:
        return f"split_draws: {exc}"
    labels_dir = data_cfg.get("labels_dir", "")
    subject, attribute = data_cfg.get("subject"), data_cfg.get("attribute")
    foreground_counts = {
        s: count_label_lines(labels_dir, s, subject=subject, attribute=attribute) for s in stems
    }
    return redraw_starved_issue(
        stems, group_key_fn, foreground_counts=foreground_counts, manifest_dir=manifest_dir,
        date=date, seed=split_cfg.get("seed"), group_by=manifest.get("group_by"),
    )


def group_split_draws(all_trials: list[dict], planned_seeds: list[int]) -> list[dict]:
    """Group ``tune_search``'s own ``all_trials`` rows by the point each draw shares (every
    param but ``hpo.SPLIT_DRAW_SEED_KEY``), each group carrying the ``split_draws`` block the
    sweep result and its manifest both record.

    ``planned_seeds`` names every seed the sweep asked for; a group is ``eligible`` for best
    only when every one of them completed for that point and the group holds no errored or
    never-answered row, never merely a count of complete rows: a point Ray repeated (grid
    search repeats every point per sample; a categorical-only random space collides) can land
    two complete values under the same seed while its other planned seed never completes, and
    counting complete values alone would call that eligible on a mean over one seed twice. Pass
    an empty list to accept any single complete row per point regardless of seed identity (a
    sweep that never asked for draws, or an older result read back).

    A group's block always carries ``n`` (every row seen for the point, ``COMPLETE`` and
    ``ERROR`` alike), ``n_complete`` (rows among them that completed with a real value) and
    ``seeds_complete`` (the distinct seeds among those complete rows, sorted), plus ``seeds``
    (one entry per complete row, not deduplicated), ``values``, ``mean``, ``std`` (the sample
    standard deviation, ``None`` under two values), ``min`` and ``max`` over ``values``. A row
    with no ``params`` at all (Ray's own never-answered case) forms its own singleton,
    ineligible group, ``point`` ``None``: it names no point to group under.
    """
    import statistics

    from tcip_mcp.pipelines.training.hpo import SPLIT_DRAW_SEED_KEY

    planned = set(planned_seeds)
    groups: dict[str, dict] = {}
    order: list[str] = []
    for i, row in enumerate(all_trials):
        params = row.get("params")
        if params is None:
            key, point = f"__unanswered_{i}__", None
        else:
            point = {k: v for k, v in params.items() if k != SPLIT_DRAW_SEED_KEY}
            key = json.dumps(point, sort_keys=True, default=str)
        entry = groups.setdefault(key, {"point": point, "rows": []})
        if key not in order:
            order.append(key)
        entry["rows"].append(row)

    out: list[dict] = []
    for key in order:
        entry = groups[key]
        rows = entry["rows"]
        complete = [r for r in rows if r.get("state") == "COMPLETE" and r.get("value") is not None]
        complete_ids = {id(r) for r in complete}
        incomplete = [r for r in rows if id(r) not in complete_ids]
        values = [float(r["value"]) for r in complete]
        seeds = [(r.get("params") or {}).get(SPLIT_DRAW_SEED_KEY) for r in complete]
        seeds_complete = sorted({s for s in seeds}, key=lambda s: (s is None, s))
        block = {
            "seeds": seeds,
            "values": values,
            "mean": statistics.fmean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "n": len(rows),
            "n_complete": len(values),
            "seeds_complete": seeds_complete,
        }
        eligible = (
            entry["point"] is not None and not incomplete
            and planned <= set(seeds_complete)
        )
        out.append({"point": entry["point"], "block": block, "eligible": eligible})
    return out


def _first_sampled_point(param_space: dict) -> dict:
    """One deterministic point from ``param_space``, spanning its declared range or choices.

    Not a real trial's sample (Ray Tune's own samplers are what a trial actually draws from),
    but enough structure to resolve what ``_apply_hpo_params`` would apply to any given trial's
    config, so a structural preflight run over it sees the same shape a trial would.
    """
    point: dict = {}
    for key, spec in param_space.items():
        if not isinstance(spec, dict):
            # Not this platform's own {"type": ...} shape (a caller-composed space bypassing
            # get_default_space): take a value outright rather than guess a range from it.
            point[key] = spec[0] if isinstance(spec, list) and spec else spec
            continue
        kind = spec.get("type")
        if kind == "categorical":
            choices = spec.get("choices") or [None]
            point[key] = choices[0]
        elif kind in ("loguniform", "uniform", "int"):
            point[key] = spec["low"] if "low" in spec else spec.get("high", 0)
        else:
            point[key] = spec.get("low", spec.get("choices", [None])[0])
    return point


def _preflight_points(param_space: dict) -> list[tuple[str, dict]]:
    """Every point ``run_hyperparameter_search``'s preflight must check: the first sampled corner, plus one variant
    per categorical choice and one per numeric bound, each holding every other axis at its first
    sampled value.

    Judging the whole space by its first sampled corner alone misses a broken choice that sits
    anywhere but first: a categorical axis naming a builder or a data path is checked at every
    value it could resolve a trial to, not just one, so a sweep whose first choice happens to
    work but whose second does not is still caught before any trial runs.
    """
    base = _first_sampled_point(param_space)
    points: list[tuple[str, dict]] = [("the first sampled point", dict(base))]
    for key, spec in param_space.items():
        if not isinstance(spec, dict):
            continue
        kind = spec.get("type")
        if kind == "categorical":
            for choice in spec.get("choices") or []:
                variant = dict(base)
                variant[key] = choice
                points.append((f"{key}={choice!r}", variant))
        elif kind in ("loguniform", "uniform", "int"):
            for bound in ("low", "high"):
                if bound in spec:
                    variant = dict(base)
                    variant[key] = spec[bound]
                    points.append((f"{key} {bound}={spec[bound]!r}", variant))
    return points


def _ensure_experiment(
    experiment_id: str, config: dict, data_source, resume_from: str, run_id: str,
    *, output_dir: str, dataset_id: str | None = None, dataset_fingerprint: str | None = None,
) -> str:
    """Create or attach the experiment for a run, enforcing experiment immutability.

    Returns the experiment id actually used. An existing id may be reused only when the experiment
    is pristine (agent pre-created it: state 'created', no metrics), in which case its
    ``config.json`` (written before tiling/seed resolution) is refreshed with the config this run
    is actually launching. Anything else, including a ``resume_from`` that targets
    an id which already has recorded history, mints a fresh ``<id>_<run_id>`` (with the old id as
    parent lineage) so the prior run's status, metrics, lineage, and registry entry stay intact
    (resuming into a non-pristine id without this would silently reuse it, discarding the
    resumed run's own metrics/lineage writes behind the terminal-state lock and letting the model
    registry replace the original's entry by name with no record of what was superseded).

    Every branch below stamps ``run_id``/``output_dir`` into the resolved experiment's
    ``status.json`` before returning, unconditionally, once, regardless of which branch
    resolved the id, so a different process can later discover this run's real artifact directory
    from ``experiment_id`` alone (``resolve_experiment_dir_for_run``/``reconstruct_run_status``/the
    disk-based ``cancel_run`` fallback all depend on this). Deliberately not a ``create_experiment``
    param: that would only cover the fresh-creation branch and silently miss the pristine-reuse
    branch, which never otherwise touches ``status.json`` at all.
    """
    from tcip_mcp.experiments import (
        create_experiment, is_pristine, metrics_logged_of, overwrite_config_if_pristine,
        read_member, stamp_run_identity, status_key,
    )

    created = create_experiment(experiment_id, config, data_source=data_source,
                                dataset_id=dataset_id, dataset_fingerprint=dataset_fingerprint)
    if "error" not in created:
        stamp_run_identity(experiment_id, run_id, output_dir)
        return experiment_id

    # is_pristine is the one implementation of the predicate: only attempt the overwrite when it
    # says pristine, so a non-pristine id mints its fresh id below with no refusal audited.
    status = read_member(status_key(experiment_id), {})
    state = status.get("state") if isinstance(status, dict) else None
    metrics_logged = metrics_logged_of(status)
    if is_pristine(state, metrics_logged):
        overwritten = overwrite_config_if_pristine(experiment_id, config)
        if "error" not in overwritten:
            stamp_run_identity(experiment_id, run_id, output_dir)
            return experiment_id

    fresh_id = f"{experiment_id}_{run_id}"
    logger.warning(
        "experiment_id %s already has a run; experiments are immutable, tracking "
        "this run as %s instead.", experiment_id, fresh_id,
    )
    # The snapshot must name itself, not whatever id the caller's own config carried in (its
    # parent, for a relaunch that set config["experiment_id"] to the picked id before this call).
    forked_config = {**config, "experiment_id": fresh_id}
    create_experiment(fresh_id, forked_config, parent_experiment=experiment_id, data_source=data_source,
                      dataset_id=dataset_id, dataset_fingerprint=dataset_fingerprint)
    stamp_run_identity(fresh_id, run_id, output_dir)
    return fresh_id


def _dataset_source_kwargs(task: str, data_cfg: dict) -> dict:
    """The ``build_dataset`` kwargs for a run's data config.

    One definition shared by the training path and the preflight smoke, so the batch the contract
    is proved against is built from the same keys as the batch the run will train on.

    ``data.date`` is the capture date the run's confirmed negatives were recorded under, threaded
    so the build reads the bucket the GUI wrote instead of one taken from the labels path. A run
    over ``annotations/<date>/`` sets it; a tree that carries no date leaves it unset.
    """
    from tcip_mcp.pipelines.model_build import DATASET_SOURCE_KEY

    if task in ("detection", "instance_seg"):
        kw = {"images_dir": data_cfg.get("images_dir", ""),
              "labels_dir": data_cfg.get("labels_dir", ""),
              "date": data_cfg.get("date")}
        # The run's subject (and optional attribute): required to read name-based labels and to
        # derive the single assign_class_ids map. Threaded so every train/val build uses one map.
        if data_cfg.get("subject"):
            kw["subject"] = data_cfg["subject"]
        if data_cfg.get("attribute"):
            kw["attribute"] = data_cfg["attribute"]
        # Thread the on-disk label format through to the dataset (json | coco).
        if data_cfg.get("label_format"):
            kw["label_format"] = data_cfg["label_format"]
        if data_cfg.get("coco_json"):
            kw["coco_json"] = data_cfg["coco_json"]
    elif task == "semantic_seg":
        kw = {"images_dir": data_cfg.get("images_dir", ""),
              "masks_dir": data_cfg.get("masks_dir", data_cfg.get("labels_dir", ""))}
    else:
        kw = {"images_dir": data_cfg.get("images_dir", "")}
        if data_cfg.get("csv_path"):
            kw["csv_path"] = data_cfg["csv_path"]
    if data_cfg.get(DATASET_SOURCE_KEY):
        # Bespoke seam (mirrors model_source): route build_dataset to the agent's builder for a task the known loaders don't cover, threaded through src so the split machinery still passes it (with stems) to every train/val build below.
        kw["dataset_source"] = data_cfg[DATASET_SOURCE_KEY]  # left names build_dataset's param
    return kw


def _one_real_batch(task: str, config: dict, n: int = 2):
    """``(batch, reason_it_failed)``, one collated ``(images, targets)`` from the run's dataset.

    Lets the model contract smoke a task it has no synthetic schema for, without the platform
    enumerating tasks. Built through the same source kwargs and augmentation the run itself uses,
    so a batch that smokes here is the batch that trains. Best-effort by design: a config that
    cannot yield a batch returns ``(None, reason)`` and the caller decides what that means, this
    function never decides whether a run proceeds. The reason is returned rather than only logged,
    so a caller that blocks can say what actually failed.
    """
    data_cfg = config.get("data") or {}
    try:
        from tcip_mcp.pipelines.data.datasets import build_dataset
        from tcip_mcp.pipelines.training.collation import task_collate

        transforms = None
        if config.get("augmentation"):
            from tcip_mcp.pipelines.data.augmentations import build_augmentation
            transforms = build_augmentation(config["augmentation"])

        src = _dataset_source_kwargs(task, data_cfg)
        ds = build_dataset(task, **src, transforms=transforms, tiling=data_cfg.get("tiling"))
        assert isinstance(ds, Sized), "every build_dataset task backend defines __len__"
        items = [ds[i] for i in range(min(n, len(ds)))]
        if not items:
            return None, "the dataset built but is empty"
        return task_collate(task)(items), None
    except Exception as exc:  # noqa: BLE001, an unbuildable batch is a caller decision, not a crash
        logger.info("could not build a real batch to smoke task %r: %s", task, exc)
        return None, f"{type(exc).__name__}: {exc}"


def _reserve_calibration_feasibility_issues(
    model_source: dict | None, data_cfg: dict, split_cfg: dict, reserve_cal_frac: float, *,
    smoke: bool,
) -> list[str]:
    """Named ``preflight_config`` issues for an explicitly-requested
    ``reserve_calibration_fraction`` that cannot be honored, so the launch refuses here rather
    than silently training without a calibration region or failing deep inside the training
    subprocess. Routed through ``preflight_config``/``launch_training``'s own validation surface
    (not ``pipelines.feedback.review_calibration``'s ``_FAILURE_MESSAGES``: that registry's only
    consumer is the Review tab reading a ``ResolvedBundle``'s sweep failures, and this refusal
    fires at training-launch time, with neither in scope).

    Structurally inapplicable configs (not detection, tiling disabled, a multi-stem dataset that
    would use the group-balanced split instead) are always flagged, cheaply, no dataset build
    needed. The single-source geometry
    :func:`~tcip_mcp.pipelines.data.split_construction.spatial_single_source_split` itself would derive
    (extent, strip-layout feasibility, an empty side after real filtering) is checked by actually
    calling it, real dataset construction, so gated on ``smoke=True`` like this function's other
    dataset/model-touching checks (a plain, non-smoke ``preflight_config`` call still catches the
    structurally-inapplicable cases above, just not this geometry).
    """
    task = (model_source.get("task") if isinstance(model_source, dict) else None) \
        or (data_cfg.get("task", "detection") if isinstance(data_cfg, dict) else "detection")
    tiling_cfg = data_cfg.get("tiling") if isinstance(data_cfg, dict) else None
    if task != "detection" or not tiling_cfg or not tiling_cfg.get("enabled", True):
        return [
            f"data.split.reserve_calibration_fraction={reserve_cal_frac} has no effect: it only "
            "applies to a detection task with tiling enabled (the single-source spatial-strip "
            f"split), this config's task is {task!r} with tiling={tiling_cfg!r}."
        ]

    images_dir, labels_dir = data_cfg.get("images_dir"), data_cfg.get("labels_dir")
    if not images_dir or not labels_dir or not Path(images_dir).is_dir() or not Path(labels_dir).is_dir():
        return []  # the existing images_dir/labels_dir structural checks already cover this

    from tcip_mcp.pipelines.image_utils import list_logical_images

    stems = sorted(list_logical_images(images_dir))
    if len(stems) >= 2:
        return [
            f"data.split.reserve_calibration_fraction={reserve_cal_frac} has no effect: "
            f"{len(stems)} source images resolve to the group-balanced multi-stem split, not the "
            "single-source spatial-strip split a calibration region reserves from."
        ]
    if len(stems) != 1 or not smoke:
        return []

    from tcip_annotation.json_io import UnreadableLabelDocument

    try:
        from tcip_mcp.pipelines.data.datasets import build_dataset
        from tcip_mcp.pipelines.data.split_construction import spatial_single_source_split

        base = build_dataset(
            "detection", images_dir=images_dir, labels_dir=labels_dir,
            subject=data_cfg.get("subject"), attribute=data_cfg.get("attribute"))
        spatial_single_source_split(
            stems[0], dict(data_cfg), tiling_cfg, base, dict(split_cfg), None)
    except (ValueError, UnreadableLabelDocument) as exc:
        return [f"data.split.reserve_calibration_fraction: {exc}"]
    except Exception as exc:  # noqa: BLE001, an unrelated build failure isn't this check's own
        logger.info(
            "reserve_calibration_fraction feasibility probe could not build a dataset to check "
            "(%s); not reported as this check's own refusal.", exc)
    return []


@mcp.tool()
@audited
def evaluate_model(
    run_id_or_ckpt: str,
    images_dir: str,
    labels_dir: str = "",
    task: str = "detection",
    conf_threshold: float | None = None,  # report/select at the ship point
    iou_threshold: float = 0.5,
    iou_type: str | None = None,
    max_dets: int | None = None,
    tiling: dict | None = None,
    use_tiled_inference: bool = False,
    global_nms_iou: float | None = None,
    postprocess: str = "nms",
    trait: str | None = None,
    subject: str | None = None,
    attribute: str | None = None,
    date: str | None = None,
    split_manifest_dir: str | None = None,
) -> dict:
    """Evaluate a trained checkpoint on a (held-out) dataset and write test_results.json.

    Computes the same per-task metrics as validation, detection/instance_seg get
    pycocotools mAP + precision/recall/F1; classification/ordinal/regression get the
    in-house scalar metrics, and writes ``test_results.json`` beside the checkpoint.

    Three detection eval regimes:
      * Untiled default (no ``tiling``, checkpoint trained without tiling) -> single full-res
        forward pass, ``eval_regime="full-frame-single-pass"``. For a checkpoint that was never
        tile-trained, this is the correct delivery gate, untiled training, untiled eval, untiled
        inference are all the same regime, so there is nothing to reconcile. Do not reach for
        ``use_tiled_inference`` for such a checkpoint; it has no persisted tile geometry to gate
        against and will refuse (see below).
      * ``tiling`` set (or a run id whose training was tiled, reused automatically) -> tile-level
        diagnostic that matches the training-run val mAP. This is not the delivery metric, it
        scores fragmented tiles against fragmented GT, not the shipped full-frame count.
      * ``use_tiled_inference=True`` -> the delivery-grade full-frame metric for a tile-trained
        checkpoint (tiled inference reconstructed to full frame, matched to full-frame GT). Report
        this to gate a delivery for such a checkpoint. Tile geometry is resolved from the
        checkpoint's own persisted or native-frame training geometry, or an explicit override; a
        checkpoint with none of those refuses rather than silently fabricating a scale, see
        ``run_full_frame_evaluation``'s docstring for the full precedence.

    Args:
        run_id_or_ckpt: A training run id (uses its ``model_best.pt``) or a checkpoint path.
            Either way the resolved checkpoint must be registered under this process's platform
            state root (``register_model``, explicit mode for a foreign or bespoke checkpoint) or
            this door refuses before loading it.
        images_dir: Images directory for the evaluation split.
        labels_dir: Labels dir (detection/instance_seg), masks dir (semantic_seg), or the GT CSV
            path (classification/ordinal/regression, one row per image stem).
        task: Task type.
        conf_threshold: Operating confidence for P/R/F1. ``None`` (default) resolves to the
            platform default (``DEFAULT_CONF``) on both regimes; an explicit value is always
            honored verbatim, on both, a stated value equal to the default included.
        iou_threshold: Operating IoU (on COCOeval's grid; 0.5 -> index 0).
        iou_type: 'bbox' or 'segm'. Default (None) auto-resolves from the task, 'segm' for
            instance_seg, 'bbox' otherwise, so a mask model isn't silently scored as boxes.
        max_dets: Full-frame/COCOeval detection cap. ``None`` (default) resolves
            per-regime, 100 (the COCOeval ``maxDets`` convention) on the tile-level diagnostic
            path, 1000 (``DEFAULT_MAX_DETS``, dense full-frame scenes aren't truncated) on the
            delivery-grade ``use_tiled_inference`` path. An explicit value is always honored
            verbatim on both paths (no rescuing substitution), the delivery-grade path stamps a
            per-image ``cap_hit``/``max_dets_cap_saturated_frac`` so an explicit cap that actually
            truncates real detections is visible rather than silently assumed safe.
        tiling: Optional detection tiling dict ({enabled, tile_size, overlap, ...}) for a
            tile-level eval. None + a run id reuses the run's training tiling; None + a
            checkpoint path stays untiled.
        use_tiled_inference: Score the delivery regime (full-frame via tiled inference).
        global_nms_iou: Cross-tile global NMS IoU threshold (tiled paths only). ``None``
            (default) resolves to the platform default (``DEFAULT_NMS_IOU``); an explicit value
            is always honored verbatim.
        postprocess: Cross-tile merge, "nms" suppresses overlaps, "nmm" unions boxes split
            across a tile seam.
        trait: When set, the trait's derived localization criterion (traits.py, e.g. a count
            trait's center-match) governs the reported count and the selection f1; AP@0.5 (``iou_threshold``)
            is kept as a labeled comparability metric. Absent -> the IoU convention governs.
        subject: Name-based GT scope. Caller-supplied wins; else resolved from the producing
            run's own config so the eval reads GT through the same id map the run trained with.
        attribute: Attribute scope for the same name-based GT resolution as ``subject``.
        date: The capture date this split's confirmed negatives were recorded under, the bucket
            key the delivery-grade path reads them by. A GT dir under ``annotations/<date>/``
            states that date; a split tree or a curated dataset carries none and leaves this
            unset. Outside ``split_manifest_dir``, never recovered from ``labels_dir``; under it,
            derived from ``labels_dir`` (``annotation_date``) the same way the manifest's own
            universe is drawn, and a stated value that disagrees refuses, naming both, so the
            negative confirmations and the calibration universe are always read under one date.
        split_manifest_dir: Score the checkpoint over this split manifest's ``calibration``
            members under ``labels_dir``'s own date instead of the whole directory: the same
            subject/attribute/date/images-root checks the calibration door applies, refusing the
            same way (detection/instance_seg only, and not combined with
            ``use_tiled_inference``), except its own floor of one foreground group, since this
            door draws no lock and halves nothing. ``test_results.json`` then records
            ``split_manifest_dir`` and the evaluated stem count, the loader's own count, refused
            by name (naming the difference and the remedy) when the loader admits fewer than the
            universe the manifest drew, since the data moved under the manifest since the split
            was drawn; omitted, the whole directory is scored, as today.
    """
    import torch
    from torch.utils.data import DataLoader

    from tcip_mcp.pipelines.training.generic_trainer import checkpoint_key
    from tcip_mcp.pipelines.training.collation import task_collate
    from tcip_mcp.pipelines.training.run_registry import get_run
    from tcip_mcp.pipelines.training.eval_runners import (
        run_full_frame_evaluation, run_test_evaluation,
    )
    from tcip_mcp.pipelines.data.datasets import build_dataset
    from tcip_mcp.pipelines.resolution import applied_operating_point

    # The tile-level/single-pass paths apply this directly below; the full-frame path resolves
    # its own sentinels internally, so its own caller passes the raw arguments through unchanged.
    applied_conf, _applied_nms_iou, _applied_max_dets = applied_operating_point(
        conf_threshold, global_nms_iou, None)

    ckpt = run_id_or_ckpt
    run = None
    if not Path(ckpt).is_file():
        run = get_run(run_id_or_ckpt)
        if run is None:
            return {"error": f"Not a checkpoint path or known run id: {run_id_or_ckpt}"}
        ckpt = str(store.blob_path(checkpoint_key(run.output_dir, "model_best")))
    if not Path(ckpt).is_file():
        return {"error": f"Checkpoint not found: {ckpt}"}

    from tcip_mcp.model_registry import UnregisteredCheckpoint, load_registered_checkpoint

    try:
        checkpoint = load_registered_checkpoint(ckpt)
    except UnregisteredCheckpoint as exc:
        return {"error": str(exc)}

    # A bare checkpoint path (run is None) carries its own stamped config["data"] too, read off
    # the object already loaded, so both paths agree without a second read.
    if run is not None:
        run_data_cfg = run.config.get("data", {}) or {}
    else:
        run_data_cfg = checkpoint.data_config
    run_tiling = run_data_cfg.get("tiling")
    # The eval scope's subject/attribute: caller-supplied wins, else the producing run's config, so
    # the name-based GT reads through the same id map the run trained with.
    if subject is None:
        subject = run_data_cfg.get("subject")
    if attribute is None:
        attribute = run_data_cfg.get("attribute")

    manifest_stems: list[str] | None = None
    if split_manifest_dir is not None:
        if task not in ("detection", "instance_seg"):
            return {"error": f"split_manifest_dir names a split manifest, and only detection "
                             f"and instance_seg admit through the trainable_stems draw a "
                             f"manifest is drawn through; task={task!r} cannot bind to one."}
        if use_tiled_inference:
            return {"error": "split_manifest_dir is not combined with use_tiled_inference: that "
                             "delivery-grade path scans images_dir/labels_dir on its own, never "
                             "narrowed to a manifest's stems."}
        from tcip_mcp.pipelines.data.splits import (
            label_image_stems, resolve_manifest_calibration_universe,
        )
        from tcip_mcp.tools.data_tools import read_split_manifest_dir

        manifest = read_split_manifest_dir(split_manifest_dir)
        present, _ = label_image_stems(labels_dir, images_dir)
        try:
            manifest_stems, _group_by, _group_key_map, _excluded, cal_date, subject, attribute = \
                resolve_manifest_calibration_universe(
                    manifest, split_manifest_dir, labels_dir, images_dir, subject, attribute,
                    present, min_foreground_groups={"calibration": 1})
        except ValueError as exc:
            return {"error": str(exc)}
        if date is not None and date != cal_date:
            return {"error": f"date={date!r} disagrees with the date labels_dir={labels_dir!r} "
                             f"is under ({cal_date!r}); a split manifest binds under one date, "
                             "so the negative confirmations and the calibration universe must be "
                             "read under the same one."}
        date = cal_date

    # Delivery-grade full-frame path: conf_threshold/global_nms_iou/max_dets pass through exactly
    # as given, run_full_frame_evaluation resolves its own sentinels (a direct caller's record).
    if use_tiled_inference and task == "detection":
        tcfg = tiling or run_tiling or {}
        # tile_size/overlap pass through as None-if-absent: run_full_frame_evaluation itself
        # resolves them from persisted training geometry (or refuses), never this wrapper fabricating.
        from tcip_annotation.json_io import UnreadableLabelDocument

        try:
            return run_full_frame_evaluation(
                checkpoint, images_dir, labels_dir, str(Path(ckpt).parent),
                subject=subject, attribute=attribute,
                conf_threshold=conf_threshold, iou_threshold=iou_threshold,
                tile_size=tcfg.get("tile_size"), overlap=tcfg.get("overlap"),
                global_nms_iou=global_nms_iou, postprocess=postprocess,
                max_dets=max_dets, trait=trait, date=date,
            )
        except (ValueError, UnreadableLabelDocument) as exc:
            return {"error": str(exc)}

    # Tile-level diagnostic (or untiled). Only detection tiles; a run id reuses its training tiling.
    if tiling is None and run is not None:
        tiling = run_tiling
    if task != "detection":
        tiling = None

    # One kwargs-builder shared with the training path (_dataset_source_kwargs), not a second
    # hand-rolled copy: two independent implementations of "which data_cfg keys does this task
    # read" is exactly what let classification/ordinal/regression drift out of sync with training
    # (evaluate_model never threaded a CSV path, so OrdinalDataset/RegressionDataset construction
    # always failed here). labels_dir doubles as the CSV path for the non-geometry tasks, the same
    # single "wherever this task's GT lives" slot it already serves for masks_dir/semantic_seg.
    data_cfg = {"images_dir": images_dir, "labels_dir": labels_dir, "masks_dir": labels_dir,
                "csv_path": labels_dir, "subject": subject, "attribute": attribute, "date": date}
    ds_kwargs = _dataset_source_kwargs(task, data_cfg)
    if manifest_stems is not None:
        ds_kwargs["stems"] = manifest_stems
    try:
        dataset = build_dataset(task, **ds_kwargs, tiling=tiling)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Failed to build dataset: {exc}"}

    evaluated_stem_count = None
    if manifest_stems is not None:
        loader_stems = list(getattr(dataset, "stems", []) or [])
        evaluated_stem_count = len(loader_stems)
        if evaluated_stem_count < len(manifest_stems):
            missing = sorted(set(manifest_stems) - set(loader_stems))
            preview = missing[:10]
            more = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
            return {"error": f"the split manifest's calibration universe for date {date!r} "
                             f"holds {len(manifest_stems)} stem(s), but the loader admitted only "
                             f"{evaluated_stem_count}: {preview}{more}. The data moved under the "
                             "manifest since the split was drawn (a label emptied, a "
                             "confirmation withdrawn); regenerate the split over the current "
                             "data."}

    loader = DataLoader(dataset, batch_size=4, collate_fn=task_collate(task))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 100 is the COCOeval maxDets convention for this tile-level/diagnostic regime, distinct
    # from the delivery-grade path's 1000 above; an explicit caller max_dets is honored verbatim.
    resolved_max_dets = 100 if max_dets is None else max_dets
    return run_test_evaluation(
        checkpoint, loader, device, task, str(Path(ckpt).parent),
        conf_threshold=applied_conf, iou_threshold=iou_threshold,
        iou_type=iou_type, max_dets=resolved_max_dets, tiling=tiling, trait=trait,
        split_manifest_dir=split_manifest_dir,
        evaluated_stem_count=evaluated_stem_count,
    )
