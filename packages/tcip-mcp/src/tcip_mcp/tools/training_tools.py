"""Training MCP tools, config validation, launch training, HPO, status."""

from __future__ import annotations

import itertools
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
    Key,
    StoreDescriptor,
    StoreError,
    check_json_value,
    register_store,
    store,
)
from tcip_store.file_backend import RootedFileLocator

from tcip_mcp.server import mcp
from tcip_mcp.audit import audited
from tcip_mcp.pipelines.resolution import DEFAULT_CONF, DEFAULT_NMS_IOU

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
    (:data:`_SPLIT_MANIFEST_CONFLICT_KEYS`). Shared by ``preflight_config`` and
    :func:`~tcip_mcp.pipelines.data.split_construction.auto_train_val`'s manifest branch, so the
    two report the identical set for one config.
    """
    conflicts = [k for k in _SPLIT_MANIFEST_CONFLICT_KEYS if split_cfg.get(k) is not None]
    if data_cfg.get("coco_json"):
        conflicts.append("coco_json")
    if (data_cfg.get("label_format") or "").lower() == "coco":
        conflicts.append("label_format")
    return sorted(conflicts)

# Lazy imports of heavy dependencies inside tool functions to keep server startup fast.


@mcp.tool()
@audited
def preflight_config(config: dict, smoke: bool = False, overfit: bool = False) -> dict:
    """Validate a training configuration before launching.

    Config structure:
        model_source: {builder, builder_kwargs, task, in_chans}
        data: {images_dir, labels_dir, task}  # known loaders, OR a bespoke
              # {dataset_source: {builder, builder_kwargs, source_files, task}, task}
        training: {batch_size, ...}  # the full key list generic_trainer.train() reads
              # (device/seed/deterministic/mixed_precision/stages/optimizer/scheduler/
              # lr_scaling/stage_warmup_epochs/enforce_monotonic_unfreeze/
              # gradient_accumulation_steps/checkpoint_every_n_epochs/early_stopping/evaluation)
              # is documented on train()'s own docstring, not repeated here, read that for the
              # canonical, always-current list. training_source: optional custom train(ctx) loop.

    Args:
        config: Full training configuration dict.
        smoke: When True, actually build the model and run ``check_model_contract`` (a train+eval
            forward at the resolved in_chans/num_classes/img_size). A contract failure is a
            guaranteed real-run failure, so it is appended to ``issues`` and blocks the launch,
            ``launch_training`` runs this before spawning the training thread. For a task the
            contract has no synthetic batch schema for, one real batch is built from ``data`` and
            used instead; if no batch can be built either, the boundary is unproven and that also
            blocks. Default False keeps the always-on web ``/validate`` path to structural checks
            plus a builder import, no model construction and no forward pass.
        overfit: When True (with ``smoke``), also run the voluntary ``overfit_check`` diagnostic and
            report it under ``overfit_check``, never gating (a noisy-but-valid model can fail it).
            The stored report is already rendered (``model_contract.render_overfit_report``): a
            diverging model's raw losses may hold ``nan``/``inf``, which this JSON-RPC tool cannot
            answer with directly.
    """
    from tcip_mcp.pipelines.schemas import validate_train_config_schema

    # Pydantic schema: type/structure of data/training.
    issues: list[str] = list(validate_train_config_schema(config))
    warnings: list[str] = []

    # model_source presence + builder importability (the one build path).
    model_source = config.get("model_source")
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
    from tcip_mcp.pipelines.model_build import DATASET_SOURCE_KEY, TRAINING_SOURCE_KEY
    training_source = config.get(TRAINING_SOURCE_KEY)
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
    data_cfg = config.get("data")
    if not data_cfg:
        issues.append("Missing 'data' section")
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
        for key in ("images_dir", "labels_dir"):
            path = data_cfg.get(key)
            if not path:
                issues.append(f"Missing 'data.{key}'")
            elif not Path(path).is_dir():
                issues.append(f"Directory not found: data.{key} = '{path}'")

    # Channel firewall: probe one sample raster and check its band count against the declared
    # in_chans, so a channel-wrong train is caught here rather than deep in the training subprocess.
    # Only fires when a raster is actually readable, never a false-fail on an empty/absent dir.
    if isinstance(model_source, dict) and data_cfg:
        from tcip_mcp.pipelines.model_build import declared_in_chans
        declared = declared_in_chans(model_source)
        images_dir = data_cfg.get("images_dir")
        if declared is not None and images_dir and Path(images_dir).is_dir():
            from tcip_mcp.pipelines.data.datasets import IMAGE_EXTS
            sample = next((f for f in sorted(Path(images_dir).iterdir())
                           if f.suffix.lower() in IMAGE_EXTS), None)
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
            from tcip_mcp.pipelines.data.datasets import IMAGE_EXTS
            stems = sorted(f.stem for f in Path(images_dir).iterdir()
                           if f.suffix.lower() in IMAGE_EXTS)
            if stems:
                from tcip_mcp.pipelines.data.splits import resolve_group_key_fn
                try:
                    resolve_group_key_fn(split_cfg_dict.get("group_by", "tile_prefix"), stems,
                                         group_key_map=split_cfg_dict.get("group_key_map"))
                except ValueError as exc:
                    issues.append(f"data.split: {exc}")

    # Split-manifest binding: only what preflight can answer without the run's real admission
    # (bind_manifest_stems itself never runs here; the checks below mirror auto_train_val's).
    manifest_dir = split_cfg_dict.get("manifest_dir")
    if manifest_dir:
        task_for_manifest = (model_source.get("task") if isinstance(model_source, dict) else None) \
            or data_cfg_dict.get("task", "detection")
        if data_cfg_dict.get("val_images_dir"):
            issues.append(
                "data.split.manifest_dir conflicts with data.val_images_dir: two membership "
                "sources for one run's validation split."
            )
        conflicts = _split_manifest_drawn_conflicts(data_cfg_dict, split_cfg_dict)
        if conflicts:
            issues.append(
                f"data.split.manifest_dir conflicts with {conflicts}: a recorded partition and "
                "a drawn split's own parameters/source cannot both govern one run."
            )
        if task_for_manifest not in ("detection", "instance_seg"):
            issues.append(
                f"data.split.manifest_dir names a split manifest, and only detection and "
                f"instance_seg admit through the trainable_stems draw a manifest is drawn "
                f"through; task={task_for_manifest!r} cannot bind to one."
            )
        else:
            from tcip_mcp.dataset_layout import annotation_date
            from tcip_mcp.pipelines.data.splits import manifest_date_key
            from tcip_mcp.tools.data_tools import read_split_manifest_dir

            try:
                manifest = read_split_manifest_dir(manifest_dir)
            except ValueError as exc:
                issues.append(str(exc))
                manifest = None
            if manifest is not None:
                norm_src = _dataset_source_kwargs(task_for_manifest, data_cfg_dict)
                subject, attribute = norm_src.get("subject"), norm_src.get("attribute")
                if (manifest.get("subject"), manifest.get("attribute")) != (subject, attribute):
                    issues.append(
                        f"split manifest was drawn for subject={manifest.get('subject')!r}, "
                        f"attribute={manifest.get('attribute')!r}, but this run is "
                        f"subject={subject!r}, attribute={attribute!r}: a run only binds to its "
                        "own subject's (and attribute's) manifest."
                    )
                labels_dir = data_cfg_dict.get("labels_dir", "")
                run_date = annotation_date(labels_dir)
                declared_date = data_cfg_dict.get("date")
                if declared_date is not None and declared_date != run_date:
                    issues.append(
                        f"data.date={declared_date!r} disagrees with the date "
                        f"data.labels_dir={labels_dir!r} is under ({run_date!r}); a split "
                        "manifest binds under one date, so the negative confirmations and the "
                        "manifest must be read under the same one."
                    )
                else:
                    date_block = (manifest.get("members") or {}).get(manifest_date_key(run_date))
                    if date_block is None:
                        issues.append(
                            f"split manifest at {manifest_dir!r} holds no members under date "
                            f"{run_date!r}; it holds members under "
                            f"{sorted(manifest.get('members') or {})}."
                        )
                    else:
                        from tcip_mcp.pipelines.data.splits import refuse_if_images_root_moved

                        try:
                            refuse_if_images_root_moved(
                                "data.images_dir", data_cfg_dict.get("images_dir"),
                                date_block.get("images_root"), run_date,
                            )
                        except ValueError as exc:
                            issues.append(str(exc))

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
                from tcip_mcp.pipelines.data.datasets import trainable_stems
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

    # Training config validation
    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 2)
    if not isinstance(batch_size, int) or batch_size < 1:
        issues.append("'training.batch_size' must be a positive integer")

    # Per-stage 'epochs' is required; 'lr' is optional (StageSpec) and the trainer
    # reads learning rates from config['optimizer'], never from a stage. Absent
    # stages are fine, launch_training supplies its own default schedule.
    for i, stage in enumerate(train_cfg.get("stages") or []):
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
    eval_cfg = train_cfg.get("evaluation") or config.get("evaluation") or {}
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

            ms = config.get("model_source") or {}
            task = ms.get("task") or (config.get("data") or {}).get("task", "detection")
            dims = resolve_contract_dims(config, task)
            model = build_model(config)
            report = check_model_contract(model, task, **dims)
            batch, why_no_batch = None, None
            if report.get("not_smokeable"):
                # The contract has no synthetic batch schema for this task. Rather than enumerate
                # tasks (a taxonomy) or skip the check (a rail made optional), smoke it against a
                # real batch from the run's own dataset, a better reference than a synthetic one
                # for every task, and the only one for a task the platform does not enumerate.
                batch, why_no_batch = _one_real_batch(task, config)
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
    process or any other concurrent run's process. Use check_training_status to monitor progress;
    it reads the run's own status/metrics from disk, not shared memory.

    Args:
        config: Full training configuration dict with model_source, data, training sections.
        output_dir: Directory for checkpoints and logs. Empty defaults to the experiment store
            (``<project>/.tcip/experiments``, the same base the experiment records use); a
            relative path resolves against the project root, never the server process's cwd.
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
    # run_hpo normalizes separately, inside _apply_hpo_params.
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
        update_status(experiment_id, "running")
    except Exception as exc:  # Experiment tracking is best-effort, but failures must be visible.
        logger.warning("Experiment tracking failed for %s: %s", experiment_id, exc)

    # The child reads its own bootstrap config from here, independent of whether experiment
    # tracking above succeeded, a filesystem hiccup in .tcip/experiments degrades tracking (as it
    # always has) without also preventing the run from training at all. experiment_id is never
    # read from here (see subprocess_worker.py), only passed as the explicit CLI arg below,
    # because this file is written before config["experiment_id"] is guaranteed resolved in the
    # fresh-id-relaunch branch.
    store.replace(launch_config_key(run.output_dir), config)

    # Captured once, beside the child's environment snapshot: the watchdog below writes about
    # this run under the root it launched under, even if this process later adopts another.
    from tcip_mcp.project_paths import project_root

    launch_root = project_root()
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
    import sys

    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    path_entries = [p for p in sys.path if p]
    if existing_pythonpath:
        path_entries = path_entries + [existing_pythonpath]
    env["PYTHONPATH"] = os.pathsep.join(path_entries)

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
    in-memory-only mark, since ``check_training_status`` always defers to disk for a pid-bearing
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
def check_training_status(run_id: str) -> dict:
    """Check the status of a training run.

    Reads the run's own status/metrics from disk whenever its training body runs in a
    subprocess, the in-memory record for a subprocess-delegated run is a launch-time placeholder
    only, since the subprocess mutates its own separate copy in its own process memory, or when
    this process never held the run in memory at all (a different process launched it).

    Args:
        run_id: Training run identifier.
    """
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
    one implementation :func:`list_training_runs` and :func:`inspect_compute_resources` both
    build on, so a subprocess-delegated run's real status is visible to both and neither
    reimplements the merge.

    A live in-memory entry (HPO trials excluded) wins by ``run_id`` over its own disk row: a
    ``pid``-bearing one takes the disk overlay for ``status``/``current_epoch``/``error``/
    ``experiment_id`` (a subprocess-delegated run mutates its own separate copy on disk, so the
    parent-side in-memory record is a stale launch-time placeholder past that point); a
    ``pid``-less one (every synchronous run) is reported from its own in-memory record,
    untouched. Both carry ``external: False`` and an ``experiment_id`` (from the disk overlay
    where there is one, else the run's own config, else ``None``). Rows: this process's own, in
    registry order, then the disk-only rows, sorted by experiment id.
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
            row["experiment_id"] = overlay["experiment_id"]
        merged.append(row)

    live_run_ids = {r["run_id"] for r in live}
    disk_only = [r for r in disk if r["run_id"] not in live_run_ids]
    return merged + disk_only


@mcp.tool()
@audited
def list_training_runs() -> dict:
    """List every training run this platform can currently account for.

    Merges this process's own in-memory registry with every launched run's own record on disk
    (a run this session launched, another process launched, or one that survived a restart); see
    :func:`_all_training_runs` for the merge rule. HPO trials are excluded (they belong to the
    Tuning view). Cost: one status read and one config read per experiment record on disk, plus
    one metrics-log read per launched record for its current epoch.
    """
    return {"runs": _all_training_runs(read_progress=True)}


@mcp.tool()
@audited
def cancel_training(run_id: str) -> dict:
    """Request graceful cancellation of a running training run.

    The trainer stops at the next batch/epoch boundary, still saves ``model_final.pt``
    (so partial progress is recoverable), and sets the run + its experiment to
    'cancelled'. Status updates asynchronously, so the returned status may still read
    'running' immediately after the request.

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

    Not an MCP tool: run through ``scripts/inspect_compute_resources.py``, per the admission
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
    """Dict subclass recording which top-level keys are ever read via ``__getitem__``/``get``/
    ``__contains__``, installed on ``run.config`` for one HPO trial's dispatch, so
    ``unconsumed_params`` reflects genuine runtime access (did anything read this key during
    this trial), not a static comparison against ``train()``'s known key list, which would
    falsely flag a bespoke ``training_source``'s own legitimate custom sweep key.

    Real, stated limitations (never gates the run, warn-only, so a false positive costs a log
    line, not a failed trial): top-level only (a nested read like
    ``ctx.config["optimizer"]["custom_key"]`` isn't seen); ``dict(cfg)``/``**cfg`` copies bypass
    the overrides entirely (CPython copies at the C level); whole-dict iteration
    (``.items()``/``.values()``/``.keys()``) isn't tracked per-key.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.accessed: set[str] = set()

    def __getitem__(self, key: Any) -> Any:
        self.accessed.add(key)
        return super().__getitem__(key)

    def get(self, key: Any, default: Any = None) -> Any:
        self.accessed.add(key)
        return super().get(key, default)

    def __contains__(self, key: Any) -> bool:
        self.accessed.add(key)
        return super().__contains__(key)


def hpo_root(output_dir: str = "", *, root: Path | str | None = None) -> Path:
    """Where HPO sweeps live: ``output_dir`` when the caller named one, else ``.tcip/hpo``
    under ``root`` (default: the platform state root). A relative ``output_dir`` resolves
    against the project root, never the server process's cwd.

    ``root`` lets a caller that already knows which project a sweep belongs to (its own
    registry entry's launch root) resolve its directory there, rather than under whatever
    root this process currently has pinned.

    The one resolver for that decision. Anything that has to find a sweep on disk (the
    Tuning routes included) calls this rather than rebuilding the same default.
    """
    from tcip_mcp.project_paths import project_root, resolve_output_path

    if output_dir:
        return resolve_output_path(output_dir)
    base = Path(root) if root is not None else project_root()
    return base / ".tcip" / "hpo"


def sweep_dir(study_name: str, output_dir: str = "", *, root: Path | str | None = None) -> Path:
    """One sweep's own directory: its manifest, its ``trial_<id>`` dirs, and (because Ray
    is handed ``storage_path=hpo_root`` and ``name=study_name``) Ray's experiment store."""
    return hpo_root(output_dir, root=root) / study_name


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
    ``last_writer_wins``: ``run_hpo`` holds the manifest in memory for the whole sweep and
    writes the whole document at each state change, so nothing merges into what is on disk.
    """
    return Key(SWEEP_MANIFEST_STORE, str(hpo_root(output_dir, root=root).resolve()),
               (_sweep_name(study_name), "manifest"))


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


def _run_hpo_trial(config: dict, report, base_config: dict, trial_dir: str) -> None:
    """Train one HPO trial and ``report`` its resolved selection metric, in whatever direction
    that metric's own declaration says is better (``evaluation.HIGHER_IS_BETTER_BY_METRIC``, via
    :func:`~tcip_mcp.pipelines.training.generic_trainer.resolve_selection_metric`), never a fixed
    minimize convention.

    ``report(value)`` feeds the Ray Tune searcher/scheduler; call it each epoch (so a scheduler
    can prune) and once at the end with the best value this trial actually reached (Tune's default
    ``get_best_result`` scope reads only the last value each trial reported, so the run's best
    epoch would otherwise be lost behind a worse later one). A trial that never reports a real
    value, before training starts or on any failure, reports the losing side of its own direction
    as that final value instead of leaving Tune with nothing for it, so a trial with nothing real
    to say can never win either a minimize or a maximize sweep. Trials train under the final run's
    regime, same augmentation, imbalance handling, and dispatch: a ``training_source`` in
    ``base_config`` actually runs under that loop here too, not always the stock trainer, or the
    selected hyperparameters won't transfer.
    """
    merged = _apply_hpo_params(base_config, config)

    from tcip_mcp.pipelines.training.envelope import TrainContext, dispatch_train_body
    from tcip_mcp.pipelines.training.evaluation import HIGHER_IS_BETTER_BY_METRIC
    from tcip_mcp.pipelines.training.generic_trainer import (
        _improves, resolve_selection_metric, task_collate, seeded_loader_kwargs,
        stamp_effective_data_geometry,
    )
    from tcip_mcp.pipelines.training.run_registry import create_run
    from tcip_mcp.pipelines.data.samplers import build_sampler
    from tcip_mcp.pipelines.data.split_construction import auto_train_val
    from torch.utils.data import DataLoader

    model_source = merged.get("model_source")
    # setdefault, not get: the geometry stamp below mutates this dict and must land in the
    # resolved-config snapshot written from merged.
    data_cfg = merged.setdefault("data", {})
    train_cfg = merged.get("training", {})
    task = (model_source.get("task") if model_source else None) or data_cfg.get("task", "detection")
    eval_cfg = merged.get("evaluation") or {}
    try:
        higher_is_better = HIGHER_IS_BETTER_BY_METRIC[resolve_selection_metric(
            task, eval_cfg.get("trait"), eval_cfg.get("selection_metric"))]
    except Exception:
        # Undeclared direction, an unregistered trait, or any other resolution failure; the
        # trial fails below either way, this only decides which sentinel that failure reports.
        higher_is_better = False
    losing_side = float("-inf") if higher_is_better else float("inf")

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
        report(best["value"])  # the trial's best reported value, or the losing side if it reported none
    except Exception as e:
        logger.warning("HPO trial failed: %s", e)
        report(losing_side)
    finally:
        # Surface any swept param no consumer touched. Warn-only, never gates the trial.
        unconsumed = sorted((set(config.keys()) - _HPO_KNOWN_KEYS) - tracked_config.accessed)
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
def run_hpo(
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
    TensorBoard logdir). The full result is written alongside as ``<study_name>.json``.

    Refuses (``{"error": ..., "issues": [...]}``, nothing minted) an unimportable builder or
    training source, or a config with no ``data`` section, at the config a trial would actually
    train under (``base_config`` with the search space's first sampled point applied), rather
    than reporting as the losing side in every trial.

    Args:
        base_config: Base training config each trial modifies.
        param_space: Param-space dict (see ``hpo.get_default_space``); default when omitted.
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
    """
    from tcip_mcp.pipelines.training.hpo import tune_search, get_default_space

    if param_space is None:
        param_space = get_default_space()
    # Both reach a stored record: the space into the sweep manifest, the base config into
    # every trial's resolved config once a sampled point is applied to it.
    check_json_value(param_space, path="param_space")
    check_json_value(base_config, path="base_config")

    # Structural preflight over every point the search space could resolve a trial's builder or
    # data section to, not only the first sampled corner, before anything is minted.
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

    from tcip_mcp.pipelines.training.evaluation import HIGHER_IS_BETTER_BY_METRIC
    from tcip_mcp.pipelines.training.generic_trainer import resolve_selection_metric

    # Ray forbids setting metric/mode anywhere but the Tuner, so the direction is resolved once
    # here, from base_config, and every trial's own resolution (_run_hpo_trial) must agree with it.
    hpo_model_source = base_config.get("model_source") or {}
    hpo_eval_cfg = (base_config.get("training") or {}).get("evaluation") \
        or base_config.get("evaluation") or {}
    hpo_task = hpo_model_source.get("task") \
        or (base_config.get("data") or {}).get("task", "detection")
    hpo_metric = resolve_selection_metric(
        hpo_task, hpo_eval_cfg.get("trait"), hpo_eval_cfg.get("selection_metric"))
    hpo_mode = "max" if HIGHER_IS_BETTER_BY_METRIC[hpo_metric] else "min"

    import uuid
    from datetime import datetime, timezone

    hpo_dir = hpo_root(output_dir)
    hpo_dir.mkdir(parents=True, exist_ok=True)
    study_name = study_name or f"hpo_{uuid.uuid4().hex[:8]}"
    sweep_root = sweep_dir(study_name, output_dir)
    sweep_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "study_name": study_name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "n_trials": n_trials,
        "search_alg": search_alg,
        "scheduler": scheduler,
        "param_space": param_space,
        "sweep_dir": str(sweep_root),
    }
    manifest_key = sweep_manifest_key(study_name, output_dir)
    store.replace(manifest_key, manifest)

    def objective_fn(config: dict, report) -> None:
        try:
            from ray import tune as _tune
            tid = _tune.get_context().get_trial_id()
        except Exception:
            tid = uuid.uuid4().hex[:8]
        _run_hpo_trial(config, report, base_config, str(sweep_root / f"trial_{tid}"))

    try:
        result = tune_search(
            objective_fn=objective_fn,
            param_space=param_space,
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
        )
    except Exception as exc:
        manifest.update(status="failed", error=str(exc),
                        finished_at=datetime.now(timezone.utc).isoformat())
        store.replace(manifest_key, manifest)
        raise

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
    # best_value_state (stored_number's sibling for a non-finite best_value) rides along whenever the search produced one.
    manifest_result = {k: result.get(k) for k in ("best_params", "best_value", "n_trials")}
    if "best_value_state" in result:
        manifest_result["best_value_state"] = result["best_value_state"]
    manifest.update(
        status="completed",
        finished_at=datetime.now(timezone.utc).isoformat(),
        result=manifest_result,
    )
    # Durable result records (best-effort, a write hiccup must not sink a completed sweep).
    try:
        store.replace(manifest_key, manifest)
        store.replace(study_result_key(study_name, output_dir), result)
    except (OSError, StoreError):
        logger.warning("could not persist the hpo result for %s", study_name, exc_info=True)
    return result


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
    """Every point ``run_hpo``'s preflight must check: the first sampled corner, plus one variant
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
    create_experiment(fresh_id, config, parent_experiment=experiment_id, data_source=data_source,
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
        from tcip_mcp.pipelines.training.generic_trainer import task_collate

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

    from tcip_mcp.pipelines.data.datasets import IMAGE_EXTS

    stems = sorted(f.stem for f in Path(images_dir).iterdir() if f.suffix.lower() in IMAGE_EXTS)
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


def _spatial_split_raster_identity(data_cfg: dict, stem: str) -> dict | None:
    """This mosaic's own :func:`~tcip_mcp.pipelines.raster_source.raster_content_identity`, best
    effort: recorded into ``spatial_manifest`` at spatial-split time (the training source is first
    known to be a raster here), read back at export time (``inference_tools.
    _export_predictions_raster``) to gate a block-calibrated bundle's claim scope to this exact
    mosaic. A provenance write must never sink a launch: an unreadable/unsupported source (a
    bespoke ``dataset_source``, a corrupt file) logs and returns ``None`` rather than raising, the
    same posture ``persist_split_manifest`` already takes for its own best-effort writes.
    """
    try:
        from tcip_mcp.pipelines.derivations import probe_channels
        from tcip_mcp.pipelines.image_utils import resolve_image_source
        from tcip_mcp.pipelines.raster_source import content_identity

        source = resolve_image_source(data_cfg.get("images_dir", ""), stem)
        nc = probe_channels(source)
        identity = content_identity(source, nc)
        import dataclasses
        return dataclasses.asdict(identity)
    except Exception as exc:  # noqa: BLE001, best-effort provenance, never sinks the split/launch
        logger.warning("raster content identity for %r could not be recorded: %s", stem, exc)
        return None


def _checked_label_format(task: str, data_cfg: dict, src: dict) -> str | None:
    """The per-image label format this run's ``data.labels_dir`` holds (``"json"``, ``"coco"``
    never returned, see below), or ``None`` for a task/config the check does not apply to.

    Refuses a dataset-level assembled COCO document sitting in ``data.labels_dir`` rather than
    per-image label files: a caller-fixable config error, since the per-image files in that
    directory would be shadowed by the assembled export, silently training on the wrong source
    being worse than refusing. Called once per run, by each caller of
    :func:`_build_full_admitted_dataset` ahead of its own handler and never from inside the
    helper, so neither the auto path's degrade handler nor a manifest bind can catch this
    refusal and fold it into "training without validation".
    """
    if task not in ("detection", "instance_seg") or data_cfg.get("label_format") or data_cfg.get("coco_json"):
        return None
    labels_dir, images_dir = src.get("labels_dir", ""), src.get("images_dir", "")
    if not (labels_dir and images_dir):
        return None
    from tcip_mcp.pipelines.data.datasets import dir_label_format, first_labels_json

    fmt = dir_label_format(labels_dir)
    if fmt == "coco":
        offending = first_labels_json(labels_dir)
        raise ValueError(
            f"data.labels_dir={labels_dir!r} holds a dataset-level COCO file "
            f"({offending}): if the per-image label files in this directory are the ones "
            "that should train, move it out of data.labels_dir; if this COCO export is "
            "the intended label source, set data.coco_json (or data.label_format='coco') "
            "to train on it directly."
        )
    return fmt


def _build_full_admitted_dataset(
    task: str, data_cfg: dict, src: dict, transforms, detected_label_format: str | None,
):
    """The full, admitted-set dataset for one run's data config, plus the ``build_dataset`` kwargs
    that produced it: one implementation, called both by the auto-split path (inside its own
    degrading handler) and by a split-manifest bind (unwrapped, so a real build failure raises
    rather than degrading to no validation over a recorded partition), so the two can never
    disagree about what this run admits.

    ``detected_label_format`` is the caller's own :func:`_checked_label_format` result: a
    dataset-level COCO document sitting in ``data.labels_dir`` is refused there, ahead of the
    caller's own handler, never re-read (and so never re-refused) here.

    Returns ``(full_ds, stems, build_src)``: ``stems`` is the task path's own admitted set
    (``full_ds.stems``/``full_ds._stems``), and ``build_src`` is ``src`` plus any assembled
    ``coco_data``/``label_format``/``num_classes`` the COCO branch added, for a caller that narrows
    it with ``stems=`` afterward.
    """
    from tcip_mcp.pipelines.data.datasets import build_dataset

    build_src = dict(src)
    labels_dir, images_dir = src.get("labels_dir", ""), src.get("images_dir", "")

    if detected_label_format == "json":
        from tcip_mcp.pipelines.data.datasets import _resolve_registry_id_map, assemble_coco
        subject, attribute = src.get("subject"), src.get("attribute")
        _reg, id_map = _resolve_registry_id_map(labels_dir, subject, attribute)
        assert subject is not None, "_resolve_registry_id_map already refused an empty subject"
        build_src["coco_data"] = assemble_coco(
            labels_dir, images_dir, subject=subject, attribute=attribute, id_map=id_map,
            date=src.get("date"))
        build_src["label_format"] = "coco"
        build_src["num_classes"] = len(id_map)

    full_ds = build_dataset(task, **build_src, transforms=transforms)
    stems = list(getattr(full_ds, "stems", None) or getattr(full_ds, "_stems", []))
    return full_ds, stems, build_src


@mcp.tool()
@audited
def evaluate_model(
    run_id_or_ckpt: str,
    images_dir: str,
    labels_dir: str = "",
    task: str = "detection",
    conf_threshold: float = DEFAULT_CONF,  # report/select at the ship point
    iou_threshold: float = 0.5,
    iou_type: str | None = None,
    max_dets: int | None = None,
    tiling: dict | None = None,
    use_tiled_inference: bool = False,
    global_nms_iou: float = DEFAULT_NMS_IOU,
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
            Either way the resolved checkpoint must be registered under this process's project
            root (``register_model``, explicit mode for a foreign or bespoke checkpoint) or this
            door refuses before loading it.
        images_dir: Images directory for the evaluation split.
        labels_dir: Labels dir (detection/instance_seg), masks dir (semantic_seg), or the GT CSV
            path (classification/ordinal/regression, one row per image stem).
        task: Task type.
        conf_threshold: Operating confidence for P/R/F1.
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
        global_nms_iou: Cross-tile global NMS IoU threshold (tiled paths only).
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

    from tcip_mcp.pipelines.training.generic_trainer import checkpoint_key, task_collate
    from tcip_mcp.pipelines.training.run_registry import get_run
    from tcip_mcp.pipelines.training.evaluation import (
        run_full_frame_evaluation, run_test_evaluation,
    )
    from tcip_mcp.pipelines.data.datasets import build_dataset
    from tcip_mcp.pipelines.resolution import DEFAULT_MAX_DETS

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
            manifest_stems, _group_by, _group_key_map, _excluded, cal_date = \
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

    # Delivery-grade full-frame path (tiled inference + full-frame GT matching).
    if use_tiled_inference and task == "detection":
        tcfg = tiling or run_tiling or {}
        # An explicit caller max_dets is honored verbatim (no rescuing sentinel);
        # None resolves to the delivery-grade default (dense full-frame scenes aren't truncated).
        resolved_max_dets = DEFAULT_MAX_DETS if max_dets is None else max_dets
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
                max_dets=resolved_max_dets, trait=trait, date=date,
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
        conf_threshold=conf_threshold, iou_threshold=iou_threshold,
        iou_type=iou_type, max_dets=resolved_max_dets, tiling=tiling, trait=trait,
        split_manifest_dir=split_manifest_dir,
        evaluated_stem_count=evaluated_stem_count,
    )
