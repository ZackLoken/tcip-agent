"""Training MCP tools, config validation, launch training, HPO, status."""

from __future__ import annotations

import contextvars
import itertools
import json
import logging
import os
import subprocess
import sys
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path, PureWindowsPath
from typing import Any, Iterator, NamedTuple, Sized

from tcip_store import (
    LOG_JSON,
    RECORD_JSON,
    BadKey,
    DecodeError,
    Key,
    StoreDescriptor,
    StoreError,
    VersionConflict,
    canonical_path,
    check_json_value,
    register_store,
    store,
    stored_number,
)
from tcip_store.file_backend import RootedFileLocator

from tcip_mcp.server import mcp
from tcip_mcp.audit import audited
from tcip_mcp.pipelines.data.split_construction import split_seed
from tcip_mcp.pipelines.data.splits import DEFAULT_GROUP_BY
from tcip_mcp.pipelines.model_build import run_task
from tcip_mcp.pipelines.resolution import DEFAULT_POSTPROCESS

logger = logging.getLogger(__name__)

# Round-robins unpinned concurrent launches across available GPUs (no-op with 0-1 devices).
_gpu_round_robin = itertools.count()

# Serializes the overfit diagnostic's reseed-run-restore, since the RNG streams are process-global.
_OVERFIT_CHECK_LOCK = threading.Lock()

# A reconstructed non-terminal run reads "running" only while its heartbeat is within this
# window, past it a live process is presumed dead and it reads "interrupted".
TCIP_HEARTBEAT_STALE_SECONDS = float(os.environ.get("TCIP_HEARTBEAT_STALE_SECONDS", "600"))

# What the calling route declared about itself, read once by launch_training on the calling
# thread before create_run/_ensure_experiment run; unset for a caller with nothing to declare.
declared_launcher: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "declared_launcher", default=None,
)


@contextmanager
def declare_launcher(name: str) -> Iterator[None]:
    """Declare ``name`` as this thread's launcher for the one ``launch_training`` call inside the
    block. Scoped to the calling thread only: a launch's own background watchdog thread and the
    training subprocess never see it.
    """
    token = declared_launcher.set(name)
    try:
        yield
    finally:
        declared_launcher.reset(token)


def _resolve_launched_by() -> dict[str, Any]:
    """The launcher declaration ``launch_training`` stamps on the experiment it creates: the
    declared name when :func:`declare_launcher` set one, else the connected MCP agent's identity
    when a handshake is in force, else ``process`` for a caller with neither. Resolved once, on
    the calling thread, before ``create_run``/``_ensure_experiment`` run.
    """
    from tcip_mcp import agent_identity

    name = declared_launcher.get()
    if name is not None:
        return {"launcher": name}
    identity = agent_identity.current()
    if identity is not None:
        return {"launcher": "agent", **agent_identity.audit_fields()}
    return {"launcher": "process"}

_SELECTION_CONFLICT_KEYS = (
    "group_by", "group_key_map", "val_ratio", "seed", "stratify_foreground",
    "test_ratio", "reserve_calibration_fraction",
)


def _split_selection_drawn_conflicts(split_cfg: dict) -> list[str]:
    """Every key under ``data.split`` that ``data.split.selection_dir`` conflicts with: a drawn
    split's own parameters (:data:`_SELECTION_CONFLICT_KEYS`). ``seed`` is admitted only when
    ``data.split.redraw_within_selection`` is true; a seed otherwise conflicts by name.
    """
    keys: tuple[str, ...] = _SELECTION_CONFLICT_KEYS
    if split_cfg.get("redraw_within_selection"):
        keys = tuple(k for k in keys if k != "seed")
    return sorted(k for k in keys if split_cfg.get(k) is not None)


def _redraw_flag_issue(split_cfg: dict) -> str | None:
    """The objection ``data.split.redraw_within_selection`` raises that
    :func:`_split_selection_drawn_conflicts` cannot: the flag set true with no ``seed`` beside it.
    ``None`` when the flag is unset, false, or paired with a seed.
    """
    if split_cfg.get("redraw_within_selection") and split_cfg.get("seed") is None:
        return (
            "data.split.redraw_within_selection=true requires data.split.seed: the seed the "
            "redraw draws train and val at."
        )
    return None


def _data_dir_issues(data_cfg: dict) -> list[str]:
    """Every objection to the data locations this run's own producer reads: ``data.images_dir``
    or ``data.labels_dir`` missing, or naming a path that does not exist, whatever the task and
    whatever builds its loaders. Empty for a config bound to a selection.
    """
    split_cfg = data_cfg.get("split")
    if isinstance(split_cfg, dict) and split_cfg.get("selection_dir"):
        return []
    issues: list[str] = []
    for name in ("images_dir", "labels_dir"):
        path = data_cfg.get(name)
        if not path:
            issues.append(f"Missing 'data.{name}'")
        elif not Path(path).exists():
            # Named without claiming a shape: what ground truth is there is the producer's own
            # read, and a config pointing at nothing is the only fact this check has.
            issues.append(f"Not found: data.{name} = '{path}'")
    return issues


class RunPopulation(NamedTuple):
    """What a run would train and validate over: its own samples, the class space they were
    admitted under, what the admission dropped, and what it refused or warned about."""

    samples: list
    scope: Any
    counts: dict[str, int]
    issues: list[str]
    warnings: list[str]


def _run_population(data_cfg: dict, selection) -> RunPopulation:
    """The samples this run would train and validate over, as the producer names them, with the
    class space it admitted them under: the recorded train and val samples of the selection it is
    bound to, or the producer's own admission over the locations its config names
    (:func:`~tcip_mcp.pipelines.data.label_queries.admit`).

    Its ``issues`` are the admission's own refusals, in the words the run would refuse with (a
    label document that will not decode, a dataset-level export where per-image documents belong, a
    document directory with no subject to admit under). Its ``warnings`` name a confirmed negative
    whose label now holds annotations, which admission excludes. A bucket naming two files under
    one stem propagates (:class:`~tcip_mcp.pipelines.image_utils.AmbiguousImageStem`).
    """
    from tcip_mcp.pipelines.data.selection import ClassScope

    if selection is not None:
        return RunPopulation(selection.trainable(), selection.scope, {}, [], [])
    empty = RunPopulation([], ClassScope(), {}, [], [])
    if _data_dir_issues(data_cfg):
        return empty  # preflight names each of them

    from tcip_annotation.json_io import UnreadableLabelDocument
    from tcip_store import SchemaVersionRefused

    from tcip_mcp.pipelines.data.label_queries import admit_run
    from tcip_mcp.pipelines.image_utils import AmbiguousImageStem

    contradicted: set[str] = set()
    try:
        admitted = admit_run(data_cfg, contradicted_out=contradicted)
    except UnreadableLabelDocument as exc:
        # A run over this ground truth fails on the same file, so this blocks, not warns.
        return empty._replace(issues=[f"data.labels_dir: {exc}"])
    except SchemaVersionRefused as exc:
        return empty._replace(
            issues=[f"a .bandgroup manifest under {data_cfg['images_dir']} could not be read: "
                    f"{exc}"])
    except AmbiguousImageStem:
        raise
    except (OSError, ValueError) as exc:
        # The launch admits through this same producer, so whatever refuses it refuses there too.
        return empty._replace(issues=[f"data: {exc}"])
    warnings: list[str] = []
    if contradicted:
        warnings.append(
            f"data: {sorted(contradicted)} are recorded negative for the subject but their label "
            "file now holds subject annotations; the stored negative is stale, they train on "
            "their labeled content instead, and the confirmation needs re-review."
        )
    return RunPopulation(
        admitted.every_sample(), admitted.scope, admitted.counts, [], warnings)


def _selection_dir_conflicts(data_cfg: dict) -> list[str]:
    """Every objection a ``data.split.selection_dir`` binding raises from a run's data section
    alone, computed without reading any selection: the drawn-split key conflicts
    (:data:`_SELECTION_CONFLICT_KEYS`), and a ``redraw_within_selection`` flag with no seed beside
    it (:func:`_redraw_flag_issue`).
    """
    split_cfg_raw = data_cfg.get("split")
    split_cfg: dict = split_cfg_raw if isinstance(split_cfg_raw, dict) else {}

    issues: list[str] = []
    conflicts = _split_selection_drawn_conflicts(split_cfg)
    if conflicts:
        issues.append(
            f"data.split.selection_dir conflicts with {conflicts}: a recorded partition and "
            "a drawn split's own parameters/source cannot both govern one run."
        )
    flag_issue = _redraw_flag_issue(split_cfg)
    if flag_issue:
        issues.append(flag_issue)
    return issues


def _selection_dependent_issues(selection, selection_dir: str) -> list[str]:
    """Every objection that needs the selection itself, read at ``selection_dir``, to answer: it
    names a subject when the ground truth is a label document
    (:func:`~tcip_mcp.pipelines.data.selection.unscoped_document_issue`), and its train and val
    sides are both populated. Whether the selected loader can read the ground truth these samples
    name is that loader's own refusal.
    """
    from tcip_mcp.pipelines.data.selection import unscoped_document_issue

    issues: list[str] = []
    unscoped = unscoped_document_issue(selection, selection_dir)
    if unscoped:
        issues.append(unscoped)
    counts = selection.counts()
    if not counts["train"] or not counts["val"]:
        issues.append(
            f"the selection at {selection_dir} leaves an empty side (train={counts['train']}, "
            f"val={counts['val']}); a run needs both."
        )
    return issues


def selection_compatibility(config: dict, selection, selection_dir: str) -> list[str]:
    """Every objection a launch binding ``config`` to ``selection`` (read at ``selection_dir``)
    would raise: :func:`_selection_dir_conflicts` (config-only) composed with
    :func:`_selection_dependent_issues` (needs the selection in hand).
    """
    issues = _selection_dir_conflicts(config.get("data") or {})
    issues.extend(_selection_dependent_issues(selection, selection_dir))
    return issues


def candidate_config_with_selection(config: dict, selection_dir: str) -> dict:
    """The launch config choosing ``selection_dir`` over ``config``'s own "As recorded" data
    section would build: ``data.split`` replaced wholesale by ``{"selection_dir": selection_dir}``,
    and the recorded class scope emptied, since a bound run reads its scope off the selection.
    """
    from tcip_mcp.pipelines.data.selection import ClassScope

    data_cfg_raw = config.get("data")
    data_cfg: dict = {**data_cfg_raw} if isinstance(data_cfg_raw, dict) else {}
    ClassScope().onto(data_cfg)
    data_cfg["split"] = {"selection_dir": selection_dir}
    return {**config, "data": data_cfg}

# Lazy imports of heavy dependencies inside tool functions to keep server startup fast.


def preflight_config(config: dict, smoke: bool = False, overfit: bool = False) -> dict:
    """Validate a training configuration before launching.

    Config structure, one placement for everything::

        model_source: {builder, builder_kwargs, task, in_chans}
        data: {images_dir, labels_dir, task}    # known loaders, or a bespoke
                                                # {dataset_source: {builder, ...}, task}
        batch_size, stages, mixed_precision, device, seed, ...   # every key
                                                # generic_trainer.train() reads, at the top
                                                # level; train()'s own docstring is the list
        evaluation: {trait, selection_metric, ...}
        training_source: optional custom train(ctx) loop.

    A nested ``training`` section is refused by name (``schemas.TrainConfigSchema``).

    Args:
        config: Full training configuration dict.
        smoke: When True, actually build the model and run ``check_model_contract`` (a train+eval
            forward at the resolved in_chans/num_classes/img_size). A contract failure is appended
            to ``issues`` and blocks the launch. For a task the contract has no synthetic batch
            schema for, one real batch is built from ``data`` and used instead; if no batch can be
            built either, that also blocks. Default False keeps a plain call to structural checks
            plus a builder import.
        overfit: When True (with ``smoke``), also run the voluntary ``overfit_check`` diagnostic
            and report it under ``overfit_check``, never gating. The stored report is already
            rendered (``model_contract.render_overfit_report``), with non-finite losses rendered.
    """
    from tcip_mcp.pipelines.schemas import validate_train_config_schema
    from tcip_mcp.pipelines.model_build import DATASET_SOURCE_KEY, MODEL_SOURCE_KEY, TRAINING_SOURCE_KEY

    issues: list[str] = list(validate_train_config_schema(config))
    warnings: list[str] = []

    # model_source presence + builder importability (the one build path).
    model_source = config.get(MODEL_SOURCE_KEY)
    if not model_source:
        issues.append("Missing 'model_source' section")
    elif not isinstance(model_source, dict) or not model_source.get("builder"):
        issues.append("model_source must be a dict with a 'builder' (module:function)")
    else:
        from tcip_mcp.pipelines.model_build import import_source_builder
        try:
            import_source_builder(model_source)
        except Exception as exc:
            issues.append(f"model_source.builder not importable: {exc}")

    # training_source seam (mirrors model_source/dataset_source above), a bare "module:function"
    # string, not a dict.
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
    elif not isinstance(data_cfg, dict):
        issues.append("'data' must be a dict")
    else:
        if data_cfg.get(DATASET_SOURCE_KEY) is not None:
            # Bespoke dataset seam (mirrors model_source): the builder must import. The data the
            # platform's own producer reads is still required, and _data_dir_issues says so.
            dataset_source = data_cfg[DATASET_SOURCE_KEY]
            if not isinstance(dataset_source, dict) or not dataset_source.get("builder"):
                issues.append(
                    "data.dataset_source must be a dict with a 'builder' (module:function)")
            else:
                from tcip_mcp.pipelines.model_build import import_source_builder
                try:
                    import_source_builder(dataset_source)
                except Exception as exc:
                    issues.append(f"data.dataset_source.builder not importable: {exc}")
        issues.extend(_data_dir_issues(data_cfg))

    data_cfg_dict: dict = data_cfg if isinstance(data_cfg, dict) else {}
    split_cfg_raw = data_cfg_dict.get("split")
    split_cfg_dict: dict = split_cfg_raw if isinstance(split_cfg_raw, dict) else {}

    # A bound selection is read once, before anything below reads a population: its own samples
    # are this run's membership, so nothing here admits out of the config's directories.
    selection_dir = split_cfg_dict.get("selection_dir")
    selection = None
    if selection_dir:
        from tcip_mcp.pipelines.data.selection import read_selection

        # Config-only issues fire before the read, so a moved selection never hides them.
        issues.extend(_selection_dir_conflicts(data_cfg_dict))
        try:
            selection = read_selection(selection_dir)
        except ValueError as exc:
            issues.append(str(exc))
        else:
            issues.extend(_selection_dependent_issues(selection, selection_dir))
            if split_cfg_dict.get("redraw_within_selection"):
                warnings.append(
                    "data.split.redraw_within_selection=true: this run redraws train and "
                    "val inside the selection's own train and val samples at this seed; "
                    "the selection's calibration side stays untouched."
                )
                from tcip_mcp.pipelines.data.splits import redraw_pool, redraw_starved_issue

                starved = redraw_starved_issue(*redraw_pool(selection), selection_dir=selection_dir,
                                               seed=split_cfg_dict.get("seed"))
                issues.extend([starved] if starved else [])

    # The run's own membership and sources, resolved once: every leg below reads it rather than
    # listing a directory or admitting again of its own.
    run = _run_population(data_cfg_dict, selection)
    population, sample_counts = run.samples, run.counts
    issues.extend(run.issues)
    warnings.extend(run.warnings)

    # The run's own sizes, read the way the run reads them, so whatever refuses the launch refuses
    # here: a channel-wrong train is caught before the subprocess, declared width or not.
    from tcip_mcp.pipelines.data.datasets import stated_sizes
    from tcip_mcp.pipelines.model_build import run_in_chans

    sizes: dict[str, int] = stated_sizes(data_cfg_dict)
    if population:
        from tcip_annotation.json_io import UnreadableLabelDocument
        from tcip_mcp.pipelines.data.datasets import resolve_sizes

        try:
            sizes = resolve_sizes(run_task(config), data_cfg_dict, population,
                                  data_cfg_dict.get(DATASET_SOURCE_KEY) or None)
        except (ValueError, UnreadableLabelDocument) as exc:
            issues.append(f"data: {exc}")
    channels = sizes.get("num_channels")
    declared = run_in_chans(model_source, None)  # the model's own, for the comparison below
    if channels is not None and declared is not None:
        from tcip_mcp.pipelines.resolution import (
            ResolvedBundle, default as _resolved_default, validate_resolved_bundle,
        )
        b = ResolvedBundle(trait="", dataset_hash=None, params={
            "in_chans": _resolved_default("in_chans", declared)})
        issues.extend(validate_resolved_bundle(b, probed_channels=int(channels)))

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
            elif population:
                known = {str(Path(s.source).resolve()) for s in population}
                bad = sorted({
                    label for label, _ in sampling_record.windows
                    if str(Path(label).resolve()) not in known
                })
                image_stats_containment = "checked"
                if bad:
                    issues.append(
                        f"model_source.image_stats_sampling names path(s) {bad} outside this "
                        f"run's own {len(known)} source(s)."
                    )
            else:
                # A run whose membership resolved to nothing here has no sources to check the
                # window paths against; say so rather than silently skip or pass.
                image_stats_containment = "not_checked"

    # Split-policy validation over the run's own members: an unrecognized ``group_by`` or an
    # incomplete ``group_key_map`` is caught here rather than deep in ``auto_train_val``.
    if population and (split_cfg_dict.get("group_by") or split_cfg_dict.get("group_key_map")):
        from tcip_mcp.pipelines.data.label_queries import admission_date
        from tcip_mcp.pipelines.data.splits import recorded_group_key_fn
        try:
            # Through the producers' own derivation, over the capture date they admit
            # under: resolving a second spelling here refuses the map the draw requires.
            recorded_group_key_fn(
                split_cfg_dict.get("group_by", DEFAULT_GROUP_BY),
                date=admission_date(data_cfg_dict.get("labels_dir", "")),
                stems=sorted({s.member for s in population}),
                group_key_map=split_cfg_dict.get("group_key_map"))
        except ValueError as exc:
            issues.append(f"data.split: {exc}")

    # Four-way spatial split feasibility (reserve_calibration_fraction, opt-in): must refuse by
    # name when infeasible, not silently degrade to no validation (see the helper's own docstring).
    reserve_cal_frac = split_cfg_dict.get("reserve_calibration_fraction")
    if reserve_cal_frac:
        issues.extend(_reserve_calibration_feasibility_issues(
            run_task(config), data_cfg_dict, split_cfg_dict, reserve_cal_frac,
            run=run, sizes=sizes, smoke=smoke))

    # Trainable-sample coverage, never gating: a run admitting a fraction of its annotated images
    # would otherwise read "valid, no warnings" while training on far fewer than expected.
    if sample_counts:
        dropped = {k: v for k, v in sample_counts.items()
                   if k not in ("annotated", "confirmed_negative") and v}
        total = sum(sample_counts.values())
        n_dropped = sum(dropped.values())
        if n_dropped and total:
            warnings.append(
                f"data: {n_dropped}/{total} candidate images ({n_dropped / total:.0%}) will "
                f"not train, {dict(sorted(dropped.items()))}. "
                f"{len(population)} stem(s) admitted.")

    # Fail fast on an explicit selection_metric that is undeclared or, with a center-match trait,
    # comparability-only, at validation time rather than mid-run.
    eval_cfg = config.get("evaluation") or {}
    if not isinstance(eval_cfg, dict):
        issues.append(
            f"'evaluation' must be a mapping (trait/selection_metric/... keys), got "
            f"{type(eval_cfg).__name__}"
        )
        eval_cfg = {}
    if eval_cfg.get("selection_metric"):
        from tcip_mcp.pipelines.training.generic_trainer import config_selection_metric

        try:
            config_selection_metric(config)
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

            task = run_task(config)
            # The sizes preflight resolved above, this run having recorded none yet; a run that
            # states none has no synthetic shape and smokes against a real batch.
            dims = resolve_contract_dims(config, task, scope=run.scope, sizes=sizes)
            model = build_model(config)
            report = check_model_contract(model, task, dims=dims)
            batch, why_no_batch = None, None
            if report.get("not_smokeable"):
                # No synthetic batch for this task or this width: smoke against a real batch from
                # the run's own dataset, the only reference the platform has not guessed at.
                batch, why_no_batch = _one_real_batch(task, config)
                if batch is not None:
                    report = check_model_contract(model, task, sample_batch=batch)
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
                        raw_report = overfit_check(model, task, sample_batch=batch, dims=dims)
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

    The run's training body (dataset build, model forward/backward, checkpointing) executes in a
    separate OS process. Use monitor_training to monitor progress and cancel_training to stop a
    run. The platform itself stops a run only when it is dead (two consecutive full training passes
    with no finite batch loss) or stagnant against its own validation metric (early stopping); a
    run launched with no validation loader gets divergence as its only automatic stop.

    Stamps the resolved experiment's ``status.json`` with who launched this run (``launched_by``):
    the name :func:`declare_launcher` declared for this thread, else the connected MCP agent's
    identity when a handshake is in force, else ``"process"``.

    Args:
        config: Full training configuration dict: model_source and data, with every training
            setting (batch_size, stages, evaluation, seed, device) at the top level. An
            ``experiment_id`` names the record to launch under (one run's immutable record,
            ``tcip_mcp.experiments``; created if absent, reused while pristine, forked if it
            already has history); absent, a fresh id is minted.
        output_dir: Base directory for checkpoints and logs. Empty defaults to the experiment store
            (``<project>/.tcip/experiments``); a relative path resolves against the platform state
            root, never the server process's cwd. The run's own artifacts land under
            ``output_dir/<experiment_id>``.
        resume_from: Optional path to a ``checkpoint_epoch_*.pt`` to resume from (restores model +
            optimizer + scheduler + scaler and continues).
        max_wall_clock_seconds: Optional hard timeout. If the training process hasn't exited on its
            own by then, it is terminated and the run marked failed with that reason, with no
            cooperative grace period. Omit for no timeout (the default).
        overfit_check: When True, runs the voluntary ``overfit_check`` diagnostic (twenty optimizer
            steps at the training tile edge, on the CPU, inside this synchronous call) on the same
            batch the contract proved, before the subprocess spawns, and records the result on the
            run's ``model_contract`` under ``overfit_check``. Never gates. Default False.

    No record, no run: everything through the experiment stamp below is one boundary. Before it:
    preflight, normalization, the model contract, and the dataset identity read (a
    ``SchemaVersionRefused`` reader-ceiling mismatch refuses the launch by name; an absent,
    malformed or otherwise-unreadable identity trains untracked, the same as an unregistered
    dataset). After it: the registry entry, the launch config, the subprocess and TensorBoard. A
    spawn failure after the stamp leaves a ``running`` record with no process, which reads
    ``interrupted`` once its heartbeat stales and forks on relaunch.
    """
    # The caller's config is stored twice, as the launch config and as the experiment's
    # snapshot, so what it holds is checked before either write.
    check_json_value(config, path="config")
    # smoke=True: build the model and run the correctness contract before spawning the training
    # subprocess, so a broken builder returns here instead of wasting a full audited run.
    validation = preflight_config(config, smoke=True, overfit=overfit_check)
    if not validation["valid"]:
        return {"error": "Invalid config", "issues": validation["issues"]}

    # A shallow copy: what follows records onto it, never onto the caller's own dict, so a
    # launch never hands back an argument it silently mutated.
    config = dict(config)

    # The top-level key, never the smoke sub-report: overfit_check runs beside the contract's
    # build, on the same batch; preflight_config already rendered it for storage.
    rendered_overfit_report = validation.get("overfit_check")

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

    from tcip_mcp.experiments import experiments_dir, mint_experiment_id
    from tcip_mcp.pipelines.training.run_registry import create_run, draw_seed_if_unset
    from tcip_mcp.project_paths import resolve_output_path

    # Training artifacts (weights, tensorboard, metrics) live with the project the run belongs
    # to, same as its experiment record; only an absolute output_dir points anywhere else.
    output_base = str(resolve_output_path(output_dir) if output_dir else experiments_dir())

    data_cfg = config.get("data", {})

    # Resolved once, before the record is written, so every status write this launch makes
    # (fresh creation, pristine reuse or a fresh-id conflict) stamps the same declaration.
    launched_by = _resolve_launched_by()

    from tcip_store import SchemaVersionRefused

    from tcip_mcp.pipelines.data.split_construction import dataset_identity

    # Read before any record exists: a version-refused identity refuses the launch by name.
    try:
        ds_id, ds_fp = dataset_identity(data_cfg)
    except SchemaVersionRefused as exc:
        return {"error": f"launch_training: dataset identity is unreadable at this reader's "
                         f"ceiling, refusing to train against it untracked: {exc}"}

    draw_seed_if_unset(config)

    requested = config.get("experiment_id") or mint_experiment_id()

    # The id becomes the run's own artifact directory name: _trial_name applies the identical
    # directory-name rule to an HTTP path segment.
    try:
        _trial_name(requested)
    except BadKey:
        return {"error": f"launch_training: experiment_id {requested!r} is not a legal directory "
                         "name (a path separator, a drive, an empty name or '.'/'..'), and the id "
                         "becomes the run's own artifact directory."}

    try:
        experiment_id, run_output_dir = _ensure_experiment(
            requested, config, data_cfg.get("images_dir"), resume_from,
            output_base=output_base, launched_by=launched_by, dataset_id=ds_id,
            dataset_fingerprint=ds_fp,
        )
    except Exception as exc:
        return {"error": f"launch_training: could not record this run's experiment: {exc}"}

    # Thread the resolved id into the live config so the child's checkpoints carry it.
    config["experiment_id"] = experiment_id

    run = create_run(config, run_output_dir, id=experiment_id)

    # The child reads its own bootstrap config from here.
    store.replace(launch_config_key(run.output_dir), config)

    # Captured once, beside the child's environment snapshot: the watchdog below writes about
    # this run under the root it launched under, even if this process later adopts another.
    from tcip_mcp.project_paths import platform_state_root

    launch_root = platform_state_root()
    child_env = _child_env_for_launch(config)

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "tcip_mcp.pipelines.training.subprocess_worker",
            "--experiment-id", experiment_id,
            "--output-dir", run.output_dir,
            "--resume-from", resume_from,
        ],
        env=child_env,
    )
    run.pid = proc.pid

    if max_wall_clock_seconds is not None:
        _watch_wall_clock(proc, run, experiment_id, max_wall_clock_seconds, root=launch_root)

    # Keyed by its own log directory (the manager's default), never by the record id.
    tb_info = {}
    try:
        from tcip_mcp.pipelines.training.tensorboard_manager import launch_tensorboard
        tb_dir = str(Path(run.output_dir) / "tensorboard")
        tb_info = launch_tensorboard(tb_dir)
    except Exception:
        pass  # TensorBoard launch is best-effort

    return {
        "experiment_id": experiment_id,
        "status": "launched",
        "output_dir": run.output_dir,
        "tensorboard": tb_info,
        "pid": proc.pid,
        "overfit_check": rendered_overfit_report,
    }


def _child_env_for_launch(config: dict) -> dict[str, str]:
    """Subprocess env for a launch: round-robin GPU pinning (``CUDA_VISIBLE_DEVICES``) when the
    config names no device, untouched when it does; this process's import search path propagated
    via ``PYTHONPATH``.
    """
    import os

    from tcip_mcp.pipelines.model_build import child_pythonpath

    env = dict(os.environ)
    env["PYTHONPATH"] = child_pythonpath()

    if config.get("device"):
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
    """Daemon watcher: hard-terminates ``proc`` if it outlives ``timeout_seconds`` and records the
    reason through the status channel every other terminal state uses. No cooperative grace period.

    ``root`` is the platform root this run launched under, captured once at launch; the write goes
    there.
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
def monitor_training(experiment_id: str | None = None, sweep_id: str | None = None) -> dict:
    """Check the status of a training run, or of a hyperparameter sweep.

    Exactly one of ``experiment_id`` and ``sweep_id`` names what to check; both or neither refuses
    by name. The two return different shapes. A read: it changes nothing and leaves no audit line.

    ``experiment_id``: reads the run's own status/metrics from disk whenever its training body runs
        in a subprocess, or when this process never held the run in memory at all. Returns
        ``{"experiment_id", "status", "epoch", "best_metric", "output_dir", "error",
        "tensorboard_url"}``, or ``{"error": "Run not found: ..."}`` for an id no record claims,
        malformed ids folded to the same answer. An HPO trial is monitored through its sweep
        (``sweep_id=``) and canceled with it (``cancel_hyperparameter_search``).

    ``sweep_id``: reads the sweep's own manifest and trial directories from disk under this
        process's own pinned platform root, through :func:`read_sweep_from_disk`, then layers the
        study result's own fields onto a completed sweep through :func:`enrich_with_study_result`.
        A sweep that has not yet written a manifest reads as not found. Returns
        ``read_sweep_from_disk``'s own shape, enriched (``{"sweep_id", "status", "error", "result",
        "manifest", "relaunched_from", "has_manifest", "trials"}``), or ``{"error": ...}`` when no
        manifest exists or ``sweep_id`` would address a record outside the HPO store.

    Args:
        experiment_id: The run's record id (``tcip_mcp.experiments``), from launch_training.
            Exactly one of ``experiment_id``/``sweep_id`` is required.
        sweep_id: Hyperparameter sweep identifier. Exactly one of ``experiment_id``/``sweep_id``.
    """
    if sweep_id is not None:
        if experiment_id is not None:
            return {"error": "exactly one of experiment_id or sweep_id is required, got "
                              f"experiment_id={experiment_id!r} sweep_id={sweep_id!r}"}
        try:
            disk_sweep = read_sweep_from_disk(sweep_id)
        except BadKey:
            return {"error": f"invalid sweep_id: {sweep_id}"}
        if disk_sweep is None:
            return {"error": f"sweep not found: {sweep_id}"}
        return enrich_with_study_result(disk_sweep, sweep_id)
    if experiment_id is None:
        return {"error": "exactly one of experiment_id or sweep_id is required, got "
                          "experiment_id=None sweep_id=None"}

    from tcip_mcp.pipelines.training.run_registry import get_run
    run = get_run(experiment_id)

    result: dict[str, Any] | None = None
    if run is None or run.pid is not None:
        from tcip_mcp.experiments import reconstruct_run_status
        disk = reconstruct_run_status(experiment_id, stale_seconds=TCIP_HEARTBEAT_STALE_SECONDS)
        if disk is not None:
            result = {
                "experiment_id": experiment_id,
                "status": disk["status"],
                "epoch": disk["current_epoch"],
                "best_metric": disk["best_metric"],
                "output_dir": disk["output_dir"],
                "error": disk.get("error"),
            }

    if result is None:
        if run is None:
            return {"error": f"Run not found: {experiment_id}"}
        result = {
            "experiment_id": run.id,
            "status": run.status,
            "epoch": run.current_epoch,
            "best_metric": run.best_metric,
            "output_dir": run.output_dir,
            "error": run.error or None,
        }

    # A run's own TensorBoard is keyed by its log directory, never by experiment_id.
    tb_url = None
    try:
        from tcip_mcp.pipelines.training.tensorboard_manager import _TB_PROCESSES
        output_dir = result.get("output_dir")
        tb_key = str((Path(output_dir) / "tensorboard").resolve()) if output_dir else None
        entry = _TB_PROCESSES.get(tb_key) if tb_key is not None else None
        if entry is not None and entry.proc.poll() is None:
            tb_url = f"http://localhost:{entry.port}"
    except Exception:
        pass
    result["tensorboard_url"] = tb_url
    return result


def _launched_training_runs(*, read_progress: bool) -> list[dict[str, Any]]:
    """Every launched training run this store holds a record for, reconstructed from disk.

    A record is a launched run when its config carries ``model_source`` and
    it is no longer pristine (:func:`~tcip_mcp.experiments.is_pristine`): a state other than
    ``"created"``, or the ``metrics_logged`` marker. Rows come back sorted by experiment id
    (``experiment_ids_with_status``'s own order), each carrying ``external: True``, a
    process-locality fact only (this record was reconstructed from disk); who launched it is the
    record's own ``launched_by`` (see :func:`~tcip_mcp.experiments.reconstruct_from_status`).
    ``read_progress`` governs whether ``current_epoch`` costs a metrics-log read per record. Cost:
    one status read and one config read per experiment record on disk, plus, when ``read_progress``
    is true, one metrics-log read per launched record.
    """
    from tcip_store import DecodeError

    from tcip_mcp.experiments import (
        config_key, experiment_ids_with_status, is_pristine, metrics_logged_of,
        reconstruct_from_status, recorded_state, status_key,
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
        if is_pristine(recorded_state(status), metrics_logged_of(status)):
            continue
        row = reconstruct_from_status(experiment_id, status, stale_seconds=TCIP_HEARTBEAT_STALE_SECONDS,
                                      read_progress=read_progress)
        row["external"] = True
        rows.append(row)
    return rows


def _all_training_runs(*, read_progress: bool) -> list[dict[str, Any]]:
    """This process's in-memory registry merged with every launched run's own disk record.

    A live in-memory entry (HPO trials excluded) wins by its own id over its own disk row: a
    ``pid``-bearing one takes the disk overlay for
    ``status``/``heartbeat``/``current_epoch``/``error`` and ``best_metric``/``best_metric_name``;
    a ``pid``-less one (every synchronous run) is reported from its own in-memory record,
    untouched, with no ``heartbeat``. Both carry ``external: False`` and ``experiment_id``, the
    row's own id. ``launched_by`` is the record's own: a ``pid``-bearing row takes the overlay's, a
    ``pid``-less row reads its resolved experiment's status record directly; a run whose stamp
    failed reads ``None``, and an id that can name no record (``BadKey``) folds to ``None``.

    Rows: this process's own, in registry order, then the disk-only rows, sorted by experiment id.
    """
    from tcip_mcp.experiments import read_member, status_key
    from tcip_mcp.pipelines.training.run_registry import list_runs

    live = list_runs()
    disk = _launched_training_runs(read_progress=read_progress)
    disk_by_id = {r["experiment_id"]: r for r in disk}

    merged: list[dict[str, Any]] = []
    for r in live:
        row = dict(r)
        experiment_id = row.pop("id")
        row["experiment_id"] = experiment_id
        row["external"] = False
        overlay = disk_by_id.get(experiment_id) if row.get("pid") is not None else None
        if overlay is not None:
            row["status"] = overlay["status"]
            row["heartbeat"] = overlay.get("heartbeat")
            if overlay["current_epoch"] is not None:
                row["current_epoch"] = overlay["current_epoch"]
            if overlay.get("error"):
                row["error"] = overlay["error"]
            if overlay.get("best_metric_name") is not None:
                row["best_metric"] = overlay["best_metric"]
                row["best_metric_name"] = overlay["best_metric_name"]
            row["launched_by"] = overlay["launched_by"]
        else:
            try:
                key = status_key(experiment_id)
            except BadKey:
                row["launched_by"] = None
            else:
                row["launched_by"] = read_member(key, {}).get("launched_by")
        merged.append(row)

    live_ids = {r["id"] for r in live}
    disk_only = [r for r in disk if r["experiment_id"] not in live_ids]
    return merged + disk_only


def list_launchable_configs() -> list[dict]:
    """Every experiment in this project with a model source, as a row the config picker can start a
    run from: the id, the builder and task, the data it names, the subject, its derived state and
    its parent when it has one.

    Cost: ``list_experiments()``'s own one status read plus one config read per experiment record,
        and a further read of that same config, plus one status read and one lineage read per
        experiment that carries a model source. State is ``derived_state`` once the record is no
        longer pristine (``is_pristine``), so a never-launched config reads ``"created"``.
    """
    from tcip_mcp.experiments import (
        config_key, derived_state, is_pristine, lineage_key, list_experiments,
        metrics_logged_of, read_member, status_key,
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
            "task": run_task(config),
            "images_dir": (data_cfg or {}).get("images_dir"),
            "subject": (data_cfg or {}).get("subject"),
            "created": exp["created"],
            "state": exp["state"] if is_pristine(exp["state"], metrics_logged_of(status))
                     else derived_state(status, TCIP_HEARTBEAT_STALE_SECONDS),
            "parent_experiment": (lineage or {}).get("parent_experiment"),
        })
    return rows


def list_split_choices(experiment_id: str) -> dict:
    """Every choice this config's own "Data" control offers a relaunch of ``experiment_id``: its
    stored data section as recorded, and every selection directory this project's own bound runs or
    the dataset's own ``splits`` directory hold, each compatibility-checked the way a launch would
    check it.

    Every check is :func:`selection_compatibility` over a selection this reader read itself,
    through :func:`~tcip_mcp.pipelines.data.selection.read_selection_checked`. A candidate
    selection is checked against the config :func:`candidate_config_with_selection` builds
    (``data.split`` replaced wholesale); "As recorded" is checked against the stored config
    unchanged, plus the directory-presence issues :func:`preflight_config` would raise.

    The listing: the selection directories other enumerable experiment configs in this project
    bound to (the picked config's own excluded, since it is "As recorded"), plus, when
    ``dataset_root_of(data.images_dir)`` resolves, that root's ``splits`` directory (offered only
    when something is recorded there directly) and every directory one level under it holding a
    selection. The own-binding exclusion and the candidate dedupe compare each directory by
    ``tcip_store.canonical_path``.

    Returns ``{"error": ...}`` for an unknown ``experiment_id``. Otherwise: ``{"as_recorded":
    {"case": "bound"|"drawn", "line": str, "compatible": bool, "reason": str | None}, "selections":
    [{"selection_dir": str, "enabled": bool, "reason": str | None, "seed": int | None, "group_by":
    str | None, "train": int, "val": int, "calibration": int, "replaced_split_keys": list[str]},
    ...]}``. The counts are the selection's whole sides. ``replaced_split_keys`` names every
    recorded ``data.split`` key other than ``selection_dir`` choosing any offered partition drops,
    the same set for every candidate. A bound config anchors the search for sibling selections at
    its own selection's first sample source; an unbound one at ``data.images_dir``. With neither
    readable, ``selections`` is empty.
    """
    from tcip_mcp.dataset_layout import dataset_root_of
    from tcip_mcp.experiments import config_key, experiment_exists, experiment_ids_with_status, read_member
    from tcip_mcp.pipelines.data.selection import read_selection_checked

    if not experiment_exists(experiment_id):
        return {"error": f"Experiment not found: {experiment_id}"}
    config = read_member(config_key(experiment_id), {})
    config = config if isinstance(config, dict) else {}
    data_cfg = config.get("data")
    data_cfg = data_cfg if isinstance(data_cfg, dict) else {}
    split_cfg = data_cfg.get("split")
    split_cfg = split_cfg if isinstance(split_cfg, dict) else {}
    own_selection_dir = split_cfg.get("selection_dir")
    replaced_split_keys = sorted(
        k for k, v in split_cfg.items() if k != "selection_dir" and v is not None
    )

    own_selection = None
    if own_selection_dir:
        as_recorded = {"case": "bound", "line": "on the partition it bound",
                       "compatible": True, "reason": None}
        own_selection, own_error = read_selection_checked(own_selection_dir)
        if own_selection is None:
            as_recorded["compatible"] = False
            as_recorded["reason"] = own_error or (
                f"no selection recorded under {own_selection_dir}; run draw_splits first."
            )
        else:
            own_issues = selection_compatibility(config, own_selection, own_selection_dir)
            if own_issues:
                as_recorded["compatible"] = False
                as_recorded["reason"] = "; ".join(own_issues)
    else:
        seed = split_seed(split_cfg)
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

    # A bound config names no locations, so its own selection's first sample source anchors the
    # sibling search the way an unbound config's images_dir does.
    if own_selection is not None and own_selection.samples:
        root_anchor = str(Path(own_selection.samples[0].source).parent)
    elif not own_selection_dir and not dir_issues:
        root_anchor = str(data_cfg["images_dir"])
    else:
        return {"as_recorded": as_recorded, "selections": []}

    own_norm = canonical_path(own_selection_dir) if own_selection_dir else None
    candidate_dirs: list[str] = []
    seen: set[str] = set()
    for other_id in experiment_ids_with_status():
        if other_id == experiment_id:
            # Its own selection_dir is already own_selection_dir, read once above; re-reading its
            # config here to derive the identical fact a second time is work this listing skips.
            continue
        other_config = read_member(config_key(other_id), {})
        if not isinstance(other_config, dict):
            continue
        other_data = other_config.get("data")
        other_data = other_data if isinstance(other_data, dict) else {}
        other_split = other_data.get("split")
        other_split = other_split if isinstance(other_split, dict) else {}
        candidate = other_split.get("selection_dir")
        if not candidate:
            continue
        candidate_norm = canonical_path(candidate)
        if candidate_norm == own_norm or candidate_norm in seen:
            continue
        seen.add(candidate_norm)
        candidate_dirs.append(candidate)
    dataset_root = dataset_root_of(root_anchor)
    if dataset_root is not None:
        default_dir = str(dataset_root / "splits")
        default_norm = canonical_path(default_dir)
        if default_norm != own_norm and default_norm not in seen:
            seen.add(default_norm)
            candidate_dirs.append(default_dir)
        # One level down: where freeze_selection writes a frozen run's own partition.
        splits_dir = Path(default_dir)
        if splits_dir.is_dir():
            for sub in sorted(p for p in splits_dir.iterdir() if p.is_dir()):
                sub_norm = canonical_path(str(sub))
                if sub_norm == own_norm or sub_norm in seen:
                    continue
                seen.add(sub_norm)
                candidate_dirs.append(str(sub))

    selections: list[dict] = []
    for candidate_dir in candidate_dirs:
        selection, error_text = read_selection_checked(candidate_dir)
        if selection is None:
            if error_text is None:
                continue  # nothing recorded there; not a real candidate
            selections.append({
                "selection_dir": candidate_dir, "enabled": False, "reason": error_text,
                "seed": None, "group_by": None, "train": 0, "val": 0, "calibration": 0,
                "replaced_split_keys": replaced_split_keys,
            })
            continue
        candidate_config = candidate_config_with_selection(config, candidate_dir)
        issues = selection_compatibility(candidate_config, selection, candidate_dir)
        counts = selection.counts()
        entry: dict = {
            "selection_dir": candidate_dir, "seed": selection.seed,
            "group_by": selection.group_by, "train": counts["train"],
            "val": counts["val"], "calibration": counts["calibration"],
            "replaced_split_keys": replaced_split_keys,
        }
        if issues:
            entry["enabled"] = False
            entry["reason"] = "; ".join(issues)
        else:
            entry["enabled"] = True
            entry["reason"] = None
        selections.append(entry)

    return {"as_recorded": as_recorded, "selections": selections}


@mcp.tool()
@audited
def cancel_training(experiment_id: str) -> dict:
    """Request graceful cancellation of a running training run.

    The trainer stops at the next batch/epoch boundary, still saves ``model_final.pt``
    (so partial progress is recoverable), and sets the run + its experiment to
    'canceled'. Status updates asynchronously, so the returned status may still read
    'running' immediately after the request. A run whose divergence verdict lands first (two
    consecutive full training passes with no finite batch loss, checked ahead of cancellation
    at the same boundary) ends 'failed' instead, with no ``model_final.pt``.

    Args:
        experiment_id: The run's record id (one run's immutable record, ``tcip_mcp.experiments``),
            from launch_training.
    """
    from tcip_mcp.pipelines.training.run_registry import cancel_run, get_run
    if not cancel_run(experiment_id):
        return {"error": f"Run not found: {experiment_id}"}
    run = get_run(experiment_id)
    if run is not None:
        status = run.status
    else:
        # Canceled via the disk fallback: no in-memory status, so reflect the disk record
        # cancel_run itself resolved to write the sentinel, if it's still discoverable.
        from tcip_mcp.experiments import reconstruct_run_status
        disk = reconstruct_run_status(experiment_id, stale_seconds=TCIP_HEARTBEAT_STALE_SECONDS)
        status = disk["status"] if disk is not None else "running"
    return {"experiment_id": experiment_id, "status": status, "cancel_requested": True}


def inspect_compute_resources() -> dict:
    """Report the host's current compute headroom, a fact to reason with before launching another
    concurrent training/HPO run, not an enforced cap.

    Returns:
        ``cpu``: ``{logical_count, percent_used}``, ``percent_used`` is ``None`` without ``psutil``
            installed.
        ``memory``: ``{total_bytes, available_bytes}``, both ``None`` without ``psutil``.
        ``gpus``: ``[{index, free_bytes, total_bytes}, ...]``, always populated when CUDA is
            available (``torch.cuda.mem_get_info``, no extra dependency); ``[]`` otherwise.
        ``active_training_runs``: count of every run whose derived state is ``"running"``, a
            heartbeat fresher than ``TCIP_HEARTBEAT_STALE_SECONDS`` (600s by default), through
            :func:`_all_training_runs` with progress reads off.
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


def hpo_root(output_dir: str = "", *, root: Path | str | None = None) -> Path:
    """Where HPO sweeps live: ``output_dir`` when the caller named one, else ``.tcip/hpo`` under
    ``root`` (default: the platform state root). A relative ``output_dir`` resolves against the
    platform state root, never the server process's cwd.
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

_CANCEL_BEFORE_START_REASON = "canceled before the sweep's first trial started"
_CANCEL_DURING_RUN_REASON = "the sweep was canceled by request before it could finish"

_TRIAL_DIR_PREFIX = "trial_"

def sweep_heartbeat_seconds() -> float:
    """How often ``run_hyperparameter_search``'s driver thread restamps the sweep manifest's
    ``heartbeat`` while ``tune_search`` runs: :data:`TCIP_HEARTBEAT_STALE_SECONDS` read fresh on
    every call, divided by ten.
    """
    return TCIP_HEARTBEAT_STALE_SECONDS / 10

_LAUNCHING_SWEEPS: dict[str, Path] = {}
_LAUNCHING_SWEEPS_LOCK = threading.Lock()
"""Every sweep a caller has minted a ``study_name`` for and is about to hand to ``run_hyperparameter_search``,
before that call's own manifest exists: the pre-manifest window in which a cancel request
would otherwise find nothing on disk and nothing in ``run_registry._RUNS`` to act on."""


def mark_sweep_launching(study_name: str, output_dir: str = "", *, root: Path | str | None = None) -> None:
    """Record that ``study_name`` is about to become a sweep at this resolved root, closing the
    window between a caller minting the id and ``run_hyperparameter_search`` writing its first
    manifest.

    ``run_hyperparameter_search`` discards the mark in a ``finally`` around everything from its own
    entry through its first manifest write; a caller that marks a study and never calls
    ``run_hyperparameter_search`` for it discards it itself (:func:`discard_sweep_launching`).
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
    """The sweep's derived liveness, the ``status`` every Tuning listing row reports.

    ``driver_live`` is true only where a process can vouch for the driver directly (a live worker
    thread for a sweep it launched), which beats the manifest's heartbeat outright. Every other
    case reads through :func:`tcip_mcp.experiments.derived_state` over ``{"state":
    manifest["status"], "heartbeat": manifest["heartbeat"]}``.
    """
    from tcip_mcp.experiments import _RECORDED_AS_DONE, derived_state

    status = manifest["status"]
    if driver_live and status not in _RECORDED_AS_DONE:
        return "running"
    return derived_state({"state": status, "heartbeat": manifest["heartbeat"]}, stale_seconds)


def _running_trial_dirs(sweep_root: Path) -> list[Path]:
    """Every ``trial_<id>`` directory under ``sweep_root`` that has not yet written its
    resolved-config record (``_run_hpo_trial``'s ``finally`` block writes it once, at the end of
    the trial).
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
    """``study_name`` once it is known to name one sweep and not a path through the store: a
    separator, a drive letter or a parent reference is refused.
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
    """The manifest a sweep is listed and read back from, keyed off the HPO root.

    Two writers share this record. ``run_hyperparameter_search`` replaces the whole document at
    each state change, re-deriving ``cancel_requested`` from the sweep's own stop file on every
    write. ``cancel_hyperparameter_search`` read-modify-writes only ``cancel_requested`` through
    the store's compare-and-set (``read_versioned`` plus ``replace(..., expect=version)``) and
    never over a manifest already in a terminal status. ``concurrency="last_writer_wins"``.
    """
    return Key(SWEEP_MANIFEST_STORE, str(hpo_root(output_dir, root=root).resolve()),
               (_sweep_name(study_name), "manifest"))


STUDY_RESULT_FIELDS = ("all_trials", "search_alg", "scheduler", "warm_start", "baseline_params")
"""The study result's own fields, absent from the manifest's completion projection."""


def study_result_key(
    study_name: str, output_dir: str = "", *, root: Path | str | None = None
) -> Key:
    """A finished sweep's result document, beside the sweep's own directory, under ``root`` as
    :func:`sweep_manifest_key` resolves it. ``last_writer_wins``: written once, when the sweep
    ends.
    """
    return Key(STUDY_RESULT_STORE, str(hpo_root(output_dir, root=root).resolve()),
               (_sweep_name(study_name),))


def _trial_name(trial_dir_name: str) -> str:
    """``trial_dir_name`` once it is known to name one trial and not a path through the sweep: a
    separator, a drive letter or a parent reference is refused.
    """
    if PureWindowsPath(trial_dir_name).name != trial_dir_name or trial_dir_name == "..":
        raise BadKey(
            f"trial name {trial_dir_name!r} is not a single name: a name carrying a path "
            "separator, a drive or a parent reference would address a record outside the sweep"
        )
    return trial_dir_name


def trial_config_key(sweep_root: Path | str, trial_dir_name: str) -> Key:
    """The point one trial actually trained at: its merged config plus the sampled params, scoped
    to the sweep. ``last_writer_wins``: one trial process writes its own document once, when the
    trial finishes.
    """
    return Key(TRIAL_CONFIG_STORE, str(Path(sweep_root).resolve()),
               (_trial_name(trial_dir_name), "resolved_config"))


def trial_metrics_key(sweep_root: Path | str, trial_dir_name: str) -> Key:
    """One trial's epoch-by-epoch metrics, one entry per row, append only, scoped to the sweep like
    :func:`trial_config_key`.
    """
    return Key(TRIAL_METRICS_STORE, str(Path(sweep_root).resolve()),
               (_trial_name(trial_dir_name), "metrics"))


def trial_metrics_key_for_dir(trial_dir: Path | str) -> Key:
    """The metrics log of the trial that writes into ``trial_dir``."""
    path = Path(trial_dir).resolve()
    return trial_metrics_key(path.parent, path.name)


def log_holds_anything(page: Any) -> bool:
    """Whether a metrics log holds anything at all: rows, a torn tail, undecodable bytes, or
    entries at a schema_version this reader does not accept.
    """
    return bool(page.records or page.torn_tail or page.corrupt or page.version_refused)


def enrich_with_study_result(
    response: dict[str, Any], sweep_id: str, *, root: Path | str | None = None
) -> dict[str, Any]:
    """Layer the study result's own fields onto ``response["result"]`` for a completed sweep, read
    through the store and never fabricated: a sweep whose study result is absent (or already
    carries these fields) is served as it already was.
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
    """One sweep's manifest-derived summary plus every trial directory it has produced, read from
    the sweep's own store records alone.

    Returns ``None`` when no manifest exists under ``root`` (the current platform root when
    ``root`` is ``None``). Otherwise: ``{"sweep_id", "status", "error", "result", "manifest",
    "relaunched_from", "has_manifest": True, "trials"}``. ``status`` is the derived liveness
    (:func:`sweep_state`, ``driver_live=False``). ``result`` is the manifest's own, exactly as
    written (:func:`enrich_with_study_result` layers the study result). ``trials`` is one entry per
    ``trial_<id>`` directory under the sweep's own root: its resolved params and whether it has
    logged any metrics yet (:func:`log_holds_anything`).
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
            })

    return {
        "sweep_id": manifest.get("study_name", sweep_id),
        "status": status,
        "error": manifest.get("error"),
        "result": result,
        "manifest": manifest,
        "relaunched_from": manifest["relaunched_from"],
        "has_manifest": True,
        "trials": trials,
    }


def _run_hpo_trial(config: dict, report, base_config: dict, trial_dir: str) -> None:
    """Train one HPO trial and ``report`` its resolved selection metric, in whatever direction that
    metric's own declaration says is better (``evaluation.HIGHER_IS_BETTER_BY_METRIC``, via
    :func:`~tcip_mcp.pipelines.training.generic_trainer.config_selection_metric`).

    ``report(value)`` feeds the Ray Tune searcher/scheduler; it is called each epoch and once at
    the end with the best value this trial reached. A trial that never reports a real value, before
    training starts or on any failure, and a trial whose run ended ``"failed"`` or ``"canceled"``,
    report the losing side of its own direction as that final value. Trials train under the final
    run's regime, same augmentation, imbalance handling, and dispatch: a ``training_source`` in
    ``base_config`` runs under that loop here too.
    """
    trial_config = _apply_hpo_params(base_config, config)

    from tcip_mcp.pipelines.training.envelope import TrainContext, dispatch_train_body
    from tcip_mcp.pipelines.training.evaluation import HIGHER_IS_BETTER_BY_METRIC
    from tcip_mcp.pipelines.training.generic_trainer import (
        _improves, config_selection_metric, run_loaders, run_transforms,
        stamp_effective_data_geometry,
    )
    from tcip_mcp.pipelines.training.run_registry import create_run, draw_seed_if_unset
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    # setdefault, not get: creates "data" if base_config omitted it, and the geometry stamp
    # below mutates the tree the resolved-config snapshot is spread from below.
    data_cfg = trial_config.setdefault("data", {})
    task = run_task(trial_config)
    try:
        higher_is_better = HIGHER_IS_BETTER_BY_METRIC[config_selection_metric(trial_config)]
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

    draw_seed_if_unset(trial_config)
    # An id no tool takes, unique across concurrent sweeps where a directory basename is not.
    trial_id = str(Path(trial_dir).resolve())
    run = create_run(trial_config, trial_dir, id=trial_id, origin="hpo_trial")  # off the Training tab

    # The best value this trial has actually reported, in the resolved direction; call_report is
    # what every reporting path below goes through, so this is the one place that tracks it.
    best = {"value": losing_side}

    def call_report(value: float) -> None:
        value = float(value)
        if _improves(value, best["value"], higher_is_better=higher_is_better):
            best["value"] = value
        report(value)

    try:
        # Auto-val gives the val_loader that the composite objective / the scheduler need.
        train_ds, val_ds, _partition = auto_train_val(
            task, data_cfg, run_transforms(trial_config))
        # Stamped before training so a pruned/failed trial's resolved-config snapshot still
        # records the geometry the trial actually trained on.
        stamp_effective_data_geometry(data_cfg, train_ds)
        # run.config's seed is draw_seed_if_unset-resolved, the value the trial actually uses.
        train_loader, val_loader = run_loaders(
            trial_config, task, train_ds, val_ds, run.config.get("seed"))

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
        # A diverged or canceled run reports the losing side, never what it reported before ending.
        if run.status in ("failed", "canceled"):
            report(losing_side)
        else:
            report(best["value"])  # the trial's best reported value, or the losing side if none
    except Exception as e:
        logger.warning("HPO trial failed: %s", e)
        report(losing_side)
    finally:
        try:
            # trial_params is the sampled point itself, the only record of which axes this
            # sweep actually varied (the config as sampled cannot say that).
            trial_path = Path(trial_dir)
            store.replace(trial_config_key(trial_path.parent, trial_path.name),
                          {**trial_config, "trial_params": dict(config)})
        except (OSError, StoreError):
            logger.warning("could not persist the resolved config for %s", trial_dir, exc_info=True)


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
    search_seed: int,
    trial_budget: int | None = None,
    relaunched_from: str | None = None,
) -> dict:
    """Run hyperparameter optimization on Ray Tune, training each trial for real.

    The search algorithm and trial scheduler are the caller's choice (call the ``hpo`` module's
    ``available_search_algs`` / ``available_schedulers`` for the live list):
      - ``search_alg``: ``random``/``grid`` (native), or a backend, ``optuna``, ``bayesopt``,
        ``hyperopt``, each constructed with ``search_seed``.
      - ``scheduler``: ``asha`` (async HyperBand), ``hyperband``, ``pbt``, ``median``, or ``none``
        to run every trial to completion.

    Trials optimize ``base_config``'s own resolved selection metric, in whatever direction that
    metric's declaration says is better (``evaluation.HIGHER_IS_BETTER_BY_METRIC``, resolved once
    for the whole sweep); each trains under the base config's regime.

    Everything one sweep writes lands under ``<output_dir or .tcip/hpo>/<study_name>/``: a
    ``manifest.json`` stamped ``running`` before the first trial starts and updated when the sweep
    ends, one ``trial_<id>/`` directory per trial, and Ray's own experiment store (also the
    TensorBoard logdir). The full result is written alongside as ``<study_name>.json``. The
    manifest also carries every argument this call resolved (``base_config`` and the resolved
    ``param_space`` included). The manifest's ``heartbeat`` is restamped every
    :func:`sweep_heartbeat_seconds` from a daemon thread for as long as the search runs, stopped
    and joined before any terminal write; a restamp a store or OS error interrupts costs one beat.
    This call discards a :func:`mark_sweep_launching` mark for ``study_name`` once its first
    manifest is written.

    Refuses (``{"error": ..., "issues": [...]}``, nothing minted): an unimportable builder or
        training source, or a config with no ``data`` section, at every point the search space
        could resolve a trial's config to; a ``param_space`` axis whose sampled points would
        resolve to a different selection metric or ranking direction than ``base_config``'s own
        (including an axis such as ``model_source.task`` that changes the metric's task-derived
        default); a ``param_space`` axis naming ``data.split.seed`` while ``split_draws`` draws at
        most one partition (:func:`caller_split_seed_refusal`); ``split_draws`` above 1 on an
        unbound, built-in detection config with tiling on that admits exactly one trainable source
        (:func:`_split_draws_refusal`); a ``split_draws`` that is not an integer, or below 1
        (:func:`_split_draws_argument_refusal`, checked first); and, on a launch above one draw or
        any call naming a ``trial_budget``, the bound refusals of :func:`_trial_budget_refusal`. A
        cancel (``cancel_hyperparameter_search``) requested before or during the run ends the sweep
        ``{"status": "canceled", ...}``, the manifest recording the same.

    Args:
        base_config: Base training config each trial modifies.
        param_space: Param-space dict (see ``hpo.get_default_space``); default when omitted. Every
            axis is checked against ``base_config``'s own resolved selection metric and direction.
            A ``data.split.seed`` axis is refused whenever ``split_draws`` draws at most one
            partition; above 1 it belongs to ``split_draws``/``split_draw_seeds``.
        n_trials: Number of trials; a whole number of at least one on a call that reads a
            ``trial_budget`` bound.
        output_dir: Base output directory for trial results (defaults under ``.tcip/hpo``).
        search_alg: Search algorithm, see the list above.
        scheduler: Trial scheduler, see the list above.
        grace_period: Minimum epochs before a halving scheduler (``asha``/``hyperband``) can stop a
            trial early.
        reduction_factor: Halving factor for ``asha``/``hyperband`` (fraction of trials kept at
            each rung).
        warm_start: Seed the search with ``baseline_params`` as a known-good starting point.
        baseline_params: Hyperparameter values to seed the search with when ``warm_start=True``.
        max_concurrent: Trials to run at once (default 1, safe for single-GPU training).
        resources_per_trial: Ray resource request per trial, omit to derive one from the host's
            real GPU count and ``max_concurrent`` (see ``hpo._default_trial_resources``); an
            explicit value always wins.
        study_name: The sweep's id, when the caller already minted one; omitted mints one here.
        auto_tensorboard: Launch a TensorBoard over the sweep root once it finishes.
        relaunched_from: The sweep this one replays, recorded on this sweep's manifest, ``None``
            when this sweep was not a relaunch. Refused when it names no sweep manifest under this
            resolved root. A stated ``trial_budget`` is checked on a relaunch exactly as on a
            launch.
        search_seed: The search algorithm's own seed, recorded on the manifest; required, and
            distinct from the split seed a trial's data draw uses.
        trial_budget: The most trials this sweep may launch, counted the way Ray will launch them
            (see :func:`~tcip_mcp.pipelines.training.hpo.planned_trial_count`). Required above one
            draw on a launch that is not a relaunch; checked whenever stated, including at one
            draw. Recorded on the sweep manifest beside ``split_draws``, ``None`` when the caller
            stated none.
        split_draws: Above 1, adds ``data.split.seed`` to the search space as a grid over
            ``split_draw_seeds`` (default: the base config's own ``data.split.seed``, else
            ``DEFAULT_SEED``, plus the draw index), paired with every sampled point through Ray's
            own ``BasicVariantGenerator(constant_grid_search=True)`` so each point trains once per
            seed. A ``base_config`` bound to a selection gains
            ``data.split.redraw_within_selection: true`` on its own copy (``data.split.seed``
            defaulting to ``DEFAULT_SEED``), so every trial redraws train and val inside the
            selection's own train-plus-val samples, calibration untouched; refused when those
            samples resolve to fewer than two foreground groups. Otherwise refused when
            ``data.auto_val`` is off, ``search_alg`` is not a native one
            (``random``/``grid``/``variant_generator``), ``scheduler`` is not ``none``,
            ``split_draw_seeds`` is given at a length other than ``split_draws`` or names the same
            seed twice, ``warm_start``'s ``baseline_params`` names ``data.split.seed``,
            ``param_space`` already sweeps ``data.split.seed`` itself, or ``param_space`` sweeps
            any other ``data.*`` axis. The result groups trials by point (params minus the seed)
            and chooses the best by mean over each point's draws; see
            ``result["best_value_spread"]``, and every point's own block at
            ``result["split_sensitivity"]`` beside ``result["n_points"]`` (planned points) and
            ``result["split_draws"]``, both mirrored onto the sweep manifest's own ``result``. 1 is
            the default; zero or below is refused naming the value.
        split_draw_seeds: The seeds ``split_draws`` pairs with every sampled point, one per draw;
            omit for the derived default (see ``split_draws``).
    """
    from tcip_mcp.pipelines.training.hpo import (
        get_default_space,
        split_draw_search_space,
        tune_search,
    )

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

        argument_refusal = _split_draws_argument_refusal(split_draws)
        if argument_refusal is not None:
            return {"error": argument_refusal, "issues": []}

        # Below the leg that makes split_draws an integer, so this comparison never meets another
        # type, and computed once so every leg that reads it agrees on whether a bound is read.
        reads_bound = trial_budget is not None or (split_draws > 1 and relaunched_from is None)

        # A bound base_config admitted to split_draws redraws inside its manifest from here on.
        base_config = _base_config_for_split_draws(base_config, split_draws)

        # Checked ahead of preflight, so its own reason is what a bound or auto_val refusal
        # reads as, not whatever preflight would have hit first.
        hpo_task = run_task(base_config)
        draws_refusal = _split_draws_refusal(
            base_config, param_space, hpo_task, search_alg, scheduler,
            split_draws, split_draw_seeds, warm_start, baseline_params)
        if draws_refusal is not None:
            return {"error": draws_refusal, "issues": []}

        seed_axis_refusal = caller_split_seed_refusal(param_space, split_draws)
        if seed_axis_refusal is not None:
            return {"error": f"{seed_axis_refusal.reason} {seed_axis_refusal.remedy}",
                    "issues": []}

        from tcip_mcp.pipelines.training.evaluation import HIGHER_IS_BETTER_BY_METRIC
        from tcip_mcp.pipelines.training.generic_trainer import config_selection_metric
        # Ray forbids setting metric/mode anywhere but the Tuner, so the direction is resolved
        # once here, from base_config; every point below is checked against it the same way.
        try:
            hpo_metric = config_selection_metric(base_config)
        except ValueError as exc:
            return {"error": str(exc), "issues": []}
        hpo_mode = "max" if HIGHER_IS_BETTER_BY_METRIC[hpo_metric] else "min"

        axis_conflict = _selection_metric_axis_conflict(
            base_config, param_space, hpo_metric, hpo_mode)
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

        search_param_space, resolved_draw_seeds = split_draw_search_space(
            param_space, base_config, split_draws, split_draw_seeds)

        budget_refusal = _trial_budget_refusal(
            reads_bound=reads_bound, split_draws=split_draws, n_trials=n_trials,
            search_alg=search_alg, warm_start=warm_start, trial_budget=trial_budget,
            relaunched_from=relaunched_from, search_param_space=search_param_space,
            baseline_params=baseline_params,
        )
        if budget_refusal is not None:
            return {"error": budget_refusal, "issues": []}

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
            "search_seed": search_seed,
            "trial_budget": trial_budget,
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
        # call reached the manifest write) records a canceled manifest rather than refusing.
        if cancel_path.exists():
            manifest.update(status="canceled", error=_CANCEL_BEFORE_START_REASON,
                            finished_at=datetime.now(timezone.utc).isoformat())
            _write_manifest()
            return {"status": "canceled", "study_name": study_name, "error": _CANCEL_BEFORE_START_REASON}

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
            seed=search_seed,
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
            manifest.update(status="canceled", error=_CANCEL_DURING_RUN_REASON,
                            finished_at=datetime.now(timezone.utc).isoformat())
            _write_manifest()
            return {"status": "canceled", "study_name": study_name, "error": _CANCEL_DURING_RUN_REASON}
        manifest.update(status="failed", error=str(exc),
                        finished_at=datetime.now(timezone.utc).isoformat())
        _write_manifest()
        raise

    heartbeat_stop.set()
    heartbeat_thread.join()

    if cancel_path.exists():
        manifest.update(status="canceled", error=_CANCEL_DURING_RUN_REASON,
                        finished_at=datetime.now(timezone.utc).isoformat())
        _write_manifest()
        return {"status": "canceled", "study_name": study_name, "error": _CANCEL_DURING_RUN_REASON}

    # Auto-launch TensorBoard on the sweep root: Ray's per-trial event files and each
    # trial's own tensorboard dir both sit under it.
    tb_info: dict = {}
    tb_logdir = result.get("tensorboard_logdir")
    if tb_logdir and auto_tensorboard:
        try:
            from tcip_mcp.pipelines.training.tensorboard_manager import launch_tensorboard
            tb_info = launch_tensorboard(tb_logdir, key=f"hpo_{study_name}")
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
            result["best_value_reason"] = (
                "no eligible point: every drawn point had an errored or never-answered draw"
            )
            result["best_value_spread"] = None

    # best_value_state carries only stored_number's own token vocabulary; best_value_reason is
    # the English sentence for why there is no best value at all.
    manifest_result = {k: result.get(k) for k in ("best_params", "best_value", "n_trials")}
    if "best_value_state" in result:
        manifest_result["best_value_state"] = result["best_value_state"]
    for key in ("best_value_spread", "split_sensitivity", "n_points", "split_draws", "best_value_reason"):
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

    Writes the sweep's own stop file (``SWEEP_CANCEL_SENTINEL``) at the sweep's root, which
    ``run_hyperparameter_search``, ``_run_hpo_trial`` and the sweep's own Tune ``Stopper`` poll.
    Also writes the run-level sentinel (``run_registry.CANCEL_SENTINEL``) into every trial
    directory that has not yet written its resolved config, so a trial mid-epoch sees the request
    at its next batch boundary.

    Refuses when the study names no sweep this process can find: no manifest under the resolved
    root, no live trial of this study registered in this process's own run registry, and no
    ``mark_sweep_launching`` entry for it at this same resolved root either. A marked study with no
    manifest yet answers ``"running"`` with ``cancel_requested`` set; a mark recorded under a
    different root does not count.

    The manifest's own ``cancel_requested`` is written through the store's compare-and-set, and
    never over a manifest already in a terminal status (see :func:`sweep_manifest_key`). The
    sentinel files are the authoritative signal; the manifest field mirrors them best-effort.

    Args:
        study_name: The sweep to cancel.
        output_dir: Where the sweep's own directory lives, as given to
            ``run_hyperparameter_search``; empty resolves the same ``.tcip/hpo`` default, under
            ``root``.
        root: The platform root this sweep launched under; omitted resolves under this process's
            own root.
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
    from tcip_mcp.experiments import _RECORDED_AS_DONE

    if manifest["status"] not in _RECORDED_AS_DONE:
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
    ``base_config``'s own progressive-unfreeze schedule is left untouched:

      - ``lr``           -> ``optimizer["head_lr"]``, plus ``optimizer["backbone_lr"]`` scaled by
                            whatever backbone/head ratio ``base_config`` already expressed
      - ``weight_decay`` -> ``optimizer["weight_decay"]``
      - anything else    -> the top level of ``cfg`` (``batch_size`` included), where
                            ``train()`` reads every key of its own
    """
    import copy

    cfg = copy.deepcopy(base_config)

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


def _selection_metric_axis_conflict(
    base_config: dict, param_space: dict, hpo_metric: str, hpo_mode: str,
) -> str | None:
    """The refusal reason, if any, when some point ``param_space`` could resolve a trial to picks a
    different selection metric or ranking direction than ``(hpo_metric, hpo_mode)``, the pair
    ``run_hyperparameter_search`` resolved from ``base_config`` and fixes on the Tuner.

    Resolves every :func:`_preflight_points` point through
    ``config_selection_metric``/``HIGHER_IS_BETTER_BY_METRIC``, so an axis that changes the
    metric's own task-derived default (``model_source.task``) is caught too. A point whose params
    fail to apply is left for the structural preflight loop to report. ``None`` when every point
    agrees with ``base_config``.
    """
    from tcip_mcp.pipelines.training.evaluation import HIGHER_IS_BETTER_BY_METRIC
    from tcip_mcp.pipelines.training.generic_trainer import config_selection_metric
    axes = sorted(param_space)
    for label, point in _preflight_points(param_space):
        try:
            point_cfg = _apply_hpo_params(base_config, point)
        except ValueError:
            continue
        try:
            point_metric = config_selection_metric(point_cfg)
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
    """``base_config`` as ``run_hyperparameter_search`` mints the sweep from: unchanged unless
    ``split_draws`` is above 1 and the config is bound to a selection, in which case a copy carries
    ``data.split.redraw_within_selection: true`` (defaulting ``data.split.seed`` to
    ``DEFAULT_SEED`` when absent). An already-true flag or an already-set seed is left as it is.
    """
    if split_draws <= 1:
        return base_config
    data_cfg = base_config.get("data") or {}
    split_cfg = data_cfg.get("split") or {}
    if not split_cfg.get("selection_dir"):
        return base_config
    new_split = {**split_cfg, "redraw_within_selection": True}
    new_split["seed"] = split_seed(split_cfg)
    return {**base_config, "data": {**data_cfg, "split": new_split}}


def _split_draws_argument_refusal(split_draws: object) -> str | None:
    """Whether ``split_draws`` itself is a draw count at all, checked before every other leg
    (including :func:`_base_config_for_split_draws`): a value that is not an ``int`` (a ``bool`` is
    an ``int`` and reads as the integer it names) refuses by name, and a value below one refuses by
    name.
    """
    if not isinstance(split_draws, int):
        return (f"split_draws={split_draws!r} is not a draw count; pass a whole number of "
                "partitions to pair.")
    if split_draws < 1:
        return (f"split_draws={split_draws} pairs no partition; a sweep pairs at least one, so "
                "pass 1 (the default, one draw and no spread) or the number of partitions to "
                "pair.")
    return None


def _trial_budget_refusal(
    *, reads_bound: bool, split_draws: int, n_trials: object, search_alg: str, warm_start: bool,
    trial_budget: object, relaunched_from: str | None, search_param_space: dict,
    baseline_params: dict | None,
) -> str | None:
    """Whether this call's own trial count fits the bound it must state or check, run only when
    ``reads_bound`` says one is read (a launch, never a relaunch, above one draw; or any call
    naming ``trial_budget``).

    Runs its two argument clauses first (an ``n_trials`` or ``trial_budget`` that is not a positive
    ``int``, a ``bool`` excluded by name), then counts Ray's own variant count over
    ``search_param_space`` via :func:`~tcip_mcp.pipelines.training.hpo.planned_trial_count` (a
    space that cannot be counted or generated answers its own refusal, naming the exception), then
    its two bound clauses: a launch above one draw naming no ``trial_budget`` refuses naming the
    budget Ray's own count would admit; a stated ``trial_budget`` the count exceeds refuses naming
    the count, the budget, and whether even one draw exceeds it (``count // split_draws`` is one
    draw's own count). Everything else answers ``None``.
    """
    if not reads_bound:
        return None
    if not isinstance(n_trials, int) or isinstance(n_trials, bool) or n_trials <= 0:
        return (f"n_trials={n_trials!r} is not a count of trials; Ray runs a negative count as "
                "an unbounded sweep and a zero count as none, so pass the number of sampled "
                "points, a whole number of at least one.")
    if trial_budget is not None and (
        not isinstance(trial_budget, int) or isinstance(trial_budget, bool) or trial_budget <= 0
    ):
        return (f"trial_budget={trial_budget!r} is not a count of trials; pass the most trials "
                "this sweep may launch, a whole number of at least one, or omit it at one draw.")

    from tcip_mcp.pipelines.training.hpo import planned_trial_count

    try:
        count = planned_trial_count(
            search_param_space, n_trials, search_alg, split_draws, warm_start, baseline_params)
    except (ValueError, KeyError, TypeError, IndexError, OverflowError) as exc:
        return (f"the sweep's param_space cannot be counted as a search space: {exc!r}; correct "
                "the axis that exception names, since tune_search builds the same space and "
                "would meet it too, or run at split_draws=1 stating no trial_budget, where no "
                "count is taken.")

    per_draw = count // split_draws
    warm_state = "on" if warm_start else "off"
    built = (f"Ray's own variant count over the space this call builds: n_trials={n_trials} "
             f"sample(s), search_alg={search_alg!r}, warm start {warm_state}, {per_draw} per draw")

    if split_draws > 1 and trial_budget is None and relaunched_from is None:
        return (f"split_draws={split_draws} pairs every sampled point with {split_draws} "
                f"partitions, so this sweep would launch {count} trials ({built}), a total the "
                f"call never stated; pass trial_budget={count} ({count} admits it as "
                "configured), or run at split_draws=1.")
    if trial_budget is not None and count > trial_budget:
        if per_draw > trial_budget:
            return (f"this sweep would launch {count} trials ({built}) against "
                    f"trial_budget={trial_budget}, which this sweep exceeds at one draw "
                    f"({per_draw} trials per draw); lower n_trials, narrow the gridded axes, or "
                    "state a larger trial_budget.")
        admitted_draws = trial_budget // per_draw
        return (f"this sweep would launch {count} trials ({built}) against "
                f"trial_budget={trial_budget}, which admits at most {admitted_draws} draw(s) at "
                f"these settings; lower split_draws to {admitted_draws}, lower n_trials, narrow "
                "the gridded axes, or state a larger trial_budget.")
    return None


def _split_draws_refusal(
    base_config: dict, param_space: dict | None, task: str, search_alg: str, scheduler: str,
    split_draws: int, split_draw_seeds: list[int] | None, warm_start: bool,
    baseline_params: dict | None,
) -> str | None:
    """Every reason of the paired path's own that ``run_hyperparameter_search`` refuses
    ``split_draws`` above 1 for, checked before minting the sweep. ``None`` when nothing here
    objects, and for one draw.

    A ``base_config`` bound to a selection (already carrying ``data.split.redraw_within_selection``
    from :func:`_base_config_for_split_draws`) skips the ``auto_val`` leg and runs the selection's
    own foreground-groups check instead. An unbound config gets its own last leg
    (:func:`_unbound_single_source_spatial_issue`).
    """
    if split_draws <= 1:
        return None
    from tcip_mcp.pipelines.training.hpo import SPLIT_DRAW_SEED_KEY, _NATIVE_SEARCH, _NO_SCHEDULER

    data_cfg = base_config.get("data") or {}
    split_cfg = data_cfg.get("split") or {}
    bound = bool(split_cfg.get("selection_dir"))
    if not bound and not data_cfg.get("auto_val", True):
        return ("split_draws needs a drawn validation split, and base_config sets "
                "data.auto_val=False.")
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
    if split_draw_seeds is not None and len(set(split_draw_seeds)) != len(split_draw_seeds):
        repeated = sorted({s for s in split_draw_seeds if split_draw_seeds.count(s) > 1})
        return (f"split_draw_seeds repeats {repeated}: a spread over the same partition drawn "
                "twice is not a spread, and group_split_draws only counts a point eligible on "
                "its distinct planned seeds, not on one seed completed twice.")
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
        return _bound_redraw_starvation_issue(split_cfg)
    return _unbound_single_source_spatial_issue(task, data_cfg, split_draws)


def _bound_redraw_starvation_issue(split_cfg: dict) -> str | None:
    """Whether ``run_hyperparameter_search``'s ``split_draws`` minting a redraw sweep over a bound
    ``base_config``'s selection would starve every trial the identical way
    (:func:`~tcip_mcp.pipelines.data.splits.redraw_starved_issue`), or ``None``.
    """
    from tcip_mcp.pipelines.data.selection import read_selection
    from tcip_mcp.pipelines.data.splits import redraw_pool, redraw_starved_issue

    selection_dir = split_cfg["selection_dir"]
    try:
        selection = read_selection(selection_dir)
    except ValueError as exc:
        return f"split_draws: {exc}"
    return redraw_starved_issue(*redraw_pool(selection), selection_dir=selection_dir,
                                seed=split_cfg.get("seed"))


def _unbound_single_source_spatial_issue(task: str, data_cfg: dict, split_draws: int) -> str | None:
    """Whether ``split_draws`` above 1 would redraw the identical partition on every trial: an
    unbound, built-in detection config with tiling on that admits exactly one source takes
    ``auto_train_val``'s single-source spatial-strip branch
    (:func:`~tcip_mcp.pipelines.data.split_construction.spatial_single_source_split`), which no
    draw of ``data.split.seed`` varies.

    Counts the members ``base_config`` admits through
    :func:`~tcip_mcp.pipelines.data.label_queries.admit`, inside one handler answering no refusal
    on any exception. ``None`` when ``task`` is not ``"detection"``, ``data.tiling`` is absent, not
    a mapping, or disabled, a bespoke ``dataset_source`` is named, the admitted count could not be
    resolved, or the admitted count is not exactly one.
    """
    if task != "detection":
        return None
    from tcip_mcp.pipelines.model_build import DATASET_SOURCE_KEY

    if data_cfg.get(DATASET_SOURCE_KEY):
        return None
    tiling = data_cfg.get("tiling")
    if not isinstance(tiling, dict) or not tiling or not tiling.get("enabled", True):
        return None

    from tcip_mcp.pipelines.data.label_queries import admit_run

    try:
        admitted = admit_run(data_cfg)
    except Exception:
        return None
    if len(admitted.records) != 1:
        return None
    return (
        f"split_draws={split_draws} redraws the split, and base_config admits one trainable "
        "source under data.labels_dir: a detection sweep with tiling on takes the single-source "
        "spatial strip path, whose partition is placed by declared order and does not vary with "
        "data.split.seed, or trains with no validation, or fails on a reserved calibration "
        "fraction; in every case no draw holds a different partition out and the spread would be "
        "training-seed noise. Run at split_draws=1, or sweep a dataset with two or more admitted "
        "sources or a config bound to a selection."
    )


class SeedAxisRefusal(NamedTuple):
    """Why a caller-supplied ``data.split.seed`` axis in ``param_space`` refuses at one draw:
    ``reason`` names the fact and the breeder's own next step; ``remedy`` names the Ray mechanics
    and what the tool's caller passes instead.
    """

    reason: str
    remedy: str


_SEED_AXIS_REASON = (
    "this sweep varied the split seed itself, so it cannot be replayed as recorded; "
    "ask the agent to run it again"
)
_SEED_AXIS_REMEDY = (
    "at one draw a caller-supplied data.split.seed axis is sampled or gridded with the point, "
    "the best is Ray's own pick over seeds and points together (so best_params would name a "
    "seed), a pruning scheduler can end one seed's trial early, and the sweep computes no "
    "spread; drop data.split.seed from param_space and pass split_draws=<n> (with "
    "split_draw_seeds for chosen, distinct seeds), which pairs every seed with every point and "
    "picks the best by mean over draws. The paired path's own conditions apply, bound and "
    "unbound alike: a config bound to a selection redraws train and val inside the selection's "
    "own train and val samples (the selection must be readable and resolve at least two "
    "foreground groups across them); an unbound config keeps auto_val on; search_alg "
    "is one the native generator builds (random, grid, variant_generator, or unset); scheduler "
    "prunes nothing (none, fifo, or unset); split_draw_seeds is one per draw and distinct; no "
    "baseline_params names the seed under a warm start; no other data.* axis is in param_space; "
    "a trial_budget is stated on a launch that is not a relaunch, and Ray's variant count over "
    "the sweep fits under it; and, for a built-in detection config with tiling on, more than one "
    "trainable source is admitted under data.labels_dir (a single admitted source's own "
    "single-source spatial-strip path pairs no distinct partition with any draw). A single fixed "
    "seed belongs in "
    "base_config's own data.split.seed: the drawn path's partition depends on it, and the "
    "single-source spatial path's, or a selection-bound config's without redraw_within_selection, "
    "never does (the spatial path still records the config's value on the split record, a "
    "different fact, not claimed here)."
)


def coerce_split_draws(split_draws: object) -> int | None:
    """``split_draws`` read from a manifest of unknown provenance: ``None`` stands for 1; an
    ``int`` is read directly, whatever its sign, a ``bool`` included; a ``str`` or a finite
    ``float`` is read as the integer its ``int()`` names when that integer equals the value it was
    given, so ``2`` and ``"2"`` and ``2.0`` all read as ``2`` while ``2.5`` reads as ``None``.
    Carries no lower bound of its own. A non-numeric string, a non-finite float and anything else
    ``int()`` cannot read answer ``None``.
    """
    if split_draws is None:
        return 1
    if isinstance(split_draws, int):
        return int(split_draws)
    if not isinstance(split_draws, (float, str)):
        return None
    try:
        coerced = int(split_draws)
    except (TypeError, ValueError, OverflowError):
        return None
    try:
        if float(coerced) != float(split_draws):
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return coerced


def caller_split_seed_refusal(
    param_space: object, split_draws: object,
) -> SeedAxisRefusal | None:
    """Whether ``param_space`` names ``data.split.seed`` as its own axis while ``split_draws``
    draws at most one partition: refused whatever the sampler.

    ``split_draws`` is read through :func:`coerce_split_draws`; a value it cannot read refuses
    nothing, and neither does a ``param_space`` that is not a mapping.
    """
    from tcip_mcp.pipelines.training.hpo import SPLIT_DRAW_SEED_KEY

    draws = coerce_split_draws(split_draws)
    if draws is None or draws > 1:
        return None
    if not isinstance(param_space, dict):
        return None
    if SPLIT_DRAW_SEED_KEY not in param_space:
        return None
    return SeedAxisRefusal(reason=_SEED_AXIS_REASON, remedy=_SEED_AXIS_REMEDY)


def group_split_draws(all_trials: list[dict], planned_seeds: list[int]) -> list[dict]:
    """Group ``tune_search``'s own ``all_trials`` rows by the point each draw shares (every param
    but ``hpo.SPLIT_DRAW_SEED_KEY``), each group carrying the ``split_draws`` block the sweep
    result and its manifest both record.

    ``planned_seeds`` names every seed the sweep asked for; a group is ``eligible`` for best only
    when every one of them completed for that point and the group holds no errored or
    never-answered row. Pass an empty list to accept any single complete row per point regardless
    of seed identity.

    A group's block always carries ``n`` (every row seen for the point, ``COMPLETE`` and ``ERROR``
    alike), ``n_complete`` (rows among them that completed with a real value) and
    ``seeds_complete`` (the distinct seeds among those complete rows, sorted), plus ``seeds`` (one
    entry per complete row, not deduplicated), ``values``, ``mean``, ``std`` (the sample standard
    deviation, ``None`` under two values), ``min`` and ``max`` over ``values``. A row with no
    ``params`` at all forms its own singleton, ineligible group, ``point`` ``None``.
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
    """One deterministic point from ``param_space``, spanning its declared range or choices."""
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
    """Every point ``run_hyperparameter_search``'s preflight must check: the first sampled corner,
    plus one variant per categorical choice and one per numeric bound, each holding every other
    axis at its first sampled value.
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
    experiment_id: str, config: dict, data_source, resume_from: str,
    *, output_base: str, launched_by: dict[str, Any], dataset_id: str | None = None,
    dataset_fingerprint: str | None = None,
) -> tuple[str, str]:
    """Create or attach the experiment for a run, enforcing experiment immutability.

    Returns ``(experiment_id, output_dir)``: the id actually used, and its directory (``output_base
    / experiment_id``). An existing id may be reused only when the experiment is pristine (state
    'created', no metrics), in which case its ``config.json`` is refreshed with the config this run
    is launching. Anything else, including a ``resume_from`` that targets an id which already has
    recorded history, mints a fresh ``<id>_<minted>`` (with the old id as parent lineage).

    Every branch stamps this experiment through :func:`~tcip_mcp.experiments.stamp_run_identity`,
    one compare-and-set transaction that moves the record to ``running`` (the pristine-reuse branch
    writing its config in that same transaction). A stamp whose precondition fails falls to the
    fork. ``launched_by`` is the one declaration ``launch_training`` resolved.
    """
    from tcip_mcp.experiments import StampPreconditionFailed, create_experiment, stamp_run_identity

    output_dir = str(Path(output_base) / experiment_id)
    created = create_experiment(experiment_id, config, data_source=data_source,
                                dataset_id=dataset_id, dataset_fingerprint=dataset_fingerprint)
    if "error" not in created:
        try:
            stamp_run_identity(experiment_id, output_dir, launched_by=launched_by)
            return experiment_id, output_dir
        except StampPreconditionFailed:
            pass  # a concurrent pristine-reuse claimed this record before this call's own stamp
    else:
        try:
            stamp_run_identity(experiment_id, output_dir, launched_by=launched_by, config=config)
            return experiment_id, output_dir
        except StampPreconditionFailed:
            pass  # lost the race for this record between the pristine check and the stamp

    from tcip_mcp.experiments import mint_experiment_id

    fresh_id = f"{experiment_id}_{mint_experiment_id()}"
    logger.warning(
        "experiment_id %s already has a run; experiments are immutable, tracking "
        "this run as %s instead.", experiment_id, fresh_id,
    )
    # The snapshot must name itself, not whatever id the caller's own config carried in (its
    # parent, for a relaunch that set config["experiment_id"] to the picked id before this call).
    forked_config = {**config, "experiment_id": fresh_id}
    forked = create_experiment(fresh_id, forked_config, parent_experiment=experiment_id,
                               data_source=data_source, dataset_id=dataset_id,
                               dataset_fingerprint=dataset_fingerprint)
    if "error" in forked:
        raise RuntimeError(
            f"_ensure_experiment: could not create the fork {fresh_id!r}: {forked['error']}")
    fresh_output_dir = str(Path(output_base) / fresh_id)
    stamp_run_identity(fresh_id, fresh_output_dir, launched_by=launched_by)
    return fresh_id, fresh_output_dir


def _one_real_batch(task: str, config: dict, n: int = 2):
    """``(batch, reason_it_failed)``, one collated ``(images, targets)`` from the run's own
    training loader, resolved through
    :func:`~tcip_mcp.pipelines.data.split_construction.auto_train_val` over a deep copy of the
    config. A config that cannot yield a batch returns ``(None, reason)``.
    """
    data_cfg = config.get("data") or {}
    try:
        import copy

        from tcip_mcp.pipelines.data.split_construction import auto_train_val
        from tcip_mcp.pipelines.training.collation import task_collate
        from tcip_mcp.pipelines.training.generic_trainer import run_transforms

        train_ds, _val_ds, _partition = auto_train_val(
            task, copy.deepcopy(data_cfg), run_transforms(config))
        assert isinstance(train_ds, Sized), "every build_dataset task backend defines __len__"
        indexable: Any = train_ds
        items = [indexable[i] for i in range(min(n, len(train_ds)))]
        if not items:
            return None, "the dataset built but is empty"
        return task_collate(task)(items), None
    except Exception as exc:  # noqa: BLE001, an unbuildable batch is a caller decision, not a crash
        logger.info("could not build a real batch to smoke task %r: %s", task, exc)
        return None, f"{type(exc).__name__}: {exc}"


def _reserve_calibration_feasibility_issues(
    task: str, data_cfg: dict, split_cfg: dict, reserve_cal_frac: float, *,
    run: RunPopulation, sizes: "Mapping[str, int]", smoke: bool,
) -> list[str]:
    """Named ``preflight_config`` issues for an explicitly-requested
    ``reserve_calibration_fraction`` that cannot be honored.

    Structurally inapplicable configs (not detection, tiling disabled, a multi-member dataset) are
    always flagged from ``run``, the run's own admitted samples and class space. The single-source
    geometry (extent, strip-layout feasibility, an empty side after real filtering) is checked by
    calling :func:`~tcip_mcp.pipelines.data.split_construction.spatial_single_source_split` over
    that same single sample, only when ``smoke=True``.
    """
    tiling_cfg = data_cfg.get("tiling") if isinstance(data_cfg, dict) else None
    if task != "detection" or not tiling_cfg or not tiling_cfg.get("enabled", True):
        return [
            f"data.split.reserve_calibration_fraction={reserve_cal_frac} has no effect: it only "
            "applies to a detection task with tiling enabled (the single-source spatial-strip "
            f"split), this config's task is {task!r} with tiling={tiling_cfg!r}."
        ]

    if _data_dir_issues(data_cfg):
        return []  # preflight names each of them

    if len(run.samples) >= 2:
        return [
            f"data.split.reserve_calibration_fraction={reserve_cal_frac} has no effect: "
            f"{len(run.samples)} admitted sources resolve to the group-balanced multi-member "
            "split, not the single-source spatial-strip split a calibration region reserves from."
        ]
    if len(run.samples) != 1 or not smoke:
        return []

    from tcip_annotation.json_io import UnreadableLabelDocument

    try:
        from tcip_mcp.pipelines.data.split_construction import spatial_single_source_split

        # The sizes preflight already resolved for this run; this probe reports feasibility over
        # the dataset that run would build and resolves nothing of its own.
        spatial_single_source_split(
            run.samples[0], run.scope, tiling_cfg, dict(split_cfg), None, sizes)
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
    experiment_id_or_ckpt: str,
    images_dir: str,
    labels_dir: str = "",
    conf_threshold: float | None = None,  # report/select at the ship point
    iou_threshold: float = 0.5,
    iou_type: str | None = None,
    max_dets: int | None = None,
    tiling: dict | None = None,
    use_tiled_inference: bool = False,
    global_nms_iou: float | None = None,
    postprocess: str = DEFAULT_POSTPROCESS,
    trait: str | None = None,
    subject: str | None = None,
    attribute: str | None = None,
    selection_dir: str | None = None,
) -> dict:
    """Evaluate a trained checkpoint on a (held-out) dataset and write test_results.json.

    Computes the same per-task metrics as validation, detection/instance_seg get pycocotools mAP +
    precision/recall/F1; classification/ordinal/regression get the in-house scalar metrics, and
    writes ``test_results.json`` beside the checkpoint.

    Three detection eval regimes:
      * Untiled default (no ``tiling``, checkpoint trained without tiling) -> single full-res
      forward pass, ``eval_regime="full-frame-single-pass"``: the delivery gate for a checkpoint
      never tile-trained. ``use_tiled_inference`` for such a checkpoint refuses (see below).
      * ``tiling`` set (or a run id whose training was tiled, reused automatically) -> tile-level
      diagnostic that matches the training-run val mAP; not the delivery metric.
      * ``use_tiled_inference=True`` -> the delivery-grade full-frame metric for a tile-trained
      checkpoint (tiled inference reconstructed to full frame, matched to full-frame GT). Tile
      geometry is resolved from the checkpoint's own persisted or native-frame training geometry,
      or an explicit override; a checkpoint with none of those refuses (see
      ``run_full_frame_evaluation``).

    Args:
        experiment_id_or_ckpt: An experiment id this process launched (uses its ``model_best.pt``,
            resolved through the in-process registry alone) or a checkpoint path. Either way the
            resolved checkpoint must be registered under this process's platform state root
            (``register_model``, explicit mode for a foreign or bespoke checkpoint) or this door
            refuses before loading it.
        images_dir: Images directory for the evaluation split.
        labels_dir: Labels dir (detection/instance_seg), masks dir (semantic_seg), or the GT CSV
            path (classification/ordinal/regression, one row per image stem); the task is the
            checkpoint's own.
        conf_threshold: Operating confidence for P/R/F1. ``None`` (default) resolves to the
            platform default (``DEFAULT_CONF``) on every regime; an explicit value is honored
            verbatim.
        iou_threshold: Operating IoU (on COCOeval's grid; 0.5 -> index 0).
        iou_type: 'bbox' or 'segm'. Default (None) auto-resolves from the task, 'segm' for
            instance_seg, 'bbox' otherwise.
        max_dets: Full-frame/COCOeval detection cap. ``None`` (default) resolves per-regime, 100
            (the COCOeval ``maxDets`` convention) on the tile-level diagnostic path, 1000
            (``DEFAULT_MAX_DETS``) on the delivery-grade ``use_tiled_inference`` path. An explicit
            value is honored verbatim on both paths; the delivery-grade path stamps a per-image
            ``cap_hit``/``max_dets_cap_saturated_frac``.
        tiling: Optional detection tiling dict ({enabled, tile_size, overlap, ...}) for a
            tile-level eval. None + a run id reuses the run's training tiling; None + a checkpoint
            path stays untiled.
        use_tiled_inference: Score the delivery regime (full-frame via tiled inference).
        global_nms_iou: Cross-tile global NMS IoU threshold (tiled paths only). ``None`` (default)
            resolves to the platform default (``DEFAULT_NMS_IOU``); an explicit value is honored
            verbatim.
        postprocess: Cross-tile merge, "nms" suppresses overlaps, "nmm" unions boxes split across a
            tile seam.
        trait: When set, the trait's derived localization criterion (traits.py, e.g. a count
            trait's center-match) governs the reported count and the selection f1; AP@0.5
            (``iou_threshold``) is kept as a labeled comparability metric. Absent -> the IoU
            convention governs.
        subject: Name-based GT scope. Caller-supplied wins; else resolved from the producing run's
            own config.
        attribute: Attribute scope for the same name-based GT resolution as ``subject``.
        selection_dir: Score the checkpoint over this selection's ``calibration`` samples whose
            label documents live under ``labels_dir`` instead of the whole directory, refusing by
            name the way the calibration door does (detection/instance_seg only, and not combined
            with ``use_tiled_inference``), with a floor of one foreground group.
            ``test_results.json`` then records ``selection_dir`` and the evaluated stem count, the
            loader's own count, refused by name (naming the difference and the remedy) when the
            loader admits fewer than the universe the selection drew; omitted, the whole directory
            is scored.
    """
    import torch
    from torch.utils.data import DataLoader

    from tcip_mcp.pipelines.training.generic_trainer import checkpoint_key
    from tcip_mcp.pipelines.training.collation import task_collate
    from tcip_mcp.pipelines.training.run_registry import get_run
    from tcip_mcp.pipelines.training.eval_runners import (
        run_full_frame_evaluation, run_test_evaluation,
    )
    from tcip_mcp.pipelines.data.datasets import build_dataset, resolve_sizes
    from tcip_mcp.pipelines.inference.predictor import build_predictor
    from tcip_mcp.pipelines.resolution import applied_operating_point

    # The tile-level/single-pass paths apply this directly below; the full-frame path resolves
    # its own sentinels internally, so its own caller passes the raw arguments through unchanged.
    applied_conf, _applied_nms_iou, _applied_max_dets = applied_operating_point(
        conf_threshold, global_nms_iou, None)

    ckpt = experiment_id_or_ckpt
    run = None
    if not Path(ckpt).is_file():
        run = get_run(experiment_id_or_ckpt)
        if run is None:
            return {"error": f"Not a checkpoint path or known experiment id: {experiment_id_or_ckpt}"}
        ckpt = str(store.blob_path(checkpoint_key(run.output_dir, "model_best")))
    if not Path(ckpt).is_file():
        return {"error": f"Checkpoint not found: {ckpt}"}

    from tcip_mcp.model_registry import UnregisteredCheckpoint, load_registered_checkpoint

    try:
        checkpoint = load_registered_checkpoint(ckpt)
        task = checkpoint.task
    except (UnregisteredCheckpoint, ValueError) as exc:
        return {"error": str(exc)}

    from tcip_mcp.pipelines.data.selection import ClassScope

    # A bare checkpoint path (run is None) carries its own stamped config["data"] too, read off
    # the object already loaded, so both paths agree without a second read.
    run_data_cfg = (run.config.get("data") or {}) if run is not None else checkpoint.data_config
    run_tiling = run_data_cfg.get("tiling")
    # Caller-supplied subject/attribute win, else the run's own recorded class space through the
    # one reader of it, so the name-based GT reads under the space the run trained in.
    recorded_scope = ClassScope.recorded_in(run_data_cfg)
    subject = subject or recorded_scope.subject
    attribute = attribute or recorded_scope.attribute

    selection_stems: list[str] | None = None
    selection_samples: dict[str, Any] = {}
    selection_scope = None
    if selection_dir is not None:
        if use_tiled_inference:
            return {"error": "selection_dir is not combined with use_tiled_inference: that "
                             "delivery-grade path scans images_dir/labels_dir on its own, never "
                             "narrowed to a selection's samples."}
        from tcip_mcp.pipelines.data.selection import read_selection
        from tcip_mcp.pipelines.data.splits import selection_calibration_universe

        selection = read_selection(selection_dir)
        try:
            (selection_stems, _group_by, _group_key_map, _excluded, _counts,
             selection_samples) = selection_calibration_universe(
                selection, labels_dir, min_foreground_groups={"calibration": 1})
        except ValueError as exc:
            return {"error": str(exc)}
        # The selection's own class space governs a selection-restricted measurement.
        selection_scope = selection.scope
        subject, attribute = selection_scope.subject, selection_scope.attribute

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
                max_dets=max_dets, trait=trait,
            )
        except (ValueError, UnreadableLabelDocument) as exc:
            return {"error": str(exc)}

    # Tile-level diagnostic (or untiled). Only detection tiles; a run id reuses its training tiling.
    if tiling is None and run is not None:
        tiling = run_tiling
    if task != "detection":
        tiling = None

    # The checkpoint's own predictor, built once: the width it reads images at sizes the loader and
    # its model scores them, at the operating point it was built with (score_threshold unstated).
    try:
        predictor = build_predictor(checkpoint, score_threshold=None)
    except ValueError as exc:
        return {"error": str(exc)}

    universe: list[Any] = []
    if selection_stems is not None:
        from tcip_mcp.pipelines.data.label_queries import refuse_inadmissible_samples

        universe = [selection_samples[s] for s in selection_stems]
        # A loader indexes exactly what it is handed, so a label emptied since the
        # draw measures as an image with no objects unless the admission refuses it here.
        try:
            refuse_inadmissible_samples(universe, selection_scope)
        except ValueError as exc:
            return {"error": str(exc)}

    evaluated_stem_count = None
    try:
        if selection_stems is not None:
            # The universe's own recorded samples under the class space the selection recorded:
            # the pixels measured are the ones the draw held out, in the run's own vocabulary.
            measured_samples, measured_scope = universe, selection_scope
        else:
            # Through the producer, over the ground truth this door was pointed at, so the door
            # and a run over the same data admit one membership.
            from tcip_mcp.pipelines.data.label_queries import admit, require_admitted

            admitted = admit(images_dir, labels_dir, subject=subject, attribute=attribute)
            require_admitted(admitted)
            measured_samples, measured_scope = admitted.every_sample(), admitted.scope
        # Read at the width the predictor reads at: the model scores these tensors, so a loader
        # sized off the references instead would hand it images of another shape.
        dataset = build_dataset(
            task, samples=measured_samples, tiling=tiling, scope=measured_scope,
            sizes=resolve_sizes(task, {"num_channels": predictor.in_chans}, measured_samples))
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Failed to build dataset: {exc}"}

    if selection_stems is not None:
        from tcip_mcp.pipelines.data.datasets import indexed_sample_keys

        # Admission answers for the samples; this answers for the loader built from them, which
        # tiling can leave naming no example at all for a held-out source.
        retained = indexed_sample_keys(dataset)
        dropped = sorted(s for s in selection_stems
                         if selection_samples[s].identity not in retained)
        if dropped:
            return {"error": f"the selection's calibration universe under {labels_dir!r} holds "
                             f"{len(selection_stems)} sample(s), and the loader indexes nothing "
                             f"for {len(dropped)} of them ({dropped[:5]}). This run's tiling keeps "
                             "no tile of those sources, so the measurement would be taken over "
                             "part of the universe the draw held out and reported as the whole of "
                             "it. Widen the tiling's keep regions, stop skipping empty tiles, or "
                             "evaluate untiled."}
        evaluated_stem_count = len(retained)

    loader = DataLoader(dataset, batch_size=4, collate_fn=task_collate(task))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 100 is the COCOeval maxDets convention for this tile-level/diagnostic regime, distinct
    # from the delivery-grade path's 1000 above; an explicit caller max_dets is honored verbatim.
    resolved_max_dets = 100 if max_dets is None else max_dets
    return run_test_evaluation(
        checkpoint, predictor.model, loader, device, str(Path(ckpt).parent),
        conf_threshold=applied_conf, iou_threshold=iou_threshold,
        iou_type=iou_type, max_dets=resolved_max_dets, tiling=tiling, trait=trait,
        selection_dir=selection_dir,
        evaluated_stem_count=evaluated_stem_count,
    )
