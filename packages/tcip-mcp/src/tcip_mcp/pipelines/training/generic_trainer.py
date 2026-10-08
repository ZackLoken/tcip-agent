"""Task-agnostic training loop for a bespoke ``model_source`` model.

This trainer works with any task type (detection, classification, ordinal, regression,
segmentation) because it delegates everything to the model's forward() which returns a loss dict in
train mode.

Provides: progressive unfreezing, early stopping, mixed precision, gradient accumulation,
    checkpoints; metrics leave through the caller's callbacks.
"""

from __future__ import annotations

import functools
import logging
import math
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from tcip_store import scalar_number, stored_numbers

from tcip_mcp.pipelines.data.datasets import indexed_sample_keys, instance_targets
from tcip_mcp.pipelines.model_contract import DETECTION_TASKS, TCIPModel
from tcip_mcp.pipelines.model_build import (
    CONFIG_KEY,
    METRICS_KEY,
    STATE_DICT_KEY,
    build_from_model_source,
    recorded_model_dims,
    run_task,
)
from tcip_mcp.pipelines.schemas import (CosineSchedule, OneCycleSchedule, PlateauSchedule,
                                        SchedulerSpec, StageSpec, TrainConfigSchema)
from tcip_mcp.pipelines.training.evaluation import (
    HIGHER_IS_BETTER_BY_METRIC,
    VAL_LOSS_KEY,
    VAL_METRIC_PREFIX,
    evaluate,
)
from tcip_mcp.pipelines.training.optimizer_factory import (
    GROUPS_KEY,
    TRAINING_STATE_KEYS,
    build_optimizer,
    capture_training_state,
    captured_sources,
    compute_lr_scale,
    restore_training_state,
)
from tcip_mcp.pipelines.training.run_registry import TrainRun

if TYPE_CHECKING:
    from tcip_mcp.traits import TraitEntry

logger = logging.getLogger(__name__)

BATCH_ROWS_PER_EPOCH = 10
"""Per-batch rows an epoch emits when ``log_every_n_batches`` is unstated
(:func:`batch_row_ends`): owner ruling."""


def batch_row_ends(n_batches: int) -> set[int]:
    """The one-based batch counts within an epoch of ``n_batches`` batches after which a
    per-batch row is emitted by default: :data:`BATCH_ROWS_PER_EPOCH` evenly spaced ends, the
    last the epoch's own, or every batch of an epoch shorter than that."""
    rows = min(BATCH_ROWS_PER_EPOCH, n_batches)
    return {math.ceil(k * n_batches / rows) for k in range(1, rows + 1)}


def run_device(spec: TrainConfigSchema) -> torch.device:
    """The device a validated config (``schemas.train_config``) trains on: its ``device``, or
    cuda when available and cpu otherwise."""
    return torch.device(spec.device or ("cuda" if torch.cuda.is_available() else "cpu"))


def _warmup_starts(model, optimizer, captured: list[dict],
                   target_lrs: list[float]) -> list[float]:
    """The learning rate each of ``optimizer``'s param groups warms up from at a stage boundary,
    read from ``captured``, the handed-off capture's param groups (its :data:`GROUPS_KEY`),
    through their membership (``optimizer_factory.captured_sources``): the one rate its members'
    captured groups ran at, ``0.0`` for a group whose members no captured group held (they were
    not training), and its own target for an empty group. Refuses (``ValueError``) a group mixing
    held and unheld members, or members whose captured groups ran at different rates."""
    starts = []
    for index, (sources, target) in enumerate(
            zip(captured_sources(model, optimizer, captured), target_lrs)):
        if not sources:
            starts.append(target)
        elif sources == {None}:
            starts.append(0.0)
        else:
            rates = {captured[i]["settings"]["lr"] for i in sources if i is not None}
            if None in sources or len(rates) != 1:
                raise ValueError(
                    f"param group {index} has no one rate to warm up from: its members ran at "
                    f"{sorted(rates)}{' and some were not training' if None in sources else ''}.")
            starts.append(rates.pop())
    return starts


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed random / numpy / torch (+ cuda) for reproducible runs.

    ``deterministic`` additionally forces cuDNN deterministic algorithms
    (``cudnn.deterministic=True``, ``cudnn.benchmark=False``).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def capture_rng_state(loader_generator: torch.Generator | None) -> dict[str, Any]:
    """Snapshot every random stream a run draws from: the four set_seed seeds and
    ``loader_generator``, the generator a loader shuffles with (``None`` for a loader that has
    none), restorable with :func:`restore_rng_state`."""
    return {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader_rng_state": None if loader_generator is None else loader_generator.get_state(),
    }


def restore_rng_state(state: dict[str, Any], loader_generator: torch.Generator | None) -> None:
    """Restore the streams :func:`capture_rng_state` captured, the loader's into
    ``loader_generator``. Refuses (``ValueError``) a capture and a loader that disagree on whether
    the loader draws from a generator of its own."""
    if (state["loader_rng_state"] is None) != (loader_generator is None):
        raise ValueError(
            "the captured random state and this loader disagree on whether the loader shuffles "
            "from a generator of its own; resume with a loader built the way the run's was.")
    torch.set_rng_state(state["torch_rng_state"])
    if state["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    random.setstate(state["python_rng_state"])
    if loader_generator is not None:
        loader_generator.set_state(state["loader_rng_state"])


def loader_worker_init(worker_id: int, seed: int | None = None,
                       num_workers: int | None = None) -> None:
    """DataLoader worker-process initializer: configures the platform GDAL cache budget at
    ``1 / num_workers`` of it per worker (the whole budget for ``None``), and seeds numpy and
    random per worker when ``seed`` is given."""
    from tcip_mcp.pipelines.raster_source import configure_gdal_cache

    share = 1.0 / num_workers if num_workers else 1.0
    configure_gdal_cache(share=share)
    if seed is not None:
        worker_seed = (seed + worker_id) % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)


def seeded_loader_kwargs(seed: int | None, num_workers: int | None = None) -> dict:
    """DataLoader kwargs wiring :func:`loader_worker_init` as the ``worker_init_fn`` (scaled by
    ``num_workers`` when given), plus a ``generator`` seeded with ``seed`` when one is given, none
    otherwise."""
    kwargs: dict = {
        "worker_init_fn": functools.partial(
            loader_worker_init, seed=None if seed is None else int(seed), num_workers=num_workers),
    }
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        kwargs["generator"] = generator
    return kwargs


def run_transforms(spec: TrainConfigSchema) -> Any:
    """The training transforms a validated config's (``schemas.train_config``) ``augmentation``
    declares, or ``None`` for none."""
    augmentation = spec.augmentation
    if not augmentation:
        return None
    from tcip_mcp.pipelines.data.augmentations import build_augmentation

    return build_augmentation(augmentation)


def run_loaders(spec: TrainConfigSchema, task: str, train_ds: Any, val_ds: Any
                ) -> tuple[DataLoader, DataLoader | None]:
    """A run's train loader and, when ``val_ds`` is given, its val loader, at a validated
    config's (``schemas.train_config``) ``batch_size``, ``num_workers`` and ``sampler``, seeded
    from its ``seed``."""
    from tcip_mcp.pipelines.data.samplers import build_sampler
    from tcip_mcp.pipelines.training.collation import task_collate

    batch_size, num_workers = spec.batch_size, spec.num_workers
    loader_kwargs = seeded_loader_kwargs(spec.seed, num_workers=num_workers)
    # Built after the loader context is known: read order depends on the worker regime too.
    sampler = build_sampler(spec.sampler, train_ds,
                            num_workers=num_workers, batch_size=batch_size)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler,
        collate_fn=task_collate(task), num_workers=num_workers, **loader_kwargs)
    val_loader = None if val_ds is None else DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=task_collate(task),
        num_workers=num_workers, **loader_kwargs)
    return train_loader, val_loader


def stamp_effective_data_geometry(data_cfg: dict, train_ds: Any) -> dict | None:
    """Record the input geometry ``train_ds`` actually serves into ``data_cfg``, in place, or
    ``None`` for a dataset the platform did not build.

    A run whose loaders came from a bespoke ``data.dataset_source`` builder stamps nothing and
    answers ``None``: this reads a dataset's own tile attributes and probes its sources, and a
    bespoke dataset exposes none of them.

    ``data_cfg`` is the live ``config["data"]`` dict the run persists, so this must run after the
    dataset is built and before training starts.

    A tiled train dataset (one carrying a ``tile_size``) stamps its effective
    ``tile_size``/``overlap`` into the tiling record, filling in defaults the caller's config
    omitted. An untiled one replaces the tiling record with ``{"enabled": False}`` outright, never
    a merge.

    ``train_native_size``: an untiled run whose training frames all share one size stamps
        ``data_cfg["train_native_size"] = [width, height]``; mixed sizes stamp nothing. Tiled runs
        stamp nothing here either: the stamped ``tile_size`` carries their frame. Probing is
        header-only for the common containers (``image_dimensions``) and needs the dataset's source
        list.

    Returns the stamped facts, ``{"tiling": dict, "tiling_replaced": bool, "train_native_size": [w,
    h] | None}``.
    """
    from tcip_mcp.pipelines.model_build import DATASET_SOURCE_KEY

    if data_cfg.get(DATASET_SOURCE_KEY):
        return None
    eff_tile = getattr(train_ds, "tile_size", None)
    if eff_tile is not None:
        tiling = data_cfg.setdefault("tiling", {})
        tiling["tile_size"] = int(eff_tile)
        eff_overlap = getattr(train_ds, "overlap", None)
        if eff_overlap is not None:
            tiling["overlap"] = float(eff_overlap)
        return {"tiling": tiling, "tiling_replaced": False, "train_native_size": None}

    data_cfg["tiling"] = {"enabled": False}
    native = _uniform_native_size(train_ds)
    if native is not None:
        data_cfg["train_native_size"] = list(native)
    return {"tiling": data_cfg["tiling"], "tiling_replaced": True,
            "train_native_size": list(native) if native is not None else None}


def _uniform_native_size(train_ds: Any) -> tuple[int, int] | None:
    """The one ``(width, height)`` every training source shares, or ``None`` when sizes differ,
    a source cannot be probed, or the dataset exposes no source list to probe."""
    stems = sorted(indexed_sample_keys(train_ds))
    resolve = getattr(train_ds, "image_of", None)
    if not stems or resolve is None:
        return None
    from tcip_mcp.pipelines.image_utils import image_dimensions

    channels = train_ds.expected_channels
    size: tuple[int, int] | None = None
    for stem in stems:
        try:
            dims = image_dimensions(resolve(stem), channels)
        except (OSError, ValueError):
            return None
        if size is None:
            size = (int(dims[0]), int(dims[1]))
        elif (int(dims[0]), int(dims[1])) != size:
            return None
    return size


def checkpoint_path(run_dir: Path | str, name: str) -> Path:
    """One checkpoint of the run whose directory is ``run_dir``: ``<run_dir>/<name>.pt``, ``name``
    being ``model_best``, ``model_final``, an epoch checkpoint or a bespoke loop's own tag. A name
    that is not one file name refuses with ``BadKeyError`` (``experiments.run_name``)."""
    from tcip_mcp.experiments import run_name

    return Path(run_dir) / f"{run_name(name)}.pt"


def write_checkpoint(payload: dict, path: Path) -> Path:
    """Publish one checkpoint at ``path`` once it is whole (``experiments.publish_once``) and
    return it. A name already written refuses with ``FileExistsError``."""
    from tcip_mcp.experiments import publish_once

    publish_once(path, lambda handle: torch.save(payload, handle))
    return path


def _checkpoint_metrics(metrics: dict) -> dict:
    """One epoch's metrics normalized the way every destination stores them
    (:func:`stored_numbers`): a diverged run's ``nan`` reads back as ``null`` plus a state
    companion.
    """
    return stored_numbers(metrics)


@dataclass(frozen=True)
class _ResumeState:
    """The resume contract beside the training state the checkpoint carries at its top level
    (:func:`capture_training_state`'s keys): each field a key :func:`_save_checkpoint` writes
    and a resume reads back.

    ``best`` is the run's best epoch so far and ``stage_best`` the current stage's, each an
    :func:`_epoch_state` or ``None``; ``warmup_groups`` are the param groups (:data:`GROUPS_KEY`)
    of the capture the current stage warms up from, which :func:`_warmup_starts` reads by
    membership, ``None`` for a stage with no warmup.
    """

    scheduler_state_dict: Any
    scaler_state_dict: Any
    stage: int
    stage_epoch: int
    epoch: int
    best: dict | None
    stage_best: dict | None
    warmup_groups: list[dict] | None
    es_best: float
    es_counter: int
    global_step: int
    torch_rng_state: Any
    numpy_rng_state: Any
    python_rng_state: Any
    cuda_rng_state: Any
    loader_rng_state: Any


def _epoch_state(model, optimizer, *, selection: float, stage: int, epoch: int,
                 metrics: dict) -> dict:
    """One epoch's training state (:func:`capture_training_state`) with the selection value,
    stage, epoch and metrics it was taken at: what a stage hands the next and what
    ``model_best.pt`` is written from."""
    return {**capture_training_state(model, optimizer), "selection": selection, "stage": stage,
            "epoch": epoch, METRICS_KEY: _checkpoint_metrics(metrics)}


def _save_checkpoint(
    path: Path, *, model, optimizer, scheduler, scaler, loader_generator, config: dict,
    stage_idx: int, stage_epoch: int, run: "TrainRun", best: dict | None,
    stage_best: dict | None, warmup_groups: list[dict] | None,
    es_best: float, es_counter: int, global_step: int, metrics: dict,
) -> None:
    """Write a resumable periodic checkpoint carrying the run's config, its current training
    state (:func:`capture_training_state`, the same representation a stage's best is held in)
    and the resume state (:class:`_ResumeState`)."""
    state = _ResumeState(
        scheduler_state_dict=scheduler.state_dict(),
        scaler_state_dict=scaler.state_dict() if scaler is not None else None,
        stage=stage_idx, stage_epoch=stage_epoch, epoch=run.current_epoch,
        best=best, stage_best=stage_best, warmup_groups=warmup_groups,
        es_best=es_best, es_counter=es_counter,
        global_step=global_step, **capture_rng_state(loader_generator),
    )
    write_checkpoint({
        **capture_training_state(model, optimizer), **vars(state), CONFIG_KEY: config,
        METRICS_KEY: _checkpoint_metrics(metrics),
    }, path)


# ====================================================================
# Scheduler builder
# ====================================================================

def _build_scheduler(optimizer, spec: SchedulerSpec, epochs: int):
    """The scheduler ``spec.type`` names (``schemas.SCHEDULER_TYPES``) over ``epochs`` epochs,
    at the settings ``spec`` carries; a onecycle schedule cycles the rate alone, the optimizer's
    stated momentum the one it trains at."""
    if isinstance(spec, CosineSchedule):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=spec.eta_min
        )
    elif isinstance(spec, PlateauSchedule):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=spec.factor, patience=spec.patience
        )
    elif isinstance(spec, OneCycleSchedule):
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=spec.max_lr, total_steps=epochs, cycle_momentum=False)
    return torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=spec.step_size, gamma=spec.gamma
    )


# ====================================================================
# Validation
# ====================================================================

@torch.no_grad()
def _validate(
    model: TCIPModel, val_loader: DataLoader, device: torch.device, task: str, *,
    dims: Mapping[str, int], conf_threshold: float | None, iou_threshold: float,
    score_weights: dict | None, trait: TraitEntry | None,
) -> dict:
    """``evaluation.evaluate`` of ``model`` over ``val_loader`` at the given thresholds and
    ``trait`` (the confirmed entry whose criterion governs a count trait's detection metrics),
    every key prefixed :data:`VAL_METRIC_PREFIX`."""
    metrics = evaluate(
        model, val_loader, device, task, dims=dims,
        conf_threshold=conf_threshold, iou_threshold=iou_threshold,
        score_weights=score_weights, trait=trait,
    )
    return {f"{VAL_METRIC_PREFIX}{k}": v for k, v in metrics.items()}


def config_trait(spec: TrainConfigSchema, project: Path) -> TraitEntry | None:
    """The confirmed entry of the trait a validated config's (``schemas.train_config``)
    ``evaluation`` block names, read from ``project`` (refusing as
    ``operationalization.latest_confirmed`` does), or ``None`` when the block names none."""
    name = spec.evaluation.trait
    if not name:
        return None
    from tcip_mcp.operationalization import latest_confirmed

    return latest_confirmed(name, project).entry


def resolve_selection_metric(
    task: str, trait: TraitEntry | None, requested: str | None, *, has_val_loader: bool = True,
) -> str:
    """Resolve the bare metric key (into ``val_metrics``, without the ``val_`` prefix) that drives
    both ``model_best.pt`` and early stopping.

    Default: ``"objective"`` for detection/instance_seg, else ``"loss"``. An explicit ``requested``
        is honored, except it is rejected when ``trait``, the trait's confirmed entry, is a
        center-match trait and ``requested`` names a metric that trait's own localization
        criterion demotes to comparability-only (``evaluation.CENTER_MATCH_COMPARABILITY_KEYS``);
        an unstated kind rejects nothing here.

    A resolved metric (default or explicit) with no declared ranking direction
    (``evaluation.HIGHER_IS_BETTER_BY_METRIC``) is rejected.

    ``has_val_loader``: every metric but ``"loss"`` needs a validation pass, so a run with no
        validation loader can only select on ``"loss"`` (the training loss); anything else,
        including the ``"objective"`` default, is rejected. Defaults to ``True`` for a caller that
        has not built a loader yet.
    """
    default = "objective" if task in DETECTION_TASKS else "loss"
    resolved = requested or default
    if resolved not in HIGHER_IS_BETTER_BY_METRIC:
        raise ValueError(
            f"evaluation.selection_metric={resolved!r} has no declared ranking direction "
            "(evaluation.HIGHER_IS_BETTER_BY_METRIC names no entry for it), so model_best.pt "
            f"and early stopping would have to guess which way it improves. Choose one of "
            f"{sorted(HIGHER_IS_BETTER_BY_METRIC)}."
        )
    if not has_val_loader and resolved != "loss":
        raise ValueError(
            f"evaluation.selection_metric={resolved!r} needs a validation loader to compute, "
            "and this run has none. Only 'loss' (the training loss) can be selected on without "
            "one; configure a validation split, or set evaluation.selection_metric='loss'."
        )
    if trait is not None:
        from tcip_mcp.pipelines.training.evaluation import CENTER_MATCH_COMPARABILITY_KEYS
        from tcip_mcp.traits import CENTER_MATCH

        if trait.localization == CENTER_MATCH and resolved in CENTER_MATCH_COMPARABILITY_KEYS:
            raise ValueError(
                f"evaluation.selection_metric={resolved!r} is a comparability-only metric for "
                f"trait {trait.name!r} (localization=center_match), it does not govern this "
                "trait's phenotype count. Select by 'objective', 'f1', 'precision', 'recall', or "
                "'loss', which resolve through the trait's own center-match criterion."
            )
    return resolved


def resolve_objective(spec: TrainConfigSchema, task: str, *, project: Path,
                      has_val_loader: bool) -> dict:
    """A run's objective: ``selection_metric``, :func:`resolve_selection_metric` over its
    ``task``, its validated config's (``schemas.train_config``) ``evaluation`` trait
    (:func:`config_trait` in ``project``) and ``selection_metric``, and ``higher_is_better``, that
    metric's declared direction."""
    metric = resolve_selection_metric(task, config_trait(spec, project),
                                      spec.evaluation.selection_metric,
                                      has_val_loader=has_val_loader)
    return {"selection_metric": metric, "higher_is_better": HIGHER_IS_BETTER_BY_METRIC[metric]}


def _selection_value(task: str, val_metrics: dict, avg_loss: float, metric: str) -> float:
    """Best-model/early-stopping driver: ``val_metrics[f'{VAL_METRIC_PREFIX}{metric}']``.

    Raises when the resolved selection metric is not among this epoch's validation metrics. The one
    exception is ``metric == "loss"`` with no validation pass at all (``val_metrics`` empty): the
    training loss ``avg_loss`` answers.

    A present value of ``None`` (a diverged metric ``evaluation.stored_number`` normalized to
    ``null``) comes back as ``nan``, which never improves.
    """
    key = f"{VAL_METRIC_PREFIX}{metric}"
    if key in val_metrics:
        value = val_metrics[key]
        return value if value is not None else float("nan")
    if metric == "loss" and not val_metrics:
        return avg_loss
    raise ValueError(
        f"selection metric {metric!r} (key {key!r}) is not among this epoch's validation "
        f"metrics for task {task!r}: {sorted(val_metrics)}."
    )


def _improves(candidate: float, incumbent: float, *, higher_is_better: bool) -> bool:
    """Whether ``candidate`` beats ``incumbent`` as a selection value, in the direction the run's
    selection metric improves in.
    """
    return candidate > incumbent if higher_is_better else candidate < incumbent


def apply_stage_freeze(
    model: TCIPModel, freeze_to: int, *, prev_trainable: int | None = None,
    enforce_monotonic: bool = True,
) -> int:
    """Apply a stage's progressive-unfreeze policy and return the resulting trainable-param count.

    ``freeze_to``: ``0`` (or a model with no ``freeze_backbone``) trains everything; ``<0`` freezes
        all backbone stages; ``>0`` freezes up to that stage, best-effort, a bespoke model need not
        expose ``freeze_backbone``. When ``enforce_monotonic`` and ``prev_trainable`` is given, an
        unfreeze that shrinks the trainable set raises (progressive unfreeze must only ever grow
        it).
    """
    if not freeze_to or not hasattr(model, "freeze_backbone"):
        for p in model.parameters():
            p.requires_grad = True
    elif freeze_to < 0:
        num_stages = getattr(getattr(model, "backbone", None), "num_stages", 4)
        model.freeze_backbone(num_stages)
    else:
        model.freeze_backbone(freeze_to)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if enforce_monotonic and prev_trainable is not None and trainable < prev_trainable:
        raise RuntimeError(
            f"Non-decreasing unfreeze violated: {trainable} < {prev_trainable} trainable params"
        )
    return trainable


def _validate_input_channels(config: dict, loader: DataLoader) -> None:
    """Fail loudly if the first batch's channel count is not the band count the model was built
    at, ``data.num_channels``.

    A built-in loader reads every source at that count; a bespoke ``dataset_source`` builder
    composes its own bands and can hand the model another, which this names before an opaque
    conv-shape error deep in the first forward pass.
    """
    expected = config["data"]["num_channels"]
    batch = next(iter(loader), None)
    if batch is None:
        return
    imgs = batch[0]
    sample = imgs[0] if isinstance(imgs, (list, tuple)) else imgs
    if not hasattr(sample, "dim") or sample.dim() < 3:
        return
    channels = int(sample.shape[-3])
    if channels != expected:
        raise ValueError(
            f"Input images have {channels} channels but this run's model is built at "
            f"data.num_channels={expected}. State data.num_channels={channels}, or hand it data "
            f"at {expected} bands."
        )


# ====================================================================
# Main train loop
# ====================================================================

def train(
    run: TrainRun,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    *,
    epoch_callback=None,
    batch_callback=None,
    resume_from: str = "",
) -> TrainRun:
    """Execute a task-agnostic training run.

    The model is built from the validated config's ``model_source``
    (``model_build.build_from_model_source``), for the task that
    model source names (``model_build.run_task``).
    ``epoch_callback(epoch:int, epoch_metrics:dict)`` is how each epoch's row reaches the run's
    metrics log and, under HPO, the pruner. It may raise to abort the run (e.g.
    ``optuna.TrialPruned``); its scalars are the epoch's TensorBoard scalars, so a caller wanting
    them passes a sink that writes both (``TrainContext.log_metrics``).
    ``batch_callback(step:int, epoch:int, metrics:dict)`` is how a per-batch row reaches the
    run's metrics log and its TensorBoard (``TrainContext.log_batch``).

    Every key below is read from ``run.spec``, the run's config validated once by its producer
    (``schemas.train_config``); ``run.config`` is otherwise an open dict, and a bespoke
    ``model_source``/``dataset_source``/``training_source`` may read its own additional keys:

    - ``device`` (str, default cuda-if-available else cpu)
    - ``seed`` (int | None), ``deterministic`` (bool, default False), RNG seeding before model
      build.
    - ``mixed_precision`` (bool, default True), AMP, only when ``device`` is cuda.
    - ``stages`` (list of ``{freeze_to, epochs, gradient_accumulation_steps}``), at least one.
    - ``optimizer`` (``schemas.OptimizerSpec``), the one source of learning rate, stated for the
      first stage's target effective batch.
    - ``scheduler`` (``schemas.SchedulerSpec``; ``type`` one of ``schemas.SCHEDULER_TYPES``).
    - ``lr_scaling`` (``{scale_power, max_lr}``, absent for none): each stage's learning rates
      scaled by ``(its target effective batch / the first stage's) ** scale_power``, capped at the
      optional ``max_lr``. The target is nominal (the loader's batch size times the stage's
      accumulation), not an epoch's shorter remainder window.
    - ``stage_warmup_epochs`` (int, default 0), ``enforce_monotonic_unfreeze`` (bool, default
      True).
    - ``gradient_accumulation_steps`` (int, default 1), and a per-stage override. The physical
      batch is ``train_loader.batch_size``; a loader with none is refused. An epoch's last window
      may hold fewer batches than the accumulation, and its loss is averaged over the batches it
      holds.
    - ``checkpoint_every_n_epochs`` (int), periodic resumable checkpoints.
    - ``log_every_n_batches`` (int): every that many training batches of the run, the batch's
      loss reaches ``batch_callback`` once; unstated, :data:`BATCH_ROWS_PER_EPOCH` times an
      epoch at the evenly spaced batch ends :func:`batch_row_ends` derives from the loader's
      length, or after every batch of a shorter epoch.
    - ``early_stopping`` (``schemas.EarlyStoppingSpec``, on unless ``enabled`` is false): with a
      validation loader, a stage whose selection value has not improved by ``min_delta`` for
      ``patience`` epochs ends, and the next stage starts. With no validation loader it is
      inert.
    - ``evaluation`` (``schemas.EvaluationSpec``), passed through to
      ``_validate``/``evaluate``.

    Best-model selection and early stopping read ``run.objective``, the metric and direction the
    run's launch resolved. Each stage after the first starts from its predecessor's best epoch
    whole (:func:`_epoch_state`: weights, buffers and optimizer state taken at one moment), with
    a freshly built optimizer at its own learning rates. A stage that ends with no selectable
    epoch fails the run, naming the stage. The run's best epoch is held until the run ends and
    written once as ``model_best.pt`` beside ``model_final.pt``, the last epoch's weights; a
    diverged run writes neither. ``run.best_metric`` is the held best epoch's selection value.
    """
    config = run.config
    run.status = "running"
    run.start_time = time.time()
    higher_is_better = run.objective["higher_is_better"]
    # The losing-side sentinel for this run's own direction: any real value beats it.
    losing_side = float("-inf") if higher_is_better else float("inf")
    best: dict | None = None      # the run's best epoch (_epoch_state)

    try:
        # Failable setup lives inside the try so an invalid/unwritable output_dir
        # marks the run "failed" instead of stranding it at "running" forever.
        spec = run.spec
        stages, optimizer_spec, scheduler_spec, ckpt_every = spec.default_trainer_regime()
        out_dir = Path(run.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        device = run_device(spec)

        # Seed before model build so pretrained=False init + shuffle are reproducible.
        seed = spec.seed
        if seed is not None:
            set_seed(seed, deterministic=spec.deterministic)

        task = run_task(config)
        dims = recorded_model_dims(config)
        model = build_from_model_source(spec.model_source, dims)
        model.to(device)
        _validate_input_channels(config, train_loader)

        use_amp = spec.mixed_precision and device.type == "cuda"
        scaler = torch.amp.GradScaler(device.type) if use_amp else None

        # Progressive-unfreezing fidelity setup.
        base_backbone_lr = optimizer_spec.backbone_lr
        base_head_lr = optimizer_spec.head_lr
        stage_warmup_epochs = spec.stage_warmup_epochs
        physical_batch = getattr(train_loader, "batch_size", None)
        if physical_batch is None:
            raise ValueError(
                "the training loader states no batch_size, so the run's effective batch is not "
                "known; build it with a batch_size (generic_trainer.run_loaders).")

        def accumulation(stage: StageSpec) -> int:
            if stage.gradient_accumulation_steps is None:
                return spec.gradient_accumulation_steps
            return stage.gradient_accumulation_steps

        first_eff_batch = physical_batch * accumulation(stages[0])
        es = spec.early_stopping
        # (patience, min_delta) when a stage can end on its plateau; inert with no val loader.
        plateau_rule = ((es.patience, es.min_delta) if es.enabled and val_loader is not None
                        else None)
        prev_trainable = None     # trainable param count of the previous stage
        trait = config_trait(spec, run.project)
        selection_metric = run.objective["selection_metric"]
        loader_generator = getattr(train_loader, "generator", None)

        global_step = 0
        handoff: dict | None = None   # the previous stage's best epoch, the next stage's start
        diverged = False
        # Consecutive full training passes with zero finite batch losses; reset at every stage
        # boundary and by any epoch with even one finite loss, uniform across loader shapes.
        diverged_epochs = 0

        # Resume from a periodic checkpoint (training state + scheduler + scaler + randomness).
        resume_stage = -1
        ckpt: _ResumeState | None = None
        current: dict | None = None
        if resume_from:
            # On CPU: the RNG byte tensors must stay there, and load_state_dict moves the rest.
            loaded = torch.load(resume_from, map_location="cpu", weights_only=False)
            names = [*TRAINING_STATE_KEYS, *(f.name for f in fields(_ResumeState))]
            missing = [name for name in names if name not in loaded]
            if missing:
                raise ValueError(
                    f"Cannot resume from {resume_from}: checkpoint is missing {missing}. Resume "
                    "from a periodic checkpoint_epoch_*.pt, or start a fresh run.")
            # The contract's one read: every restore below reads these, never loaded.
            current = {name: loaded[name] for name in TRAINING_STATE_KEYS}
            ckpt = _ResumeState(**{f.name: loaded[f.name] for f in fields(_ResumeState)})
            del loaded
            resume_stage = ckpt.stage
            run.current_epoch = ckpt.epoch
            best = ckpt.best
            global_step = ckpt.global_step
            # After the fresh set_seed() above, which also configures cudnn, so the resumed
            # streams overwrite the freshly-seeded ones.
            restore_rng_state(vars(ckpt), loader_generator)
            logger.info("Resuming from %s at stage %d, stage_epoch %d (global epoch %d)",
                        resume_from, resume_stage, ckpt.stage_epoch, run.current_epoch)

        for stage_idx, stage in enumerate(stages):
            if diverged:
                break
            # Skip stages already completed before the resume checkpoint.
            if stage_idx < resume_stage:
                continue
            run.current_stage = stage_idx
            diverged_epochs = 0

            # Progressive unfreezing (+ monotonic guard), the shared craft primitive a custom
            # train(ctx) reuses via ctx.apply_stage_freeze.
            trainable = apply_stage_freeze(
                model, stage.freeze_to, prev_trainable=prev_trainable,
                enforce_monotonic=spec.enforce_monotonic_unfreeze,
            )
            prev_trainable = trainable

            # Per-stage accumulation + optional effective-batch LR scaling.
            stage_accum = accumulation(stage)
            target_eff_batch = physical_batch * stage_accum
            stage_backbone_lr, stage_head_lr = base_backbone_lr, base_head_lr
            if spec.lr_scaling is not None:
                mult = compute_lr_scale(target_eff_batch, first_eff_batch,
                                        spec.lr_scaling.scale_power)
                stage_backbone_lr *= mult
                stage_head_lr *= mult
                max_lr = spec.lr_scaling.max_lr
                if max_lr is not None:
                    stage_backbone_lr = min(stage_backbone_lr, max_lr)
                    stage_head_lr = min(stage_head_lr, max_lr)

            optimizer = build_optimizer(optimizer_spec, model, backbone_lr=stage_backbone_lr,
                                        head_lr=stage_head_lr)

            target_lrs = [g["lr"] for g in optimizer.param_groups]
            stage_best: dict | None = None
            warmup_groups: list[dict] | None = None
            es_best, es_counter = losing_side, 0
            if ckpt is not None and stage_idx == resume_stage:
                stage_best, warmup_groups = ckpt.stage_best, ckpt.warmup_groups
                es_best, es_counter = ckpt.es_best, ckpt.es_counter
            elif handoff is not None:
                restore_training_state(model, optimizer, handoff, group_settings=False)
                if stage_warmup_epochs:
                    warmup_groups = handoff[GROUPS_KEY]
                logger.info("Stage %d starts from stage %d's best epoch %d", stage_idx,
                            handoff["stage"], handoff["epoch"])
            warmup_starts = (None if warmup_groups is None
                             else _warmup_starts(model, optimizer, warmup_groups, target_lrs))

            stage_epochs = stage.epochs
            # Inter-stage LR warmup from the handed-off learning rates (default off).
            warmup_n = min(stage_warmup_epochs, stage_epochs) if warmup_starts is not None else 0
            sched_epochs = max(1, stage_epochs - warmup_n)
            scheduler = _build_scheduler(optimizer, scheduler_spec, sched_epochs)
            is_plateau = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

            start_epoch = 0
            if ckpt is not None and current is not None and stage_idx == resume_stage:
                start_epoch = ckpt.stage_epoch
                scheduler.load_state_dict(ckpt.scheduler_state_dict)
                # After the scheduler's construction, which writes its own settings into every
                # group: the captured groups carry the scheduled values the run had reached.
                restore_training_state(model, optimizer, current, group_settings=True)
                if scaler is not None:
                    scaler.load_state_dict(ckpt.scaler_state_dict)
                # Restored: past the best epochs it holds, the run keeps no reference to the
                # checkpoint.
                ckpt = current = None

            for epoch in range(start_epoch, stage_epochs):
                if run.should_cancel() or (plateau_rule and es_counter >= plateau_rule[0]):
                    break
                run.current_epoch += 1
                model.train()
                epoch_loss = 0.0
                n_batches = 0
                optimizer_steps = 0
                epoch_had_finite_loss = False
                optimizer.zero_grad()

                # Per-group linear LR warmup at the stage boundary.
                in_warmup = warmup_starts is not None and epoch < warmup_n
                if warmup_starts is not None and in_warmup:
                    alpha = (epoch + 1) / warmup_n
                    for group, start, target in zip(optimizer.param_groups, warmup_starts,
                                                    target_lrs):
                        group["lr"] = start + alpha * (target - start)

                n_loader = len(train_loader)
                # Batches past this index fall in the epoch's last, shorter window.
                full_windows_end = n_loader - n_loader % stage_accum
                row_ends = batch_row_ends(n_loader)
                for batch_idx, batch in enumerate(train_loader):
                    if run.should_cancel():
                        break
                    if task in DETECTION_TASKS:
                        images, targets = batch
                        images = [img.to(device) for img in images]
                        targets = instance_targets([
                            {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in t.items()}
                            for t in targets
                        ])
                    else:
                        images, targets = batch
                        images = images.to(device)
                        targets = {
                            k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in targets.items()
                        }

                    window = (stage_accum if batch_idx < full_windows_end
                              else n_loader - full_windows_end)
                    loss: Any
                    with torch.amp.autocast(device.type, enabled=use_amp):
                        loss_dict = model(images, targets)
                        loss = sum(loss_dict.values()) if isinstance(loss_dict, dict) else loss_dict
                    scaled = loss / window
                    (scaled if scaler is None else scaler.scale(scaled)).backward()
                    if (batch_idx + 1) % stage_accum == 0 or (batch_idx + 1) == n_loader:
                        if scaler is None:
                            optimizer.step()
                        else:
                            scaler.step(optimizer)
                            scaler.update()
                        optimizer.zero_grad()
                        optimizer_steps += 1

                    loss_value = loss.item()
                    epoch_loss += loss_value
                    n_batches += 1
                    global_step += 1

                    row_due = (global_step % spec.log_every_n_batches == 0
                               if spec.log_every_n_batches else batch_idx + 1 in row_ends)
                    if batch_callback is not None and row_due:
                        batch_callback(global_step, run.current_epoch, {"batch_loss": loss_value})

                    if math.isfinite(loss_value):
                        epoch_had_finite_loss = True

                diverged_epochs = (
                    diverged_epochs + 1 if n_batches and not epoch_had_finite_loss else 0
                )
                if diverged_epochs >= 2:
                    run.status = "failed"
                    run.status_error = (
                        "Training loss was non-finite for 2 consecutive full training passes "
                        "(no batch in either pass produced a finite loss); the run stopped as "
                        "diverged."
                    )
                    diverged = True
                    break  # skip this epoch's epoch-end block; stop before validation/checkpoints

                avg_loss = epoch_loss / max(n_batches, 1)
                current_lr = optimizer.param_groups[0]["lr"]

                val_metrics = {}
                if val_loader is not None:
                    val_metrics = _validate(
                        model, val_loader, device, task, dims=dims,
                        conf_threshold=spec.evaluation.conf_threshold,
                        iou_threshold=spec.evaluation.iou_threshold,
                        score_weights=spec.evaluation.score_weights,
                        trait=trait,
                    )
                sel = _selection_value(task, val_metrics, avg_loss, selection_metric)

                # Suppress the scheduler during warmup epochs.
                if not in_warmup:
                    if is_plateau:
                        scheduler.step(val_metrics.get(VAL_LOSS_KEY, avg_loss))
                    else:
                        scheduler.step()

                epoch_metrics = {
                    "stage": stage_idx,
                    "train_loss": round(avg_loss, 6),
                    "lr": current_lr,
                    "target_eff_batch": target_eff_batch,
                    "optimizer_steps": optimizer_steps,
                    "trainable_params": trainable,
                    "selection": round(sel, 6),
                    "selection_metric": selection_metric,
                    "selection_trait": trait.name if trait is not None else None,
                    **val_metrics,
                }
                run.metrics_history.append(epoch_metrics)

                if epoch_callback is not None:
                    epoch_callback(run.current_epoch, epoch_metrics)

                extra_val_metrics = " ".join(
                    f"{k}={v:.4f}" for k, v in val_metrics.items()
                    if k != VAL_LOSS_KEY and scalar_number(v))
                # A task whose evaluation reports no loss at all carries val_loss=None, which a
                # float format raises on: say what it is rather than print a 0 nobody measured.
                val_loss = val_metrics.get(VAL_LOSS_KEY)
                logger.info("Epoch %d stage %d loss=%.4f val_loss=%s lr=%.2e%s",
                    run.current_epoch, stage_idx, avg_loss,
                    f"{val_loss:.4f}" if scalar_number(val_loss) else val_loss, current_lr,
                    f" {extra_val_metrics}" if extra_val_metrics else "")

                # The run's best epoch is always also its stage's best, so one capture serves both.
                stage_incumbent = stage_best["selection"] if stage_best else losing_side
                if _improves(sel, stage_incumbent, higher_is_better=higher_is_better):
                    stage_best = _epoch_state(model, optimizer, selection=sel, stage=stage_idx,
                                              epoch=run.current_epoch, metrics=epoch_metrics)
                    run_incumbent = best["selection"] if best else losing_side
                    if _improves(sel, run_incumbent, higher_is_better=higher_is_better):
                        best = stage_best

                # The stage's plateau, on the same selection objective; the margin applies on the
                # same side of es_best that higher_is_better says an improvement lands on.
                if plateau_rule:
                    patience, min_delta = plateau_rule
                    margin = min_delta if higher_is_better else -min_delta
                    if _improves(sel, es_best + margin, higher_is_better=higher_is_better):
                        es_best, es_counter = sel, 0
                    else:
                        es_counter += 1
                    if es_counter >= patience:
                        logger.info("Stage %d plateaued at epoch %d", stage_idx, run.current_epoch)

                if ckpt_every > 0 and run.current_epoch % ckpt_every == 0:
                    _save_checkpoint(
                        checkpoint_path(out_dir, f"checkpoint_epoch_{run.current_epoch}"),
                        model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                        loader_generator=loader_generator, config=config, stage_idx=stage_idx,
                        stage_epoch=epoch + 1, run=run, best=best, stage_best=stage_best,
                        warmup_groups=warmup_groups, es_best=es_best, es_counter=es_counter,
                        global_step=global_step, metrics=epoch_metrics,
                    )

            if diverged or run.should_cancel():
                break  # stop before starting the next stage
            if stage_best is None:
                raise RuntimeError(
                    f"stage {stage_idx} produced no selectable epoch: its {selection_metric!r} "
                    "never took a value that ranks, so there is no state to start the next stage "
                    "from or to deliver.")
            handoff = stage_best

        # Saved on a normal completion or a cancellation; skipped for a diverged run (its
        # status_error is the record) and for a raised exception (caught below).
        if not diverged:
            if best is not None:
                run.saved["model_best"] = write_checkpoint(
                    {**{key: best[key] for key in (STATE_DICT_KEY, METRICS_KEY, "stage", "epoch")},
                     CONFIG_KEY: config}, checkpoint_path(out_dir, "model_best"))
            last_epoch_metrics = run.metrics_history[-1] if run.metrics_history else {}
            run.saved["model_final"] = write_checkpoint({
                STATE_DICT_KEY: model.state_dict(),
                CONFIG_KEY: config,
                METRICS_KEY: _checkpoint_metrics(last_epoch_metrics),
            }, checkpoint_path(out_dir, "model_final"))

        if diverged:
            logger.info("Training run %s stopped: %s", run.id, run.status_error)
        elif run.should_cancel():
            run.status = "canceled"
            logger.info("Training run %s canceled at epoch %d", run.id, run.current_epoch)
        else:
            run.status = "completed"

    except Exception as e:
        # Let HPO pruning signals propagate to Optuna (duck-typed to avoid the dep).
        if type(e).__name__ == "TrialPruned":
            raise
        run.status = "failed"
        run.status_error = str(e)
        logger.exception("Training failed: %s", e)

    finally:
        run.end_time = time.time()
        run.best_metric = best["selection"] if best is not None else losing_side

    return run
