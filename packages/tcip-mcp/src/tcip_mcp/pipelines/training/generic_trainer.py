"""Task-agnostic training loop for a bespoke ``model_source`` model.

This trainer works with any task type (detection, classification, ordinal, regression,
segmentation) because it delegates everything to the model's forward() which returns a loss dict in
train mode.

Provides: TensorBoard, JSONL metrics, progressive unfreezing, early stopping, mixed precision,
    gradient accumulation, checkpoints.
"""

from __future__ import annotations

import functools
import logging
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from tcip_store import Key, StoreDescriptor, register_store, store, stored_numbers
from tcip_store.file_backend import RootedFileLocator

from tcip_mcp.pipelines.data.datasets import indexed_sample_keys, instance_targets
from tcip_mcp.pipelines.model_contract import TCIPModel
from tcip_mcp.pipelines.model_build import (
    MODEL_SOURCE_KEY,
    STATE_DICT_KEY,
    build_model,
    run_in_chans,
    stamp_model_ref,
)
from tcip_mcp.pipelines.resolution import DEFAULT_CONF
from tcip_mcp.pipelines.schemas import DEFAULT_BATCH_SIZE
from tcip_mcp.pipelines.training.evaluation import (
    HIGHER_IS_BETTER_BY_METRIC,
    VAL_METRIC_PREFIX,
    evaluate,
)
from tcip_mcp.pipelines.training.optimizer_factory import (
    DEFAULT_BACKBONE_LR,
    DEFAULT_HEAD_LR,
    DEFAULT_WEIGHT_DECAY,
    build_optimizer,
    compute_lr_scale,
    restore_optimizer_state,
    snapshot_optimizer_state,
)
from tcip_mcp.pipelines.training.run_registry import TrainRun

logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[misc,assignment]


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


def capture_rng_state() -> dict[str, Any]:
    """Snapshot the four generator streams set_seed seeds, restorable with restore_rng_state."""
    return {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore generator state captured by capture_rng_state."""
    torch.set_rng_state(state["torch_rng_state"])
    if state["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    random.setstate(state["python_rng_state"])


def loader_worker_init(worker_id: int, seed: int | None = None,
                       num_workers: int | None = None) -> None:
    """DataLoader worker-process initializer, module-level so it pickles under Windows spawn
    (a closure cannot, and spawn pickles ``worker_init_fn`` to every worker).

    Always configures the platform GDAL cache budget: a spawned worker starts a fresh process
    on GDAL's stock default, not the budget the parent configured. Scaled by ``num_workers``
    (each worker gets ``1 / num_workers`` of the budget) so the fleet of workers together
    commits what the platform intended, not that amount each; ``None`` gives every worker the
    whole budget, unscaled. Per-worker numpy/random seeding applies only when the run is seeded.
    """
    from tcip_mcp.pipelines.raster_source import configure_gdal_cache

    share = 1.0 / num_workers if num_workers else 1.0
    configure_gdal_cache(share=share)
    if seed is not None:
        worker_seed = (seed + worker_id) % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)


def seeded_loader_kwargs(seed: int | None, num_workers: int | None = None) -> dict:
    """DataLoader kwargs wiring :func:`loader_worker_init` as the one ``worker_init_fn`` (every
    run, seeded or not, so spawned workers get the platform GDAL cache budget, scaled by
    ``num_workers`` when given) plus a seeded ``generator`` when ``seed`` is set, the same
    value ``set_seed`` uses, making shuffling and worker randomness reproducible. An unseeded
    run gets no generator: it stays unseeded end to end rather than silently becoming
    reproducible only in its loader.
    """
    kwargs: dict = {
        "worker_init_fn": functools.partial(
            loader_worker_init, seed=None if seed is None else int(seed), num_workers=num_workers),
    }
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        kwargs["generator"] = generator
    return kwargs


def run_transforms(config: dict) -> Any:
    """The training transforms ``config["augmentation"]`` declares, or ``None`` for none."""
    if not config.get("augmentation"):
        return None
    from tcip_mcp.pipelines.data.augmentations import build_augmentation

    return build_augmentation(config["augmentation"])


def run_loaders(config: dict, task: str, train_ds: Any, val_ds: Any, seed: int | None
                ) -> tuple[DataLoader, DataLoader | None]:
    """A run's train loader and, when ``val_ds`` is given, its val loader, at the config's
    ``batch_size`` (:data:`~tcip_mcp.pipelines.schemas.DEFAULT_BATCH_SIZE` when unstated),
    ``num_workers`` and ``sampler``, seeded from ``seed``."""
    from tcip_mcp.pipelines.data.samplers import build_sampler
    from tcip_mcp.pipelines.training.collation import task_collate

    batch_size = config.get("batch_size", DEFAULT_BATCH_SIZE)
    num_workers = config.get("num_workers", 0)
    loader_kwargs = seeded_loader_kwargs(seed, num_workers=num_workers)
    # Built after the loader context is known: read order depends on the worker regime too.
    sampler = build_sampler(config.get("sampler", "random"), train_ds,
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
    resolve = getattr(train_ds, "_resolve_path", None)
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


RUN_CHECKPOINT_STORE = "run_checkpoint"
register_store(
    StoreDescriptor(
        name=RUN_CHECKPOINT_STORE,
        kind="blob",
        key_fields=("name",),
        frozen=True,
        locator=RootedFileLocator(suffix=".pt"),
        path_readable=True,
    )
)
"""A run's checkpoints, keyed by the run's own output directory and the checkpoint's name.

Path-readable because a checkpoint is handed on as a path, not as bytes: ``torch.load``, the
model registry's sha256 of the real file, and the GUI's inference tab all take one.
"""


def checkpoint_key(output_dir: Path | str, name: str) -> Key:
    """One checkpoint of the run that writes into ``output_dir``.

    ``name`` is the checkpoint's own name without its extension: ``model_best``,
    ``model_final``, an epoch checkpoint, or a bespoke loop's own tag.
    """
    return Key(RUN_CHECKPOINT_STORE, str(Path(output_dir).resolve()), (name,))


def write_checkpoint(payload: dict, key: Key) -> Path:
    """Write one checkpoint's bytes and return where they landed.

    A crash or an OOM mid-save cannot destroy the previous checkpoint, and a concurrent reader
    (the GUI's inference tab) never observes a half-written file: the stream becomes the
    checkpoint only on a clean exit.
    """
    with store.write_blob(key) as handle:
        torch.save(payload, handle)
    return store.blob_path(key)


def _checkpoint_metrics(metrics: dict) -> dict:
    """One epoch's metrics normalized the way every destination stores them
    (:func:`stored_numbers`): a diverged run's ``nan`` reads back as ``null`` plus a state
    companion.
    """
    return stored_numbers(metrics)


_RESUME_KEYS = (
    STATE_DICT_KEY, "optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict",
    "stage", "stage_epoch", "epoch", "best_metric", "es_best", "es_counter", "global_step",
    "torch_rng_state", "numpy_rng_state", "python_rng_state", "cuda_rng_state",
)
"""The resume contract: every key :func:`_save_checkpoint` writes and a resume reads."""


def _save_checkpoint(
    key: Key, *, model, optimizer, scheduler, scaler, config: dict,
    stage_idx: int, stage_epoch: int, run: "TrainRun",
    es_best: float, es_counter: int, global_step: int, seed, metrics: dict,
) -> None:
    """Write a resumable periodic checkpoint carrying ``model_source``, the weights and the run's
    width.
    """
    state = {
        STATE_DICT_KEY: model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "stage": stage_idx,
        "stage_epoch": stage_epoch,
        "epoch": run.current_epoch,
        "best_metric": run.best_metric,
        "es_best": es_best,
        "es_counter": es_counter,
        "global_step": global_step,
        **capture_rng_state(),
    }
    write_checkpoint(stamp_model_ref({
        **{k: state[k] for k in _RESUME_KEYS}, "config": config, "seed": seed,
        "metrics": _checkpoint_metrics(metrics),
    }, config), key)


# ====================================================================
# Scheduler builder
# ====================================================================

def _build_scheduler(optimizer, config: dict, epochs: int):
    name = config.get("type", "cosine")
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=config.get("eta_min", 0))
    elif name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=config.get("factor", 0.5), patience=config.get("patience", 3))
    elif name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.get("max_lr", optimizer.defaults["lr"]), total_steps=epochs)
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.get("step_size", 10), gamma=config.get("gamma", 0.1))
    else:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


# ====================================================================
# Validation
# ====================================================================

@torch.no_grad()
def _validate(
    model: TCIPModel, val_loader: DataLoader, device: torch.device, task: str, *,
    conf_threshold: float = DEFAULT_CONF, iou_threshold: float = 0.5,
    iou_type: str | None = None, max_dets: int = 100, score_weights: dict | None = None,
    trait: str | None = None,
) -> dict:
    """Task-aware validation, delegates to ``evaluation.evaluate`` and ``val_``-prefixes.

    detection/instance_seg → precision/recall/F1/mAP50/mAP + composite objective;
    classification → accuracy/F1; ordinal → MAE/rank_acc; regression → MAE/RMSE;
    semantic_seg → mIoU/dice/pixel_acc/per-class IoU (``evaluation.semantic_seg_metrics``). Always
    returns ``val_loss``. Only detection/instance_seg's composite objective (or an explicit
    ``evaluation.selection_metric``) drives ``model_best.pt``/early stopping, every other
    task, including semantic_seg, selects by ``val_loss`` (see ``resolve_selection_metric``).

    ``trait``: when set, a count trait's derived localization criterion governs the reported
    detection count/F1 instead of the IoU@0.5 comparability convention (see ``evaluate``).
    """
    metrics = evaluate(
        model, val_loader, device, task,
        conf_threshold=conf_threshold, iou_threshold=iou_threshold,
        iou_type=iou_type, max_dets=max_dets, score_weights=score_weights,
        trait=trait,
    )
    return {f"{VAL_METRIC_PREFIX}{k}": v for k, v in metrics.items()}


def resolve_selection_metric(
    task: str, trait: str | None, requested: str | None, *, has_val_loader: bool = True,
) -> str:
    """Resolve the bare metric key (into ``val_metrics``, without the ``val_`` prefix) that drives
    both ``model_best.pt`` and early stopping.

    Default: ``"objective"`` for detection/instance_seg, else ``"loss"``. An explicit ``requested``
        is honored, except it is rejected when ``trait`` is a center-match trait and ``requested``
        names a metric that trait's own localization criterion demotes to comparability-only
        (``evaluation.CENTER_MATCH_COMPARABILITY_KEYS``).

    Reads the trait's recorded localization kind (``TraitSpec.localization``); an unrecorded kind
    (``spec.localization == ""``) rejects nothing here.

    A resolved metric (default or explicit) with no declared ranking direction
    (``evaluation.HIGHER_IS_BETTER_BY_METRIC``) is rejected.

    ``has_val_loader``: every metric but ``"loss"`` needs a validation pass, so a run with no
        validation loader can only select on ``"loss"`` (the training loss); anything else,
        including the ``"objective"`` default, is rejected. Defaults to ``True`` for a caller that
        has not built a loader yet.
    """
    default = "objective" if task in ("detection", "instance_seg") else "loss"
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
    if trait:
        from tcip_mcp.pipelines.training.evaluation import CENTER_MATCH_COMPARABILITY_KEYS
        from tcip_mcp.traits import CENTER_MATCH, get_trait

        spec = get_trait(trait)
        if spec.localization == CENTER_MATCH and resolved in CENTER_MATCH_COMPARABILITY_KEYS:
            raise ValueError(
                f"evaluation.selection_metric={resolved!r} is a comparability-only metric for "
                f"trait {trait!r} (localization=center_match), it does not govern this trait's "
                "phenotype count. Select by 'objective', 'f1', 'precision', 'recall', or 'loss', "
                "which resolve through the trait's own center-match criterion."
            )
    return resolved


def config_selection_metric(config: dict, *, has_val_loader: bool = True) -> str:
    """:func:`resolve_selection_metric` for a run config: its task
    (:func:`~tcip_mcp.pipelines.model_build.run_task`) and its ``evaluation`` block's ``trait``
    and ``selection_metric``."""
    from tcip_mcp.pipelines.model_build import run_task

    eval_cfg = config.get("evaluation") or {}
    return resolve_selection_metric(run_task(config), eval_cfg.get("trait"),
                                    eval_cfg.get("selection_metric"),
                                    has_val_loader=has_val_loader)


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


def _is_scalar_metric(value: Any) -> bool:
    """Whether ``value`` is a real number TensorBoard's ``add_scalar`` (and the console line's
    own summary) can log. A center-match trait's ``val_metrics`` carries non-scalar entries
    (``governing_criterion``: dict, ``map50_role``: str) that both must skip alike."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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
    """Fail loudly if the data's channel count doesn't match the width this run reads at.

    Catches an N-channel/RGB mismatch up front with a clear message instead of an opaque
    conv-shape error deep in the first forward pass. The width is the run's own
    (:func:`run_in_chans`), and a run that states none anywhere states nothing here to disagree
    with, which is a dataset the platform did not build.
    """
    expected = run_in_chans(config.get(MODEL_SOURCE_KEY), config.get("data"))
    if expected is None:
        return
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
            f"Input images have {channels} channels but this run reads at in_chans={expected}. "
            f"Set model_source.in_chans={channels}, or hand it data at {expected} bands."
        )


# ====================================================================
# Main train loop
# ====================================================================

def train(
    run: TrainRun,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    *,
    task: str,
    epoch_callback=None,
    resume_from: str = "",
) -> TrainRun:
    """Execute a task-agnostic training run.

    The model is built from run.config["model_source"] via build_model().
    ``epoch_callback(epoch:int, epoch_metrics:dict)`` is how each epoch's row reaches the run's
    metrics log and, under HPO, the pruner. It may raise to abort the run (e.g.
    ``optuna.TrialPruned``).

    Every ``run.config`` key this function reads; ``run.config`` is an open dict
    (``TrainConfigSchema`` keeps ``extra="allow"``), and a bespoke
    ``model_source``/``dataset_source``/``training_source`` may read its own additional keys:

    - ``device`` (str, default cuda-if-available else cpu)
    - ``batch_size`` (int), only as a fallback when ``train_loader`` itself has no ``.batch_size``.
    - ``seed`` (int | None), ``deterministic`` (bool, default False), RNG seeding before model
      build.
    - ``mixed_precision`` (bool, default True), AMP, only when ``device`` is cuda.
    - ``stages`` (list of ``{freeze_to, epochs}``), default a single 10-epoch full-unfreeze stage.
    - ``optimizer`` (``{name, backbone_lr, head_lr, weight_decay}``, default adamw/1e-4/1e-3/1e-4),
      the one source of learning rate, applied uniformly across every stage.
    - ``scheduler`` (``{type, ...}``; ``type`` in cosine/plateau/onecycle/step, default cosine).
    - ``lr_scaling`` (``{enabled, reference_effective_batch, scale_power, max_lr}``, default
      disabled), effective-batch LR scaling at stage boundaries.
    - ``stage_warmup_epochs`` (int, default 0), ``enforce_monotonic_unfreeze`` (bool, default
      True).
    - ``gradient_accumulation_steps`` (int, default 1), and a per-stage
      ``gradient_accumulation_steps`` override.
    - ``checkpoint_every_n_epochs`` (int, default 5), periodic resumable checkpoints.
    - ``early_stopping`` (``{enabled, patience, min_delta}``, default enabled-if-val_loader,
      patience 7, min_delta 1e-4).
    - ``evaluation`` (``{trait, selection_metric, conf_threshold, iou_threshold, iou_type,
      max_dets, score_weights}``, all optional). ``trait`` and ``selection_metric`` drive
      ``resolve_selection_metric``; the rest pass through to ``_validate``/``evaluate``.

    Every key sits at the top level of the config.
    """
    config = run.config
    run.status = "running"
    run.start_time = time.time()

    tb_writer = None

    # Early stopping
    es = config.get("early_stopping", {})
    es_enabled = es.get("enabled", val_loader is not None)
    es_patience = es.get("patience", 7)
    es_min_delta = es.get("min_delta", 1e-4)
    es_counter = 0
    es_best = float("inf")

    accum_steps = config.get("gradient_accumulation_steps", 1)
    ckpt_every = config.get("checkpoint_every_n_epochs", 5)

    try:
        # Failable setup lives inside the try so an invalid/unwritable output_dir
        # marks the run "failed" instead of stranding it at "running" forever.
        out_dir = Path(run.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if SummaryWriter is not None:
            tb_writer = SummaryWriter(log_dir=str(out_dir / "tensorboard"))

        device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        # Seed before model build so pretrained=False init + shuffle are reproducible.
        seed = config.get("seed")
        if seed is not None:
            set_seed(int(seed), deterministic=config.get("deterministic", False))

        model = build_model(config)
        model.to(device)
        _validate_input_channels(config, train_loader)

        stages = config.get("stages", [{"freeze_to": 0, "epochs": 10}])
        use_amp = config.get("mixed_precision", True) and device.type == "cuda"
        scaler = torch.amp.GradScaler(device.type) if use_amp else None

        opt_cfg = config.get("optimizer", {})
        sched_cfg = config.get("scheduler", {"type": "cosine"})

        # Progressive-unfreezing fidelity setup.
        base_backbone_lr = opt_cfg.get("backbone_lr", DEFAULT_BACKBONE_LR)
        base_head_lr = opt_cfg.get("head_lr", DEFAULT_HEAD_LR)
        lr_scaling_cfg = config.get("lr_scaling", {})
        stage_warmup_epochs = int(config.get("stage_warmup_epochs", 0))
        enforce_monotonic_unfreeze = config.get("enforce_monotonic_unfreeze", True)
        physical_batch = (getattr(train_loader, "batch_size", None)
                          or config.get("batch_size", DEFAULT_BATCH_SIZE))
        pending_snapshot = None   # best optimizer state from the previous stage
        prev_trainable = None     # trainable param count of the previous stage
        eval_cfg = config.get("evaluation") or {}
        trait = eval_cfg.get("trait")
        selection_metric = config_selection_metric(config, has_val_loader=val_loader is not None)
        run.best_metric_name = selection_metric
        # The losing-side sentinel for this run's own direction: any real value beats it.
        higher_is_better = HIGHER_IS_BETTER_BY_METRIC[selection_metric]
        losing_side = float("-inf") if higher_is_better else float("inf")
        run.best_metric = losing_side
        es_best = losing_side

        global_step = 0
        stopped_early = False
        diverged = False
        # Consecutive full training passes with zero finite batch losses; reset at every stage
        # boundary and by any epoch with even one finite loss, uniform across loader shapes.
        diverged_epochs = 0

        # Resume from a periodic checkpoint (model + optimizer + scheduler + scaler).
        resume_stage = -1
        resume_stage_epoch = 0
        ckpt = None
        if resume_from:
            # On CPU: the RNG byte tensors must stay there, and load_state_dict moves the rest.
            loaded = torch.load(resume_from, map_location="cpu", weights_only=False)
            # The contract's one read: every restore below indexes this projection, never loaded.
            try:
                ckpt = {k: loaded[k] for k in _RESUME_KEYS}
            except KeyError:
                missing = [k for k in _RESUME_KEYS if k not in loaded]
                raise ValueError(
                    f"Cannot resume from {resume_from}: checkpoint is missing {missing}. Resume "
                    "from a periodic checkpoint_epoch_*.pt, or start a fresh run."
                ) from None
            model.load_state_dict(ckpt[STATE_DICT_KEY])
            resume_stage = ckpt["stage"]
            resume_stage_epoch = ckpt["stage_epoch"]
            run.current_epoch = ckpt["epoch"]
            run.best_metric = ckpt["best_metric"]
            es_best = ckpt["es_best"]
            es_counter = ckpt["es_counter"]
            global_step = ckpt["global_step"]
            # After the fresh set_seed() above, which also configures cudnn, so the resumed
            # streams overwrite the freshly-seeded ones.
            restore_rng_state(ckpt)
            logger.info("Resuming from %s at stage %d, stage_epoch %d (global epoch %d)",
                        resume_from, resume_stage, resume_stage_epoch, run.current_epoch)

        for stage_idx, stage in enumerate(stages):
            if stopped_early or diverged:
                break
            run.current_stage = stage_idx
            diverged_epochs = 0

            # Skip stages already completed before the resume checkpoint.
            if stage_idx < resume_stage:
                continue

            # Progressive unfreezing (+ monotonic guard), the shared craft primitive a custom
            # train(ctx) reuses via ctx.apply_stage_freeze.
            trainable = apply_stage_freeze(
                model, stage.get("freeze_to", 0), prev_trainable=prev_trainable,
                enforce_monotonic=enforce_monotonic_unfreeze,
            )
            prev_trainable = trainable

            # Per-stage accumulation + optional effective-batch LR scaling.
            stage_accum = stage.get("gradient_accumulation_steps", accum_steps)
            eff_batch = physical_batch * stage_accum
            stage_backbone_lr, stage_head_lr = base_backbone_lr, base_head_lr
            if lr_scaling_cfg.get("enabled", False):
                mult = compute_lr_scale(
                    eff_batch,
                    lr_scaling_cfg.get("reference_effective_batch", 64),
                    lr_scaling_cfg.get("scale_power", 0.5),
                )
                stage_backbone_lr *= mult
                stage_head_lr *= mult
                max_lr = lr_scaling_cfg.get("max_lr")
                if max_lr is not None:
                    stage_backbone_lr = min(stage_backbone_lr, max_lr)
                    stage_head_lr = min(stage_head_lr, max_lr)

            optimizer = build_optimizer(
                opt_cfg.get("name", "adamw"),
                model,
                backbone_lr=stage_backbone_lr,
                head_lr=stage_head_lr,
                weight_decay=opt_cfg.get("weight_decay", DEFAULT_WEIGHT_DECAY),
            )

            # Hand off momentum from the previous stage's best epoch.
            if pending_snapshot is not None:
                restored = restore_optimizer_state(optimizer, model, pending_snapshot)
                logger.info("Stage %d: restored optimizer state for %d params", stage_idx, restored)

            target_lrs = [g["lr"] for g in optimizer.param_groups]
            prev_end_lrs: list[float] | None = (
                pending_snapshot.get("end_lrs") if pending_snapshot else None)

            stage_epochs = stage.get("epochs", 10)
            # Inter-stage LR warmup (boundaries only; default off).
            warmup_n = (
                min(stage_warmup_epochs, stage_epochs)
                if (stage_idx > 0 and pending_snapshot is not None)
                else 0
            )
            sched_epochs = max(1, stage_epochs - warmup_n)
            scheduler = _build_scheduler(optimizer, sched_cfg, sched_epochs)
            is_plateau = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
            stage_best = losing_side
            stage_snapshot = None

            # For the resumed stage, restore optimizer/scheduler/scaler and start
            # mid-stage; later stages keep fresh state (and the momentum handoff above).
            start_epoch = 0
            if stage_idx == resume_stage and ckpt is not None:
                start_epoch = resume_stage_epoch
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if scheduler is not None:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                if scaler is not None:
                    scaler.load_state_dict(ckpt["scaler_state_dict"])
                ckpt = None

            for epoch in range(start_epoch, stage_epochs):
                if stopped_early or diverged or run.should_cancel():
                    break
                run.current_epoch += 1
                model.train()
                epoch_loss = 0.0
                n_batches = 0
                epoch_had_finite_loss = False
                optimizer.zero_grad()

                # Per-group linear LR warmup at the stage boundary.
                in_warmup = warmup_n > 0 and epoch < warmup_n
                if in_warmup:
                    alpha = (epoch + 1) / warmup_n
                    for gi, group in enumerate(optimizer.param_groups):
                        start = prev_end_lrs[gi] if (prev_end_lrs and gi < len(prev_end_lrs)) else 0.0
                        group["lr"] = start + alpha * (target_lrs[gi] - start)

                for batch_idx, batch in enumerate(train_loader):
                    if run.should_cancel():
                        break
                    if task in ("detection", "instance_seg"):
                        images, targets = batch
                        images = [img.to(device) for img in images]
                        targets = instance_targets([{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets])
                    else:
                        images, targets = batch
                        images = images.to(device)
                        targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}

                    loss: Any
                    if use_amp:
                        assert scaler is not None  # use_amp implies scaler was built above
                        with torch.amp.autocast(device.type):
                            loss_dict = model(images, targets)
                            loss = sum(loss_dict.values()) if isinstance(loss_dict, dict) else loss_dict
                        scaled = loss / stage_accum
                        scaler.scale(scaled).backward()
                        if (batch_idx + 1) % stage_accum == 0 or (batch_idx + 1) == len(train_loader):
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                    else:
                        loss_dict = model(images, targets)
                        loss = sum(loss_dict.values()) if isinstance(loss_dict, dict) else loss_dict
                        scaled = loss / stage_accum
                        scaled.backward()
                        if (batch_idx + 1) % stage_accum == 0 or (batch_idx + 1) == len(train_loader):
                            optimizer.step()
                            optimizer.zero_grad()

                    loss_value = loss.item()
                    epoch_loss += loss_value
                    n_batches += 1
                    global_step += 1

                    if tb_writer and global_step % 10 == 0:
                        tb_writer.add_scalar("train/loss_step", loss_value, global_step)

                    if math.isfinite(loss_value):
                        epoch_had_finite_loss = True

                diverged_epochs = diverged_epochs + 1 if n_batches and not epoch_had_finite_loss else 0
                if diverged_epochs >= 2:
                    run.status = "failed"
                    run.error = (
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
                        model, val_loader, device, task,
                        # A fixed default unless eval_cfg explicitly overrides it, not the
                        # resolved ship-point conf (that is derived later by resolve_operating_point).
                        conf_threshold=eval_cfg.get("conf_threshold", DEFAULT_CONF),
                        iou_threshold=eval_cfg.get("iou_threshold", 0.5),
                        iou_type=eval_cfg.get("iou_type"),
                        max_dets=eval_cfg.get("max_dets", 100),
                        score_weights=eval_cfg.get("score_weights"),
                        trait=trait,
                    )
                sel = _selection_value(task, val_metrics, avg_loss, selection_metric)

                # Suppress the scheduler during warmup epochs.
                if not in_warmup:
                    if is_plateau:
                        scheduler.step(val_metrics.get("val_loss", avg_loss))
                    else:
                        scheduler.step()

                epoch_metrics = {
                    "epoch": run.current_epoch,
                    "stage": stage_idx,
                    "train_loss": round(avg_loss, 6),
                    "lr": current_lr,
                    "eff_batch": eff_batch,
                    "trainable_params": trainable,
                    "selection": round(sel, 6),
                    "selection_metric": selection_metric,
                    "selection_trait": trait,
                    **val_metrics,
                }
                run.metrics_history.append(epoch_metrics)

                if tb_writer:
                    tb_writer.add_scalar("train/loss", avg_loss, run.current_epoch)
                    tb_writer.add_scalar("train/lr", current_lr, run.current_epoch)
                    for k, v in val_metrics.items():
                        if _is_scalar_metric(v):
                            tb_writer.add_scalar(f"val/{k}", v, run.current_epoch)
                    tb_writer.flush()

                if epoch_callback is not None:
                    epoch_callback(run.current_epoch, epoch_metrics)

                extra_val_metrics = " ".join(
                    f"{k}={v:.4f}" for k, v in val_metrics.items()
                    if k != "val_loss" and _is_scalar_metric(v))
                # A task whose evaluation reports no loss at all carries val_loss=None, which a
                # float format raises on: say what it is rather than print a 0 nobody measured.
                val_loss = val_metrics.get("val_loss")
                logger.info("Epoch %d stage %d loss=%.4f val_loss=%s lr=%.2e%s",
                    run.current_epoch, stage_idx, avg_loss,
                    f"{val_loss:.4f}" if _is_scalar_metric(val_loss) else val_loss, current_lr,
                    f" {extra_val_metrics}" if extra_val_metrics else "")

                # Best model checkpoint, selected by the selection objective.
                if _improves(sel, run.best_metric, higher_is_better=higher_is_better):
                    run.best_metric = sel
                    try:
                        write_checkpoint(stamp_model_ref({
                            STATE_DICT_KEY: model.state_dict(),
                            "config": config,
                            "metrics": _checkpoint_metrics(epoch_metrics),
                            "stage": stage_idx, "epoch": run.current_epoch,
                        }, config), checkpoint_key(out_dir, "model_best"))
                    except PermissionError:
                        # Windows: a concurrent reader can hold model_best.pt open past the
                        # replace retries; keep the previous best rather than failing the run.
                        logger.warning(
                            "model_best.pt held open by a reader; keeping previous best "
                            "(epoch %d not persisted).", run.current_epoch)

                # Remember this stage's best optimizer state for the handoff.
                if _improves(sel, stage_best, higher_is_better=higher_is_better):
                    stage_best = sel
                    stage_snapshot = snapshot_optimizer_state(optimizer, model)

                if ckpt_every > 0 and run.current_epoch % ckpt_every == 0:
                    _save_checkpoint(
                        checkpoint_key(out_dir, f"checkpoint_epoch_{run.current_epoch}"),
                        model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                        config=config, stage_idx=stage_idx, stage_epoch=epoch + 1, run=run,
                        es_best=es_best, es_counter=es_counter, global_step=global_step,
                        seed=seed, metrics=epoch_metrics,
                    )

                # Early stopping, on the same selection objective; the margin applies on the
                # same side of es_best that higher_is_better says an improvement lands on.
                if es_enabled and val_loader is not None:
                    margin = es_min_delta if higher_is_better else -es_min_delta
                    if _improves(sel, es_best + margin, higher_is_better=higher_is_better):
                        es_best = sel
                        es_counter = 0
                    else:
                        es_counter += 1
                        if es_counter >= es_patience:
                            logger.info("Early stopping at epoch %d", run.current_epoch)
                            stopped_early = True

            # Carry this stage's best optimizer state into the next stage.
            if stage_snapshot is not None:
                pending_snapshot = stage_snapshot

            if diverged or run.should_cancel():
                break  # stop before starting the next stage

        # Final checkpoint: saved on a normal completion or a cancellation; skipped for a
        # diverged run (run.error is the record) and for a raised exception (caught below).
        if not diverged:
            last_epoch_metrics = run.metrics_history[-1] if run.metrics_history else {}
            write_checkpoint(stamp_model_ref({
                STATE_DICT_KEY: model.state_dict(),
                "config": config,
                "metrics": _checkpoint_metrics(last_epoch_metrics),
            }, config), checkpoint_key(out_dir, "model_final"))

        if diverged:
            logger.info("Training run %s stopped: %s", run.id, run.error)
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
        run.error = str(e)
        logger.exception("Training failed: %s", e)

    finally:
        run.end_time = time.time()
        if tb_writer:
            tb_writer.close()

    return run
