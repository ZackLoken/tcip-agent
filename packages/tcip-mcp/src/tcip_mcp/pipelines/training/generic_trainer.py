"""Task-agnostic training loop for a bespoke ``model_source`` model.

This trainer works with *any* task type (detection, classification,
ordinal, regression, segmentation) because it delegates everything
to the model's forward() which returns a loss dict in train mode.

Preserves: TensorBoard, JSONL metrics, progressive unfreezing,
early stopping, mixed precision, gradient accumulation, checkpoints.
"""

from __future__ import annotations

import functools
import logging
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from tcip_store import Key, StoreDescriptor, register_store, store
from tcip_store.file_backend import RootedFileLocator

from tcip_mcp.pipelines.model_contract import TCIPModel
from tcip_mcp.pipelines.model_build import (
    MODEL_SOURCE_KEY,
    STATE_DICT_KEY,
    build_model,
    stamp_model_ref,
)
from tcip_mcp.pipelines.resolution import DEFAULT_CONF
from tcip_mcp.pipelines.training.evaluation import evaluate
from tcip_mcp.pipelines.training.optimizer_factory import (
    build_optimizer,
    compute_lr_scale,
    restore_optimizer_state,
    snapshot_optimizer_state,
)

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


def loader_worker_init(worker_id: int, seed: int | None = None,
                       num_workers: int | None = None) -> None:
    """DataLoader worker-process initializer, module-level so it pickles under Windows spawn
    (a closure cannot, and spawn pickles ``worker_init_fn`` to every worker).

    Always configures the platform GDAL cache budget: a spawned worker starts a fresh process
    on GDAL's stock default, not the budget the parent configured. Scaled by ``num_workers``
    (each worker gets ``1 / num_workers`` of the budget) so the fleet of workers together
    commits what the platform intended, not that amount each; ``None`` (a caller that has not
    been updated to pass it) keeps the pre-scaling full-budget behavior. Per-worker numpy/random
    seeding applies only when the run is seeded.
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


def stamp_effective_data_geometry(data_cfg: dict, train_ds: Any) -> dict:
    """Record the input geometry ``train_ds`` actually serves into ``data_cfg``, in place.

    ``data_cfg`` is the live ``config["data"]`` dict the run persists (checkpoints embed the
    run config by reference; an HPO trial's resolved-config snapshot spreads the merged
    config), so this must run after the dataset is built and before training starts.

    A tiled train dataset (one carrying a ``tile_size``) stamps its effective
    ``tile_size``/``overlap`` into the tiling record, filling in defaults the caller's config
    omitted. An untiled one replaces the tiling record with ``{"enabled": False}`` outright,
    never a merge: a requested-but-unrealized ``tile_size`` surviving in the persisted config
    would read back as the frame the model trained on (predictor tile geometry, contract-dim
    resolution, tiled-eval defaults).

    ``train_native_size``: an untiled run whose training frames all share one size stamps
    ``data_cfg["train_native_size"] = [width, height]``; mixed sizes stamp nothing, since
    there is no single native frame to record. Tiled runs stamp nothing here either: their
    native frame is the tile, and the stamped ``tile_size`` already carries it. Probing is
    header-only for the common containers (``image_dimensions``) and needs the dataset's
    source list; a dataset that exposes none cannot be probed, so nothing is stamped.

    Returns the stamped facts, ``{"tiling": dict, "tiling_replaced": bool,
    "train_native_size": [w, h] | None}``, so a caller can mirror the identical stamp into a
    durable experiment record instead of re-deriving it.
    """
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
    stems = getattr(train_ds, "stems", None)
    resolve = getattr(train_ds, "_resolve_path", None)
    if not stems or resolve is None:
        return None
    from tcip_mcp.pipelines.image_utils import image_dimensions

    channels = getattr(train_ds, "expected_channels", 3)
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


# ====================================================================
# TrainConfig
# ====================================================================

@dataclass
class TrainConfig:
    """Constructor-time config carrier for ``launch_training``'s data-loader setup.

    Only ``sampler``/``batch_size``/``num_workers`` are ever read off an instance, the
    trainer itself reads every other setting straight from ``run.config`` (see ``train()``'s own
    ``config.get(...)`` calls), so this dataclass does not double as a config record. Do not add a
    field here expecting ``train()`` to honor it; wire it into ``run.config`` instead.
    """
    model_source: dict
    dataset: dict           # {task, images_dir, labels_dir, ...}
    augmentation: dict = field(default_factory=dict)
    sampler: str = "random"
    batch_size: int = 4
    num_workers: int = 2


# ====================================================================
# TrainRun state
# ====================================================================

@dataclass
class TrainRun:
    run_id: str
    config: dict
    status: str = "created"
    current_epoch: int = 0
    current_stage: int = 0
    best_metric: float = float("inf")  # best selection objective (composite for detection/instance_seg, else val_loss; lower=better)
    metrics_history: list[dict] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    error: str = ""
    output_dir: str = ""
    # "training" (standalone/GUI/agent run) vs "hpo_trial" (a run created by an HPO
    # sweep's objective). The Training view lists only "training" so sweeps of dozens of
    # trials don't flood it, HPO trials belong to the Tuning view.
    origin: str = "training"
    # Set by cancel_run() to request a graceful stop; the train loop polls it.
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    # Set on resume, True if the checkpoint carried RNG state and it was restored, False
    # if the checkpoint predates RNG capture (fresh-seed stream stands). None on a non-resumed run.
    rng_state_restored: bool | None = None
    # Set by the parent right after spawning the subprocess this run's training body actually
    # executes in. None means the loop is running in-process (every existing test; cancel_event alone
    # is authoritative). Not None means should_cancel() must also check the cross-process sentinel
    # file, since the parent's own cancel_event lives in different process memory than the child.
    pid: int | None = None

    def should_cancel(self) -> bool:
        """True if cancellation was requested, in-process (``cancel_event``) or via the sentinel
        file a (possibly different) process may have written at ``<output_dir>/.cancel_requested``.
        Checked unconditionally, not gated on ``pid`` being set: the object checking this is
        typically the child's own attached ``TrainRun`` (which has no reason to know its own OS
        pid), while ``pid`` is meaningful on the parent's copy for a different purpose (deciding
        whether ``cancel_run`` should set the in-memory ``Event`` or write the sentinel). The single
        check every poll site and every ``train(ctx)`` loop must use, so a sentinel-triggered stop is
        never invisible to a check written against the in-memory ``Event`` alone."""
        if self.cancel_event.is_set():
            return True
        if self.output_dir:
            return (Path(self.output_dir) / ".cancel_requested").exists()
        return False

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "current_epoch": self.current_epoch,
            "current_stage": self.current_stage,
            "best_metric": self.best_metric,
            "metrics_history": self.metrics_history,
            "origin": self.origin,
            "elapsed_seconds": (self.end_time or time.time()) - self.start_time if self.start_time else 0,
            "pid": self.pid,
        }


_RUNS: dict[str, TrainRun] = {}
_RUNS_LOCK = threading.Lock()


def create_run(config: dict, output_dir: str, origin: str = "training") -> TrainRun:
    # uuid suffix (not len(_RUNS)): same-second launches, from this process, or from
    # the web backend's separate registry sharing the same experiments dir, must never
    # collide, or two runs would silently share one run_id and output directory.
    run_id = f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    # Reproducibility: never start an unseeded run. If the caller set no seed (same
    # top-level/training.seed read as train()), draw one from OS entropy and write it
    # into the config in place, launch_training snapshots this dict into the
    # experiment record *after* create_run, and every checkpoint embeds it, so the
    # effective seed is always on record and the run can be rerun exactly.
    if config.get("seed", config.get("training", {}).get("seed")) is None:
        config["seed"] = random.SystemRandom().randrange(2**31)
        logger.info("Run %s: no seed configured; drew seed=%d.", run_id, config["seed"])
    run = TrainRun(run_id=run_id, config=config, output_dir=output_dir, origin=origin)
    with _RUNS_LOCK:
        _RUNS[run_id] = run
    return run


def attach_run(run_id: str, config: dict, output_dir: str, origin: str = "training") -> TrainRun:
    """Construct a ``TrainRun`` for an id the caller already owns, unlike ``create_run``,
    which unconditionally mints a fresh random id, this never mints one. Used by the subprocess
    worker to adopt the exact ``run_id`` the parent already returned to its own caller and baked
    into ``output_dir``/``env.json``/audit events; ``create_run`` cannot do that.

    Does not draw a seed: the parent already called ``create_run`` (which does) before spawning,
    so ``config`` (read back from the persisted ``config.json``) already carries the resolved seed.
    Inserts into *this process's own* ``_RUNS``, safe even though the id may already be a key in a
    different process's registry, since that's different process memory entirely.
    """
    run = TrainRun(run_id=run_id, config=config, output_dir=output_dir, origin=origin)
    with _RUNS_LOCK:
        _RUNS[run_id] = run
    return run


def get_run(run_id: str) -> TrainRun | None:
    with _RUNS_LOCK:
        return _RUNS.get(run_id)


def list_runs(include_hpo_trials: bool = False) -> list[dict]:
    """Return runs as dicts. HPO trial runs (``origin='hpo_trial'``) are excluded by
    default so a sweep's trials don't leak into the Training view; pass
    ``include_hpo_trials=True`` for the full registry."""
    with _RUNS_LOCK:
        runs = list(_RUNS.values())
    return [
        r.to_dict()
        for r in runs
        if include_hpo_trials or r.origin != "hpo_trial"
    ]


def cancel_run(run_id: str) -> bool:
    """Request a graceful cancellation of a training run. Returns False if unknown.

    A run whose training body executes in a subprocess (``run.pid is not None``) can't be
    stopped by setting an in-memory ``Event``, that memory lives in a different process. Writes a
    sentinel file at ``<output_dir>/.cancel_requested`` instead, which ``TrainRun.should_cancel()``
    polls in the child. When this process has no local record of the run at all (it was launched by
    a *different* process, e.g. the web backend cancelling a run the agent's MCP server's
    subprocess is running), falls back to the run's own status record for the real output directory
    rather than guessing one; an unresolvable run is refused (``False``), never a silent write to a
    path nobody polls. The record is read through the store the launching process wrote it to, so a
    live run is still cancellable when that store is a database rather than a file beside the run.
    """
    with _RUNS_LOCK:
        run = _RUNS.get(run_id)
    if run is not None:
        if run.pid is None:
            run.cancel_event.set()
        else:
            Path(run.output_dir).mkdir(parents=True, exist_ok=True)
            (Path(run.output_dir) / ".cancel_requested").touch()
        return True

    from tcip_mcp.experiments import read_member, resolve_experiment_for_run, status_key

    experiment_id = resolve_experiment_for_run(run_id)
    if experiment_id is None:
        return False
    status = read_member(status_key(experiment_id), {})
    output_dir = status.get("output_dir") if isinstance(status, dict) else None
    if not output_dir:
        return False
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / ".cancel_requested").touch()
    return True


RUN_CHECKPOINT_STORE = "run_checkpoint"
register_store(
    StoreDescriptor(
        name=RUN_CHECKPOINT_STORE,
        kind="blob",
        key_fields=("name",),
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

    A crash or an OOM mid-save cannot destroy the previous checkpoint, and a concurrent
    reader (GUI inference tab, ``evaluate_model`` on a live run) never observes a
    half-written file: the stream becomes the checkpoint only on a clean exit.
    """
    with store.write_blob(key) as handle:
        torch.save(payload, handle)
    return store.blob_path(key)


def _save_checkpoint(
    key: Key, *, model, optimizer, scheduler, scaler, config: dict,
    stage_idx: int, stage_epoch: int, run: "TrainRun",
    es_best: float, es_counter: int, global_step: int, seed, metrics: dict,
) -> None:
    """Write a resumable periodic checkpoint.

    Superset of the previous payload, ``GenericPredictor`` reads only the model reference
    (``model_source``) + the weights and stays compatible.
    """
    write_checkpoint(stamp_model_ref({
        STATE_DICT_KEY: model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "config": config,
        "stage": stage_idx,
        "stage_epoch": stage_epoch,
        "epoch": run.current_epoch,
        "best_metric": run.best_metric,
        "es_best": es_best,
        "es_counter": es_counter,
        "global_step": global_step,
        "seed": seed,
        "metrics": metrics,
        # Full RNG state at save time, so a resume can pick the streams up exactly where
        # they were rather than silently re-seeding from stream position zero.
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, config), key)


# ====================================================================
# Collate functions
# ====================================================================

def _detection_collate(batch):
    """Detection/instance_seg: list of (img, target) → (list[img], list[target])."""
    images, targets = zip(*batch)
    return list(images), list(targets)


def _stack_collate(batch):
    """Classification/ordinal/regression/semantic_seg: stack into tensors."""
    images, targets = zip(*batch)
    images = torch.stack(images)
    # Merge target dicts, stack numeric values
    merged: dict[str, Any] = {}
    for key in targets[0]:
        vals = [t[key] for t in targets]
        if isinstance(vals[0], (int, float)):
            merged[key] = torch.tensor(vals)
        elif isinstance(vals[0], torch.Tensor):
            merged[key] = torch.stack(vals)
        else:
            merged[key] = vals
    return images, merged


def task_collate(task: str):
    """Return the right collate_fn for a task type."""
    if task in ("detection", "instance_seg"):
        return _detection_collate
    return _stack_collate


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
    return {f"val_{k}": v for k, v in metrics.items()}


def resolve_selection_metric(task: str, trait: str | None, requested: str | None) -> str:
    """Resolve the bare metric key (into ``val_metrics``, without the ``val_`` prefix) that drives
    both ``model_best.pt`` and early stopping, deliberately the same key for both.

    Default (unchanged from the prior hardcoded behavior): ``"objective"`` for detection/
    instance_seg, else ``"loss"``. An explicit ``requested`` is honored, except it is rejected when
    ``trait`` is a center-match trait and ``requested`` names a metric that trait's own
    localization criterion demotes to comparability-only (``evaluation.CENTER_MATCH_COMPARABILITY_KEYS``)
, selecting checkpoints by a metric the trait doesn't trust is a defensibility regression, not a
    legitimate choice.

    Reads the trait's recorded localization kind (``TraitSpec.localization`` is derived
    once from real GT and persisted, never authored). This runs at preflight time, before any GT is
    loaded, so it cannot itself derive a kind, an unrecorded kind (``spec.localization == ""``, a
    trait never yet calibrated/evaluated against real data) means nothing is known yet, so no
    metric is rejected here; ``resolve_match_criterion`` is what fills the recording in the first
    time real GT is available, and every later preflight call sees it.
    """
    default = "objective" if task in ("detection", "instance_seg") else "loss"
    if not requested:
        return default
    if trait:
        from tcip_mcp.pipelines.training.evaluation import CENTER_MATCH_COMPARABILITY_KEYS
        from tcip_mcp.traits import CENTER_MATCH, get_trait

        spec = get_trait(trait)
        if spec.localization == CENTER_MATCH and requested in CENTER_MATCH_COMPARABILITY_KEYS:
            raise ValueError(
                f"evaluation.selection_metric={requested!r} is a comparability-only metric for "
                f"trait {trait!r} (localization=center_match), it does not govern this trait's "
                "phenotype count. Select by 'objective', 'f1', 'precision', 'recall', or 'loss', "
                "which resolve through the trait's own center-match criterion."
            )
    return requested


def _selection_value(task: str, val_metrics: dict, avg_loss: float, metric: str) -> float:
    """Best-model/early-stopping driver: ``val_metrics[f'val_{metric}']``, falling back to ``val_loss``."""
    key = f"val_{metric}"
    if key in val_metrics:
        return val_metrics[key]
    return val_metrics.get("val_loss", avg_loss)


def apply_stage_freeze(
    model: TCIPModel, freeze_to: int, *, prev_trainable: int | None = None,
    enforce_monotonic: bool = True,
) -> int:
    """Apply a stage's progressive-unfreeze policy and return the resulting trainable-param count.

    ``freeze_to``: ``0`` (or a model with no ``freeze_backbone``) trains everything; ``<0`` freezes
    all backbone stages; ``>0`` freezes up to that stage, best-effort, a bespoke model need not
    expose ``freeze_backbone``. When ``enforce_monotonic`` and ``prev_trainable`` is given, an
    unfreeze that shrinks the trainable set raises (progressive unfreeze must only ever grow it).
    Extracted so a hand-rolled ``train(ctx)`` gets the identical policy + guard the default trainer uses.
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


def _expected_in_chans(config: dict) -> int:
    """Input channels the model expects, from ``model_source.in_chans``."""
    src = config.get(MODEL_SOURCE_KEY)
    if isinstance(src, dict):
        return int(src.get("in_chans", 3))
    return 3


def _validate_input_channels(config: dict, loader: DataLoader) -> None:
    """Fail loudly if the data's channel count doesn't match the model's expected ``in_chans``.

    Catches an N-channel/RGB mismatch up front with a clear message instead of an opaque
    conv-shape error deep in the first forward pass. Reads the bespoke ``model_source``
    (which declares ``in_chans``).
    """
    expected = _expected_in_chans(config)
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
            f"Input images have {channels} channels but the model's backbone expects "
            f"in_chans={expected}. Set backbone.in_chans={channels} (or provide matching data)."
        )


# ====================================================================
# Main train loop
# ====================================================================

def train(
    run: TrainRun,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    task: str = "detection",
    epoch_callback=None,
    resume_from: str = "",
) -> TrainRun:
    """Execute a task-agnostic training run.

    The model is built from run.config["model_source"] via build_model().
    ``epoch_callback(epoch:int, epoch_metrics:dict)`` is how each epoch's row reaches the
    run's metrics log and, under HPO, the pruner: this loop composes the row and hands it over
    rather than writing a log of its own, so one sink decides where a row lands. It may raise
    to abort the run (e.g. ``optuna.TrialPruned``).

    This is the canonical, ground-truth list of every ``run.config`` key this function reads
    (every other surface, e.g. ``preflight_config``'s docstring and the training skill's
    example, points here rather than re-deriving its own copy). ``run.config`` is an open dict,
    not a fixed schema (``TrainConfigSchema``/``StageSpec`` both keep ``extra="allow"``), a bespoke
    ``model_source``/``dataset_source``/``training_source`` may read its own additional keys this
    function never touches.

    - ``device`` (str, default cuda-if-available else cpu)
    - ``batch_size`` (int), only as a fallback when ``train_loader`` itself has no ``.batch_size``
      (the DataLoader's own batch size, set at construction, is the primary source).
    - ``seed`` (int | None), ``deterministic`` (bool, default False), RNG seeding before model
      build; also under ``training.seed``/``training.deterministic``. ``create_run`` already draws
      and records a seed when none is configured, so ``seed`` is never actually ``None`` here in
      practice.
    - ``mixed_precision`` (bool, default True), AMP, only when ``device`` is cuda.
    - ``stages`` (list of ``{freeze_to, epochs}``; a per-stage ``lr`` is accepted but ignored,
      see ``optimizer`` below), default a single 10-epoch full-unfreeze stage.
    - ``optimizer`` (``{name, backbone_lr, head_lr, weight_decay}``, default adamw/1e-4/1e-3/1e-4)
, the one source of learning rate, applied uniformly across every stage.
    - ``scheduler`` (``{type, ...}``; ``type`` in cosine/plateau/onecycle/step, default cosine).
    - ``lr_scaling`` (``{enabled, reference_effective_batch, scale_power, max_lr}``, default
      disabled), effective-batch LR scaling at stage boundaries.
    - ``stage_warmup_epochs`` (int, default 0), ``enforce_monotonic_unfreeze`` (bool, default True).
    - ``gradient_accumulation_steps`` (int, default 1; a stage may override its own), and a
      per-stage ``gradient_accumulation_steps`` override.
    - ``checkpoint_every_n_epochs`` (int, default 5), periodic resumable checkpoints.
    - ``early_stopping`` (``{enabled, patience, min_delta}``, default enabled-if-val_loader,
      patience 7, min_delta 1e-4).
    - ``evaluation`` (``{trait, selection_metric, conf_threshold, iou_threshold, iou_type,
      max_dets, score_weights}``, all optional), ``trait`` and ``selection_metric`` drive
      ``resolve_selection_metric``; the rest pass through to ``_validate``/``evaluate``.
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
        seed = config.get("seed", config.get("training", {}).get("seed"))
        if seed is not None:
            set_seed(int(seed), deterministic=config.get(
                "deterministic", config.get("training", {}).get("deterministic", False)))

        model = build_model(config)
        model.to(device)
        _validate_input_channels(config, train_loader)

        stages = config.get("stages", [{"freeze_to": 0, "epochs": 10}])
        use_amp = config.get("mixed_precision", True) and device.type == "cuda"
        scaler = torch.amp.GradScaler(device.type) if use_amp else None

        opt_cfg = config.get("optimizer", {"name": "adamw", "backbone_lr": 1e-4, "head_lr": 1e-3, "weight_decay": 1e-4})
        sched_cfg = config.get("scheduler", {"type": "cosine"})

        # Progressive-unfreezing fidelity setup.
        base_backbone_lr = opt_cfg.get("backbone_lr", 1e-4)
        base_head_lr = opt_cfg.get("head_lr", 1e-3)
        lr_scaling_cfg = config.get("lr_scaling", {})
        stage_warmup_epochs = int(config.get("stage_warmup_epochs", 0))
        enforce_monotonic_unfreeze = config.get("enforce_monotonic_unfreeze", True)
        physical_batch = getattr(train_loader, "batch_size", None) or config.get("batch_size") or 1
        pending_snapshot = None   # best optimizer state from the previous stage
        prev_trainable = None     # trainable param count of the previous stage
        eval_cfg = config.get("evaluation", {})  # metric / selection params
        trait = eval_cfg.get("trait")
        selection_metric = resolve_selection_metric(task, trait, eval_cfg.get("selection_metric"))

        global_step = 0
        stopped_early = False

        # Resume from a periodic checkpoint (model + optimizer + scheduler + scaler).
        resume_stage = -1
        resume_stage_epoch = 0
        ckpt = None
        if resume_from:
            ckpt = torch.load(resume_from, map_location=device, weights_only=False)
            missing = [k for k in (STATE_DICT_KEY, "optimizer_state_dict") if k not in ckpt]
            if missing:
                # Fail loudly instead of silently restarting from scratch.
                raise ValueError(
                    f"Cannot resume from {resume_from}: checkpoint is missing {missing} "
                    "(a non-resumable checkpoint, e.g. model_best.pt, which never carries resume "
                    "state). Resume "
                    "from a periodic checkpoint_epoch_*.pt, or start a fresh run."
                )
            model.load_state_dict(ckpt[STATE_DICT_KEY])
            resume_stage = ckpt.get("stage", 0)
            resume_stage_epoch = ckpt.get("stage_epoch", 0)
            run.current_epoch = ckpt.get("epoch", 0)
            run.best_metric = ckpt.get("best_metric", run.best_metric)
            es_best = ckpt.get("es_best", es_best)
            es_counter = ckpt.get("es_counter", es_counter)
            global_step = ckpt.get("global_step", 0)
            # Restore RNG state after the fresh `set_seed()` call above (never skip that
            # call, it also configures cudnn.deterministic/benchmark) so the resumed streams
            # overwrite the freshly-seeded ones rather than starting over from stream position
            # zero. Older checkpoints predating this field degrade gracefully to the fresh seed.
            if "torch_rng_state" in ckpt:
                torch.set_rng_state(ckpt["torch_rng_state"])
                if torch.cuda.is_available() and ckpt.get("cuda_rng_state") is not None:
                    torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
                np.random.set_state(ckpt["numpy_rng_state"])
                random.setstate(ckpt["python_rng_state"])
                run.rng_state_restored = True
            else:
                run.rng_state_restored = False
            logger.info("Resuming from %s at stage %d, stage_epoch %d (global epoch %d)",
                        resume_from, resume_stage, resume_stage_epoch, run.current_epoch)

        for stage_idx, stage in enumerate(stages):
            if stopped_early:
                break
            run.current_stage = stage_idx

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
                weight_decay=opt_cfg.get("weight_decay", 1e-4),
            )

            # Hand off momentum from the previous stage's best epoch.
            if pending_snapshot is not None:
                restored = restore_optimizer_state(optimizer, model, pending_snapshot)
                logger.info("Stage %d: restored optimizer state for %d params", stage_idx, restored)

            target_lrs = [g["lr"] for g in optimizer.param_groups]
            prev_end_lrs = pending_snapshot.get("end_lrs") if pending_snapshot else None

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
            stage_best = float("inf")
            stage_snapshot = None

            # For the resumed stage, restore optimizer/scheduler/scaler and start
            # mid-stage; later stages keep fresh state (and the momentum handoff above).
            start_epoch = 0
            if stage_idx == resume_stage and ckpt is not None:
                start_epoch = resume_stage_epoch
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
                        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
                        scaler.load_state_dict(ckpt["scaler_state_dict"])
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Resume: optimizer/scheduler restore failed (%s); using fresh state.", exc)
                ckpt = None

            for epoch in range(start_epoch, stage_epochs):
                if stopped_early or run.should_cancel():
                    break
                run.current_epoch += 1
                model.train()
                epoch_loss = 0.0
                n_batches = 0
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
                        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
                    else:
                        images, targets = batch
                        images = images.to(device)
                        targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}

                    if use_amp:
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

                    epoch_loss += loss.item()
                    n_batches += 1
                    global_step += 1

                    if tb_writer and global_step % 10 == 0:
                        tb_writer.add_scalar("train/loss_step", loss.item(), global_step)

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
                        # A center-match trait's val_metrics carries non-scalar entries
                        # (governing_criterion: dict, map50_role: str) that add_scalar rejects.
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            tb_writer.add_scalar(f"val/{k}", v, run.current_epoch)
                    tb_writer.flush()

                if epoch_callback is not None:
                    epoch_callback(run.current_epoch, epoch_metrics)

                logger.info("Epoch %d stage %d loss=%.4f val_loss=%.4f lr=%.2e",
                    run.current_epoch, stage_idx, avg_loss, val_metrics.get("val_loss", 0), current_lr)

                # Best model checkpoint, selected by the selection objective.
                if sel < run.best_metric:
                    run.best_metric = sel
                    try:
                        write_checkpoint(stamp_model_ref({
                            STATE_DICT_KEY: model.state_dict(),
                            "config": config,
                            "metrics": epoch_metrics,
                            "stage": stage_idx, "epoch": run.current_epoch,
                        }, config), checkpoint_key(out_dir, "model_best"))
                    except PermissionError:
                        # Windows: a concurrent reader (GUI inference / evaluate_model)
                        # holding model_best.pt open can outlast the replace retries.
                        # Keep the previous best on disk rather than failing the run.
                        logger.warning(
                            "model_best.pt held open by a reader; keeping previous best "
                            "(epoch %d not persisted).", run.current_epoch)

                # Remember this stage's best optimizer state for the handoff.
                if sel < stage_best:
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

                # Early stopping, on the same selection objective.
                if es_enabled and val_loader is not None:
                    if sel < es_best - es_min_delta:
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

            if run.should_cancel():
                break  # stop before starting the next stage

        # Final checkpoint (saved even on cancellation so partial progress is recoverable).
        write_checkpoint(stamp_model_ref({
            STATE_DICT_KEY: model.state_dict(),
            "config": config,
            "metrics": run.metrics_history,
        }, config), checkpoint_key(out_dir, "model_final"))

        if run.should_cancel():
            run.status = "cancelled"
            logger.info("Training run %s cancelled at epoch %d", run.run_id, run.current_epoch)
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
