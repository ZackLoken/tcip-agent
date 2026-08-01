"""K24: the subprocess entry point ``launch_training`` spawns to run one bespoke training run's
actual body — dataset/loader construction, the audited envelope, ``run_training_envelope()`` — in
an isolated OS process, so a leak/OOM/hang in one run can't take down the launching process or any
other concurrent run's process. Everything here mirrors what ``launch_training`` did synchronously
in-process before K24; only the process boundary moved.

Invoked as ``python -m tcip_mcp.pipelines.training.subprocess_worker --run-id ... --experiment-id
... --config-path ... --output-dir ... --resume-from ...`` — never imported for its functions
elsewhere, only run as ``__main__``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--config-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--resume-from", default="")
    return p.parse_args()


def _patch_experiment_config_tiling(experiment_id: str, tiling_cfg: dict) -> None:
    """Best-effort: patch the EFFECTIVE tiling geometry (CV2) into the durable experiment record's
    own ``config.json`` — a small merge, not a rewrite. Never sinks the run if the experiment
    directory doesn't exist (experiment tracking is best-effort throughout this path, same as
    every other write in it)."""
    try:
        from tcip_mcp.experiments import experiments_dir
        from tcip_mcp.utils.atomic_io import atomic_write_json, file_transaction

        exp_config_path = experiments_dir() / experiment_id / "config.json"
        if not exp_config_path.is_file():
            return
        with file_transaction(exp_config_path):
            cfg = json.loads(exp_config_path.read_text())
            cfg.setdefault("data", {}).setdefault("tiling", {}).update(tiling_cfg)
            atomic_write_json(exp_config_path, cfg)
    except Exception:
        logger.warning("tiling geometry patch-back failed for %s", experiment_id, exc_info=True)


def run(run_id: str, experiment_id: str, config_path: str, output_dir: str, resume_from: str) -> None:
    """The training body — identical in substance to what ``launch_training`` ran synchronously
    in-process before K24, just executing in this dedicated process instead."""
    from tcip_mcp.pipelines.training.generic_trainer import (
        attach_run, seeded_loader_kwargs, task_collate,
    )
    from tcip_mcp.tools.training_tools import _auto_train_val, _dataset_identity, _persist_split_manifest

    config = json.loads(Path(config_path).read_text())
    run_obj = attach_run(run_id, config, output_dir)

    model_source = config.get("model_source", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    # Task drives collate + measurement routing: the bespoke model_source declares it, falling
    # back to the data section.
    task = model_source.get("task") or data_cfg.get("task", "detection")

    aug_config = config.get("augmentation", {})
    transforms = None
    if aug_config:
        from tcip_mcp.pipelines.data.augmentations import build_augmentation
        transforms = build_augmentation(aug_config)

    train_ds, val_ds = _auto_train_val(task, data_cfg, transforms)

    # CV2: resolve the EFFECTIVE tiling geometry (the 224/0.2 defaults used when the tiling dict
    # omitted them, not just caller-pinned values) — only knowable once the dataset is actually
    # built, so patched into the durable experiment record here rather than before launch.
    tiling_cfg = data_cfg.get("tiling")
    if tiling_cfg and tiling_cfg.get("enabled", True):
        eff_tile = getattr(train_ds, "tile_size", None)
        eff_overlap = getattr(train_ds, "overlap", None)
        if eff_tile is not None:
            tiling_cfg["tile_size"] = int(eff_tile)
        if eff_overlap is not None:
            tiling_cfg["overlap"] = float(eff_overlap)
        _patch_experiment_config_tiling(experiment_id, tiling_cfg)

    from tcip_mcp.pipelines.data.samplers import build_sampler
    from torch.utils.data import DataLoader

    sampler = build_sampler(config.get("sampler", "random"), train_ds)
    # K11: seed the loader's shuffle/worker RNG from the run's own (parent-resolved) seed — the
    # same value set_seed uses, so a resumed run's data order is reproducible too.
    loader_kwargs = seeded_loader_kwargs(config.get("seed"))
    batch_size = train_cfg.get("batch_size", 2)
    num_workers = train_cfg.get("num_workers", 0)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=(sampler is None), sampler=sampler,
        collate_fn=task_collate(task),
        num_workers=num_workers,
        **loader_kwargs,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=batch_size,
            shuffle=False,
            collate_fn=task_collate(task),
            num_workers=num_workers,
            **loader_kwargs,
        )
    if val_loader is None and task in ("detection", "instance_seg"):
        logger.warning(
            "No validation loader for %s run %s: best-model selection and early "
            "stopping will fall back to training loss (no val mAP/composite). "
            "Provide a val split (data.val_images_dir) or enable auto_val.",
            task, run_id,
        )

    # The dataset identity this run trains on — recomputed here (recompute-on-read is this fact's
    # own stated authority) rather than threaded across the process boundary; same deterministic
    # result the parent's own copy (used for the lineage record) already produced.
    ds_id, ds_fp = _dataset_identity(data_cfg)
    _persist_split_manifest(experiment_id, train_ds, val_ds, data_cfg,
                            dataset_id=ds_id, dataset_fingerprint=ds_fp)

    from tcip_mcp.pipelines.training.envelope import TrainContext, run_training_envelope

    ctx = TrainContext(
        run=run_obj, train_loader=train_loader, val_loader=val_loader,
        task=task, resume_from=resume_from, experiment_id=experiment_id,
    )
    run_training_envelope(ctx)


def main() -> None:
    args = _parse_args()
    run(args.run_id, args.experiment_id, args.config_path, args.output_dir, args.resume_from)


if __name__ == "__main__":
    main()
