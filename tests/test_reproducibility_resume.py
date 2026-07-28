"""W7 — global seeding + resume-from-checkpoint."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import numpy as np  # noqa: E402

from tcip_mcp.pipelines.training.generic_trainer import set_seed  # noqa: E402


def test_set_seed_reproducible():
    set_seed(123)
    a = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    set_seed(123)
    b = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert a == b


def test_set_seed_sets_cudnn_flags():
    det0, bench0 = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    try:
        set_seed(1, deterministic=True)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
    finally:
        torch.backends.cudnn.deterministic = det0
        torch.backends.cudnn.benchmark = bench0


# --------------------------------------------------------------------------
# Integration (needs torchvision)
# --------------------------------------------------------------------------

pytest.importorskip("torchvision")

import tcip_mcp.pipelines.components.backbones  # noqa: F401,E402
import tcip_mcp.pipelines.components.necks  # noqa: F401,E402
import tcip_mcp.pipelines.components.heads  # noqa: F401,E402
import tcip_mcp.pipelines.components.losses  # noqa: F401,E402
from tcip_mcp.pipelines.data.datasets import build_dataset  # noqa: E402
from tcip_mcp.pipelines.training.generic_trainer import create_run, task_collate, train  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def _classification_data(tmp_path: Path, n: int = 6):
    from PIL import Image
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        Image.new("RGB", (32, 32), (40 * (i % 5), 50, 60)).save(images_dir / f"img{i}.png")
        rows.append((f"img{i}", i % 2))
    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("stem", "label"))
        w.writerows(rows)
    return str(images_dir), str(csv_path)


def _model_source():
    return {"builder": "tests.bespoke_models:build_bespoke_classifier",
            "builder_kwargs": {"num_classes": 2}, "task": "classification"}


def _cfg(stages, **extra):
    cfg = {
        "model_source": _model_source(), "device": "cpu", "stages": stages,
        "mixed_precision": False,
        "optimizer": {"name": "adamw", "backbone_lr": 1e-4, "head_lr": 1e-3, "weight_decay": 0},
        "early_stopping": {"enabled": False}, "checkpoint_every_n_epochs": 1,
    }
    cfg.update(extra)
    return cfg


def test_seeded_train_reproducible(tmp_path):
    images_dir, csv_path = _classification_data(tmp_path)

    def run_once(out):
        ds = build_dataset("classification", images_dir=images_dir, csv_path=csv_path, num_classes=2)
        loader = DataLoader(ds, batch_size=2, collate_fn=task_collate("classification"))
        run = create_run(_cfg([{"freeze_to": -1, "epochs": 1}], seed=7), str(out))
        run = train(run, loader, task="classification")
        return run.metrics_history[0]["train_loss"]

    assert run_once(tmp_path / "a") == pytest.approx(run_once(tmp_path / "b"))


def test_resume_continues_epochs(tmp_path):
    images_dir, csv_path = _classification_data(tmp_path)
    ds = build_dataset("classification", images_dir=images_dir, csv_path=csv_path, num_classes=2)
    loader = DataLoader(ds, batch_size=2, collate_fn=task_collate("classification"))
    cfg = _cfg([{"freeze_to": -1, "epochs": 2}])

    train(create_run(cfg, str(tmp_path / "out")), loader, task="classification")
    ckpt = tmp_path / "out" / "checkpoint_epoch_1.pt"
    assert ckpt.is_file()

    run2 = create_run(cfg, str(tmp_path / "out2"))
    run2 = train(run2, loader, task="classification", resume_from=str(ckpt))
    assert run2.status == "completed"
    assert run2.current_epoch == 2          # continued global epoch count
    assert len(run2.metrics_history) == 1   # only the one remaining epoch
    assert (tmp_path / "out2" / "model_final.pt").is_file()


def test_resume_skips_completed_stage_and_restores_optimizer(tmp_path):
    images_dir, csv_path = _classification_data(tmp_path)
    ds = build_dataset("classification", images_dir=images_dir, csv_path=csv_path, num_classes=2)
    loader = DataLoader(ds, batch_size=2, collate_fn=task_collate("classification"))
    cfg = _cfg([{"freeze_to": -1, "epochs": 1}, {"freeze_to": 0, "epochs": 1}])

    train(create_run(cfg, str(tmp_path / "out")), loader, task="classification")
    ckpt = tmp_path / "out" / "checkpoint_epoch_1.pt"  # end of stage 0
    assert ckpt.is_file()

    run2 = create_run(cfg, str(tmp_path / "out2"))
    run2 = train(run2, loader, task="classification", resume_from=str(ckpt))
    assert run2.status == "completed"
    assert run2.current_stage == 1          # stage 0 skipped, stage 1 ran
    assert run2.current_epoch == 2          # restored optimizer + ran stage 1's epoch


def test_resume_restores_rng_state_not_just_reseeds(tmp_path):
    """K11 (F5): resuming must restore the RNG stream position, not just reseed from scratch —
    a resumed epoch's loss must match the straight-through run's epoch at the same point, even
    if the global RNG is deliberately corrupted between save and resume.

    ``shuffle=True`` with no explicit ``generator=`` is deliberate: PyTorch's ``RandomSampler``
    draws a fresh per-epoch seed from the GLOBAL torch RNG on every ``__iter__`` call when no
    generator is given, so the resumed epoch's batch order is genuinely sensitive to whatever
    ``torch.set_rng_state`` last set it to — a sequential (unshuffled) loader would make this
    test pass identically whether or not RNG restoration actually ran.
    """
    images_dir, csv_path = _classification_data(tmp_path)

    def build_loader():
        ds = build_dataset("classification", images_dir=images_dir, csv_path=csv_path, num_classes=2)
        return DataLoader(ds, batch_size=2, shuffle=True, collate_fn=task_collate("classification"))

    cfg = _cfg([{"freeze_to": -1, "epochs": 2}], seed=11)

    # Straight-through baseline: both epochs in one uninterrupted run.
    straight = train(create_run(cfg, str(tmp_path / "straight")), build_loader(), task="classification")
    baseline_epoch2_loss = straight.metrics_history[1]["train_loss"]

    # Split run: epoch 1 checkpointed, global RNG deliberately corrupted, then resumed for epoch 2.
    train(create_run(cfg, str(tmp_path / "out")), build_loader(), task="classification")
    ckpt = tmp_path / "out" / "checkpoint_epoch_1.pt"
    assert ckpt.is_file()
    assert "torch_rng_state" in torch.load(ckpt, weights_only=False)

    torch.manual_seed(999)
    np.random.seed(999)
    random.seed(999)

    resumed = train(create_run(cfg, str(tmp_path / "out2")), build_loader(),
                    task="classification", resume_from=str(ckpt))
    assert resumed.rng_state_restored is True
    resumed_epoch2_loss = resumed.metrics_history[0]["train_loss"]  # the one epoch this run ran

    assert resumed_epoch2_loss == pytest.approx(baseline_epoch2_loss)


def test_resume_from_checkpoint_without_rng_state_degrades_gracefully(tmp_path):
    """K11 (F5): a checkpoint predating RNG capture must resume via the fresh seed (old
    behavior), not crash — and must honestly record rng_state_restored=False rather than
    claiming a restore that didn't happen."""
    images_dir, csv_path = _classification_data(tmp_path)
    ds = build_dataset("classification", images_dir=images_dir, csv_path=csv_path, num_classes=2)
    loader = DataLoader(ds, batch_size=2, collate_fn=task_collate("classification"))
    cfg = _cfg([{"freeze_to": -1, "epochs": 2}])

    train(create_run(cfg, str(tmp_path / "out")), loader, task="classification")
    ckpt_path = tmp_path / "out" / "checkpoint_epoch_1.pt"
    ckpt = torch.load(ckpt_path, weights_only=False)
    for key in ("torch_rng_state", "numpy_rng_state", "python_rng_state", "cuda_rng_state"):
        ckpt.pop(key, None)
    torch.save(ckpt, ckpt_path)  # simulate a pre-K11 checkpoint

    run2 = create_run(cfg, str(tmp_path / "out2"))
    run2 = train(run2, loader, task="classification", resume_from=str(ckpt_path))
    assert run2.status == "completed"
    assert run2.rng_state_restored is False


def test_seeded_loader_kwargs_reproducible_shuffle():
    """K11 (F5): the platform's own DataLoader construction sites are seeded — two loaders built
    from the same seed shuffle identically; an unseeded run stays honestly unseeded."""
    from tcip_mcp.pipelines.training.generic_trainer import seeded_loader_kwargs

    kw1 = seeded_loader_kwargs(42)
    kw2 = seeded_loader_kwargs(42)
    perm1 = torch.randperm(10, generator=kw1["generator"])
    perm2 = torch.randperm(10, generator=kw2["generator"])
    assert torch.equal(perm1, perm2)

    assert seeded_loader_kwargs(None) == {}


def test_resume_from_non_resumable_checkpoint_fails_loudly(tmp_path):
    # 2.3: resuming a checkpoint without optimizer state (e.g. model_best.pt) must fail
    # loudly, not silently restart from scratch.
    images_dir, csv_path = _classification_data(tmp_path)
    ds = build_dataset("classification", images_dir=images_dir, csv_path=csv_path, num_classes=2)
    loader = DataLoader(ds, batch_size=2, collate_fn=task_collate("classification"))
    cfg = _cfg([{"freeze_to": -1, "epochs": 1}])

    train(create_run(cfg, str(tmp_path / "out")), loader, task="classification")
    best = tmp_path / "out" / "model_best.pt"   # has model_state_dict but no optimizer_state_dict
    assert best.is_file()

    run2 = create_run(cfg, str(tmp_path / "out2"))
    run2 = train(run2, loader, task="classification", resume_from=str(best))
    assert run2.status == "failed"
    assert "resume" in (run2.error or "").lower()
    assert not (tmp_path / "out2" / "model_final.pt").is_file()  # did not silently train
