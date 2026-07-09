"""W2 — progressive-unfreezing fidelity.

Covers the three optimizer_factory helpers (LR scaling, name-keyed optimizer
snapshot/restore) and their integration in generic_trainer.train(): the
non-decreasing-unfreeze guard, inter-stage LR warmup, effective-batch LR
scaling, and a two-stage handoff smoke test. Tiny synthetic data, CPU,
``pretrained=False`` so it stays under the CI per-test timeout.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import tcip_mcp.pipelines.components.backbones  # noqa: F401,E402
import tcip_mcp.pipelines.components.necks  # noqa: F401,E402
import tcip_mcp.pipelines.components.heads  # noqa: F401,E402
import tcip_mcp.pipelines.components.losses  # noqa: F401,E402
from tcip_mcp.pipelines.composer import compose_model  # noqa: E402
from tcip_mcp.pipelines.data.datasets import build_dataset  # noqa: E402
from tcip_mcp.pipelines.training.generic_trainer import (  # noqa: E402
    create_run, task_collate, train,
)
from tcip_mcp.pipelines.training.optimizer_factory import (  # noqa: E402
    compute_lr_scale, snapshot_optimizer_state, restore_optimizer_state,
)

IMG = 64
BASE_BB_LR = 1e-3


# --------------------------------------------------------------------------
# Unit tests — pure helpers
# --------------------------------------------------------------------------

def test_compute_lr_scale():
    assert compute_lr_scale(128, 64, 0.5) == pytest.approx(2 ** 0.5)
    assert compute_lr_scale(1024, 64, 0.5) == pytest.approx(4.0)
    assert compute_lr_scale(64, 64, 0.5) == 1.0
    assert compute_lr_scale(8, 0, 0.5) == 1.0  # reference_batch <= 0 guard


def test_snapshot_restore_roundtrip():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    for p in model[0].parameters():  # freeze first Linear
        p.requires_grad = False

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    x = torch.randn(3, 4)
    opt.zero_grad()
    model(x).sum().backward()
    opt.step()  # creates exp_avg / exp_avg_sq for the trainable params

    snap = snapshot_optimizer_state(opt, model)
    assert set(snap["state_by_name"]) == {"1.weight", "1.bias"}
    assert len(snap["end_lrs"]) == 1

    # Unfreeze everything; restore into a fresh optimizer over the larger set.
    for p in model.parameters():
        p.requires_grad = True
    new_opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    restored = restore_optimizer_state(new_opt, model, snap)
    assert restored == 2

    assert torch.allclose(
        new_opt.state[model[1].weight]["exp_avg"],
        snap["state_by_name"]["1.weight"]["exp_avg"],
    )
    # Newly unfrozen params carry no optimizer state yet.
    assert model[0].weight not in new_opt.state


def test_freeze_to_is_per_stage_for_tv_backbones():
    """tv_* fallback backbones must freeze per stage, not all-or-nothing.

    Regression: _freeze_sequential_fraction saw _MultiStageExtractor's sole
    named child (the ``stages`` ModuleList) and froze the entire backbone for
    any freeze_to >= 1, so intermediate schedule stages silently trained
    heads-only.
    """
    from tcip_mcp.pipelines.registry import BACKBONES

    bb = BACKBONES.build("tv_resnet50", pretrained=False)
    counts = []
    for stage in range(bb.num_stages + 1):  # 0 .. 4
        bb.freeze_to(stage)
        counts.append(sum(p.numel() for p in bb.model.parameters() if p.requires_grad))
    assert counts[0] > 0  # freeze_to=0 leaves everything trainable
    assert counts[-1] == 0  # freeze_to=num_stages freezes everything
    # Strictly decreasing: each extra stage frozen removes trainable params.
    assert all(b < a for a, b in zip(counts, counts[1:])), counts


# --------------------------------------------------------------------------
# Integration helpers
# --------------------------------------------------------------------------

def _save_png(path: Path, bright: bool = False) -> None:
    from torchvision.utils import save_image

    base = 0.7 if bright else 0.0
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(torch.rand(3, IMG, IMG) * 0.3 + base, str(path))


def _classification_loader(tmp_path: Path, n: int = 6, batch_size: int = 2) -> DataLoader:
    images_dir = tmp_path / "images"
    rows = []
    for i in range(n):
        _save_png(images_dir / f"img{i}.png", bright=(i % 2 == 0))
        rows.append((f"img{i}", i % 2))
    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("stem", "label"))
        w.writerows(rows)
    ds = build_dataset("classification", images_dir=str(images_dir), csv_path=str(csv_path), num_classes=2)
    return DataLoader(ds, batch_size=batch_size, collate_fn=task_collate("classification"))


def _cls_spec() -> dict:
    return {
        "backbone": {"name": "tv_resnet50", "pretrained": False},
        "neck": {"name": "gap"},
        "heads": [{"name": "classification", "num_classes": 2}],
    }


def _cfg(stages, **extra) -> dict:
    cfg = {
        "model_spec": _cls_spec(),
        "device": "cpu",
        "stages": stages,
        "mixed_precision": False,
        "optimizer": {"name": "adamw", "backbone_lr": BASE_BB_LR, "head_lr": 1e-3, "weight_decay": 0},
        "scheduler": {"type": "cosine"},
        "early_stopping": {"enabled": False},
    }
    cfg.update(extra)
    return cfg


# --------------------------------------------------------------------------
# Integration tests — train()
# --------------------------------------------------------------------------

def test_monotonic_unfreeze_guard_fails(tmp_path: Path):
    loader = _classification_loader(tmp_path)
    spec = _cls_spec()
    compose_model(spec)
    # Stage 0 fully unfreezes; stage 1 re-freezes the backbone -> guard must fire.
    cfg = _cfg([{"freeze_to": 0, "epochs": 1}, {"freeze_to": -1, "epochs": 1}])
    run = create_run(cfg, str(tmp_path / "out"))
    run = train(run, loader, val_loader=None, task="classification")
    assert run.status == "failed"
    assert "Non-decreasing unfreeze" in run.error


def test_warmup_lr_ramps_at_stage_boundary(tmp_path: Path):
    loader = _classification_loader(tmp_path)
    spec = _cls_spec()
    compose_model(spec)
    cfg = _cfg(
        [{"freeze_to": -1, "epochs": 2}, {"freeze_to": 0, "epochs": 3}],
        stage_warmup_epochs=2,
    )
    run = create_run(cfg, str(tmp_path / "out"))
    run = train(run, loader, val_loader=None, task="classification")
    assert run.status == "completed", getattr(run, "error", run.status)

    stage1 = [m for m in run.metrics_history if m["stage"] == 1]
    assert len(stage1) == 3
    assert stage1[0]["lr"] > 0 and stage1[1]["lr"] > 0
    assert stage1[1]["lr"] >= stage1[0]["lr"]  # ramping up toward target
    for m in run.metrics_history:
        assert "eff_batch" in m and "trainable_params" in m


def test_lr_scaling_applied(tmp_path: Path):
    spec = _cls_spec()
    compose_model(spec)
    stages = [{"freeze_to": -1, "epochs": 1, "gradient_accumulation_steps": 2}]

    # batch_size 2 * accum 2 -> eff_batch 4. ref 4 -> mult 1.0.
    loader_a = _classification_loader(tmp_path / "a", batch_size=2)
    cfg_a = _cfg(stages, lr_scaling={"enabled": True, "reference_effective_batch": 4, "scale_power": 0.5})
    run_a = train(create_run(cfg_a, str(tmp_path / "a" / "out")), loader_a, task="classification")
    assert run_a.status == "completed", getattr(run_a, "error", run_a.status)
    assert run_a.metrics_history[0]["eff_batch"] == 4
    assert run_a.metrics_history[0]["lr"] == pytest.approx(BASE_BB_LR * 1.0)

    # ref 1 -> mult (4/1)^0.5 == 2.0.
    loader_b = _classification_loader(tmp_path / "b", batch_size=2)
    cfg_b = _cfg(stages, lr_scaling={"enabled": True, "reference_effective_batch": 1, "scale_power": 0.5})
    run_b = train(create_run(cfg_b, str(tmp_path / "b" / "out")), loader_b, task="classification")
    assert run_b.metrics_history[0]["lr"] == pytest.approx(BASE_BB_LR * 2.0)


def test_two_stage_handoff_smoke(tmp_path: Path):
    loader = _classification_loader(tmp_path)
    spec = _cls_spec()
    compose_model(spec)
    cfg = _cfg([{"freeze_to": -1, "epochs": 1}, {"freeze_to": 0, "epochs": 1}])
    run = create_run(cfg, str(tmp_path / "out"))
    run = train(run, loader, val_loader=None, task="classification")
    assert run.status == "completed", getattr(run, "error", run.status)
    assert all(math.isfinite(m["train_loss"]) for m in run.metrics_history)
    tps = [m["trainable_params"] for m in run.metrics_history]
    assert all(b >= a for a, b in zip(tps, tps[1:]))  # non-decreasing across stages
    assert (tmp_path / "out" / "model_best.pt").is_file()
