"""S0 characterization goldens — pin the CURRENT craft-layer train/eval + provenance shapes.

These lock the behavior the Phase-3 model-layer refactor must preserve byte-for-byte on the
default (``model_spec`` → ``compose_model``) path: what ``train()`` writes (``model_best.pt`` /
``model_final.pt`` / periodic checkpoint / ``metrics.jsonl`` keys), that resume continues, and
that a trained checkpoint round-trips through ``register_model_from_experiment`` (kind + metrics
+ sha256 + lineage). If a later refactor shifts any of these, this file fails loudly.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import tcip_mcp.pipelines.components.backbones  # noqa: F401,E402
import tcip_mcp.pipelines.components.necks  # noqa: F401,E402
import tcip_mcp.pipelines.components.heads  # noqa: F401,E402
import tcip_mcp.pipelines.components.losses  # noqa: F401,E402
from tcip_mcp.pipelines.inference.predictor import KIND_TORCHVISION_COMPOSED  # noqa: E402
from tcip_mcp.pipelines.training.generic_trainer import (  # noqa: E402
    create_run,
    task_collate,
    train,
)
from torch.utils.data import DataLoader  # noqa: E402


# --------------------------------------------------------------------------
# Fixtures — tiny seeded classification run (fast, CPU, deterministic keys)
# --------------------------------------------------------------------------

def _cls_data(tmp_path: Path, n: int = 6):
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


def _cls_spec() -> dict:
    return {
        "backbone": {"name": "resnet18", "pretrained": False},
        "neck": {"name": "gap"},
        "heads": [{"name": "classification", "num_classes": 2}],
    }


def _cfg(stages, **extra) -> dict:
    cfg = {
        "model_spec": _cls_spec(), "device": "cpu", "stages": stages,
        "mixed_precision": False,
        "optimizer": {"name": "adamw", "backbone_lr": 1e-4, "head_lr": 1e-3, "weight_decay": 0},
        "early_stopping": {"enabled": False}, "checkpoint_every_n_epochs": 1,
    }
    cfg.update(extra)
    return cfg


def _loader(tmp_path: Path):
    from tcip_mcp.pipelines.data.datasets import build_dataset
    images_dir, csv_path = _cls_data(tmp_path)
    ds = build_dataset("classification", images_dir=images_dir, csv_path=csv_path, num_classes=2)
    return DataLoader(ds, batch_size=2, collate_fn=task_collate("classification"))


# --------------------------------------------------------------------------
# GOLDEN 1 — train() writes the expected artifacts + metrics.jsonl keys
# --------------------------------------------------------------------------

def test_golden_train_writes_artifacts_and_metric_keys(tmp_path):
    loader = _loader(tmp_path)
    out = tmp_path / "out"
    run = create_run(_cfg([{"freeze_to": -1, "epochs": 2}], seed=7), str(out))
    run = train(run, loader, val_loader=loader, task="classification")

    assert run.status == "completed", getattr(run, "error", run.status)
    assert (out / "model_best.pt").is_file()
    assert (out / "model_final.pt").is_file()
    assert (out / "checkpoint_epoch_1.pt").is_file()  # ckpt_every=1
    assert (out / "metrics.jsonl").is_file()

    lines = (out / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2  # one row per epoch
    row = json.loads(lines[-1])
    for k in ("epoch", "stage", "train_loss", "lr", "eff_batch", "trainable_params", "selection"):
        assert k in row, f"metrics.jsonl missing craft key {k!r}"
    # classification + val_loader → val_loss/val_accuracy/val_f1 recorded (val_-prefixed).
    for k in ("val_loss", "val_accuracy", "val_f1"):
        assert k in row, f"metrics.jsonl missing {k!r}"


# --------------------------------------------------------------------------
# GOLDEN 2 — checkpoint payload schemas (best / final / periodic) + kind stamp
# --------------------------------------------------------------------------

def test_golden_checkpoint_payload_schemas(tmp_path):
    loader = _loader(tmp_path)
    out = tmp_path / "out"
    run = create_run(_cfg([{"freeze_to": -1, "epochs": 2}], seed=11), str(out))
    train(run, loader, val_loader=loader, task="classification")

    best = torch.load(out / "model_best.pt", weights_only=False)
    assert best["kind"] == KIND_TORCHVISION_COMPOSED
    assert set(best) == {"kind", "model_state_dict", "model_spec", "config", "metrics", "stage", "epoch"}
    assert best["model_spec"] == _cls_spec()

    final = torch.load(out / "model_final.pt", weights_only=False)
    assert final["kind"] == KIND_TORCHVISION_COMPOSED
    assert set(final) == {"kind", "model_state_dict", "model_spec", "config", "metrics"}

    periodic = torch.load(out / "checkpoint_epoch_1.pt", weights_only=False)
    assert periodic["kind"] == KIND_TORCHVISION_COMPOSED
    assert set(periodic) == {
        "kind", "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
        "scaler_state_dict", "model_spec", "config", "stage", "stage_epoch", "epoch",
        "best_metric", "es_best", "es_counter", "global_step", "seed", "metrics",
    }


# --------------------------------------------------------------------------
# GOLDEN 3 — resume from a periodic checkpoint continues the run
# --------------------------------------------------------------------------

def test_golden_resume_continues(tmp_path):
    loader = _loader(tmp_path)
    cfg = _cfg([{"freeze_to": -1, "epochs": 2}])

    train(create_run(cfg, str(tmp_path / "out")), loader, task="classification")
    ckpt = tmp_path / "out" / "checkpoint_epoch_1.pt"
    assert ckpt.is_file()

    run2 = create_run(cfg, str(tmp_path / "out2"))
    run2 = train(run2, loader, task="classification", resume_from=str(ckpt))
    assert run2.status == "completed"
    assert run2.current_epoch == 2          # continued global epoch count
    assert len(run2.metrics_history) == 1   # only the remaining epoch ran
    assert (tmp_path / "out2" / "model_final.pt").is_file()


# --------------------------------------------------------------------------
# GOLDEN 4 — provenance round-trip: a trained checkpoint registers with its own
# embedded metrics + kind + sha256 + lineage back-reference.
# --------------------------------------------------------------------------

def test_golden_provenance_round_trip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # .tcip/experiments + .tcip/models live under cwd
    from tcip_mcp.experiments import (
        create_experiment,
        register_model_from_experiment,
        update_status,
    )
    from tcip_mcp.model_registry import ModelRegistry

    loader = _loader(tmp_path)
    out = tmp_path / "out"
    cfg = _cfg([{"freeze_to": -1, "epochs": 1}], seed=3)

    create_experiment("expA", cfg, data_source="imgs")
    update_status("expA", "running")
    run = create_run(cfg, str(out))
    train(run, loader, val_loader=loader, task="classification")
    update_status("expA", "completed")

    best_path = out / "model_best.pt"
    embedded = torch.load(best_path, weights_only=False)["metrics"]
    result = register_model_from_experiment("expA", str(best_path))

    # Metrics come from the checkpoint payload (the best epoch), never fabricated.
    assert result["metrics"]["epoch"] == embedded["epoch"]
    assert result["metrics"]["val_loss"] == pytest.approx(embedded["val_loss"])

    entry = ModelRegistry(".").get_model("expA")
    assert entry is not None
    assert entry["kind"] == KIND_TORCHVISION_COMPOSED   # round-tripped from the checkpoint
    assert entry["sha256"] and len(entry["sha256"]) == 64
    assert "experiment:expA" in entry["tags"]

    lineage = json.loads((tmp_path / ".tcip" / "experiments" / "expA" / "lineage.json").read_text())
    assert lineage["model_weights"] == str(best_path)
