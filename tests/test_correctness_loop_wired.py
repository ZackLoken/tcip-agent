"""W4 — the correctness loop is wired to a sanctioned surface.

Before this, ``check_model_contract`` / ``overfit_check`` had zero production callers: a broken
bespoke builder wasted a full audited run. These prove the loop is now reachable and gating:
``preflight_config(smoke=True)`` builds + smokes at the RESOLVED dims and blocks a broken builder,
and ``TrainContext`` exposes the checks + craft primitives so a hand-rolled ``train(ctx)`` self-proves.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tcip_mcp.pipelines.model_build import resolve_contract_dims  # noqa: E402
from tcip_mcp.pipelines.training.envelope import TrainContext  # noqa: E402
from tcip_mcp.pipelines.training.generic_trainer import create_run  # noqa: E402


def _broken_builder(**kwargs):
    """An 'agent-written' builder that imports fine but fails the measurement contract: its
    eval-mode forward returns a bare tensor, not the list[dict] detection scorers consume."""
    import torch.nn as nn

    class _Broken(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, images, targets=None):
            if self.training:
                return {"loss": self.lin(torch.rand(1, 4)).sum()}
            return torch.rand(3)  # not list[dict] — violates the detection boundary

    return _Broken()


# --------------------------------------------------------------------------
# resolve_contract_dims — resolved, not the tiny 64px default (critic G5)
# --------------------------------------------------------------------------

def test_resolve_contract_dims_prefers_tile_edge_over_default():
    cfg = {
        "model_source": {"builder_kwargs": {"num_classes": 5, "in_chans": 4}, "task": "detection"},
        "data": {"tiling": {"enabled": True, "tile_size": 512}},
    }
    dims = resolve_contract_dims(cfg, "detection")
    assert dims == {"in_chans": 4, "num_classes": 5, "img_size": 512}


def test_resolve_contract_dims_falls_back_without_inventing():
    # No num_classes / in_chans / tile_size — resolved to safe fallbacks, never a hard fail.
    dims = resolve_contract_dims({"model_source": {}}, "detection")
    assert dims == {"in_chans": 3, "num_classes": 1, "img_size": 224}


# --------------------------------------------------------------------------
# preflight_config(smoke=True) — builds + smokes, blocks a broken builder
# --------------------------------------------------------------------------

def test_preflight_smoke_blocks_broken_builder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = tmp_path / "images"
    lbls = tmp_path / "labels"
    imgs.mkdir()
    lbls.mkdir()
    cfg = {
        "model_source": {"builder": f"{__name__}:_broken_builder", "task": "detection"},
        "data": {"images_dir": str(imgs), "labels_dir": str(lbls)},
        "training": {"batch_size": 1, "stages": [{"freeze_to": 0, "epochs": 1}]},
    }
    # Fast path (no smoke) is structurally valid — the builder imports fine.
    assert preflight_config(cfg)["valid"] is True
    # Smoke path builds + runs the contract and catches the measurement-boundary violation.
    r = preflight_config(cfg, smoke=True)
    assert r["valid"] is False
    assert any("model contract" in i for i in r["issues"])
    assert r["smoke"]["dims"]["img_size"] == 224  # resolved, not 64


def test_preflight_smoke_passes_valid_builder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = tmp_path / "images"
    lbls = tmp_path / "labels"
    imgs.mkdir()
    lbls.mkdir()
    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1, "min_size": 64, "max_size": 128},
                         "task": "detection"},
        "data": {"images_dir": str(imgs), "labels_dir": str(lbls)},
        "training": {"batch_size": 1, "stages": [{"freeze_to": 0, "epochs": 1}]},
    }
    r = preflight_config(cfg, smoke=True, overfit=True)
    assert r["valid"] is True, r["issues"]
    assert r["smoke"]["ok"] is True
    assert "overfit_check" in r  # voluntary diagnostic reported, non-gating


# --------------------------------------------------------------------------
# ctx surface — a custom loop self-proves + reuses the craft primitives
# --------------------------------------------------------------------------

def _ctx_for(task: str, builder: str, **builder_kwargs):
    config = {"model_source": {"builder": builder, "builder_kwargs": builder_kwargs, "task": task},
              "device": "cpu"}
    run = create_run(config, "out")
    return TrainContext(run=run, train_loader=None, val_loader=None, task=task)


def test_ctx_check_contract_and_overfit_check():
    ctx = _ctx_for("classification", "tests.bespoke_models:build_bespoke_classifier", num_classes=2)
    report = ctx.check_contract()
    assert report["ok"], report["issues"]
    # overfit is voluntary + non-gating; steps flow through as an override kwarg.
    over = ctx.overfit_check(steps=15)
    assert over["passed"], over["issue"]


def test_ctx_apply_stage_freeze_matches_trainer_guard():
    from tcip_mcp.pipelines.training.generic_trainer import apply_stage_freeze

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    ctx = TrainContext(run=create_run({}, "out"), train_loader=None, task="classification")
    full = ctx.apply_stage_freeze(model, 0)
    assert full == sum(p.numel() for p in model.parameters())
    # A shrink relative to the previous stage violates the monotonic guard.
    with pytest.raises(RuntimeError, match="Non-decreasing unfreeze"):
        apply_stage_freeze(model, 0, prev_trainable=full + 1)
