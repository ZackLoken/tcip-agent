"""W8 — imbalance losses + augmentation presets + HPO upgrades."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from tcip_mcp.pipelines.components.losses import (  # noqa: E402
    FocalLoss, build_loss, compute_class_weights,
)
from tcip_mcp.pipelines.data.augmentations import (  # noqa: E402
    RandomRotation, ToTensor, build_augmentation, get_augmentation_preset,
)


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------

def test_compute_class_weights_balanced():
    w = compute_class_weights({0: 90, 1: 10})
    assert len(w) == 2
    assert w[1] > w[0]  # rarer class up-weighted
    assert float(w.mean()) == pytest.approx(1.0, abs=1e-5)  # normalized over present


def test_focal_loss_scalar_alpha_unchanged():
    fl = FocalLoss(alpha=0.25, gamma=2.0)
    preds = torch.randn(8, 5, requires_grad=True)
    targets = torch.randint(0, 5, (8,))
    loss = fl(preds, targets)
    assert loss.ndim == 0
    loss.backward()


def test_focal_loss_class_weights():
    torch.manual_seed(0)
    w = compute_class_weights({0: 90, 1: 10, 2: 50})
    fl = FocalLoss(weight=w)
    preds = torch.randn(12, 3)
    targets = torch.randint(0, 3, (12,))
    loss = fl(preds.clone().requires_grad_(True), targets)
    assert loss.ndim == 0
    loss.backward()
    assert not torch.allclose(fl(preds, targets), FocalLoss()(preds, targets))  # weighting changes it


def test_weighted_ce_registered_and_built():
    from tcip_mcp.pipelines.registry import LOSSES
    assert "weighted_ce" in LOSSES
    loss = build_loss("weighted_ce", class_distribution={0: 90, 1: 10}, num_classes=2)
    assert loss.ce.weight is not None  # class weight injected


def test_classification_head_loss_optional():
    from tcip_mcp.pipelines.components.heads import ClassificationHead
    feats = torch.randn(4, 512)
    targets = {"labels": torch.randint(0, 5, (4,))}

    h = ClassificationHead(in_channels=512, num_classes=5, loss="focal")
    out = h(feats)
    assert h.compute_loss(out, targets)["cls_loss"].requires_grad

    h0 = ClassificationHead(in_channels=512, num_classes=5)
    out0 = h0(feats)
    assert torch.allclose(
        h0.compute_loss(out0, targets)["cls_loss"],
        F.cross_entropy(out0["logits"], targets["labels"]),
    )


def test_semantic_seg_head_weighted_ce():
    from tcip_mcp.pipelines.components.heads import SemanticSegHead
    h = SemanticSegHead(in_channels=64, num_classes=3, loss="weighted_ce", class_weights=[1.0, 2.0, 3.0])
    out = h(torch.randn(2, 64, 16, 16))
    losses = h.compute_loss(out, {"masks": torch.randint(0, 3, (2, 16, 16))})
    assert "ce_loss" in losses and "dice_loss" in losses
    assert losses["ce_loss"].requires_grad and losses["dice_loss"].requires_grad


# --------------------------------------------------------------------------
# Augmentation
# --------------------------------------------------------------------------

def test_nadir_preset_omits_mosaic_copypaste():
    p = get_augmentation_preset("nadir_rotation")
    assert {"rotation", "horizontal_flip", "vertical_flip"} <= set(p)
    assert "mosaic" not in p and "copy_paste" not in p and "mixup" not in p


def test_build_augmentation_from_preset_string():
    aug = build_augmentation("nadir_rotation")
    assert isinstance(aug.transforms[-1], ToTensor)
    assert any(isinstance(t, RandomRotation) for t in aug.transforms)


def test_random_rotation_detection_keeps_boxes_valid():
    torch.manual_seed(0)
    img = Image.new("RGB", (64, 64), (120, 120, 120))
    target = {"boxes": torch.tensor([[10.0, 10.0, 40.0, 40.0]]), "labels": torch.tensor([1])}
    out, t = RandomRotation(degrees=90, p=1.0)(img, target)
    assert out.size == (64, 64)
    boxes = t["boxes"]
    if len(boxes):
        assert (boxes[:, 2] > boxes[:, 0]).all() and (boxes[:, 3] > boxes[:, 1]).all()
        assert (boxes >= 0).all() and (boxes <= 64).all()
    assert len(t["labels"]) == len(boxes)


def test_random_rotation_classification_passthrough():
    img = Image.new("RGB", (64, 64), (100, 100, 100))
    out, t = RandomRotation(degrees=90, p=1.0)(img, {"labels": 3})
    assert out.size == (64, 64)
    assert t["labels"] == 3


# --------------------------------------------------------------------------
# HPO
# --------------------------------------------------------------------------

def test_get_default_baseline_params_subset_of_space():
    from tcip_mcp.pipelines.training.hpo import get_default_baseline_params, get_default_optuna_space
    assert set(get_default_baseline_params()).issubset(set(get_default_optuna_space()))


def test_build_pruner_asha():
    pytest.importorskip("optuna")
    from optuna.pruners import SuccessiveHalvingPruner, MedianPruner
    from tcip_mcp.pipelines.training.hpo import _build_pruner
    assert isinstance(_build_pruner("asha"), SuccessiveHalvingPruner)
    assert isinstance(_build_pruner("median"), MedianPruner)


def test_optuna_warm_start_trial0_is_baseline():
    pytest.importorskip("optuna")
    from tcip_mcp.pipelines.training.hpo import optuna_search
    space = {
        "lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
        "batch_size": {"type": "categorical", "choices": [2, 4, 8]},
    }
    baseline = {"lr": 3e-4, "batch_size": 4}
    result = optuna_search(
        lambda config, trial: config["lr"], param_space=space, n_trials=2,
        direction="minimize", warm_start=True, baseline_params=baseline,
    )
    assert result["warm_start"] is True
    assert result["all_trials"][0]["params"]["lr"] == pytest.approx(3e-4)
    assert result["all_trials"][0]["params"]["batch_size"] == 4


def test_run_hpo_still_importable():
    from tcip_mcp.tools.training_tools import run_hpo
    result = run_hpo({"model_spec": {"heads": [{"task": "detection"}]}}, n_trials=2, use_optuna=False)
    assert "configs" in result  # random-search path unaffected
