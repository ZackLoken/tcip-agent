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


def test_weighted_ce_built():
    loss = build_loss("weighted_ce", class_distribution={0: 90, 1: 10}, num_classes=2)
    assert loss.ce.weight is not None  # class weight injected


def test_class_distribution_on_an_unweightable_loss_refuses():
    """Imbalance handling that silently vanishes is worse than a build that refuses."""
    with pytest.raises(ValueError, match="not weightable"):
        build_loss("dice", class_distribution={0: 90, 1: 10}, num_classes=2)


def test_combined_loss_routes_each_kwarg_to_the_terms_that_accept_it():
    """A per-term hyperparameter reaches its own term, not every term."""
    loss = build_loss("cross_entropy+dice", label_smoothing=0.1, smooth=2.0)
    ce, dice = loss.losses
    assert ce.ce.label_smoothing == 0.1
    assert dice.smooth == 2.0


def test_combined_loss_weights_only_the_weightable_term():
    loss = build_loss("cross_entropy+dice", class_distribution={0: 90, 1: 10}, num_classes=2)
    ce, _dice = loss.losses
    assert ce.ce.weight is not None  # dice never sees class_distribution, so it builds fine


def test_combined_loss_refuses_a_kwarg_no_term_accepts():
    with pytest.raises(ValueError, match="not accepted by any term"):
        build_loss("cross_entropy+dice", nonexistent_param=1)


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
    h = SemanticSegHead(in_channels=64, num_classes=3, class_weights=[1.0, 2.0, 3.0])
    out = h(torch.randn(2, 64, 16, 16))
    losses = h.compute_loss(out, {"masks": torch.randint(0, 3, (2, 16, 16))})
    assert "ce_loss" in losses and "dice_loss" in losses
    assert losses["ce_loss"].requires_grad and losses["dice_loss"].requires_grad


def test_semantic_seg_head_advertises_no_loss_choice():
    """The head welds CE + multi-class Dice; it must not accept a loss name it cannot honor.

    Previously `loss=` was accepted and silently discarded. There is no registry loss to route
    to — `build_loss("cross_entropy+dice")` raises at forward, because the registry's DiceLoss is
    binary while this head emits multi-class logits.
    """
    import pytest as _pytest
    from tcip_mcp.pipelines.components.heads import SemanticSegHead
    with _pytest.raises(TypeError):
        SemanticSegHead(in_channels=64, num_classes=3, loss="weighted_ce")
    assert SemanticSegHead.default_loss == ""


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


# --- Geometric transforms must keep masks aligned with image + boxes ------

def _instance_target(x1: int, y1: int, x2: int, y2: int, size: int = 64) -> dict:
    """One box with an exactly-matching [1, H, W] instance mask."""
    mask = torch.zeros((1, size, size), dtype=torch.uint8)
    mask[0, y1:y2, x1:x2] = 1
    return {
        "boxes": torch.tensor([[float(x1), float(y1), float(x2), float(y2)]]),
        "labels": torch.tensor([1]),
        "masks": mask,
    }


def _mask_bbox(mask: "torch.Tensor") -> list[float]:
    ys, xs = torch.nonzero(mask, as_tuple=True)
    return [float(xs.min()), float(ys.min()), float(xs.max()) + 1, float(ys.max()) + 1]


def test_horizontal_flip_flips_instance_masks_with_boxes():
    from tcip_mcp.pipelines.data.augmentations import RandomHorizontalFlip
    img = Image.new("RGB", (64, 64))
    out, t = RandomHorizontalFlip(p=1.0)(img, _instance_target(10, 10, 40, 40))
    assert t["masks"].shape == (1, 64, 64)
    assert _mask_bbox(t["masks"][0]) == t["boxes"][0].tolist() == [24.0, 10.0, 54.0, 40.0]


def test_vertical_flip_flips_semantic_mask():
    from tcip_mcp.pipelines.data.augmentations import RandomVerticalFlip
    img = Image.new("RGB", (64, 64))
    sem = torch.zeros((64, 64), dtype=torch.long)
    sem[:10, :] = 2  # class stripe at top
    out, t = RandomVerticalFlip(p=1.0)(img, {"masks": sem.clone()})
    assert torch.equal(t["masks"], torch.flip(sem, dims=[-2]))
    assert (t["masks"][-10:, :] == 2).all()


def test_resize_scales_masks_with_image_and_boxes():
    from tcip_mcp.pipelines.data.augmentations import Resize
    img = Image.new("RGB", (64, 64))
    out, t = Resize(size=(128, 128))(img, _instance_target(10, 10, 40, 40))
    assert out.size == (128, 128)
    assert t["masks"].shape == (1, 128, 128)
    assert _mask_bbox(t["masks"][0]) == t["boxes"][0].tolist() == [20.0, 20.0, 80.0, 80.0]

    # Semantic [H, W] mask resizes too (nearest keeps class indices intact)
    sem = torch.full((64, 64), 3, dtype=torch.long)
    _, t2 = Resize(size=(128, 96))(Image.new("RGB", (64, 64)), {"masks": sem})
    assert t2["masks"].shape == (96, 128)  # (h, w) for (w, h) size
    assert set(t2["masks"].unique().tolist()) == {3}


def test_random_resized_crop_keeps_masks_aligned():
    from tcip_mcp.pipelines.data.augmentations import RandomResizedCrop
    # Full-scale crop is deterministic: pure 2x upscale
    img = Image.new("RGB", (64, 64))
    crop = RandomResizedCrop(size=(128, 128), min_scale=1.0, max_scale=1.0)
    out, t = crop(img, _instance_target(10, 10, 40, 40))
    assert out.size == (128, 128)
    assert t["masks"].shape == (1, 128, 128)
    assert _mask_bbox(t["masks"][0]) == t["boxes"][0].tolist() == [20.0, 20.0, 80.0, 80.0]


def test_random_resized_crop_random_scale_masks_track_boxes():
    import random as _random
    from tcip_mcp.pipelines.data.augmentations import RandomResizedCrop
    _random.seed(0)
    crop = RandomResizedCrop(size=(64, 64), min_scale=0.5, max_scale=0.5)
    for _ in range(10):
        _, t = crop(Image.new("RGB", (64, 64)), _instance_target(10, 10, 40, 40))
        assert t["masks"].shape[0] == len(t["boxes"]) == len(t["labels"])
        assert t["masks"].shape[1:] == (64, 64)
        for box, mask in zip(t["boxes"], t["masks"]):
            assert mask.any()  # surviving box must have surviving mask pixels
            mb = _mask_bbox(mask)
            for a, b in zip(mb, box.tolist()):
                assert abs(a - b) <= 3  # nearest-vs-continuous rounding


def test_random_resized_crop_semantic_mask_follows_image():
    from tcip_mcp.pipelines.data.augmentations import RandomResizedCrop
    sem = torch.full((64, 64), 5, dtype=torch.long)
    crop = RandomResizedCrop(size=(32, 48), min_scale=0.5, max_scale=1.0)
    _, t = crop(Image.new("RGB", (64, 64)), {"masks": sem})
    assert t["masks"].shape == (48, 32)  # (h, w) for (w, h) size
    assert set(t["masks"].unique().tolist()) == {5}


def test_random_rotation_rotates_semantic_mask():
    import random as _random
    _random.seed(1)
    sem = torch.zeros((64, 64), dtype=torch.long)
    sem[:16, :] = 1  # asymmetric stripe
    out, t = RandomRotation(degrees=180, p=1.0)(Image.new("RGB", (64, 64)), {"masks": sem.clone()})
    assert t["masks"].shape == (64, 64)
    assert t["masks"].dtype == sem.dtype
    assert set(t["masks"].unique().tolist()) <= {0, 1}  # nearest preserves class ids
    assert not torch.equal(t["masks"], sem)  # actually rotated


# --------------------------------------------------------------------------
# HPO — Ray Tune: search algorithms + schedulers are agent-selectable
# --------------------------------------------------------------------------

def test_get_default_baseline_params_subset_of_space():
    from tcip_mcp.pipelines.training.hpo import get_default_baseline_params, get_default_space
    assert set(get_default_baseline_params()).issubset(set(get_default_space()))


def test_to_tune_space_maps_every_param_type():
    pytest.importorskip("ray")
    from ray import tune
    from tcip_mcp.pipelines.training.hpo import _to_tune_space
    space = _to_tune_space({
        "lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
        "wd": {"type": "uniform", "low": 0.0, "high": 0.1},
        "bs": {"type": "categorical", "choices": [2, 4]},
        "k": {"type": "int", "low": 1, "high": 3},
    })
    assert isinstance(space["lr"], tune.search.sample.Float)
    assert isinstance(space["bs"], tune.search.sample.Categorical)
    assert isinstance(space["k"], tune.search.sample.Integer)


def test_grid_mode_enumerates_discrete_axes():
    pytest.importorskip("ray")
    from tcip_mcp.pipelines.training.hpo import _to_tune_space
    space = _to_tune_space({"bs": {"type": "categorical", "choices": [2, 4, 8]}}, grid=True)
    # grid_search wraps a plain dict with a "grid_search" key, not a sampler.
    assert space["bs"] == {"grid_search": [2, 4, 8]}


def test_build_scheduler_aliases():
    pytest.importorskip("ray")
    from ray.tune.schedulers import (
        AsyncHyperBandScheduler, MedianStoppingRule, PopulationBasedTraining,
    )
    from ray import tune
    from tcip_mcp.pipelines.training.hpo import build_scheduler
    assert isinstance(build_scheduler("asha"), AsyncHyperBandScheduler)
    assert isinstance(build_scheduler("median"), MedianStoppingRule)
    # PBT mutates hyperparameters mid-training, so it needs the search space as mutations.
    assert isinstance(
        build_scheduler("pbt", hyperparam_mutations={"lr": tune.loguniform(1e-5, 1e-2)}),
        PopulationBasedTraining,
    )
    assert build_scheduler("none") is None


def test_build_search_alg_native_and_backend():
    pytest.importorskip("ray")
    from tcip_mcp.pipelines.training.hpo import build_search_alg
    # Native random/grid need no searcher (BasicVariantGenerator handles them).
    assert build_search_alg("random") is None
    assert build_search_alg("grid") is None
    # A backend the agent picks that isn't installed raises clearly (never silently swapped).
    with pytest.raises(ValueError, match="not installed"):
        build_search_alg("hebo")


def test_available_search_algs_lists_natives_and_installed_backends():
    pytest.importorskip("ray")
    from tcip_mcp.pipelines.training.hpo import available_search_algs
    algs = available_search_algs()
    assert "random" in algs and "grid" in algs
    pytest.importorskip("optuna")
    assert "optuna" in algs  # backend installed in this env


def test_tune_search_warm_start_and_optimizes(tmp_path):
    """End-to-end Ray Tune: a real sweep finds the minimum, honors warm_start, and reports
    each trial. Uses a pure-math objective so no training is needed."""
    pytest.importorskip("ray")

    def obj(config, report):
        report((config["x"] - 2.0) ** 2)

    from tcip_mcp.pipelines.training.hpo import tune_search
    result = tune_search(
        obj,
        param_space={"x": {"type": "uniform", "low": -5.0, "high": 5.0}},
        metric="objective", mode="min", num_samples=6,
        search_alg="random", scheduler="none",
        warm_start=True, baseline_params={"x": 2.0},
    )
    assert result["warm_start"] is True
    assert result["n_trials"] == 6
    assert result["search_alg"] == "random" and result["scheduler"] == "none"
    # The x=2.0 warm-start point is the exact minimum (objective 0.0).
    assert result["best_value"] == pytest.approx(0.0, abs=1e-9)
    assert result["best_params"]["x"] == pytest.approx(2.0)


def test_run_hpo_exposes_agent_search_choices_not_pinned():
    """run_hpo lets the agent choose search_alg + scheduler; the old Optuna/pruner pins
    are gone (capability-not-method)."""
    import inspect

    from tcip_mcp.tools.training_tools import run_hpo
    params = inspect.signature(run_hpo).parameters
    assert "search_alg" in params and "scheduler" in params
    assert "pruner" not in params and "direction" not in params
    assert params["search_alg"].default == "random"
    assert params["scheduler"].default == "asha"
