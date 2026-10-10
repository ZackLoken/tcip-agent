"""Imbalance losses, the augmentation chain a config states, and the HPO search."""

from __future__ import annotations

from unittest import mock

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402
from torch.nn import functional  # noqa: E402
from PIL import Image  # noqa: E402

from tcip_mcp.pipelines.components.losses import (  # noqa: E402
    FocalLoss, build_loss, compute_class_weights,
)
from tcip_mcp.pipelines.data.augmentations import (  # noqa: E402
    RandomRotation, ToTensor, build_augmentation,
)
from tests._training_values import asha_scheduler, tune_arguments  # noqa: E402


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------

def test_compute_class_weights_balanced():
    w = compute_class_weights({0: 90, 1: 10})
    assert len(w) == 2
    assert w[1] > w[0]  # rarer class up-weighted
    assert float(w.mean()) == pytest.approx(1.0, abs=1e-5)  # normalized over present


def test_class_weights_are_sized_by_the_one_class_count_derivation(monkeypatch):
    """A weight vector and a run's loaders are sized by one rule: the weigher asks the platform's
    own derivation for the count a distribution implies, so a distribution whose top id is above
    its own length weights every class the ground truth reaches rather than only those present."""
    from tcip_mcp.pipelines import derivations
    from tcip_mcp.pipelines.components import losses

    asked: list[dict] = []

    def _record(distribution):
        asked.append(dict(distribution))
        return derivations.num_classes_from_distribution(distribution)

    monkeypatch.setattr(losses, "num_classes_from_distribution", _record)

    w = compute_class_weights({0: 40, 4: 10})

    assert asked == [{0: 40, 4: 10}]  # the count came from the shared derivation
    assert len(w) == 5  # ids 0 and 4 present, 1 to 3 counted but unseen
    assert float(w[1]) == float(w[2]) == float(w[3]) > 0  # an unseen class is weighted, not dropped


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
    # weighting changes it
    assert not torch.allclose(fl(preds, targets), FocalLoss()(preds, targets))


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
        functional.cross_entropy(out0["logits"], targets["labels"]),
    )


def test_regression_head_loss_optional():
    from tcip_mcp.pipelines.components.heads import RegressionHead
    feats = torch.randn(4, 256)
    targets = {"values": torch.tensor([1.0, 2.5, 0.3, -1.2])}

    h = RegressionHead(in_channels=256, loss="huber")
    out = h(feats)
    assert h.compute_loss(out, targets)["reg_loss"].requires_grad

    h0 = RegressionHead(in_channels=256)
    out0 = h0(feats)
    assert torch.allclose(
        h0.compute_loss(out0, targets)["reg_loss"],
        functional.smooth_l1_loss(out0["values"], targets["values"]),
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

    There is no registry loss to route a name to: `build_loss("cross_entropy+dice")` raises at
    forward, since the registry's DiceLoss is binary while this head emits multi-class logits.
    """
    import pytest as _pytest
    from tcip_mcp.pipelines.components.heads import SemanticSegHead
    with _pytest.raises(TypeError, match="'loss'"):
        SemanticSegHead(in_channels=64, num_classes=3, loss="weighted_ce")  # type: ignore[call-arg]  # the refused kwarg is the subject; the raises pins it to loss


# --------------------------------------------------------------------------
# Augmentation
# --------------------------------------------------------------------------

EVERY_TRANSFORM: dict[str, dict] = {
    "rotation": {"degrees": 180, "p": 1.0},
    "horizontal_flip": {"p": 0.5},
    "vertical_flip": {"p": 0.5},
    "color_jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2},
    "random_crop": {"size": [64, 64], "min_scale": 0.5, "max_scale": 1.0},
    "gaussian_blur": {"p": 0.1, "radius": 2.0},
    "resize": {"size": [64, 64]},
}
"""A sample augmentation config naming every registered transform, each entry stating every
value its transform takes."""


def test_build_augmentation_builds_each_transform_at_the_values_its_entry_states():
    """guard. Every entry of the dict form builds its transform, in the config's order, holding
    exactly the values the entry states, the chain ending in ToTensor."""
    chain = build_augmentation(EVERY_TRANSFORM).transforms

    assert [vars(t) for t in chain[:-1]] == list(EVERY_TRANSFORM.values())
    assert isinstance(chain[-1], ToTensor)


@pytest.mark.parametrize("config, named", [
    ({"horizontal_flip": True}, "'horizontal_flip'"),
    ({"horizontal_flip": 0.5}, "'horizontal_flip'"),
    ({"resize": [640, 640]}, "'resize'"),
    ("nadir_rotation", "'nadir_rotation'"),
    ({"color_jitter": {"hue": 0.1}}, "'color_jitter'"),
    ({"color_jitter": {**EVERY_TRANSFORM["color_jitter"], "hue": 0.1}}, "'hue'"),
    ({"rotation": {"degrees": 90}}, "'rotation'"),
    ({"horizontal_flip": {"p": 0.5}, "mosaic": {}}, "mosaic"),
], ids=["true", "a-number", "a-list", "a-preset-name", "hue-alone", "hue-beside-the-rest",
        "a-value-unstated", "an-unknown-transform"])
def test_build_augmentation_refuses_an_entry_not_stating_its_values_by_name(config, named):
    """guard. A shorthand (True, a number, a list), a preset name, a value the transform does not
    take (hue) and a value it takes left unstated each refuse naming the entry, and an unknown
    transform refuses by its name: nothing in the chain is a value the config did not state."""
    with pytest.raises(ValueError, match=named):
        build_augmentation(config)


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
    from tests._producer_fixtures import painted_array

    mask = torch.as_tensor(painted_array(size, size, [((x1, y1, x2, y2), 1)]))[None]
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
    from tests._producer_fixtures import painted_array
    img = Image.new("RGB", (64, 64))
    sem = torch.as_tensor(painted_array(64, 64, [((0, 0, 64, 10), 2)])).long()  # top stripe
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

    from tests._producer_fixtures import painted_array
    _random.seed(1)
    sem = torch.as_tensor(painted_array(64, 64, [((0, 0, 64, 16), 1)])).long()  # asymmetric
    out, t = RandomRotation(degrees=180, p=1.0)(Image.new("RGB", (64, 64)), {"masks": sem.clone()})
    assert t["masks"].shape == (64, 64)
    assert t["masks"].dtype == sem.dtype
    assert set(t["masks"].unique().tolist()) <= {0, 1}  # nearest preserves class ids
    assert not torch.equal(t["masks"], sem)  # actually rotated


# --------------------------------------------------------------------------
# HPO: Ray Tune search algorithms + schedulers are agent-selectable
# --------------------------------------------------------------------------

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


def test_tune_search_normalizes_search_alg_case_before_deciding_grid(tmp_path, monkeypatch):
    """tune_search reads a normalized local, not the caller's own casing, at the point it
    decides whether _to_tune_space builds a grid space: search_alg="Grid" (the agent's own
    casing, not this module's lower-cased spelling) still reaches the grid branch. Stopped
    right after _to_tune_space records its own grid keyword, before Ray is ever touched."""
    pytest.importorskip("ray")
    import tcip_mcp.pipelines.training.hpo as hpo

    class _StoppedAfterSpaceError(Exception):
        pass

    captured: dict = {}

    def fake_to_tune_space(param_space, grid=False, grid_keys=frozenset()):
        captured["grid"] = grid
        raise _StoppedAfterSpaceError

    monkeypatch.setattr(hpo, "_to_tune_space", fake_to_tune_space)

    with pytest.raises(_StoppedAfterSpaceError):
        hpo.tune_search(
            objective_fn=lambda config, report: None,
            param_space={"bs": {"type": "categorical", "choices": [2, 4]}},
            sweep_dir=tmp_path / "sweep", **tune_arguments(search_alg="Grid"))

    assert captured["grid"] is True


PBT_SCHEDULER = {
    "name": "pbt", "time_attr": "training_iteration", "perturbation_interval": 2,
    "burn_in_period": 0, "quantile_fraction": 0.25, "resample_probability": 0.25,
    "perturbation_factors": [1.2, 0.8], "custom_explore_fn": None, "log_config": True,
    "require_attrs": True, "synch": False,
}
MEDIAN_SCHEDULER = {"name": "median", "time_attr": "training_iteration", "grace_period": 2,
                    "min_samples_required": 3, "min_time_slice": 0, "hard_stop": True}
HYPERBAND_SCHEDULER = {"name": "hyperband", "time_attr": "training_iteration", "max_t": 9,
                       "reduction_factor": 3, "stop_last_trials": True}
"""Sample trial scheduler blocks, each stating every setting its Ray class takes."""


def test_build_scheduler_builds_each_scheduler_at_the_settings_its_block_states():
    """guard. Each scheduler is Ray's own class built at its block's settings, max_t among them
    (never Ray's default cap), pbt mutating over the space it is handed, fifo pruning nothing."""
    pytest.importorskip("ray")
    from ray import tune
    from ray.tune.schedulers import (
        AsyncHyperBandScheduler, FIFOScheduler, HyperBandScheduler, MedianStoppingRule,
        PopulationBasedTraining,
    )
    from tcip_mcp.pipelines.training.hpo import build_scheduler

    space = {"lr": tune.loguniform(1e-5, 1e-2)}
    asha = build_scheduler(asha_scheduler(max_t=7), hyperparam_mutations=space)
    assert isinstance(asha, AsyncHyperBandScheduler) and asha._max_t == 7
    hyperband = build_scheduler(HYPERBAND_SCHEDULER, hyperparam_mutations=space)
    assert isinstance(hyperband, HyperBandScheduler) and hyperband._max_t_attr == 9
    assert isinstance(build_scheduler(MEDIAN_SCHEDULER, hyperparam_mutations=space),
                      MedianStoppingRule)
    pbt = build_scheduler(PBT_SCHEDULER, hyperparam_mutations=space)
    assert isinstance(pbt, PopulationBasedTraining) and pbt._hyperparam_mutations == space
    assert type(build_scheduler({"name": "fifo"}, hyperparam_mutations=space)) is FIFOScheduler


@pytest.mark.parametrize("scheduler, named", [
    ({k: v for k, v in asha_scheduler().items() if k != "max_t"}, "unstated ['max_t']"),
    ({k: v for k, v in asha_scheduler().items() if k != "grace_period"},
     "unstated ['grace_period']"),
    ({**HYPERBAND_SCHEDULER, "grace_period": 2}, "not taken ['grace_period']"),
    ({"name": "fifo", "max_t": 5}, "not taken ['max_t']"),
    ({"name": "none"}, "Unknown scheduler 'none'"),
    ({"max_t": 5}, "Unknown scheduler ''"),
], ids=["asha-without-max_t", "asha-without-grace_period", "hyperband-with-grace_period",
        "fifo-with-max_t", "an-unknown-name", "no-name"])
def test_scheduler_settings_refuses_a_setting_unstated_or_not_taken_by_name(scheduler, named):
    """guard. A setting its scheduler's Ray class takes and the block leaves unstated (so Ray
    would supply it), and one the class does not take (so Ray would drop it), each refuse by
    name, as does a name no scheduler carries."""
    pytest.importorskip("ray")
    import re

    from tcip_mcp.pipelines.training.hpo import scheduler_settings

    with pytest.raises(ValueError, match=re.escape(named)):
        scheduler_settings(scheduler)


def _first_sampled_lr(searcher, storage_path) -> float:
    """The ``lr`` of the first trial ``searcher`` samples from a one-axis continuous space."""
    from ray import tune
    from ray.tune.experiment import Experiment

    searcher.add_configurations(Experiment(
        name="seed_probe", run=lambda config: None,
        config={"lr": tune.loguniform(1e-5, 1e-2)}, num_samples=1, storage_path=str(storage_path)))
    return searcher.next_trial().config["lr"]


def test_the_native_sampler_draws_the_sweep_seed_s_points(tmp_path):
    pytest.importorskip("ray")
    from tcip_mcp.pipelines.training.hpo import build_search_alg

    first = _first_sampled_lr(build_search_alg("random", seed=7), tmp_path / "a")
    again = _first_sampled_lr(build_search_alg("random", seed=7), tmp_path / "b")
    other = _first_sampled_lr(build_search_alg("grid", seed=8), tmp_path / "c")
    assert first == again
    assert first != other


def test_every_backend_searcher_retains_the_sweep_seed():
    """Each real backend searcher holds the stated seed where its own sampler reads it: optuna's
    ``_seed``, hyperopt's ``rstate`` (a ``RandomState`` drawing what one seeded with the same
    value draws), bayesopt's ``_random_state``."""
    pytest.importorskip("ray")
    import numpy as np

    from tcip_mcp.pipelines.training.hpo import build_search_alg

    points = [{"lr": 1e-3}]
    optuna = build_search_alg("optuna", seed=713, points_to_evaluate=points)
    hyperopt = build_search_alg("hyperopt", seed=713, points_to_evaluate=points)
    bayesopt = build_search_alg("bayesopt", seed=713, points_to_evaluate=points)

    assert optuna._seed == 713
    assert hyperopt.rstate.randint(2**31 - 1) == np.random.RandomState(713).randint(2**31 - 1)
    assert bayesopt._random_state == 713


@pytest.mark.parametrize("name", ["nevergrad", "ax"])
def test_a_searcher_this_platform_cannot_seed_or_run_is_refused_by_name(name):
    pytest.importorskip("ray")
    from tcip_mcp.pipelines.training.hpo import build_search_alg

    with pytest.raises(ValueError, match=f"Unknown search_alg '{name}'"):
        build_search_alg(name, seed=0)


def test_a_blank_search_alg_refuses_naming_the_vocabulary():
    pytest.importorskip("ray")
    from tcip_mcp.pipelines.training.hpo import build_search_alg

    with pytest.raises(ValueError, match="Unknown search_alg ''.*random"):
        build_search_alg("", seed=0)


def test_build_search_alg_native_and_backend():
    pytest.importorskip("ray")
    from ray.tune.search.basic_variant import BasicVariantGenerator

    from tcip_mcp.pipelines.training.hpo import build_search_alg

    assert isinstance(build_search_alg("random", seed=0), BasicVariantGenerator)
    assert isinstance(build_search_alg("grid", seed=0), BasicVariantGenerator)
    # A backend the agent picks that isn't installed raises clearly (never silently swapped).
    # Absence is simulated rather than relying on a package that happens to be missing: every
    # offered backend now installs by default, so nothing real is left to stand in for one.
    import tcip_mcp.pipelines.training.hpo as hpo_mod

    with mock.patch.object(hpo_mod, "find_spec", return_value=None):
        with pytest.raises(ValueError, match="not installed"):
            build_search_alg("optuna", seed=0)


def test_available_search_algs_lists_natives_and_installed_backends():
    pytest.importorskip("ray")
    from tcip_mcp.pipelines.training.hpo import available_search_algs
    algs = available_search_algs()
    assert "random" in algs and "grid" in algs
    pytest.importorskip("optuna")
    assert "optuna" in algs  # backend installed in this env


@pytest.mark.ray_cluster
def test_tune_search_evaluates_its_baseline_and_optimizes(tmp_path):
    """End-to-end Ray Tune: a real sweep runs every trial and evaluates the baseline point.
    Uses a pure-math objective, each trial writing the point it trained, so no training is
    needed."""
    pytest.importorskip("ray")
    import json
    import uuid

    seen = tmp_path / "seen"
    seen.mkdir()

    def obj(config, report):
        (seen / f"{uuid.uuid4().hex}.json").write_text(json.dumps(config["x"]), encoding="utf-8")
        report((config["x"] - 2.0) ** 2)

    from tcip_mcp.pipelines.training.hpo import tune_search
    tune_search(
        obj,
        param_space={"x": {"type": "uniform", "low": -5.0, "high": 5.0}},
        sweep_dir=tmp_path / "hpo" / "sweep",
        **tune_arguments(num_samples=6, baseline_params={"x": 2.0}),
    )
    points = [json.loads(p.read_text(encoding="utf-8")) for p in seen.iterdir()]
    assert len(points) == 6
    assert 2.0 in points  # the baseline point, the exact minimum, was trained


def test_run_hyperparameter_search_exposes_agent_search_choices_not_pinned():
    """guard. run_hyperparameter_search takes the trial count, the search algorithm and the
    scheduler block as required arguments with no default; it takes no ``pruner``,
    ``direction`` or separate halving setting, and the search space is the caller's to state, no
    default space or default baseline shipping beside it."""
    import inspect

    from tcip_mcp.pipelines.training import hpo
    from tcip_mcp.tools.training_tools import run_hyperparameter_search
    params = inspect.signature(run_hyperparameter_search).parameters
    assert "search_alg" in params and "scheduler" in params
    assert "pruner" not in params and "direction" not in params
    assert params["param_space"].default is inspect.Parameter.empty
    assert "warm_start" not in params
    assert not hasattr(hpo, "get_default_space")
    assert not hasattr(hpo, "get_default_baseline_params")
    for name in ("n_trials", "search_alg", "scheduler", "search_seed"):
        assert params[name].default is inspect.Parameter.empty, name
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, name
    assert not {"grace_period", "reduction_factor", "max_t"} & set(params)
