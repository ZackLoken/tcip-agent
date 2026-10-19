"""Sample run-config, assessment and ranking blocks.

They are a run's optimizer, schedule, stop rule and evaluation blocks, a data section's drawn
split and tiling blocks, a sample search space and search, and the sample settings of an
assessment and a failure-case ranking.
"""

from typing import Any

from tcip_mcp.pipelines.active_learning import ScorerSpec
from tcip_mcp.pipelines.execution import MaskCut, Stated

RATIONALE = "a sample reason the test states for this block"
"""The reason every sample block here states (``schemas`` requires one per stated block)."""

EQUIVALENCE_Z = 1.645
"""A sample equivalence multiplier an assessment states: a one-sided 95 percent test."""
DISPERSION_QUANTILE = 0.9
"""A sample dispersion quantile an assessment states: the ninetieth percentile."""
COUNT_TEST = {
    "equivalence_z": EQUIVALENCE_Z,
    "dispersion_quantile": DISPERSION_QUANTILE,
}
"""The sample count test settings, by ``operating_point.count_criterion`` parameter."""
COLLECTION_CONF = 0.01
"""A sample conf a detector assessment states its predictions are collected at."""


def collection_pass(
    **stated: object,
) -> Stated:
    """Return a ``Stated`` of :data:`COLLECTION_CONF` and :data:`RATIONALE`, ``stated`` over."""
    return Stated(
        **{"conf": COLLECTION_CONF, "rationale": RATIONALE, **stated}
    )


TRIAGE_WEIGHTS = {
    "shortfall_weight": 2.0,
    "surplus_weight": 1.0,
    "confidence_weight": 1.0,
}
"""Sample failure-case ranking weights, by door parameter."""
COMBINED_SCORER = ScorerSpec.model_validate(
    {"name": "combined", "uncertainty_weight": 0.6, "diversity_weight": 0.4}
)
"""A sample review-queue scorer block."""

SCHEDULE_SETTINGS: dict[str, dict[str, Any]] = {
    "cosine": {"eta_min": 1e-6, "horizon_epochs": 3},
    "plateau": {
        "factor": 0.3,
        "patience": 2,
        "threshold": 1e-4,
        "threshold_mode": "rel",
        "cooldown": 0,
        "min_lr": 0.0,
        "eps": 1e-8,
    },
    "onecycle": {
        "max_lr": 2e-2,
        "horizon_epochs": 3,
        "pct_start": 0.3,
        "anneal_strategy": "cos",
        "cycle_momentum": False,
        "base_momentum": 0.85,
        "max_momentum": 0.95,
        "div_factor": 25.0,
        "final_div_factor": 1e4,
        "three_phase": False,
    },
    "step": {"step_size": 3, "gamma": 0.5},
}
"""A sample of each scheduler type's own settings."""


def schedule(kind: str, **stated: object) -> dict:
    """Return a ``scheduler`` block for ``kind``.

    It merges ``type`` ``kind``, ``SCHEDULE_SETTINGS[kind]``, :data:`RATIONALE` and then
    ``stated``.
    """
    return {
        "type": kind,
        **SCHEDULE_SETTINGS[kind],
        "rationale": RATIONALE,
        **stated,
    }


def stop_rule(
    patience: int = 50, min_delta: float = 0.0, nonfinite_passes: int = 2
) -> dict:
    """Return an ``early_stopping`` block of its three arguments and :data:`RATIONALE`."""
    return {
        "patience": patience,
        "min_delta": min_delta,
        "nonfinite_passes": nonfinite_passes,
        "rationale": RATIONALE,
    }


def stage(freeze_to: int = 0, **stated: object) -> dict:
    """Return a ``stages`` entry at ``freeze_to`` stating :data:`RATIONALE`, ``stated`` over it."""
    return {"freeze_to": freeze_to, "rationale": RATIONALE, **stated}


def adamw_optimizer(**stated: object) -> dict:
    """Return a sample AdamW ``optimizer`` block, with ``stated`` over it."""
    return {
        "name": "adamw",
        "backbone_lr": 1e-4,
        "head_lr": 1e-3,
        "weight_decay": 0.0,
        "rationale": RATIONALE,
        **stated,
    }


def sgd_optimizer(**stated: object) -> dict:
    """Return a sample SGD ``optimizer`` block, its momentum stated, with ``stated`` over it."""
    return {
        "name": "sgd",
        "backbone_lr": 1e-3,
        "head_lr": 1e-2,
        "weight_decay": 0.0,
        "momentum": 0.9,
        "rationale": RATIONALE,
        **stated,
    }


def drawn_split(**stated: object) -> dict:
    """Return a drawn ``data.split`` block, and ``draw_splits``'s statement of a draw.

    It states seed 0, validation ratio 0.5, calibration and holdout ratios 0, stem grouping,
    foreground stratification and :data:`RATIONALE`, with ``stated`` merged over them.
    """
    return {
        "seed": 0,
        "val_ratio": 0.5,
        "calibration_ratio": 0.0,
        "holdout_ratio": 0.0,
        "grouping": "stem",
        "stratify_foreground": True,
        "rationale": RATIONALE,
        **stated,
    }


def tiling_block(**stated: object) -> dict:
    """Return a ``data.tiling`` block: ``skip_empty`` False, :data:`RATIONALE`, ``stated`` over."""
    return {"skip_empty": False, "rationale": RATIONALE, **stated}


def augmentation(**transforms: object) -> dict:
    """Return an ``augmentation`` block of :data:`RATIONALE` with ``transforms`` merged over it.

    Each of ``transforms`` maps a transform's name to its values.
    """
    return {"rationale": RATIONALE, **transforms}


def sampler(name: str, **values: object) -> dict:
    """Return a ``sampler`` block of ``name`` and :data:`RATIONALE`, ``values`` merged over them."""
    return {"name": name, "rationale": RATIONALE, **values}


VALIDATION_CONF = 0.35
"""A sample confidence a detector run's validation counts boxes at."""

MASK_BINARIZE = 0.5
"""A sample probability an instance segmentation pass or validation states its masks are cut
at."""

VERTEX_SPACING = 4.0
"""A sample ``vertex_spacing``: pixels of a mask ring's perimeter per vertex it is simplified to."""

MASK_CUT = MaskCut(mask_binarize=MASK_BINARIZE, vertex_spacing=VERTEX_SPACING)
"""The sample mask cut, at :data:`MASK_BINARIZE` and :data:`VERTEX_SPACING`."""


IOU_THRESHOLD = 0.5
"""A sample IoU a match must reach under the IoU convention."""


SCORE_WEIGHTS = {"loss": 0.5, "f1": 0.3, "map50": 0.2}
"""Sample weights a run selecting on the composite objective states for its three terms."""


def evaluation_block(**stated: object) -> dict:
    """Return an ``evaluation`` block, ``stated`` merged over its sample settings.

    The settings are ``selection_metric`` ``objective``, :data:`VALIDATION_CONF`, a copy of
    :data:`SCORE_WEIGHTS` and :data:`RATIONALE`.
    """
    return {
        "selection_metric": "objective",
        "conf_threshold": VALIDATION_CONF,
        "score_weights": dict(SCORE_WEIGHTS),
        "rationale": RATIONALE,
        **stated,
    }


def sweep_space() -> dict:
    """Return a fresh copy of the sample search space.

    The space is the head learning rate, sampled log-uniformly.
    """
    return {
        "optimizer.head_lr": {"type": "loguniform", "low": 1e-4, "high": 1e-2}
    }


def fifo_scheduler() -> dict:
    """Return a ``fifo`` trial ``scheduler`` block."""
    return {"name": "fifo"}


def fifo_search() -> dict:
    """Return a sweep's sample ``search_alg`` and ``scheduler``.

    They are random sampling and the ``fifo`` scheduler.
    """
    return {"search_alg": "random", "scheduler": fifo_scheduler()}


def asha_scheduler(**stated: object) -> dict:
    """Return a sample ``asha`` trial ``scheduler`` block, ``stated`` merged over its settings.

    The settings are ``time_attr``, ``max_t``, ``grace_period``, ``reduction_factor``,
    ``brackets`` and ``stop_last_trials``.
    """
    return {
        "name": "asha",
        "time_attr": "training_iteration",
        "max_t": 3,
        "grace_period": 1,
        "reduction_factor": 2,
        "brackets": 1,
        "stop_last_trials": True,
        **stated,
    }


def tune_arguments(**stated: object) -> dict:
    """Return sample keyword arguments for ``hpo.tune_search``, ``stated`` merged over them.

    They state metric ``objective`` at mode ``min``, one sample, random search, the ``fifo``
    scheduler, seed 0, one concurrent trial and one split draw.
    """
    return {
        "metric": "objective",
        "mode": "min",
        "num_samples": 1,
        "search_alg": "random",
        "scheduler": fifo_scheduler(),
        "seed": 0,
        "max_concurrent": 1,
        "baseline_params": None,
        "resources_per_trial": None,
        "split_draws": 1,
        **stated,
    }
