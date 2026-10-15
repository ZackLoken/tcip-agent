"""The sample blocks a test config states, since the platform ships no default for any of them:
the optimizer, schedule, stop rule and evaluation blocks ``_chain_fixtures.training_config``
builds its run from (and a test states over it as an override), the drawn split and tiling
blocks a data section states, and a sample search space and search; beside them the sample
settings an assessment and a failure-case ranking state.
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
"""A sample conf a detector assessment states its predictions are collected at, low enough that
hesitant detections survive to be swept."""


def collection_pass(**stated: Any) -> Stated:
    """A detector assessment's execution statement: its predictions collected at
    :data:`COLLECTION_CONF`, stating :data:`RATIONALE`, with ``stated`` over it.
    """
    return Stated(
        **{"conf": COLLECTION_CONF, "rationale": RATIONALE, **stated}
    )


TRIAGE_WEIGHTS = {
    "shortfall_weight": 2.0,
    "surplus_weight": 1.0,
    "confidence_weight": 1.0,
}
"""Sample failure-case ranking weights a test states, by door parameter."""
COMBINED_SCORER = ScorerSpec.model_validate(
    {"name": "combined", "uncertainty_weight": 0.6, "diversity_weight": 0.4}
)
"""A sample review-queue scorer block a test states."""

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


def schedule(kind: str, **stated) -> dict:
    """A ``scheduler`` block of type ``kind`` stating its own sample settings
    (:data:`SCHEDULE_SETTINGS`) and :data:`RATIONALE`, with ``stated`` over them.
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
    """An ``early_stopping`` block; the sample patience sits past the horizons
    :data:`SCHEDULE_SETTINGS` states, so a sample stage under a horizon-bound schedule ends at
    its horizon unless the test states a longer one.
    """
    return {
        "patience": patience,
        "min_delta": min_delta,
        "nonfinite_passes": nonfinite_passes,
        "rationale": RATIONALE,
    }


def stage(freeze_to: int = 0, **stated) -> dict:
    """A ``stages`` entry at ``freeze_to`` stating :data:`RATIONALE`, with ``stated`` over it."""
    return {"freeze_to": freeze_to, "rationale": RATIONALE, **stated}


def adamw_optimizer(**stated) -> dict:
    """A sample AdamW ``optimizer`` block, with ``stated`` over it."""
    return {
        "name": "adamw",
        "backbone_lr": 1e-4,
        "head_lr": 1e-3,
        "weight_decay": 0.0,
        "rationale": RATIONALE,
        **stated,
    }


def sgd_optimizer(**stated) -> dict:
    """A sample SGD ``optimizer`` block, its momentum stated, with ``stated`` over it."""
    return {
        "name": "sgd",
        "backbone_lr": 1e-3,
        "head_lr": 1e-2,
        "weight_decay": 0.0,
        "momentum": 0.9,
        "rationale": RATIONALE,
        **stated,
    }


def drawn_split(**stated) -> dict:
    """A drawn ``data.split`` block, and ``draw_splits``'s statement of a draw: half the members
    on the validation side, no reserved side, one group per member, balanced by foreground count,
    at seed 0, with ``stated`` over it.
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


def tiling_block(**stated) -> dict:
    """A ``data.tiling`` block keeping object-free tiles, with ``stated`` over it."""
    return {"skip_empty": False, "rationale": RATIONALE, **stated}


def augmentation(**transforms) -> dict:
    """An ``augmentation`` block of ``transforms``, each a transform's name to its values,
    stating :data:`RATIONALE`.
    """
    return {"rationale": RATIONALE, **transforms}


def sampler(name: str, **values) -> dict:
    """A ``sampler`` block naming ``name`` at ``values``, stating :data:`RATIONALE`."""
    return {"name": name, "rationale": RATIONALE, **values}


VALIDATION_CONF = 0.35
"""A sample confidence a detector run's validation counts boxes at."""

MASK_BINARIZE = 0.5
"""A sample probability an instance segmentation pass or validation states its masks are cut
at."""

VERTEX_SPACING = 4.0
"""A sample number of pixels of each ring's perimeter per vertex an instance segmentation pass,
validation or COCO import states its mask rings are simplified to."""

MASK_CUT = MaskCut(mask_binarize=MASK_BINARIZE, vertex_spacing=VERTEX_SPACING)
"""The sample mask cut an instance segmentation pass or validation states."""


IOU_THRESHOLD = 0.5
"""A sample IoU a match must reach under the IoU convention, handed to ``evaluate`` directly."""


SCORE_WEIGHTS = {"loss": 0.5, "f1": 0.3, "map50": 0.2}
"""Sample weights a run selecting on the composite objective states for its three terms."""


def evaluation_block(**stated) -> dict:
    """An ``evaluation`` block selecting on the composite objective at :data:`VALIDATION_CONF`,
    weighting it by :data:`SCORE_WEIGHTS`, with ``stated`` over it.
    """
    return {
        "selection_metric": "objective",
        "conf_threshold": VALIDATION_CONF,
        "score_weights": dict(SCORE_WEIGHTS),
        "rationale": RATIONALE,
        **stated,
    }


def sweep_space() -> dict:
    """A fresh copy of the sample search space: the head learning rate, sampled log-uniformly."""
    return {
        "optimizer.head_lr": {"type": "loguniform", "low": 1e-4, "high": 1e-2}
    }


def fifo_scheduler() -> dict:
    """A trial ``scheduler`` block that runs every trial to completion."""
    return {"name": "fifo"}


def fifo_search() -> dict:
    """A sweep's sample ``search_alg`` and ``scheduler``: random sampling, every trial run to
    completion.
    """
    return {"search_alg": "random", "scheduler": fifo_scheduler()}


def asha_scheduler(**stated) -> dict:
    """A sample ``asha`` trial ``scheduler`` block stating every setting its Ray class takes,
    with ``stated`` over them.
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


def tune_arguments(**stated) -> dict:
    """``hpo.tune_search``'s arguments for a sample one-trial random search scheduling nothing
    and seeking the lowest ``objective``, with ``stated`` over them; ``objective_fn``,
    ``param_space`` and ``sweep_dir`` are the caller's own.
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
