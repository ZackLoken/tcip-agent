"""HPO — hyperparameter optimization with Optuna integration + TensorBoard logging.

Supports:
  - Optuna TPE/ASHA search that trains each trial (requires `pip install optuna`)
  - Per-trial TensorBoard logging (each trial gets its own subdirectory)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

try:
    import optuna
    from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, HyperbandPruner
    from optuna.samplers import TPESampler

    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    SummaryWriter = None  # type: ignore[misc,assignment]
    HAS_TB = False


# ---------------------------------------------------------------------------
# Optuna integration
# ---------------------------------------------------------------------------

def get_default_optuna_space() -> dict:
    """Return Optuna-compatible search space definition.

    Each key maps to a dict with 'type' and parameters:
      - 'categorical': {'choices': [...]}
      - 'loguniform': {'low': float, 'high': float}
      - 'uniform': {'low': float, 'high': float}
      - 'int': {'low': int, 'high': int}
    """
    return {
        "lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
        "batch_size": {"type": "categorical", "choices": [2, 4, 8]},
        "head": {"type": "categorical", "choices": ["faster_rcnn", "fcos", "retinanet"]},
        "weight_decay": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
        "backbone": {"type": "categorical", "choices": ["resnet50", "resnet101"]},
    }


def get_default_baseline_params() -> dict:
    """A known-good baseline to warm-start trial 0 (subset of the default space)."""
    return {
        "lr": 3e-4,
        "batch_size": 4,
        "weight_decay": 1e-4,
        "head": "faster_rcnn",
        "backbone": "resnet50",
    }


def _build_pruner(
    name: str = "asha", *, grace_period: int = 5, reduction_factor: int = 3,
    n_startup_trials: int = 3, n_warmup_steps: int = 5,
):
    """Build an Optuna pruner. ``asha``/``successive_halving`` -> ASHA;
    ``hyperband`` -> Hyperband; ``median`` -> MedianPruner; else/``none`` -> NopPruner.

    Switching the default to ASHA is safe: with no intermediate ``trial.report``
    calls ``SuccessiveHalvingPruner`` never prunes.
    """
    name = (name or "none").lower()
    if name in ("asha", "successive_halving"):
        return SuccessiveHalvingPruner(min_resource=grace_period, reduction_factor=reduction_factor)
    if name == "hyperband":
        return HyperbandPruner(min_resource=grace_period, reduction_factor=reduction_factor)
    if name == "median":
        return MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps)
    return optuna.pruners.NopPruner()


def _suggest_param(trial: Any, name: str, spec: dict) -> Any:
    """Suggest a single parameter from an Optuna trial."""
    ptype = spec["type"]
    if ptype == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    elif ptype == "loguniform":
        return trial.suggest_float(name, spec["low"], spec["high"], log=True)
    elif ptype == "uniform":
        return trial.suggest_float(name, spec["low"], spec["high"])
    elif ptype == "int":
        return trial.suggest_int(name, spec["low"], spec["high"])
    else:
        raise ValueError(f"Unknown param type: {ptype}")


def optuna_search(
    objective_fn: Callable[..., float],
    param_space: dict | None = None,
    n_trials: int = 20,
    direction: str = "maximize",
    study_name: str = "tcip_hpo",
    storage: str | None = None,
    seed: int = 42,
    pruning: bool = True,
    pruner: str = "asha",
    grace_period: int = 5,
    reduction_factor: int = 3,
    warm_start: bool = False,
    baseline_params: dict | None = None,
    tb_logdir: str | None = None,
) -> dict:
    """Run HPO using Optuna with TPE sampler and optional ASHA pruning.

    Args:
        objective_fn: Callable taking (config dict, trial number) and returning a scalar metric.
                     For detection, this should return val mAP50.
        param_space: Optuna-compatible space dict (see get_default_optuna_space).
                    If None, uses default space.
        n_trials: Number of trials to run.
        direction: 'maximize' (for mAP) or 'minimize' (for loss).
        study_name: Name of the Optuna study.
        storage: Optional database URL for persistent study (e.g. 'sqlite:///hpo.db').
        seed: Random seed for reproducibility.
        pruning: Whether to use MedianPruner for early stopping of poor trials.
        tb_logdir: Optional base directory for TensorBoard logs. Each trial writes
                  to tb_logdir/trial_{n}/. If None, TensorBoard logging is skipped.

    Returns:
        Dict with 'best_params', 'best_value', 'n_trials', 'study_name',
        'all_trials', and 'tensorboard_logdir' (if TB logging enabled).

    Raises:
        ImportError: If optuna is not installed.
    """
    if not HAS_OPTUNA:
        raise ImportError(
            "Optuna is required for optuna_search. Install with: pip install optuna"
        )

    if param_space is None:
        param_space = get_default_optuna_space()

    sampler = TPESampler(seed=seed)
    pruner_obj = _build_pruner(
        pruner, grace_period=grace_period, reduction_factor=reduction_factor,
    ) if pruning else optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner_obj,
        storage=storage,
        load_if_exists=True,
    )

    # Warm-start trial 0 with a known-good baseline (filtered to the search space).
    if warm_start:
        baseline = baseline_params or get_default_baseline_params()
        enqueued = {k: v for k, v in baseline.items() if k in param_space}
        if enqueued:
            study.enqueue_trial(enqueued)

    def wrapped_objective(trial: optuna.Trial) -> float:
        config = {}
        for name, spec in param_space.items():
            config[name] = _suggest_param(trial, name, spec)
        logger.info("Trial %d: %s", trial.number, config)

        # Pass the trial (not just its number) so the objective can report
        # intermediate values for ASHA pruning (trial.report / should_prune).
        value = objective_fn(config, trial)

        # Log trial result to TensorBoard
        if tb_logdir and HAS_TB:
            trial_dir = str(Path(tb_logdir) / f"trial_{trial.number}")
            writer = SummaryWriter(log_dir=trial_dir)
            # Log each param as a scalar at step=trial.number
            for pname, pval in config.items():
                if isinstance(pval, (int, float)):
                    writer.add_scalar(f"hpo/params/{pname}", pval, trial.number)
            writer.add_scalar("hpo/objective", value, trial.number)
            # Log params as hparams for TensorBoard HParams plugin
            writer.add_hparams(
                {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))},
                {"hpo/objective": value},
            )
            writer.close()

        return value

    study.optimize(wrapped_objective, n_trials=n_trials)

    all_trials = []
    for t in study.trials:
        all_trials.append({
            "number": t.number,
            "params": t.params,
            "value": t.value,
            "state": str(t.state),
        })

    result = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
        "study_name": study_name,
        "all_trials": all_trials,
        "pruner": pruner,
        "warm_start": warm_start,
        "baseline_params": (baseline_params or get_default_baseline_params()) if warm_start else None,
    }
    if tb_logdir:
        result["tensorboard_logdir"] = tb_logdir
    return result
