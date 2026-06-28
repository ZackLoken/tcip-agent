"""HPO — hyperparameter optimization with Optuna integration + TensorBoard logging.

Supports:
  - Random search (always available, no dependencies)
  - Optuna TPE/ASHA (optional, requires `pip install optuna`)
  - Per-trial TensorBoard logging (each trial gets its own subdirectory)
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

try:
    import optuna
    from optuna.pruners import MedianPruner
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


def random_search(
    param_space: dict[str, list[Any]],
    n_trials: int = 10,
    seed: int = 42,
) -> list[dict]:
    """Generate n_trials random configurations from a parameter space.

    Args:
        param_space: Dict mapping parameter names to lists of possible values.
        n_trials: Number of configurations to sample.
        seed: Random seed.

    Returns:
        List of config dicts.

    Example:
        param_space = {
            "lr": [1e-4, 3e-4, 1e-3, 3e-3],
            "head": ["faster_rcnn", "fcos"],
            "batch_size": [2, 4, 8],
        }
    """
    rng = random.Random(seed)
    configs = []
    for _ in range(n_trials):
        config = {k: rng.choice(v) for k, v in param_space.items()}
        configs.append(config)
    return configs


def validate_param_space(param_space: dict) -> list[str]:
    """Validate an HPO parameter space definition."""
    issues: list[str] = []
    if not isinstance(param_space, dict):
        issues.append("param_space must be a dict")
        return issues
    for key, values in param_space.items():
        if not isinstance(values, list) or len(values) == 0:
            issues.append(f"'{key}' must be a non-empty list of candidate values")
    return issues


def get_default_param_space() -> dict:
    """Return a reasonable default HPO search space for detection."""
    return {
        "lr": [1e-4, 3e-4, 5e-4, 1e-3],
        "batch_size": [2, 4],
        "head": ["faster_rcnn"],
        "weight_decay": [1e-4, 1e-3],
        "min_size": [640, 800],
    }


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
    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=5) if pruning else optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    def wrapped_objective(trial: optuna.Trial) -> float:
        config = {}
        for name, spec in param_space.items():
            config[name] = _suggest_param(trial, name, spec)
        logger.info("Trial %d: %s", trial.number, config)

        value = objective_fn(config, trial.number)

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
    }
    if tb_logdir:
        result["tensorboard_logdir"] = tb_logdir
    return result
