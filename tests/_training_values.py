"""The sample blocks a test config states, since the platform ships no default for any of them:
the optimizer, schedule, stop rule and evaluation blocks ``_chain_fixtures.training_config``
builds its run from (and a test states over it as an override), and a sample search space and
search."""

SCHEDULE_SETTINGS = {
    "cosine": {"eta_min": 1e-6, "horizon_epochs": 3},
    "plateau": {"factor": 0.3, "patience": 2},
    "onecycle": {"max_lr": 2e-2, "horizon_epochs": 3},
    "step": {"step_size": 3, "gamma": 0.5},
}
"""A sample of each scheduler type's own settings."""


def schedule(kind: str, **stated) -> dict:
    """A ``scheduler`` block of type ``kind`` stating its own sample settings
    (:data:`SCHEDULE_SETTINGS`), with ``stated`` over them."""
    return {"type": kind, **SCHEDULE_SETTINGS[kind], **stated}


def stop_rule(patience: int = 50, min_delta: float = 0.0) -> dict:
    """An ``early_stopping`` block; the default patience sits past the horizons
    :data:`SCHEDULE_SETTINGS` states, so a sample stage under a horizon-bound schedule ends at
    its horizon unless the test states a longer one."""
    return {"patience": patience, "min_delta": min_delta}


def adamw_optimizer() -> dict:
    """A sample AdamW ``optimizer`` block."""
    return {"name": "adamw", "backbone_lr": 1e-4, "head_lr": 1e-3, "weight_decay": 0.0}


def sgd_optimizer() -> dict:
    """A sample SGD ``optimizer`` block, its momentum stated."""
    return {"name": "sgd", "backbone_lr": 1e-3, "head_lr": 1e-2, "weight_decay": 0.0,
            "momentum": 0.9}


VALIDATION_CONF = 0.35
"""A sample confidence a detector run's validation counts boxes at."""


def evaluation_block(**stated) -> dict:
    """An ``evaluation`` block at :data:`VALIDATION_CONF`, with ``stated`` over it."""
    return {"conf_threshold": VALIDATION_CONF, **stated}


def sweep_space() -> dict:
    """A fresh copy of the sample search space: the head learning rate, sampled log-uniformly."""
    return {"optimizer.head_lr": {"type": "loguniform", "low": 1e-4, "high": 1e-2}}


def tune_arguments(**stated) -> dict:
    """``hpo.tune_search``'s arguments for a sample one-trial random search scheduling nothing
    and seeking the lowest ``objective``, with ``stated`` over them; ``objective_fn``,
    ``param_space`` and ``sweep_dir`` are the caller's own."""
    return {"metric": "objective", "mode": "min", "num_samples": 1, "search_alg": "random",
            "scheduler": None, "grace_period": 1, "reduction_factor": 2, "seed": 0,
            "max_concurrent": 1, "baseline_params": None, "resources_per_trial": None,
            "split_draws": 1, **stated}
