"""HPO, hyperparameter optimization on Ray Tune.

Search *algorithms* and trial *schedulers* are agent-selectable per task/data; no single
method is welded in. The agent picks from whatever backends are installed (Ray degrades to
what imports on this machine) and overrides the derivable defaults when the data warrants.

Facts (not a recipe, the agent chooses):
  - search algorithms: ``random``/``grid`` are native; ``optuna``, ``bayesopt``, ``hyperopt``,
    ``nevergrad``, ``ax`` need their pip backend and are installed by default.
  - trial schedulers: ``asha`` (async HyperBand), ``hyperband``, ``pbt``, ``median``; ``none``
    runs every trial to completion.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Generator
from contextlib import contextmanager
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable

from tcip_store import RECORD_JSON, Key, StoreDescriptor, register_store, stored_number
from tcip_store.file_backend import RootedFileLocator

logger = logging.getLogger(__name__)

# Ray is one cluster per process, shared by every concurrent sweep, so its lifetime is
# refcounted rather than owned by whichever sweep happened to start first.
_ray_lifecycle = threading.Lock()
_active_searches = 0
_ray_started_here = False
# The PYTHONPATH this module handed ray.init() for the still-running cluster; None until it starts one.
_ray_runtime_pythonpath: str | None = None
_external_cluster_warned = False

# Ray Tune raises on these deprecated variables' mere presence; storage_path alone decides
# where trial results land, so a sweep drops them for its duration and restores them after.
_ENV_VARS_RAY_TUNE_REFUSES = ("TUNE_RESULT_DIR", "RAY_AIR_LOCAL_CACHE_DIR")

# Native samplers (BasicVariantGenerator), no extra dependency. ``grid`` becomes a grid
# over the discrete axes of the space; ``random`` samples them.
_NATIVE_SEARCH = {"random", "grid", "variant_generator", "", None}
# The backend module each searcher imports at instantiation (probe with find_spec so discovery
# has no side effects). All of these install by default; the probe covers an install that skipped
# the hpo extra. Backends resting on abandoned upstreams are not offered: zoopt and hpbandster
# (which bohb needs, for both its searcher and its scheduler) have no maintained release, and hebo
# cannot build on Windows/Py3.12.
_SEARCH_BACKEND_MODULE = {
    "optuna": "optuna", "hyperopt": "hyperopt", "bayesopt": "bayes_opt",
    "nevergrad": "nevergrad", "ax": "ax",
}
# Seed kwarg differs per searcher; pass it only where the constructor accepts one.
_SEARCHER_SEED_KWARG = {"optuna": "seed", "hyperopt": "random_state_seed", "bayesopt": "random_state"}
# Scheduler aliases -> Ray's create_scheduler name.
_SCHEDULER_ALIASES = {
    "asha": "async_hyperband", "async_hyperband": "async_hyperband",
    "hyperband": "hyperband", "pbt": "pbt",
    "median": "median_stopping_rule", "median_stopping_rule": "median_stopping_rule",
}
_NO_SCHEDULER = {"none", "fifo", ""}
# Schedulers that consume the grace-period / reduction-factor early-stopping knobs.
_HALVING_SCHEDULERS = {"async_hyperband", "hyperband"}


def get_default_space() -> dict:
    """A small, safe starting space the agent overrides per dataset (never a fixed recipe).

    Each key maps to ``{'type': ..., ...}``:
      - ``categorical``: ``{'choices': [...]}``
      - ``loguniform`` / ``uniform``: ``{'low': float, 'high': float}``
      - ``int``: ``{'low': int, 'high': int}`` (both bounds inclusive)
    """
    return {
        "lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
        "batch_size": {"type": "categorical", "choices": [2, 4, 8]},
        "weight_decay": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
    }


def get_default_baseline_params() -> dict:
    """A known-good point to warm-start (subset of the default space)."""
    return {"lr": 3e-4, "batch_size": 4, "weight_decay": 1e-4}


def available_search_algs() -> list[str]:
    """Search algorithms usable on this machine: natives + backends whose module imports."""
    algs = ["random", "grid"]
    for name, module in _SEARCH_BACKEND_MODULE.items():
        if find_spec(module) is not None:
            algs.append(name)
    return algs


def available_schedulers() -> list[str]:
    """Trial schedulers Ray Tune offers (all native, none need an extra backend)."""
    return ["asha", "hyperband", "pbt", "median", "none"]


def _to_tune_space(param_space: dict, grid: bool = False) -> dict:
    """Convert the platform param-space dict into a Ray Tune search space.

    ``grid=True`` enumerates the discrete axes (categorical / int) via ``grid_search``;
    continuous axes stay sampled (a grid can't enumerate a continuous range).
    """
    from ray import tune

    space: dict[str, Any] = {}
    for name, spec in param_space.items():
        ptype = spec["type"]
        if ptype == "loguniform":
            space[name] = tune.loguniform(spec["low"], spec["high"])
        elif ptype == "uniform":
            space[name] = tune.uniform(spec["low"], spec["high"])
        elif ptype == "int":
            vals = list(range(int(spec["low"]), int(spec["high"]) + 1))
            space[name] = tune.grid_search(vals) if grid else tune.randint(spec["low"], spec["high"] + 1)
        elif ptype == "categorical":
            choices = list(spec["choices"])
            space[name] = tune.grid_search(choices) if grid else tune.choice(choices)
        else:
            raise ValueError(f"Unknown param type: {ptype}")
    return space


def build_search_alg(
    name: str | None, *, seed: int = 42, points_to_evaluate: list[dict] | None = None,
):
    """Build a Ray Tune searcher, or ``None`` for the native random/grid sampler.

    ``metric``/``mode`` are set once on the Tuner (Ray forbids setting them in both places),
    so they are not passed here. Raises a clear error if the agent picks a backend that is not
    installed, the choice is honored, never silently swapped for another algorithm.
    """
    key = (name or "random").lower()
    if key in _NATIVE_SEARCH:
        if points_to_evaluate:
            from ray.tune.search.basic_variant import BasicVariantGenerator
            return BasicVariantGenerator(points_to_evaluate=list(points_to_evaluate))
        return None

    module = _SEARCH_BACKEND_MODULE.get(key)
    if module is not None and find_spec(module) is None:
        raise ValueError(
            f"search_alg '{key}' needs the '{module}' backend, which is not installed. "
            f"Available here: {available_search_algs()}"
        )

    from ray.tune.search import create_searcher

    kwargs: dict[str, Any] = {}
    seed_kw = _SEARCHER_SEED_KWARG.get(key)
    if seed_kw:
        kwargs[seed_kw] = seed
    if points_to_evaluate:
        kwargs["points_to_evaluate"] = list(points_to_evaluate)
    try:
        return create_searcher(key, **kwargs)
    except TypeError:
        # Searcher rejected an optional kwarg (seed / points_to_evaluate); retry minimal.
        return create_searcher(key)


def build_scheduler(
    name: str | None, *, grace_period: int = 5, reduction_factor: int = 3,
    hyperparam_mutations: dict | None = None,
):
    """Build a Ray Tune trial scheduler, or ``None`` to run every trial to completion.

    ``metric``/``mode`` are set once on the Tuner, not here (Ray forbids both). PBT mutates
    hyperparameters mid-training, so it needs ``hyperparam_mutations`` (the search space).
    """
    key = (str(name).lower() if name is not None else None)
    if key is None or key in _NO_SCHEDULER:
        return None
    ray_name = _SCHEDULER_ALIASES.get(key, key)

    from ray.tune.schedulers import create_scheduler

    kwargs: dict[str, Any] = {}
    if ray_name in _HALVING_SCHEDULERS:
        kwargs.update(grace_period=grace_period, reduction_factor=reduction_factor)
    if ray_name == "pbt" and hyperparam_mutations:
        kwargs["hyperparam_mutations"] = hyperparam_mutations
    return create_scheduler(ray_name, **kwargs)


def _default_trial_resources(max_concurrent: int) -> dict[str, float]:
    """Derive a per-trial Ray resource request from the host's actual GPU count and the caller's
    own requested concurrency, never a pinned number. ``gpu=0.0`` with no CUDA device;
    otherwise ``device_count / max_concurrent`` capped at 1.0, so ``max_concurrent`` trials the
    agent asked to run at once actually get non-overlapping (or fairly-shared) GPU allocations
    instead of every trial silently defaulting to Ray's own 0-GPU request and all contending for
    whatever `TrainContext.device` happens to resolve to. ``max_concurrent=1`` (today's default)
    yields ``gpu=1.0``, byte-identical to today's implicit whole-device behavior when unset.
    """
    try:
        import torch
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        count = 0
    gpu = 0.0 if count == 0 else min(1.0, count / max(max_concurrent, 1))
    return {"cpu": 1.0, "gpu": gpu}


RAY_DASHBOARD_STORE = "ray_dashboard"
register_store(
    StoreDescriptor(
        name=RAY_DASHBOARD_STORE,
        kind="record",
        key_fields=("document",),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="last_writer_wins",
        locator=RootedFileLocator(prefix=(".tcip", "state"), suffix=".json"),
    )
)


def ray_dashboard_key() -> Key:
    """Where a running cluster's dashboard URL is written down, under the platform root.

    ``last_writer_wins``: the whole document is composed from what ``ray.init()`` just
    returned and written in one shot, never merged into.
    """
    from tcip_mcp.project_paths import platform_state_root

    return Key(RAY_DASHBOARD_STORE, str(platform_state_root().resolve()), ("ray_dashboard",))


def _publish_ray_dashboard(dashboard_url: str | None) -> None:
    """Record the dashboard of the cluster this process just started.

    ``ray.init()`` reports the dashboard as a bare ``host:port``; what is written down is a
    fetchable URL, so every reader agrees on one form. The pid is the initiating process,
    which is what tells a later reader (in any process) whether the cluster is still up.
    """
    if not dashboard_url:
        logger.info("Ray started without a dashboard; no URL to publish")
        return
    from datetime import datetime, timezone

    from tcip_store import StoreError, store

    try:
        store.replace(ray_dashboard_key(), {
            "url": f"http://{dashboard_url}",
            "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        })
    except (OSError, StoreError):
        # A dashboard URL nobody could record is a missing convenience, never a reason to
        # sink the sweep the cluster was started for.
        logger.warning("could not record the Ray dashboard URL", exc_info=True)


def _clear_ray_dashboard() -> None:
    """Drop the recorded dashboard so a torn-down cluster's URL is never served."""
    from tcip_store import StoreError, store

    try:
        store.delete(ray_dashboard_key())
    except (OSError, StoreError):
        logger.warning("could not clear the recorded Ray dashboard URL", exc_info=True)


def read_ray_dashboard() -> dict | None:
    """The live Ray dashboard's ``{url, pid, started_at}``, or ``None`` if none is up.

    Ray's dashboard is its own OS process on a real port, so any process on the machine can
    serve its URL once it has been written down: an agent-launched sweep initializes Ray
    inside the MCP server, and the web backend reads the same record. A recorded URL whose
    initiating process is gone is stale, since that process's exit takes the cluster with it.

    An unreadable record is reported and then answered the way an absent one is: this record
    describes a live process, so bytes that do not decode cannot name one either.
    """
    import psutil

    from tcip_store import DecodeError, store

    try:
        state = store.read(ray_dashboard_key(), default=None)
    except DecodeError:
        logger.warning("the recorded Ray dashboard does not decode", exc_info=True)
        return None
    if not isinstance(state, dict) or not state.get("url"):
        return None
    pid = state.get("pid")
    if not isinstance(pid, int) or not psutil.pid_exists(pid):
        return None
    return state


@contextmanager
def _ray_session(ray: Any) -> Generator[None]:
    """Keep Ray up for the duration of one sweep, shutting it down only when the last
    concurrent sweep leaves and only if this module is what started it.

    Two conditions, both required. The count is what makes a finishing sweep leave a
    still-running sibling's cluster alone; the ownership flag is what keeps a sweep running
    inside a process that initialized Ray for its own reasons (a notebook, an embedding
    application) from tearing that cluster down. The lock covers both, so the
    check-then-init and the decrement-then-shutdown sequences cannot interleave.
    """
    global _active_searches, _ray_started_here, _ray_runtime_pythonpath, _external_cluster_warned

    from tcip_mcp.pipelines.model_build import child_pythonpath

    with _ray_lifecycle:
        if not ray.is_initialized():
            # A bare ray[tune] install has no aiohttp; probe rather than assume, so a sweep still runs without it.
            include_dashboard = find_spec("aiohttp") is not None
            # Only this branch configures Ray; the not-taken branch below leaves an
            # already-initialized cluster's runtime_env alone, whoever started it.
            pythonpath = child_pythonpath()
            context = ray.init(include_dashboard=include_dashboard, dashboard_host="127.0.0.1",
                               log_to_driver=False, ignore_reinit_error=True,
                               configure_logging=False,
                               runtime_env={"env_vars": {"PYTHONPATH": pythonpath}})
            _ray_started_here = True
            _ray_runtime_pythonpath = pythonpath
            if include_dashboard:
                _publish_ray_dashboard(context.dashboard_url)
        elif _ray_started_here:
            if child_pythonpath() != _ray_runtime_pythonpath:
                logger.warning(
                    "the running Ray cluster's workers keep the import path captured at "
                    "cluster start; a bespoke source directory added since will not import "
                    "in trial workers until the cluster is restarted"
                )
        elif not _external_cluster_warned:
            _external_cluster_warned = True
            logger.warning(
                "Ray was already initialized outside this module; no import-path "
                "propagation was applied to its workers"
            )
        _active_searches += 1
    try:
        yield
    finally:
        with _ray_lifecycle:
            _active_searches -= 1
            if _active_searches == 0 and _ray_started_here:
                _ray_started_here = False
                _ray_runtime_pythonpath = None
                _clear_ray_dashboard()
                ray.shutdown()


def tune_search(
    objective_fn: Callable[[dict, Callable[[float], None]], Any],
    param_space: dict | None = None,
    *,
    metric: str = "objective",
    mode: str = "min",
    num_samples: int = 20,
    search_alg: str | None = "random",
    scheduler: str | None = "asha",
    grace_period: int = 5,
    reduction_factor: int = 3,
    seed: int = 42,
    max_concurrent: int = 1,
    warm_start: bool = False,
    baseline_params: dict | None = None,
    storage_path: str | None = None,
    study_name: str = "tcip_hpo",
    resources_per_trial: dict | None = None,
) -> dict:
    """Run an HPO sweep on Ray Tune.

    Args:
        objective_fn: ``fn(config, report)``, trains one trial for the trial's ``config``
            and calls ``report(value)`` for each step it wants the searcher/scheduler to see
            (report at least once; the last value is the trial's result under ``mode``).
        param_space: platform param-space dict (see ``get_default_space``); ``None`` uses it.
        metric / mode: the reported metric name and whether to ``min`` or ``max`` it.
        num_samples: number of trials (with a grid space, samples over the grid).
        search_alg / scheduler: agent-selected names (see module docstring). ``None`` schedules
            nothing; native ``random``/``grid`` need no searcher backend.
        max_concurrent: trials to run at once (default 1, safe for single-GPU training).
        warm_start: seed the search with ``baseline_params`` (or the default baseline).
        storage_path: where Ray persists trial results (also the TensorBoard logdir root).
            Required: trial results land where the caller says, never Ray's own
            home-directory default. ``run_hpo`` resolves it for a training sweep (the
            project's own ``.tcip/hpo``); a bespoke search names its own directory. Accepts
            a local path today; Ray's ``storage_path`` also takes a cloud URI, the seam a
            central store would use.
        resources_per_trial: Ray resource request per trial (``{"cpu": ..., "gpu": ...}``, GPU as
            a fraction for sharing). Omit to derive one from the host's real GPU count and
            ``max_concurrent``, an explicit value always wins over the derivation.

    Returns dict with ``best_params``, ``best_value``, ``n_trials``, ``all_trials``,
    ``search_alg``, ``scheduler``, ``study_name`` (+ ``warm_start``/``baseline_params``).
    """
    if not storage_path:
        raise ValueError(
            "tune_search needs storage_path: trial results are persisted where the caller "
            "says, never Ray's own home-directory default. run_hpo resolves it for a "
            "training sweep (the project's own .tcip/hpo via hpo_root/sweep_dir); a bespoke "
            "search names its own directory."
        )

    import ray
    from ray import tune

    space = _to_tune_space(param_space or get_default_space(), grid=(search_alg == "grid"))
    resources = resources_per_trial or _default_trial_resources(max_concurrent)

    points = None
    if warm_start:
        baseline = baseline_params or get_default_baseline_params()
        filtered = {k: v for k, v in baseline.items() if k in space}
        points = [filtered] if filtered else None

    searcher = build_search_alg(search_alg, seed=seed, points_to_evaluate=points)
    sched = build_scheduler(
        scheduler, grace_period=grace_period, reduction_factor=reduction_factor,
        hyperparam_mutations=space,
    )

    # Concurrency: a backend searcher must be wrapped (TuneConfig.max_concurrent_trials is
    # ignored once Ray wraps a searcher). The native sampler (None / BasicVariantGenerator)
    # honors max_concurrent_trials on the Tuner directly.
    from ray.tune.search import Searcher
    from ray.tune.search.basic_variant import BasicVariantGenerator

    tune_max: int | None = max_concurrent
    if (isinstance(searcher, Searcher) and not isinstance(searcher, BasicVariantGenerator)
            and max_concurrent and max_concurrent > 0):
        from ray.tune.search import ConcurrencyLimiter
        searcher = ConcurrencyLimiter(searcher, max_concurrent=max_concurrent)
        tune_max = None

    def trainable(config: dict) -> None:
        # A trial body runs in a Ray worker process, which is its own storage entry point.
        from tcip_store.binding import bind_default

        bind_default()
        objective_fn(config, lambda value: tune.report({metric: float(value)}))

    trainable = tune.with_resources(trainable, resources=resources)

    run_kwargs: dict[str, Any] = {
        "verbose": 0,
        "storage_path": Path(storage_path).resolve().as_posix(),
        "name": study_name,
    }

    removed_env: dict[str, str] = {}
    for var in _ENV_VARS_RAY_TUNE_REFUSES:
        value = os.environ.pop(var, None)
        if value is not None:
            removed_env[var] = value
            logger.warning(
                "ignoring %s=%s for this sweep: Ray deprecated the variable and refuses to "
                "run while it is set; trial results go to storage_path (%s).",
                var, value, run_kwargs["storage_path"],
            )
    try:
        with _ray_session(ray):
            tuner = tune.Tuner(
                trainable,
                param_space=space,
                tune_config=tune.TuneConfig(
                    num_samples=num_samples, search_alg=searcher, scheduler=sched,
                    max_concurrent_trials=tune_max, metric=metric, mode=mode,
                    reuse_actors=False,
                ),
                run_config=tune.RunConfig(**run_kwargs),
            )
            results = tuner.fit()
    finally:
        os.environ.update(removed_env)

    all_trials = []
    for i in range(len(results)):
        r = results[i]
        # Every trial that reaches result assembly reported at least once, so its last_result
        # carries config; a Ray-killed never-reporting trial is a known separate defect on record.
        assert r.config is not None
        all_trials.append({
            "params": {k: r.config.get(k) for k in space},
            **stored_number("value", (r.metrics or {}).get(metric)),
            "iterations": (r.metrics or {}).get("training_iteration"),
            "state": "ERROR" if r.error else "COMPLETE",
        })

    best = results.get_best_result(metric=metric, mode=mode)
    assert best.config is not None
    result: dict[str, Any] = {
        "best_params": {k: best.config.get(k) for k in space},
        **stored_number("best_value", (best.metrics or {}).get(metric)),
        "n_trials": len(results),
        "study_name": study_name,
        "all_trials": all_trials,
        "search_alg": (search_alg or "random"),
        "scheduler": (scheduler if scheduler is not None else "none"),
        "warm_start": warm_start,
        "baseline_params": (baseline_params or get_default_baseline_params()) if warm_start else None,
    }
    result["tensorboard_logdir"] = f"{run_kwargs['storage_path']}/{study_name}"
    return result
