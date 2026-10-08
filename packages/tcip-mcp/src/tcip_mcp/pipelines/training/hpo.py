"""HPO, hyperparameter optimization on Ray Tune.

The search algorithms and trial schedulers a sweep can name:
  - search algorithms: ``random``/``grid`` are native; ``optuna``, ``bayesopt``, ``hyperopt``
    need their pip backend and are installed by default. Each is constructed with the sweep's
    own seed.
  - trial schedulers: ``asha`` (async HyperBand), ``hyperband``, ``pbt``, ``median``; ``none``
    runs every trial to completion.
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
import sys
import threading
from collections.abc import Generator
from contextlib import contextmanager
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Ray is one cluster per process, shared by every concurrent sweep, so its lifetime is
# refcounted rather than owned by whichever sweep happened to start first.
_ray_lifecycle = threading.Lock()
# The live sweeps: the cluster lives while any runs.
_active_searches = 0
# Whether the running cluster is one this module started.
_ray_started = False
# The PYTHONPATH this module handed ray.init() for the still-running cluster; None until it starts
# one.
_ray_runtime_pythonpath: str | None = None
_external_cluster_warned = False

# Ray Tune raises on these deprecated variables' mere presence; storage_path alone decides
# where trial results land, so a sweep drops them for its duration and restores them after.
_ENV_VARS_RAY_TUNE_REFUSES = ("TUNE_RESULT_DIR", "RAY_AIR_LOCAL_CACHE_DIR")

# Native samplers (BasicVariantGenerator), no extra dependency. ``grid`` becomes a grid
# over the discrete axes of the space; ``random`` samples them.
_NATIVE_SEARCH = {"random", "grid", "variant_generator"}


def search_alg_key(search_alg: str | None) -> str:
    """The searcher a sweep names: ``random`` where it names none, its own name lower-cased
    otherwise, a blank one included."""
    return "random" if search_alg is None else search_alg.lower()


_SEARCH_BACKENDS: dict[str, tuple[str, str, str, str]] = {
    "optuna": ("optuna", "ray.tune.search.optuna", "OptunaSearch", "seed"),
    "hyperopt": ("hyperopt", "ray.tune.search.hyperopt", "HyperOptSearch", "random_state_seed"),
    "bayesopt": ("bayes_opt", "ray.tune.search.bayesopt", "BayesOptSearch", "random_state"),
}
"""Each offered backend searcher: the backend module it needs (probed with ``find_spec``), the
Ray wrapper class constructed directly, and that class's own seed keyword. Not offered: nevergrad
(its wrapper takes no seed), ax (its wrapper fails to set up a real space under the installed
Ray), and zoopt, hpbandster, hebo (abandoned upstreams)."""
# Scheduler aliases -> Ray's create_scheduler name.
_SCHEDULER_ALIASES = {
    "asha": "async_hyperband", "async_hyperband": "async_hyperband",
    "hyperband": "hyperband", "pbt": "pbt",
    "median": "median_stopping_rule", "median_stopping_rule": "median_stopping_rule",
}
_NO_SCHEDULER = {"none", "fifo", ""}
# Schedulers that consume the grace-period / reduction-factor early-stopping knobs.
_HALVING_SCHEDULERS = {"async_hyperband", "hyperband"}

SPLIT_DRAW_SEED_KEY = "data.split.seed"
"""The dotted param-space key a sweep's ``split_draws`` axis sweeps, paired with every sampled
point."""


def available_search_algs() -> list[str]:
    """Search algorithms usable on this machine: natives + backends whose module imports."""
    algs = ["random", "grid"]
    for name, (module, _wrapper_module, _wrapper_class, _seed_kw) in _SEARCH_BACKENDS.items():
        if find_spec(module) is not None:
            algs.append(name)
    return algs


def available_schedulers() -> list[str]:
    """Trial schedulers Ray Tune offers (all native, none need an extra backend)."""
    return ["asha", "hyperband", "pbt", "median", "none"]


GRID_AXIS_LIMIT = 10_000
"""The most values one gridded int axis may enumerate: an engineering bound, chosen and not
measured, on what a grid materializes as a list before Ray expands it into trials, so a range
meant for sampling is refused by name rather than allocated."""


def space_axes(param_space: dict, *, grid: bool) -> dict[str, tuple[str, list]]:
    """Each axis of the platform param-space dict as its type and the values it reaches the ends
    of: a ``categorical`` axis's ``choices``, a ``loguniform``/``uniform`` axis's float
    ``low``/``high``, an ``int`` axis's inclusive ``low``/``high``. Each key is a config key or a
    dotted path into one (``optimizer.head_lr``). Refuses an axis not in that shape: a missing
    key (``KeyError``) or a spec that is no mapping (``TypeError``), and by name (``ValueError``)
    an unknown type, an ``int`` axis whose ``low`` exceeds its ``high`` and a ``categorical`` axis
    with no choices, neither of which yields a value to train, an ``int`` axis whose inclusive
    ``low`` or ``high`` is no int64 value, the values Ray's integer sampler draws (numpy's
    ``Generator.integers``, whose exclusive end may reach ``2**63``), and, under ``grid`` (a grid
    search enumerating every discrete axis), an ``int`` axis spanning more than
    :data:`GRID_AXIS_LIMIT` values, before anything is enumerated."""
    import numpy as np

    drawable = np.iinfo(np.int64)
    axes: dict[str, tuple[str, list]] = {}
    for name, spec in param_space.items():
        ptype = spec["type"]
        if ptype in ("loguniform", "uniform"):
            values = [spec["low"], spec["high"]]
        elif ptype == "int":
            values = [int(spec["low"]), int(spec["high"])]
            if values[0] > values[1]:
                raise ValueError(f"param_space axis {name!r}: int low {spec['low']} exceeds high "
                                 f"{spec['high']}")
            if values[0] < drawable.min or values[1] > drawable.max:
                raise ValueError(
                    f"param_space axis {name!r}: int [{values[0]}, {values[1]}] reaches past "
                    f"[{drawable.min}, {drawable.max}], the int64 values the sampler draws")
            span = values[1] - values[0] + 1
            if grid and span > GRID_AXIS_LIMIT:
                raise ValueError(
                    f"param_space axis {name!r}: a grid over int [{values[0]}, {values[1]}] "
                    f"enumerates {span} values, past the {GRID_AXIS_LIMIT} one grid axis may "
                    "hold; narrow it or sample it under a search_alg other than grid")
        elif ptype == "categorical":
            values = list(spec["choices"])
            if not values:
                raise ValueError(f"param_space axis {name!r}: categorical with no choices")
        else:
            raise ValueError(f"param_space axis {name!r}: unknown param type {ptype!r}")
        axes[name] = (ptype, values)
    return axes


def _to_tune_space(
    param_space: dict, grid: bool = False, grid_keys: frozenset[str] = frozenset(),
) -> dict:
    """Convert the platform param-space dict (:func:`space_axes`) into a Ray Tune search space.
    ``grid=True`` enumerates every discrete axis (categorical / int) via ``grid_search``;
    continuous axes stay sampled. ``grid_keys`` names categorical axes forced to ``grid_search``
    regardless of ``grid``."""
    from ray import tune

    space: dict[str, Any] = {}
    for name, (ptype, values) in space_axes(param_space, grid=grid).items():
        as_grid = grid or name in grid_keys
        if ptype == "loguniform":
            space[name] = tune.loguniform(*values)
        elif ptype == "uniform":
            space[name] = tune.uniform(*values)
        elif ptype == "int" and as_grid:
            space[name] = tune.grid_search(list(range(values[0], values[1] + 1)))
        elif ptype == "int":
            space[name] = tune.randint(values[0], values[1] + 1)
        else:
            space[name] = tune.grid_search(values) if as_grid else tune.choice(values)
    return space


def build_search_alg(
    name: str | None, *, seed: int, points_to_evaluate: list[dict] | None = None,
    constant_grid_search: bool = False,
):
    """Build the Ray Tune searcher ``name`` names, constructed with ``seed``.

    A native name (``random``/``grid``) builds Ray's ``BasicVariantGenerator`` with
    ``random_state=seed``; ``constant_grid_search`` pairs every sampled point with each value of
    the space's grid axes. A backend name constructs its own Ray wrapper class
    (:data:`_SEARCH_BACKENDS`) with the seed in that class's own keyword. ``metric``/``mode`` are
    not passed. Raises ``ValueError`` naming the choice for a searcher not offered here, for an
    offered backend
    that is not installed, and for ``constant_grid_search`` asked of a backend; the choice is
    honored, never swapped for another algorithm.
    """
    from tcip_mcp.pipelines.model_build import resolve_named

    key = search_alg_key(name)
    backend = resolve_named(key, {**dict.fromkeys(_NATIVE_SEARCH), **_SEARCH_BACKENDS},
                            kind="search_alg")
    points = list(points_to_evaluate) if points_to_evaluate else None
    if backend is None:
        from ray.tune.search.basic_variant import BasicVariantGenerator

        return BasicVariantGenerator(points_to_evaluate=points, random_state=seed,
                                     constant_grid_search=constant_grid_search)
    if constant_grid_search:
        raise ValueError(f"constant_grid_search is the native sampler's, and search_alg '{key}' "
                         "is a backend searcher; choose random or grid.")
    module, wrapper_module, wrapper_class, seed_kw = backend
    if find_spec(module) is None:
        raise ValueError(
            f"search_alg '{key}' needs the '{module}' backend, which is not installed. "
            f"Available here: {available_search_algs()}"
        )

    import importlib

    wrapper = getattr(importlib.import_module(wrapper_module), wrapper_class)
    return wrapper(points_to_evaluate=points, **{seed_kw: seed})


def build_scheduler(
    name: str | None, *, grace_period: int = 5, reduction_factor: int = 3,
    hyperparam_mutations: dict | None = None,
):
    """Build a Ray Tune trial scheduler, or ``None`` to run every trial to completion.

    ``metric``/``mode`` are not passed. ``pbt`` takes ``hyperparam_mutations`` (the search
    space).
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
    own requested concurrency: ``gpu=0.0`` with no CUDA device; otherwise ``device_count /
    max_concurrent`` capped at 1.0. ``max_concurrent=1`` (the default) yields ``gpu=1.0``.
    """
    try:
        import torch
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        count = 0
    gpu = 0.0 if count == 0 else min(1.0, count / max(max_concurrent, 1))
    return {"cpu": 1.0, "gpu": gpu}


def _has_attached_console() -> bool:
    """Whether this process has a console to signal Ray's daemons through: always on a platform
    other than Windows; on Windows, when ``GetConsoleCP`` reads nonzero."""
    if sys.platform != "win32":
        return True
    import ctypes

    return ctypes.windll.kernel32.GetConsoleCP() != 0


def _kill_ray_daemons_before_shutdown(ray: Any) -> None:
    """Kill every daemon this process's Ray cluster started (the reaper last) and each one's
    descendants, ahead of ``ray.shutdown()``, waiting one second on them together; a process
    already gone or unreachable is skipped. A no-op for a cluster this module did not start."""
    import psutil

    private = getattr(ray, "_private", None)
    worker = getattr(private, "worker", None) if private is not None else None
    global_worker = getattr(worker, "global_worker", None) if worker is not None else None
    node = getattr(global_worker, "node", None) if global_worker is not None else None
    all_processes = getattr(node, "all_processes", None) if node is not None else None
    if not all_processes:
        return

    order = ["raylet", "gcs_server"]
    order += [process_type for process_type in all_processes if process_type not in order
              and process_type != "reaper"]
    order.append("reaper")

    for process_type in order:
        for process_info in all_processes.get(process_type, []):
            process = process_info.process
            if process.poll() is not None:
                continue

            try:
                descendants = psutil.Process(process.pid).children(recursive=True)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                descendants = []
            for descendant in descendants:
                try:
                    descendant.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            if descendants:
                psutil.wait_procs(descendants, timeout=1)

            process.kill()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Ray daemon %s (pid %s) did not exit within 1s of being killed; "
                    "ray.shutdown() will still try to signal it", process_type, process.pid,
                )


@contextmanager
def _ray_session(ray: Any, num_cpus: int) -> Generator[None]:
    """Keep Ray up for the duration of one sweep, shutting it down only when the last concurrent
    sweep leaves and only if this module is what started it. The lock covers both the
    check-then-init and the decrement-then-shutdown sequences.

    ``num_cpus`` is the CPU count the cluster is started with when this call starts it: the sweep's
    own request, every concurrent trial's CPUs together. A sweep that joins a cluster a sibling
    started runs on the sibling's size.
    """
    global _ray_started, _ray_runtime_pythonpath, _active_searches
    global _external_cluster_warned

    from tcip_mcp.pipelines.model_build import child_pythonpath

    with _ray_lifecycle:
        if not ray.is_initialized():
            # Only this branch configures Ray; the not-taken branch below leaves an
            # already-initialized cluster's runtime_env alone, whoever started it.
            pythonpath = child_pythonpath()
            ray.init(num_cpus=num_cpus, include_dashboard=False, log_to_driver=False,
                     ignore_reinit_error=True, configure_logging=False,
                     runtime_env={"env_vars": {"PYTHONPATH": pythonpath}})
            _ray_started = True
            _ray_runtime_pythonpath = pythonpath
        elif _ray_started:
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
            if not _active_searches and _ray_started:
                _ray_started = False
                _ray_runtime_pythonpath = None
                if not _has_attached_console():
                    _kill_ray_daemons_before_shutdown(ray)
                ray.shutdown()


def _build_sweep_stopper(sweep_root: Path) -> tuple[Any, Any]:
    """A ``ray.tune.Stopper`` for cooperative-first, Ray-hard-stop-as-fallback sweep cancel, paired
    with the ``ray.tune.Callback`` that feeds its whole-experiment decision.

    Per-trial (``Stopper.__call__``): True whenever a cancel of ``sweep_root`` is requested
    (``experiments.cancel_requested``), so Tune ends that trial after the report it just made.
    Whole-experiment
    (``Stopper.stop_all``): True once a cancel is requested and either the callback's own live set
    (every trial id Ray has started and not yet completed or errored, kept current from
    ``on_trial_start``/``on_trial_complete``/``on_trial_error``) is empty, or the heartbeat stale
    window has passed since the first request's recorded time (``experiments.request_cancel``), at
    which point Ray's own stop kills whatever actor a trial that never polls ``should_cancel`` left
    running.

    Returns ``(stopper, callback)``; the caller wires the stopper onto ``run_config.stop`` and the
    callback into ``run_config.callbacks``.
    """
    from ray.tune import Callback, Stopper

    from tcip_mcp.experiments import cancel_requested

    class _LiveTrialsCallback(Callback):
        """The trial ids Ray currently holds live: started and not yet completed or errored."""

        def __init__(self) -> None:
            self.live_trial_ids: set[str] = set()

        def on_trial_start(self, iteration: int, trials: list, trial: Any, **info: Any) -> None:
            self.live_trial_ids.add(trial.trial_id)

        def on_trial_complete(self, iteration: int, trials: list, trial: Any, **info: Any) -> None:
            self.live_trial_ids.discard(trial.trial_id)

        on_trial_error = on_trial_complete

    live_trials = _LiveTrialsCallback()

    class _SweepStopper(Stopper):
        def __call__(self, trial_id: str, result: dict) -> bool:
            return cancel_requested(sweep_root)

        def stop_all(self) -> bool:
            if not cancel_requested(sweep_root):
                return False
            from datetime import datetime, timezone

            from tcip_mcp.experiments import CANCEL_FILE, HEARTBEAT_STALE_SECONDS, read_record

            requested = datetime.fromisoformat(read_record(sweep_root / CANCEL_FILE)["requested"])
            waited = (datetime.now(timezone.utc) - requested).total_seconds()
            return waited > HEARTBEAT_STALE_SECONDS or not live_trials.live_trial_ids

    return _SweepStopper(), live_trials


def split_draw_search_space(
    param_space: dict, base_config: dict, split_draws: int, split_draw_seeds: list[int] | None,
) -> tuple[dict, list[int] | None]:
    """The search space for ``tune_search`` and :func:`planned_trial_count`: ``param_space``
    itself, unaugmented, at ``split_draws`` of one or below; above one, a copy carrying
    :data:`SPLIT_DRAW_SEED_KEY` as a categorical grid axis over the resolved draw seeds
    (``split_draw_seeds`` when given, else ``base_config``'s own ``data.split.seed`` plus the draw
    index, refused when it states none).

    Returns ``(search_param_space, resolved_draw_seeds)``: the second element is ``None`` at one
    draw and the list of seeds paired above it.
    """
    if split_draws <= 1:
        return param_space, None
    from tcip_mcp.pipelines.data.split_construction import split_seed

    if split_draw_seeds is not None:
        resolved_draw_seeds = list(split_draw_seeds)
    else:
        base_seed = split_seed((base_config.get("data") or {}).get("split") or {})
        resolved_draw_seeds = [base_seed + i for i in range(split_draws)]
    search_param_space = {
        **param_space,
        SPLIT_DRAW_SEED_KEY: {"type": "categorical", "choices": resolved_draw_seeds},
    }
    return search_param_space, resolved_draw_seeds


def _search_space_and_points(
    param_space: dict, search_alg: str | None, split_draws: int, baseline_params: dict | None,
) -> tuple[dict, list[dict] | None, str]:
    """The Ray Tune space, warm-start preset points, and the normalized search-algorithm name: the
    platform's own ``param_space`` turned into Ray's own space, gridded over every discrete axis
    under ``grid`` and over :data:`SPLIT_DRAW_SEED_KEY` above one draw, plus ``baseline_params``,
    when given, filtered to the space's own keys.
    """
    normalized_search_alg = search_alg_key(search_alg)
    grid_keys = frozenset({SPLIT_DRAW_SEED_KEY}) if split_draws > 1 else frozenset()
    space = _to_tune_space(param_space, grid=(normalized_search_alg == "grid"),
                           grid_keys=grid_keys)
    filtered = {k: v for k, v in (baseline_params or {}).items() if k in space}
    return space, [filtered] if filtered else None, normalized_search_alg


def planned_trial_count(
    param_space: dict, num_samples: int, search_alg: str | None, split_draws: int,
    baseline_params: dict | None,
) -> int:
    """How many trials ``tune_search`` would launch for this sweep, read off the specification
    :func:`_search_space_and_points` prepares for the launch itself: under a native search, Ray's
    own variant count over that space and its warm-start points (``num_samples`` times every grid
    axis's size, a preset counting its own grid); under a backend searcher, ``num_samples``, the
    trials ``ray.tune.TuneConfig`` asks it for. A negative ``num_samples`` counts zero. Builds no
    searcher and draws no trial.

    Propagates whatever ``_to_tune_space`` raises on a space it cannot build.
    """
    from ray.tune.search.variant_generator import _count_variants

    space, points, normalized_search_alg = _search_space_and_points(
        param_space, search_alg, split_draws, baseline_params)
    if normalized_search_alg not in _NATIVE_SEARCH:
        return max(num_samples, 0)
    return _count_variants({"config": space, "num_samples": num_samples}, points or [])


def tune_search(
    objective_fn: Callable[[dict, Callable[[float], None]], Any],
    param_space: dict,
    *,
    metric: str,
    mode: str,
    num_samples: int,
    search_alg: str | None,
    scheduler: str | None,
    grace_period: int,
    reduction_factor: int,
    seed: int,
    max_concurrent: int,
    baseline_params: dict | None,
    sweep_dir: Path,
    resources_per_trial: dict | None,
    split_draws: int,
) -> None:
    """Run an HPO sweep on Ray Tune, its results in Ray's experiment store at ``sweep_dir``.
    A cancel requested of that directory
    (``experiments.request_cancel``) stops it (:func:`_build_sweep_stopper`). Every argument is
    the sweep's own, stated by its caller.

    Args:
        objective_fn: ``fn(config, report)``, trains one trial for the trial's ``config`` and calls
            ``report(value)`` for each step it wants the searcher/scheduler to see.
        param_space: platform param-space dict (see :func:`_to_tune_space`).
        metric / mode: the reported metric name and whether to ``min`` or ``max`` it.
        num_samples: number of trials (with a grid space, samples over the grid); the count
            launched is :func:`planned_trial_count`'s.
        search_alg / scheduler: agent-selected names (see module docstring). ``None`` schedules
        nothing; native ``random``/``grid`` need no searcher backend.
        seed: the searcher's own seed, the sweep's stated one; required.
        max_concurrent: trials to run at once.
        baseline_params: a point to seed the search with, ``None`` for none.
        sweep_dir: the local directory Ray persists the sweep's trial results in, its parent
            Ray's ``storage_path`` and its name Ray's experiment name.
        resources_per_trial: Ray resource request per trial (``{"cpu": ..., "gpu": ...}``, GPU as a
            fraction for sharing). ``None`` derives one from the host's real GPU count and
            ``max_concurrent``. A cluster this sweep starts is sized to ``cpu`` times
            ``max_concurrent`` CPUs (:func:`_ray_session`).
        split_draws: Above 1, ``param_space`` must already carry a ``SPLIT_DRAW_SEED_KEY`` grid
            axis (:func:`split_draw_search_space`; raises ``ValueError`` naming the axis when it is
            missing) and :func:`build_search_alg` builds the native sampler with
            ``constant_grid_search``, so every sampled point is trained once per seed whether
            ``search_alg`` is ``random`` or ``grid``; a backend ``search_alg`` refuses.
    """
    if split_draws > 1 and SPLIT_DRAW_SEED_KEY not in param_space:
        raise ValueError(
            f"tune_search: split_draws={split_draws} pairs {SPLIT_DRAW_SEED_KEY!r} as a grid "
            "axis with every sampled point, and param_space carries no such axis: pass it "
            "explicitly, or call through run_hyperparameter_search's own split_draws, which adds "
            "it for you."
        )

    import ray
    from ray import tune

    # The normalized name is the helper's third return, read at every branch point below;
    # result["search_alg"] stays the caller's own string.
    space, points, normalized_search_alg = _search_space_and_points(
        param_space, search_alg, split_draws, baseline_params)
    resources = resources_per_trial or _default_trial_resources(max_concurrent)

    searcher = build_search_alg(normalized_search_alg, seed=seed, points_to_evaluate=points,
                                constant_grid_search=split_draws > 1)
    sched = build_scheduler(
        scheduler, grace_period=grace_period, reduction_factor=reduction_factor,
        hyperparam_mutations=space,
    )

    # Concurrency: a backend searcher must be wrapped (TuneConfig.max_concurrent_trials is
    # ignored once Ray wraps a searcher). The native BasicVariantGenerator honors
    # max_concurrent_trials on the Tuner directly.
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
        from tcip_store import bind

        bind()
        objective_fn(config, lambda value: tune.report({metric: float(value)}))

    trainable = tune.with_resources(trainable, resources=resources)
    # A trainable with no CPU request gets one CPU from Ray, so the cluster is sized the same way.
    cluster_cpus = math.ceil(float(resources.get("cpu", 1.0)) * max(max_concurrent, 1))

    stopper, live_trials_callback = _build_sweep_stopper(sweep_dir)
    run_kwargs: dict[str, Any] = {
        "verbose": 0,
        "storage_path": sweep_dir.parent.resolve().as_posix(),
        "name": sweep_dir.name,
        "stop": stopper,
        "callbacks": [live_trials_callback],
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
        with _ray_session(ray, cluster_cpus):
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
            tuner.fit()
    finally:
        os.environ.update(removed_env)
