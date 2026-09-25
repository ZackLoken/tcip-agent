"""In-process registry of live training runs: ``TrainRun``, its cancel-sentinel protocol, and
the create/get/list/cancel operations over the process-global ``_RUNS`` map.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

CANCEL_SENTINEL = ".cancel_requested"
"""The run-level cooperative-cancel sentinel's filename, written under a run's own
``output_dir`` and polled by :meth:`TrainRun.should_cancel`. A sweep's own stop file is a
separate protocol (``training_tools.SWEEP_CANCEL_SENTINEL``, at the sweep root rather than a
run's own directory); the two never share a name."""


@dataclass
class TrainRun:
    id: str
    config: dict
    status: str = "created"
    current_epoch: int = 0
    current_stage: int = 0
    # Best selection value so far; train() resets this to the losing-side infinity for the run's
    # resolved selection_metric before the first epoch.
    best_metric: float = float("inf")
    # The bare name train() resolved best_metric on, stamped once before the first epoch (the
    # trainer's own resolution; nothing here re-derives a name from config). None until then.
    best_metric_name: str | None = None
    metrics_history: list[dict] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    error: str = ""
    output_dir: str = ""
    # "training" (standalone/GUI/agent run) vs "hpo_trial" (an HPO sweep's own trial run);
    # the Training view lists only "training" so a sweep of dozens of trials doesn't flood it.
    origin: str = "training"
    # Set by cancel_run() to request a graceful stop; the train loop polls it.
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    # None means the loop runs in-process (cancel_event alone is authoritative); set once the
    # parent spawns the subprocess a run's body executes in, when should_cancel polls the sentinel.
    pid: int | None = None

    def should_cancel(self) -> bool:
        """True if cancellation was requested, in-process (``cancel_event``) or via the sentinel
        file a (possibly different) process may have written at ``<output_dir>/<CANCEL_SENTINEL>``.
        Checked whether or not ``pid`` is set.
        """
        if self.cancel_event.is_set():
            return True
        if self.output_dir:
            return (Path(self.output_dir) / CANCEL_SENTINEL).exists()
        return False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "current_epoch": self.current_epoch,
            "current_stage": self.current_stage,
            "best_metric": self.best_metric,
            "best_metric_name": self.best_metric_name,
            "metrics_history": self.metrics_history,
            "origin": self.origin,
            "elapsed_seconds": (self.end_time or time.time()) - self.start_time if self.start_time else 0,
            "pid": self.pid,
        }


_RUNS: dict[str, TrainRun] = {}
_RUNS_LOCK = threading.Lock()


def draw_seed_if_unset(config: dict) -> None:
    """Draw a seed from OS entropy into ``config`` in place, unless the caller already set one.
    Never start an unseeded run.
    """
    if config.get("seed") is None:
        config["seed"] = random.SystemRandom().randrange(2**31)
        logger.info("no seed configured; drew seed=%d.", config["seed"])


def create_run(config: dict, output_dir: str, *, id: str, origin: str = "training") -> TrainRun:
    """Register a run under ``id`` with ``config`` and ``output_dir``, and return it. Mints no id
    and draws no seed: ``id`` and ``config["seed"]`` are taken as given."""
    run = TrainRun(id=id, config=config, output_dir=output_dir, origin=origin)
    with _RUNS_LOCK:
        _RUNS[id] = run
    return run


def get_run(id: str) -> TrainRun | None:
    with _RUNS_LOCK:
        return _RUNS.get(id)


def list_runs(include_hpo_trials: bool = False) -> list[dict]:
    """Return runs as dicts. HPO trial runs (``origin='hpo_trial'``) are excluded by
    default so a sweep's trials don't leak into the Training view; pass
    ``include_hpo_trials=True`` for the full registry."""
    with _RUNS_LOCK:
        runs = list(_RUNS.values())
    return [
        r.to_dict()
        for r in runs
        if include_hpo_trials or r.origin != "hpo_trial"
    ]


def cancel_run(id: str) -> bool:
    """Request a graceful cancellation of a training run. Returns False if unknown.

    A run whose training body executes in a subprocess (``run.pid is not None``) gets a sentinel
    file at ``<output_dir>/<CANCEL_SENTINEL>``, which ``TrainRun.should_cancel()`` polls in the
    child. When this process has no local record of the run at all, the output directory is read
    from the experiment record's own status, through the store; an id no record could carry (a path
    separator, an empty or dot name) and an unresolvable run are refused (``False``).
    """
    with _RUNS_LOCK:
        run = _RUNS.get(id)
    if run is not None:
        if run.pid is None:
            run.cancel_event.set()
        else:
            Path(run.output_dir).mkdir(parents=True, exist_ok=True)
            (Path(run.output_dir) / CANCEL_SENTINEL).touch()
        return True

    from tcip_store import BadKey

    from tcip_mcp.experiments import read_member, status_key

    try:
        key = status_key(id)
    except BadKey:
        return False
    status = read_member(key, {})
    output_dir = status.get("output_dir") if isinstance(status, dict) else None
    if not output_dir:
        return False
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / CANCEL_SENTINEL).touch()
    return True
