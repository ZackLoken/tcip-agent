"""In-process registry of live training runs: ``TrainRun``, its cancel-sentinel protocol, and
the create/attach/get/list/cancel operations over the process-global ``_RUNS`` map.
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
    # Set on resume, True if the checkpoint carried RNG state and it was restored, False
    # if the checkpoint carried none, the fresh-seed stream stands. None on a non-resumed run.
    rng_state_restored: bool | None = None
    # None means the loop runs in-process (cancel_event alone is authoritative); set once the
    # parent spawns the subprocess a run's body executes in, when should_cancel polls the sentinel.
    pid: int | None = None

    def should_cancel(self) -> bool:
        """True if cancellation was requested, in-process (``cancel_event``) or via the sentinel
        file a (possibly different) process may have written at ``<output_dir>/<CANCEL_SENTINEL>``.
        Checked unconditionally, not gated on ``pid`` being set: the object checking this is
        typically the child's own attached ``TrainRun`` (which has no reason to know its own OS
        pid), while ``pid`` is meaningful on the parent's copy for a different purpose (deciding
        whether ``cancel_run`` should set the in-memory ``Event`` or write the sentinel). The single
        check every poll site and every ``train(ctx)`` loop must use, so a sentinel-triggered stop is
        never invisible to a check written against the in-memory ``Event`` alone."""
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

    Never start an unseeded run. Called by ``launch_training`` before the experiment record is
    written (so the record's snapshot and a pristine reuse's config refresh both carry the seed
    the run actually trains with) and by the HPO trial dispatch on its own tracking config
    object, before that trial's own ``create_run``, so a swept ``seed`` still reads as consumed by
    the trainer.
    """
    if config.get("seed", config.get("training", {}).get("seed")) is None:
        config["seed"] = random.SystemRandom().randrange(2**31)
        logger.info("no seed configured; drew seed=%d.", config["seed"])


def create_run(config: dict, output_dir: str, *, id: str, origin: str = "training") -> TrainRun:
    """Register a run under ``id``, the caller's own: this mints nothing and draws no seed,
    both resolved by the caller before this is called (:func:`draw_seed_if_unset` for the seed;
    an experiment id, minted or caller-named, for ``id``). ``id`` is a training run's own
    experiment id, or an HPO trial's resolved trial-directory path; either way the caller
    supplies it, since nothing here can tell the two apart.
    """
    run = TrainRun(id=id, config=config, output_dir=output_dir, origin=origin)
    with _RUNS_LOCK:
        _RUNS[id] = run
    return run


def attach_run(id: str, config: dict, output_dir: str, origin: str = "training") -> TrainRun:
    """Construct a ``TrainRun`` for an id the caller already owns, exactly the shape
    ``create_run`` returns. Used by the subprocess worker to adopt the exact experiment id the
    parent already resolved and baked into ``output_dir``/``env.json``/audit events; nothing
    here draws a seed, since ``config`` (read back from the persisted ``config.json``) already
    carries the one the parent resolved. Inserts into *this process's own* ``_RUNS``, safe even
    though the id may already be a key in a different process's registry, since that's different
    process memory entirely.
    """
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

    A run whose training body executes in a subprocess (``run.pid is not None``) can't be
    stopped by setting an in-memory ``Event``, that memory lives in a different process. Writes a
    sentinel file at ``<output_dir>/<CANCEL_SENTINEL>`` instead, which ``TrainRun.should_cancel()``
    polls in the child. When this process has no local record of the run at all (it was launched by
    a *different* process, e.g. the web backend cancelling a run the agent's MCP server's
    subprocess is running), falls back to reading the experiment record's own status directly for
    the real output directory rather than guessing one, an id no record could ever carry (a path
    separator, an empty or dot name) folding to the same refusal an absent record gets; an
    unresolvable run is refused (``False``), never a silent write to a path nobody polls. The
    record is read through the store the launching process wrote it to, so a live run is still
    cancellable when that store is a database rather than a file beside the run.
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
