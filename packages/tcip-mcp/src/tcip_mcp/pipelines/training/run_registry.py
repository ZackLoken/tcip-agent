"""``TrainRun``, the state of one run's body in the process running it."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tcip_mcp.experiments import RunObservation
    from tcip_mcp.pipelines.schemas import TrainConfigSchema

logger = logging.getLogger(__name__)


@dataclass
class TrainRun:
    id: str
    config: dict
    # ``config`` validated once by the run's producer (``schemas.train_config``), the one typed
    # form every reader of the run's settings reads.
    spec: TrainConfigSchema
    # The run's resolved objective, ``{"selection_metric", "higher_is_better"}``
    # (``generic_trainer.resolve_objective``), read by the body and never resolved again.
    objective: dict
    # The project the run belongs to, obtained once by the process running it.
    project: Path
    status: str = "running"
    current_epoch: int = 0
    current_stage: int = 0
    # The held best epoch's selection value, which train() sets when it ends (the losing-side
    # infinity for the run's objective when no epoch was selectable).
    best_metric: float = float("inf")
    metrics_history: list[dict] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    # The error the run's final status will name (``experiments.write_final_status``).
    status_error: str = ""
    output_dir: str = ""
    # "training" (a launched run) vs "hpo_trial" (an HPO sweep's own trial run, which a cancel
    # of its sweep also stops).
    origin: str = "training"
    # The POSIX time past which the run stops as over its wall clock; None for no limit.
    deadline: float | None = None
    # Every checkpoint the run's body saved, by tag, and the one it declared its deliverable.
    saved: dict[str, Path] = field(default_factory=dict)
    deliverable: Path | None = None

    @property
    def wall_clock_exceeded(self) -> bool:
        """Whether the run has passed its :attr:`deadline`."""
        return self.deadline is not None and time.time() > self.deadline

    def should_cancel(self) -> bool:
        """True once a cancellation of the run's own directory or, for an HPO trial, of its
        sweep's directory is requested (``experiments.cancel_requested``), or once
        :attr:`wall_clock_exceeded`.
        """
        from tcip_mcp.experiments import cancel_requested

        if self.wall_clock_exceeded:
            return True
        if not self.output_dir:
            return False
        run_dir = Path(self.output_dir)
        directories = (run_dir, run_dir.parent) if self.origin == "hpo_trial" else (run_dir,)
        return any(cancel_requested(d) for d in directories)


def observed_run(observation: RunObservation, *, origin: str = "training") -> TrainRun:
    """The :class:`TrainRun` of an observed run, of ``origin``: named for its directory, training
    under its launch ``config`` with the ``data`` section its launch resolved laid over it,
    validated once (``schemas.train_config``, which refuses an invalid one), toward the objective
    that resolution carries. A run whose input resolved to nothing ended at its opening and has
    none."""
    from tcip_mcp.experiments import project_of_run
    from tcip_mcp.pipelines.schemas import train_config

    resolved = observation.resolution
    assert resolved is not None, "a run whose input resolved to nothing ended at its opening"
    config = {**observation.record["config"], "data": resolved["data"]}
    return TrainRun(id=observation.directory.name, config=config, spec=train_config(config),
                    objective=resolved["objective"], project=project_of_run(observation.directory),
                    output_dir=str(observation.directory), origin=origin)


def draw_seed_if_unset(config: dict) -> None:
    """Draw a seed from OS entropy into ``config`` in place, unless the caller already set one.
    Never start an unseeded run.
    """
    if config.get("seed") is None:
        config["seed"] = random.SystemRandom().randrange(2**31)
        logger.info("no seed configured; drew seed=%d.", config["seed"])
