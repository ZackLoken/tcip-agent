"""``TrainRun``, the state of one run's body in the process running it."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tcip_mcp.experiments import RunObservation
    from tcip_mcp.pipelines.model_build import SourceLayout
    from tcip_mcp.pipelines.schemas import (
        Objective,
        TrainConfigSchema,
        TrainerReads,
        TrainGeometry,
    )

logger = logging.getLogger(__name__)


@dataclass
class TrainRun:
    id: str
    # The run's validated config.
    spec: TrainConfigSchema
    # The objective the run's launch resolved.
    objective: Objective
    # The geometry the run's launch read off its training frames.
    geometry: TrainGeometry
    # The project the run belongs to.
    project: Path
    # The run's laid-out source files.
    layout: SourceLayout
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

    @cached_property
    def reads(self) -> TrainerReads:
        """What training this run reads, its spec admitted once (``trainer_reads``, which
        refuses naming whatever is unstated).
        """
        return self.spec.trainer_reads()

    @cached_property
    def evaluation_arguments(self) -> dict:
        """The keywords ``evaluation.evaluate`` reads off this run, resolved once: the dims its
        config records (``model_build.recorded_model_dims``), the confidence it reads
        (:attr:`reads`), its stated ``evaluation.iou_threshold`` and ``score_weights``, the trait
        entry at the revision its :attr:`objective` records (read from :attr:`project`, ``None``
        for a run naming no trait; a revision whose entry no longer carries the recorded hash
        refuses with ``traits.RevisionMovedError``), and its :attr:`geometry`'s
        ``object_density``.
        """
        from tcip_mcp.pipelines.model_build import recorded_model_dims
        from tcip_mcp.traits import read_trait

        spec, ref = self.spec, self.objective.revision
        return {
            "dims": recorded_model_dims(spec),
            "conf_threshold": self.reads.conf_threshold,
            "iou_threshold": spec.evaluation.iou_threshold,
            "score_weights": spec.evaluation.score_weights,
            "trait": None
            if ref is None
            else read_trait(ref.trait, self.project)
            .revision(ref.trait_revision, ref.trait_revision_sha256)
            .entry,
            "density": self.geometry.object_density,
        }

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
        directories = (
            (run_dir, run_dir.parent)
            if self.origin == "hpo_trial"
            else (run_dir,)
        )
        return any(cancel_requested(d) for d in directories)


def observed_run(
    observation: RunObservation, *, origin: str = "training"
) -> TrainRun:
    """The :class:`TrainRun` of an observed run, of ``origin``: named for its directory, training
    under its validated launch config (``RunObservation.spec``) with the data block its launch
    resolved (``RunObservation.resolved_data``) laid over it and its source files bound to the
    run's own snapshot (``model_build.snapshot_bound``), which it imports from as the record
    states it (``model_build.run_layout``), toward the objective and at the geometry that
    resolution carries. A run whose input resolved to nothing ended at its opening and has none.
    """
    from tcip_mcp.experiments import project_of_run
    from tcip_mcp.pipelines.model_build import run_layout, snapshot_bound

    data, geometry = observation.resolved_data, observation.resolved_geometry
    assert data is not None and geometry is not None, (
        "a run whose input resolved to nothing ended at its opening"
    )
    spec = snapshot_bound(
        observation.spec.model_copy(update={"data": data}),
        observation.record["source"],
        observation.directory,
    )
    return TrainRun(
        id=observation.directory.name,
        spec=spec,
        objective=observation.objective,
        geometry=geometry,
        project=project_of_run(observation.directory),
        layout=run_layout(observation.directory, observation.record["source"]),
        output_dir=str(observation.directory),
        origin=origin,
    )


def seeded(spec: TrainConfigSchema) -> TrainConfigSchema:
    """``spec``, or when it states no ``seed`` a copy carrying one drawn from OS entropy, so no
    run starts unseeded.
    """
    if spec.seed is not None:
        return spec
    seed = random.SystemRandom().randrange(2**31)
    logger.info("no seed configured; drew seed=%d.", seed)
    return spec.model_copy(update={"seed": seed})
