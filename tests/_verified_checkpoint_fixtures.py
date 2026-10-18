"""Run and checkpoint fixtures built through the platform's own producers.

They hold a run the launcher's own producer resolves and opens, run by the child's own entry to
the final status it writes (whose completion registers its checkpoint), and a foreign checkpoint
registered through ``register_model``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from tcip_mcp.dataset_layout import UNDATED_BUCKET
from tests._chain_fixtures import (
    BESPOKE_DETECTION,
    BESPOKE_MODELS,
    FRAME_STATISTICS,
    SAVE_BUILT_WEIGHTS,
    training_config,
)
from tests._training_values import (
    MASK_CUT,
    RATIONALE,
    drawn_split,
    sweep_space,
)

torch = pytest.importorskip("torch")

SCOPED_DATA = {
    "num_channels": 3,
    "scope": {"subject": "bud"},
    "split": drawn_split(val_ratio=0.15, grouping="tile_prefix"),
}
"""A three-band detection run's data section, scoped to ``bud`` and drawing its own split at a
0.15 validation ratio grouped by tile prefix."""

BUILT_DETECTOR: dict[str, Any] = {
    "builder": BESPOKE_DETECTION,
    "source_files": [BESPOKE_MODELS],
    "task": "detection",
    "normalization": FRAME_STATISTICS,
}
"""A tiny detection builder's ``model_source``, normalizing by its own frames' statistics."""

SAVE_A_SECOND_CHECKPOINT = (
    "bespoke_models:save_built_weights_and_a_second_checkpoint"
)
"""A ``training_source`` saving the built weights as the deliverable and again beside it."""


def built_detector(**builder_kwargs: Any) -> dict:
    """Return :data:`BUILT_DETECTOR` with ``builder_kwargs`` as its builder keywords."""
    return {**BUILT_DETECTOR, "builder_kwargs": builder_kwargs}


def detector_declaring(*files: str) -> dict:
    """Return :data:`BUILT_DETECTOR` whose ``source_files`` also declare ``files``."""
    return {
        **BUILT_DETECTOR,
        "source_files": [*BUILT_DETECTOR["source_files"], *files],
    }


def detection_images(
    where: Path,
    scope: dict,
    *,
    n: int = 2,
    polygons: bool = False,
    values: dict[str, str] | None = None,
    registry: Any = None,
) -> dict:
    """Write ``n`` labeled frames under ``where`` through ``seed_labeled_images``.

    The frames go to ``where/images/<undated>`` at 64 by 48, each labeled with one annotation of
    ``scope``'s subject on the box (8, 8, 24, 24), or that square as a polygon when
    ``polygons``, carrying ``values`` as its attributes. ``registry``, when given, first
    replaces ``where``'s subject registry. Returns ``{"images_dir": ...}``.
    """
    from tcip_annotation.state import Annotation, BBox, Polygon
    from tests._producer_fixtures import registry_over, seed_labeled_images

    if registry is not None:
        registry_over(where, registry)
    geometry = (
        Polygon([[(8, 8), (24, 8), (24, 24), (8, 24)]])
        if polygons
        else BBox(8, 8, 24, 24)
    )
    annotation = Annotation(
        subject=scope["subject"],
        geometry=geometry,
        attributes=dict(values or {}),
    )
    images_dir = seed_labeled_images(
        where / "images" / UNDATED_BUCKET,
        [annotation],
        n=n,
        width=64,
        height=48,
    )
    return {"images_dir": str(images_dir)}


def table_images(where: Path, *, n: int = 2) -> dict:
    """Write a tiny table-labeled dataset under ``where``.

    :func:`~tests.tiny_trainer_fixtures.write_regression_dataset` writes ``n`` frames 64 wide by
    48 high at ``distinct_intensities(n)``, labeled ``i % 2``. Returns its ``images_dir`` and
    ``labels_dir``.
    """
    from tests.tiny_trainer_fixtures import (
        distinct_intensities,
        write_regression_dataset,
    )

    images_dir, table = write_regression_dataset(
        where,
        distinct_intensities(n),
        [i % 2 for i in range(n)],
        height=48,
        width=64,
    )
    return {"images_dir": str(images_dir), "labels_dir": str(table)}


def detection_config(
    where: Path,
    *,
    project: Path,
    model_source: dict = BUILT_DETECTOR,
    split: dict | None = None,
    **config: Any,
) -> dict:
    """Return a :func:`~tests._chain_fixtures.training_config` over two detection frames.

    The config names ``model_source`` (by default :data:`BUILT_DETECTOR`) under ``project``
    over two :func:`detection_images` frames under ``where``, scoped as :data:`SCOPED_DATA` and
    drawing ``split`` when one is given, with ``config``'s keys in place of their own.
    """
    from tests._chain_fixtures import training_config

    return training_config(
        project,
        model_source,
        {
            **detection_images(where, SCOPED_DATA["scope"]),
            **SCOPED_DATA,
            **({} if split is None else {"split": split}),
        },
        **config,
    )


def fixture_data_dir(root: str | Path, name: str) -> Path:
    """Return ``<root>-data/<name>``, a directory beside the project ``root`` for fixture images."""
    base = Path(root)
    return base.parent / f"{base.name}-data" / name


def opened_run(
    root: str | Path,
    config: dict,
    *,
    experiment_id: str | None = None,
    **facts: Any,
) -> Path:
    """Open a run directory under the project ``root`` over ``config`` validated.

    Its sources are staged a single time (``model_build.staged_sources``), resolved by the
    launcher's own producer (``split_construction.resolve_run``) and opened by its own writer
    (``training_tools.open_run``); ``facts`` are ``open_run``'s other keywords. Returns the
    directory.
    """
    from tcip_mcp import experiments
    from tcip_mcp.pipelines.data.split_construction import resolve_run
    from tcip_mcp.pipelines.model_build import staged_sources
    from tcip_mcp.pipelines.schemas import train_config
    from tcip_mcp.tools.training_tools import open_run

    run_dir = experiments.experiment_dir(
        experiment_id or experiments.mint_experiment_id(), project=root
    )
    spec = train_config(config)
    sources = staged_sources(spec, Path(root))
    resolution = resolve_run(spec, sources.layout, project=Path(root))
    open_run(run_dir, resolution.spec, sources, resolution.record, **facts)
    return run_dir


SWEEP_ARGUMENTS: dict[str, Any] = {
    "param_space": sweep_space(),
    "n_trials": 1,
    "search_alg": "random",
    "scheduler": {"name": "fifo"},
    "baseline_params": None,
    "max_concurrent": 1,
    "resources_per_trial": None,
    "split_draws": 1,
    "split_draw_seeds": None,
    "search_seed": 0,
    "trial_budget": None,
    "relaunched_from": None,
}
"""A one-trial random sweep over the head learning rate, run to completion; ``open_sweep``'s other
arguments."""


def opened_sweep(root: str | Path, base_config: dict, **given: Any) -> Path:
    """Open a sweep directory under the project ``root`` through the sweep door's own producer.

    ``training_tools.open_sweep`` opens it over ``base_config``, :data:`SWEEP_ARGUMENTS` with
    ``given`` laid over them. Returns the directory.
    """
    from tcip_mcp.tools.training_tools import open_sweep

    arguments = {**SWEEP_ARGUMENTS, **given}
    opened = open_sweep(
        Path(root),
        base_config,
        arguments.pop("param_space"),
        actor=None,
        **arguments,
    )
    assert isinstance(opened, Path), opened
    return opened


def partition_side(partition: dict, side: str) -> list[str]:
    """Return the members a resolved partition records on ``side``, sorted.

    They are read through ``split_construction.partition_samples``.
    """
    from tcip_mcp.pipelines.data.split_construction import partition_samples

    return sorted(
        s.member for s in partition_samples(partition) if s.side == side
    )


def log_epoch(run_dir: Path, epoch: int, metrics: dict) -> None:
    """Append one epoch row to ``run_dir``'s metrics log through the envelope's own sink."""
    from tcip_mcp.experiments import observe
    from tcip_mcp.pipelines.training.envelope import TrainContext
    from tcip_mcp.pipelines.training.run_registry import observed_run

    TrainContext(
        run=observed_run(observe(run_dir)), train_loader=None
    )._epoch_sink(epoch, metrics)


def finished_run(
    root: str | Path,
    *,
    experiment_id: str | None = None,
    model_source: dict | None = None,
    data: dict | None = None,
    metrics: dict | None = None,
    rows: list[dict] | None = None,
    training_source: str = SAVE_BUILT_WEIGHTS,
    wall_clock_passed: bool = False,
    cancel_requested: bool = False,
    seed: int | None = None,
    registry: Any = None,
    **config: Any,
) -> Path:
    """Open a run under ``root`` and run it by the child's own entry to its final status.

    :func:`opened_run` opens it over two frames of its own (unless ``data`` names its own
    ``images_dir``): :func:`detection_images` of its scope for a detection model, polygons for an
    ``instance_seg`` one, and :func:`table_images` for any other task; the child's own entry
    (``subprocess_worker.run_directory``) runs it to the final status it writes.
    ``training_source`` is its body, by default one logging ``rows`` (each an ``epoch`` plus its
    metrics) and saving the model its config builds under its seed with ``metrics`` as its
    checkpoint metrics; ``seed`` states that seed (the sample config's when unset);
    ``wall_clock_passed`` opens it with ``max_wall_clock_seconds`` 1e-9, and
    ``cancel_requested`` requests its cancellation before the entry runs it.
    ``model_source`` defaults to :data:`BUILT_DETECTOR` and ``data`` to :data:`SCOPED_DATA` for a
    detection or instance_seg model, three bands otherwise; ``registry`` is the subject registry
    its own frames' dataset declares. A run naming no selection draws its split at validation
    ratio 0.15, grouped by tile prefix, at seed 0, with ``data.split`` over that, and
    ``config``'s keys take the place of its config's own. Returns the run directory.
    """
    from tcip_mcp import experiments
    from tcip_mcp.pipelines.model_contract import DETECTOR_TASKS
    from tcip_mcp.pipelines.training.subprocess_worker import run_directory

    run_id = experiment_id or experiments.mint_experiment_id()
    model_source = model_source or dict(BUILT_DETECTOR)
    task = model_source.get("task")
    geometric = task in DETECTOR_TASKS
    stated = data or (dict(SCOPED_DATA) if geometric else {"num_channels": 3})
    if "images_dir" not in stated:
        where = fixture_data_dir(root, run_id)
        frames = (
            detection_images(
                where,
                stated["scope"],
                polygons=task == "instance_seg",
                registry=registry,
            )
            if geometric
            else table_images(where)
        )
        stated = {**frames, **stated}
    split = stated.get("split") or {}
    if "selection_dir" not in split:
        stated = {
            **stated,
            "split": drawn_split(
                **{"val_ratio": 0.15, "grouping": "tile_prefix", **split}
            ),
        }
    run_config = training_config(
        Path(root),
        model_source,
        stated,
        training_source=training_source,
        **config,
    )
    if metrics:
        run_config["fixture_metrics"] = metrics
    if rows:
        run_config["fixture_rows"] = rows
    if seed is not None:
        run_config["seed"] = seed
    run_dir = opened_run(
        root,
        run_config,
        experiment_id=run_id,
        max_wall_clock_seconds=1e-9 if wall_clock_passed else None,
    )
    if cancel_requested:
        experiments.request_cancel(run_dir)
    run_directory(run_dir)
    return run_dir


def resolved_run(
    root: str | Path,
    data: dict,
    *,
    task: str = "detection",
    experiment_id: str | None = None,
) -> Path:
    """Open a run under ``root`` of a ``task`` model (:func:`unbuilt_source`) and run no body.

    :func:`opened_run` opens it over a :func:`~tests._chain_fixtures.training_config` with
    ``data`` as its data section. Returns the run directory.
    """
    return opened_run(
        root,
        training_config(Path(root), unbuilt_source(task), data),
        experiment_id=experiment_id,
    )


def unbuilt_source(task: str) -> dict:
    """Return a ``task`` ``model_source`` naming the tiny detection builder.

    Its normalization is the statistics of the train side's frames
    (:data:`~tests._chain_fixtures.FRAME_STATISTICS`).
    """
    return {
        "builder": BESPOKE_DETECTION,
        "source_files": [BESPOKE_MODELS],
        "task": task,
        "normalization": FRAME_STATISTICS,
    }


def worker_run(
    root: str | Path, config: dict, *, experiment_id: str | None = None
) -> Path:
    """Open a run under ``root`` over ``config`` and run it in-process through the child's entry.

    :func:`opened_run` opens it and ``subprocess_worker.run_directory`` runs it: its loaders
    built from its record, its body run, its final status written. Returns the run directory.
    """
    from tcip_mcp.pipelines.training.subprocess_worker import run_directory

    run_dir = opened_run(root, config, experiment_id=experiment_id)
    run_directory(run_dir)
    return run_dir


def run_to_end(
    project: str | Path, experiment_id: str, *, seconds: float = 120
) -> dict:
    """Return the row ``monitor_training`` answers for ``experiment_id`` when it is terminal.

    The row is polled for at most ``seconds``; the test fails naming the last row read
    otherwise.
    """
    import time

    from tcip_mcp.experiments import TERMINAL_STATES
    from tcip_mcp.tools.training_tools import monitor_training

    deadline = time.monotonic() + seconds
    row: dict = {}
    while time.monotonic() < deadline:
        row = monitor_training(Path(project), experiment_id)["run"]
        if row.get("state") in TERMINAL_STATES:
            return row
        time.sleep(0.2)
    pytest.fail(
        f"{experiment_id} reached no terminal state within {seconds}s: {row}"
    )


def registered_checkpoint(project_root: str | Path, **kwargs: Any) -> str:
    """Return the path of the checkpoint a :func:`finished_run` registered by completing.

    The run is under ``project_root``; ``kwargs`` are its own.
    """
    from tcip_mcp.experiments import observe

    observation = observe(finished_run(project_root, **kwargs))
    checkpoint = observation.checkpoint
    assert checkpoint is not None, observation.final
    return checkpoint["path"]


def foreign_checkpoint(
    project_root: str | Path, *, name: str | None = None
) -> str:
    """Register a :func:`secondary_checkpoint` of ``project_root`` into it and return its path.

    It is registered under ``name`` (by default ``model-<run directory name>``) through
    ``register_model``'s explicit mode.
    """
    path = secondary_checkpoint(Path(project_root))
    register_checkpoint(
        project_root, path, name=name or f"model-{Path(path).parent.name}"
    )
    return path


def secondary_checkpoint(project_root: Path, **config: Any) -> str:
    """Return the path of a checkpoint no completion registered.

    It is the second one a :func:`finished_run` of ``project_root`` (``config``'s keys taking
    the place of its config's own) saved through its own writer
    (``TrainContext.save_checkpoint``) under :data:`~tests.bespoke_models.SECOND_CHECKPOINT_TAG`,
    beside the deliverable its completion registered.
    """
    from tcip_mcp.pipelines.training.generic_trainer import checkpoint_path
    from tests.bespoke_models import SECOND_CHECKPOINT_TAG

    run_dir = finished_run(
        project_root, training_source=SAVE_A_SECOND_CHECKPOINT, **config
    )
    return str(checkpoint_path(run_dir, SECOND_CHECKPOINT_TAG))


_PRODUCED: dict[str, bytes] = {}
"""The bytes of the checkpoint one completed run produced per content string, for the session."""


def produced_checkpoint(path: Path, content: str) -> Path:
    """Write at ``path`` a checkpoint that a run completed under a root of its own.

    The bytes are a :func:`registered_checkpoint`'s from a temporary project, produced once per
    ``content`` for the session and reused for the same ``content``; ``content`` itself is not
    written into them. Returns ``path``.
    """
    import tempfile

    import tcip_store as ts

    if content not in _PRODUCED:
        with tempfile.TemporaryDirectory() as producer:
            _PRODUCED[content] = Path(
                registered_checkpoint(Path(producer) / "project")
            ).read_bytes()
            ts.release_root(Path(producer))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_PRODUCED[content])
    return path


def register_checkpoint(
    project_root: str | Path, path: str, *, name: str
) -> None:
    """Register the checkpoint at ``path`` into ``project_root`` under ``name``.

    The registration goes through ``register_model``'s explicit mode, asserting it admitted it.
    """
    from tcip_mcp.tools.model_tools import register_model

    result = register_model(project_root, name=name, checkpoint_path=path)
    assert "error" not in result, result


_PROJECT_CHECKPOINTS: dict[str, str] = {}


def project_checkpoint(project_root: str | Path, **kwargs: Any) -> str:
    """Return one :func:`registered_checkpoint` per project root and ``kwargs``.

    It is made on first call and answered again after.
    """
    root = Path(project_root)
    key = f"{root}|{sorted(kwargs.items())!r}"
    if key not in _PROJECT_CHECKPOINTS:
        _PROJECT_CHECKPOINTS[key] = registered_checkpoint(root, **kwargs)
    return _PROJECT_CHECKPOINTS[key]


def verified_checkpoint(project_root: str | Path, **kwargs: Any):
    """Return the registry's own ``VerifiedCheckpoint`` over :func:`project_checkpoint`'s."""
    from tcip_mcp.model_registry import load_registered_checkpoint

    return load_registered_checkpoint(
        project_checkpoint(project_root, **kwargs), project=Path(project_root)
    )


SAMPLE_CROSS_TILE_NMS = 0.45
"""A sample IoU cross-tile merge threshold for a tiled pass."""
SAMPLE_CONF = 0.4
"""A sample confidence for a detector pass."""
SAMPLE_OVERLAP = 0.25
"""A sample tile overlap for a tiled pass."""
SAMPLE_DETECTOR_PASS = {
    "conf": SAMPLE_CONF,
    "postprocess": "nms",
    "cross_tile_nms": SAMPLE_CROSS_TILE_NMS,
    "rationale": RATIONALE,
}
"""Sample execution values for a detector pass, by ``execution.Stated`` field."""
SAMPLE_INSTANCE_SEG_PASS = {
    **SAMPLE_DETECTOR_PASS,
    "masks": True,
    "mask_cut": MASK_CUT,
}
"""Sample execution values for an instance segmentation pass: a detector's, ``masks`` on, and
the sample mask cut."""


def objects_over(boxes: list[list[float]], area: float):
    """Return the region ``datasets.object_region`` makes of one-class objects at ``boxes``.

    ``boxes`` are xyxy, none a crowd region, over ``area`` pixels.
    """
    from tcip_mcp.pipelines.data.datasets import object_region

    return object_region(
        {
            "boxes": boxes,
            "labels": [1] * len(boxes),
            "iscrowd": [0] * len(boxes),
        },
        area,
    )


def tiled_record(
    *,
    tile_size: int,
    overlap: float,
    conf: float,
    tile_resize: tuple[int, int] | None = None,
    cross_tile_nms: float = SAMPLE_CROSS_TILE_NMS,
):
    """Return a detector's tiled execution record at ``tile_size``, ``overlap`` and ``tile_resize``.

    Each of those is stated; the record states ``conf`` and a merge by NMS at the stated
    ``cross_tile_nms``, its density derived from a reference counting a hundred objects in one
    tile, built by ``execution.execution_record``.
    """
    from types import SimpleNamespace

    from tcip_mcp.pipelines.execution import (
        Reference,
        Stated,
        execution_record,
    )
    from tcip_mcp.pipelines.slicing import TileGeometry

    detector = SimpleNamespace(task="detection", path="detector.pt")
    geometry = TileGeometry(
        tile_size=tile_size,
        tile_size_source="explicit",
        overlap=overlap,
        overlap_source="explicit",
        tile_resize=tile_resize,
    )
    stated = Stated(
        conf=conf,
        postprocess="nms",
        cross_tile_nms=cross_tile_nms,
        rationale=RATIONALE,
    )
    reference = Reference(
        regions=[
            objects_over([[0.0, 0.0, 1.0, 1.0]] * 100, tile_size * tile_size)
        ]
    )
    return execution_record(cast(Any, detector), stated, geometry, reference)


def predicted_over(
    project: Path,
    checkpoint_path: str,
    images_dir: str,
    *,
    device: str | None = None,
    **stated: Any,
):
    """Prepare and run a pass of the detector registered at ``checkpoint_path``, publishing nothing.

    The pass reads ``images_dir`` (``execution.prepare`` under the ``stated`` execution values
    over :data:`SAMPLE_DETECTOR_PASS`) and predicts each of its paths. Returns
    ``(pass, results)``.
    """
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.execution import Stated, prepare

    p = prepare(
        load_registered_checkpoint(checkpoint_path, project=Path(project)),
        Stated(**{**SAMPLE_DETECTOR_PASS, **stated}),
        images_dir=images_dir,
        device=device,
    ).runnable()
    return p, [r for path in p.paths for r in p.predict([path])]


def completed_checkpoint(run_dir: Path) -> dict | None:
    """Return the checkpoint ``run_dir``'s final status completed with.

    The checkpoint's ``path`` stands beside what it says of itself
    (``model_registry.entry_facts``); ``None`` when the run did not complete one.
    """
    from tcip_mcp.experiments import observe, project_of_run
    from tcip_mcp.model_registry import entry_facts, run_entry

    entry = run_entry(observe(run_dir))
    return (
        None
        if entry is None
        else {
            "path": entry["checkpoint_path"],
            "sha256": entry["sha256"],
            **entry_facts(entry, project_of_run(run_dir)),
        }
    )
