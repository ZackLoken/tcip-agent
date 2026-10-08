"""Run and checkpoint fixtures built through the platform's own producers: a run the launcher's
own producer resolves and opens, run by the child's own entry to the final status it writes (whose
completion registers its checkpoint), and a foreign checkpoint registered through
``register_model``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from tcip_mcp.dataset_layout import UNDATED_BUCKET
from tests._chain_fixtures import BESPOKE_DETECTION, SAVE_BUILT_WEIGHTS, training_config
from tests._training_values import sweep_space

torch = pytest.importorskip("torch")

SCOPED_DATA = {
    "num_channels": 3, "scope": {"subject": "bud"}, "split": {"seed": 0, "val_ratio": 0.15}
}
"""A three-band detection run's data section, stating the one subject it is scoped to and the
seed its own draw is made at; the admission reads the attributes the registry declares for it."""

BUILT_DETECTOR: dict[str, Any] = {
    "builder": BESPOKE_DETECTION,
    "builder_kwargs": {"min_size": 64, "max_size": 128},
    "task": "detection",
}
"""A tiny detection builder's ``model_source``."""


def built_detector(**builder_kwargs: Any) -> dict:
    """:data:`BUILT_DETECTOR` with ``builder_kwargs`` laid over its own builder keywords."""
    return {**BUILT_DETECTOR,
            "builder_kwargs": {**BUILT_DETECTOR["builder_kwargs"], **builder_kwargs}}


ONE_BAND_DETECTOR = built_detector(image_mean=[0.4], image_std=[0.2])
""":data:`BUILT_DETECTOR` normalizing one band, for a checkpoint whose data records one channel."""

SQUARE_64_DETECTOR = built_detector(max_size=64)
""":data:`BUILT_DETECTOR` with its resize bounds both at 64 px."""


def detection_images(where: Path, scope: dict, *, n: int = 2, polygons: bool = False,
                     values: dict[str, str] | None = None, registry: Any = None) -> dict:
    """A tiny detection dataset rooted at ``where``: ``n`` three-band frames, each label document
    holding one box (``polygons``: one square polygon) of ``scope``'s subject, carrying
    ``values`` as its attribute values, and ``registry`` as the dataset's subject registry when
    one is given. Returns the ``images_dir`` a data section names it by."""
    from tcip_annotation.state import Annotation, BBox, Polygon

    from tests._producer_fixtures import registry_over, seed_labeled_images

    if registry is not None:
        registry_over(where, registry)
    geometry = (Polygon([[(8, 8), (24, 8), (24, 24), (8, 24)]]) if polygons
                else BBox(8, 8, 24, 24))
    annotation = Annotation(subject=scope["subject"], geometry=geometry,
                            attributes=dict(values or {}))
    images_dir = seed_labeled_images(where / "images" / UNDATED_BUCKET, [annotation],
                                     n=n, width=64, height=48)
    return {"images_dir": str(images_dir)}


def table_images(where: Path, *, n: int = 2) -> dict:
    """A tiny table-labeled dataset under ``where``: ``n`` three-band frames and a ``labels.csv``
    naming each by stem with a class label, alternating 0 and 1. Returns the ``images_dir`` and
    ``labels_dir`` a data section names it by."""
    import csv

    from PIL import Image

    images_dir = where / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True, exist_ok=True)
    table = where / "labels.csv"
    with open(table, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("stem", "label"))
        for i in range(n):
            Image.new("RGB", (64, 48), color=(60 + 40 * i, 90, 60)).save(
                images_dir / f"frame{i}.png")
            writer.writerow((f"frame{i}", i % 2))
    return {"images_dir": str(images_dir), "labels_dir": str(table)}


def detection_config(where: Path, *, model_source: dict = BUILT_DETECTOR,
                     **config: Any) -> dict:
    """A :func:`~tests._chain_fixtures.training_config` of ``model_source`` (by default
    :data:`BUILT_DETECTOR`) over two :func:`detection_images` frames under ``where``, scoped as
    :data:`SCOPED_DATA`, with ``config``'s keys in place of their own."""
    from tests._chain_fixtures import training_config

    return training_config(
        model_source, {**detection_images(where, SCOPED_DATA["scope"]), **SCOPED_DATA},
        **config)


def fixture_data_dir(root: str | Path, name: str) -> Path:
    """Where a fixture run under the project ``root`` keeps the images it trains on: a directory
    beside the project named after it, so the project's own tree holds only what the platform
    wrote."""
    base = Path(root)
    return base.parent / f"{base.name}-data" / name


def opened_run(root: str | Path, config: dict, *, experiment_id: str | None = None,
               **facts: Any) -> Path:
    """A run directory under the project ``root`` resolved by the launcher's own producer
    (``split_construction.resolve_run``) and opened by its own writer (``training_tools.open_run``)
    over a copy of ``config``, launched by ``process``; ``facts`` are ``open_run``'s other
    keywords. Returns the directory."""
    from tcip_mcp import experiments
    from tcip_mcp.pipelines.data.split_construction import resolve_run
    from tcip_mcp.pipelines.schemas import train_config
    from tcip_mcp.tools.training_tools import open_run

    run_dir = experiments.experiment_dir(experiment_id or experiments.mint_experiment_id(),
                                         project=root)
    config = dict(config)
    open_run(run_dir, config,
             resolve_run(config, train_config(config), project=Path(root)).record, **facts)
    return run_dir


SWEEP_ARGUMENTS: dict[str, Any] = {
    "param_space": sweep_space(), "n_trials": 1,
    "search_alg": "random", "scheduler": "none", "grace_period": 1, "reduction_factor": 2,
    "baseline_params": None, "max_concurrent": 1,
    "resources_per_trial": None, "split_draws": 1, "split_draw_seeds": None, "search_seed": 0,
    "trial_budget": None, "relaunched_from": None,
}
"""A one-trial random sweep over the head learning rate, run to completion; ``open_sweep``'s other
arguments."""


def opened_sweep(root: str | Path, base_config: dict, **given: Any) -> Path:
    """A sweep directory under the project ``root`` opened by the sweep door's own producer
    (``training_tools.open_sweep``) over ``base_config``, :data:`SWEEP_ARGUMENTS` with ``given``
    laid over them. Returns the directory; its trials open through ``training_tools.open_trial``."""
    from tcip_mcp.tools.training_tools import open_sweep

    arguments = {**SWEEP_ARGUMENTS, **given}
    opened = open_sweep(Path(root), base_config, arguments.pop("param_space"), actor=None,
                        **arguments)
    assert isinstance(opened, Path), opened
    return opened


def partition_side(partition: dict, side: str) -> list[str]:
    """The members a resolved partition records on ``side``, sorted, read through the platform's
    own partition reader."""
    from tcip_mcp.pipelines.data.split_construction import partition_samples

    return sorted(s.member for s in partition_samples(partition) if s.side == side)


def log_epoch(run_dir: Path, epoch: int, metrics: dict) -> None:
    """Append one epoch row to ``run_dir``'s metrics log through the envelope's own sink."""
    from tcip_mcp.experiments import observe
    from tcip_mcp.pipelines.training.envelope import TrainContext
    from tcip_mcp.pipelines.training.run_registry import observed_run

    TrainContext(run=observed_run(observe(run_dir)), train_loader=None)._epoch_sink(epoch, metrics)


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
) -> Path:
    """A run under ``root`` opened by :func:`opened_run` over two frames of its own (unless
    ``data`` names its own ``images_dir``): :func:`detection_images` of its scope for a detection
    model, polygons for an ``instance_seg`` one, and :func:`table_images` for any other task; run
    by the child's own entry
    (``subprocess_worker.run_directory``) to the
    final status it writes. ``training_source`` is its body, by default one logging ``rows``
    (each an ``epoch`` plus its metrics) and saving the model its config builds under its seed
    with ``metrics`` as its checkpoint metrics, which completes it; ``seed`` states that seed
    (drawn when unset); ``wall_clock_passed`` launches it with a wall clock it has passed by the
    time it ends, and ``cancel_requested`` requests its cancellation before it starts.
    ``model_source`` defaults to :data:`BUILT_DETECTOR` and ``data`` to :data:`SCOPED_DATA` for a
    detection or instance_seg model, three bands otherwise; ``registry`` is the subject registry
    its own frames' dataset declares, so its admission records that registry's attributes. A run
    drawing its own split draws it at ``data.split.seed`` 0 unless ``data`` states one.
    Returns the run directory."""
    from tcip_mcp import experiments
    from tcip_mcp.pipelines.training.subprocess_worker import run_directory

    run_id = experiment_id or experiments.mint_experiment_id()
    model_source = model_source or dict(BUILT_DETECTOR)
    task = model_source.get("task")
    geometric = task in ("detection", "instance_seg")
    stated = data or (dict(SCOPED_DATA) if geometric else {"num_channels": 3})
    if "images_dir" not in stated:
        where = fixture_data_dir(root, run_id)
        frames = (detection_images(where, stated["scope"], polygons=task == "instance_seg",
                                   registry=registry)
                  if geometric else table_images(where))
        stated = {**frames, **stated}
    split = stated.get("split") or {}
    if "selection_dir" not in split:
        stated = {**stated, "split": {"seed": 0, "val_ratio": 0.15, **split}}
    config = training_config(model_source, stated, training_source=training_source)
    if metrics:
        config["fixture_metrics"] = metrics
    if rows:
        config["fixture_rows"] = rows
    if seed is not None:
        config["seed"] = seed
    run_dir = opened_run(root, config, experiment_id=run_id,
                         max_wall_clock_seconds=1e-9 if wall_clock_passed else None)
    if cancel_requested:
        experiments.request_cancel(run_dir)
    run_directory(run_dir)
    return run_dir


def resolved_run(root: str | Path, data: dict, *, task: str = "detection",
                 experiment_id: str | None = None) -> Path:
    """A run under ``root`` opened by :func:`opened_run` over ``data``, whose launch record holds
    the data section, partition and objective the launcher's producer resolved; no body runs.
    Returns the run directory."""
    return opened_run(root, training_config({"task": task}, data), experiment_id=experiment_id)


def worker_run(root: str | Path, config: dict, *,
               experiment_id: str | None = None) -> Path:
    """A run under ``root`` opened by :func:`opened_run` over ``config`` and run in-process
    through the child's own entry (``subprocess_worker.run_directory``): its loaders built from
    its record, its body run, its final status written. Returns the run directory."""
    from tcip_mcp.pipelines.training.subprocess_worker import run_directory

    run_dir = opened_run(root, config, experiment_id=experiment_id)
    run_directory(run_dir)
    return run_dir


def run_to_end(project: str | Path, experiment_id: str, *, seconds: float = 120) -> dict:
    """The row ``monitor_training`` answers for ``experiment_id`` once its state is terminal,
    polled for at most ``seconds``; fails the test naming the last row read otherwise."""
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
    pytest.fail(f"{experiment_id} reached no terminal state within {seconds}s: {row}")


def registered_checkpoint(project_root: str | Path, **kwargs: Any) -> str:
    """The path of the checkpoint a :func:`finished_run` under ``project_root`` registered by
    completing (``kwargs`` are its own)."""
    from tcip_mcp.experiments import observe

    observation = observe(finished_run(project_root, **kwargs))
    checkpoint = observation.checkpoint
    assert checkpoint is not None, observation.final
    return checkpoint["path"]


def foreign_checkpoint(project_root: str | Path, *, name: str | None = None,
                       **kwargs: Any) -> str:
    """A checkpoint a run of another project completed (a sibling of ``project_root`` named
    after it), registered into ``project_root`` under ``name`` (by default one naming the run)
    through ``register_model``'s explicit mode; its path. ``kwargs`` are :func:`finished_run`'s
    own."""
    project_root = Path(project_root)
    path = registered_checkpoint(project_root.parent / f"{project_root.name}-elsewhere", **kwargs)
    register_checkpoint(project_root, path, name=name or f"model-{Path(path).parent.name}")
    return path


def register_checkpoint(project_root: str | Path, path: str, *, name: str) -> None:
    """Register the checkpoint at ``path`` into ``project_root`` under ``name`` through
    ``register_model``'s explicit mode, asserting it admitted it."""
    from tcip_mcp.tools.model_tools import register_model

    result = register_model(project_root, name=name, checkpoint_path=path, config={})
    assert "error" not in result, result


_PROJECT_CHECKPOINTS: dict[str, str] = {}


def project_checkpoint(project_root: str | Path, **kwargs: Any) -> str:
    """One :func:`foreign_checkpoint` per project root and ``kwargs``, made on first call and
    answered again after: for a door whose inference pass a test stubs but whose checkpoint load
    is real."""
    root = Path(project_root)
    key = f"{root}|{sorted(kwargs.items())!r}"
    if key not in _PROJECT_CHECKPOINTS:
        _PROJECT_CHECKPOINTS[key] = foreign_checkpoint(root, **kwargs)
    return _PROJECT_CHECKPOINTS[key]


def verified_checkpoint(project_root: str | Path, **kwargs: Any):
    """The registry's own ``VerifiedCheckpoint`` over :func:`project_checkpoint`'s checkpoint."""
    from tcip_mcp.model_registry import load_registered_checkpoint

    return load_registered_checkpoint(project_checkpoint(project_root, **kwargs),
                                      project=Path(project_root))


SAMPLE_CROSS_TILE_NMS = 0.45
"""A sample IoU cross-tile merge threshold a test's tiled pass states."""
SAMPLE_CONF = 0.4
"""A sample confidence a test's detector pass states."""
SAMPLE_MAX_DETS = 300
"""A sample detection cap a test's detector pass states."""
SAMPLE_DETECTOR_PASS = {"conf": SAMPLE_CONF, "max_dets": SAMPLE_MAX_DETS,
                        "cross_tile_nms": SAMPLE_CROSS_TILE_NMS}
"""The execution values a test's detector pass states, by ``execution.Stated`` field."""


def tiled_record(*, tile_size: int, overlap: float, conf: float,
                 tile_resize: tuple[int, int] | None = None,
                 cross_tile_nms: float = SAMPLE_CROSS_TILE_NMS):
    """A detector's tiled execution record at ``tile_size``, ``overlap`` and ``tile_resize``
    (each stated), at ``conf`` and the sample cap, merging by NMS at the stated
    ``cross_tile_nms``, built by the producer a pass builds its own with
    (``execution.execution_record``)."""
    from types import SimpleNamespace

    from tcip_mcp.pipelines.execution import Stated, execution_record
    from tcip_mcp.pipelines.slicing import TileGeometry

    detector = SimpleNamespace(task="detection", path="detector.pt")
    geometry = TileGeometry(tile_size=tile_size, tile_size_source="explicit",
                            tile_size_derived_from=None, overlap=overlap,
                            overlap_source="explicit", tile_resize=tile_resize)
    stated = Stated(conf=conf, max_dets=SAMPLE_MAX_DETS, postprocess="nms",
                    cross_tile_nms=cross_tile_nms)
    return execution_record(cast(Any, detector), stated, geometry, None)


def predicted_over(project: Path, checkpoint_path: str, images_dir: str, *,
                   device: str | None = None, **stated: Any):
    """The pass ``run_inference`` prepares for the detector registered at ``checkpoint_path``
    over ``images_dir`` (``execution.prepare`` under the ``stated`` execution values over
    :data:`SAMPLE_DETECTOR_PASS`), run without publishing: ``(pass, results)``."""
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.execution import Stated, prepare

    p = prepare(load_registered_checkpoint(checkpoint_path, project=Path(project)),
                Stated(**{**SAMPLE_DETECTOR_PASS, **stated}),
                images_dir=images_dir, device=device).runnable()
    return p, [r for path in p.paths for r in p.predict([path])]


def completed_checkpoint(run_dir: Path) -> dict | None:
    """The checkpoint ``run_dir``'s final status completed with, its ``path`` beside what it says
    of itself (``model_registry.entry_facts``), or ``None`` when the run did not complete one."""
    from tcip_mcp.experiments import observe
    from tcip_mcp.model_registry import entry_facts, run_entry

    entry = run_entry(observe(run_dir))
    return None if entry is None else {"path": entry["checkpoint_path"],
                                       "sha256": entry["sha256"], **entry_facts(entry)}


def checkpoint_file(path: Path, content: str) -> Path:
    """A checkpoint file at ``path`` the verified reader admits, holding ``content`` beside an
    empty set of weights, in torch's pickle format so that one content always writes the same
    bytes. Returns ``path``."""
    from tcip_mcp.pipelines.model_build import STATE_DICT_KEY

    torch.save({"content": content, STATE_DICT_KEY: {}}, path,
               _use_new_zipfile_serialization=False)
    return path


def dummy_checkpoint(tmp_path: Path) -> str:
    """A checkpoint path that exists on disk and whose bytes are never read."""
    p = tmp_path / "m.pt"
    if not p.exists():
        p.write_bytes(b"x")
    return str(p)
