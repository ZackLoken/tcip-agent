"""Run and checkpoint fixtures built through the platform's own producers: a run the launcher's
own producer resolves and opens, run by the child's own entry to the final status it writes (whose
completion registers its checkpoint), and a foreign checkpoint registered through
``register_model``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

SCOPED_DATA = {"num_channels": 3, "scope": {"subject": "bud", "id_map": {"bud": 0}}}
"""A three-band detection run's data section, scoped to one subject."""

BUILT_DETECTOR = {
    "builder": "tests.bespoke_models:build_bespoke_detection",
    "builder_kwargs": {"min_size": 64, "max_size": 128},
    "task": "detection",
}
"""A tiny detection builder's ``model_source``."""


def detection_images(where: Path, scope: dict, *, n: int = 2, polygons: bool = False) -> dict:
    """A tiny detection dataset under ``where``: ``n`` three-band frames, each label document
    holding one box (``polygons``: one square polygon) of ``scope``'s subject, carrying the first
    value its ``id_map`` names under its ``attribute`` when it names one. Returns the
    ``images_dir`` and ``labels_dir`` a data section names it by."""
    from PIL import Image

    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox, Polygon

    images_dir, labels_dir = where / "images", where / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    attribute = scope.get("attribute")
    attributes = {attribute: next(iter(scope["id_map"]))} if attribute else {}
    geometry = (Polygon([[(8, 8), (24, 8), (24, 24), (8, 24)]]) if polygons
                else BBox(8, 8, 24, 24))
    for i in range(n):
        stem = f"frame{i}"
        Image.new("RGB", (64, 48), color=(60 + 40 * i, 90, 60)).save(images_dir / f"{stem}.png")
        json_io.write_annotations(
            str(labels_dir / f"{stem}.json"),
            [Annotation(subject=scope["subject"], geometry=geometry, attributes=attributes)],
            64, 48, keep_empty=True)
    return {"images_dir": str(images_dir), "labels_dir": str(labels_dir)}


def table_images(where: Path, *, n: int = 2) -> dict:
    """A tiny table-labeled dataset under ``where``: ``n`` three-band frames and a ``labels.csv``
    naming each by stem with a class label, alternating 0 and 1. Returns the ``images_dir`` and
    ``labels_dir`` a data section names it by."""
    import csv

    from PIL import Image

    images_dir = where / "images"
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


def detection_config(where: Path, **config: Any) -> dict:
    """A :data:`BUILT_DETECTOR` run's config over two :func:`detection_images` frames under
    ``where``, scoped as :data:`SCOPED_DATA`, with ``config``'s keys laid over it."""
    return {"model_source": dict(BUILT_DETECTOR),
            "data": {**detection_images(where, SCOPED_DATA["scope"]), **SCOPED_DATA}, **config}


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
    from tcip_mcp.tools.training_tools import open_run

    run_dir = experiments.experiment_dir(experiment_id or experiments.mint_experiment_id(),
                                         project=root)
    config = dict(config)
    open_run(run_dir, config, resolve_run(config, project=Path(root)).record,
             launched_by={"launcher": "process"}, **facts)
    return run_dir


def partition_side(partition: dict, side: str) -> list[str]:
    """The members a resolved partition records on ``side``, sorted, read through the platform's
    own partition reader."""
    from tcip_mcp.pipelines.data.split_construction import partition_samples

    return sorted(s.member for s in partition_samples(partition) if s.side == side)


def log_epoch(run_dir: Path, epoch: int, metrics: dict) -> None:
    """Append one epoch row to ``run_dir``'s metrics log through the envelope's own sink."""
    from tcip_mcp.experiments import RUN_FILE, project_of_run, read_record
    from tcip_mcp.pipelines.training.envelope import TrainContext
    from tcip_mcp.pipelines.training.run_registry import TrainRun

    record = read_record(run_dir / RUN_FILE)
    run = TrainRun(id=run_dir.name, config=record["config"],
                   objective=record["resolved"]["objective"], project=project_of_run(run_dir),
                   output_dir=str(run_dir))
    TrainContext(run=run, train_loader=None)._epoch_sink(epoch, metrics)


def finished_run(
    root: str | Path,
    *,
    experiment_id: str | None = None,
    model_source: dict | None = None,
    data: dict | None = None,
    metrics: dict | None = None,
    rows: list[dict] | None = None,
    training_source: str = "tests.bespoke_models:save_built_weights",
    wall_clock_passed: bool = False,
    cancel_requested: bool = False,
    seed: int | None = None,
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
    detection or instance_seg model, three bands otherwise. Returns the run directory."""
    from tcip_mcp import experiments
    from tcip_mcp.pipelines.training.subprocess_worker import run_directory

    run_id = experiment_id or experiments.mint_experiment_id()
    model_source = model_source or dict(BUILT_DETECTOR)
    task = model_source.get("task")
    geometric = task in ("detection", "instance_seg")
    stated = data or (dict(SCOPED_DATA) if geometric else {"num_channels": 3})
    if "images_dir" not in stated:
        where = fixture_data_dir(root, run_id)
        frames = (detection_images(where, stated["scope"], polygons=task == "instance_seg")
                  if geometric else table_images(where))
        stated = {**frames, **stated}
    config: dict[str, Any] = {
        "model_source": model_source,
        "data": stated,
        "training_source": training_source,
    }
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
    return opened_run(root, {"model_source": {"task": task}, "data": data},
                      experiment_id=experiment_id)


def worker_run(root: str | Path, config: dict, *,
               experiment_id: str | None = None) -> Path:
    """A run under ``root`` opened by :func:`opened_run` over ``config`` and run in-process
    through the child's own entry (``subprocess_worker.run_directory``): its loaders built from
    its record, its body run, its final status written. Returns the run directory."""
    from tcip_mcp.pipelines.training.subprocess_worker import run_directory

    run_dir = opened_run(root, config, experiment_id=experiment_id)
    run_directory(run_dir)
    return run_dir


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
    from tcip_mcp.tools.model_tools import register_model

    project_root = Path(project_root)
    path = registered_checkpoint(project_root.parent / f"{project_root.name}-elsewhere", **kwargs)
    result = register_model(project_root, name=name or f"model-{Path(path).parent.name}",
                            checkpoint_path=path, config={})
    assert "error" not in result, result
    return path


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


def run_inference_verified(project: Path, checkpoint_path: str, **overrides: Any):
    """``_run_inference_verified`` for ``project`` over the checkpoint registered at
    ``checkpoint_path``, with no bucket persisted: ``overrides`` sets any of its keyword
    arguments, the rest take ``run_inference``'s unstated defaults, and ``results`` comes back as
    a list. A checkpoint the registry refuses returns ``{"error": ...}``."""
    import inspect

    from tcip_mcp.model_registry import UnregisteredCheckpoint, load_registered_checkpoint
    from tcip_mcp.tools.inference_tools import _run_inference_verified, run_inference

    try:
        checkpoint = load_registered_checkpoint(checkpoint_path, project=Path(project))
    except UnregisteredCheckpoint as exc:
        return {"error": str(exc)}
    # Read off run_inference's own defaults rather than restate them, so this stand-in for its
    # private pass cannot drift from the door it stands in for.
    door_defaults = inspect.signature(run_inference).parameters
    kwargs: dict[str, Any] = {
        "images_dir": None, "conf_threshold": None, "device": None,
        "tile": None, "tile_size": None, "overlap": None,
        "tile_batch_size": door_defaults["tile_batch_size"].default,
        "cross_tile_nms": None, "max_dets": None, "postprocess": "nms", "trait": None,
        "calibration_labels_dir": None, "calibration_images_dir": None,
        "group_by": None, "group_key_map": None,
        "split_seed": door_defaults["split_seed"].default,
        "split_holdout_ratio": door_defaults["split_holdout_ratio"].default,
        "selection_dir": None,
    }
    kwargs.update(overrides)
    result = _run_inference_verified(Path(project), checkpoint, **kwargs)
    if "results" in result:
        result["results"] = list(result["results"])
    return result


def completed_checkpoint(run_dir: Path) -> dict | None:
    """The checkpoint ``run_dir``'s final status completed with, its ``path`` beside what it says
    of itself (``model_registry.entry_facts``), or ``None`` when the run did not complete one."""
    from tcip_mcp.experiments import observe
    from tcip_mcp.model_registry import entry_facts, run_entry

    entry = run_entry(observe(run_dir))
    return None if entry is None else {"path": entry["checkpoint_path"],
                                       "sha256": entry["sha256"], **entry_facts(entry)}


def checkpoint_file(path: Path, content: str) -> Path:
    """A checkpoint file at ``path`` the verified reader admits, holding ``content`` alone, in
    torch's pickle format so that one content always writes the same bytes. Returns ``path``."""
    torch.save({"content": content}, path, _use_new_zipfile_serialization=False)
    return path


def dummy_checkpoint(tmp_path: Path) -> str:
    """A checkpoint path that exists on disk and whose bytes are never read."""
    p = tmp_path / "m.pt"
    if not p.exists():
        p.write_bytes(b"x")
    return str(p)
