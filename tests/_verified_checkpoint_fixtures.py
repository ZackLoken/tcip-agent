"""Fixtures for the checkpoint-digest rail: a real checkpoint registered through the platform's
own producer, and a stub ``VerifiedCheckpoint`` for a test that stubs a build.

``build_predictor`` and every measurement-path checkpoint load take a
``tcip_mcp.model_registry.VerifiedCheckpoint`` from ``load_registered_checkpoint``, never a
bare path: a real-checkpoint test builds and registers one through :func:`registered_checkpoint`,
and a test that stubs ``build_predictor``/``load_registered_checkpoint`` builds the stub object
through :func:`stub_verified_checkpoint`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

SCOPED_DATA = {"num_channels": 3, "scope": {"subject": "bud", "id_map": {"bud": 0}}}
"""A three-band detection run's recorded data section, scoped to one subject."""


def registered_checkpoint(
    tmp_path: Path,
    *,
    project_root: str | Path,
    name: str = "test-model",
    model_source: dict | None = None,
    data: dict | None = None,
    stamp: dict | None = None,
    filename: str = "model_best.pt",
) -> str:
    """Write a bespoke checkpoint through ``build_model``, register it via ``register_model``'s
    explicit mode against ``project_root``, and return its path.

    ``model_source`` defaults to a tiny detection builder. ``data`` is the run's own recorded data
    section the model is built at and the payload carries, by default a three-band run scoped to
    one subject; ``stamp`` merges extra top-level keys onto the saved payload (e.g.
    ``experiment_id``) the way the trainer stamps one.
    """
    from tcip_mcp.pipelines.model_build import build_model, recorded_model_dims
    from tcip_mcp.tools.model_tools import register_model

    src = model_source or {
        "builder": "tests.bespoke_models:build_bespoke_detection",
        "builder_kwargs": {"min_size": 64, "max_size": 128},
        "task": "detection",
    }
    config = {"model_source": src, "data": data or dict(SCOPED_DATA)}
    model = build_model(config, recorded_model_dims(config))
    payload: dict[str, Any] = {"config": config, "model_state_dict": model.state_dict()}
    if stamp:
        payload.update(stamp)
    ckpt_path = Path(tmp_path) / filename
    torch.save(payload, str(ckpt_path))
    result = register_model(name=name, checkpoint_path=str(ckpt_path), config={},
                            project_path=str(project_root))
    assert "error" not in result, result
    return str(ckpt_path)


def run_inference_verified(checkpoint_path: str, **overrides: Any):
    """The ephemeral in-memory pass, for a test that wants inference results with no bucket
    persisted: loads the registered checkpoint and calls ``_run_inference_verified``
    directly, the same private pass ``run_inference`` itself calls once it has resolved a bucket.

    ``overrides`` supplies whichever of the pass' own keyword arguments a test cares about; every
    other one takes the unstated-sentinel default the tool forwards when a caller states nothing.
    The pass' ``results`` stream is drawn into a list, the way a door with no bucket consumes it.
    A checkpoint the registry refuses (``UnregisteredCheckpoint``) returns ``{"error": ...}``.
    """
    import inspect

    from tcip_mcp.model_registry import UnregisteredCheckpoint, load_registered_checkpoint
    from tcip_mcp.tools.inference_tools import _run_inference_verified, run_inference

    try:
        checkpoint = load_registered_checkpoint(checkpoint_path)
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
        "calibration_labels_dir": None, "calibration_images_dir": None, "experiment_id": None,
        "group_by": None, "group_key_map": None,
        "split_seed": door_defaults["split_seed"].default,
        "split_holdout_ratio": door_defaults["split_holdout_ratio"].default,
        "selection_dir": None,
    }
    kwargs.update(overrides)
    result = _run_inference_verified(checkpoint, **kwargs)
    if "results" in result:
        result["results"] = list(result["results"])
    return result


def stub_verified_checkpoint(
    path: str,
    *,
    sha256: str = "stub-sha256",
    entry: dict | None = None,
    config_data: dict | None = None,
    experiment_id: str | None = None,
    producer: str | None = None,
    kind: str | None = "tcip_module",
):
    """A ``VerifiedCheckpoint``-shaped stub for a test that stubs ``build_predictor``/
    ``load_registered_checkpoint``: the fields a door under test reads off the object
    (``path``, ``sha256``, ``entries``, ``producer``, and a ``payload`` carrying a detection
    run's ``config``, its data section ``config_data`` or :data:`SCOPED_DATA`, and
    ``experiment_id``), with no registry lookup or file read behind it. ``kind`` stamps the
    payload's own ``kind`` key (the tcip module default) so ``build_predictor``'s kind sniff
    succeeds without needing a real builder or state dict; pass ``None`` for a test that means to
    exercise the kind-sniff failure itself.
    """
    from tcip_mcp.model_registry import VerifiedCheckpoint

    payload: dict[str, Any] = {"config": {"model_source": {"task": "detection"},
                                          "data": config_data or dict(SCOPED_DATA)}}
    if kind is not None:
        payload["kind"] = kind
    if experiment_id is not None:
        payload["experiment_id"] = experiment_id
    entries = (entry,) if entry is not None else ()
    return VerifiedCheckpoint(
        path=path, sha256=sha256, payload=payload, entries=entries, producer=producer,
    )
