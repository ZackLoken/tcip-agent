"""Fixtures for the checkpoint-digest rail: a real checkpoint registered through the platform's
own producer, and a stub ``VerifiedCheckpoint`` for a test that stubs a build.

``build_predictor`` and every measurement-path checkpoint load now take a
``tcip_mcp.model_registry.VerifiedCheckpoint`` from ``load_registered_checkpoint`` rather than a
bare path: a real-checkpoint test builds and registers one through :func:`registered_checkpoint`,
and a test that stubs ``build_predictor``/``load_registered_checkpoint`` builds the stub object
through :func:`stub_verified_checkpoint`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")


def registered_checkpoint(
    tmp_path: Path,
    *,
    project_root: str | Path,
    name: str = "test-model",
    model_source: dict | None = None,
    stamp: dict | None = None,
    filename: str = "model_best.pt",
) -> str:
    """Write a bespoke checkpoint through ``build_model``, register it via ``register_model``'s
    explicit mode against ``project_root``, and return its path.

    ``model_source`` defaults to a tiny detection builder; ``stamp`` merges extra top-level keys
    onto the saved payload (e.g. ``experiment_id``, tiling config) the way the trainer stamps one.
    """
    from tcip_mcp.pipelines.model_build import build_model
    from tcip_mcp.tools.model_tools import register_model

    src = model_source or {
        "builder": "tests.bespoke_models:build_bespoke_detection",
        "builder_kwargs": {"num_classes": 1, "min_size": 64, "max_size": 128},
        "task": "detection",
    }
    model = build_model({"model_source": src})
    payload: dict[str, Any] = {"model_source": src, "model_state_dict": model.state_dict()}
    if stamp:
        payload.update(stamp)
    ckpt_path = Path(tmp_path) / filename
    torch.save(payload, str(ckpt_path))
    result = register_model(name=name, checkpoint_path=str(ckpt_path), config={},
                            project_path=str(project_root))
    assert "error" not in result, result
    return str(ckpt_path)


def run_inference_verified(checkpoint_path: str, **overrides: Any):
    """The ephemeral in-memory pass, for a test that wants exactly what the old, non-persisting
    ``run_inference`` tool used to hand back before the merge folded it into the door that
    persists a bucket: loads the registered checkpoint and calls ``_run_inference_verified``
    directly, the same private pass the merged tool itself calls once it has resolved a bucket.

    ``overrides`` supplies whichever of the pass' own keyword arguments (``image_paths`` included,
    since the merged tool no longer exposes it) a test cares about; every other one takes the same
    unstated-sentinel default the old tool forwarded when a caller stated nothing. A checkpoint
    the registry refuses (``UnregisteredCheckpoint``) returns ``{"error": ...}``, the same catch
    every real caller of ``load_registered_checkpoint`` wraps it in, rather than raising out of
    this stand-in for one.
    """
    from tcip_mcp.model_registry import UnregisteredCheckpoint, load_registered_checkpoint
    from tcip_mcp.tools.inference_tools import _run_inference_verified

    try:
        checkpoint = load_registered_checkpoint(checkpoint_path)
    except UnregisteredCheckpoint as exc:
        return {"error": str(exc)}
    kwargs: dict[str, Any] = {
        "image_paths": None, "images_dir": None, "conf_threshold": None, "device": None,
        "tile": None, "tile_size": None, "overlap": None, "tile_batch_size": 96,
        "global_nms_iou": None, "max_dets": None, "postprocess": "nms", "trait": None,
        "calibration_labels_dir": None, "calibration_images_dir": None, "experiment_id": None,
        "group_by": None, "group_key_map": None, "split_seed": 0, "split_holdout_ratio": 0.5,
        "split_manifest_dir": None,
    }
    kwargs.update(overrides)
    return _run_inference_verified(checkpoint, **kwargs)


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
    (``path``, ``sha256``, ``entries``, ``producer``, and a ``payload`` carrying
    ``config["data"]``/``experiment_id``), with no registry lookup or file read behind it.
    ``kind`` stamps the payload's own ``kind`` key (the tcip module default) so
    ``build_predictor``'s kind sniff succeeds without needing a real ``model_source``/state dict;
    pass ``None`` for a test that means to exercise the kind-sniff failure itself.
    """
    from tcip_mcp.model_registry import VerifiedCheckpoint

    payload: dict[str, Any] = {"config": {"data": config_data or {}}}
    if kind is not None:
        payload["kind"] = kind
    if experiment_id is not None:
        payload["experiment_id"] = experiment_id
    entries = (entry,) if entry is not None else ()
    return VerifiedCheckpoint(
        path=path, sha256=sha256, payload=payload, entries=entries, producer=producer,
    )
