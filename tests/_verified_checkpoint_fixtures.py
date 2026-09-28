"""Checkpoint fixtures: a real checkpoint registered through the platform's own producer, a stub
``VerifiedCheckpoint``, and a stubbed ``load_registered_checkpoint``."""

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
    """``_run_inference_verified`` over the registered checkpoint at ``checkpoint_path``, with no
    bucket persisted: ``overrides`` sets any of its keyword arguments, the rest take
    ``run_inference``'s unstated defaults, and ``results`` comes back as a list. A checkpoint the
    registry refuses returns ``{"error": ...}``."""
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


def admit_any_checkpoint(monkeypatch, *, file_digest: bool = False) -> None:
    """Stub ``load_registered_checkpoint`` to admit whatever path it is given as a
    :func:`stub_verified_checkpoint`. With ``file_digest``, a path naming a file carries the
    digest of its bytes; otherwise every path carries ``"stub-sha256"``."""
    import tcip_mcp.model_registry as model_registry_mod

    def _stub(path, *a, **kw):
        p = Path(path)
        sha = (model_registry_mod._sha256_of_bytes(p.read_bytes())
               if file_digest and p.is_file() else "stub-sha256")
        return stub_verified_checkpoint(str(path), sha256=sha)

    monkeypatch.setattr(model_registry_mod, "load_registered_checkpoint", _stub)


def dummy_checkpoint(tmp_path: Path) -> str:
    """A checkpoint path that exists on disk and whose bytes are never read."""
    p = tmp_path / "m.pt"
    if not p.exists():
        p.write_bytes(b"x")
    return str(p)
