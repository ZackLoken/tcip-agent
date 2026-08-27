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
