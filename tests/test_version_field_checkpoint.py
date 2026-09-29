"""load_registered_checkpoint loads a registered checkpoint of the platform's own payload shape
once its digest has verified it."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")


def _bespoke_checkpoint(path: Path) -> str:
    """A real, unpicklable tcip checkpoint at path, the platform's own producer's shape."""
    from tcip_mcp.pipelines.model_build import build_model, recorded_model_dims

    model_source = {
        "builder": "tests.bespoke_models:build_bespoke_detection",
        "builder_kwargs": {"min_size": 64, "max_size": 128},
        "task": "detection",
    }
    config = {"model_source": model_source,
              "data": {"num_channels": 3, "scope": {"subject": "bud", "id_map": {"bud": 0}}}}
    payload = {
        "config": config,
        "model_state_dict": build_model(config, recorded_model_dims(config)).state_dict(),
    }
    torch.save(payload, str(path))
    return str(path)


def _register(tmp_path: Path, ckpt_path: str, name: str) -> None:
    from tcip_mcp.tools.model_tools import register_model

    result = register_model(name=name, checkpoint_path=ckpt_path, config={}, project_path=str(tmp_path))
    assert "error" not in result, result


def test_a_version_one_checkpoint_loads_through_the_platforms_own_registration(tmp_path):
    from tcip_mcp.model_registry import load_registered_checkpoint

    ckpt = _bespoke_checkpoint(tmp_path / "m.pt")
    _register(tmp_path, ckpt, "version-one-model")

    verified = load_registered_checkpoint(ckpt, project_path=str(tmp_path))
    assert "model_state_dict" in verified.payload
