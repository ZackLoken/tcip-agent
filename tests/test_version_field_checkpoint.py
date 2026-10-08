"""load_registered_checkpoint loads a registered checkpoint of the platform's own payload shape
once its digest has verified it."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")

from tcip_mcp.pipelines.model_build import CONFIG_KEY, STATE_DICT_KEY  # noqa: E402
from tests._verified_checkpoint_fixtures import BUILT_DETECTOR, register_checkpoint  # noqa: E402


def _bespoke_checkpoint(path: Path) -> str:
    """A real, unpicklable tcip checkpoint at path, the platform's own producer's shape."""
    from tcip_mcp.pipelines.model_build import build_model, recorded_model_dims

    config = {"model_source": dict(BUILT_DETECTOR),
              "data": {"num_channels": 3, "scope": {"subject": "bud", "attributes": []}}}
    payload = {
        CONFIG_KEY: config,
        STATE_DICT_KEY: build_model(config, recorded_model_dims(config)).state_dict(),
    }
    torch.save(payload, str(path))
    return str(path)


def test_a_version_one_checkpoint_loads_through_the_platforms_own_registration(tmp_path):
    from tcip_mcp.model_registry import load_registered_checkpoint

    ckpt = _bespoke_checkpoint(tmp_path / "m.pt")
    register_checkpoint(tmp_path, ckpt, name="version-one-model")

    verified = load_registered_checkpoint(ckpt, project=tmp_path)
    assert STATE_DICT_KEY in verified.payload
