"""load_registered_checkpoint loads a checkpoint the platform registered.

The checkpoint is of the platform's own payload shape, and the loaded payload holds its state
dict.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tcip_mcp.pipelines.model_build import STATE_DICT_KEY  # noqa: E402
from tests._verified_checkpoint_fixtures import registered_checkpoint  # noqa: E402


def test_a_version_one_checkpoint_loads_through_the_platforms_own_registration(
    tmp_path,
):
    from tcip_mcp.model_registry import load_registered_checkpoint

    ckpt = registered_checkpoint(tmp_path)

    verified = load_registered_checkpoint(ckpt, project=tmp_path)
    assert STATE_DICT_KEY in verified.payload
