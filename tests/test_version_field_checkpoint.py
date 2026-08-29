"""load_registered_checkpoint runs the schema-version check on the checkpoint payload once the
digest has already verified its producer, raising ValueError the same way an unrecognized kind
sniff does. No writer stamps a version yet (the field is lazy), so the refusal case stamps one
onto a real, registered checkpoint's payload directly rather than through a writer that does not
exist.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")


def _bespoke_checkpoint(path: Path, *, extra: dict | None = None) -> str:
    """A real, unpicklable tcip checkpoint at path, the platform's own producer's shape."""
    from tcip_mcp.pipelines.model_build import build_model

    model_source = {
        "builder": "tests.bespoke_models:build_bespoke_detection",
        "builder_kwargs": {"num_classes": 1, "min_size": 64, "max_size": 128},
        "task": "detection",
    }
    payload = {
        "model_source": model_source,
        "model_state_dict": build_model({"model_source": model_source}).state_dict(),
    }
    if extra:
        payload.update(extra)
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


def test_a_checkpoint_above_the_ceiling_refuses_as_a_value_error(tmp_path):
    from tcip_mcp.model_registry import load_registered_checkpoint

    ckpt = _bespoke_checkpoint(tmp_path / "m.pt", extra={"schema_version": 2})
    _register(tmp_path, ckpt, "version-two-model")

    with pytest.raises(ValueError):
        load_registered_checkpoint(ckpt, project_path=str(tmp_path))
