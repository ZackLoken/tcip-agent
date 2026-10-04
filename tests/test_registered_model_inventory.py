"""What the registry holds, and where it holds it.

The registered-model inventory of a project is its registry index record: registration records a checkpoint where it already lives instead of copying it under the registry
directory, and a checkpoint registered again supersedes its earlier entry rather than adding a
second one. Both facts decide what any reader counting a project's models can honestly report.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

import pytest

import tcip_store as ts
from tcip_mcp.model_registry import ModelRegistry, registry_index_key
from tcip_mcp.pipelines.data.label_queries import registry_scope

pytest.importorskip("torch")

from tests._verified_checkpoint_fixtures import checkpoint_file  # noqa: E402

# Distinct payloads per run, so a hash attributed to the wrong entry is visible.
_RUNS = {
    "currant_bud_detector_v1": "run-a-weights",
    "chestnut_leaf_area_seg_v2": "run-b-weights-with-a-longer-payload",
    "currant_bush_detector_v1": "run-c",
}


def test_registered_models_are_recorded_in_the_index_not_copied_into_the_registry_dir(
    tmp_path: Path,
) -> None:
    """Registration leaves each checkpoint under its own run directory and records it in the
    index. The registry directory holds the index alone, so counting checkpoint files there
    counts nothing and the index is the only source of a project's registered-model inventory."""
    root = tmp_path / "proj"
    reg = ModelRegistry(str(root))
    for name, content in _RUNS.items():
        run_dir = root / ".tcip" / "experiments" / name / "artifacts"
        run_dir.mkdir(parents=True)
        ckpt = checkpoint_file(run_dir / "model_best.pt", content)
        reg.register_model(name, str(ckpt), {"data": {"scope": asdict(registry_scope(root, "bud"))}},
                           metrics={"val_map50": 0.42})

    models_dir = root / ".tcip" / "models"
    assert ts.exists(registry_index_key(root))
    assert list(models_dir.glob("*.pt")) == []

    reread = ModelRegistry(str(root)).list_models()
    assert {m["name"] for m in reread} == set(_RUNS)
    for entry in reread:
        ckpt = Path(entry["checkpoint_path"])
        assert ckpt.is_file()
        assert ckpt.parent.parent.parent.name == "experiments"
        assert ckpt.parent.parent.name == entry["name"]
        assert entry["sha256"] == hashlib.sha256(ckpt.read_bytes()).hexdigest()


def test_registering_a_checkpoint_again_supersedes_its_earlier_entry(tmp_path: Path) -> None:
    """A checkpoint registered again, under its old name or another, replaces that checkpoint's
    one entry; two different checkpoints are two entries whatever they are named. The inventory
    keeps one entry per checkpoint, so a reader counting registered models never counts one
    checkpoint twice."""
    root = tmp_path / "proj"
    root.mkdir()
    reg = ModelRegistry(str(root))

    ckpt_v1 =checkpoint_file(tmp_path / "model_epoch8.pt", "epoch-8-weights")
    ckpt_v2 = checkpoint_file(tmp_path / "model_epoch19.pt", "epoch-19-weights-after-resume")
    companion = checkpoint_file(tmp_path / "leaf_model.pt", "a separate run")

    reg.register_model("currant_bud_detector_v1", str(ckpt_v1), {},
                       metrics={"val_map50": 0.61})
    reg.register_model("chestnut_leaf_area_seg_v2", str(companion), {},
                       metrics={"val_map50": 0.55})
    reg.register_model("currant_bud_detector_v1_final", str(ckpt_v1), {},
                       metrics={"val_map50": 0.74})
    reg.register_model("currant_bud_detector_v1", str(ckpt_v2), {},
                       metrics={"val_map50": 0.80})

    inventory = ModelRegistry(str(root)).list_models()
    assert len(inventory) == 3
    by_sha = {m["sha256"]: m for m in inventory}
    superseding = by_sha[hashlib.sha256(ckpt_v1.read_bytes()).hexdigest()]
    assert superseding["name"] == "currant_bud_detector_v1_final"
    assert superseding["metrics"]["val_map50"] == 0.74
    assert superseding["checkpoint_path"] == str(ckpt_v1)
    assert by_sha[hashlib.sha256(ckpt_v2.read_bytes()).hexdigest()]["name"] == (
        "currant_bud_detector_v1")
