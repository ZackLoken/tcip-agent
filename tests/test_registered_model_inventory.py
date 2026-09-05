"""What the registry holds, and where it holds it.

The registered-model inventory of a project is the index at ``.tcip/models/registry.json``:
registration records a checkpoint where it already lives instead of copying it under the registry
directory, and a name registered again supersedes its earlier entry rather than adding a second
one. Both facts decide what any reader counting a project's models can honestly report.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import tcip_store as ts
from tcip_mcp.model_registry import ModelRegistry, registry_index_key

# Distinct payload sizes per run, so a size or hash attributed to the wrong entry is visible.
_RUNS = {
    "currant_bud_detector_v1": b"run-a-weights",
    "chestnut_leaf_area_seg_v2": b"run-b-weights-with-a-longer-payload",
    "currant_bush_detector_v1": b"run-c",
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
        ckpt = run_dir / "model_best.pt"
        ckpt.write_bytes(content)
        reg.register_model(name, str(ckpt), {"data": {"subject": "bud"}},
                           metrics={"val_map50": 0.42}, metrics_source="caller")

    models_dir = root / ".tcip" / "models"
    assert ts.exists(registry_index_key(root))
    assert list(models_dir.glob("*.pt")) == []

    reread = ModelRegistry(str(root)).list_models()
    assert {m["name"] for m in reread} == set(_RUNS)
    for entry in reread:
        ckpt = Path(entry["checkpoint_path"])
        assert ckpt.is_file()
        assert ckpt.parent.parent.parent.name == "experiments"
        assert entry["file_size_bytes"] == len(_RUNS[entry["name"]])
        assert entry["sha256"] == hashlib.sha256(_RUNS[entry["name"]]).hexdigest()


def test_registering_a_name_again_supersedes_its_earlier_entry(tmp_path: Path) -> None:
    """A resumed or re-registered run under an existing name replaces that entry. The inventory
    keeps one entry per name and it carries the newer checkpoint, so a reader counting registered
    models counts runs and never counts one run twice."""
    root = tmp_path / "proj"
    reg = ModelRegistry(str(root))

    first, second = b"epoch-8-weights", b"epoch-19-weights-after-resume"
    ckpt_v1 = tmp_path / "model_epoch8.pt"
    ckpt_v1.write_bytes(first)
    ckpt_v2 = tmp_path / "model_epoch19.pt"
    ckpt_v2.write_bytes(second)
    companion = tmp_path / "leaf_model.pt"
    companion.write_bytes(b"a separate run")

    reg.register_model("currant_bud_detector_v1", str(ckpt_v1), {},
                       metrics={"val_map50": 0.61}, metrics_source="caller")
    reg.register_model("chestnut_leaf_area_seg_v2", str(companion), {},
                       metrics={"val_map50": 0.55}, metrics_source="caller")
    reg.register_model("currant_bud_detector_v1", str(ckpt_v2), {},
                       metrics={"val_map50": 0.74}, metrics_source="caller")

    inventory = ModelRegistry(str(root)).list_models()
    assert len(inventory) == 2
    assert [m["name"] for m in inventory].count("currant_bud_detector_v1") == 1

    superseding = ModelRegistry(str(root)).get_model("currant_bud_detector_v1")
    assert superseding["sha256"] == hashlib.sha256(second).hexdigest()
    assert superseding["file_size_bytes"] == len(second)
    assert superseding["metrics"]["val_map50"] == 0.74
    assert superseding["checkpoint_path"] == str(ckpt_v2)
