"""The registry entry shape as its readers actually consume it.

``ModelRegistry`` writes ``.tcip/models/registry.json``; other readers consume those entries by
key: the data-state doctor's registry-pollution check and the provenance identity resolver. Both
sides of each agreement here run the real implementations against a registry the real
``register_model`` wrote, so a key the writer stops emitting shows up as a reader going quiet
rather than as a check still passing against a hand-written entry.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

from tcip_mcp.model_registry import ModelRegistry, resolve_model_identity

DOCTOR_PATH = Path(__file__).parent.parent / "scripts" / "doctor.py"

# Checkpoints of distinct size and content, so an entry can never be matched by accident.
_RUNS = {
    "hazelnut_catkin_detector_v1": b"weights-a",
    "chestnut_leaf_area_seg_v2": b"weights-b-longer-payload",
}


def _doctor():
    """The data-state doctor loaded as a module, for its own read of the registry index."""
    spec = importlib.util.spec_from_file_location("tcip_data_state_doctor", DOCTOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def polluted_project(tmp_path: Path) -> tuple[Path, ModelRegistry]:
    """A project whose registry entries point at checkpoints under a test-fixture directory."""
    root = tmp_path / "proj"
    root.mkdir()
    leak_dir = tmp_path / "pytest-of-fixture" / "run"
    leak_dir.mkdir(parents=True)
    reg = ModelRegistry(str(root))
    for i, (name, content) in enumerate(_RUNS.items()):
        ckpt = leak_dir / f"{name}.pt"
        ckpt.write_bytes(content)
        reg.register_model(
            name, str(ckpt), {"data": {"subject": "catkin"}},
            metrics={"val_map50": 0.5 + 0.1 * i}, tags=["detector", f"experiment:run{i}"],
        )
    return root, reg


def test_doctor_flags_every_registry_entry_the_registry_wrote(polluted_project) -> None:
    """The doctor's registry check reads the index by key; every entry the registry holds must be
    visible to it, named and with its checkpoint path, or the mandated data-state check degrades
    into a silent pass on a polluted project."""
    root, reg = polluted_project
    registered = reg.list_models()
    assert len(registered) == len(_RUNS)

    findings: list[tuple[str, str]] = []
    _doctor().check_registry(root, findings)

    errors = [msg for level, msg in findings if level == "error"]
    assert len(errors) == len(_RUNS)
    for entry in registered:
        assert any(
            entry["name"] in msg and entry["checkpoint_path"] in msg for msg in errors
        ), f"no doctor finding names the registered model {entry['name']}"


def test_doctor_reports_nothing_for_a_project_with_no_registered_models(tmp_path: Path) -> None:
    """A models directory with an empty index is a clean state, not a finding: constructing the
    registry creates the directory before anything is registered."""
    root = tmp_path / "proj"
    root.mkdir()
    reg = ModelRegistry(str(root))
    assert (root / ".tcip" / "models").is_dir()
    assert reg.list_models() == []

    findings: list[tuple[str, str]] = []
    _doctor().check_registry(root, findings)
    assert findings == []


def test_identity_resolution_matches_a_registered_checkpoint_by_content(tmp_path: Path) -> None:
    """A checkpoint copied to a path the registry never saw still resolves to the run that
    produced it, matched on the content hash the registry stored. Resolution scans the whole
    index, so the entry whose hash matches wins over an earlier entry that does not."""
    root = tmp_path / "proj"
    root.mkdir()
    reg = ModelRegistry(str(root))

    other = tmp_path / "other_run.pt"
    other.write_bytes(b"a different run entirely")
    reg.register_model("chestnut_leaf_area_seg_v2", str(other), {},
                       tags=["segmenter", "experiment:leaf_run"])

    content = b"the checkpoint that produced the phenotype"
    trained = tmp_path / "model_best.pt"
    trained.write_bytes(content)
    reg.register_model("hazelnut_catkin_detector_v1", str(trained), {},
                       tags=["detector", "experiment:catkin_run3"])

    delivered = tmp_path / "delivery" / "model_copy.pt"
    delivered.parent.mkdir()
    delivered.write_bytes(content)

    identity = resolve_model_identity(delivered, project_path=str(root))
    assert identity["sha256"] == hashlib.sha256(content).hexdigest()
    assert identity["experiment_id"] == "catkin_run3"
    assert identity["checkpoint"] == "model_copy"
