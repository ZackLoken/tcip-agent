"""The registry entry shape as its readers actually consume it.

Each agreement here runs a reader of the registry index's entries against a registry the real
``register_model`` wrote.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tcip_mcp.model_registry import ModelRegistry, load_registered_checkpoint

# Checkpoints of distinct content, so an entry can never be matched by accident.
_RUNS = {
    "currant_bud_detector_v1": "weights-a",
    "chestnut_leaf_area_seg_v2": "weights-b-longer-payload",
}


def _checkpoint(path: Path, content: str) -> Path:
    """A checkpoint at ``path`` a run completed under a root of its own, its bytes one per
    ``content`` (``_verified_checkpoint_fixtures.produced_checkpoint``)."""
    pytest.importorskip("torch")
    from tests._verified_checkpoint_fixtures import produced_checkpoint

    return produced_checkpoint(path, content)


def _doctor():
    """The data-state doctor loaded as a module, for its own read of the registry index."""
    from tcip_mcp.cli import doctor

    return doctor


@pytest.fixture()
def polluted_project(tmp_path: Path) -> tuple[Path, ModelRegistry]:
    """A project whose registry entries point at checkpoints under a test-fixture directory."""
    root = tmp_path / "proj"
    root.mkdir()
    leak_dir = tmp_path / "pytest-of-fixture" / "run"
    leak_dir.mkdir(parents=True)
    reg = ModelRegistry(str(root))
    for i, (name, content) in enumerate(_RUNS.items()):
        ckpt = _checkpoint(leak_dir / f"{name}.pt", content)
        reg.register_model(
            name, str(ckpt),
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
    """A project that has registered nothing is a clean state, not a finding."""
    root = tmp_path / "proj"
    root.mkdir()
    reg = ModelRegistry(str(root))
    assert reg.list_models() == []

    findings: list[tuple[str, str]] = []
    _doctor().check_registry(root, findings)
    assert findings == []


def test_identity_resolution_matches_a_registered_checkpoint_by_content(
    tmp_path: Path,
) -> None:
    """A checkpoint copied to a path the registry never saw still resolves to the run that
    produced it, matched on the content hash its final status names: the binding a run's own
    completion recorded, not a caller-asserted tag."""
    pytest.importorskip("torch")
    from tests._verified_checkpoint_fixtures import finished_run, registered_checkpoint

    root = tmp_path / "proj"
    root.mkdir()

    finished_run(root, experiment_id="leaf_run")
    trained = Path(registered_checkpoint(root, experiment_id="bud_run3"))

    delivered = tmp_path / "delivery" / "model_copy.pt"
    delivered.parent.mkdir()
    delivered.write_bytes(trained.read_bytes())

    checkpoint = load_registered_checkpoint(delivered, project=root)
    assert checkpoint.producer == {
        "checkpoint_sha256": hashlib.sha256(trained.read_bytes()).hexdigest(),
        "experiment_id": "bud_run3"}


def test_doctor_reports_an_index_that_will_not_decode_rather_than_reading_it_as_no_models(
    tmp_path: Path,
) -> None:
    """A registry the reader cannot decode is a finding, not silence.

    Absence and corruption are different states: a project that registered nothing has no models,
    while a project whose index will not parse has models nobody can see. Folding the second into
    the first hands a breeder a clean bill of health for a registry that is unreadable.
    """
    from tcip_mcp.model_registry import registry_index_key
    from tests._record_damage_fixtures import damage_record

    root = tmp_path / "proj"
    root.mkdir()
    reg = ModelRegistry(str(root))
    ckpt = _checkpoint(tmp_path / "model_best.pt", "weights-a")
    reg.register_model("currant_bud_detector_v1", str(ckpt))

    damage_record(registry_index_key(root), b'{"entries": [{"name": "currant')

    findings: list[tuple[str, str]] = []
    _doctor().check_registry(root, findings)
    assert [level for level, _ in findings] == ["error"]
    assert "will not decode" in findings[0][1]


def test_a_readable_index_is_still_read_entry_by_entry(polluted_project) -> None:
    """The decode refusal is scoped to an index that will not parse.

    A readable index is still walked entry by entry, so raising on corruption cannot collapse
    every project into the one finding that says nothing about which models it holds.
    """
    root, reg = polluted_project
    assert len(reg.list_models()) == len(_RUNS)

    findings: list[tuple[str, str]] = []
    _doctor().check_registry(root, findings)
    assert len(findings) == len(_RUNS)
    assert not any("decode" in message for _, message in findings)
    for entry in reg.list_models():
        assert any(entry["name"] in message for _, message in findings)
