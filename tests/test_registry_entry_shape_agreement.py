"""The registry entry shape as its readers actually consume it.

``ModelRegistry`` writes ``.tcip/models/registry.json``; other readers consume those entries by
key: the data-state doctor's registry-pollution check, the provenance identity resolver, and the
browser's own ``RegisteredModel``. Both sides of each agreement here run the real implementations
against a registry the real ``register_model`` wrote, so a key the writer stops emitting shows up
as a reader going quiet rather than as a check still passing against a hand-written entry.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from tcip_mcp.model_registry import (
    ModelRegistry, load_registered_checkpoint, resolve_model_identity,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_SRC = REPO_ROOT / "packages" / "tcip-web" / "frontend" / "src"
ENTRY_DECLARATION = FRONTEND_SRC / "api" / "inference.ts"

_ENTRY_INTERFACE_RE = re.compile(r"export interface RegisteredModel \{(.*?)\n\}", re.S)
_TS_BLOCK_RE = re.compile(r"(?:interface|type)\s+\w+\s*=?\s*\{(.*?)\n\}", re.S)
_TS_FIELD_RE = re.compile(r"^\s+(\w+)(\??):", re.M)

# Checkpoints of distinct size and content, so an entry can never be matched by accident.
_RUNS = {
    "currant_bud_detector_v1": b"weights-a",
    "chestnut_leaf_area_seg_v2": b"weights-b-longer-payload",
}


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
        ckpt = leak_dir / f"{name}.pt"
        ckpt.write_bytes(content)
        reg.register_model(
            name, str(ckpt), {"data": {"subject": "bud"}},
            metrics={"val_map50": 0.5 + 0.1 * i}, tags=["detector", f"experiment:run{i}"],
            metrics_source="caller",
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


def test_identity_resolution_matches_a_registered_checkpoint_by_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A checkpoint copied to a path the registry never saw still resolves to the run that
    produced it, matched on the content hash the registry stored: the binding a run's own
    completion recorded, not a caller-asserted tag."""
    torch = pytest.importorskip("torch")
    from tcip_mcp.experiments import complete_run, create_experiment, register_model_from_experiment

    root = tmp_path / "proj"
    root.mkdir()
    monkeypatch.setenv("TCIP_STATE_ROOT", str(root))

    other = tmp_path / "other_run.pt"
    torch.save({"model_state_dict": {}, "note": "a different run entirely"}, other)
    create_experiment("leaf_run", {"model_source": {"builder": "x:y"}})
    assert "error" not in complete_run("leaf_run", str(other))
    assert "error" not in register_model_from_experiment("leaf_run", str(other))

    trained = tmp_path / "model_best.pt"
    torch.save({"model_state_dict": {}, "note": "the checkpoint that produced the phenotype"},
              trained)
    create_experiment("bud_run3", {"model_source": {"builder": "x:y"}})
    assert "error" not in complete_run("bud_run3", str(trained))
    assert "error" not in register_model_from_experiment("bud_run3", str(trained))

    delivered = tmp_path / "delivery" / "model_copy.pt"
    delivered.parent.mkdir()
    delivered.write_bytes(trained.read_bytes())

    checkpoint = load_registered_checkpoint(delivered, project_path=str(root))
    identity = resolve_model_identity(checkpoint)
    assert identity["sha256"] == hashlib.sha256(trained.read_bytes()).hexdigest()
    assert identity["experiment_id"] == "bud_run3"
    assert identity["checkpoint"] == "model_copy"


def test_doctor_reports_an_index_that_will_not_decode_rather_than_reading_it_as_no_models(
    tmp_path: Path,
) -> None:
    """A registry the reader cannot decode is a finding, not silence.

    Absence and corruption are different states: a project that registered nothing has no models,
    while a project whose index will not parse has models nobody can see. Folding the second into
    the first hands a breeder a clean bill of health for a registry that is unreadable.

    Bound to the file backend on purpose: corruption is injected by truncating the index's own
    raw bytes on disk, which only means something for a document a file actually holds.
    """
    from tcip_store import bind
    from tcip_store.file_backend import FileBackend

    bind(FileBackend())
    root = tmp_path / "proj"
    root.mkdir()
    reg = ModelRegistry(str(root))
    ckpt = tmp_path / "model_best.pt"
    ckpt.write_bytes(b"weights-a")
    reg.register_model("currant_bud_detector_v1", str(ckpt), {}, metrics_source=None)

    index = root / ".tcip" / "models" / "registry.json"
    index.write_text(index.read_text(encoding="utf-8")[:-8], encoding="utf-8")

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


def _declared_entry_fields() -> dict[str, bool]:
    """The browser's registry-entry fields, each mapped to whether it is declared optional."""
    match = _ENTRY_INTERFACE_RE.search(ENTRY_DECLARATION.read_text(encoding="utf-8"))
    assert match is not None, "the RegisteredModel interface is no longer where this test reads it"
    fields = {name: bool(optional) for name, optional in _TS_FIELD_RE.findall(match.group(1))}
    assert fields, "no fields parsed out of the RegisteredModel interface"
    return fields


def test_every_field_the_browser_reads_off_an_entry_is_one_the_registry_writes(
    tmp_path: Path,
) -> None:
    """The browser's entry type is held against an entry the real writer produced.

    Transcribed by hand, it drifts the moment a key is renamed: the UI keeps compiling and the
    field it reads arrives undefined, which for the checkpoint path means an inference launch
    against nothing. A field declared required must also carry a value, since a key the writer
    always sets to null is not a string the browser can render.
    """
    root = tmp_path / "proj"
    root.mkdir()
    ckpt = tmp_path / "model_best.pt"
    ckpt.write_bytes(b"weights-a")
    entry = ModelRegistry(str(root)).register_model(
        "currant_bud_detector_v1", str(ckpt), {"data": {"subject": "bud"}},
        metrics={"val_map50": 0.5}, tags=["detector"], metrics_source="caller",
    )

    declared = _declared_entry_fields()
    missing = sorted(name for name in declared if name not in entry)
    assert not missing, (
        f"the browser reads fields the registry entry does not carry: {missing}; "
        f"the entry's own keys are {sorted(entry)}"
    )
    unset = sorted(name for name, optional in declared.items() if not optional and entry[name] is None)
    assert not unset, f"the browser declares these required but the registry leaves them unset: {unset}"


def test_no_other_frontend_module_declares_a_registry_entry_of_its_own() -> None:
    """One declaration in the browser, so the backend has one counterpart to stay equal to.

    A second type naming the same entry is what drifts: the module that was updated keeps
    working and the one that was not reads a key the writer no longer emits. Frontend test files
    are left out, since a shape written there is the expectation being asserted.
    """
    entry_fields = {"name", "checkpoint_path"}
    sources = [
        p for p in sorted(list(FRONTEND_SRC.rglob("*.ts")) + list(FRONTEND_SRC.rglob("*.tsx")))
        if ".test." not in p.name and "test" not in p.parent.name and p != ENTRY_DECLARATION
    ]
    assert sources, "no frontend sources found, so nothing was checked"
    offenders = [
        p.relative_to(REPO_ROOT).as_posix()
        for p in sources
        for block in _TS_BLOCK_RE.findall(p.read_text(encoding="utf-8"))
        if entry_fields <= {name for name, _ in _TS_FIELD_RE.findall(block)}
    ]
    assert not offenders, (
        "these declare a registry entry instead of importing RegisteredModel:\n"
        + "\n".join(offenders)
    )
