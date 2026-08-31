"""The relative-paths family's own proofs, beyond the battery of pre-existing suites re-run
unmodified: the version-2 document boundary's refusal and its partners, the grammar-aware
external test, the shared containment core between the checkpoint and dataset registries, and
every response surface answering a resolved absolute path for a relative stored entry.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import tcip_store as ts

from tcip_mcp.model_registry import (
    ModelRegistry,
    RegistryVersionRefused,
    read_registry_index,
    registry_index_key,
)
from tcip_mcp.registry_paths import is_external_form


def _seed_v1(root: Path, entries: list[dict]) -> None:
    ts.replace(registry_index_key(root), entries, expect=ts.Version.ABSENT)


# ── the document-boundary refusal and its partners ─────────────────────────────────────────


def test_absent_registry_answers_empty_for_a_fresh_project(tmp_path: Path):
    assert read_registry_index(tmp_path) == []


def test_a_bare_array_refuses_read_naming_the_conform_script(tmp_path: Path):
    _seed_v1(tmp_path, [{"name": "legacy", "checkpoint_path": "x.pt"}])

    with pytest.raises(RegistryVersionRefused, match="conform_model_registry_paths"):
        read_registry_index(tmp_path)


def test_a_bare_array_refuses_a_write_naming_the_conform_script(tmp_path: Path):
    _seed_v1(tmp_path, [{"name": "legacy", "checkpoint_path": "x.pt"}])
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"weights")

    with pytest.raises(RegistryVersionRefused, match="conform_model_registry_paths"):
        ModelRegistry(str(tmp_path)).register_model("m", str(ckpt), {}, metrics_source=None)


def test_the_refusal_is_not_a_store_error(tmp_path: Path):
    """Deliberate: a StoreError catch (bundle's own) must never swallow this into an empty
    answer, so it is checked as its own type, not merely as raising something."""
    _seed_v1(tmp_path, [{"name": "legacy", "checkpoint_path": "x.pt"}])

    with pytest.raises(RegistryVersionRefused):
        try:
            read_registry_index(tmp_path)
        except ts.StoreError:
            pytest.fail("RegistryVersionRefused must not be a StoreError")


def test_bundle_propagates_the_refusal_rather_than_reading_an_unconformed_registry_as_empty(
    tmp_path: Path,
):
    from tcip_mcp.tools.bundle import account_for

    from tcip_mcp.tools.project_tools import init_project

    init_project(str(tmp_path), site="north orchard")
    _seed_v1(tmp_path, [{"name": "legacy", "checkpoint_path": "x.pt"}])

    with pytest.raises(RegistryVersionRefused):
        account_for(tmp_path)


def test_archive_project_refuses_loudly_on_an_unconformed_registry(tmp_path: Path):
    from tcip_mcp.tools.project_tools import archive_project, init_project

    init_project(str(tmp_path), site="north orchard")
    _seed_v1(tmp_path, [{"name": "legacy", "checkpoint_path": "x.pt"}])

    result = archive_project(str(tmp_path), str(tmp_path.parent / "out.zip"))

    assert "error" in result
    assert "conform_model_registry_paths" in result["error"]


def test_doctor_reports_the_refusal_as_its_own_finding(tmp_path: Path):
    import importlib.util

    from tcip_mcp.tools.project_tools import init_project

    init_project(str(tmp_path), site="north orchard")
    _seed_v1(tmp_path, [{"name": "legacy", "checkpoint_path": "x.pt"}])

    doctor_path = Path(__file__).resolve().parent.parent / "scripts" / "doctor.py"
    spec = importlib.util.spec_from_file_location("tcip_doctor_under_test", doctor_path)
    assert spec and spec.loader
    doctor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(doctor)

    findings: list[tuple[str, str]] = []
    doctor.check_registry(tmp_path, findings)

    assert findings and findings[0][0] == "error"
    assert "could not be checked" in findings[0][1]


def test_a_conformed_registry_reads_clean(tmp_path: Path):
    from tcip_mcp.model_registry import conform_registry_paths

    _seed_v1(tmp_path, [])
    conform_registry_paths(tmp_path)

    assert read_registry_index(tmp_path) == []


def test_a_malformed_mapping_refuses_naming_what_it_found():
    from tcip_mcp.model_registry import _read_registry_document

    with pytest.raises(RegistryVersionRefused):
        _read_registry_document({"schema_version": 3, "entries": []})
    with pytest.raises(RegistryVersionRefused):
        _read_registry_document({"schema_version": 2, "entries": "not-a-list"})


# ── the grammar-aware external test, both spellings, both directions ───────────────────────


@pytest.mark.parametrize("spelling", [
    "C:/Users/breeder/model.pt",
    r"C:\Users\breeder\model.pt",
    "/home/breeder/model.pt",
    r"\\fileserver\share\model.pt",
    "//fileserver/share/model.pt",
])
def test_is_external_form_recognizes_every_absolute_spelling(spelling: str):
    assert is_external_form(spelling) is True


@pytest.mark.parametrize("spelling", [".", "models/m.pt", "a/b/c.pt"])
def test_is_external_form_rejects_every_relative_spelling(spelling: str):
    assert is_external_form(spelling) is False


# ── the shared containment core between the checkpoint and dataset registries ──────────────


def test_dataset_and_checkpoint_spellers_agree_on_the_same_geometry(tmp_path: Path):
    """One containment core: a checkpoint and a dataset both sitting under the same project
    root spell relative, and both sitting on a genuinely separate tree spell absolute, agreeing
    with each other rather than each registry re-deriving its own notion of containment."""
    from tcip_mcp.tools.project_tools import entry_is_external, registry_path_for

    project = tmp_path / "proj"
    nested_dataset = project / "datasets" / "main"
    nested_dataset.mkdir(parents=True)
    ckpt_dir = project / ".tcip" / "models"
    ckpt_dir.mkdir(parents=True)
    ckpt = ckpt_dir / "m.pt"
    ckpt.write_bytes(b"weights")

    reg = ModelRegistry(str(project))
    entry = reg.register_model("m", str(ckpt), {}, metrics_source=None)
    stored_ckpt = read_registry_index(project)[0]["checkpoint_path"]
    dataset_path = registry_path_for(nested_dataset, project)

    assert not is_external_form(stored_ckpt)
    assert not entry_is_external({"path": dataset_path})
    assert Path(entry["checkpoint_path"]).is_absolute()

    outside = tmp_path / "elsewhere"
    outside.mkdir()
    outside_ckpt = outside / "m2.pt"
    outside_ckpt.write_bytes(b"other weights")
    reg.register_model("m2", str(outside_ckpt), {}, metrics_source=None)
    stored_outside = read_registry_index(project)[1]["checkpoint_path"]
    outside_dataset_path = registry_path_for(outside, project)

    assert is_external_form(stored_outside)
    assert entry_is_external({"path": outside_dataset_path})


# ── response surfaces: resolved absolute, including a relative-root process case ───────────


def _register_internal_checkpoint(project: Path) -> tuple[ModelRegistry, str]:
    ckpt_dir = project / ".tcip" / "experiments" / "exp1"
    ckpt_dir.mkdir(parents=True)
    ckpt = ckpt_dir / "model_final.pt"
    ckpt.write_bytes(b"a resolved-response fixture's own weights")
    reg = ModelRegistry(str(project))
    reg.register_model("m", str(ckpt), {}, metrics_source=None)
    return reg, str(ckpt)


def test_model_registry_list_get_best_all_answer_resolved_absolute(tmp_path: Path):
    project = tmp_path / "proj"
    _, ckpt = _register_internal_checkpoint(project)
    reg = ModelRegistry(str(project))

    listed = reg.list_models()[0]["checkpoint_path"]
    got = reg.get_model("m")["checkpoint_path"]

    assert Path(listed) == Path(ckpt).resolve()
    assert Path(got) == Path(ckpt).resolve()


def test_list_registered_models_tool_answers_resolved_absolute(tmp_path: Path):
    from tcip_mcp.tools.model_tools import list_registered_models

    project = tmp_path / "proj"
    _, ckpt = _register_internal_checkpoint(project)

    result = list_registered_models(str(project))

    assert Path(result["models"][0]["checkpoint_path"]) == Path(ckpt).resolve()


def test_select_best_model_tool_answers_resolved_absolute(tmp_path: Path):
    from tcip_mcp.tools.model_tools import select_best_model

    project = tmp_path / "proj"
    ckpt_dir = project / ".tcip" / "experiments" / "exp1"
    ckpt_dir.mkdir(parents=True)
    ckpt = ckpt_dir / "model_final.pt"
    ckpt.write_bytes(b"best-model fixture weights")
    ModelRegistry(str(project)).register_model(
        "m", str(ckpt), {}, metrics={"val_map50": 0.9}, metrics_source="trainer",
    )

    result = select_best_model(str(project), metric="val_map50", higher_is_better=True)

    assert "error" not in result, result
    assert Path(result["checkpoint_path"]) == ckpt.resolve()


def test_explicit_register_model_return_is_resolved_absolute(tmp_path: Path):
    project = tmp_path / "proj"
    ckpt_dir = project / ".tcip" / "models"
    ckpt_dir.mkdir(parents=True)
    ckpt = ckpt_dir / "m.pt"
    ckpt.write_bytes(b"explicit-mode weights")

    entry = ModelRegistry(str(project)).register_model("m", str(ckpt), {}, metrics_source=None)

    assert Path(entry["checkpoint_path"]) == ckpt.resolve()


def test_register_model_from_experiment_checkpoint_field_is_resolved_absolute(
    tmp_path: Path, monkeypatch,
):
    from tcip_mcp.experiments import (
        complete_run, create_experiment, experiment_dir, register_model_from_experiment,
        update_status,
    )
    from tcip_mcp.tools.project_tools import init_project

    project = tmp_path / "proj"
    init_project(str(project), site="north orchard")
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(project))
    create_experiment("exp1", {"model_source": {"builder": "x:y"}})
    update_status("exp1", "running")
    ckpt_dir = experiment_dir("exp1")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    weights = ckpt_dir / "model_final.pt"
    weights.write_bytes(b"experiment-mode weights")
    assert "error" not in complete_run("exp1", str(weights))

    registered = register_model_from_experiment("exp1", str(weights), project_path=str(project))

    assert "error" not in registered, registered
    assert Path(registered["checkpoint"]) == weights.resolve()


# ── one shared at-or-under predicate, not a second copy per module ─────────────────────────


def test_model_registry_and_bundle_share_the_same_at_or_under_function():
    import tcip_mcp.model_registry as model_registry
    import tcip_mcp.tools.bundle as bundle
    from tcip_mcp.registry_paths import is_at_or_under

    assert model_registry.is_at_or_under is is_at_or_under
    assert bundle._is_at_or_under is is_at_or_under


# ── resolved_registry_path's traversal refusal, both grammars ──────────────────────────────


@pytest.mark.parametrize("stored", ["../outside/evil.pt", r"..\..\outside\evil.pt"])
def test_resolved_registry_path_refuses_a_traversal_in_either_grammar(tmp_path: Path, stored: str):
    from tcip_mcp.registry_paths import RegistryPathTraversal, resolved_registry_path

    with pytest.raises(RegistryPathTraversal):
        resolved_registry_path(tmp_path, stored)


def test_resolved_registry_path_admits_a_legitimate_relative_value(tmp_path: Path):
    from tcip_mcp.registry_paths import resolved_registry_path

    result = resolved_registry_path(tmp_path, ".tcip/models/m.pt")

    assert result == (tmp_path / ".tcip" / "models" / "m.pt").resolve()


def test_a_relative_root_still_answers_an_absolute_response(tmp_path: Path, monkeypatch):
    """The relative-root case section 6 names: the resolver's own root argument, not just the
    entry's stored path, must still answer absolute. The registry key itself refuses a relative
    project root (``require_absolute_root``), so this is the resolver's own contract, exercised
    directly rather than through the full ``ModelRegistry`` stack."""
    from tcip_mcp.registry_paths import resolved_registry_path

    monkeypatch.chdir(tmp_path)
    (tmp_path / "subdir").mkdir()

    result = resolved_registry_path("subdir", "m.pt")

    assert result.is_absolute()
    assert result == (tmp_path / "subdir" / "m.pt").resolve()
