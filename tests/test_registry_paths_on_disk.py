"""The model registry's paths across an archive and import: an in-project checkpoint's entry is
stored relative to its project root, so the restored project's registry names the restored file.
"""

from __future__ import annotations

from pathlib import Path


def test_archive_then_import_lands_a_registry_naming_the_restored_files(
    tmp_path: Path, monkeypatch,
):
    """The real archive and import round trip: a foreign entry stays relative and the restored
    run's own entry names the checkpoint under the destination."""
    from tcip_mcp.model_registry import ModelRegistry, read_registry_index
    from tcip_mcp.tools.model_tools import register_model
    from tcip_mcp.tools.project_tools import archive_project, import_project, initialize_project
    from tests._verified_checkpoint_fixtures import checkpoint_file, finished_run

    src = tmp_path / "src_project"
    initialize_project(str(src), site="north orchard")
    monkeypatch.setenv("TCIP_STATE_ROOT", str(src))
    finished_run(None, experiment_id="exp1")
    foreign = src / ".tcip" / "models" / "foreign.pt"
    foreign.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_file(foreign, "a foreign checkpoint's own weights")
    register_model(name="foreign", checkpoint_path=str(foreign), config={},
                   project_path=str(src))

    zip_path = tmp_path / "export.zip"
    assert "error" not in archive_project(str(src), str(zip_path), include_models=True)
    dest = tmp_path / "restored"
    imported = import_project(str(zip_path), str(dest))
    assert "error" not in imported, imported

    entries = read_registry_index(dest)
    assert [e["name"] for e in entries] == ["foreign"]
    assert not Path(entries[0]["checkpoint_path"]).is_absolute()
    listed = {m["name"]: m for m in ModelRegistry(str(dest)).list_models()}
    assert "foreign" in listed
    run_entry = listed["exp1"]
    assert Path(run_entry["checkpoint_path"]).is_file()
    assert Path(run_entry["checkpoint_path"]).is_relative_to(dest)
