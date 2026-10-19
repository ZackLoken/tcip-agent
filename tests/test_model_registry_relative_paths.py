"""A model registry entry's path is stored relative to the project root and resolved on read.

These tests cover the external-form check, the checkpoint and dataset registries' agreement on
relative and absolute spellings, and the response surfaces answering a resolved absolute path
for a relative stored entry.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

import tcip_store as ts
from tcip_mcp.model_registry import (
    ModelRegistry,
    RegistryIndex,
    read_registry_index,
    registry_index_key,
)
from tcip_mcp.registry_paths import is_external_form


def _checkpoint(path: Path, content: str) -> None:
    """Write a checkpoint at ``path`` through ``produced_checkpoint(path, content)``."""
    pytest.importorskip("torch")
    from tests._verified_checkpoint_fixtures import produced_checkpoint

    produced_checkpoint(path, content)


def test_absent_registry_answers_empty_for_a_fresh_project(
    tmp_path: Path,
) -> None:
    assert read_registry_index(tmp_path) == []


def test_archive_project_carries_a_registered_checkpoint_outside_models(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.project_tools import ZipBundle, archive_project
    from tests._web_fixtures import new_project

    project = tmp_path / "proj"
    new_project(project)
    weights_dir = project / "weights"
    weights_dir.mkdir()
    ckpt = weights_dir / "m.pt"
    _checkpoint(ckpt, "weights")
    ModelRegistry(str(project)).register_model("m", str(ckpt))
    out = tmp_path / "out.zip"

    result = archive_project(
        project, ZipBundle(output_path=str(out)), include_models=True
    )

    assert "error" not in result, result
    with zipfile.ZipFile(out) as zf:
        assert "weights/m.pt" in zf.namelist()

    without = archive_project(
        project, ZipBundle(output_path=str(tmp_path / "without.zip"))
    )
    assert "error" not in without and without["checkpoints_excluded"] == 1
    with zipfile.ZipFile(tmp_path / "without.zip") as zf:
        assert "weights/m.pt" not in zf.namelist()


# ── the grammar-aware external test, both spellings, both directions ───────────────────────


@pytest.mark.parametrize(
    "spelling",
    [
        "C:/Users/breeder/model.pt",
        r"C:\Users\breeder\model.pt",
        "/home/breeder/model.pt",
        r"\\fileserver\share\model.pt",
        "//fileserver/share/model.pt",
    ],
)
def test_is_external_form_recognizes_every_absolute_spelling(
    spelling: str,
) -> None:
    assert is_external_form(spelling) is True


@pytest.mark.parametrize("spelling", [".", "models/m.pt", "a/b/c.pt"])
def test_is_external_form_rejects_every_relative_spelling(
    spelling: str,
) -> None:
    assert is_external_form(spelling) is False


# ── the shared containment core between the checkpoint and dataset registries ──────────────


def test_dataset_and_checkpoint_spellers_agree_on_the_same_geometry(
    tmp_path: Path,
) -> None:
    """The dataset and checkpoint registries spell the same geometry the same way.

    A checkpoint and a dataset both under the project root are recorded relative by their own
    registries, and both on a separate tree are recorded absolute.
    """
    from tcip_mcp.tools.project_tools import read_datasets, register_dataset

    project = tmp_path / "proj"
    nested_dataset = project / "datasets" / "main"
    nested_dataset.mkdir(parents=True)
    _checkpoint(project / ".tcip" / "models" / "m.pt", "weights")

    reg = ModelRegistry(str(project))
    entry = reg.register_model(
        "m", str(project / ".tcip" / "models" / "m.pt")
    ).entry
    assert "error" not in register_dataset(
        project, str(nested_dataset), "chestnut"
    )
    stored_ckpt = read_registry_index(project)[0].checkpoint_path
    (dataset_entry,) = read_datasets(project)

    assert not is_external_form(stored_ckpt)
    assert not is_external_form(dataset_entry.path)
    assert Path(entry.checkpoint_path).is_absolute()

    outside = tmp_path / "elsewhere"
    _checkpoint(outside / "m2.pt", "other weights")
    reg.register_model("m2", str(outside / "m2.pt"))
    assert "error" not in register_dataset(project, str(outside), "chestnut")
    stored_outside = read_registry_index(project)[1].checkpoint_path
    outside_entry = next(
        e for e in read_datasets(project) if e.id != dataset_entry.id
    )

    assert is_external_form(stored_outside)
    assert is_external_form(outside_entry.path)


# ── response surfaces: resolved absolute, including a relative-root process case ───────────


def _register_internal_checkpoint(project: Path) -> tuple[ModelRegistry, str]:
    ckpt = project / ".tcip" / "models" / "m.pt"
    _checkpoint(ckpt, "weights")
    reg = ModelRegistry(str(project))
    reg.register_model("m", str(ckpt))
    return reg, str(ckpt)


def test_model_registry_listing_answers_resolved_absolute(
    tmp_path: Path,
) -> None:
    project = tmp_path / "proj"
    _, ckpt = _register_internal_checkpoint(project)

    listed = ModelRegistry(str(project)).list_models()[0].entry.checkpoint_path

    assert Path(listed) == Path(ckpt).resolve()


def test_rank_registered_models_listing_view_answers_resolved_absolute(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.model_tools import rank_registered_models

    project = tmp_path / "proj"
    _, ckpt = _register_internal_checkpoint(project)

    result = rank_registered_models(project)

    assert Path(result["models"][0]["checkpoint_path"]) == Path(ckpt).resolve()


def test_rank_registered_models_tool_answers_resolved_absolute(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.model_tools import rank_registered_models

    project = tmp_path / "proj"
    ckpt = project / ".tcip" / "models" / "m.pt"
    _checkpoint(ckpt, "weights")
    ModelRegistry(str(project)).register_model(
        "m", str(ckpt), metrics={"val_map50": 0.9}
    )

    result = rank_registered_models(
        project,
        metric="val_map50",
        higher_is_better=True,
        include_unverified=True,
    )

    assert "error" not in result, result
    assert Path(result["checkpoint_path"]) == ckpt.resolve()


def test_explicit_register_model_return_is_resolved_absolute(
    tmp_path: Path,
) -> None:
    project = tmp_path / "proj"
    ckpt = project / ".tcip" / "models" / "m.pt"
    _checkpoint(ckpt, "weights")

    entry = ModelRegistry(str(project)).register_model("m", str(ckpt)).entry

    assert Path(entry.checkpoint_path) == ckpt.resolve()


def test_a_completed_runs_registry_entry_answers_its_checkpoint_resolved_absolute(
    tmp_path: Path,
) -> None:
    """A completed run's registry entry answers its checkpoint resolved absolute.

    A run's final status stores its checkpoint relative to the project; the registry's
    listing of that run answers the path resolved absolute.
    """
    from tcip_mcp.experiments import experiment_dir
    from tests._verified_checkpoint_fixtures import finished_run
    from tests._web_fixtures import new_project

    project = tmp_path / "proj"
    new_project(project)
    finished_run(project, experiment_id="exp1")

    (entry,) = [
        m.entry
        for m in ModelRegistry(str(project)).list_models()
        if m.entry.name == "exp1"
    ]

    assert Path(entry.checkpoint_path).is_absolute()
    assert (
        Path(entry.checkpoint_path).resolve()
        == (
            experiment_dir("exp1", project=project) / "model_final.pt"
        ).resolve()
    )


# ── resolved_registry_path's traversal refusal, both grammars ──────────────────────────────


@pytest.mark.parametrize(
    "stored", ["../outside/evil.pt", r"..\..\outside\evil.pt"]
)
def test_resolved_registry_path_refuses_a_traversal_in_either_grammar(
    tmp_path: Path, stored: str
) -> None:
    from tcip_mcp.registry_paths import (
        RegistryPathTraversalError,
        resolved_registry_path,
    )

    with pytest.raises(RegistryPathTraversalError):
        resolved_registry_path(tmp_path, stored)


def test_resolved_registry_path_admits_a_legitimate_relative_value(
    tmp_path: Path,
) -> None:
    from tcip_mcp.registry_paths import resolved_registry_path, stored_path

    target = (tmp_path / ".tcip" / "models" / "m.pt").resolve()
    result = resolved_registry_path(tmp_path, stored_path(target, tmp_path))

    assert result == target


# ── resolved_registry_path's empty-value refusal, and its admits-valid-work partner ────────


def test_resolved_registry_path_refuses_an_empty_value(tmp_path: Path) -> None:
    from tcip_mcp.registry_paths import (
        RegistryPathEmptyError,
        resolved_registry_path,
    )

    with pytest.raises(RegistryPathEmptyError):
        resolved_registry_path(tmp_path, "")


def test_resolved_registry_path_admits_a_non_empty_value(
    tmp_path: Path,
) -> None:
    from tcip_mcp.registry_paths import resolved_registry_path, stored_path

    target = (tmp_path / "m.pt").resolve()
    assert (
        resolved_registry_path(tmp_path, stored_path(target, tmp_path))
        == target
    )


# ── checkpoint_registry_path_for's root gate, and its admits-valid-work partner ────────────


def test_checkpoint_registry_path_for_refuses_a_missing_root(
    tmp_path: Path,
) -> None:
    from tcip_mcp.registry_paths import (
        CheckpointRegistryRootUnusableError,
        checkpoint_registry_path_for,
    )

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"weights")
    missing_root = tmp_path / "does_not_exist"

    with pytest.raises(CheckpointRegistryRootUnusableError):
        checkpoint_registry_path_for(ckpt, missing_root)


def test_checkpoint_registry_path_for_refuses_a_file_shaped_root(
    tmp_path: Path,
) -> None:
    from tcip_mcp.registry_paths import (
        CheckpointRegistryRootUnusableError,
        checkpoint_registry_path_for,
    )

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"weights")
    file_root = tmp_path / "not_a_directory"
    file_root.write_bytes(b"not a directory")

    with pytest.raises(CheckpointRegistryRootUnusableError):
        checkpoint_registry_path_for(ckpt, file_root)


def test_checkpoint_registry_path_for_admits_an_existing_directory_root(
    tmp_path: Path,
) -> None:
    from tcip_mcp.registry_paths import checkpoint_registry_path_for

    ckpt_dir = tmp_path / ".tcip" / "models"
    ckpt_dir.mkdir(parents=True)
    ckpt = ckpt_dir / "m.pt"
    ckpt.write_bytes(b"weights")

    assert checkpoint_registry_path_for(ckpt, tmp_path) == ".tcip/models/m.pt"


def test_list_models_refuses_a_stored_entry_whose_path_climbs_out_of_the_project(
    tmp_path: Path,
) -> None:
    """A stored entry whose checkpoint path carries ``..`` refuses the listing that reads it."""
    from tcip_mcp.registry_paths import RegistryPathTraversalError

    project = tmp_path / "proj"
    reg, _ckpt = _register_internal_checkpoint(project)
    (entry,) = read_registry_index(project)
    ts.replace(
        registry_index_key(project),
        RegistryIndex(
            entries=[
                entry.model_copy(
                    update={"checkpoint_path": "../outside/evil.pt"}
                )
            ]
        ).model_dump(mode="json"),
    )

    with pytest.raises(RegistryPathTraversalError):
        reg.list_models()


@pytest.mark.parametrize(
    "reader", ["checkpoint_files", "registry_checkpoint_disclosures"]
)
def test_the_archive_readers_refuse_a_stored_entry_whose_path_climbs_out_of_the_project(
    tmp_path: Path, reader: str
) -> None:
    """A stored entry whose checkpoint path carries ``..`` refuses each archive reader of it."""
    import tcip_mcp.model_registry as model_registry
    from tcip_mcp.registry_paths import RegistryPathTraversalError

    project = tmp_path / "proj"
    _register_internal_checkpoint(project)
    (entry,) = read_registry_index(project)
    ts.replace(
        registry_index_key(project),
        RegistryIndex(
            entries=[
                entry.model_copy(
                    update={"checkpoint_path": "../outside/evil.pt"}
                )
            ]
        ).model_dump(mode="json"),
    )

    with pytest.raises(RegistryPathTraversalError):
        getattr(model_registry, reader)(project)


# ── the symlinked-checkpoint spelling rule: stores the resolved location, not the symlink ──


def test_a_symlink_to_a_checkpoint_outside_root_stores_the_resolved_external_location(
    tmp_path: Path,
) -> None:
    """A symlink to a checkpoint outside the root stores the resolved external location.

    A symlink under the project root pointing at a file outside it stores that file's resolved
    external location, not a relative spelling of the link's own in-tree name.
    """
    project = tmp_path / "proj"
    real_dir = tmp_path / "real_weights"
    real_dir.mkdir()
    real = real_dir / "m.pt"
    _checkpoint(real, "weights")
    link_dir = project / ".tcip" / "models"
    link_dir.mkdir(parents=True)
    link = link_dir / "m_link.pt"
    try:
        link.symlink_to(real)
    except OSError as exc:
        pytest.skip(f"this environment cannot create a symlink: {exc}")

    reg = ModelRegistry(str(project))
    reg.register_model("m", str(link))

    stored = read_registry_index(project)[0].checkpoint_path
    assert stored == str(real.resolve())
    assert is_external_form(stored)


def test_a_symlink_to_a_checkpoint_under_root_stores_the_resolved_internal_location(
    tmp_path: Path,
) -> None:
    """A symlink to a checkpoint under the root stores the target's relative spelling.

    A symlink whose resolved target sits under the project root stores that target's own
    project-relative spelling, not the link's.
    """
    project = tmp_path / "proj"
    real_dir = project / "weights_home"
    real_dir.mkdir(parents=True)
    real = real_dir / "m.pt"
    _checkpoint(real, "weights")
    link_dir = project / ".tcip" / "models"
    link_dir.mkdir(parents=True)
    link = link_dir / "m_link.pt"
    try:
        link.symlink_to(real)
    except OSError as exc:
        pytest.skip(f"this environment cannot create a symlink: {exc}")

    reg = ModelRegistry(str(project))
    reg.register_model("m", str(link))

    stored = read_registry_index(project)[0].checkpoint_path
    assert stored == "weights_home/m.pt"
    assert not is_external_form(stored)


def test_a_relative_root_still_answers_an_absolute_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``resolved_registry_path`` over a relative root answers the absolute path under the cwd."""
    from tcip_mcp.registry_paths import resolved_registry_path

    monkeypatch.chdir(tmp_path)
    (tmp_path / "subdir").mkdir()

    result = resolved_registry_path("subdir", "m.pt")

    assert result.is_absolute()
    assert result == (tmp_path / "subdir" / "m.pt").resolve()
