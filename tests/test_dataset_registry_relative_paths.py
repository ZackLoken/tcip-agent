"""The dataset registry's ``path``, relative inside the project and absolute outside it.

The path is relative to the project root when the dataset is the project's own tree, absolute
for a genuinely external dataset, and resolved on read by ``dataset_entry_path``.
"""

from __future__ import annotations

import json
from pathlib import Path

from tcip_annotation.state import Annotation, BBox
from tcip_mcp.dataset_layout import image_dir
from tcip_mcp.subject_registry import Subject, SubjectRegistry
from tcip_mcp.tools.project_tools import (
    DatasetEntry,
    read_datasets,
    register_dataset,
)
from tests._cli_fixtures import run_tcip
from tests._producer_fixtures import label_image, registry_over, write_image


def _make_dataset(root: Path) -> None:
    """Write a minimal nested-schema dataset: one image, its label and a registry."""
    write_image(image_dir(root, "2-11-26") / "img_000.jpg", (32, 32))
    label_image(
        image_dir(root, "2-11-26") / "img_000.jpg",
        [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))],
        32,
        32,
    )
    registry_over(
        root, SubjectRegistry(subjects=(Subject(name="bud", description=""),))
    )


# ── the writer: stores "." for the project's own tree ──────────────────────────


def test_register_dataset_stores_a_relative_path_for_the_projects_own_tree(
    tmp_path: Path,
) -> None:
    """A dataset that is the project's own tree registers as ``"."``, not as an absolute path."""
    src = tmp_path / "proj"
    _make_dataset(src)

    registered = register_dataset(src, str(src), crop="currant", actor=None)

    assert "error" not in registered
    assert read_datasets(src) == [DatasetEntry(id=registered["id"], path=".")]


# ── the accessor: resolves a relative entry, leaves an absolute one alone ──────


def test_dataset_entry_path_resolves_a_relative_entry_against_the_project_root(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.project_tools import dataset_entry_path

    src = tmp_path / "proj"
    _make_dataset(src)
    register_dataset(src, str(src), crop="currant", actor=None)

    entry = read_datasets(src)[0]

    assert dataset_entry_path(src, entry).resolve() == src.resolve()


def test_dataset_entry_path_leaves_an_external_absolute_entry_unchanged(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.project_tools import dataset_entry_path

    project = tmp_path / "proj"
    project.mkdir()
    external = tmp_path / "external"
    _make_dataset(external)
    assert "error" not in register_dataset(
        project, str(external), crop="currant", actor=None
    )

    (entry,) = read_datasets(project)
    assert entry.path == str(external.resolve())
    assert dataset_entry_path(project, entry) == external.resolve()


def test_register_dataset_stores_a_nested_relative_path_with_posix_separators(
    tmp_path: Path,
) -> None:
    """A dataset nested under the project registers a relative path with posix separators.

    A dataset under a subdirectory of the project (not the project's own tree) stores a deeper
    relative form than "."; it is posix-separated, and the accessor resolves it back to the
    nested directory.
    """
    from tcip_mcp.tools.project_tools import dataset_entry_path

    project = tmp_path / "proj"
    nested = project / "datasets" / "main"
    _make_dataset(nested)

    registered = register_dataset(
        project, str(nested), crop="currant", actor=None
    )

    assert "error" not in registered
    entries = read_datasets(project)
    assert entries[0].path == "datasets/main"
    assert "\\" not in entries[0].path
    assert (
        dataset_entry_path(project, entries[0]).resolve() == nested.resolve()
    )


def test_stored_path_returns_the_resolved_absolute_path_when_not_contained(
    tmp_path: Path,
) -> None:
    """A dataset genuinely outside the project is stored as its resolved absolute path.

    A spelling through '..' is stored resolved, with no '..' left in it.
    """
    from tcip_mcp.registry_paths import stored_path

    project = tmp_path / "proj"
    project.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    messy = external / ".." / "external"

    result = stored_path(messy, project)

    assert ".." not in result
    assert Path(result) == external.resolve()


def test_stored_path_recognizes_containment_through_a_symlinked_project_root(
    tmp_path: Path,
) -> None:
    """A symlinked project root recognizes a dataset under the directory it points at.

    Through a symlink aliasing the project root, a dataset under the real directory stores as a
    relative path.
    """
    import pytest

    from tcip_mcp.registry_paths import stored_path

    real_project = tmp_path / "real_proj"
    dataset = real_project / "datasets" / "main"
    dataset.mkdir(parents=True)
    alias = tmp_path / "alias_proj"
    try:
        alias.symlink_to(real_project, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks not available on this machine: {exc}")

    assert stored_path(dataset, alias) == "datasets/main"


# ── check_dataset_identity: identity-based moved, not a stored-string comparison ────────


def test_check_dataset_identity_stays_quiet_for_a_self_registered_project(
    tmp_path: Path,
) -> None:
    from tests._web_fixtures import new_project

    src = new_project(tmp_path / "proj").root
    _make_dataset(src)
    register_dataset(src, str(src), crop="currant", actor=None)

    result = run_tcip(
        "check-dataset-identity",
        [
            "--dataset-root",
            str(src),
            "--project",
            str(src),
            "--user",
            "tester",
        ],
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert [
        entry["state"] for entry in json.loads(result.stdout)["registered"]
    ] == ["in_place"]


def test_check_dataset_identity_still_fires_for_a_genuinely_moved_dataset(
    tmp_path: Path,
) -> None:
    import shutil

    import tcip_store
    from tests._web_fixtures import new_project

    orig = new_project(tmp_path / "orig").root
    _make_dataset(orig)
    register_dataset(orig, str(orig), crop="currant", actor=None)

    moved = tmp_path / "moved"
    tcip_store.release_root(orig)
    shutil.copytree(orig, moved)

    result = run_tcip(
        "check-dataset-identity",
        [
            *("--dataset-root", str(moved)),
            *("--project", str(orig), "--user", "tester"),
        ],
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert [
        entry["state"] for entry in json.loads(result.stdout)["registered"]
    ] == ["moved"]


# ── traits.py: refusal messages name the resolved root ──────────────────────────────────────


def test_resolve_statement_registry_names_the_resolved_root_not_the_stored_dot(
    tmp_path: Path,
) -> None:
    """The statement-registry refusal names each dataset's resolved root, not the stored dot.

    A project registering two datasets under its subdirectories refuses, and the refusal names
    both resolved roots and no ``'.'``.
    """
    from tcip_mcp.traits import resolve_statement_registry

    project = tmp_path / "proj"
    first = project / "datasets" / "first"
    second = project / "datasets" / "second"
    _make_dataset(first)
    _make_dataset(second)
    register_dataset(project, str(first), crop="currant", actor=None)
    register_dataset(project, str(second), crop="currant", actor=None)

    try:
        resolve_statement_registry(str(project), "")
        raised = None
    except ValueError as exc:
        raised = str(exc)

    assert raised is not None
    assert (
        "'.'" not in raised
    )  # never the registry's own stored spelling for a relative entry
    assert repr(str(first.resolve())) in raised
    assert repr(str(second.resolve())) in raised


# ── routes/results.py: evidence roots resolve a relatively-registered dataset ───────────────


def test_evidence_roots_resolves_a_relatively_registered_dataset(
    tmp_path: Path,
) -> None:
    """The evidence roots include a relatively registered dataset, resolved.

    A dataset registered under a subdirectory of the project is among the roots.
    """
    from tcip_web.routes.results import _evidence_roots

    project = tmp_path / "proj"
    dataset = project / "datasets" / "main"
    _make_dataset(dataset)
    register_dataset(project, str(dataset), crop="currant", actor=None)

    roots = _evidence_roots(project)

    assert any(p.resolve() == dataset.resolve() for p in roots)
