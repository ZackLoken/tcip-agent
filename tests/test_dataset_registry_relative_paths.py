"""The dataset registry's ``path``: relative to the project root when the dataset is the
project's own tree, absolute for a genuinely external dataset, resolved on read through one
accessor (``dataset_entry_path``). Every consumer that used to read a registry entry's ``path``
directly now resolves it through that accessor instead; this file exercises the accessor itself
and the consumers that are not already covered by their own test files
(``tcip_web.paths.allowed_roots`` has its own test in test_web_path_guard_permanent_on.py, and
the door's own round trip lands with the import door's commits).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox
from tcip_mcp import class_registry
from tcip_mcp.class_registry import ClassRegistry, Subject
from tcip_mcp.tools.project_tools import read_datasets, register_dataset

# dataset_entry_path is imported inside each test that needs it, not at module scope, so this
# module still collects at the pre-row baseline (where it does not exist yet).

_CHECK_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_dataset_identity.py"


def _make_dataset(root: Path) -> None:
    """A minimal nested-schema dataset (image + label + registry), so its fingerprint is real
    rather than the ``None`` a check_dataset_identity.py run reads as bespoke-or-empty."""
    (root / "images" / "2-11-26").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32)).save(root / "images" / "2-11-26" / "img_000.jpg")
    (root / "annotations" / "2-11-26").mkdir(parents=True, exist_ok=True)
    json_io.write_annotations(
        str(root / "annotations" / "2-11-26" / "img_000.json"),
        [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], 32, 32)
    class_registry.write_registry(
        root / "classes.json", ClassRegistry(subjects=(Subject(name="bud"),)))


def _run_check(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_CHECK_SCRIPT), *args],
        capture_output=True, text=True, timeout=60,
    )


# ── the writer: stores "." for the project's own tree ──────────────────────────


def test_register_dataset_stores_a_relative_path_for_the_projects_own_tree(tmp_path: Path):
    """At the pre-row baseline every dataset's registered path is absolute, the project's own
    tree included; this is the behavior the row changes."""
    src = tmp_path / "proj"
    _make_dataset(src)

    registered = register_dataset(str(src), crop="currant", project_root=str(src))

    assert "error" not in registered
    regs = read_datasets(src)
    assert regs[0]["path"] == "."


# ── the accessor: resolves a relative entry, leaves an absolute one alone ──────


def test_dataset_entry_path_resolves_a_relative_entry_against_the_project_root(tmp_path: Path):
    from tcip_mcp.tools.project_tools import dataset_entry_path

    src = tmp_path / "proj"
    _make_dataset(src)
    register_dataset(str(src), crop="currant", project_root=str(src))

    entry = read_datasets(src)[0]

    assert dataset_entry_path(src, entry).resolve() == src.resolve()


def test_dataset_entry_path_leaves_an_external_absolute_entry_unchanged(tmp_path: Path):
    from tcip_mcp.tools.project_tools import dataset_entry_path

    external = tmp_path / "external"
    external.mkdir()
    entry = {"path": str(external)}

    assert dataset_entry_path(tmp_path / "proj", entry) == Path(str(external))


def test_register_dataset_stores_a_nested_relative_path_with_posix_separators(tmp_path: Path):
    """A dataset under a subdirectory of the project (not the project's own tree) stores a
    deeper relative form than "."; it is posix-separated so the entry reads the same after a
    cross-machine move, and the accessor resolves it back to the same directory."""
    from tcip_mcp.tools.project_tools import dataset_entry_path

    project = tmp_path / "proj"
    nested = project / "datasets" / "main"
    _make_dataset(nested)

    registered = register_dataset(str(nested), crop="currant", project_root=str(project))

    assert "error" not in registered
    entries = read_datasets(project)
    assert entries[0]["path"] == "datasets/main"
    assert "\\" not in entries[0]["path"]
    assert dataset_entry_path(project, entries[0]).resolve() == nested.resolve()


def test_registry_path_for_returns_the_resolved_absolute_path_when_not_contained(tmp_path: Path):
    """A dataset genuinely outside the project is stored absolute; the fall-through must return
    the resolved form rather than echoing whatever spelling the caller passed, since the
    docstring promises "absolute ... and a '..' form is never produced"."""
    from tcip_mcp.tools.project_tools import registry_path_for

    project = tmp_path / "proj"
    project.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    messy = external / ".." / "external"

    result = registry_path_for(messy, project)

    assert ".." not in result
    assert Path(result) == external.resolve()


def test_registry_path_for_recognizes_containment_through_a_symlinked_project_root(tmp_path: Path):
    """Coverage: containment is decided by filesystem identity, so an alias of the project root
    (a symlink standing in for a junction) still recognizes a dataset under the real directory
    it points at."""
    import pytest

    from tcip_mcp.tools.project_tools import registry_path_for

    real_project = tmp_path / "real_proj"
    dataset = real_project / "datasets" / "main"
    dataset.mkdir(parents=True)
    alias = tmp_path / "alias_proj"
    try:
        alias.symlink_to(real_project, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks not available on this machine: {exc}")

    assert registry_path_for(dataset, alias) == "datasets/main"


# ── project_roots (scripts/_store_bootstrap.py): reaches a relatively-registered dataset ───


def test_project_roots_reaches_a_relatively_registered_datasets_state(tmp_path: Path):
    """The dataset is registered under a subdirectory of the project, not the project's own
    tree, so a resolved entry can only come from the registry-driven root project_roots adds,
    never from the project root project_roots always adds regardless of the registry."""
    from scripts._store_bootstrap import project_roots
    from tcip_store.layout_claims import ROOT

    project = tmp_path / "proj"
    dataset = project / "datasets" / "main"
    _make_dataset(dataset)
    register_dataset(str(dataset), crop="currant", project_root=str(project))

    roots = project_roots(project)

    root_entries = [Path(r) for r, layout in roots if layout == ROOT]
    assert any(p.resolve() == dataset.resolve() for p in root_entries)
    assert len(root_entries) == len(set(root_entries))


# ── check_dataset_identity.py: identity-based MOVED, not a stored-string comparison ────────


def test_check_dataset_identity_stays_quiet_for_a_self_registered_project(tmp_path: Path):
    src = tmp_path / "proj"
    _make_dataset(src)
    register_dataset(str(src), crop="currant", project_root=str(src))

    result = _run_check(str(src), "--project", str(src))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "MOVED" not in result.stdout


def test_check_dataset_identity_still_fires_for_a_genuinely_moved_dataset(tmp_path: Path):
    import shutil

    orig = tmp_path / "orig"
    _make_dataset(orig)
    register_dataset(str(orig), crop="currant", project_root=str(orig))

    moved = tmp_path / "moved"
    shutil.copytree(orig, moved)

    result = _run_check(str(moved), "--project", str(orig))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "MOVED" in result.stdout


# ── operationalization.py: refusal messages name the resolved root ─────────────────────────


def test_resolve_statement_registry_names_the_resolved_root_not_the_stored_dot(tmp_path: Path):
    """A project registering more than one dataset refuses to guess which one a statement's
    classes belong to. Both datasets are registered under subdirectories of the project (never
    the project's own tree), so the resolved paths the refusal names can only have come from the
    registry-driven ``roots`` list, never from the message's own echo of its ``project_root``
    argument (which the un-nested form let this assertion pass without discriminating)."""
    from tcip_mcp.operationalization import resolve_statement_registry

    project = tmp_path / "proj"
    first = project / "datasets" / "first"
    second = project / "datasets" / "second"
    _make_dataset(first)
    _make_dataset(second)
    register_dataset(str(first), crop="currant", project_root=str(project))
    register_dataset(str(second), crop="currant", project_root=str(project))

    try:
        resolve_statement_registry(str(project), "")
        raised = None
    except ValueError as exc:
        raised = str(exc)

    assert raised is not None
    assert "'.'" not in raised  # never the registry's own stored spelling for a relative entry
    assert repr(str(first.resolve())) in raised
    assert repr(str(second.resolve())) in raised


# ── routes/results.py: evidence roots resolve a relatively-registered dataset ───────────────


def test_evidence_roots_resolves_a_relatively_registered_dataset(tmp_path: Path):
    """The dataset is registered under a subdirectory of the project, not the project's own
    tree, so a resolved root can only come from resolving the registry entry, never from the
    project root _evidence_roots always includes regardless of the registry."""
    from tcip_web.routes.results import _evidence_roots

    project = tmp_path / "proj"
    dataset = project / "datasets" / "main"
    _make_dataset(dataset)
    register_dataset(str(dataset), crop="currant", project_root=str(project))

    roots = _evidence_roots(project)

    assert any(p.resolve() == dataset.resolve() for p in roots)
