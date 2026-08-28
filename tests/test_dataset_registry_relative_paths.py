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
        [Annotation(subject="catkin", geometry=BBox(1, 1, 9, 9))], 32, 32)
    class_registry.write_registry(
        root / "classes.json", ClassRegistry(subjects=(Subject(name="catkin"),)))


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

    registered = register_dataset(str(src), crop="hazelnut", project_root=str(src))

    assert "error" not in registered
    regs = read_datasets(src)
    assert regs[0]["path"] == "."


# ── the accessor: resolves a relative entry, leaves an absolute one alone ──────


def test_dataset_entry_path_resolves_a_relative_entry_against_the_project_root(tmp_path: Path):
    from tcip_mcp.tools.project_tools import dataset_entry_path

    src = tmp_path / "proj"
    _make_dataset(src)
    register_dataset(str(src), crop="hazelnut", project_root=str(src))

    entry = read_datasets(src)[0]

    assert dataset_entry_path(src, entry).resolve() == src.resolve()


def test_dataset_entry_path_leaves_an_external_absolute_entry_unchanged(tmp_path: Path):
    from tcip_mcp.tools.project_tools import dataset_entry_path

    external = tmp_path / "external"
    external.mkdir()
    entry = {"path": str(external)}

    assert dataset_entry_path(tmp_path / "proj", entry) == Path(str(external))


# ── project_roots (scripts/_store_bootstrap.py): reaches a relatively-registered dataset ───


def test_project_roots_reaches_a_relatively_registered_datasets_state(tmp_path: Path):
    from scripts._store_bootstrap import project_roots
    from tcip_store.layout_claims import ROOT

    src = tmp_path / "proj"
    _make_dataset(src)
    register_dataset(str(src), crop="hazelnut", project_root=str(src))

    roots = project_roots(src)

    root_entries = [Path(r) for r, layout in roots if layout == ROOT]
    assert any(p.resolve() == src.resolve() for p in root_entries)
    # The project root itself and its own dataset are the same directory; project_roots must
    # not add it twice just because one came from the registry and one from the project itself.
    assert len(root_entries) == len(set(root_entries))


# ── check_dataset_identity.py: identity-based MOVED, not a stored-string comparison ────────


def test_check_dataset_identity_stays_quiet_for_a_self_registered_project(tmp_path: Path):
    src = tmp_path / "proj"
    _make_dataset(src)
    register_dataset(str(src), crop="hazelnut", project_root=str(src))

    result = _run_check(str(src), "--project", str(src))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "MOVED" not in result.stdout


def test_check_dataset_identity_still_fires_for_a_genuinely_moved_dataset(tmp_path: Path):
    import shutil

    orig = tmp_path / "orig"
    _make_dataset(orig)
    register_dataset(str(orig), crop="hazelnut", project_root=str(orig))

    moved = tmp_path / "moved"
    shutil.copytree(orig, moved)

    result = _run_check(str(moved), "--project", str(orig))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "MOVED" in result.stdout


# ── operationalization.py: refusal messages name the resolved root ─────────────────────────


def test_resolve_statement_registry_names_the_resolved_root_not_the_stored_dot(tmp_path: Path):
    """A project registering more than one dataset refuses to guess which one a statement's
    classes belong to; the refusal used to quote the registry's own stored spelling, which
    reads as the meaningless "." for a project's own relatively-registered tree."""
    from tcip_mcp.operationalization import resolve_statement_registry

    src = tmp_path / "proj"
    _make_dataset(src)
    register_dataset(str(src), crop="hazelnut", project_root=str(src))
    external = tmp_path / "second_dataset"
    _make_dataset(external)
    register_dataset(str(external), crop="hazelnut", project_root=str(src))

    try:
        resolve_statement_registry(str(src), "")
        raised = None
    except ValueError as exc:
        raised = str(exc)

    assert raised is not None
    assert "'.'" not in raised  # never the registry's own stored spelling for the project's own tree
    assert repr(str(src.resolve())) in raised


# ── routes/results.py: evidence roots resolve a relatively-registered dataset ───────────────


def test_evidence_roots_resolves_a_relatively_registered_dataset(tmp_path: Path):
    from tcip_web.routes.results import _evidence_roots

    src = tmp_path / "proj"
    _make_dataset(src)
    register_dataset(str(src), crop="hazelnut", project_root=str(src))

    roots = _evidence_roots(src)

    assert any(p.resolve() == src.resolve() for p in roots)
