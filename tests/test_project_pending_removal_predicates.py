"""Tests for the pending-removal predicate split in ``tcip_mcp.workspace``.

A pending-removal marker is written directly through the store here, standing in for the
removal door itself: these tests are about what the marker means to every reader that already
exists (``adoptable_project_root``, ``active_project_if_present``, ``marker_problem``,
``ingest_images``, ``workspace_project_name``), not about the door itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tcip_store as ts
from tcip_mcp import workspace
from tcip_mcp.tools.ingest_tools import ingest_images

PENDING_RECORD = {
    "requested_at": "20260304T120000Z",
    "requested_by": "user:tester",
    "archive_path": "archive.zip",
    "holding_dir": "holding",
    "external_roots": [],
    "dependent_projects": [],
}


@pytest.fixture(autouse=True)
def _isolate_workspace(tmp_path_factory, monkeypatch):
    ws = tmp_path_factory.mktemp("workspace")
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    return ws


def _mark_pending(project_root: Path) -> None:
    ts.replace(workspace.pending_removal_key(project_root), PENDING_RECORD, expect=ts.Version.ABSENT)


def _make_project(name: str) -> Path:
    root = workspace.project_path(name)
    (root / ".tcip").mkdir(parents=True)
    return root


def test_adoptable_project_root_raises_for_a_pending_project():
    root = _make_project("sample_plot_alpha")
    _mark_pending(root)

    with pytest.raises(workspace.ProjectPendingRemoval, match="20260304T120000Z"):
        workspace.adoptable_project_root("sample_plot_alpha")


def test_adoptable_project_root_admits_an_ordinary_project():
    _make_project("sample_plot_beta")

    assert workspace.adoptable_project_root("sample_plot_beta").name == "sample_plot_beta"


def test_workspace_project_root_ignores_the_pending_marker():
    root = _make_project("sample_plot_gamma")
    _mark_pending(root)

    assert workspace.workspace_project_root("sample_plot_gamma") == root


def test_workspace_project_name_still_names_a_pending_project():
    root = _make_project("sample_plot_delta")
    _mark_pending(root)

    assert workspace.workspace_project_name(root) == "sample_plot_delta"


def test_a_marker_the_store_refuses_to_read_leaves_the_project_adoptable():
    """A store refusal reading the marker (an undecodable document) reads as no marker at
    all: the project stays adoptable exactly as it would with nothing written."""
    root = _make_project("sample_plot_epsilon")
    path = root / ".tcip" / "pending_removal.json"
    path.write_bytes(b"not json")

    assert workspace.adoptable_project_root("sample_plot_epsilon") == root


def test_active_project_if_present_reads_a_pending_marker_as_none():
    root = _make_project("sample_plot_zeta")
    _mark_pending(root)
    ts.replace(workspace.active_project_key(), "sample_plot_zeta")

    assert workspace.active_project_if_present() is None


def test_marker_problem_names_why_once_active_project_if_present_answers_none():
    root = _make_project("sample_plot_eta")
    _mark_pending(root)
    ts.replace(workspace.active_project_key(), "sample_plot_eta")

    assert workspace.active_project_if_present() is None
    problem = workspace.marker_problem()
    assert problem is not None
    assert "20260304T120000Z" in problem


def test_ingest_images_refuses_a_pending_project_by_name(tmp_path):
    from PIL import Image

    root = _make_project("sample_plot_theta")
    _mark_pending(root)

    src = tmp_path / "raw"
    src.mkdir()
    Image.new("RGB", (8, 8), (0, 0, 0)).save(src / "a.jpg")

    manifest = ingest_images(source=str(src), name="sample_plot_theta", site="a site")

    assert "error" in manifest
    assert "20260304T120000Z" in manifest["error"]
    assert not (root / "images").exists()


def test_ingest_images_admits_an_ordinary_project(tmp_path):
    from PIL import Image

    src = tmp_path / "raw"
    src.mkdir()
    Image.new("RGB", (8, 8), (0, 0, 0)).save(src / "a.jpg")

    manifest = ingest_images(source=str(src), name="sample_plot_iota", site="a site")

    assert "error" not in manifest
