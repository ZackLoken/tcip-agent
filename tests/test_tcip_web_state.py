"""Tests for tcip_web.state.StateStore: versioning, flush targeting, rehydrate."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import tcip_store
from tcip_mcp.web_client import gui_snapshot_key
from tcip_web.state import DatasetSelection, GuiState, StateStore


def test_version_increments_on_mutate() -> None:
    store = StateStore()
    assert store.version == 0
    asyncio.run(store.mutate({"active_tab": "review"}))
    assert store.version == 1
    asyncio.run(store.mutate({"mode": "polygon"}))
    assert store.version == 2


def test_flush_targets_the_project_open_at_flush_time_not_at_schedule_time(tmp_path: Path) -> None:
    """A project switch inside the debounce window must not write the new project's snapshot
    into the old project's gui.json: the destination is the project open when the flush runs."""
    store = StateStore()
    proj_a = tmp_path / "A"
    proj_b = tmp_path / "B"

    store.open_project(proj_a)
    store.open_project(proj_b)
    store._flush_sync()

    assert tcip_store.exists(gui_snapshot_key(str(proj_b)))
    assert not tcip_store.exists(gui_snapshot_key(str(proj_a)))


def test_nothing_persists_while_no_project_is_open(tmp_path: Path) -> None:
    """The open project is the only persistence root; a snapshot naming a project inside its own
    state is not one, so state edited to name a root never redirects a flush there."""
    store = StateStore()
    store._state = GuiState(dataset=DatasetSelection(project_root=str(tmp_path / "named")))
    store._flush_sync()
    assert not tcip_store.exists(gui_snapshot_key(str(tmp_path / "named")))


def test_load_from_disk_roundtrip(tmp_path: Path) -> None:
    store = StateStore()
    store.open_project(tmp_path)
    store._state = GuiState(
        active_tab="results",
        dataset=DatasetSelection(project_root=str(tmp_path), dataset_root=str(tmp_path / "ds")),
    )
    store._flush_sync()

    # A fresh store simulates a backend restart.
    restarted = StateStore()
    assert restarted.open_project(tmp_path) is True
    assert restarted.state.active_tab == "results"
    assert restarted.state.dataset.dataset_root == str(tmp_path / "ds")


def test_load_from_disk_missing_returns_false(tmp_path: Path) -> None:
    assert StateStore().load_from_disk(tmp_path) is False


def test_mutate_refuses_an_unknown_tab_and_holds_nothing() -> None:
    from tcip_web.state import GuiMutationInvalid

    store = StateStore()
    with pytest.raises(GuiMutationInvalid):
        asyncio.run(store.mutate({"active_tab": "nonexistent"}))
    assert store.version == 0
    assert store.state.active_tab == "annotate"


def test_mutate_refuses_a_built_model_with_a_wrongly_typed_field_and_holds_nothing() -> None:
    """A pre-built model instance passes model_copy untouched (revalidate_instances="never"), so
    mutate must dump it and validate the merged result rather than trust it as already valid."""
    from tcip_web.state import GuiMutationInvalid, ReviewFilters

    store = StateStore()
    bad_filters = ReviewFilters.model_construct(iou_threshold="banana")
    with pytest.raises(GuiMutationInvalid):
        asyncio.run(store.mutate({"review": bad_filters}))
    assert store.version == 0
    assert store.state.review.iou_threshold == 0.5


def test_mutate_refuses_an_unknown_top_level_key_and_holds_nothing() -> None:
    """A misspelled top-level key (``activ_tab`` for ``active_tab``) must not be silently
    dropped: before ``GuiState`` forbade extra fields this returned 200, bumped the version and
    scheduled a save with the typo simply ignored."""
    from tcip_web.state import GuiMutationInvalid

    store = StateStore()
    with pytest.raises(GuiMutationInvalid):
        asyncio.run(store.mutate({"activ_tab": "review"}))
    assert store.version == 0
    assert store.state.active_tab == "annotate"


def test_load_from_disk_holds_defaults_on_an_undecodable_snapshot(tmp_path: Path) -> None:
    """A snapshot that will not validate must not leave the previous project's state live under
    the new project's root: the held state resets to ``GuiState()`` defaults, and a mutation on
    the new project afterward persists no field carried over from the old one."""
    store = StateStore()
    proj_a = tmp_path / "A"
    proj_b = tmp_path / "B"

    store.open_project(proj_a)
    asyncio.run(store.mutate({"active_subject": "bud"}))
    assert store.state.active_subject == "bud"

    tcip_store.replace(gui_snapshot_key(str(proj_b)), {"active_tab": "not-a-real-tab"},
                       expect=tcip_store.Version.ABSENT)

    assert store.open_project(proj_b) is False
    assert store.state == GuiState()

    asyncio.run(store.mutate({"active_subject": "leaf"}))
    store._flush_sync()
    stored = tcip_store.read(gui_snapshot_key(str(proj_b)))
    assert stored["active_subject"] == "leaf"
