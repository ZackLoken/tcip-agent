"""Tests for tcip_web.state.StateStore: versioning, persistence to the open project, reopening,
the retained panel events and the one project admission."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError

import tcip_store
from tcip_mcp.web_client import DatasetSelection, GuiState, gui_snapshot_key
from tcip_web.state import GuiMutationInvalid, ProjectNotOpen, StateStore

from tests._web_fixtures import new_project


def test_version_increments_on_mutate() -> None:
    store = StateStore()
    assert store.version == 0
    asyncio.run(store.mutate({"active_tab": "training"}))
    assert store.version == 1
    asyncio.run(store.mutate({"mode": "polygon"}))
    assert store.version == 2


def test_a_change_persists_to_the_project_open_when_it_is_made(tmp_path: Path) -> None:
    store = StateStore()
    proj_a = new_project(tmp_path / "A")
    proj_b = new_project(tmp_path / "B")

    asyncio.run(store.open_project(proj_a))
    asyncio.run(store.open_project(proj_b))
    asyncio.run(store.mutate({"active_tab": "results"}))

    assert tcip_store.read(gui_snapshot_key(proj_b))["active_tab"] == "results"
    assert not tcip_store.exists(gui_snapshot_key(proj_a))


def test_nothing_persists_while_no_project_is_open(tmp_path: Path) -> None:
    """The open project is the only persistence root, so a change with none open writes
    nothing, and closing a project stops its snapshot from being written."""
    store = StateStore()
    project = new_project(tmp_path / "A")
    asyncio.run(store.mutate({"active_tab": "results"}))
    assert not tcip_store.exists(gui_snapshot_key(project))

    asyncio.run(store.open_project(project))
    asyncio.run(store.close_project())
    asyncio.run(store.mutate({"active_tab": "training"}))
    assert not tcip_store.exists(gui_snapshot_key(project))


def test_a_state_that_cannot_be_persisted_raises_and_is_not_held(tmp_path: Path, monkeypatch):
    import tcip_mcp.web_client as web_client

    store = StateStore()
    asyncio.run(store.open_project(new_project(tmp_path / "A")))

    def _refuse(*args, **kwargs):
        raise tcip_store.StoreError("the store refused the write")

    monkeypatch.setattr(web_client.tcip_store, "replace", _refuse)
    with pytest.raises(tcip_store.StoreError):
        asyncio.run(store.mutate({"active_tab": "results"}))
    assert store.state.active_tab == "annotate"


def test_a_reopened_project_holds_the_state_it_persisted(tmp_path: Path) -> None:
    project = new_project(tmp_path / "A")
    (project / "ds").mkdir()
    store = StateStore()
    asyncio.run(store.open_project(project))
    asyncio.run(store.mutate({"active_tab": "results",
                              "dataset": DatasetSelection(dataset_root=str(project / "ds"))}))

    # A fresh store simulates a backend restart.
    restarted = StateStore()
    asyncio.run(restarted.open_project(project))
    assert restarted.state.active_tab == "results"
    assert restarted.state.dataset.dataset_root == str(project / "ds")


def test_opening_a_project_with_no_snapshot_holds_a_fresh_state(tmp_path: Path) -> None:
    store = StateStore()
    asyncio.run(store.open_project(new_project(tmp_path / "A")))
    assert store.state == GuiState()


def test_opening_a_project_whose_snapshot_does_not_decode_refuses_and_leaves_the_open_one(
    tmp_path: Path,
) -> None:
    """A present snapshot that is not its producer's whole shape is reported, never read as a
    fresh state, and the project open before stays open."""
    store = StateStore()
    proj_a = new_project(tmp_path / "A")
    proj_b = new_project(tmp_path / "B")
    asyncio.run(store.open_project(proj_a))
    tcip_store.replace(gui_snapshot_key(proj_b), {"active_tab": "not-a-real-tab"},
                       expect=tcip_store.Version.ABSENT)

    with pytest.raises(ValidationError):
        asyncio.run(store.open_project(proj_b))
    assert store.project_root == proj_a


def test_a_persisted_snapshot_stating_a_nested_field_partly_is_reported_not_defaulted(
    tmp_path: Path,
) -> None:
    """A snapshot whose ``view`` states none of its fields is not its producer's whole shape; the
    open refuses rather than filling the view with defaults. The whole snapshot the producer
    writes opens."""
    store = StateStore()
    project = new_project(tmp_path / "A")
    asyncio.run(store.open_project(project))
    asyncio.run(store.mutate({"active_tab": "results"}))
    written = tcip_store.read(gui_snapshot_key(project))

    reopened = StateStore()
    asyncio.run(reopened.open_project(project))
    assert reopened.state.active_tab == "results"

    tcip_store.replace(gui_snapshot_key(project), {**written, "view": {}})
    with pytest.raises(ValueError, match="view"):
        asyncio.run(StateStore().open_project(project))


def test_mutate_refuses_a_partial_nested_object_and_holds_nothing() -> None:
    store = StateStore()
    with pytest.raises(GuiMutationInvalid, match="view"):
        asyncio.run(store.mutate({"view": {"scale": 2.0}}))
    assert store.version == 0
    assert store.state.view.scale == 1.0


def test_mutate_refuses_an_unknown_tab_and_holds_nothing() -> None:
    store = StateStore()
    with pytest.raises(GuiMutationInvalid):
        asyncio.run(store.mutate({"active_tab": "nonexistent"}))
    assert store.version == 0
    assert store.state.active_tab == "annotate"


def test_mutate_refuses_a_built_model_with_a_wrongly_typed_field_and_holds_nothing() -> None:
    """A pre-built model instance passes model_copy untouched (revalidate_instances="never"), so
    mutate must dump it and validate the merged result rather than trust it as already valid."""
    store = StateStore()
    bad_dataset = DatasetSelection.model_construct(current_image_index="banana")
    with pytest.raises(GuiMutationInvalid):
        asyncio.run(store.mutate({"dataset": bad_dataset}))
    assert store.version == 0
    assert store.state.dataset.current_image_index == 0


def test_mutate_refuses_an_unknown_top_level_key_and_holds_nothing() -> None:
    """A misspelled top-level key (``activ_tab`` for ``active_tab``) must not be silently
    dropped."""
    store = StateStore()
    with pytest.raises(GuiMutationInvalid):
        asyncio.run(store.mutate({"activ_tab": "results"}))
    assert store.version == 0
    assert store.state.active_tab == "annotate"


def test_retained_events_are_the_open_projects_and_go_when_another_opens(tmp_path: Path) -> None:
    store = StateStore()
    proj_a = new_project(tmp_path / "A")
    proj_b = new_project(tmp_path / "B")
    asyncio.run(store.open_project(proj_a))
    store.retain_event("meta", {"event_type": "for_a"})
    assert [e["event_type"] for e in store.retained_events("meta")] == ["for_a"]

    asyncio.run(store.open_project(proj_b))
    assert store.retained_events("meta") == []
    asyncio.run(store.open_project(proj_a))
    assert store.retained_events("meta") == []


def test_admission_answers_the_open_project_and_refuses_any_other(tmp_path: Path) -> None:
    from tcip_mcp.project_record import read_record

    store = StateStore()
    project = new_project(tmp_path / "A")
    with pytest.raises(ProjectNotOpen) as none_open:
        store.admit(read_record(project)["id"])
    assert none_open.value.open_project_id is None

    asyncio.run(store.open_project(project))
    assert store.admit(read_record(project)["id"]) == project
    with pytest.raises(ProjectNotOpen) as other:
        store.admit("0" * 12)
    assert other.value.open_project_id == read_record(project)["id"]
