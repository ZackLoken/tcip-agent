"""Tests for tcip_web.state.StateStore: versioning, persistence to the open project, reopening,
the retained panel events and the one project admission."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError

import tcip_store
from tcip_mcp.web_client import DatasetSelection, GuiState, gui_snapshot_key
from tcip_web.state import (
    GuiMutationInvalidError, Held, NoProjectOpenError, OpenProject, ProjectNotOpenError,
    StateStore,
)

from tests._web_fixtures import named_project


def test_version_increments_on_mutate() -> None:
    store = StateStore()
    assert store.version == 0
    asyncio.run(store.mutate({"active_tab": "training"}, project=None))
    assert store.version == 1
    asyncio.run(store.mutate({"mode": "polygon"}, project=None))
    assert store.version == 2


def test_a_change_persists_to_the_project_open_when_it_is_made(tmp_path: Path) -> None:
    store = StateStore()
    proj_a = named_project(tmp_path / "A", "A")
    proj_b = named_project(tmp_path / "B", "B")

    asyncio.run(store.open_project(proj_a))
    asyncio.run(store.open_project(proj_b))
    asyncio.run(store.mutate({"active_tab": "results"}, project=store.held()))

    assert tcip_store.read(gui_snapshot_key(proj_b.root))["active_tab"] == "results"
    assert not tcip_store.exists(gui_snapshot_key(proj_a.root))


def test_a_mutation_made_for_a_project_no_longer_open_is_refused_and_not_held(
    tmp_path: Path,
) -> None:
    """A route captures the project it acts on once; the write admits that project under the
    lock, so a mutation computed for one project is never held or persisted as another's."""
    store = StateStore()
    proj_a = named_project(tmp_path / "A", "A")
    proj_b = named_project(tmp_path / "B", "B")
    asyncio.run(store.open_project(proj_a))
    captured = store.held()
    asyncio.run(store.open_project(proj_b))

    with pytest.raises(ProjectNotOpenError) as refused:
        asyncio.run(store.mutate({"active_tab": "results"}, project=captured))
    assert refused.value.open_project_id == proj_b.id
    assert store.state.active_tab == "annotate"
    assert not tcip_store.exists(gui_snapshot_key(proj_a.root))
    assert not tcip_store.exists(gui_snapshot_key(proj_b.root))


def test_a_mutation_made_for_no_project_persists_nowhere(tmp_path: Path) -> None:
    """A mutation made while no project was open is held and persisted nowhere, whatever is
    open by the time it is applied: not under a project opened meanwhile, and not after a
    close."""
    store = StateStore()
    project = named_project(tmp_path / "A", "A")
    asyncio.run(store.mutate({"active_tab": "results"}, project=None))
    assert not tcip_store.exists(gui_snapshot_key(project.root))

    asyncio.run(store.open_project(project))
    asyncio.run(store.mutate({"active_tab": "training"}, project=None))
    assert store.state.active_tab == "training"
    assert not tcip_store.exists(gui_snapshot_key(project.root))

    asyncio.run(store.close_project(project.id))
    assert store.opened is None
    asyncio.run(store.mutate({"active_tab": "results"}, project=None))
    assert not tcip_store.exists(gui_snapshot_key(project.root))


def test_a_state_that_cannot_be_persisted_raises_and_is_not_held(tmp_path: Path, monkeypatch):
    import tcip_mcp.web_client as web_client

    store = StateStore()
    asyncio.run(store.open_project(named_project(tmp_path / "A", "A")))

    def _refuse(*args, **kwargs):
        raise tcip_store.StoreError("the store refused the write")

    monkeypatch.setattr(web_client.tcip_store, "replace", _refuse)
    with pytest.raises(tcip_store.StoreError):
        asyncio.run(store.mutate({"active_tab": "results"}, project=store.held()))
    assert store.state.active_tab == "annotate"


def test_a_reopened_project_holds_the_state_it_persisted(tmp_path: Path) -> None:
    project = named_project(tmp_path / "A", "A")
    (project.root / "ds" / "images" / "undated").mkdir(parents=True)
    store = StateStore()
    asyncio.run(store.open_project(project))
    asyncio.run(store.mutate({"active_tab": "results", "dataset": DatasetSelection(
        dataset_root=str(project.root / "ds"), date="undated")}, project=store.held()))

    # A fresh store simulates a backend restart.
    restarted = StateStore()
    asyncio.run(restarted.open_project(project))
    assert restarted.state.active_tab == "results"
    assert restarted.state.dataset.dataset_root == str(project.root / "ds")


def test_opening_a_project_with_no_snapshot_holds_a_fresh_state(tmp_path: Path) -> None:
    store = StateStore()
    asyncio.run(store.open_project(named_project(tmp_path / "A", "A")))
    assert store.state == GuiState()


def test_opening_a_project_whose_snapshot_does_not_decode_refuses_and_leaves_the_open_one(
    tmp_path: Path,
) -> None:
    """A present snapshot that is not its producer's whole shape is reported, never read as a
    fresh state, and the project open before stays open."""
    store = StateStore()
    proj_a = named_project(tmp_path / "A", "A")
    proj_b = named_project(tmp_path / "B", "B")
    asyncio.run(store.open_project(proj_a))
    tcip_store.replace(gui_snapshot_key(proj_b.root), {"active_tab": "not-a-real-tab"},
                       expect=tcip_store.Version.ABSENT)

    with pytest.raises(ValidationError):
        asyncio.run(store.open_project(proj_b))
    assert store.opened == proj_a


def test_a_persisted_snapshot_stating_a_nested_field_partly_is_reported_not_defaulted(
    tmp_path: Path,
) -> None:
    """A snapshot whose ``view`` states none of its fields is not its producer's whole shape; the
    open refuses rather than filling the view with defaults. The whole snapshot the producer
    writes opens."""
    store = StateStore()
    project = named_project(tmp_path / "A", "A")
    asyncio.run(store.open_project(project))
    asyncio.run(store.mutate({"active_tab": "results"}, project=store.held()))
    written = tcip_store.read(gui_snapshot_key(project.root))

    reopened = StateStore()
    asyncio.run(reopened.open_project(project))
    assert reopened.state.active_tab == "results"

    tcip_store.replace(gui_snapshot_key(project.root), {**written, "view": {}})
    with pytest.raises(ValueError, match="view"):
        asyncio.run(StateStore().open_project(project))


def test_mutate_refuses_a_partial_nested_object_and_holds_nothing() -> None:
    store = StateStore()
    with pytest.raises(GuiMutationInvalidError, match="view"):
        asyncio.run(store.mutate({"view": {"scale": 2.0}}, project=None))
    assert store.version == 0
    assert store.state.view.scale == 1.0


def test_mutate_refuses_an_unknown_tab_and_holds_nothing() -> None:
    store = StateStore()
    with pytest.raises(GuiMutationInvalidError):
        asyncio.run(store.mutate({"active_tab": "nonexistent"}, project=None))
    assert store.version == 0
    assert store.state.active_tab == "annotate"


def test_mutate_refuses_a_built_model_with_a_wrongly_typed_field_and_holds_nothing() -> None:
    """A pre-built model instance passes model_copy untouched (revalidate_instances="never"), so
    mutate must dump it and validate the merged result rather than trust it as already valid."""
    store = StateStore()
    bad_dataset = DatasetSelection.model_construct(current_image_index="banana")
    with pytest.raises(GuiMutationInvalidError):
        asyncio.run(store.mutate({"dataset": bad_dataset}, project=None))
    assert store.version == 0
    assert store.state.dataset.current_image_index == 0


def test_mutate_refuses_an_unknown_top_level_key_and_holds_nothing() -> None:
    """A misspelled top-level key (``activ_tab`` for ``active_tab``) must not be silently
    dropped."""
    store = StateStore()
    with pytest.raises(GuiMutationInvalidError):
        asyncio.run(store.mutate({"activ_tab": "results"}, project=None))
    assert store.version == 0
    assert store.state.active_tab == "annotate"


def test_retained_events_are_the_open_projects_and_go_when_another_opens(tmp_path: Path) -> None:
    store = StateStore()
    proj_a = named_project(tmp_path / "A", "A")
    proj_b = named_project(tmp_path / "B", "B")
    asyncio.run(store.open_project(proj_a))
    store.retain_event("meta", {"event_type": "for_a"})
    assert [e["event_type"] for e in store.retained_events("meta")] == ["for_a"]

    asyncio.run(store.open_project(proj_b))
    assert store.retained_events("meta") == []
    asyncio.run(store.open_project(proj_a))
    assert store.retained_events("meta") == []


def test_admission_answers_the_open_project_and_refuses_any_other(tmp_path: Path) -> None:
    store = StateStore()
    project = named_project(tmp_path / "A", "A")
    with pytest.raises(NoProjectOpenError):
        store.admit(project.id)

    asyncio.run(store.open_project(project))
    assert store.admit(project.id) == project
    with pytest.raises(ProjectNotOpenError) as other:
        store.admit("0" * 12)
    assert other.value.open_project_id == project.id


class _SwitchingStore(StateStore):
    """A store whose open project is another one at every read after the first, standing in for
    a switch that lands between two reads of it."""

    def __init__(self, first: OpenProject, second: OpenProject) -> None:
        super().__init__()
        self._reads = [Held(first, GuiState()), Held(second, GuiState())]

    @property
    def _now(self) -> Held:
        return self._reads.pop(0) if len(self._reads) > 1 else self._reads[0]

    @_now.setter
    def _now(self, value) -> None:
        """``StateStore.__init__`` assigns the field; the switching reads stand regardless."""


def test_admission_reads_the_open_project_once(tmp_path: Path) -> None:
    """An admission that read the id and the root in two steps would admit one project's id and
    answer the other's root when a switch lands between them; reading the one value once, it
    answers the project it compared against."""
    proj_a = named_project(tmp_path / "A", "A")
    proj_b = named_project(tmp_path / "B", "B")
    store = _SwitchingStore(proj_a, proj_b)

    assert store.admit(proj_a.id) == proj_a
    with pytest.raises(ProjectNotOpenError) as refused:
        store.admit(proj_a.id)
    assert refused.value.open_project_id == proj_b.id


def test_a_switch_publishes_nothing_until_its_departure_commits(tmp_path: Path, monkeypatch):
    """Opening B departs from A in A's stats transaction; while that transaction is open, and when
    its commit fails, the store still holds A with A's state, and B is published, identity and
    state together, only once the commit has returned."""
    import contextlib

    store = StateStore()
    proj_a = named_project(tmp_path / "A", "A")
    proj_b = named_project(tmp_path / "B", "B")
    asyncio.run(store.open_project(proj_a))
    asyncio.run(store.mutate({"active_tab": "results"}, project=proj_a))
    held_a = (store.opened, store.state)
    real = tcip_store.transaction
    seen: list = []
    failing = [True]

    @contextlib.contextmanager
    def observed(*keys, **kwargs):
        with real(*keys, **kwargs) as txn:
            yield txn
            seen.append((store.opened, store.state))
            if failing[0]:
                raise tcip_store.StoreError("the commit failed")

    monkeypatch.setattr(tcip_store, "transaction", observed)
    with pytest.raises(tcip_store.StoreError):
        asyncio.run(store.open_project(proj_b))
    assert seen == [held_a]
    assert (store.opened, store.state) == held_a

    failing[0] = False
    asyncio.run(store.open_project(proj_b))
    assert seen[-1] == held_a
    assert (store.opened, store.state) == (proj_b, GuiState())


def test_a_close_naming_another_project_closes_nothing(tmp_path: Path) -> None:
    """The close compares and closes in one step under the lock: a close naming a project that
    is not the open one leaves the open one open, and a switch that lands while the close waits
    for the lock is seen by the comparison, so the project opened meanwhile stays open."""
    store = StateStore()
    proj_a = named_project(tmp_path / "A", "A")
    proj_b = named_project(tmp_path / "B", "B")
    asyncio.run(store.open_project(proj_a))
    asyncio.run(store.close_project("0" * 12))
    assert store.opened == proj_a

    async def switch_while_the_close_waits() -> None:
        async with store._lock:
            closing = asyncio.create_task(store.close_project(proj_a.id))
            await asyncio.sleep(0)
            store._now = Held(proj_b, store.state)
        await closing

    asyncio.run(switch_while_the_close_waits())
    assert store.opened == proj_b
    asyncio.run(store.close_project(proj_b.id))
    assert store.opened is None
