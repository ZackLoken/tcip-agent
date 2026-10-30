"""Tests for tcip_web.state.StateStore.

The tests cover versioning, persistence to the open project, reopening, the retained panel
events and the one project admission.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import pytest
from pydantic import ValidationError

import tcip_store
from tcip_mcp.agent_identity import RecordedIdentity
from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir
from tcip_mcp.web_client import (
    GUI_STATE_DEFAULTS,
    Banner,
    DatasetChoice,
    gui_snapshot_key,
    read_gui_snapshot,
    selection_for,
)
from tcip_web.state import (
    GuiMutationInvalidError,
    Held,
    NoProjectOpenError,
    OpenProject,
    PanelEnvelope,
    ProjectNotOpenError,
    StateStore,
)
from tests._web_fixtures import named_project

if TYPE_CHECKING:
    from collections.abc import Generator

    from tcip_store import Key, Txn


def test_version_increments_on_mutate() -> None:
    store = StateStore()
    assert store.version == 0
    asyncio.run(
        store.mutate(lambda _held: {"active_tab": "training"}, project=None)
    )
    assert store.version == 1
    asyncio.run(store.mutate(lambda _held: {"mode": "polygon"}, project=None))
    assert store.version == 2


def test_a_change_persists_to_the_project_open_when_it_is_made(
    tmp_path: Path,
) -> None:
    store = StateStore()
    proj_a = named_project(tmp_path / "A", "A")
    proj_b = named_project(tmp_path / "B", "B")

    asyncio.run(store.open_project(proj_a))
    asyncio.run(store.open_project(proj_b))
    asyncio.run(
        store.mutate(
            lambda _held: {"active_tab": "results"}, project=store.held()
        )
    )

    snapshot = read_gui_snapshot(proj_b.root)
    assert snapshot is not None and snapshot.active_tab == "results"
    assert not tcip_store.exists(gui_snapshot_key(proj_a.root))


def test_a_mutation_made_for_a_project_no_longer_open_is_refused_and_not_held(
    tmp_path: Path,
) -> None:
    """A mutation made for a project no longer open is refused and not held.

    A mutation captured for one project and applied after a switch is refused naming the open
    project; the state is unchanged and no snapshot is persisted for either project.
    """
    store = StateStore()
    proj_a = named_project(tmp_path / "A", "A")
    proj_b = named_project(tmp_path / "B", "B")
    asyncio.run(store.open_project(proj_a))
    captured = store.held()
    asyncio.run(store.open_project(proj_b))

    with pytest.raises(ProjectNotOpenError) as refused:
        asyncio.run(
            store.mutate(
                lambda _held: {"active_tab": "results"}, project=captured
            )
        )
    assert refused.value.open_project_id == proj_b.id
    assert store.state.active_tab == "annotate"
    assert not tcip_store.exists(gui_snapshot_key(proj_a.root))
    assert not tcip_store.exists(gui_snapshot_key(proj_b.root))


def test_a_mutation_made_for_no_project_persists_nowhere(
    tmp_path: Path,
) -> None:
    """A mutation made while no project was open is persisted nowhere.

    It is persisted neither with nothing open, nor under a project opened meanwhile, nor after
    a close.
    """
    store = StateStore()
    project = named_project(tmp_path / "A", "A")
    asyncio.run(
        store.mutate(lambda _held: {"active_tab": "results"}, project=None)
    )
    assert not tcip_store.exists(gui_snapshot_key(project.root))

    asyncio.run(store.open_project(project))
    asyncio.run(
        store.mutate(lambda _held: {"active_tab": "training"}, project=None)
    )
    assert store.state.active_tab == "training"
    assert not tcip_store.exists(gui_snapshot_key(project.root))

    asyncio.run(store.close_project(project.id))
    assert store.opened is None
    asyncio.run(
        store.mutate(lambda _held: {"active_tab": "results"}, project=None)
    )
    assert not tcip_store.exists(gui_snapshot_key(project.root))


def test_a_state_that_cannot_be_persisted_raises_and_is_not_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tcip_mcp.web_client as web_client

    store = StateStore()
    asyncio.run(store.open_project(named_project(tmp_path / "A", "A")))

    def _refuse(*args: object, **kwargs: object) -> NoReturn:
        raise tcip_store.StoreError("the store refused the write")

    monkeypatch.setattr(web_client.tcip_store, "replace", _refuse)
    with pytest.raises(tcip_store.StoreError):
        asyncio.run(
            store.mutate(
                lambda _held: {"active_tab": "results"}, project=store.held()
            )
        )
    assert store.state.active_tab == "annotate"


def test_a_reopened_project_holds_the_state_it_persisted(
    tmp_path: Path,
) -> None:
    project = named_project(tmp_path / "A", "A")
    image_dir(project.root / "ds", UNDATED_BUCKET).mkdir(parents=True)
    store = StateStore()
    asyncio.run(store.open_project(project))
    asyncio.run(
        store.mutate(
            lambda _held: {
                "active_tab": "results",
                "dataset": selection_for(
                    DatasetChoice(
                        dataset_root=str(project.root / "ds"),
                        date=UNDATED_BUCKET,
                        current_image_index=0,
                    )
                ),
            },
            project=store.held(),
        )
    )

    # A fresh store simulates a backend restart.
    restarted = StateStore()
    asyncio.run(restarted.open_project(project))
    assert restarted.state.active_tab == "results"
    dataset = restarted.state.dataset
    assert dataset is not None
    assert dataset.dataset_root == str(project.root / "ds")


def test_a_held_selection_survives_a_mutation_of_another_field(
    tmp_path: Path,
) -> None:
    """A selection mutated in is held whole through a later mutation of another field.

    Both mutations merge the held state, whose selection carries its images; the snapshot holds
    the choice alone.
    """
    from tests._producer_fixtures import write_image

    project = named_project(tmp_path / "A", "A")
    write_image(image_dir(project.root / "ds", "2026-01-01") / "a.jpg", (8, 8))
    selection = selection_for(
        DatasetChoice(
            dataset_root=str(project.root / "ds"),
            date="2026-01-01",
            current_image_index=0,
        )
    )
    store = StateStore()
    asyncio.run(store.open_project(project))

    asyncio.run(
        store.mutate(
            lambda _held: {"dataset": selection}, project=store.held()
        )
    )
    asyncio.run(
        store.mutate(
            lambda _held: {"active_tab": "results"}, project=store.held()
        )
    )

    assert store.state.dataset == selection
    assert [image.name for image in selection.images] == ["a.jpg"]
    persisted = tcip_store.read(gui_snapshot_key(project.root))
    assert isinstance(persisted, dict)
    assert set(persisted["dataset"]) == set(DatasetChoice.model_fields)


def test_every_served_image_name_is_the_one_a_capture_scan_derives(
    tmp_path: Path,
) -> None:
    """A selection's images, a labels-written event and a label load name an image alike.

    Each is the name the capture scan keys its logical images by, a photograph's and a band
    group's both.
    """
    import numpy as np
    import tifffile

    from tcip_mcp.pipelines.image_utils import logical_images_by_name
    from tcip_mcp.web_client import LabelsWritten
    from tcip_web.routes.annotate import load_labels
    from tests._producer_fixtures import write_band_group, write_image

    images = image_dir(tmp_path / "ds", "2026-01-01")
    write_image(images / "a.jpg", (8, 8))
    bands = {band: images / f"cap_{band}.tif" for band in ("Green", "Red")}
    for path in bands.values():
        tifffile.imwrite(path, np.zeros((8, 8), np.uint16))
    write_band_group(images, "cap", bands, central_wavelength_nm=None)

    selection = selection_for(
        DatasetChoice(
            dataset_root=str(tmp_path / "ds"),
            date="2026-01-01",
            current_image_index=0,
        )
    )

    scanned = list(logical_images_by_name(images))
    assert [image.name for image in selection.images] == scanned
    assert [
        LabelsWritten(image_path=image.path).name for image in selection.images
    ] == scanned
    assert [
        load_labels(image.path).name for image in selection.images
    ] == scanned


def test_opening_a_project_with_no_snapshot_holds_a_fresh_state(
    tmp_path: Path,
) -> None:
    store = StateStore()
    asyncio.run(store.open_project(named_project(tmp_path / "A", "A")))
    assert store.state == GUI_STATE_DEFAULTS


def test_opening_a_project_whose_snapshot_does_not_decode_refuses_and_leaves_the_open_one(
    tmp_path: Path,
) -> None:
    """Opening a project whose snapshot does not decode refuses and leaves the open one.

    A present snapshot naming a tab outside the vocabulary raises a validation error, and the
    project already open stays open.
    """
    store = StateStore()
    proj_a = named_project(tmp_path / "A", "A")
    proj_b = named_project(tmp_path / "B", "B")
    asyncio.run(store.open_project(proj_a))
    tcip_store.replace(
        gui_snapshot_key(proj_b.root),
        {"active_tab": "not-a-real-tab"},
        expect=tcip_store.Version.ABSENT,
    )

    with pytest.raises(ValidationError):
        asyncio.run(store.open_project(proj_b))
    assert store.opened == proj_a


def test_a_persisted_snapshot_stating_a_nested_field_partly_is_reported_not_defaulted(
    tmp_path: Path,
) -> None:
    """A snapshot whose ``view`` states none of its fields is refused, not defaulted.

    Opening such a snapshot raises naming ``view``; the whole snapshot the producer writes
    opens.
    """
    store = StateStore()
    project = named_project(tmp_path / "A", "A")
    asyncio.run(store.open_project(project))
    asyncio.run(
        store.mutate(
            lambda _held: {"active_tab": "results"}, project=store.held()
        )
    )
    written = tcip_store.read(gui_snapshot_key(project.root))
    assert isinstance(written, dict)

    reopened = StateStore()
    asyncio.run(reopened.open_project(project))
    assert reopened.state.active_tab == "results"

    tcip_store.replace(gui_snapshot_key(project.root), {**written, "view": {}})
    with pytest.raises(ValueError, match="view"):
        asyncio.run(StateStore().open_project(project))


def test_mutate_refuses_a_partial_nested_object_and_holds_nothing() -> None:
    store = StateStore()
    with pytest.raises(GuiMutationInvalidError, match="view"):
        asyncio.run(
            store.mutate(lambda _held: {"view": {"scale": 2.0}}, project=None)
        )
    assert store.version == 0
    assert store.state.view.scale == 1.0


def test_mutate_refuses_a_view_at_no_positive_zoom_and_holds_nothing() -> None:
    store = StateStore()
    view = {"scale": 0.0, "offset_x": 0.0, "offset_y": 0.0}
    with pytest.raises(GuiMutationInvalidError, match="scale"):
        asyncio.run(store.mutate(lambda _held: {"view": view}, project=None))
    assert store.version == 0


def test_mutate_refuses_an_unknown_tab_and_holds_nothing() -> None:
    store = StateStore()
    with pytest.raises(GuiMutationInvalidError):
        asyncio.run(
            store.mutate(
                lambda _held: {"active_tab": "nonexistent"}, project=None
            )
        )
    assert store.version == 0
    assert store.state.active_tab == "annotate"


def test_mutate_refuses_a_built_model_with_a_wrongly_typed_field_and_holds_nothing(
    tmp_path: Path,
) -> None:
    """A mutation carrying a built selection with a wrongly typed field is refused, holding nothing.

    The selection is the producer's over a capture holding an image, its index replaced; the
    store's version and state are unchanged.
    """
    from tests._producer_fixtures import write_image

    write_image(image_dir(tmp_path / "ds", "2026-01-01") / "a.jpg", (8, 8))
    bad_selection = selection_for(
        DatasetChoice(
            dataset_root=str(tmp_path / "ds"),
            date="2026-01-01",
            current_image_index=0,
        )
    ).model_copy(update={"current_image_index": "banana"})
    store = StateStore()
    with pytest.raises(GuiMutationInvalidError):
        asyncio.run(
            store.mutate(
                lambda _held: {"dataset": bad_selection}, project=None
            )
        )
    assert store.version == 0
    assert store.state.dataset is None


def test_mutate_refuses_an_unknown_top_level_key_and_holds_nothing() -> None:
    """A misspelled top-level key is refused.

    The key is ``activ_tab`` for ``active_tab``, and the store holds nothing.
    """
    store = StateStore()
    with pytest.raises(GuiMutationInvalidError):
        asyncio.run(
            store.mutate(lambda _held: {"activ_tab": "results"}, project=None)
        )
    assert store.version == 0
    assert store.state.active_tab == "annotate"


def test_retained_events_are_the_open_projects_and_go_when_another_opens(
    tmp_path: Path,
) -> None:
    store = StateStore()
    proj_a = named_project(tmp_path / "A", "A")
    proj_b = named_project(tmp_path / "B", "B")
    asyncio.run(store.open_project(proj_a))
    store.retain_event(
        PanelEnvelope(
            panel="meta",
            event=Banner(text="for a"),
            event_id="for_a#1",
            identity=RecordedIdentity(),
        )
    )
    assert [e.event_id for e in store.retained_events("meta")] == ["for_a#1"]

    asyncio.run(store.open_project(proj_b))
    assert store.retained_events("meta") == []
    asyncio.run(store.open_project(proj_a))
    assert store.retained_events("meta") == []


def test_admission_answers_the_open_project_and_refuses_any_other(
    tmp_path: Path,
) -> None:
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
    """A store whose open project is ``first`` at its first read and ``second`` after."""

    def __init__(self, first: OpenProject, second: OpenProject) -> None:
        super().__init__()
        self._reads = [
            Held(first, GUI_STATE_DEFAULTS),
            Held(second, GUI_STATE_DEFAULTS),
        ]

    @property
    def _now(self) -> Held:
        return self._reads.pop(0) if len(self._reads) > 1 else self._reads[0]

    @_now.setter
    def _now(self, value: Held) -> None:
        """Ignore the assignment, so the switching reads stand."""


def test_admission_reads_the_open_project_once(tmp_path: Path) -> None:
    """The admission reads the open project once and answers the project it compared against.

    A switch that lands after the admission's one read is refused at the next admission, naming
    the project now open.
    """
    proj_a = named_project(tmp_path / "A", "A")
    proj_b = named_project(tmp_path / "B", "B")
    store = _SwitchingStore(proj_a, proj_b)

    assert store.admit(proj_a.id) == proj_a
    with pytest.raises(ProjectNotOpenError) as refused:
        store.admit(proj_a.id)
    assert refused.value.open_project_id == proj_b.id


def test_a_switch_publishes_nothing_until_its_departure_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A switch publishes nothing until its departure commits.

    Opening B departs from A in A's stats transaction; while that transaction is open, and when
    its commit fails, the store still holds A with A's state, and B is published, identity and
    state together, only once the commit has returned.
    """
    import contextlib

    store = StateStore()
    proj_a = named_project(tmp_path / "A", "A")
    proj_b = named_project(tmp_path / "B", "B")
    asyncio.run(store.open_project(proj_a))
    asyncio.run(
        store.mutate(lambda _held: {"active_tab": "results"}, project=proj_a)
    )
    held_a = (store.opened, store.state)
    real = tcip_store.transaction
    seen: list = []
    failing = [True]

    @contextlib.contextmanager
    def observed(*keys: Key, **kwargs: float | None) -> Generator[Txn]:
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
    assert (store.opened, store.state) == (proj_b, GUI_STATE_DEFAULTS)


def test_a_close_naming_another_project_closes_nothing(tmp_path: Path) -> None:
    """A close naming a project that is not the open one closes nothing.

    A close naming no open project leaves A open, and a close of A waiting on the lock while B
    is switched in leaves B open.
    """
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
