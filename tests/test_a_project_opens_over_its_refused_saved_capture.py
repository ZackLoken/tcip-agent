"""A project whose saved selection lies in a capture the enumerator refuses still opens.

The refusal is carried in the held state as the saved choice and the enumerator's sentence, which
the state route and the agent's ``view_gui_state`` both serve, an act on the selection answers,
and a select that builds replaces. Each scene is made through the web application's routes: the
select route saves a choice, an agent's focus event (``focus_human_attention``'s own, posted
through the panel route) and the tab route save a mode, a subject and a tab, the open route
leaves for another project and returns, and a second photograph of one case-folded stem written
into the capture makes it ambiguous, as ``ingest_images`` refuses to bring one in.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import pytest

from tcip_mcp.dataset_layout import image_dir
from tests._mapping_fixtures import CROP
from tests._producer_fixtures import (
    GroupedDataset,
    labeled_grouped_capture,
    register_dataset_at,
    write_image,
)

if TYPE_CHECKING:
    import httpx
    from fastapi.testclient import TestClient

    from tcip_web.state import OpenProject

CAPTURE = "2026-04-01"
"""The capture saved as the selection: a band group and a photograph."""

SOUND = "2026-04-02"
"""A second capture of the same dataset, holding one photograph."""

SIZE = 16
"""The edge, in pixels, of every image written here."""

NEXT_STEP = "Rename so each logical image has its own stem."
"""The next step the enumerator's sentence for an ambiguous stem ends with."""

SAVED_TAB = "results"
"""The tab saved through the tab route, not the default one."""


class Scene(NamedTuple):
    """The project, the project opened between, and the dataset whose capture is saved."""

    project: OpenProject
    other: OpenProject
    written: GroupedDataset


def _opened(client: TestClient, project: OpenProject) -> httpx.Response:
    """Open ``project`` through the open route as a grower; return the answer."""
    return client.post(
        "/api/projects/open", json={"id": project.id, "user": "grower"}
    )


def _selected(client: TestClient, root: Path, date: str) -> httpx.Response:
    """Select ``date`` of the dataset at ``root`` through the select route; return the answer."""
    return client.post(
        "/api/dataset/select", json={"dataset_root": str(root), "date": date}
    )


def _reopened(client: TestClient, scene: Scene) -> httpx.Response:
    """Open the other project, asserted to succeed and name it, then reopen the scene's own."""
    away = _opened(client, scene.other)
    assert away.status_code == 200, away.text
    assert away.json()["project"]["id"] == scene.other.id
    return _opened(client, scene.project)


def _ambiguous(written: GroupedDataset) -> Path:
    """Write a second photograph of ``plain_002``'s case-folded stem beside it; return it."""
    return write_image(
        written.images_dir / "Plain_002.png", (SIZE, SIZE), (9, 9, 9)
    )


@pytest.fixture
def scene(
    client: TestClient,
    opened: OpenProject,
    monkeypatch: pytest.MonkeyPatch,
) -> Scene:
    """Save a selection of :data:`CAPTURE`, a focus and :data:`SAVED_TAB`, then make it ambiguous.

    The dataset holds :data:`CAPTURE` and :data:`SOUND`; another project stands beside this one
    in the workspace.
    """
    from tcip_mcp.tools.gui_tools import focus_human_attention
    from tests._web_fixtures import named_project, posted_focus

    written = labeled_grouped_capture(opened.root / "ds", CAPTURE)
    register_dataset_at(opened.root, written.root, crop=CROP)
    write_image(image_dir(written.root, SOUND) / "later.jpg", (SIZE, SIZE))
    other = named_project(opened.root.parent / "other", "other")
    chosen = _selected(client, written.root, CAPTURE)
    assert chosen.status_code == 200, chosen.text
    focused = posted_focus(
        client,
        opened,
        monkeypatch,
        lambda bound: focus_human_attention(
            bound, str(written.root), "bud", CAPTURE, mode="point"
        ),
    )
    assert focused.status_code == 200, focused.text
    tab = client.post("/api/state/tab", json={"active_tab": SAVED_TAB})
    assert tab.status_code == 200, tab.text
    _ambiguous(written)
    return Scene(opened, other, written)


def _assert_states_the_refused_capture(
    refused: dict, written: GroupedDataset
) -> None:
    """Assert ``refused`` holds :data:`CAPTURE` of ``written`` and the enumerator's sentence.

    The sentence is asserted by the capture's directory and the next step it names.
    """
    assert refused["choice"]["date"] == CAPTURE
    assert Path(refused["choice"]["dataset_root"]).samefile(written.root)
    assert str(written.images_dir) in refused["sentence"], refused
    assert refused["sentence"].endswith(NEXT_STEP), refused


def test_a_project_whose_saved_capture_is_refused_opens_and_states_it(
    client: TestClient, scene: Scene
) -> None:
    """The reopen answers 200, and both doors serve the saved state with the refused choice.

    The state route and ``view_gui_state`` each carry no selection, the saved tab, mode and
    subject, and the saved capture with the enumerator's sentence, the same in each.
    """
    from tcip_mcp.tools.project_tools import view_gui_state

    reopened = _reopened(client, scene)

    assert reopened.status_code == 200, reopened.text
    assert client.get("/api/projects").json()["open_id"] == scene.project.id
    served = client.get("/api/state").json()
    read = view_gui_state(scene.project.root)
    for state in (served, read):
        assert state["dataset"] is None
        assert (
            state["active_tab"],
            state["mode"],
            state["active_subject"],
        ) == (SAVED_TAB, "point", "bud")
        _assert_states_the_refused_capture(
            state["refused_choice"], scene.written
        )
    assert served["refused_choice"] == read["refused_choice"]


def test_the_refused_choice_survives_a_change_that_selects_nothing(
    client: TestClient, scene: Scene
) -> None:
    """After the reopen, each tab change persists the refused choice, read afresh each time."""
    from tcip_mcp.web_client import read_gui_snapshot

    assert _reopened(client, scene).status_code == 200
    reads = []
    for tab in ("training", "annotate"):
        moved = client.post("/api/state/tab", json={"active_tab": tab})
        assert moved.status_code == 200, moved.text
        read = read_gui_snapshot(scene.project.root)
        assert read is not None and read.refused_choice is not None
        reads.append(read.refused_choice)
        _assert_states_the_refused_capture(
            read.refused_choice.model_dump(mode="json"), scene.written
        )
    assert reads[0] == reads[1]


def test_an_act_on_the_selection_states_the_refused_capture(
    client: TestClient, scene: Scene
) -> None:
    """After the reopen, the nav route answers 400 with the enumerator's sentence."""
    assert _reopened(client, scene).status_code == 200

    stepped = client.post("/api/dataset/nav", json={"current_image_index": 0})

    assert stepped.status_code == 400, stepped.text
    detail = stepped.json()["detail"]
    assert (
        detail == client.get("/api/state").json()["refused_choice"]["sentence"]
    )
    assert str(scene.written.images_dir) in detail and detail.endswith(
        NEXT_STEP
    )


def test_a_select_that_builds_replaces_the_refused_choice(
    client: TestClient, scene: Scene
) -> None:
    """After the reopen, a sound capture selects and empties the field; the refused one is 400."""
    assert _reopened(client, scene).status_code == 200

    sound = _selected(client, scene.written.root, SOUND)

    assert sound.status_code == 200, sound.text
    assert [
        image["name"] for image in sound.json()["selection"]["images"]
    ] == ["later.jpg"]
    state = client.get("/api/state").json()
    assert state["dataset"]["date"] == SOUND
    assert state["refused_choice"] is None
    refused = _selected(client, scene.written.root, CAPTURE)
    assert refused.status_code == 400, refused.text
    assert str(scene.written.images_dir) in refused.json()["detail"]
    assert refused.json()["detail"].endswith(NEXT_STEP)


def test_a_repaired_capture_selects_and_reopens_with_no_refused_choice(
    client: TestClient, scene: Scene
) -> None:
    """With the second photograph gone, a select of the capture and a reopen each hold it."""
    assert _reopened(client, scene).status_code == 200
    (scene.written.images_dir / "Plain_002.png").unlink()

    chosen = _selected(client, scene.written.root, CAPTURE)

    assert chosen.status_code == 200, chosen.text
    for answer in (None, _reopened(client, scene)):
        assert answer is None or answer.status_code == 200, answer
        state = client.get("/api/state").json()
        assert state["dataset"]["date"] == CAPTURE
        assert state["refused_choice"] is None


def test_a_sound_saved_capture_reopens_at_its_saved_index(
    client: TestClient, opened: OpenProject
) -> None:
    """A capture saved at its second image through the nav route reopens there, nothing refused."""
    from tests._web_fixtures import named_project

    written = labeled_grouped_capture(opened.root / "ds", CAPTURE)
    register_dataset_at(opened.root, written.root, crop=CROP)
    scene = Scene(
        opened, named_project(opened.root.parent / "other", "other"), written
    )
    assert _selected(client, written.root, CAPTURE).status_code == 200
    stepped = client.post("/api/dataset/nav", json={"current_image_index": 1})
    assert stepped.status_code == 200, stepped.text

    assert _reopened(client, scene).status_code == 200

    state = client.get("/api/state").json()
    assert (
        state["dataset"]["date"],
        state["dataset"]["current_image_index"],
    ) == (
        CAPTURE,
        1,
    )
    assert len(state["dataset"]["images"]) == 2
    assert state["refused_choice"] is None


def test_a_project_with_no_snapshot_opens_with_the_defaults(
    client: TestClient, opened: OpenProject
) -> None:
    """A project the application never saved a state for opens on the defaults, nothing refused."""
    from tcip_mcp.web_client import GUI_STATE_DEFAULTS
    from tests._web_fixtures import named_project

    fresh = named_project(opened.root.parent / "fresh", "fresh")

    answer = _opened(client, fresh)

    assert answer.status_code == 200, answer.text
    state = client.get("/api/state").json()
    assert state == GUI_STATE_DEFAULTS.model_dump(mode="json")
    assert state["refused_choice"] is None
