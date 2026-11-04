"""A stored record its model refuses answers, at every door, the sentence its reader raises.

Each record is written past its writer through ``tcip_store.replace`` and read through the
platform's own door. The reader's error names where the record lives and carries the model's
issues, never the validation library's documentation link; a door answering text answers that
sentence, and a door answering the refusal as a record answers the issues the sentence states.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import tcip_store

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi.testclient import TestClient

    from tcip_web.state import OpenProject

LIBRARY_LINK = "errors.pydantic.dev"
"""The host of the validation library's documentation link, whose path names its release."""


def _states(sentence: str, served: dict) -> bool:
    """Return whether ``sentence`` ends in the issues the served refusal record ``served`` holds."""
    from tcip_store import Invalid

    return sentence.endswith(f": {Invalid.model_validate(served)}")


@pytest.mark.usefixtures("seed_bud_operationalization")
def test_a_confirmation_over_a_refused_trait_record_answers_its_readers_sentence(
    client: TestClient, opened_project: Path
) -> None:
    """Confirming a trait whose stored record its schema refuses answers 400 with the sentence.

    The sentence is the one ``read_trait`` raises, naming the trait and the refused field.
    """
    from tcip_mcp import traits

    record = traits.read_trait("bud_opening", opened_project)
    revision = record.latest
    stored = record.model_dump(mode="json")
    stored["revisions"][0]["entry"]["localization"] = "unicorn_match"
    tcip_store.replace(traits.trait_key(opened_project, "bud_opening"), stored)

    with pytest.raises(ValueError) as refused:
        traits.read_trait("bud_opening", opened_project)
    resp = client.post(
        "/api/results/traits/confirm",
        json={
            "trait": "bud_opening",
            "revision": revision.number,
            "entry_sha256": revision.entry_sha256,
            "user": "tester",
            "confirmed": True,
        },
    )

    sentence = str(refused.value)
    assert LIBRARY_LINK not in sentence
    assert "'bud_opening'" in sentence
    assert "revisions.0.entry.localization" in sentence
    assert resp.status_code == 400
    assert resp.json()["detail"] == sentence


@pytest.mark.usefixtures("seed_bud_operationalization", "opened_project")
def test_a_confirmation_of_a_trait_the_project_lacks_answers_its_refusal_as_text(
    client: TestClient,
) -> None:
    """Confirming a trait the project holds no record of answers 400 with the refusal's text.

    The detail is the sentence naming the project's traits, as the route's other refusals answer.
    """
    resp = client.post(
        "/api/results/traits/confirm",
        json={
            "trait": "unicorn",
            "revision": 1,
            "entry_sha256": "0" * 64,
            "user": "tester",
            "confirmed": True,
        },
    )

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert isinstance(detail, str)
    assert "'unicorn'" in detail and "bud_opening" in detail


def test_opening_a_project_whose_gui_snapshot_is_refused_answers_its_readers_issues(
    client: TestClient, made: OpenProject
) -> None:
    """Opening a project whose GUI snapshot its model refuses answers 409 with the issues.

    The sentence ``read_gui_snapshot`` raises names the project, and the route's detail holds the
    issues it states.
    """
    from tcip_mcp.web_client import gui_snapshot_key, read_gui_snapshot

    tcip_store.replace(
        gui_snapshot_key(made.root),
        {"active_tab": "not-a-real-tab"},
        expect=tcip_store.Version.ABSENT,
    )

    with pytest.raises(ValueError) as refused:
        read_gui_snapshot(made.root)
    resp = client.post(
        "/api/projects/open", json={"id": made.id, "user": "grower"}
    )

    sentence = str(refused.value)
    assert LIBRARY_LINK not in sentence
    assert str(made.root) in sentence
    assert resp.status_code == 409
    assert _states(sentence, resp.json()["detail"])


def test_opening_a_project_whose_gui_snapshot_its_writer_wrote_opens_it(
    client: TestClient, made: OpenProject
) -> None:
    """A GUI snapshot written by ``write_gui_snapshot`` reads, so the project opens."""
    from tcip_mcp.web_client import GUI_STATE_DEFAULTS, write_gui_snapshot

    write_gui_snapshot(made.root, GUI_STATE_DEFAULTS)

    resp = client.post(
        "/api/projects/open", json={"id": made.id, "user": "grower"}
    )

    assert resp.status_code == 200, resp.text


def test_a_refused_memory_record_lists_with_the_issues_its_reader_raises(
    project: Path,
) -> None:
    """A report its model refuses lists as unread, with the issues ``read_memory`` raises.

    The sentence names the record's id; a report its writer wrote beside it still reads.
    """
    from tcip_mcp.tools.meta_tools import (
        FrictionReport,
        MemoryKind,
        load_project_memory,
        memory_key,
        read_memory,
        report_friction,
    )

    kept = report_friction(
        project, category="missing_tool", detail="kept", actor=None
    )
    damaged = report_friction(
        project, category="missing_tool", detail="damaged", actor=None
    )
    stored = read_memory(FrictionReport, str(project), damaged["id"])
    tcip_store.replace(
        memory_key(FrictionReport, str(project), damaged["id"]),
        stored.model_dump(mode="json") | {"category": "not_a_category"},
    )

    with pytest.raises(ValueError) as refused:
        read_memory(FrictionReport, str(project), damaged["id"])
    entries = load_project_memory(project, MemoryKind.reports)["entries"]

    sentence = str(refused.value)
    assert LIBRARY_LINK not in sentence
    assert damaged["id"] in sentence
    by_id = {entry["id"]: entry for entry in entries}
    assert by_id[kept["id"]]["record"]["detail"] == "kept"
    assert _states(sentence, by_id[damaged["id"]]["reason"])


def test_a_refused_delivery_event_refuses_the_listing_with_its_readers_issues(
    client: TestClient, opened_project: Path
) -> None:
    """A delivery event its model refuses refuses the listing (400) with the issues it raises.

    The sentence ``read_delivery_events`` raises names the event's id.
    """
    from tcip_mcp import delivery

    tcip_store.replace(
        delivery.delivery_event_key(opened_project, "e1"), {"event_id": "e1"}
    )

    with pytest.raises(ValueError) as refused:
        delivery.read_delivery_events(opened_project)
    resp = client.get("/api/results/delivery-events")

    sentence = str(refused.value)
    assert LIBRARY_LINK not in sentence
    assert "e1" in sentence
    assert resp.status_code == 400
    assert _states(sentence, resp.json()["detail"])
