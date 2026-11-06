"""A stored record its model refuses answers, at every door, the sentence its reader raises.

Each record is written past its writer through ``tcip_store.replace`` and read through the
platform's own door. The reader's error names where the record lives and carries the model's
issues, never the validation library's documentation link; a door answering text answers that
sentence, and a door answering the refusal as a record answers the place and the issues the
sentence states.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

import tcip_store

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi.testclient import TestClient

    from tcip_web.state import OpenProject

LIBRARY_LINK = "errors.pydantic.dev"
"""The host of the validation library's documentation link, whose path names its release."""


def _states(sentence: str, served: dict) -> bool:
    """Return whether ``sentence`` is the served refused read ``served`` read as text."""
    from tcip_store import RefusedRead

    return sentence == str(RefusedRead.model_validate(served))


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

    The snapshot ``write_gui_snapshot`` wrote has its tab changed to one outside the vocabulary.
    The sentence ``read_gui_snapshot`` raises names the project and that one field, and the
    route's detail holds the issues it states.
    """
    from tcip_mcp.web_client import (
        GUI_STATE_DEFAULTS,
        gui_snapshot_key,
        read_gui_snapshot,
        write_gui_snapshot,
    )

    write_gui_snapshot(made.root, GUI_STATE_DEFAULTS)
    key = gui_snapshot_key(made.root)
    tcip_store.replace(
        key, tcip_store.read(key) | {"active_tab": "not-a-real-tab"}
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
    detail = resp.json()["detail"]
    assert [issue["loc"] for issue in detail["refusal"]["issues"]] == [
        ["active_tab"]
    ]
    assert _states(sentence, detail)


def test_opening_a_project_whose_snapshot_names_no_capture_folder_answers_its_issues(
    client: TestClient, made: OpenProject
) -> None:
    """Opening a project whose snapshot's capture is no path segment answers 409, invalid.

    The snapshot ``write_gui_snapshot`` wrote over a real capture has its date changed to one
    holding a separator. The sentence ``read_gui_snapshot`` raises carries no library link, and
    the route's detail is the ``invalid`` refusal of the issues it states.
    """
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.web_client import (
        GUI_STATE_DEFAULTS,
        DatasetChoice,
        gui_snapshot_key,
        read_gui_snapshot,
        selection_for,
        write_gui_snapshot,
    )
    from tests._producer_fixtures import write_image

    write_image(image_dir(made.root, "2026-02-11") / "img.png", (8, 8))
    choice = DatasetChoice(
        dataset_root=str(made.root), date="2026-02-11", current_image_index=0
    )
    write_gui_snapshot(
        made.root,
        GUI_STATE_DEFAULTS.model_copy(
            update={"dataset": selection_for(choice)}
        ),
    )
    key = gui_snapshot_key(made.root)
    stored = cast("dict[str, Any]", tcip_store.read(key))
    stored["dataset"]["date"] = "2026/02/11"
    tcip_store.replace(key, stored)

    with pytest.raises(ValueError) as refused:
        read_gui_snapshot(made.root)
    resp = client.post(
        "/api/projects/open", json={"id": made.id, "user": "grower"}
    )

    sentence = str(refused.value)
    assert LIBRARY_LINK not in sentence
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["refusal"]["kind"] == "invalid"
    assert _states(sentence, detail)


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

    The sentence names the record's id, as the served entry's place does; a report its writer
    wrote beside it still reads.
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
    reason = by_id[damaged["id"]]["reason"]
    assert damaged["id"] in reason["where"]
    assert _states(sentence, reason)


def test_a_refused_delivery_event_refuses_the_listing_with_its_readers_issues(
    client: TestClient, opened_project: Path
) -> None:
    """A delivery event its model refuses refuses the listing (400) with the issues it raises.

    The event a real delivery recorded has its time changed to no instant. The sentence
    ``read_delivery_events`` raises names the event's id and that one field.
    """
    from tcip_mcp import delivery
    from tests._trait_fixtures import seed_confirmed_count
    from tests.test_delivery_events import _record

    event_id = _record(
        opened_project, seed_confirmed_count(opened_project)
    ).delivery_event_id
    key = delivery.delivery_event_key(opened_project, event_id)
    tcip_store.replace(
        key, tcip_store.read(key) | {"produced_at": "not an instant"}
    )

    with pytest.raises(ValueError) as refused:
        delivery.read_delivery_events(opened_project)
    resp = client.get("/api/results/delivery-events")

    sentence = str(refused.value)
    assert LIBRARY_LINK not in sentence
    assert event_id in sentence
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert [issue["loc"] for issue in detail["refusal"]["issues"]] == [
        ["produced_at"]
    ]
    assert _states(sentence, detail)


def test_a_preflight_over_a_detector_whose_objects_span_no_pixel_names_the_geometry(
    project: Path,
) -> None:
    """A preflight whose detector geometry cannot derive states why, naming the geometry.

    Every box the labels hold is half a pixel across, so the finest anchor size floors to
    nothing; the preflight's issue names the derived detector geometry and the refused field,
    and carries no library link.
    """
    pytest.importorskip("torch")
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir
    from tcip_mcp.tools.training_tools import preflight_config
    from tests._producer_fixtures import (
        label_image,
        painted_frame,
        registry_of,
        registry_over,
        small_detection_config,
    )

    registry_over(project, registry_of("bud"))
    images = image_dir(project, UNDATED_BUCKET)
    images.mkdir(parents=True)
    for i in range(4):
        image = images / f"img{i}.png"
        painted_frame(
            64, 64, (90, 90, 90), [((20, 20, 40, 40), (250, 250, 250))]
        ).save(image)
        label_image(
            image,
            [Annotation(subject="bud", geometry=BBox(10, 10, 10.5, 10.5))],
            64,
            64,
        )

    issues = preflight_config(
        project, small_detection_config(images, project=project)
    )["issues"]

    (issue,) = [i for i in issues if "anchor_base_size" in i]
    assert LIBRARY_LINK not in issue
    assert "detector geometry" in issue


def _registered_dataset(project: Path) -> dict:
    """Return a training configuration over ``project`` as a registered dataset of labeled images.

    The subject registry is written by ``write_subject_registry`` and the identity by
    ``register_dataset``.
    """
    pytest.importorskip("torch")
    from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir
    from tcip_mcp.tools.annotation_tools import write_subject_registry
    from tcip_mcp.tools.project_tools import register_dataset
    from tests._producer_fixtures import (
        seed_bud_images,
        small_detection_config,
    )

    images = seed_bud_images(image_dir(project, UNDATED_BUCKET))
    assert "error" not in write_subject_registry(
        project, str(project), {"bud": {"description": "one bud"}}, actor=None
    )
    assert "error" not in register_dataset(
        project, str(project), "currant", actor=None
    )
    return small_detection_config(images, project=project)


def _rewritten(path: Path, change: dict[str, Any]) -> None:
    """Rewrite the stored document at ``path`` with the top-level keys of ``change`` replaced."""
    stored = tcip_store.read_blob_versioned(path)
    document = cast("dict[str, Any]", tcip_store.decode_value(stored.value))
    tcip_store.put_blob(
        path,
        tcip_store.encode_record(document | change),
        expect=stored.version,
    )


def test_a_preflight_over_a_subject_registry_its_model_refuses_states_the_issues(
    project: Path,
) -> None:
    """A preflight over a registry its model refuses states the issues, with no library link.

    The registry ``write_subject_registry`` wrote has its subject's description changed to no
    text; the issue names the registry and that one field.
    """
    from tcip_mcp.dataset_layout import subjects_path
    from tcip_mcp.tools.training_tools import preflight_config

    config = _registered_dataset(project)
    assert preflight_config(project, config)["issues"] == []
    _rewritten(subjects_path(project), {"bud": {"description": 5}})

    (issue,) = preflight_config(project, config)["issues"]

    assert LIBRARY_LINK not in issue
    assert "the subject registry" in issue
    assert "subjects.0.description" in issue


def test_writing_a_registry_its_model_refuses_answers_its_readers_sentence_as_an_error(
    project: Path,
) -> None:
    """Writing a registry its model refuses answers an error holding the sentence.

    The subject states no description; the error names the dataset the subjects were stated for
    and that one field, and a registry that states one is written.
    """
    from tcip_mcp.tools.annotation_tools import write_subject_registry

    refused = write_subject_registry(
        project, str(project), {"bud": {}}, actor=None
    )
    written = write_subject_registry(
        project, str(project), {"bud": {"description": "one bud"}}, actor=None
    )

    assert LIBRARY_LINK not in refused["error"]
    assert refused["error"].startswith(f"the subjects stated for {project}: ")
    assert "subjects.0.description" in refused["error"]
    assert "error" not in written


def test_loading_a_registry_its_model_refuses_answers_500_with_its_readers_sentence(
    client: TestClient, opened_project: Path
) -> None:
    """Loading a stored registry its model refuses answers 500 with the sentence.

    The registry ``write_subject_registry`` wrote has its subject's description changed to no
    text; the detail names the registry's path and that one field.
    """
    from tcip_mcp.dataset_layout import subjects_path
    from tcip_mcp.tools.annotation_tools import write_subject_registry

    assert "error" not in write_subject_registry(
        opened_project,
        str(opened_project),
        {"bud": {"description": "one bud"}},
        actor=None,
    )
    _rewritten(subjects_path(opened_project), {"bud": {"description": 5}})

    resp = client.get(
        "/api/subjects/load",
        params={"dataset_root": str(opened_project), "date": "2026-03-02"},
    )

    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert LIBRARY_LINK not in detail
    assert (
        f"the subject registry at {subjects_path(opened_project)}: "
        "subjects.0.description"
    ) in detail


def test_a_check_of_a_dataset_identity_its_model_refuses_states_the_issues(
    project: Path,
) -> None:
    """A check of an identity its model refuses states the issues, with no library link.

    The identity ``register_dataset`` wrote has its crop changed to no text; the reason
    ``check_dataset_identity`` answers names the document and that one field. Bytes that do not
    decode refuse naming the document too.
    """
    from tcip_mcp.dataset_layout import dataset_identity_path
    from tcip_mcp.tools.project_tools import check_dataset_identity

    _registered_dataset(project)
    assert check_dataset_identity(project, str(project)).state == "unchanged"
    path = dataset_identity_path(project)
    _rewritten(path, {"crop": 5})

    checked = check_dataset_identity(project, str(project))

    assert checked.reason is not None
    assert LIBRARY_LINK not in checked.reason
    assert str(path) in checked.reason
    assert "crop:" in checked.reason

    tcip_store.put_blob(
        path,
        b"{ not json",
        expect=tcip_store.read_blob_versioned(path).version,
    )
    undecodable = check_dataset_identity(project, str(project))

    assert undecodable.reason is not None
    assert str(path) in undecodable.reason
