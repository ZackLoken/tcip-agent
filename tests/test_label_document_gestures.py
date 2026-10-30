"""The editor's gestures through the label save, and what readers read back.

These tests cover what the gestures write into the label document, the verdict shard and the
completion marks, and what the admission, the queue and the editor's own listing read back.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import tcip_store as ts
from tcip_annotation import json_io
from tcip_annotation.verdicts import Verdict, encode_verdict, read_verdicts
from tcip_mcp.dataset_layout import (
    Decisions,
    Gestures,
    image_dir,
    save_label_document,
    verdict_key_of,
)
from tcip_mcp.pipelines.proposal import Proposal
from tests._producer_fixtures import (
    image_label_key,
    registry_of,
    save_labels,
    staged_box,
    write_image,
)

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from tcip_annotation.state import Annotation

DATE = "2026-02-11"
WIDTH, HEIGHT = 100, 80
BOX = [40.0, 30.0, 60.0, 50.0]
RING = [[10.0, 10.0], [30.0, 10.0], [30.0, 25.0], [10.0, 25.0]]


def _stored(root: Path, image: Path) -> json_io.LabelDocument:
    return json_io.read_label_document(image_label_key(image))


def _as_loaded(root: Path, image: Path) -> list[dict]:
    """Return ``json_io.ClientAnnotation`` of each annotation stored for ``image``, dumped."""
    return [
        json_io.ClientAnnotation.of(a).model_dump(mode="json")
        for a in _stored(root, image).annotations
    ]


# ── corrections keep what they do not correct ──────────────────────────────


def test_correcting_a_value_keeps_the_geometry(tmp_path: Path) -> None:
    image = write_image(image_dir(tmp_path, DATE) / "a.jpg", (WIDTH, HEIGHT))
    drawn = {
        "subject": "bud",
        "rings": [RING],
        "attributes": {"stage": "open"},
    }
    save_labels(tmp_path, image, [drawn], author="user:first")

    (loaded,) = _as_loaded(tmp_path, image)
    save_labels(
        tmp_path,
        image,
        [{**loaded, "attributes": {"stage": "closed"}}],
        author="user:second",
    )

    (after,) = _stored(tmp_path, image).annotations
    assert after.geometry == json_io.annotation_from_payload(drawn).geometry
    assert after.attributes == {"stage": "closed"}
    assert after.created_by == "user:second"


def test_correcting_the_geometry_keeps_the_confirmed_value(
    tmp_path: Path,
) -> None:
    image = write_image(image_dir(tmp_path, DATE) / "a.jpg", (WIDTH, HEIGHT))
    save_labels(
        tmp_path,
        image,
        [{"subject": "bud", "bbox": BOX, "attributes": {"stage": "open"}}],
    )

    (loaded,) = _as_loaded(tmp_path, image)
    save_labels(
        tmp_path, image, [{**loaded, "bbox": [41.0, 31.0, 61.0, 51.0]}]
    )

    (after,) = _stored(tmp_path, image).annotations
    assert after.attributes == {"stage": "open"}
    assert (after.geometry.x1, after.geometry.y2) == (41.0, 51.0)


def test_adding_a_missed_object_writes_it_with_its_value(
    tmp_path: Path,
) -> None:
    image = write_image(image_dir(tmp_path, DATE) / "a.jpg", (WIDTH, HEIGHT))
    save_labels(tmp_path, image, [{"subject": "bud", "bbox": BOX}])

    save_labels(
        tmp_path,
        image,
        [
            *_as_loaded(tmp_path, image),
            {
                "subject": "bud",
                "bbox": [5.0, 5.0, 15.0, 15.0],
                "attributes": {"stage": "open"},
            },
        ],
        author="user:second",
    )

    added = _stored(tmp_path, image).annotations[1]
    assert added.attributes == {"stage": "open"}
    assert added.created_by == "user:second"


# ── a proposal is accepted or rejected, once, into the one decision log ────


def test_accepting_a_paired_proposal_confirms_the_annotation_without_a_second_one(
    tmp_path: Path,
) -> None:
    image = write_image(image_dir(tmp_path, DATE) / "a.jpg", (WIDTH, HEIGHT))
    bucket = staged_box(tmp_path, image, BOX)
    save_labels(tmp_path, image, [{"subject": "bud", "bbox": BOX}])

    save_labels(
        tmp_path,
        image,
        _as_loaded(tmp_path, image),
        gestures=Gestures(decide=Decisions(bucket=bucket, verdicts={0: "accepted"})),
    )

    assert len(_stored(tmp_path, image).annotations) == 1


def test_accepting_an_unpaired_proposal_adds_it_authored_by_its_producer(
    tmp_path: Path,
) -> None:
    image = write_image(image_dir(tmp_path, DATE) / "a.jpg", (WIDTH, HEIGHT))
    bucket = staged_box(tmp_path, image, BOX)

    save_labels(
        tmp_path,
        image,
        [],
        gestures=Gestures(decide=Decisions(bucket=bucket, verdicts={0: "accepted"})),
    )

    (accepted,) = _stored(tmp_path, image).annotations
    assert (accepted.created_by, accepted.accepted_by) == (
        "detector",
        "user:breeder",
    )
    assert accepted.score is None


def test_accepting_a_model_proposal_carries_no_attribute_value_into_the_document(
    tmp_path: Path,
) -> None:
    """Accepting a model proposal carries none of its attribute values into the document.

    A proposal whose model called ``color`` blue, accepted with no annotation paired, joins
    ground truth with its geometry and subject and no attribute value.
    """
    pytest.importorskip("torch")
    from tcip_mcp import subject_registry as cr
    from tests._chain_fixtures import predicted, published

    image = write_image(image_dir(tmp_path, DATE) / "a.jpg", (WIDTH, HEIGHT))
    color = cr.Attribute("color", "categorical", ("red", "blue"))
    registry = registry_of("bud", color)
    result = {
        **predicted(image, ["blue"], (color,)),
        "width": WIDTH,
        "height": HEIGHT,
    }
    bucket = published(
        tmp_path,
        f"model/{DATE}",
        [result],
        scope={"subject": "bud"},
        registry=registry,
    )
    (proposal,) = json_io.read_predictions(
        bucket.document_key(image.stem)
    ).annotations
    assert proposal.attributes == {"color": "blue"}

    save_labels(
        tmp_path,
        image,
        [],
        gestures=Gestures(decide=Decisions(bucket=bucket.name, verdicts={0: "accepted"})),
    )

    (accepted,) = _stored(tmp_path, image).annotations
    assert (accepted.subject, accepted.attributes) == ("bud", {})
    assert accepted.geometry == proposal.geometry


def test_the_shard_records_each_decision_once_and_refuses_an_entry_it_cannot_read(
    tmp_path: Path,
) -> None:
    image = write_image(image_dir(tmp_path, DATE) / "a.jpg", (WIDTH, HEIGHT))
    bucket = staged_box(tmp_path, image, BOX)

    save_labels(
        tmp_path,
        image,
        [],
        gestures=Gestures(decide=Decisions(bucket=bucket, verdicts={0: "accepted"})),
    )

    key = verdict_key_of(image_label_key(image), bucket)
    (decided,) = read_verdicts(key)
    assert decided == Verdict(
        proposal=0, action="accepted", by="user:breeder", at=decided.at
    )

    ts.append(key, {**encode_verdict(decided), "action": "maybe"})
    with pytest.raises(ValueError, match="maybe"):
        read_verdicts(key)


def test_a_decision_or_a_completion_mark_by_no_person_does_not_read(
    tmp_path: Path,
) -> None:
    """A verdict or completion mark by no person does not read.

    A verdict's and a completion mark's ``by`` name a person; one naming a producer that is no
    person refuses as the record it is read from.
    """
    from dataclasses import replace

    from tcip_annotation.verdicts import decode_verdict, encode_verdict

    image = write_image(image_dir(tmp_path, DATE) / "a.jpg", (WIDTH, HEIGHT))
    bucket = staged_box(tmp_path, image, BOX)
    save_labels(
        tmp_path,
        image,
        [],
        gestures=Gestures(
            decide=Decisions(bucket=bucket, verdicts={0: "accepted"}),
            complete={"bud": True},
        ),
    )

    (decided,) = read_verdicts(verdict_key_of(image_label_key(image), bucket))
    with pytest.raises(ValueError, match="a person"):
        decode_verdict(encode_verdict(replace(decided, by="save_annotations")))
    stored = ts.read(image_label_key(image))
    assert isinstance(stored, dict)
    marks = stored[json_io.COMPLETION_KEY]
    assert isinstance(marks, dict) and isinstance(marks["bud"], list)
    (mark,) = marks["bud"]
    assert isinstance(mark, dict)
    assert _stored(tmp_path, image).marks["bud"][0].by == mark["by"]
    ts.replace(
        image_label_key(image),
        {
            **stored,
            json_io.COMPLETION_KEY: {
                "bud": [{**mark, "by": "save_annotations"}]
            },
        },
    )
    with pytest.raises(
        json_io.UnreadableLabelDocumentError, match="recorded identity"
    ):
        _stored(tmp_path, image)


def test_rejecting_a_paired_proposal_records_the_rejection_and_keeps_the_annotation(
    tmp_path: Path, client: TestClient
) -> None:
    from tests._web_fixtures import open_new_project

    root = open_new_project(tmp_path / "proj").root
    image = write_image(image_dir(root, DATE) / "a.jpg", (WIDTH, HEIGHT))
    bucket = staged_box(root, image, BOX)
    save_labels(root, image, [{"subject": "bud", "bbox": BOX}])
    body = {
        "image_path": str(image),
        "user": "breeder",
        "annotations": _as_loaded(root, image),
        "gestures": {"decide": {"bucket": bucket, "verdicts": {"0": "rejected"}}},
    }

    resp = client.post("/api/annotate/labels", json=body)

    assert resp.status_code == 200, resp.text
    assert len(resp.json()["document"]["annotations"]) == 1
    (decided,) = read_verdicts(verdict_key_of(image_label_key(image), bucket))
    assert (decided.proposal, decided.action) == (0, "rejected")
    served = client.get(
        "/api/annotate/proposals",
        params={"image_path": str(image), "bucket": bucket},
    ).json()
    (proposal,) = served["proposals"]
    assert (proposal["paired"], proposal["decision"]) == (0, "rejected")


def test_accepting_a_paired_proposal_signs_the_tools_annotation_off(
    tmp_path: Path,
) -> None:
    image = write_image(image_dir(tmp_path, DATE) / "a.jpg", (WIDTH, HEIGHT))
    bucket = staged_box(tmp_path, image, BOX)
    save_labels(
        tmp_path, image, [{"subject": "bud", "bbox": BOX}], author="detector"
    )

    save_labels(
        tmp_path,
        image,
        _as_loaded(tmp_path, image),
        gestures=Gestures(decide=Decisions(bucket=bucket, verdicts={0: "accepted"})),
    )

    (after,) = _stored(tmp_path, image).annotations
    assert (after.created_by, after.accepted_by) == (
        "detector",
        "user:breeder",
    )
    assert json_io.authorship_of(after.created_by, after.accepted_by) == (
        "tool_accepted"
    )
    (decided,) = read_verdicts(verdict_key_of(image_label_key(image), bucket))
    assert (decided.proposal, decided.action) == (0, "accepted")


def test_rejecting_a_paired_proposal_signs_nothing_off(tmp_path: Path) -> None:
    image = write_image(image_dir(tmp_path, DATE) / "a.jpg", (WIDTH, HEIGHT))
    bucket = staged_box(tmp_path, image, BOX)
    save_labels(
        tmp_path, image, [{"subject": "bud", "bbox": BOX}], author="detector"
    )

    save_labels(
        tmp_path,
        image,
        _as_loaded(tmp_path, image),
        gestures=Gestures(decide=Decisions(bucket=bucket, verdicts={0: "rejected"})),
    )

    (after,) = _stored(tmp_path, image).annotations
    assert after.accepted_by is None
    assert json_io.authorship_of(after.created_by, after.accepted_by) == "tool"


def test_authorship_counts_only_a_persons_hand_as_standing_behind_a_record() -> (
    None
):
    def of(**provenance: str) -> str:
        return json_io.authorship_of(
            provenance.get("created_by"), provenance.get("accepted_by")
        )

    assert of(created_by="user:breeder") == "person"
    assert of(created_by="detector") == "tool"
    assert (
        of(created_by="detector", accepted_by="user:breeder")
        == "tool_accepted"
    )
    assert of(created_by="detector", accepted_by="other-tool") == "tool"
    assert of() == "unattributed"
    assert of(accepted_by="user:breeder") == "tool_accepted"


# ── an annotation is confirmed as the person's own call ────────────────────


def test_confirming_a_tools_annotation_signs_it_off_without_changing_it(
    tmp_path: Path,
) -> None:
    image = write_image(image_dir(tmp_path, DATE) / "a.jpg", (WIDTH, HEIGHT))
    save_labels(
        tmp_path, image, [{"subject": "bud", "bbox": BOX}], author="detector"
    )
    (before,) = _stored(tmp_path, image).annotations
    assert (
        json_io.authorship_of(before.created_by, before.accepted_by) == "tool"
    )

    save_labels(
        tmp_path,
        image,
        _as_loaded(tmp_path, image),
        gestures=Gestures(confirm=frozenset({0})),
    )

    (after,) = _stored(tmp_path, image).annotations
    assert (after.accepted_by, after.created_by) == (
        "user:breeder",
        "detector",
    )
    assert json_io.authorship_of(after.created_by, after.accepted_by) == (
        "tool_accepted"
    )
    assert after.geometry == before.geometry


def test_confirming_a_position_the_save_writes_nothing_at_refuses_before_any_write(
    tmp_path: Path,
) -> None:
    image = write_image(image_dir(tmp_path, DATE) / "a.jpg", (WIDTH, HEIGHT))

    with pytest.raises(ValueError, match="none at position 1 to confirm"):
        save_labels(
            tmp_path,
            image,
            [{"subject": "bud", "bbox": BOX}],
            gestures=Gestures(confirm=frozenset({1})),
        )

    assert ts.read(image_label_key(image), default=None) is None


def test_the_save_route_confirms_an_annotation_and_answers_the_document_it_wrote(
    tmp_path: Path, client: TestClient
) -> None:
    from tests._web_fixtures import open_new_project

    root = open_new_project(tmp_path / "proj").root
    image = write_image(image_dir(root, DATE) / "a.jpg", (WIDTH, HEIGHT))
    save_labels(
        root, image, [{"subject": "bud", "bbox": BOX}], author="detector"
    )
    body = {
        "image_path": str(image),
        "user": "breeder",
        "annotations": _as_loaded(root, image),
    }

    once = client.post(
        "/api/annotate/labels", json={**body, "gestures": {"confirm": [0]}}
    )
    assert once.status_code == 200, once.text
    assert _stored(root, image).annotations[0].accepted_by == "user:breeder"
    # The save answers the document it wrote with the token that names it, as a load does.
    loaded = client.get(
        "/api/annotate/labels", params={"image_path": str(image)}
    ).json()
    assert once.json()["document"] == loaded
    assert loaded["annotations"][0]["authorship"] == "tool_accepted"


def test_an_accepted_proposal_answers_the_annotation_it_resolved_to(
    tmp_path: Path,
) -> None:
    """An accepted proposal answers the annotation it resolved to.

    An accepted proposal that pairs with a submitted annotation signs that annotation off and
    appends nothing; one that pairs with none is appended. The answer maps each proposal to its
    annotation's index.
    """
    images = image_dir(tmp_path, DATE)
    image = write_image(images / "a.jpg", (WIDTH, HEIGHT))
    bucket = staged_box(tmp_path, image, BOX)
    drawn = {"subject": "leaf", "bbox": [5.0, 5.0, 15.0, 15.0]}
    boxed = {"subject": "bud", "bbox": BOX}

    later = {"subject": "leaf", "bbox": [60.0, 60.0, 70.0, 70.0]}
    _, paired_doc, paired = save_labels(
        tmp_path,
        image,
        [drawn, boxed, later],
        gestures=Gestures(decide=Decisions(bucket=bucket, verdicts={0: "accepted"})),
    )
    assert paired == {0: 1} and len(paired_doc.annotations) == 3

    other = write_image(images / "b.jpg", (WIDTH, HEIGHT))
    other_bucket = staged_box(tmp_path, other, BOX)
    _, appended_doc, appended = save_labels(
        tmp_path,
        other,
        [drawn],
        gestures=Gestures(
            decide=Decisions(bucket=other_bucket, verdicts={0: "accepted"})
        ),
    )
    assert appended == {0: 1} and len(appended_doc.annotations) == 2


def test_the_save_route_answers_the_annotation_an_accepted_proposal_resolved_to(
    tmp_path: Path, client: TestClient
) -> None:
    from tests._web_fixtures import open_new_project

    root = open_new_project(tmp_path / "proj").root
    image = write_image(image_dir(root, DATE) / "a.jpg", (WIDTH, HEIGHT))
    bucket = staged_box(root, image, BOX)
    resp = client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(image),
            "user": "breeder",
            "gestures": {"decide": {"bucket": bucket, "verdicts": {"0": "accepted"}}},
            "annotations": [
                {"subject": "leaf", "bbox": [60.0, 60.0, 70.0, 70.0]}
            ],
        },
    )

    assert resp.status_code == 200, resp.text
    answer = resp.json()
    assert answer["accepted"] == {"0": 1}
    assert len(answer["document"]["annotations"]) == 2


def test_the_save_route_hands_the_save_the_gestures_a_direct_call_states(
    tmp_path: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The route's save reaches ``save_label_document`` with the gestures a direct call builds.

    A save stating every gesture through the route and the same gestures built for a direct
    call are compared as the records each hands the one save.
    """
    import tcip_mcp.dataset_layout as dataset_layout
    from tcip_annotation.flags import FlagRequest, ProposalTarget
    from tests._web_fixtures import open_new_project

    root = open_new_project(tmp_path / "proj").root
    image = write_image(image_dir(root, DATE) / "a.jpg", (WIDTH, HEIGHT))
    bucket = staged_box(root, image, BOX)
    raised = client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(image),
            "user": "breeder",
            "annotations": [{"subject": "bud", "bbox": BOX}],
            "gestures": {
                "flag": [
                    {
                        "text": "look",
                        "target": {"point": [50.0, 40.0], "subject": "bud"},
                    }
                ]
            },
        },
    )
    assert raised.status_code == 200, raised.text
    (flag,) = raised.json()["document"]["flags"]

    handed: list[Gestures] = []
    real = dataset_layout.save_label_document

    def spied(*args: object, gestures: Gestures, **kwargs: object) -> object:
        handed.append(gestures)
        return real(*args, gestures=gestures, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(dataset_layout, "save_label_document", spied)
    resp = client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(image),
            "user": "breeder",
            "annotations": raised.json()["document"]["annotations"],
            "gestures": {
                "decide": {"bucket": bucket, "verdicts": {"0": "accepted"}},
                "confirm": [0],
                "complete": {"bud": True},
                "rect": [0.0, 0.0, 100.0, 80.0],
                "proposals_hidden": True,
                "flag": [
                    {"text": "again", "target": {"bucket": bucket, "index": 0}}
                ],
                "resolve": {flag["id"]: "seen"},
            },
        },
    )

    assert resp.status_code == 200, resp.text
    assert handed == [
        Gestures(
            decide=Decisions(bucket=bucket, verdicts={0: "accepted"}),
            confirm=frozenset({0}),
            complete={"bud": True},
            rect=(0.0, 0.0, 100.0, 80.0),
            proposals_hidden=True,
            flag=(
                FlagRequest(
                    text="again", target=ProposalTarget(bucket=bucket, index=0)
                ),
            ),
            resolve={flag["id"]: "seen"},
        )
    ]


def test_a_save_stating_no_annotations_answers_422_and_writes_nothing(
    tmp_path: Path, client: TestClient
) -> None:
    """A save body naming no ``annotations`` is refused, the stored document as it was."""
    from tests._web_fixtures import open_new_project

    root = open_new_project(tmp_path / "proj").root
    image = write_image(image_dir(root, DATE) / "a.jpg", (WIDTH, HEIGHT))
    body = {"image_path": str(image), "user": "breeder"}
    saved = client.post(
        "/api/annotate/labels",
        json={
            **body,
            "annotations": [{"subject": "bud", "point": [1.0, 2.0]}],
        },
    )
    assert saved.status_code == 200, saved.text

    refused = client.post("/api/annotate/labels", json=body)

    assert refused.status_code == 422, refused.text
    loaded = client.get(
        "/api/annotate/labels", params={"image_path": str(image)}
    ).json()
    assert loaded["base_mtime"] == saved.json()["document"]["base_mtime"]
    assert [a["point"] for a in loaded["annotations"]] == [[1.0, 2.0]]


def test_the_save_answers_the_document_it_stored(
    tmp_path: Path, client: TestClient
) -> None:
    """The save answers the document it stored, the same answer a load gives.

    After an edit to a completed subject, the save answers the point rounded to two decimals
    and the subject's completion as partial, matching what a load then answers.
    """
    from tests._web_fixtures import open_new_project

    root = open_new_project(tmp_path / "proj").root
    image = write_image(image_dir(root, DATE) / "a.jpg", (WIDTH, HEIGHT))
    body = {"image_path": str(image), "user": "breeder"}
    first = client.post(
        "/api/annotate/labels",
        json={
            **body,
            "annotations": [{"subject": "bud", "point": [1.0, 2.0]}],
            "gestures": {"complete": {"bud": True}},
        },
    )
    assert first.json()["document"]["completion"] == {"bud": "complete"}

    edited = client.post(
        "/api/annotate/labels",
        json={
            **body,
            "annotations": [{"subject": "bud", "point": [1.23456, 2.34567]}],
        },
    )

    loaded = client.get(
        "/api/annotate/labels", params={"image_path": str(image)}
    ).json()
    assert edited.json()["document"] == loaded
    assert loaded["completion"] == {"bud": "partial"}
    assert loaded["annotations"][0]["point"] == [1.23, 2.35]


# ── a save from a stale read never lands ───────────────────────────────────


def test_a_save_from_a_stale_read_conflicts_and_writes_nothing(
    tmp_path: Path,
) -> None:
    image = write_image(image_dir(tmp_path, DATE) / "a.jpg", (WIDTH, HEIGHT))
    first, _document, _accepted = save_labels(
        tmp_path, image, [{"subject": "bud", "bbox": BOX}]
    )
    save_labels(tmp_path, image, [], author="user:second", expect=first)

    with pytest.raises(ts.VersionConflictError):
        save_labels(
            tmp_path,
            image,
            [{"subject": "bud", "bbox": [1.0, 1.0, 9.0, 9.0]}],
            author="user:third",
            expect=first,
        )

    assert _stored(tmp_path, image).annotations == []


def test_a_bucket_document_that_will_not_read_answers_400_at_the_proposals_route(
    tmp_path: Path, client: TestClient
) -> None:
    from tcip_mcp.buckets import read_bucket
    from tests._web_fixtures import open_new_project

    root = open_new_project(tmp_path / "proj").root
    image = write_image(image_dir(root, DATE) / "a.jpg", (WIDTH, HEIGHT))
    bucket = staged_box(root, image, BOX)
    params = {"image_path": str(image), "bucket": bucket}
    assert (
        client.get("/api/annotate/proposals", params=params).status_code == 200
    )

    document = read_bucket(root, bucket).document_key(image.stem)
    assert document is not None
    ts.replace(
        document,
        {
            "annotations": [
                {"subject": "bud", "bbox": [5, 5, 0, 5], "score": 0.9}
            ]
        },
    )

    resp = client.get("/api/annotate/proposals", params=params)
    assert resp.status_code == 400, resp.text
    assert "record 0" in resp.text


def test_the_queue_refuses_a_label_document_that_will_not_read_by_name(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.feedback_tools import _prepare_queue_sources

    images = image_dir(tmp_path, DATE)
    image = write_image(images / "a.jpg", (WIDTH, HEIGHT))
    save_labels(tmp_path, image, [], gestures=Gestures(complete={"bud": True}))
    queued = _prepare_queue_sources(str(images), "bud")
    assert not isinstance(queued, dict), queued
    assert queued[1] == 1

    ts.replace(image_label_key(image), {"annotations": {}})

    refused = _prepare_queue_sources(str(images), "bud")
    assert isinstance(refused, dict) and "annotations" in refused["error"]


def test_the_proposals_payload_carries_what_the_editor_reads_and_no_more(
    tmp_path: Path, client: TestClient
) -> None:
    from tests._web_fixtures import open_new_project

    root = open_new_project(tmp_path / "proj").root
    image = write_image(image_dir(root, DATE) / "a.jpg", (WIDTH, HEIGHT))
    bucket = staged_box(root, image, BOX)
    save_labels(root, image, [], gestures=Gestures(complete={"bud": True}))

    proposals = client.get(
        "/api/annotate/proposals",
        params={"image_path": str(image), "bucket": bucket},
    ).json()
    loaded = client.get(
        "/api/annotate/labels", params={"image_path": str(image)}
    ).json()

    assert set(proposals) == {"operating_point", "proposals"}
    assert set(proposals["operating_point"]) == {"conf", "reason"}
    assert set(proposals["proposals"][0]) >= {"index", "paired", "decision"}
    assert "admitted" not in proposals["proposals"][0]
    assert loaded["completion"] == {"bud": "negative"}


def test_a_staged_bucket_serves_no_operating_point_and_says_why(
    tmp_path: Path, client: TestClient
) -> None:
    """A staged bucket serves no operating point and says why.

    A bucket staged under no assessment serves a null conf and a reason beside its proposals.
    """
    from tests._web_fixtures import open_new_project

    root = open_new_project(tmp_path / "proj").root
    image = write_image(image_dir(root, DATE) / "a.jpg", (WIDTH, HEIGHT))
    bucket = staged_box(root, image, BOX)

    served = client.get(
        "/api/annotate/proposals",
        params={"image_path": str(image), "bucket": bucket},
    ).json()

    assert served["operating_point"]["conf"] is None
    assert served["operating_point"]["reason"]


def test_the_loaded_annotations_carry_the_index_a_pairing_names(
    tmp_path: Path, client: TestClient
) -> None:
    """The loaded annotations carry the document index a proposal's pairing names.

    Each loaded annotation states its document index, and a proposal's ``paired`` names the
    index of the loaded annotation it pairs with.
    """
    from tests._web_fixtures import open_new_project

    root = open_new_project(tmp_path / "proj").root
    image = write_image(image_dir(root, DATE) / "a.jpg", (WIDTH, HEIGHT))
    bucket = staged_box(root, image, BOX)
    save_labels(
        root,
        image,
        [{"subject": "leaf", "points": RING}, {"subject": "bud", "bbox": BOX}],
    )

    loaded = client.get(
        "/api/annotate/labels", params={"image_path": str(image)}
    ).json()
    served = client.get(
        "/api/annotate/proposals",
        params={"image_path": str(image), "bucket": bucket},
    ).json()

    assert [a["index"] for a in loaded["annotations"]] == [0, 1]
    (proposal,) = served["proposals"]
    assert proposal["paired"] == 1
    assert loaded["annotations"][proposal["paired"]]["subject"] == "bud"


# ── completion marks ───────────────────────────────────────────────────────


def test_an_edit_invalidates_the_marks_of_its_own_subject_only(
    tmp_path: Path,
) -> None:
    image = write_image(image_dir(tmp_path, DATE) / "a.jpg", (WIDTH, HEIGHT))
    save_labels(
        tmp_path,
        image,
        [{"subject": "bud", "bbox": BOX}],
        gestures=Gestures(complete={"bud": True, "leaf": True}),
    )
    assert _stored(tmp_path, image).state("bud") == "complete"
    assert _stored(tmp_path, image).state("leaf") == "negative"

    save_labels(
        tmp_path,
        image,
        [
            *_as_loaded(tmp_path, image),
            {"subject": "bud", "bbox": [5.0, 5.0, 15.0, 15.0]},
        ],
    )
    assert _stored(tmp_path, image).state("bud") == "partial"
    assert _stored(tmp_path, image).state("leaf") == "negative"

    save_labels(
        tmp_path,
        image,
        [
            *_as_loaded(tmp_path, image),
            {"subject": "leaf", "bbox": [70.0, 5.0, 90.0, 15.0]},
        ],
    )
    assert _stored(tmp_path, image).state("leaf") == "partial"


def test_a_mark_made_with_proposals_hidden_says_so(tmp_path: Path) -> None:
    image = write_image(image_dir(tmp_path, DATE) / "a.jpg", (WIDTH, HEIGHT))

    save_labels(
        tmp_path,
        image,
        [],
        gestures=Gestures(complete={"bud": True}, proposals_hidden=True),
    )
    save_labels(
        tmp_path, image, [], gestures=Gestures(complete={"leaf": True})
    )

    marks = _stored(tmp_path, image).marks
    assert [m.proposals_hidden for m in marks["bud"]] == [True]
    assert [m.proposals_hidden for m in marks["leaf"]] == [False]


def test_a_negative_is_an_empty_subject_and_a_mark_at_every_reader(
    tmp_path: Path, client: TestClient
) -> None:
    """Three readers see an empty document as a negative when marked and not while unmarked.

    Admission, the review queue's source preparation and the editor's listing each read an
    empty unmarked document as no negative, and the same document marked complete as one; the
    listing answers the registry's subject the document does not name as unannotated.
    """
    from tcip_mcp.pipelines.data.label_queries import admit, registry_scope
    from tcip_mcp.tools.feedback_tools import _prepare_queue_sources
    from tests._producer_fixtures import registry_of, registry_over
    from tests._web_fixtures import open_new_project

    root = open_new_project(tmp_path / "proj").root
    registry_over(root, registry_of("bud"))
    images = image_dir(root, DATE)
    image = write_image(images / "a.jpg", (WIDTH, HEIGHT))
    save_labels(root, image, [])

    def readers() -> tuple[int, int, str]:
        counts = admit(
            images,
            members=[image.stem],
            scope=registry_scope(images, "bud"),
        ).tallies
        queued = _prepare_queue_sources(str(images), "bud")
        assert not isinstance(queued, dict), queued
        listed = client.get(
            "/api/annotate/labels", params={"image_path": str(image)}
        ).json()
        return (
            counts.get("negative", 0),
            queued[1],
            listed["completion"]["bud"],
        )

    assert readers() == (0, 0, "unannotated")

    save_labels(root, image, [], gestures=Gestures(complete={"bud": True}))

    assert readers() == (1, 1, "negative")


def test_saved_provenance_is_the_requests_actor_whatever_the_browser_sends(
    tmp_path: Path, client: TestClient
) -> None:
    from tests._web_fixtures import open_new_project

    root = open_new_project(tmp_path / "proj").root
    image = write_image(image_dir(root, DATE) / "a.jpg", (WIDTH, HEIGHT))

    resp = client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(image),
            "user": "breeder",
            "annotations": [
                {
                    "subject": "bud",
                    "bbox": BOX,
                    "created_by": "user:mallory",
                    "created_at": "2020-01-01T00:00:00+00:00",
                    "accepted_by": "user:mallory",
                }
            ],
        },
    )
    assert resp.status_code == 200, resp.text

    (saved,) = _stored(root, image).annotations
    assert saved.created_by == "user:breeder"
    assert saved.created_at != "2020-01-01T00:00:00+00:00"
    assert saved.accepted_by is None


# ── one matcher answers the assessment and the editor ──────────────────────


def _counted(annotations: list, proposals: list, criterion: dict) -> dict:
    """Return the assessment's governing count over one image's annotations and proposals."""
    from tcip_annotation.json_io import xywh
    from tcip_mcp.pipelines.training.evaluation import (
        dt_record,
        governing_counts,
        gt_record,
        prediction_record,
    )

    def box(a: Annotation) -> list[float]:
        g = a.geometry
        return xywh(g.x1, g.y1, g.x2, g.y2)

    record = prediction_record(
        [dt_record(box(p), 1, p.score) for p in proposals],
        [gt_record(box(a), 1, a.iscrowd) for a in annotations],
        width=WIDTH,
        height=HEIGHT,
        cap=None,
        count=len(proposals),
    )
    return governing_counts([record], criterion, conf_threshold=0.0)


def test_the_assessment_count_and_the_editor_pairing_agree_under_a_crowd_region(
    tmp_path: Path,
) -> None:
    """The assessment count and the editor's pairing agree under a crowd region.

    The assessment's governing count and the editor's proposal pairing, each over the same
    annotations and a bucket published under no assessment, a crowd region among them, agree on
    what pairs.
    """
    from tcip_mcp.dataset_layout import image_proposals, proposal_pairs
    from tcip_mcp.pipelines.schemas import COMPARABILITY_IOU
    from tcip_mcp.pipelines.training.evaluation import comparability_criterion
    from tcip_mcp.tools.proposal_tools import stage_proposals

    image = write_image(image_dir(tmp_path, DATE) / "a.jpg", (WIDTH, HEIGHT))
    save_labels(
        tmp_path,
        image,
        [
            {"subject": "bud", "bbox": [10.0, 10.0, 30.0, 30.0]},
            {
                "subject": "bud",
                "bbox": [50.0, 10.0, 90.0, 70.0],
                "iscrowd": True,
            },
        ],
    )

    def norm(x1: float, y1: float, x2: float, y2: float, conf: float) -> dict:
        return {
            "subject": "bud",
            "conf": conf,
            "cx": (x1 + x2) / 2 / WIDTH,
            "cy": (y1 + y2) / 2 / HEIGHT,
            "w": (x2 - x1) / WIDTH,
            "h": (y2 - y1) / HEIGHT,
        }

    staged = stage_proposals(
        tmp_path,
        str(image),
        model_name="detector",
        actor=None,
        boxes=[
            norm(11, 11, 31, 31, 0.9),
            norm(60, 20, 70, 30, 0.8),
            norm(5, 60, 15, 75, 0.7),
        ],
    )
    assert "error" not in staged, staged
    bucket, proposals = image_proposals(staged.bucket, image_label_key(image))
    annotations = _stored(tmp_path, image).annotations

    counted = _counted(
        annotations,
        proposals,
        comparability_criterion(COMPARABILITY_IOU, by_mask=False),
    )
    paired = proposal_pairs(tmp_path, bucket, annotations, proposals)

    assert counted["tp"] == len(paired) == 1
    assert (
        counted["fp"] == 1
    )  # the proposal centered in the crowd region is neither
    (pair,) = paired.items()
    assert annotations[pair[1]].iscrowd is False


def test_the_editor_pairs_a_bucket_under_its_assessments_center_match(
    tmp_path: Path,
) -> None:
    """The editor pairs a bucket under its assessment's center match.

    A bucket published under a center-match assessment pairs in the editor as that
    assessment counts: an annotation shifted off its proposal by under the assessment's
    tolerance, but by more than the comparability IoU admits, pairs and counts alike.
    """
    pytest.importorskip("torch")
    from tcip_mcp.assessment import read_assessment
    from tcip_mcp.dataset_layout import image_proposals, proposal_pairs
    from tcip_mcp.pipelines.schemas import COMPARABILITY_IOU
    from tcip_mcp.pipelines.training.evaluation import comparability_criterion
    from tests._chain_fixtures import IMG, run_the_chain

    chain = run_the_chain(tmp_path, experiment_id="exp-editor-pairing")
    bucket = chain.read()
    image = next(
        p
        for p in sorted(chain.images_dir.iterdir())
        if bucket.document_key(p.stem) is not None
    )
    _bucket_record, proposals = image_proposals(
        bucket.name, image_label_key(image)
    )
    top = max(
        (p for p in proposals if p.score is not None), key=lambda p: p.score
    )
    g = top.geometry
    size = g.x2 - g.x1
    shifted = {
        "subject": top.subject,
        "bbox": [
            g.x1 + 0.2 * size,
            g.y1 + 0.2 * size,
            g.x2 + 0.2 * size,
            g.y2 + 0.2 * size,
        ],
    }
    save_label_document(
        tmp_path,
        image_label_key(image),
        [shifted],
        width=IMG,
        height=IMG,
        author="user:breeder",
        actor="user:breeder",
    )
    annotations = json_io.read_label_document(
        image_label_key(image)
    ).annotations
    count = read_assessment(tmp_path, bucket.assessment_id).criterion["count"]
    assert isinstance(count, dict)
    criterion = count["localization"]
    assert isinstance(criterion, dict)

    paired = proposal_pairs(tmp_path, bucket, annotations, proposals)

    assert criterion["kind"] == "center_match"
    assert (
        len(paired) == _counted(annotations, proposals, criterion)["tp"] == 1
    )
    convention = comparability_criterion(COMPARABILITY_IOU, by_mask=False)
    assert _counted(annotations, proposals, convention)["tp"] == 0


# ── the proposer is named ──────────────────────────────────────────────────


class DottedProposer:
    """A proposer the agent brings by dotted path, returning one fixed candidate for any image."""

    def propose(self, image_path: str, **params: object) -> Proposal:
        return Proposal(
            [
                {
                    "candidate_id": 1,
                    "bbox": [10.0, 10.0, 20.0, 20.0],
                    "area": 400.0,
                    "rings": [
                        [
                            (10.0, 10.0),
                            (30.0, 10.0),
                            (30.0, 30.0),
                            (10.0, 30.0),
                        ]
                    ],
                    "score": 0.9,
                }
            ],
            None,
        )


def test_propose_annotations_refuses_an_unnamed_proposer_and_runs_a_dotted_one(
    tmp_path: Path,
) -> None:
    from tcip_mcp.pipelines.proposal import register_proposal_engine
    from tcip_mcp.tools.proposal_tools import propose_annotations

    register_proposal_engine("registered_here", DottedProposer())
    image = write_image(image_dir(tmp_path, DATE) / "a.jpg", (WIDTH, HEIGHT))

    refused = propose_annotations(tmp_path, str(image), engine="", actor=None)
    assert "registered_here" in refused["error"]

    proposed = propose_annotations(
        tmp_path,
        str(image),
        engine="tests.test_label_document_gestures:DottedProposer",
        actor=None,
    )
    assert proposed.staging_id is not None and len(proposed.candidates) == 1
