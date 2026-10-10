"""The editor's gestures through the one label save: what each writes into the label document,
the verdict shard and the completion marks, and what the admission, the queue and the editor's
own listing read back from it."""

from __future__ import annotations

from pathlib import Path

import pytest

import tcip_store as ts
from tcip_annotation import json_io
from tcip_annotation.verdicts import Verdict, read_verdicts
from tcip_mcp.dataset_layout import Gestures, image_dir, save_label_document, verdict_key_of
from tests._producer_fixtures import image_label_key, write_image

ANSWER_ONLY = {"status", "accepted"}
"""What a save answers beside the document a load answers."""
DATE = "2026-02-11"
WIDTH, HEIGHT = 100, 80
BOX = [40.0, 30.0, 60.0, 50.0]
RING = [[10.0, 10.0], [30.0, 10.0], [30.0, 25.0], [10.0, 25.0]]


def _image(root: Path, stem: str = "a") -> Path:
    return write_image(image_dir(root, DATE) / f"{stem}.jpg", (WIDTH, HEIGHT), (90, 110, 70))


def _save(root: Path, image: Path, payloads: list[dict], *, author: str = "user:breeder",
          expect: ts.Version | None = None, gestures: Gestures = Gestures()):
    return save_label_document(root, image_label_key(image), payloads, width=WIDTH,
                               height=HEIGHT, author=author, actor=author, expect=expect,
                               gestures=gestures)


def _stored(root: Path, image: Path) -> json_io.LabelDocument:
    return json_io.read_label_document(image_label_key(image))


def _as_loaded(root: Path, image: Path) -> list[dict]:
    """The document's annotations as the editor loads them, the payloads it saves back."""
    return [json_io.client_annotation(a) for a in _stored(root, image).annotations]


def _bucket(root: Path, image: Path) -> str:
    """A proposal bucket for ``image`` holding one ``bud`` box over :data:`BOX`, staged through
    the platform's own door; its name."""
    from tcip_mcp.tools.proposal_tools import stage_proposals

    x1, y1, x2, y2 = BOX
    staged = stage_proposals(root, str(image), model_name="detector", boxes=[{
        "subject": "bud", "conf": 0.9, "cx": (x1 + x2) / 2 / WIDTH, "cy": (y1 + y2) / 2 / HEIGHT,
        "w": (x2 - x1) / WIDTH, "h": (y2 - y1) / HEIGHT}])
    assert "error" not in staged, staged
    return staged["bucket"]


# ── corrections keep what they do not correct ──────────────────────────────


def test_correcting_a_value_keeps_the_geometry(tmp_path: Path) -> None:
    image = _image(tmp_path)
    drawn = {"subject": "bud", "rings": [RING], "attributes": {"stage": "open"}}
    _save(tmp_path, image, [drawn], author="user:first")

    (loaded,) = _as_loaded(tmp_path, image)
    _save(tmp_path, image, [{**loaded, "attributes": {"stage": "closed"}}], author="user:second")

    (after,) = _stored(tmp_path, image).annotations
    assert after.geometry == json_io.annotation_from_payload(drawn).geometry
    assert after.attributes == {"stage": "closed"}
    assert after.created_by == "user:second"


def test_correcting_the_geometry_keeps_the_confirmed_value(tmp_path: Path) -> None:
    image = _image(tmp_path)
    _save(tmp_path, image, [{"subject": "bud", "bbox": BOX, "attributes": {"stage": "open"}}])

    (loaded,) = _as_loaded(tmp_path, image)
    _save(tmp_path, image, [{**loaded, "bbox": [41.0, 31.0, 61.0, 51.0]}])

    (after,) = _stored(tmp_path, image).annotations
    assert after.attributes == {"stage": "open"}
    assert (after.geometry.x1, after.geometry.y2) == (41.0, 51.0)


def test_adding_a_missed_object_writes_it_with_its_value(tmp_path: Path) -> None:
    image = _image(tmp_path)
    _save(tmp_path, image, [{"subject": "bud", "bbox": BOX}])

    _save(tmp_path, image, [*_as_loaded(tmp_path, image),
                            {"subject": "bud", "bbox": [5.0, 5.0, 15.0, 15.0],
                             "attributes": {"stage": "open"}}], author="user:second")

    added = _stored(tmp_path, image).annotations[1]
    assert added.attributes == {"stage": "open"}
    assert added.created_by == "user:second"


# ── a proposal is accepted or rejected, once, into the one decision log ────


def test_accepting_a_paired_proposal_confirms_the_annotation_without_a_second_one(
        tmp_path: Path) -> None:
    image = _image(tmp_path)
    bucket = _bucket(tmp_path, image)
    _save(tmp_path, image, [{"subject": "bud", "bbox": BOX}])

    _save(tmp_path, image, _as_loaded(tmp_path, image),
          gestures=Gestures(bucket=bucket, accept=frozenset({0})))

    assert len(_stored(tmp_path, image).annotations) == 1


def test_accepting_an_unpaired_proposal_adds_it_authored_by_its_producer(tmp_path: Path) -> None:
    image = _image(tmp_path)
    bucket = _bucket(tmp_path, image)

    _save(tmp_path, image, [], gestures=Gestures(bucket=bucket, accept=frozenset({0})))

    (accepted,) = _stored(tmp_path, image).annotations
    assert (accepted.created_by, accepted.accepted_by) == ("detector", "user:breeder")
    assert accepted.score is None


def test_accepting_a_model_proposal_carries_no_attribute_value_into_the_document(
        tmp_path: Path) -> None:
    """An accepted proposal pairing no annotation joins ground truth with its geometry and subject
    alone: every attribute the model called is left unassessed for the person to fill in."""
    pytest.importorskip("torch")
    from tcip_mcp import subject_registry as cr
    from tests._chain_fixtures import predicted, published

    image = _image(tmp_path)
    color = cr.Attribute("color", "categorical", ("red", "blue"))
    registry = cr.SubjectRegistry(subjects=(cr.Subject(name="bud", attributes=(color,)),))
    result = {**predicted(image, ["blue"], (color,)), "width": WIDTH, "height": HEIGHT}
    bucket = published(tmp_path, f"model/{DATE}", [result], scope={"subject": "bud"},
                       registry=registry)
    (proposal,) = json_io.read_predictions(bucket.document_key(image.stem)).annotations
    assert proposal.attributes == {"color": "blue"}

    _save(tmp_path, image, [], gestures=Gestures(bucket=bucket.name, accept=frozenset({0})))

    (accepted,) = _stored(tmp_path, image).annotations
    assert (accepted.subject, accepted.attributes) == ("bud", {})
    assert accepted.geometry == proposal.geometry


def test_the_shard_records_each_decision_once_and_refuses_an_entry_it_cannot_read(
        tmp_path: Path) -> None:
    image = _image(tmp_path)
    bucket = _bucket(tmp_path, image)

    _save(tmp_path, image, [], gestures=Gestures(bucket=bucket, accept=frozenset({0})))

    key = verdict_key_of(image_label_key(image), bucket)
    (decided,) = read_verdicts(key)
    assert decided == Verdict(proposal=0, action="accepted", by="user:breeder", at=decided.at)

    ts.append(key, {"proposal": 0, "action": "maybe", "by": "user:breeder", "at": decided.at})
    with pytest.raises(ValueError, match="maybe"):
        read_verdicts(key)


def test_a_decision_or_a_completion_mark_by_no_person_does_not_read(tmp_path: Path) -> None:
    """A verdict's and a completion mark's ``by`` name a person; one naming a producer that is no
    person refuses as the record it is read from."""
    from dataclasses import replace

    from tcip_annotation.verdicts import decode_verdict, encode_verdict

    image = _image(tmp_path)
    bucket = _bucket(tmp_path, image)
    _save(tmp_path, image, [], gestures=Gestures(bucket=bucket, accept=frozenset({0}),
                                                 complete={"bud": True}))

    (decided,) = read_verdicts(verdict_key_of(image_label_key(image), bucket))
    with pytest.raises(ValueError, match="a person"):
        decode_verdict(encode_verdict(replace(decided, by="save_annotations")))
    stored = ts.read(image_label_key(image))
    (mark,) = stored[json_io.COMPLETION_KEY]["bud"]
    assert _stored(tmp_path, image).marks["bud"][0].by == mark["by"]
    ts.replace(image_label_key(image), {**stored, json_io.COMPLETION_KEY: {
        "bud": [{**mark, "by": "save_annotations"}]}})
    with pytest.raises(json_io.UnreadableLabelDocumentError, match="recorded identity"):
        _stored(tmp_path, image)


def test_rejecting_a_paired_proposal_records_the_rejection_and_keeps_the_annotation(
        tmp_path: Path, client) -> None:
    from tests._web_fixtures import open_new_project

    root = open_new_project(tmp_path / "proj").root
    image = _image(root)
    bucket = _bucket(root, image)
    _save(root, image, [{"subject": "bud", "bbox": BOX}])
    body = {"image_path": str(image), "user": "breeder", "annotations": _as_loaded(root, image),
            "bucket": bucket, "reject": [0]}

    resp = client.post("/api/annotate/labels", json=body)

    assert resp.status_code == 200, resp.text
    assert len(resp.json()["annotations"]) == 1
    (decided,) = read_verdicts(verdict_key_of(image_label_key(image), bucket))
    assert (decided.proposal, decided.action) == (0, "rejected")
    served = client.get("/api/annotate/proposals", params={
        "image_path": str(image), "bucket": bucket}).json()
    (proposal,) = served["proposals"]
    assert (proposal["paired"], proposal["decision"]) == (0, "rejected")


def test_accepting_a_paired_proposal_signs_the_tools_annotation_off(tmp_path: Path) -> None:
    image = _image(tmp_path)
    bucket = _bucket(tmp_path, image)
    _save(tmp_path, image, [{"subject": "bud", "bbox": BOX}], author="detector")

    _save(tmp_path, image, _as_loaded(tmp_path, image),
          gestures=Gestures(bucket=bucket, accept=frozenset({0})))

    (after,) = _stored(tmp_path, image).annotations
    assert (after.created_by, after.accepted_by) == ("detector", "user:breeder")
    assert json_io.authorship_of(after) == "tool_accepted"
    (decided,) = read_verdicts(verdict_key_of(image_label_key(image), bucket))
    assert (decided.proposal, decided.action) == (0, "accepted")


def test_rejecting_a_paired_proposal_signs_nothing_off(tmp_path: Path) -> None:
    image = _image(tmp_path)
    bucket = _bucket(tmp_path, image)
    _save(tmp_path, image, [{"subject": "bud", "bbox": BOX}], author="detector")

    _save(tmp_path, image, _as_loaded(tmp_path, image),
          gestures=Gestures(bucket=bucket, reject=frozenset({0})))

    (after,) = _stored(tmp_path, image).annotations
    assert after.accepted_by is None and json_io.authorship_of(after) == "tool"


def test_authorship_counts_only_a_persons_hand_as_standing_behind_a_record() -> None:
    from tcip_annotation.state import Annotation

    def of(**provenance) -> str:
        return json_io.authorship_of(Annotation("bud", **provenance))

    assert of(created_by="user:breeder") == "person"
    assert of(created_by="detector") == "tool"
    assert of(created_by="detector", accepted_by="user:breeder") == "tool_accepted"
    assert of(created_by="detector", accepted_by="other-tool") == "tool"
    assert of() == "unattributed"
    assert of(accepted_by="user:breeder") == "tool_accepted"


# ── an annotation is confirmed as the person's own call ────────────────────


def test_confirming_a_tools_annotation_signs_it_off_without_changing_it(tmp_path: Path) -> None:
    image = _image(tmp_path)
    _save(tmp_path, image, [{"subject": "bud", "bbox": BOX}], author="detector")
    (before,) = _stored(tmp_path, image).annotations
    assert json_io.authorship_of(before) == "tool"

    _save(tmp_path, image, _as_loaded(tmp_path, image), gestures=Gestures(confirm=frozenset({0})))

    (after,) = _stored(tmp_path, image).annotations
    assert (after.accepted_by, after.created_by) == ("user:breeder", "detector")
    assert json_io.authorship_of(after) == "tool_accepted"
    assert after.geometry == before.geometry


def test_confirming_a_position_the_save_writes_nothing_at_refuses_before_any_write(
        tmp_path: Path) -> None:
    image = _image(tmp_path)

    with pytest.raises(ValueError, match="none at position 1 to confirm"):
        _save(tmp_path, image, [{"subject": "bud", "bbox": BOX}],
              gestures=Gestures(confirm=frozenset({1})))

    assert ts.read(image_label_key(image), default=None) is None


def test_the_save_route_confirms_an_annotation_once_and_refuses_it_named_twice(
        tmp_path: Path, client) -> None:
    from tests._web_fixtures import open_new_project

    root = open_new_project(tmp_path / "proj").root
    image = _image(root)
    _save(root, image, [{"subject": "bud", "bbox": BOX}], author="detector")
    body = {"image_path": str(image), "user": "breeder", "annotations": _as_loaded(root, image)}

    twice = client.post("/api/annotate/labels", json={**body, "confirm": [0, 0]})
    assert twice.status_code == 400 and "more than once" in twice.text
    assert _stored(root, image).annotations[0].accepted_by is None

    once = client.post("/api/annotate/labels", json={**body, "confirm": [0]})
    assert once.status_code == 200, once.text
    assert _stored(root, image).annotations[0].accepted_by == "user:breeder"
    # The save answers the document it wrote with the token that names it, as a load does.
    loaded = client.get("/api/annotate/labels", params={"image_path": str(image)}).json()
    assert {k: v for k, v in once.json().items() if k not in ANSWER_ONLY} == loaded
    assert loaded["annotations"][0]["authorship"] == "tool_accepted"


def test_an_accepted_proposal_answers_the_annotation_it_resolved_to(tmp_path: Path) -> None:
    """An accepted proposal that pairs with a submitted annotation signs that annotation off and
    appends nothing; one that pairs with none is appended. The answer names each one's annotation,
    whatever else the save carries."""
    image = _image(tmp_path)
    bucket = _bucket(tmp_path, image)
    drawn = {"subject": "leaf", "bbox": [5.0, 5.0, 15.0, 15.0]}
    boxed = {"subject": "bud", "bbox": BOX}

    later = {"subject": "leaf", "bbox": [60.0, 60.0, 70.0, 70.0]}
    _, paired_doc, paired = _save(tmp_path, image, [drawn, boxed, later],
                                  gestures=Gestures(bucket=bucket, accept=frozenset({0})))
    assert paired == {0: 1} and len(paired_doc.annotations) == 3

    other = _image(tmp_path, "b")
    other_bucket = _bucket(tmp_path, other)
    _, appended_doc, appended = _save(tmp_path, other, [drawn],
                                      gestures=Gestures(bucket=other_bucket,
                                                        accept=frozenset({0})))
    assert appended == {0: 1} and len(appended_doc.annotations) == 2


def test_the_save_route_answers_the_annotation_an_accepted_proposal_resolved_to(
        tmp_path: Path, client) -> None:
    from tests._web_fixtures import open_new_project

    root = open_new_project(tmp_path / "proj").root
    image = _image(root)
    bucket = _bucket(root, image)
    resp = client.post("/api/annotate/labels", json={
        "image_path": str(image), "user": "breeder", "bucket": bucket, "accept": [0],
        "annotations": [{"subject": "leaf", "bbox": [60.0, 60.0, 70.0, 70.0]}]})

    assert resp.status_code == 200, resp.text
    answer = resp.json()
    assert answer["accepted"] == {"0": 1}
    assert len(answer["annotations"]) == 2


def test_the_save_answers_the_document_it_stored(tmp_path: Path, client) -> None:
    """What the save writes is what it answers: rounded geometry and only the completion marks
    that still name the subject's annotations, the same answer a load gives."""
    from tests._web_fixtures import open_new_project

    root = open_new_project(tmp_path / "proj").root
    image = _image(root)
    body = {"image_path": str(image), "user": "breeder"}
    first = client.post("/api/annotate/labels", json={
        **body, "annotations": [{"subject": "bud", "point": [1.0, 2.0]}],
        "complete": {"bud": True}})
    assert first.json()["completion"] == {"bud": "complete"}

    edited = client.post("/api/annotate/labels", json={
        **body, "annotations": [{"subject": "bud", "point": [1.23456, 2.34567]}]})

    loaded = client.get("/api/annotate/labels", params={"image_path": str(image)}).json()
    assert {k: v for k, v in edited.json().items() if k not in ANSWER_ONLY} == loaded
    assert loaded["completion"] == {"bud": "partial"}
    assert loaded["annotations"][0]["point"] == [1.23, 2.35]


# ── a save from a stale read never lands ───────────────────────────────────


def test_a_save_from_a_stale_read_conflicts_and_writes_nothing(tmp_path: Path) -> None:
    image = _image(tmp_path)
    first, _document, _accepted = _save(tmp_path, image, [{"subject": "bud", "bbox": BOX}])
    _save(tmp_path, image, [], author="user:second", expect=first)

    with pytest.raises(ts.VersionConflictError):
        _save(tmp_path, image, [{"subject": "bud", "bbox": [1.0, 1.0, 9.0, 9.0]}],
              author="user:third", expect=first)

    assert _stored(tmp_path, image).annotations == []


def _nothing_written(image: Path, bucket: str) -> bool:
    key = verdict_key_of(image_label_key(image), bucket)
    return (ts.read(image_label_key(image), default=None) is None
            and not ts.keys(key.store, key.root))


def test_a_proposal_both_accepted_and_rejected_refuses_before_any_write(tmp_path: Path) -> None:
    image = _image(tmp_path)
    bucket = _bucket(tmp_path, image)

    with pytest.raises(ValueError, match=r"\[0\] are both accepted and rejected"):
        _save(tmp_path, image, [], gestures=Gestures(
            bucket=bucket, accept=frozenset({0}), reject=frozenset({0})))

    assert _nothing_written(image, bucket)


def test_a_proposal_named_twice_refuses_at_the_save_route_before_any_write(
        tmp_path: Path, client) -> None:
    from tests._web_fixtures import open_new_project

    root = open_new_project(tmp_path / "proj").root
    image = _image(root)
    bucket = _bucket(root, image)
    body = {"image_path": str(image), "user": "breeder", "annotations": [], "bucket": bucket}

    resp = client.post("/api/annotate/labels", json={**body, "accept": [0, 0]})
    assert resp.status_code == 400, resp.text
    assert "more than once" in resp.text
    assert _nothing_written(image, bucket)

    assert client.post("/api/annotate/labels", json={**body, "accept": [0]}).status_code == 200
    assert len(_stored(root, image).annotations) == 1


def test_a_bucket_document_that_will_not_read_answers_400_at_the_proposals_route(
        tmp_path: Path, client) -> None:
    from tcip_mcp.buckets import read_bucket
    from tests._web_fixtures import open_new_project

    root = open_new_project(tmp_path / "proj").root
    image = _image(root)
    bucket = _bucket(root, image)
    params = {"image_path": str(image), "bucket": bucket}
    assert client.get("/api/annotate/proposals", params=params).status_code == 200

    document = read_bucket(root, bucket).document_key(image.stem)
    assert document is not None
    ts.replace(document, {"annotations": [{"subject": "bud", "bbox": [5, 5, 0, 5],
                                           "score": 0.9}]})

    resp = client.get("/api/annotate/proposals", params=params)
    assert resp.status_code == 400, resp.text
    assert "record 0" in resp.text


def test_the_queue_refuses_a_label_document_that_will_not_read_by_name(tmp_path: Path) -> None:
    from tcip_mcp.tools.feedback_tools import _prepare_queue_sources

    image = _image(tmp_path)
    _save(tmp_path, image, [], gestures=Gestures(complete={"bud": True}))
    _sources, skipped, error = _prepare_queue_sources(str(image.parent), "bud")
    assert (skipped, error) == (1, None)

    ts.replace(image_label_key(image), {"annotations": {}})

    _sources, _skipped, error = _prepare_queue_sources(str(image.parent), "bud")
    assert error is not None and "annotations" in error["error"]


def test_the_proposals_payload_carries_what_the_editor_reads_and_no_more(
        tmp_path: Path, client) -> None:
    from tests._web_fixtures import open_new_project

    root = open_new_project(tmp_path / "proj").root
    image = _image(root)
    bucket = _bucket(root, image)
    _save(root, image, [], gestures=Gestures(complete={"bud": True}))

    proposals = client.get("/api/annotate/proposals", params={
        "image_path": str(image), "bucket": bucket}).json()
    loaded = client.get("/api/annotate/labels", params={"image_path": str(image)}).json()

    assert set(proposals) == {"bucket", "operating_point", "proposals"}
    assert set(proposals["operating_point"]) == {"conf", "reason"}
    assert set(proposals["proposals"][0]) >= {"index", "paired", "decision"}
    assert "admitted" not in proposals["proposals"][0]
    assert loaded["completion"] == {"bud": "negative"}


def test_a_staged_bucket_serves_no_operating_point_and_says_why(
        tmp_path: Path, client) -> None:
    """A bucket staged under no assessment has no conf a review may accept at: the route says
    so beside the proposals rather than deciding admission for each of them."""
    from tests._web_fixtures import open_new_project

    root = open_new_project(tmp_path / "proj").root
    image = _image(root)
    bucket = _bucket(root, image)

    served = client.get("/api/annotate/proposals", params={
        "image_path": str(image), "bucket": bucket}).json()

    assert served["operating_point"]["conf"] is None
    assert served["operating_point"]["reason"]


def test_the_loaded_annotations_carry_the_index_a_pairing_names(tmp_path: Path, client) -> None:
    """Each loaded annotation states its document index, the index a proposal's ``paired``
    refers to, so the editor pairs by the server's enumeration and never by position in a
    list it re-split by geometry."""
    from tests._web_fixtures import open_new_project

    root = open_new_project(tmp_path / "proj").root
    image = _image(root)
    bucket = _bucket(root, image)
    _save(root, image, [{"subject": "leaf", "points": RING}, {"subject": "bud", "bbox": BOX}])

    loaded = client.get("/api/annotate/labels", params={"image_path": str(image)}).json()
    served = client.get("/api/annotate/proposals", params={
        "image_path": str(image), "bucket": bucket}).json()

    assert [a["index"] for a in loaded["annotations"]] == [0, 1]
    (proposal,) = served["proposals"]
    assert proposal["paired"] == 1
    assert loaded["annotations"][proposal["paired"]]["subject"] == "bud"


# ── completion marks ───────────────────────────────────────────────────────


def test_an_edit_invalidates_the_marks_of_its_own_subject_only(tmp_path: Path) -> None:
    image = _image(tmp_path)
    _save(tmp_path, image, [{"subject": "bud", "bbox": BOX}],
          gestures=Gestures(complete={"bud": True, "leaf": True}))
    assert _stored(tmp_path, image).state("bud") == "complete"
    assert _stored(tmp_path, image).state("leaf") == "negative"

    _save(tmp_path, image, [*_as_loaded(tmp_path, image),
                            {"subject": "bud", "bbox": [5.0, 5.0, 15.0, 15.0]}])
    assert _stored(tmp_path, image).state("bud") == "partial"
    assert _stored(tmp_path, image).state("leaf") == "negative"

    _save(tmp_path, image, [*_as_loaded(tmp_path, image),
                            {"subject": "leaf", "bbox": [70.0, 5.0, 90.0, 15.0]}])
    assert _stored(tmp_path, image).state("leaf") == "partial"


def test_a_mark_made_with_proposals_hidden_says_so(tmp_path: Path) -> None:
    image = _image(tmp_path)

    _save(tmp_path, image, [], gestures=Gestures(complete={"bud": True}, proposals_hidden=True))
    _save(tmp_path, image, [], gestures=Gestures(complete={"leaf": True}))

    marks = _stored(tmp_path, image).marks
    assert [m.proposals_hidden for m in marks["bud"]] == [True]
    assert [m.proposals_hidden for m in marks["leaf"]] == [False]


def test_a_negative_is_an_empty_subject_and_a_mark_at_every_reader(
        tmp_path: Path, client) -> None:
    """The admission that trains, the review queue and the editor's own listing each read an
    empty document as nothing until a person marks it, and as a negative once they have."""
    from tcip_mcp.pipelines.data.label_queries import admit, registry_scope
    from tcip_mcp.tools.feedback_tools import _prepare_queue_sources
    from tests._web_fixtures import open_new_project

    root = open_new_project(tmp_path / "proj").root
    image = _image(root)
    _save(root, image, [])

    def readers() -> tuple[int, int, str]:
        counts = admit(image.parent, members=[image.stem],
                       scope=registry_scope(image.parent, "bud")).tallies
        _sources, skipped, _error = _prepare_queue_sources(str(image.parent), "bud")
        listed = client.get("/api/annotate/labels", params={"image_path": str(image)}).json()
        return counts.get("negative", 0), skipped, listed["completion"].get("bud", "unannotated")

    assert readers() == (0, 0, "unannotated")

    _save(root, image, [], gestures=Gestures(complete={"bud": True}))

    assert readers() == (1, 1, "negative")


def test_saved_provenance_is_the_requests_actor_whatever_the_browser_sends(
        tmp_path: Path, client) -> None:
    from tests._web_fixtures import open_new_project

    root = open_new_project(tmp_path / "proj").root
    image = _image(root)

    resp = client.post("/api/annotate/labels", json={
        "image_path": str(image), "user": "breeder",
        "annotations": [{"subject": "bud", "bbox": BOX, "created_by": "user:mallory",
                         "created_at": "2020-01-01T00:00:00+00:00",
                         "accepted_by": "user:mallory"}]})
    assert resp.status_code == 200, resp.text

    (saved,) = _stored(root, image).annotations
    assert saved.created_by == "user:breeder"
    assert saved.created_at != "2020-01-01T00:00:00+00:00"
    assert saved.accepted_by is None


# ── one matcher answers the assessment and the editor ──────────────────────


def _counted(annotations: list, proposals: list, criterion: dict) -> dict:
    """The assessment's governing count over one image's annotations and proposals."""
    from tcip_annotation.json_io import xywh
    from tcip_mcp.pipelines.training.evaluation import (
        dt_record, governing_counts, gt_record, prediction_record,
    )

    def box(a) -> list[float]:
        g = a.geometry
        return xywh(g.x1, g.y1, g.x2, g.y2)

    record = prediction_record(
        [dt_record(box(p), 1, p.score) for p in proposals],
        [gt_record(box(a), 1, a.iscrowd) for a in annotations],
        width=WIDTH, height=HEIGHT, cap=None, count=len(proposals))
    return governing_counts([record], criterion, conf_threshold=0.0)


def test_the_assessment_count_and_the_editor_pairing_agree_under_a_crowd_region(
        tmp_path: Path) -> None:
    """The assessment's governing count and the editor's proposal pairing, each over the same
    annotations and a bucket published under no assessment, a crowd region among them, agree on
    what pairs."""
    from tcip_mcp.dataset_layout import image_proposals, proposal_pairs
    from tcip_mcp.pipelines.training.evaluation import resolve_match_criterion
    from tcip_mcp.tools.proposal_tools import stage_proposals

    image = _image(tmp_path)
    _save(tmp_path, image, [{"subject": "bud", "bbox": [10.0, 10.0, 30.0, 30.0]},
                            {"subject": "bud", "bbox": [50.0, 10.0, 90.0, 70.0], "iscrowd": True}])

    def norm(x1: float, y1: float, x2: float, y2: float, conf: float) -> dict:
        return {"subject": "bud", "conf": conf, "cx": (x1 + x2) / 2 / WIDTH,
                "cy": (y1 + y2) / 2 / HEIGHT, "w": (x2 - x1) / WIDTH, "h": (y2 - y1) / HEIGHT}

    staged = stage_proposals(tmp_path, str(image), model_name="detector", boxes=[
        norm(11, 11, 31, 31, 0.9), norm(60, 20, 70, 30, 0.8), norm(5, 60, 15, 75, 0.7)])
    assert "error" not in staged, staged
    bucket, proposals = image_proposals(staged["bucket"], image_label_key(image))
    annotations = _stored(tmp_path, image).annotations

    counted = _counted(annotations, proposals, resolve_match_criterion(None, []))
    paired = proposal_pairs(tmp_path, bucket, annotations, proposals)

    assert counted["tp"] == len(paired) == 1
    assert counted["fp"] == 1  # the proposal centered in the crowd region is neither
    (pair,) = paired.items()
    assert annotations[pair[1]].iscrowd is False


def test_the_editor_pairs_a_bucket_under_its_assessments_center_match(tmp_path: Path) -> None:
    """A bucket published under a center-match assessment pairs in the editor as that
    assessment counts: an annotation shifted off its proposal by under the assessment's
    tolerance, but by more than the comparability IoU admits, pairs and counts alike."""
    pytest.importorskip("torch")
    from tcip_mcp.assessment import read_assessment
    from tcip_mcp.dataset_layout import image_proposals, proposal_pairs
    from tcip_mcp.pipelines.training.evaluation import resolve_match_criterion
    from tests._chain_fixtures import IMG, run_the_chain

    chain = run_the_chain(tmp_path, experiment_id="exp-editor-pairing")
    bucket = chain.read()
    image = next(p for p in sorted(chain.images_dir.iterdir())
                 if bucket.document_key(p.stem) is not None)
    _bucket_record, proposals = image_proposals(bucket.name, image_label_key(image))
    top = max((p for p in proposals if p.score is not None), key=lambda p: p.score)
    g = top.geometry
    size = g.x2 - g.x1
    shifted = {"subject": top.subject, "bbox": [g.x1 + 0.2 * size, g.y1 + 0.2 * size,
                                                g.x2 + 0.2 * size, g.y2 + 0.2 * size]}
    save_label_document(tmp_path, image_label_key(image), [shifted], width=IMG, height=IMG,
                        author="user:breeder", actor="user:breeder")
    annotations = json_io.read_label_document(image_label_key(image)).annotations
    criterion = read_assessment(tmp_path, bucket.assessment_id).criterion["count"]["localization"]

    paired = proposal_pairs(tmp_path, bucket, annotations, proposals)

    assert criterion["kind"] == "center_match"
    assert len(paired) == _counted(annotations, proposals, criterion)["tp"] == 1
    assert _counted(annotations, proposals, resolve_match_criterion(None, []))["tp"] == 0


# ── the proposer is named ──────────────────────────────────────────────────


class DottedProposer:
    """A proposer the agent brings by dotted path: one candidate over the whole frame."""

    def propose(self, image_path: str, **params) -> list[dict]:
        return [{"candidate_id": 1, "bbox": [10.0, 10.0, 20.0, 20.0], "area": 400.0,
                 "rings": [[(10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)]],
                 "score": 0.9}]


def test_propose_annotations_refuses_an_unnamed_proposer_and_runs_a_dotted_one(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tcip_mcp.pipelines import proposal
    from tcip_mcp.tools.proposal_tools import propose_annotations

    monkeypatch.setitem(proposal._ENGINES, "registered_here", DottedProposer())
    image = _image(tmp_path)

    refused = propose_annotations(tmp_path, str(image), engine="")
    assert "registered_here" in refused["error"]

    proposed = propose_annotations(tmp_path, str(image),
                                   engine="tests.test_label_document_gestures:DottedProposer")
    assert proposed["staged"] is True and proposed["candidate_count"] == 1
