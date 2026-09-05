"""Attribution and completeness of the confirmed negatives a review harvest materializes.

A rejection verdict is a human's statement that one subject is absent from one image, and it only
survives into training through the dataset-native status store, keyed by subject. So every rejected
image has to reach that store, the subject it is keyed under has to be the one whose absence those
rejections actually attest, and a review that answers for no single subject has to leave its
negatives unattributed and say so rather than pick one. Each confirmation records who established
it and when, so a negative a harvest wrote is not read back as a person's own Complete. The schema
stamp written beside them decides whether quarantine can ever protect them, so it has to be the
source registry's own digest for that subject.
"""

from __future__ import annotations

import json

from PIL import Image

import tcip_store as ts
from tcip_mcp import class_registry
from tcip_mcp.class_registry import Attribute, ClassRegistry, Subject, attribute_schema_digest
from tcip_mcp.dataset_layout import image_status_digest_key, image_status_key, status_bucket
from tcip_mcp.pipelines.data.label_queries import confirmed_negative_names
from tcip_mcp.pipelines.feedback.materialize import curated_manifest_key, materialize_dataset


def _image(images_dir, name: str, size) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (70, 110, 160)).save(images_dir / name)


def _accepted(class_name: str, gt) -> dict:
    return {"action": "accepted", "class_name": class_name, "gt_bbox_norm": gt,
            "pred_bbox_norm": None}


def _rejected(class_name: str, pred, reviewed_by: str = "breeder") -> dict:
    return {"action": "rejected", "class_name": class_name, "gt_bbox_norm": None,
            "pred_bbox_norm": pred, "reviewed_by": reviewed_by}


def _completed(detections: list[dict]) -> dict:
    return {"img_status": "completed", "detections": detections}


_TWO_SUBJECTS = (
    Subject(name="bud", attributes=(
        Attribute(name="stage", type="ordinal", values=("closed", "partial", "shedding")),)),
    Subject(name="leaf", attributes=(
        Attribute(name="damage", type="categorical", values=("none", "blight")),)),
)


def test_every_confirmed_negative_reaches_the_status_store(tmp_path):
    """All rejected-only images land in the store the canonical locator resolves, under one bucket."""
    src = tmp_path / "src"
    _image(src, "pos.png", (120, 40))
    for name in ("neg_a.png", "neg_b.png", "neg_c.png"):
        _image(src, name, (40, 120))
    out = tmp_path / "out"
    state = {"image": {
        "pos.png": _completed([_accepted("bud", [0.5, 0.5, 0.2, 0.3])]),
        "neg_a.png": _completed([_rejected("bud", [0.2, 0.8, 0.1, 0.1])]),
        "neg_b.png": _completed([_rejected("bud", [0.6, 0.3, 0.2, 0.1])]),
        "neg_c.png": _completed([_rejected("bud", [0.4, 0.4, 0.3, 0.2]),
                                 _rejected("bud", [0.9, 0.1, 0.05, 0.05])]),
    }}

    r = materialize_dataset(state, str(src), str(out))
    assert r["hard_negative"] == 3

    store_key = image_status_key(out)
    assert ts.exists(store_key)
    store = ts.read(store_key)
    assert list(store) == [status_bucket("bud", None)]
    bucket = store[status_bucket("bud", None)]
    assert sorted(bucket) == ["neg_a.png", "neg_b.png", "neg_c.png"]
    assert {name: rec["status"] for name, rec in bucket.items()} == {
        "neg_a.png": "negative", "neg_b.png": "negative", "neg_c.png": "negative"}

    assert confirmed_negative_names(out / "annotations", subject="bud", date=None) == {
        "neg_a.png", "neg_b.png", "neg_c.png"}

    # No source registry to stamp against, so nothing is stamped, and an unstamped confirmation is
    # admitted rather than quarantined.
    assert not ts.exists(image_status_digest_key(out))


def test_explicit_subject_outranks_the_derived_one(tmp_path):
    """The review's own subject keys the negatives even when the positives name another."""
    src = tmp_path / "src"
    _image(src, "pos.png", (100, 25))
    _image(src, "neg.png", (25, 100))
    out = tmp_path / "out"
    state = {"image": {
        "pos.png": _completed([_accepted("bud", [0.5, 0.5, 0.2, 0.2])]),
        "neg.png": _completed([_rejected("bush", [0.3, 0.3, 0.4, 0.4])]),
    }}

    r = materialize_dataset(state, str(src), str(out), subject="bush")
    assert r["subject"] == "bush"
    assert r["subjects"] == ["bud"]
    assert ts.read(curated_manifest_key(out))["subject"] == "bush"

    assert confirmed_negative_names(out / "annotations", subject="bush", date=None) == {"neg.png"}
    assert confirmed_negative_names(out / "annotations", subject="bud", date=None) == set()


def test_multi_subject_review_leaves_negatives_unattributed(tmp_path):
    """With no threaded subject and several in the verdicts, no subject may claim the negatives."""
    src = tmp_path / "src"
    _image(src, "buds.png", (140, 35))
    _image(src, "leaves.png", (35, 140))
    _image(src, "neg.png", (60, 90))
    out = tmp_path / "out"
    state = {"image": {
        "buds.png": _completed([_accepted("bud", [0.5, 0.5, 0.2, 0.2])]),
        "leaves.png": _completed([_accepted("leaf", [0.25, 0.75, 0.3, 0.1])]),
        "neg.png": _completed([_rejected("bud", [0.5, 0.5, 0.1, 0.1])]),
    }}

    r = materialize_dataset(state, str(src), str(out))
    assert r["subjects"] == ["bud", "leaf"]
    assert r["subject"] is None
    assert r["hard_negative"] == 1

    # The response names the image it left unconfirmed and why, rather than dropping it quietly.
    assert r["unconfirmed_negative"] == 1
    assert [e["image"] for e in r["unconfirmed_negatives"]] == ["neg.png"]
    assert "no single subject" in r["unconfirmed_negatives"][0]["reason"]

    assert not ts.exists(image_status_key(out))
    for subject in ("bud", "leaf"):
        assert confirmed_negative_names(out / "annotations", subject=subject, date=None) == set()

    # The image is still in the dataset, as an unconfirmed empty rather than a mis-keyed negative.
    assert json.loads((out / "annotations" / "neg.json").read_text())["annotations"] == []


def test_negative_stamps_match_the_source_registry_schema(tmp_path):
    """Each harvested negative is stamped with its own subject's current attribute-schema digest."""
    root = tmp_path / "dataset"
    registry = ClassRegistry(subjects=_TWO_SUBJECTS)
    root.mkdir()
    class_registry.write_registry(root / "classes.json", registry)
    images = root / "images"
    _image(images, "pos.png", (130, 45))
    _image(images, "neg_a.png", (45, 130))
    _image(images, "neg_b.png", (90, 30))
    out = tmp_path / "out"
    state = {"image": {
        "pos.png": _completed([_accepted("bud", [0.5, 0.5, 0.2, 0.2])]),
        "neg_a.png": _completed([_rejected("bud", [0.2, 0.2, 0.1, 0.1])]),
        "neg_b.png": _completed([_rejected("bud", [0.7, 0.4, 0.2, 0.3])]),
    }}

    materialize_dataset(state, str(images), str(out))

    bud_digest = attribute_schema_digest(registry, "bud")
    leaf_digest = attribute_schema_digest(registry, "leaf")
    assert bud_digest and leaf_digest and bud_digest != leaf_digest

    stamps = ts.read(image_status_digest_key(out))
    assert stamps[status_bucket("bud", None)] == {
        "neg_a.png": bud_digest, "neg_b.png": bud_digest}

    quarantined: set[str] = set()
    admitted = confirmed_negative_names(
        out / "annotations", subject="bud", date=None, quarantined_out=quarantined)
    assert admitted == {"neg_a.png", "neg_b.png"}
    assert quarantined == set()


def test_materialized_dataset_carries_its_own_registry_copy(tmp_path):
    """The output gets a classes.json of its own, so a later schema change is detectable here."""
    root = tmp_path / "dataset"
    registry = ClassRegistry(subjects=_TWO_SUBJECTS)
    root.mkdir()
    class_registry.write_registry(root / "classes.json", registry)
    images = root / "images"
    _image(images, "pos.png", (110, 55))
    _image(images, "neg.png", (55, 110))
    out = tmp_path / "out"
    state = {"image": {
        "pos.png": _completed([_accepted("bud", [0.5, 0.5, 0.2, 0.2])]),
        "neg.png": _completed([_rejected("bud", [0.4, 0.6, 0.1, 0.2])]),
    }}

    materialize_dataset(state, str(images), str(out))

    assert (out / "classes.json").is_file()
    copied = class_registry.read_registry(out / "classes.json")
    assert {s.name for s in copied.subjects} == {"bud", "leaf"}
    assert attribute_schema_digest(copied, "bud") == attribute_schema_digest(registry, "bud")


def test_a_rejection_of_one_subject_never_confirms_another(tmp_path):
    """An image disputed only for another object is not this subject's confirmed negative.

    The review affirmed buds on one image and rejected a bush prediction on a second. Nothing
    on the second image answers for buds, so nothing may key it as a bud negative: doing so
    asserts an image full of buds is empty of them.
    """
    src = tmp_path / "src"
    _image(src, "buds.png", (120, 40))
    _image(src, "disputed_bush.png", (40, 120))
    out = tmp_path / "out"
    state = {"image": {
        "buds.png": _completed([_accepted("bud", [0.5, 0.5, 0.2, 0.3])]),
        "disputed_bush.png": _completed([_rejected("bush", [0.2, 0.8, 0.6, 0.6])]),
    }}

    r = materialize_dataset(state, str(src), str(out))

    assert confirmed_negative_names(out / "annotations", subject="bud", date=None) == set()
    assert confirmed_negative_names(out / "annotations", subject="bush", date=None) == set()
    assert r["subject"] is None
    assert r["verdict_subjects"] == ["bud", "bush"]
    assert [e["image"] for e in r["unconfirmed_negatives"]] == ["disputed_bush.png"]
    assert r["unconfirmed_negatives"][0]["rejected_subjects"] == ["bush"]


def test_a_stated_subject_never_claims_another_subjects_rejections(tmp_path):
    """Keying the harvest under a subject does not make every rejected image its negative."""
    src = tmp_path / "src"
    _image(src, "answers_for_bud.png", (100, 30))
    _image(src, "answers_for_bush.png", (30, 100))
    out = tmp_path / "out"
    state = {"image": {
        "answers_for_bud.png": _completed([_rejected("bud", [0.5, 0.5, 0.2, 0.2])]),
        "answers_for_bush.png": _completed([_rejected("bush", [0.3, 0.3, 0.4, 0.4])]),
    }}

    r = materialize_dataset(state, str(src), str(out), subject="bud")

    assert confirmed_negative_names(out / "annotations", subject="bud", date=None) == {
        "answers_for_bud.png"}
    assert [e["image"] for e in r["unconfirmed_negatives"]] == ["answers_for_bush.png"]
    assert "not for 'bud'" in r["unconfirmed_negatives"][0]["reason"]

    # Left in the dataset as an unconfirmed empty, which is what nobody has answered for.
    assert json.loads((out / "annotations" / "answers_for_bush.json").read_text())[
        "annotations"] == []


def test_a_harvested_negative_records_who_confirmed_it_and_when(tmp_path):
    """The reviewer whose rejections established the negative is what the store records."""
    src = tmp_path / "src"
    _image(src, "pos.png", (90, 60))
    _image(src, "neg.png", (60, 90))
    out = tmp_path / "out"
    state = {"image": {
        "pos.png": _completed([_accepted("bud", [0.5, 0.5, 0.2, 0.2])]),
        "neg.png": _completed([_rejected("bud", [0.4, 0.4, 0.2, 0.2], reviewed_by="rowan")]),
    }}

    materialize_dataset(state, str(src), str(out))

    record = ts.read(image_status_key(out))[status_bucket("bud", None)]["neg.png"]
    assert isinstance(record, dict), "a stored status is a record, not a bare token"
    assert record["status"] == "negative"
    assert record["recorded_by"] == "user:rowan"
    assert record["recorded_at"].startswith("20")


def test_classified_scope_never_confirms_a_negative_even_when_a_value_names_the_subject(tmp_path):
    """Under a classified scope, ``rejected_subjects`` are the reviewed bucket's attribute
    values, never object classes: a value spelled like the subject itself is a vocabulary
    coincidence, not a claim the object is absent, so the rejected-only image is left
    unconfirmed like any other one, never read as a confirmed negative of the object.
    """
    from tcip_mcp.pipelines.resolution import BucketScope

    root = tmp_path / "dataset"
    registry = ClassRegistry(subjects=_TWO_SUBJECTS)
    root.mkdir()
    class_registry.write_registry(root / "classes.json", registry)
    images = root / "images"
    _image(images, "pos.png", (100, 30))
    _image(images, "neg.png", (30, 100))
    out = tmp_path / "out"
    state = {"image": {
        "pos.png": _completed([_accepted("bud", [0.5, 0.5, 0.2, 0.2])]),
        # The confirmed value on this rejection happens to be spelled "bud", the object
        # class's own name: exactly the coincidence the classified branch must not confirm on.
        "neg.png": _completed([_rejected("bud", [0.4, 0.4, 0.2, 0.2])]),
    }}
    scope = BucketScope(subject="bud", attribute="stage")

    r = materialize_dataset(state, str(images), str(out), scope=scope)

    assert r["hard_negative"] == 1
    assert r["unconfirmed_negative"] == 1
    assert [e["image"] for e in r["unconfirmed_negatives"]] == ["neg.png"]
    assert "never that the object itself is absent" in r["unconfirmed_negatives"][0]["reason"]
    assert not ts.exists(image_status_key(out))
    assert confirmed_negative_names(out / "annotations", subject="bud", date=None) == set()


def test_classified_scope_refuses_a_confirmed_value_outside_the_bucket_vocabulary(tmp_path):
    """A ``vocabulary`` given to the harvest checks a classified verdict's confirmed value before
    it is written: a value the bucket's own ``id_map`` never declared is reported, not written,
    the same posture a degenerate box already gets."""
    from tcip_mcp.pipelines.resolution import BucketScope

    root = tmp_path / "dataset"
    registry = ClassRegistry(subjects=_TWO_SUBJECTS)
    root.mkdir()
    class_registry.write_registry(root / "classes.json", registry)
    images = root / "images"
    _image(images, "pos.png", (100, 30))
    out = tmp_path / "out"
    state = {"image": {
        "pos.png": _completed([_accepted("shedding", [0.5, 0.5, 0.2, 0.2])]),
    }}
    scope = BucketScope(subject="bud", attribute="stage")

    r = materialize_dataset(
        state, str(images), str(out), scope=scope, vocabulary={"closed", "partial"})

    assert r["positive"] == 0
    assert [e["image"] for e in r["boundary_refused"]] == ["pos.png"]
    assert "not a value" in r["boundary_refused"][0]["reason"]
    assert "shedding" in r["boundary_refused"][0]["reason"]


def test_a_classified_scope_with_no_vocabulary_refuses_rather_than_writing_unchecked(tmp_path):
    """A classified scope requires the bucket's own vocabulary to check a confirmed value
    against: with none given, the positive is reported in ``boundary_refused`` naming the
    requirement, never written unchecked."""
    from tcip_mcp.pipelines.resolution import BucketScope

    root = tmp_path / "dataset"
    registry = ClassRegistry(subjects=_TWO_SUBJECTS)
    root.mkdir()
    class_registry.write_registry(root / "classes.json", registry)
    images = root / "images"
    _image(images, "pos.png", (100, 30))
    out = tmp_path / "out"
    state = {"image": {
        "pos.png": _completed([_accepted("closed", [0.5, 0.5, 0.2, 0.2])]),
    }}
    scope = BucketScope(subject="bud", attribute="stage")

    r = materialize_dataset(state, str(images), str(out), scope=scope)

    assert r["positive"] == 0
    assert [e["image"] for e in r["boundary_refused"]] == ["pos.png"]
    assert "requires the bucket's own recorded vocabulary" in r["boundary_refused"][0]["reason"]


def test_a_classified_scope_with_its_vocabulary_admits_a_value_it_declares(tmp_path):
    """The admitting case: a classified scope with its own bucket vocabulary writes a value that
    vocabulary declares, the object class in ``subject`` and the value under the attribute."""
    from tcip_annotation.json_io import read_annotations
    from tcip_mcp.pipelines.resolution import BucketScope

    root = tmp_path / "dataset"
    registry = ClassRegistry(subjects=_TWO_SUBJECTS)
    root.mkdir()
    class_registry.write_registry(root / "classes.json", registry)
    images = root / "images"
    _image(images, "pos.png", (100, 30))
    out = tmp_path / "out"
    state = {"image": {
        "pos.png": _completed([_accepted("closed", [0.5, 0.5, 0.2, 0.2])]),
    }}
    scope = BucketScope(subject="bud", attribute="stage")

    r = materialize_dataset(
        state, str(images), str(out), scope=scope, vocabulary={"closed", "partial"})

    assert r["positive"] == 1
    assert r["boundary_refused"] == []
    written = read_annotations(str(out / "annotations" / "pos.json"))
    assert written[0].subject == "bud"
    assert written[0].attributes == {"stage": "closed"}


def test_a_negative_no_one_reviewer_answers_for_names_the_harvest(tmp_path):
    """Two reviewers disputing one image leaves the writing tool as the honest actor."""
    src = tmp_path / "src"
    _image(src, "pos.png", (60, 90))
    _image(src, "neg.png", (90, 60))
    out = tmp_path / "out"
    state = {"image": {
        "pos.png": _completed([_accepted("bud", [0.5, 0.5, 0.2, 0.2])]),
        "neg.png": _completed([
            _rejected("bud", [0.2, 0.2, 0.1, 0.1], reviewed_by="rowan"),
            _rejected("bud", [0.7, 0.7, 0.1, 0.1], reviewed_by="sam"),
        ]),
    }}

    materialize_dataset(state, str(src), str(out))

    record = ts.read(image_status_key(out))[status_bucket("bud", None)]["neg.png"]
    assert isinstance(record, dict), "a stored status is a record, not a bare token"
    assert not record["recorded_by"].startswith("user:"), (
        "a tool producer stays bare, so a reader can tell it from a person's own Complete")

    from tcip_mcp.pipelines.feedback.materialize import MATERIALIZER_IDENTITY

    assert record["recorded_by"] == MATERIALIZER_IDENTITY
