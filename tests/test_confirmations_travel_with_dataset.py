"""Confirmations travel with the dataset, quarantined on subject-definition mismatch.

Confirmed negatives now live dataset-native (``<dataset_root>/.tcip/state/image_status.json``, a
sibling of ``classes.json``) rather than in whichever project's private ``.tcip/`` happened to be an
ancestor of the labels dir. ``dataset_fingerprint`` folds this store in as a 4th term (it previously
could not detect confirming/un-confirming a negative at all). A confirmation is quarantined only when
a stamped attribute-schema digest positively disagrees with the subject's current schema: an
unstamped confirmation is admitted, never punished for predating the mechanism.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from tcip_annotation import json_io
from tcip_mcp import class_registry
from tcip_mcp.class_registry import Attribute, ClassRegistry, Subject
from tcip_mcp.dataset_layout import stamp_image_status_digests, status_bucket


def _write_image(images_dir: Path, stem: str, size=(64, 64)) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(100, 100, 100)).save(images_dir / f"{stem}.jpg")


def _write_registry(root: Path, *subjects: Subject) -> ClassRegistry:
    registry = ClassRegistry(subjects=tuple(subjects))
    class_registry.write_registry(root / "classes.json", registry)
    return registry


def _confirm_negative(root: Path, subject: str, image_name: str, *, date=None,
                      digest: str | None = None) -> None:
    """Confirms through the writer the real routes call, so the stored shape is theirs and a test
    can confirm several images into the same bucket across separate calls."""
    from tcip_mcp.dataset_layout import CONFIRMED_NEGATIVE, record_image_statuses

    bucket = status_bucket(subject, date)
    record_image_statuses(root, bucket, {image_name: CONFIRMED_NEGATIVE},
                          recorded_by="user:breeder")
    if digest is not None:
        stamp_image_status_digests(root, bucket, [image_name], digest)


def _dataset(tmp_path: Path, *, negative: bool = False, subjects=(Subject(name="bud"),)):
    _write_registry(tmp_path, *subjects)
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "annotations"
    labels_dir.mkdir(parents=True)
    _write_image(images_dir, "img_001")
    json_io.write_annotations(labels_dir / "img_001.json", [], 64, 64, keep_empty=True)
    if negative:
        _confirm_negative(tmp_path, "bud", "img_001.jpg")
    return tmp_path


# (a) fail-before: the fingerprint was blind to confirmed-negative membership.
def test_dataset_fingerprint_changes_with_confirmed_negatives(tmp_path):
    from tcip_mcp.pipelines.data.dataset_fingerprint import dataset_fingerprint

    root = _dataset(tmp_path, negative=False)
    before = dataset_fingerprint(root)
    assert before is not None

    _confirm_negative(root, "bud", "img_001.jpg")
    after = dataset_fingerprint(root)
    assert after is not None
    assert after != before, (
        "confirming a negative must change dataset_fingerprint, otherwise compare_experiments / "
        "check_dataset_identity.py can read two runs with different trainable membership as identical"
    )


def test_dataset_fingerprint_stable_with_no_confirmations_store(tmp_path):
    """The confirmations term must not null the whole fingerprint (matches the registry term's
    optional/additive convention); a dataset with zero confirmed negatives is still valid content."""
    from tcip_mcp.pipelines.data.dataset_fingerprint import dataset_fingerprint

    root = _dataset(tmp_path, negative=False)
    assert dataset_fingerprint(root) is not None


# (b) confirmations are dataset-native: an unrelated ancestor's store is never consulted, even
# when the dataset itself has none of its own (this is the case that actually discriminates
# ancestor-walking from dataset-local resolution; a dataset-root-local store would find the
# right answer either way without ever consulting the ancestor, so a naive fixture could pass for
# the wrong reason).
def test_confirmed_negative_names_ignores_an_unrelated_ancestor_store(tmp_path):
    from tcip_mcp.pipelines.data.label_queries import confirmed_negative_names

    # A project ancestor that happens to sit above the dataset, with its own negatives: simulates
    # a project referencing a dataset that lives elsewhere.
    unrelated_ancestor = tmp_path
    dataset_root = tmp_path / "shared_dataset"
    dataset_root.mkdir()
    _write_registry(dataset_root, Subject(name="bud"))
    labels_dir = dataset_root / "annotations"
    labels_dir.mkdir()
    json_io.write_annotations(labels_dir / "img_002.json", [], 64, 64, keep_empty=True)

    _confirm_negative(unrelated_ancestor, "bud", "img_002.jpg")  # a foreign, ancestor-only store

    got = confirmed_negative_names(labels_dir, subject="bud", date=None)
    assert got == set(), (
        "a store found by walking up to an unrelated ancestor must never be silently borrowed, "
        "even when the dataset itself has no confirmations of its own"
    )


# (c) quarantine: a stamped digest that no longer matches the current schema is excluded.
def test_quarantine_excludes_a_confirmation_stamped_under_a_since_changed_schema(tmp_path):
    from tcip_mcp.class_registry import attribute_schema_digest
    from tcip_mcp.pipelines.data.label_queries import confirmed_negative_names

    root = _dataset(tmp_path, negative=False, subjects=(
        Subject(name="bud", attributes=(
            Attribute(name="opening", type="categorical", values=("closed", "open")),
        )),
    ))
    old_registry = class_registry.read_registry(root / "classes.json")
    old_digest = attribute_schema_digest(old_registry, "bud")
    _confirm_negative(root, "bud", "img_001.jpg", digest=old_digest)

    # The subject's attribute schema changes (a new value added) after the confirmation was stamped.
    _write_registry(root, Subject(name="bud", attributes=(
        Attribute(name="opening", type="categorical",
                 values=("closed", "partial", "open")),
    )))

    quarantined: set[str] = set()
    got = confirmed_negative_names(root / "annotations", subject="bud", date=None,
                                   quarantined_out=quarantined)
    assert got == set()
    assert quarantined == {"img_001.jpg"}


def test_quarantine_does_not_fire_when_schema_is_unchanged(tmp_path):
    from tcip_mcp.class_registry import attribute_schema_digest
    from tcip_mcp.pipelines.data.label_queries import confirmed_negative_names

    root = _dataset(tmp_path, negative=False, subjects=(
        Subject(name="bud", attributes=(
            Attribute(name="opening", type="categorical", values=("closed", "open")),
        )),
    ))
    digest = attribute_schema_digest(class_registry.read_registry(root / "classes.json"), "bud")
    _confirm_negative(root, "bud", "img_001.jpg", digest=digest)

    quarantined: set[str] = set()
    got = confirmed_negative_names(root / "annotations", subject="bud", date=None,
                                   quarantined_out=quarantined)
    assert got == {"img_001.jpg"}
    assert quarantined == set()


# (d) a rail must admit valid work: absence of a stamp is not evidence of staleness.
def test_unstamped_confirmation_is_admitted_not_quarantined(tmp_path):
    """Every image_status.json predating the digest-stamp mechanism (real projects, every existing
    test fixture) has no digest sidecar at all. Punishing that by quarantine-by-default would
    silently empty every one of them, exactly the 'strengthened rail rejects legitimate work'
    failure to avoid."""
    from tcip_mcp.pipelines.data.label_queries import confirmed_negative_names

    root = _dataset(tmp_path, negative=False)
    _confirm_negative(root, "bud", "img_001.jpg")  # no digest= kwarg -> no sidecar written

    quarantined: set[str] = set()
    got = confirmed_negative_names(root / "annotations", subject="bud", date=None,
                                   quarantined_out=quarantined)
    assert got == {"img_001.jpg"}
    assert quarantined == set()


# (e) trainable_stems surfaces a quarantine event as its own count, distinct from unconfirmed-empty.
def test_trainable_stems_reports_quarantined_stale_definition(tmp_path):
    from tcip_mcp.class_registry import attribute_schema_digest
    from tcip_mcp.pipelines.data.label_queries import trainable_stems

    root = _dataset(tmp_path, negative=False, subjects=(
        Subject(name="bud", attributes=(
            Attribute(name="opening", type="categorical", values=("closed", "open")),
        )),
    ))
    old_digest = attribute_schema_digest(
        class_registry.read_registry(root / "classes.json"), "bud")
    _confirm_negative(root, "bud", "img_001.jpg", digest=old_digest)
    _write_registry(root, Subject(name="bud", attributes=(
        Attribute(name="opening", type="categorical",
                 values=("closed", "partial", "open")),
    )))

    stems, counts = trainable_stems(root / "annotations", root / "images", subject="bud", date=None)
    assert stems == []
    assert counts["quarantined_stale_definition"] == 1
    assert counts["skipped_unconfirmed_empty"] == 0


# (f) quarantine is per-image, not per-bucket: a later, unrelated write to the same bucket must
# never resurrect a different image's stale, never-re-reviewed confirmation.
def test_quarantine_is_per_image_not_per_bucket(tmp_path):
    from tcip_mcp.class_registry import attribute_schema_digest
    from tcip_mcp.pipelines.data.label_queries import confirmed_negative_names

    root = _dataset(tmp_path, negative=False, subjects=(
        Subject(name="bud", attributes=(
            Attribute(name="opening", type="categorical", values=("closed", "open")),
        )),
    ))
    old_digest = attribute_schema_digest(
        class_registry.read_registry(root / "classes.json"), "bud")
    _confirm_negative(root, "bud", "img_001.jpg", digest=old_digest)

    # The schema changes after img_001's confirmation.
    _write_registry(root, Subject(name="bud", attributes=(
        Attribute(name="opening", type="categorical",
                 values=("closed", "partial", "open")),
    )))
    new_digest = attribute_schema_digest(
        class_registry.read_registry(root / "classes.json"), "bud")

    # An unrelated image is confirmed under the new (current) schema, into the same bucket; the
    # re-stamp must not silently un-quarantine img_001 too.
    _confirm_negative(root, "bud", "img_002.jpg", digest=new_digest)

    quarantined: set[str] = set()
    got = confirmed_negative_names(root / "annotations", subject="bud", date=None,
                                   quarantined_out=quarantined)
    assert got == {"img_002.jpg"}, "img_002 was stamped fresh and must be admitted"
    assert quarantined == {"img_001.jpg"}, (
        "img_001's stale stamp must still be caught even though a later write touched the bucket"
    )


# (g) materialize_dataset's review-harvested negatives must be quarantine-capable, not a permanent
# no-op for lack of any classes.json to compare against.
def test_materialize_dataset_carries_a_quarantine_capable_stamp(tmp_path):
    import tcip_store
    from tcip_mcp.dataset_layout import image_status_digest_key
    from tcip_mcp.pipelines.feedback.materialize import materialize_dataset

    src_root = tmp_path / "src"
    _write_registry(src_root, Subject(name="bud", attributes=(
        Attribute(name="opening", type="categorical", values=("closed", "open")),
    )))
    src_images = src_root / "images"
    _write_image(src_images, "imgB")
    expected_digest = class_registry.attribute_schema_digest(
        class_registry.read_registry(src_root / "classes.json"), "bud")

    review_state = {"image": {
        "imgB.jpg": {"img_status": "completed", "detections": [
            {"action": "rejected", "class_name": "bud", "gt_bbox_norm": None,
             "pred_bbox_norm": [0.5, 0.5, 0.1, 0.1]}]},
    }}
    out = tmp_path / "out"
    materialize_dataset(review_state, str(src_images), str(out), subject="bud")

    assert (out / "classes.json").is_file(), "the materialized dataset must be self-describing"
    stamps = tcip_store.read(image_status_digest_key(out), default={}).get("bud", {})
    assert stamps.get("imgB.jpg") == expected_digest
