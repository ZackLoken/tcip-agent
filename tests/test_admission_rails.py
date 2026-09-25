"""The admission rails: which samples a place holding ground truth admits, and why each one that
is dropped was dropped.

Covers the one shape read over a place (``ground_truth_shape``), the label store's own
subject-scoped admission and its human confirmations, the attribute-completeness rail, the
quarantine over a since-changed schema, and the targets the loaders read off each admitted
sample's own document."""

import json

import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox, Polygon  # noqa: E402
from tcip_mcp.dataset_layout import (
    record_image_statuses, stamp_image_status_digests, status_bucket,
)  # noqa: E402
from tcip_mcp.subject_registry import (  # noqa: E402
    Attribute, SubjectRegistry, Subject, assign_class_ids,
)

from tests._producer_fixtures import admit_over, dataset_over  # noqa: E402

BUD = "bud"


def _make_images(images_dir, stems):
    images_dir.mkdir(parents=True, exist_ok=True)
    for s in stems:
        Image.new("RGB", (100, 100)).save(images_dir / f"{s}.jpg")


def _box(x1, y1, x2, y2, *, subject=BUD, score=None, **attrs):
    """A name-based detection annotation (a prediction when ``score`` is set)."""
    return Annotation(subject=subject, geometry=BBox(x1, y1, x2, y2), score=score,
                      attributes=dict(attrs))


def _poly(points, *, subject=BUD, **attrs):
    """A one-ring polygon annotation: the ordinary case, one contour."""
    return Annotation(subject=subject, geometry=Polygon([points]), attributes=dict(attrs))


def _multi_poly(rings, *, subject=BUD, **attrs):
    """One occlusion-split instance: several disjoint rings, still a single annotation."""
    return Annotation(subject=subject, geometry=Polygon(list(rings)), attributes=dict(attrs))


def _reg_id_map(subject=BUD, attribute=None, values=()):
    """A registry + its ``assign_class_ids`` map (the single name→id map) for a training scope."""
    attrs = ((Attribute(name=attribute, type="categorical", values=tuple(values)),)
             if attribute else ())
    reg = SubjectRegistry(subjects=(Subject(name=subject, attributes=attrs),))
    return reg, assign_class_ids(reg, subject, attribute)


# ── the one shape read ──────────────────────────────────────────────────────

def test_ground_truth_shape_reads_a_directory_of_documents(tmp_path):
    from tcip_mcp.pipelines.data.label_queries import ground_truth_shape

    d = tmp_path / "detect"
    d.mkdir()
    json_io.write_annotations(d / "a.json", [_box(10, 10, 50, 50)], 100, 100)
    assert ground_truth_shape(d) == "document"


def test_ground_truth_shape_reads_masks_and_tables(tmp_path):
    from tcip_mcp.pipelines.data.label_queries import ground_truth_shape

    masks = tmp_path / "masks"
    masks.mkdir()
    Image.new("L", (8, 8)).save(masks / "a.png")
    assert ground_truth_shape(masks) == "mask"

    table = tmp_path / "gt.csv"
    table.write_text("stem,value\na,1\n", encoding="utf-8")
    assert ground_truth_shape(table) == "table"


def test_ground_truth_shape_reads_an_empty_directory_as_documents(tmp_path):
    """Nothing annotated yet is the document shape with nothing in it; the admission below names
    what is missing image by image rather than the shape read refusing the whole directory."""
    from tcip_mcp.pipelines.data.label_queries import ground_truth_shape

    d = tmp_path / "nothing"
    d.mkdir()
    assert ground_truth_shape(d) == "document"


def test_ground_truth_shape_ignores_a_bucket_sidecar(tmp_path):
    """A directory holding only a bucket's own provenance stamp has no label document in it."""
    from tcip_mcp.pipelines.data.label_queries import ground_truth_shape

    d = tmp_path / "detect"
    d.mkdir()
    (d / "operating_point.json").write_text("{}")
    assert ground_truth_shape(d) == "document"


def test_ground_truth_shape_treats_the_old_objects_schema_as_documents(tmp_path):
    """The shape read names which kind of ground truth a place holds and never reads a document:
    a directory holding an old 'objects' document is the document shape, whose reader refuses
    that document when admission reads it."""
    from tcip_mcp.pipelines.data.label_queries import ground_truth_shape

    d = tmp_path / "detect"
    d.mkdir()
    (d / "a.json").write_text(json.dumps({"objects": [{"label": "bud"}]}))
    assert ground_truth_shape(d) == "document"


def _subject_bearing_coco(path):
    """A two-image dataset-level COCO whose records also carry the per-image ``subject`` key, the
    shape the per-image decoder would otherwise read as one image's labels."""
    path.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "img0.jpg"}, {"id": 2, "file_name": "img1.jpg"}],
        "categories": [{"id": 1, "name": BUD}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "subject": BUD, "bbox": [1, 1, 9, 9]},
            {"id": 2, "image_id": 2, "category_id": 1, "subject": BUD, "bbox": [20, 20, 9, 9]},
        ],
    }))


def test_a_dataset_level_coco_at_a_label_path_is_refused_by_the_one_reader(tmp_path):
    """No training or calibration reader reads a COCO document under another name: the per-image
    reader every one of them shares refuses it, the loaders' target read and admission alike."""
    from tcip_mcp.pipelines.data.label_queries import json_det_targets
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    images = tmp_path / "images"
    labels = tmp_path / "detect"
    labels.mkdir()
    _make_images(images, ["img0", "img1"])
    json_io.write_annotations(labels / "img1.json", [_box(10, 10, 50, 50)], 100, 100)
    _subject_bearing_coco(labels / "img0.json")

    with pytest.raises(json_io.UnreadableLabelDocument, match="import_coco"):
        json_det_targets(str(labels / "img0.json"), BUD, None, {BUD: 0})
    with pytest.raises(json_io.UnreadableLabelDocument, match="dataset-level COCO"):
        auto_train_val("detection", {"images_dir": str(images), "labels_dir": str(labels),
                                     "subject": BUD}, None)


def test_a_same_stem_provenance_sidecar_is_never_read_as_that_images_label(tmp_path):
    """A bucket's own provenance stamp shares a stem with an image of that name. Admission pairs
    each candidate with the document its directory actually holds for it, so the sidecar is never
    opened as a label: that image is unannotated, and the rest of the directory still trains."""
    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    labels.mkdir(parents=True)
    _make_images(images, ["img0", "img1", "operating_point"])
    for stem in ("img0", "img1"):
        json_io.write_annotations(labels / f"{stem}.json", [_box(10, 10, 50, 50)], 100, 100)
    (labels / "operating_point.json").write_text("{}", encoding="utf-8")

    admitted = admit_over(images, labels, subject=BUD)
    assert sorted(r.member for r in admitted.records) == ["img0", "img1"]


def test_a_document_with_no_image_is_not_admitted(tmp_path):
    """Membership is the images the place holds ground truth for. A document whose image was
    deleted or renamed names nothing to train on, so it never enters the run."""
    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    labels.mkdir(parents=True)
    _make_images(images, ["kept"])
    for stem in ("kept", "orphan"):
        json_io.write_annotations(labels / f"{stem}.json", [_box(10, 10, 50, 50)], 100, 100)

    admitted = admit_over(images, labels, subject=BUD)
    assert [r.member for r in admitted.records] == ["kept"]


def test_a_noncanonical_labelme_document_refuses_rather_than_reading_as_empty(tmp_path):
    """A document in another tool's schema is unreadable ground truth, not an empty one: reading
    it as empty would train a labeled image as entirely background."""
    from tcip_annotation.json_io import UnreadableLabelDocument

    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    labels.mkdir(parents=True)
    _make_images(images, ["a"])
    (labels / "a.json").write_text(json.dumps({
        "version": "5.0.1", "imageWidth": 100, "imageHeight": 100,
        "shapes": [{"label": BUD, "shape_type": "rectangle",
                    "points": [[10, 10], [50, 50]]}],
    }), encoding="utf-8")

    with pytest.raises(UnreadableLabelDocument):
        admit_over(images, labels, subject=BUD)


# ── targets read off each sample's own document ─────────────────────────────

def test_detection_reads_its_samples_own_documents(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "detect"
    labels.mkdir()
    _make_images(images, ["img0"])
    json_io.write_annotations(labels / "img0.json", [_box(10, 10, 50, 50)], 100, 100)

    ds = dataset_over("detection", images, labels, subject=BUD)
    _, target = ds[0]
    assert target["boxes"].shape == (1, 4)
    assert target["boxes"].tolist()[0] == pytest.approx([10, 10, 50, 50])
    assert target["labels"].tolist() == [1]  # 0-indexed cid 0 -> 1-indexed
    assert ds.class_distribution == {0: 1}


def test_a_point_document_is_admitted_and_trains_through_the_builder_that_reads_points(tmp_path):
    """Admission asks whether the document carries the subject at all, so a document carrying it
    as a point is admitted; which geometries answer for a measurement is the selected loader's own
    fact, so the detection loader refuses that sample by name rather than training the image as
    background, while a builder that reads points is handed those same samples and trains over the
    coordinates their own documents record."""
    import math

    from tcip_annotation.state import Point

    images, labels = tmp_path / "images", tmp_path / "annotations"
    labels.mkdir(parents=True)
    _make_images(images, ["p0", "p1"])
    for index, stem in enumerate(("p0", "p1")):
        json_io.write_annotations(
            labels / f"{stem}.json",
            [Annotation(subject=BUD, geometry=Point(20 + index, 30 + index))], 100, 100)

    admitted = admit_over(images, labels, subject=BUD)
    assert sorted(r.member for r in admitted.records) == ["p0", "p1"]

    with pytest.raises(ValueError, match="only in geometries a detection loader does not read"):
        dataset_over("detection", images, labels, subject=BUD)

    # Admits valid work: a builder that reads points is handed the same admitted samples, reads
    # each document's own coordinates, and trains over them.
    built = dataset_over(
        "keypoints", images, labels, subject=BUD,
        dataset_source={"builder": "tests.test_dataset_source_seam:build_point_ds"})
    assert sorted(s.member for s in built.samples) == ["p0", "p1"]
    assert built.points == [(20.0, 30.0), (21.0, 31.0)]

    pixels = torch.stack([built[i][0].mean(dim=(1, 2)) for i in range(len(built))])
    targets = torch.stack([built[i][1] for i in range(len(built))])
    model = torch.nn.Linear(pixels.shape[1], 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    losses = []
    for _ in range(3):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(pixels), targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert all(math.isfinite(value) for value in losses)
    assert losses[-1] < losses[0]


def test_class_distribution_counts_only_this_loaders_own_samples(tmp_path):
    """Two loaders over two sides of one dataset report their own distributions, never the
    dataset's unsplit whole: each reads the documents its own samples name."""
    from tcip_mcp.pipelines.data.datasets import build_dataset, resolve_sizes

    images, labels = tmp_path / "images", tmp_path / "annotations"
    stems = [f"img{i}" for i in range(4)]
    _make_images(images, stems)
    labels.mkdir(parents=True)
    for i, stem in enumerate(stems):
        json_io.write_annotations(labels / f"{stem}.json", [_box(10, 10, 30, 30)] * (i + 1),
                                  100, 100)

    admitted = admit_over(images, labels, subject=BUD)
    samples = admitted.samples({stem: "train" for stem in stems}, lambda s: s)
    by_stem = {sample.member: sample for sample in samples}
    # The run's own sizes, resolved once over both sides, as a run builds its loaders.
    sizes = resolve_sizes("detection", {}, samples)
    train_ds = build_dataset("detection", samples=[by_stem[s] for s in stems[:2]],
                             scope=admitted.scope, sizes=sizes)
    val_ds = build_dataset("detection", samples=[by_stem[s] for s in stems[2:]],
                           scope=admitted.scope, sizes=sizes)

    # img0 (1 box) + img1 (2 boxes) = 3; img2 (3 boxes) + img3 (4 boxes) = 7.
    assert train_ds.class_distribution == {0: 3}
    assert val_ds.class_distribution == {0: 7}


def test_instance_seg_reads_its_samples_own_documents(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "segment"
    labels.mkdir()
    _make_images(images, ["img0"])
    json_io.write_annotations(
        labels / "img0.json",
        [_poly([(10, 10), (50, 10), (50, 50), (10, 50)])], 100, 100)

    ds = dataset_over("instance_seg", images, labels, subject=BUD)
    _, target = ds[0]
    assert target["boxes"].shape == (1, 4)
    assert target["masks"].shape[0] == 1
    assert target["labels"].tolist() == [1]
    assert int(target["masks"].sum()) > 0  # polygon rasterized to a non-empty mask


# Two disjoint lobes of one instance, with a clear gap between them.
LOBE_A = [(10, 10), (30, 10), (30, 30), (10, 30)]
LOBE_B = [(60, 10), (80, 10), (80, 30), (60, 30)]


def _assert_one_mask_over_both_lobes(target) -> None:
    """A 2-ring instance is one mask covering both lobes, not two instances, not one lobe."""
    assert target["masks"].shape[0] == 1, "a multi-ring instance must not split into several"
    assert target["labels"].tolist() == [1]
    # The box spans the union of the rings.
    assert target["boxes"].tolist() == [[10.0, 10.0, 80.0, 30.0]]
    mask = target["masks"][0].numpy()
    assert mask[10:31, 10:31].sum() > 0, "the first lobe is missing from the mask"
    assert mask[10:31, 60:81].sum() > 0, "the second lobe is missing from the mask"
    # The occluded gap between the lobes stays background: the union of rings, not their hull.
    assert mask[:, 35:55].sum() == 0


def test_instance_seg_rasterizes_a_two_ring_instance_into_one_mask(tmp_path):
    """An occlusion-split instance (a bud behind a branch) is one object with two regions, and
    every ring of it rasterizes into that instance's single mask."""
    images, labels = tmp_path / "images", tmp_path / "annotations"
    labels.mkdir(parents=True)
    _make_images(images, ["img0"])
    json_io.write_annotations(labels / "img0.json", [_multi_poly([LOBE_A, LOBE_B])], 100, 100)

    ds = dataset_over("instance_seg", images, labels, subject=BUD)
    _, target = ds[0]
    _assert_one_mask_over_both_lobes(target)


def test_instance_seg_two_single_ring_instances_stay_two_masks(tmp_path):
    """The rail admits the ordinary case too: the same two lobes authored as separate annotations
    are two instances with two masks; multi-ring support must not merge distinct objects."""
    images, labels = tmp_path / "images", tmp_path / "annotations"
    labels.mkdir(parents=True)
    _make_images(images, ["img0"])
    json_io.write_annotations(
        labels / "img0.json", [_poly(LOBE_A), _poly(LOBE_B)], 100, 100)

    ds = dataset_over("instance_seg", images, labels, subject=BUD)
    _, target = ds[0]
    assert target["masks"].shape[0] == 2
    assert target["boxes"].tolist() == [[10.0, 10.0, 30.0, 30.0], [60.0, 10.0, 80.0, 30.0]]


def test_count_label_lines_reads_json_objects(tmp_path):
    from tcip_mcp.pipelines.data.splits import count_label_lines
    labels = tmp_path / "detect"
    labels.mkdir()
    json_io.write_annotations(labels / "a.json", [_box(0, 0, 10, 10), _box(0, 0, 20, 20)], 100, 100)
    json_io.write_annotations(labels / "neg.json", [], 100, 100, keep_empty=True)
    assert count_label_lines(labels / "a.json") == 2
    assert count_label_lines(labels / "neg.json") == 0
    assert count_label_lines(labels / "missing.json") == 0


# ── the label store's own rails ─────────────────────────────────────────────

def _rail_fixture(tmp_path):
    """One annotated image, one empty-unconfirmed, one with no label file, one confirmed negative."""
    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    labels.mkdir(parents=True)
    _make_images(images, ["ann", "empty", "nolabel", "neg"])
    json_io.write_annotations(labels / "ann.json", [_box(4, 4, 12, 12)], 100, 100, keep_empty=True)
    json_io.write_annotations(labels / "empty.json", [], 100, 100, keep_empty=True)
    json_io.write_annotations(labels / "neg.json", [], 100, 100, keep_empty=True)
    record_image_statuses(tmp_path, status_bucket(BUD, None), {"neg.jpg": "negative"},
                          recorded_by="user:breeder")
    return images, labels


def test_only_annotated_and_confirmed_negatives_train(tmp_path):
    """Samples come from the annotated set, never from an image list: a project where the breeder
    labeled 30 of 400 images must not train on the other 370 asserted to be empty."""
    images, labels = _rail_fixture(tmp_path)

    admitted = admit_over(images, labels, subject=BUD)
    assert sorted(r.member for r in admitted.records) == ["ann", "neg"]
    assert admitted.counts["annotated"] == 1
    assert admitted.counts["confirmed_negative"] == 1
    assert admitted.counts["skipped_unannotated"] >= 1


def test_a_corrupt_confirmed_negative_refuses_the_admission(tmp_path):
    """A stored 'negative' whose label document is corrupt must never train as a zero-object
    negative: the admission raises rather than reading the file as empty."""
    from tcip_annotation.json_io import UnreadableLabelDocument

    images, labels = _rail_fixture(tmp_path)
    (labels / "neg.json").write_text("not json {][", encoding="utf-8")

    with pytest.raises(UnreadableLabelDocument):
        admit_over(images, labels, subject=BUD)


def test_caller_supplied_members_are_filtered_too(tmp_path):
    """A narrowed candidate set goes through the same gate, otherwise a narrowing reintroduces the
    fabrications."""
    images, labels = _rail_fixture(tmp_path)
    admitted = admit_over(images, labels, subject=BUD,
                          members=["ann", "empty", "nolabel", "neg"])
    assert sorted(r.member for r in admitted.records) == ["ann", "neg"]


def test_no_trainable_samples_raises_rather_than_training_on_nothing(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    labels.mkdir(parents=True)
    _make_images(images, ["a", "b"])
    json_io.write_annotations(labels / "a.json", [], 100, 100, keep_empty=True)  # unconfirmed empty

    with pytest.raises(ValueError, match="no trainable samples"):
        admit_over(images, labels, subject=BUD)


def test_instance_seg_applies_the_same_rail(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    labels.mkdir(parents=True)
    _make_images(images, ["ann", "nolabel"])
    json_io.write_annotations(labels / "ann.json",
                              [_poly([(4, 4), (12, 4), (12, 12), (4, 12)])], 100, 100,
                              keep_empty=True)

    ds = dataset_over("instance_seg", images, labels, subject=BUD)
    assert ds.record_stems == ["ann"]


def test_instance_seg_excludes_a_partially_labeled_stem_from_training(tmp_path):
    """An image with any instance never assessed for ``attribute`` is held out whole, never
    trained on its labeled subset."""
    root = tmp_path / "ds"
    images_dir, labels_dir = root / "images", root / "annotations"
    _make_images(images_dir, ["complete", "partial"])
    labels_dir.mkdir(parents=True)
    json_io.write_annotations(labels_dir / "complete.json", [
        _poly([(4, 4), (12, 4), (12, 12), (4, 12)], opening="open"),
    ], 100, 100)
    json_io.write_annotations(labels_dir / "partial.json", [
        _poly([(4, 4), (12, 4), (12, 12), (4, 12)], opening="closed"),
        _poly([(40, 40), (60, 40), (60, 60), (40, 60)]),  # unlabeled: no opening attribute
    ], 100, 100)
    _reg, id_map = _reg_id_map(attribute="opening", values=("open", "closed"))
    _write_registry_for(root, attribute="opening", values=("open", "closed"))

    admitted = admit_over(images_dir, labels_dir, subject=BUD, attribute="opening")

    assert [r.member for r in admitted.records] == ["complete"]
    assert admitted.id_map == id_map
    assert admitted.counts["skipped_incomplete_attribute"] == 1
    assert admitted.counts["annotated"] == 1


def _write_registry_for(root, *, attribute=None, values=()):
    """The dataset's own subjects.json, which an attribute-scoped admission reads its class order
    from."""
    from tcip_mcp.subject_registry import write_registry

    reg, _ = _reg_id_map(attribute=attribute, values=values)
    write_registry(root / "subjects.json", reg)
    return reg


@pytest.mark.parametrize("ext", [".jpg", ".JPG"])
def test_confirmed_negative_survives_an_uppercase_extension(tmp_path, ext):
    """The name compared against the status store must be the real one on disk.

    `Path.exists()` is case-insensitive on Windows and macOS, so probing constructed paths returns
    the name that was built, not the one on disk. The status store is keyed on the real filename,
    so a fabricated name matches nothing and every human-confirmed negative is silently dropped,
    including the review loop's hard negatives. `IMG_*.JPG` is this repo's canonical camera name.
    """
    from PIL import Image as _Image

    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for stem in ("IMG_0001", "IMG_0002"):
        _Image.new("RGB", (100, 100)).save(images / f"{stem}{ext}")
    json_io.write_annotations(labels / "IMG_0001.json", [_box(4, 4, 12, 12)], 100, 100,
                              keep_empty=True)
    json_io.write_annotations(labels / "IMG_0002.json", [], 100, 100, keep_empty=True)
    record_image_statuses(tmp_path, status_bucket(BUD, None), {f"IMG_0002{ext}": "negative"},
                          recorded_by="user:breeder")

    admitted = admit_over(images, labels, subject=BUD)
    assert sorted(r.member for r in admitted.records) == ["IMG_0001", "IMG_0002"]
    assert admitted.counts["confirmed_negative"] == 1


def test_semantic_seg_requires_a_mask_but_admits_an_all_background_one(tmp_path):
    """Existence is the whole rail for masks: an all-background mask is a real annotation."""
    import numpy as np
    from PIL import Image as _Image

    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    for stem in ("has_mask", "all_background", "no_mask"):
        _Image.new("RGB", (32, 32)).save(images / f"{stem}.jpg")
    _Image.fromarray(np.ones((32, 32), dtype=np.uint8)).save(masks / "has_mask.png")
    _Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(masks / "all_background.png")

    ds = dataset_over("semantic_seg", images, masks)
    assert sorted(ds.record_stems) == ["all_background", "has_mask"]


def test_sample_counts_distinguish_unannotated_from_unconfirmed_empty(tmp_path):
    """"Annotate this" and "confirm this empty one" are different jobs: the count must say which."""
    images, labels = _rail_fixture(tmp_path)
    admitted = admit_over(images, labels, subject=BUD)
    assert admitted.counts == {"annotated": 1, "confirmed_negative": 1, "skipped_unannotated": 1,
                               "skipped_unconfirmed_empty": 1, "skipped_incomplete_attribute": 0,
                               "quarantined_stale_definition": 0}


def test_a_confirmation_does_not_leak_across_subjects(tmp_path):
    """A Complete is a statement about one trait. Re-applying it elsewhere trains an image full of
    bushes as containing no bushes."""
    from tcip_mcp.pipelines.data.label_queries import confirmed_negative_names

    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    labels.mkdir(parents=True)
    _make_images(images, ["ann", "shared"])
    # Every subject's annotation records share one per-image file; subject is a field in the record.
    json_io.write_annotations(labels / "ann.json",
                              [_box(4, 4, 12, 12, subject="bud"),
                               _box(4, 4, 12, 12, subject="bush")], 100, 100, keep_empty=True)
    json_io.write_annotations(labels / "shared.json", [], 100, 100, keep_empty=True)
    # Confirmed negative for bud only; the breeder never judged it for bush.
    record_image_statuses(tmp_path, status_bucket("bud", None), {"shared.jpg": "negative"},
                          recorded_by="user:breeder")

    assert confirmed_negative_names(labels, subject="bud", date=None) == {"shared.jpg"}
    assert confirmed_negative_names(labels, subject="bush", date=None) == set()
    assert sorted(r.member for r in admit_over(images, labels, subject="bud").records) == [
        "ann", "shared"]
    assert [r.member for r in admit_over(images, labels, subject="bush").records] == ["ann"]


def test_unresolvable_subject_refuses_rather_than_dropping_negatives(tmp_path):
    """Silently returning nothing would discard every hard negative the review loop harvested.

    A flat ``labels/`` dir can't name its subject from its path; the confirmations live
    dataset-native, a sibling of ``labels/``'s own resolved root, not found by walking arbitrarily
    far up an ancestor chain.
    """
    from tcip_mcp.pipelines.data.label_queries import confirmed_negative_names

    labels = tmp_path / "labels"
    labels.mkdir(parents=True)
    record_image_statuses(tmp_path, status_bucket("bud", None), {"a.jpg": "negative"},
                          recorded_by="user:breeder")

    with pytest.raises(ValueError, match="needs an explicit subject"):
        confirmed_negative_names(labels, subject=None, date=None)


def test_a_derived_tree_without_negatives_does_not_refuse(tmp_path):
    """Refuse only when there is something to lose.

    A split or curated export cannot name its subject. Raising there would block the platform's
    own documented split -> train path; with no confirmed negatives in the project there is nothing
    to drop, so it must proceed.
    """
    from tcip_mcp.pipelines.data.label_queries import confirmed_negative_names

    labels = tmp_path / "labels"
    labels.mkdir(parents=True)
    record_image_statuses(tmp_path, status_bucket("bud", None), {"a.jpg": "complete"},
                          recorded_by="user:breeder")

    assert confirmed_negative_names(labels, subject=None, date=None) == set()


def test_a_stale_complete_confirmation_is_quarantined(tmp_path):
    """A complete confirmation is an image trained by its label file's content alone: a bud image
    finished under a two-value attribute vocabulary that grew to three is held out exactly as a
    stale negative already is, never admitted as ``annotated`` by the boxes it happens to carry."""
    from tcip_mcp import subject_registry
    from tcip_mcp.pipelines.data.label_queries import admit, admitted_documents, require_admitted

    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    labels.mkdir(parents=True)
    registry = _write_registry_for(tmp_path, attribute="opening", values=("closed", "open"))
    current_digest = subject_registry.attribute_schema_digest(registry, BUD)

    _make_images(images, ["a"])
    json_io.write_annotations(labels / "a.json", [_box(4, 4, 12, 12)], 100, 100)

    record_image_statuses(tmp_path, status_bucket(BUD, None), {"a.jpg": "complete"},
                          recorded_by="user:breeder")
    assert current_digest != "stale-digest"
    stamp_image_status_digests(tmp_path, status_bucket(BUD, None), ["a.jpg"], "stale-digest")

    records, counts = admitted_documents(str(labels), str(images), subject=BUD, date=None)
    keep = [record.member for record in records]

    assert keep == []
    assert counts["quarantined_stale_definition"] == 1
    assert counts["annotated"] == 0
    with pytest.raises(ValueError, match="quarantined"):
        require_admitted(admit(images, labels, subject=BUD))


def test_a_quarantined_negative_reads_as_quarantined(tmp_path):
    """A human-confirmed-but-schema-stale negative reads as ``quarantined_stale_definition``
    ("looked, but the schema changed since"), never as ``skipped_unconfirmed_empty`` ("nobody ever
    looked")."""
    from tcip_mcp import subject_registry
    from tcip_mcp.pipelines.data.label_queries import admitted_documents

    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    labels.mkdir(parents=True)
    registry = _write_registry_for(tmp_path, attribute="opening", values=("closed", "open"))
    current_digest = subject_registry.attribute_schema_digest(registry, BUD)

    _make_images(images, ["a"])
    json_io.write_annotations(labels / "a.json", [], 100, 100, keep_empty=True)

    record_image_statuses(tmp_path, status_bucket(BUD, None), {"a.jpg": "negative"},
                          recorded_by="user:breeder")
    assert current_digest != "stale-digest"
    stamp_image_status_digests(tmp_path, status_bucket(BUD, None), ["a.jpg"], "stale-digest")

    _records, counts = admitted_documents(str(labels), str(images), subject=BUD, date=None)
    assert counts["quarantined_stale_definition"] == 1
    assert counts["skipped_unconfirmed_empty"] == 0


def test_a_reconfirmed_complete_trains_again_after_the_schema_change(tmp_path):
    """Re-confirming restamps the current digest, so the same image trains once a human has
    looked again under the vocabulary now in effect."""
    from tcip_mcp import subject_registry
    from tcip_mcp.pipelines.data.label_queries import admitted_documents

    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    labels.mkdir(parents=True)
    registry = _write_registry_for(tmp_path, attribute="opening", values=("closed", "open"))
    current_digest = subject_registry.attribute_schema_digest(registry, BUD)

    _make_images(images, ["a"])
    json_io.write_annotations(labels / "a.json", [_box(4, 4, 12, 12)], 100, 100)
    record_image_statuses(tmp_path, status_bucket(BUD, None), {"a.jpg": "complete"},
                          recorded_by="user:breeder")
    stamp_image_status_digests(tmp_path, status_bucket(BUD, None), ["a.jpg"], current_digest)

    records, counts = admitted_documents(str(labels), str(images), subject=BUD, date=None)
    keep = [record.member for record in records]
    assert keep == ["a"]
    assert counts["annotated"] == 1
    assert counts["quarantined_stale_definition"] == 0


def test_an_unstamped_complete_trains(tmp_path):
    """A complete confirmation the stamp transaction never reached is admitted, not quarantined:
    absence of a stamp is never evidence of staleness."""
    from tcip_mcp.pipelines.data.label_queries import admitted_documents

    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    labels.mkdir(parents=True)
    _write_registry_for(tmp_path, attribute="opening", values=("closed", "open"))

    _make_images(images, ["a"])
    json_io.write_annotations(labels / "a.json", [_box(4, 4, 12, 12)], 100, 100)
    record_image_statuses(tmp_path, status_bucket(BUD, None), {"a.jpg": "complete"},
                          recorded_by="user:breeder")

    records, counts = admitted_documents(str(labels), str(images), subject=BUD, date=None)
    keep = [record.member for record in records]
    assert keep == ["a"]
    assert counts["quarantined_stale_definition"] == 0


def test_a_complete_under_an_unchanged_subject_trains(tmp_path):
    """Another subject's own schema change never quarantines a bucket the change had no part in:
    only bud's digest moves, so bush's complete, stamped under its own still-current digest,
    admits."""
    from tcip_mcp import subject_registry
    from tcip_mcp.subject_registry import write_registry
    from tcip_mcp.pipelines.data.label_queries import admitted_documents

    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    labels.mkdir(parents=True)
    registry = SubjectRegistry(subjects=(
        Subject(name=BUD, attributes=(
            Attribute(name="opening", type="categorical", values=("closed", "open")),
        )),
        Subject(name="bush"),
    ))
    write_registry(tmp_path / "subjects.json", registry)
    bush_digest = subject_registry.attribute_schema_digest(registry, "bush")

    _make_images(images, ["a"])
    json_io.write_annotations(labels / "a.json", [_box(4, 4, 12, 12, subject="bush")], 100, 100)
    record_image_statuses(tmp_path, status_bucket("bush", None), {"a.jpg": "complete"},
                          recorded_by="user:breeder")
    stamp_image_status_digests(tmp_path, status_bucket("bush", None), ["a.jpg"], bush_digest)
    # bud's own schema changes; bush's bucket, and its stamp, must be untouched by it.
    stamp_image_status_digests(tmp_path, status_bucket(BUD, None), ["b.jpg"], "stale-digest")

    records, counts = admitted_documents(str(labels), str(images), subject="bush", date=None)
    keep = [record.member for record in records]
    assert keep == ["a"]
    assert counts["quarantined_stale_definition"] == 0


def test_a_partial_carrying_a_stale_stamp_still_trains(tmp_path):
    """A partial is not a human's assertion (it carries no Complete), so a stamp on it, however
    stale, never quarantines: the quarantine is over finished statuses only."""
    from tcip_mcp.pipelines.data.label_queries import admitted_documents

    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    labels.mkdir(parents=True)
    _write_registry_for(tmp_path, attribute="opening", values=("closed", "open"))

    _make_images(images, ["a"])
    json_io.write_annotations(labels / "a.json", [_box(4, 4, 12, 12)], 100, 100)
    record_image_statuses(tmp_path, status_bucket(BUD, None), {"a.jpg": "partial"},
                          recorded_by="user:breeder")
    stamped = stamp_image_status_digests(
        tmp_path, status_bucket(BUD, None), ["a.jpg"], "stale-digest")
    assert stamped == ["a.jpg"]

    records, counts = admitted_documents(str(labels), str(images), subject=BUD, date=None)
    keep = [record.member for record in records]
    assert keep == ["a"]
    assert counts["quarantined_stale_definition"] == 0


def test_a_stale_and_contradicted_negative_still_trains_by_content(tmp_path):
    """Real content contradicts a stored negative outright; staleness never overrides that, and
    the contradiction is still named for the caller to surface."""
    from tcip_mcp.pipelines.data.label_queries import admitted_documents

    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    labels.mkdir(parents=True)
    _write_registry_for(tmp_path, attribute="opening", values=("closed", "open"))

    _make_images(images, ["a"])
    # Recorded negative, but the label file now carries real content: a contradiction.
    json_io.write_annotations(labels / "a.json", [_box(4, 4, 12, 12)], 100, 100)
    record_image_statuses(tmp_path, status_bucket(BUD, None), {"a.jpg": "negative"},
                          recorded_by="user:breeder")
    stamp_image_status_digests(tmp_path, status_bucket(BUD, None), ["a.jpg"], "stale-digest")

    contradicted: set[str] = set()
    records, counts = admitted_documents(str(labels), str(images), subject=BUD, date=None,
                                         contradicted_out=contradicted)
    assert [record.member for record in records] == ["a"]
    assert counts["annotated"] == 1
    assert counts["quarantined_stale_definition"] == 0
    assert contradicted == {"a.jpg"}


def test_admission_with_subject_none_over_only_complete_statuses_does_not_refuse(tmp_path):
    """A tree holding only complete confirmations has no negative to lose, so an unthreaded
    subject does not refuse the way it would with a confirmed negative present (coverage)."""
    from tcip_mcp.pipelines.data.label_queries import admitted_documents

    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    labels.mkdir(parents=True)
    _make_images(images, ["a"])
    json_io.write_annotations(labels / "a.json", [_box(4, 4, 12, 12)], 100, 100)
    record_image_statuses(tmp_path, status_bucket(BUD, None), {"a.jpg": "complete"},
                          recorded_by="user:breeder")

    records, counts = admitted_documents(str(labels), str(images), subject=None, date=None)
    keep = [record.member for record in records]
    assert keep == ["a"]
    assert counts["annotated"] == 1


def test_json_det_targets_skips_unlabeled_instead_of_raising(tmp_path):
    """The loader's own per-image target reader accepts partially-attributed data: an unlabeled
    instance is excluded, not a hard abort, while an undecodable value still raises."""
    from tcip_mcp.pipelines.data.label_queries import json_det_targets

    path = tmp_path / "IMG_A.json"
    json_io.write_annotations(path, [
        Annotation(subject="bud", geometry=BBox(10, 10, 30, 30),
                  attributes={"opening": "closed"}),
        Annotation(subject="bud", geometry=BBox(40, 40, 60, 60), attributes={}),  # unlabeled
    ], 100, 100)

    id_map = {"open": 0, "closed": 1}
    target, n_unlabeled = json_det_targets(str(path), "bud", "opening", id_map)
    # 0-indexed 1 ("closed") + 1 for background
    assert len(target["boxes"]) == 1 and target["labels"] == [2] and target["iscrowd"] == [False]
    assert n_unlabeled == 1  # the second instance, disclosed rather than silently dropped

    undecodable = tmp_path / "IMG_B.json"
    json_io.write_annotations(undecodable, [
        Annotation(subject="bud", geometry=BBox(10, 10, 30, 30),
                  attributes={"opening": "not-a-real-value"}),
    ], 100, 100)
    with pytest.raises(ValueError):
        json_det_targets(str(undecodable), "bud", "opening", id_map)


def test_detection_excludes_a_partially_labeled_stem_from_training(tmp_path):
    """A stem with any instance unlabeled for ``attribute`` never reaches the loader: a fixed-length
    dataset cannot act on this per ``__getitem__`` call, so the exclusion happens at admission,
    matching the delivery-gating paths that already exclude the whole image rather than silently
    training on its labeled subset."""
    root = tmp_path / "ds"
    images_dir, labels_dir = root / "images", root / "annotations"
    _make_images(images_dir, ["complete", "partial"])
    labels_dir.mkdir(parents=True)
    _write_registry_for(root, attribute="opening", values=("open", "closed"))
    json_io.write_annotations(labels_dir / "complete.json", [
        _box(10, 10, 30, 30, opening="open"),
    ], 100, 100)
    json_io.write_annotations(labels_dir / "partial.json", [
        _box(10, 10, 30, 30, opening="closed"),
        _box(40, 40, 60, 60),  # unlabeled: no opening attribute at all
    ], 100, 100)

    admitted = admit_over(images_dir, labels_dir, subject=BUD, attribute="opening")
    assert [r.member for r in admitted.records] == ["complete"]
    # The drop is recorded under its real reason, never one its absence downstream resembles.
    assert admitted.counts["skipped_incomplete_attribute"] == 1
    assert admitted.counts["annotated"] == 1
    assert admitted.counts["skipped_unconfirmed_empty"] == 0

    ds = dataset_over("detection", images_dir, labels_dir, subject=BUD, attribute="opening")
    assert len(ds.det_targets(ds.stems[0])["boxes"]) == 1


def test_tiled_detection_indexes_no_tile_from_an_attribute_incomplete_image(tmp_path):
    """The tiler expands the admitted samples into one training sample per tile, so an image the
    attribute-completeness rail held out contributes no tile at all. Asserting on the tile index,
    not on the admitted members, is what pins that: a tile carrying the image's real but unlabeled
    objects would train them as background, one tile at a time."""
    root = tmp_path / "ds"
    images_dir, labels_dir = root / "images", root / "annotations"
    _make_images(images_dir, ["complete", "partial"])
    labels_dir.mkdir(parents=True)
    _write_registry_for(root, attribute="opening", values=("open", "closed"))
    json_io.write_annotations(labels_dir / "complete.json", [
        _box(10, 10, 30, 30, opening="open")], 100, 100)
    json_io.write_annotations(labels_dir / "partial.json", [
        _box(10, 10, 30, 30, opening="closed"),
        _box(40, 40, 60, 60),  # unlabeled
    ], 100, 100)

    tiled = dataset_over("detection", images_dir, labels_dir, subject=BUD, attribute="opening",
                         tiling={"enabled": True, "tile_size": 64, "overlap": 0.0})

    assert set(tiled.record_stems) == {"complete"}  # tiles are per-index, so this is every tile
    assert len(tiled) > 0  # the rail admits the fully-attributed image's tiles
