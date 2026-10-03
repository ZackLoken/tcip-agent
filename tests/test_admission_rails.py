"""The admission rails: which samples a place holding ground truth admits, and why each one that
is dropped was dropped.

Covers the one shape read over a place (``ground_truth_shape``), the label store's own
subject-scoped admission and its completion marks, and the targets the loaders read off each
admitted sample's own document, an unassessed attribute among them."""

import json

import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox, Polygon  # noqa: E402
from tcip_mcp.pipelines.data.selection import ClassScope  # noqa: E402
from tcip_mcp.subject_registry import Attribute, SubjectRegistry, Subject  # noqa: E402

from tests._producer_fixtures import (  # noqa: E402
    admit_over, dataset_over, mark_complete, registry_over,
)

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


def test_ground_truth_shape_ignores_a_bucket_record(tmp_path):
    """A directory holding only a bucket's own record has no label document in it."""
    from tcip_mcp.pipelines.data.label_queries import ground_truth_shape

    d = tmp_path / "detect"
    d.mkdir()
    (d / "bucket.json").write_text("{}")
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

    from tcip_mcp.pipelines.data.label_queries import registry_scope

    with pytest.raises(json_io.UnreadableLabelDocument, match="import_coco"):
        json_det_targets(str(labels / "img0.json"), registry_scope(labels, BUD))
    with pytest.raises(json_io.UnreadableLabelDocument, match="dataset-level COCO"):
        auto_train_val(tmp_path, "detection", {"images_dir": str(images), "labels_dir": str(labels),
                                     "scope": {"subject": BUD}}, None)


def test_a_same_stem_bucket_record_is_never_read_as_that_images_label(tmp_path):
    """A bucket's own record shares a stem with an image of that name. Admission pairs each
    candidate with the document its directory actually holds for it, so the record is never
    opened as a label: that image is unannotated, and the rest of the directory still trains."""
    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    labels.mkdir(parents=True)
    _make_images(images, ["img0", "img1", "bucket"])
    for stem in ("img0", "img1"):
        json_io.write_annotations(labels / f"{stem}.json", [_box(10, 10, 50, 50)], 100, 100)
    (labels / "bucket.json").write_text("{}", encoding="utf-8")

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
    assert count_label_lines(labels / "a.json", ClassScope()) == 2
    assert count_label_lines(labels / "neg.json", ClassScope()) == 0
    with pytest.raises(FileNotFoundError):
        count_label_lines(labels / "missing.json", ClassScope())


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
    mark_complete(images / "neg.jpg", labels / "neg.json", BUD, project=tmp_path)
    return images, labels


def test_only_annotated_and_confirmed_negatives_train(tmp_path):
    """Samples come from the annotated set, never from an image list: a project where the breeder
    labeled 30 of 400 images must not train on the other 370 asserted to be empty."""
    images, labels = _rail_fixture(tmp_path)

    admitted = admit_over(images, labels, subject=BUD)
    assert sorted(r.member for r in admitted.records) == ["ann", "neg"]
    assert admitted.tallies["partial"] == 1
    assert admitted.tallies["negative"] == 1
    assert admitted.tallies["absent"] >= 1


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
    assert [ds.sample_of(k).member for k in ds.stems] == ["ann"]


def test_instance_seg_admits_a_partially_assessed_stem_on_its_subject_marks(tmp_path):
    """An image with an instance never assessed for an attribute is admitted on its subject's
    marks alone, every attribute of the registry read into the scope."""
    root = tmp_path / "ds"
    images_dir, labels_dir = root / "images", root / "annotations"
    _make_images(images_dir, ["complete", "partial"])
    labels_dir.mkdir(parents=True)
    json_io.write_annotations(labels_dir / "complete.json", [
        _poly([(4, 4), (12, 4), (12, 12), (4, 12)], opening="open"),
    ], 100, 100)
    json_io.write_annotations(labels_dir / "partial.json", [
        _poly([(4, 4), (12, 4), (12, 12), (4, 12)], opening="closed"),
        _poly([(40, 40), (60, 40), (60, 60), (40, 60)]),  # unassessed: no opening value
    ], 100, 100)
    reg = _write_registry_for(root, attribute="opening", values=("open", "closed"))

    admitted = admit_over(images_dir, labels_dir, subject=BUD)

    assert [r.member for r in admitted.records] == ["complete", "partial"]
    assert admitted.scope.attributes == reg.subjects[0].attributes
    assert admitted.tallies["partial"] == 2


def _write_registry_for(root, *, attribute=None, values=()):
    """The dataset's own subjects.json declaring ``attribute`` over ``values`` on :data:`BUD`
    (no attribute when ``None``), which an admission reads its attributes from."""
    attrs = ((Attribute(name=attribute, type="categorical", values=tuple(values)),)
             if attribute else ())
    reg = SubjectRegistry(subjects=(Subject(name=BUD, attributes=attrs),))
    registry_over(root, reg)
    return reg


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
    assert sorted(ds.sample_of(k).member for k in ds.stems) == ["all_background", "has_mask"]


def test_sample_counts_distinguish_unannotated_from_unconfirmed_empty(tmp_path):
    """"Annotate this" and "confirm this empty one" are different jobs: the count must say which,
    each tally the state its ground truth reads as, or absent."""
    images, labels = _rail_fixture(tmp_path)
    admitted = admit_over(images, labels, subject=BUD)
    assert admitted.tallies == {"partial": 1, "negative": 1, "absent": 1, "unannotated": 1}


def test_a_confirmation_does_not_leak_across_subjects(tmp_path):
    """A Complete is a statement about one trait. Re-applying it elsewhere trains an image full of
    bushes as containing no bushes."""
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
    mark_complete(images / "shared.jpg", labels / "shared.json", "bud", project=tmp_path)

    assert sorted(r.member for r in admit_over(images, labels, subject="bud").records) == [
        "ann", "shared"]
    assert [r.member for r in admit_over(images, labels, subject="bush").records] == ["ann"]


def test_a_negative_mark_dies_with_an_edit_of_its_subject(tmp_path):
    """A mark names its subject's annotations when it was made: once a bud is drawn on a marked
    negative, the mark no longer holds and the image trains as annotated, never as a negative."""
    images, labels = _rail_fixture(tmp_path)
    marks = json_io.read_label_document(labels / "neg.json").marks
    json_io.write_annotations(labels / "neg.json", [_box(4, 4, 12, 12)], 100, 100, marks=marks)

    assert json_io.read_label_document(labels / "neg.json").marks == {}
    tallies = admit_over(images, labels, subject=BUD).tallies
    assert (tallies["partial"], tallies.get("negative", 0)) == (2, 0)


def test_json_det_targets_marks_an_unassessed_row_and_refuses_an_undeclared_value(tmp_path):
    """The loader's own per-image target reader keeps an unassessed instance as a row whose
    attribute column carries the unassessed mark, while a value its attribute does not declare
    refuses."""
    from tcip_mcp.pipelines.data.label_queries import json_det_targets, registry_scope

    root = tmp_path / "ds"
    labels = root / "annotations"
    labels.mkdir(parents=True)
    _write_registry_for(root, attribute="opening", values=("open", "closed"))
    path = labels / "IMG_A.json"
    json_io.write_annotations(path, [
        Annotation(subject="bud", geometry=BBox(10, 10, 30, 30),
                  attributes={"opening": "closed"}),
        Annotation(subject="bud", geometry=BBox(40, 40, 60, 60), attributes={}),  # unassessed
    ], 100, 100)

    scope = registry_scope(labels, BUD)
    target = json_det_targets(str(path), scope)
    assert len(target["boxes"]) == 2 and target["labels"] == [1, 1]
    assert target["attributes"].tolist() == [[1], [json_io.UNASSESSED]]

    undeclared = labels / "IMG_B.json"
    json_io.write_annotations(undeclared, [
        Annotation(subject="bud", geometry=BBox(10, 10, 30, 30),
                  attributes={"opening": "not-a-real-value"}),
    ], 100, 100)
    with pytest.raises(json_io.UndeclaredValue):
        json_det_targets(str(undeclared), scope)


def test_detection_and_its_tiles_keep_a_partially_assessed_stem(tmp_path):
    """A stem with an instance unassessed for an attribute reaches the loader and the tiler whole,
    its unassessed row carried in step with its box through every row filter, so no real object
    trains as background."""
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
        _box(40, 40, 60, 60),  # unassessed: no opening value at all
    ], 100, 100)

    ds = dataset_over("detection", images_dir, labels_dir, subject=BUD)
    _image, target = ds[ds.stems.index(next(k for k in ds.stems
                                            if ds.sample_of(k).member == "partial"))]
    assert target["attributes"].tolist() == [[1], [json_io.UNASSESSED]]

    tiled = dataset_over("detection", images_dir, labels_dir, subject=BUD,
                         tiling={"enabled": True, "tile_size": 64, "overlap": 0.0,
                                 "sliver_frac": 0.5})  # stated: one box derives no spread
    assert {tiled.sample_of(k).member for k in tiled.stems} == {"complete", "partial"}
    for index in range(len(tiled)):
        _tile, rows = tiled[index]
        assert rows["attributes"].shape == (len(rows["boxes"]), 1)
