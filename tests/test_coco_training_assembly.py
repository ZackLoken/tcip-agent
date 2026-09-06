"""COCO-training-assembly: training/eval reads the canonical per-image JSON label store by
assembling a dataset-level COCO on the fly (``build_dataset`` auto-routes a JSON label dir onto
the ``label_format='coco'`` path). Covers detection + instance-seg, format detection, and the
stratification line-count that must see JSON objects (not only YOLO ``.txt``)."""

import json

import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox, Polygon  # noqa: E402
from tcip_mcp.dataset_layout import (
    record_image_statuses, stamp_image_status_digests, status_bucket,
)  # noqa: E402
from tcip_mcp.class_registry import (  # noqa: E402
    Attribute, ClassRegistry, Subject, assign_class_ids,
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


def _reg_id_map(subject=BUD, attribute=None, values=()):
    """A registry + its ``assign_class_ids`` map (the single name→id map) for a training scope."""
    attrs = ((Attribute(name=attribute, type="categorical", values=tuple(values)),)
             if attribute else ())
    reg = ClassRegistry(subjects=(Subject(name=subject, attributes=attrs),))
    return reg, assign_class_ids(reg, subject, attribute)


# ── format detection ────────────────────────────────────────────────────────

def test_dir_label_format_detects_json(tmp_path):
    from tcip_mcp.pipelines.data.label_queries import dir_label_format
    d = tmp_path / "detect"
    d.mkdir()
    json_io.write_annotations(d / "a.json", [_box(10, 10, 50, 50)], 100, 100)
    assert dir_label_format(d) == "json"


def test_dir_label_format_ignores_non_canonical_json(tmp_path):
    """A LabelMe-style .json (``shapes``, no ``annotations``) is not the canonical store: don't claim
    it, so it never gets silently assembled as though it were per-image JSON."""
    from tcip_mcp.pipelines.data.label_queries import dir_label_format
    d = tmp_path / "labels"
    d.mkdir()
    (d / "a.json").write_text(json.dumps({"shapes": [{"label": "x", "points": [[1, 1], [2, 2]]}]}))
    assert dir_label_format(d) is None


def test_dir_label_format_empty(tmp_path):
    from tcip_mcp.pipelines.data.label_queries import dir_label_format
    assert dir_label_format(tmp_path / "nope") is None


def test_dir_label_format_detects_a_dataset_level_coco(tmp_path):
    """A dataset-level COCO's 'images'/'categories' markers are checked before 'annotations', the
    same priority format_io reads a file's shape with everywhere else, so a labels_dir holding one
    reads 'coco' here too rather than being misclaimed as our per-image json."""
    from tcip_mcp.pipelines.data.label_queries import dir_label_format
    d = tmp_path / "detect"
    d.mkdir()
    (d / "dataset.json").write_text(json.dumps(
        {"images": [{"id": 1, "file_name": "a.jpg"}], "annotations": [], "categories": []}))
    assert dir_label_format(d) == "coco"


def test_dir_label_format_treats_the_old_objects_schema_as_unrecognized(tmp_path):
    """The old 'objects' schema, which format_io's detection raises on, is never our per-image
    json; dir_label_format's own never-raise contract folds that raise into None."""
    from tcip_mcp.pipelines.data.label_queries import dir_label_format
    d = tmp_path / "detect"
    d.mkdir()
    (d / "a.json").write_text(json.dumps({"objects": [{"label": "bud"}]}))
    assert dir_label_format(d) is None


def test_first_labels_json_excludes_a_bucket_sidecar(tmp_path):
    """A directory holding only a bucket's own provenance stamp has no first label document: the
    sidecar is never mistaken for one, the same walk ``dir_label_format`` shares."""
    from tcip_mcp.pipelines.data.label_queries import first_labels_json
    d = tmp_path / "detect"
    d.mkdir()
    (d / "operating_point.json").write_text("{}")
    assert first_labels_json(d) is None

    json_io.write_annotations(d / "a.json", [_box(10, 10, 50, 50)], 100, 100)
    assert first_labels_json(d) == d / "a.json"


# ── assemble_coco ───────────────────────────────────────────────────────────

def test_assemble_coco_pairs_labels_with_images(tmp_path):
    from tcip_mcp.pipelines.data.label_queries import assemble_coco
    images = tmp_path / "images"
    labels = tmp_path / "detect"
    labels.mkdir()
    _make_images(images, ["img0", "img1"])
    # Two classes now ride a categorical attribute of one subject (not two class_ids on the box).
    _reg, id_map = _reg_id_map(attribute="opening", values=("closed", "open"))
    json_io.write_annotations(
        labels / "img0.json",
        [_box(10, 10, 50, 50, opening="closed"), _box(0, 0, 20, 20, opening="open")],
        100, 100)
    json_io.write_annotations(labels / "img1.json", [], 100, 100, keep_empty=True)  # confirmed negative

    coco = assemble_coco(labels, images, subject=BUD, date=None, attribute="opening", id_map=id_map)
    # img1's empty file is not human-confirmed negative -> excluded (treated as unannotated)
    assert {im["file_name"] for im in coco["images"]} == {"img0.jpg"}
    assert len(coco["annotations"]) == 2
    assert {a["category_id"] for a in coco["annotations"]} == {0, 1}


def test_assemble_coco_with_no_stems_excludes_a_bucket_sidecar(tmp_path):
    """With no explicit ``stems``, the stem universe comes from prediction_documents, so a
    sidecar beside a real label never mints a spurious entry, even when an image happens to
    share the sidecar's own stem (a stray file the platform never ingested this way)."""
    from tcip_mcp.pipelines.data.label_queries import assemble_coco
    images = tmp_path / "images"
    labels = tmp_path / "detect"
    labels.mkdir()
    _make_images(images, ["img0", "operating_point"])
    _reg, id_map = _reg_id_map()
    json_io.write_annotations(labels / "img0.json", [_box(10, 10, 50, 50)], 100, 100)
    (labels / "operating_point.json").write_text("{}")

    coco = assemble_coco(labels, images, subject=BUD, date=None, id_map=id_map)

    assert {im["file_name"] for im in coco["images"]} == {"img0.jpg"}


def test_assemble_coco_includes_only_confirmed_negatives(tmp_path):
    from tcip_mcp.pipelines.data.label_queries import assemble_coco
    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    labels.mkdir(parents=True)
    _make_images(images, ["img0", "img1"])
    json_io.write_annotations(labels / "img0.json", [], 100, 100, keep_empty=True)  # confirmed below
    json_io.write_annotations(labels / "img1.json", [], 100, 100, keep_empty=True)  # emptied mid-work
    record_image_statuses(tmp_path, status_bucket(BUD, None),
                          {"img0.jpg": "negative", "img1.jpg": "partial"}, recorded_by="user:breeder")
    _reg, id_map = _reg_id_map()
    coco = assemble_coco(labels, images, subject=BUD, date=None, id_map=id_map)
    assert {im["file_name"] for im in coco["images"]} == {"img0.jpg"}  # human-confirmed only


def test_assemble_coco_skips_stem_without_image(tmp_path):
    from tcip_mcp.pipelines.data.label_queries import assemble_coco
    images = tmp_path / "images"
    labels = tmp_path / "detect"
    labels.mkdir()
    _make_images(images, ["img0"])
    _reg, id_map = _reg_id_map()
    json_io.write_annotations(labels / "img0.json", [_box(10, 10, 50, 50)], 100, 100)
    json_io.write_annotations(labels / "orphan.json", [_box(5, 5, 9, 9)], 100, 100)  # no image

    coco = assemble_coco(labels, images, subject=BUD, date=None, id_map=id_map)
    assert [im["file_name"] for im in coco["images"]] == ["img0.jpg"]


def test_class_distribution_on_a_shared_coco_scopes_to_its_own_stems(tmp_path):
    """training_tools.py's auto train/val split assembles the dataset-level COCO once and threads
    the same dict into the full/train/val builds, to avoid re-assembling it three times, so
    ``self._coco`` covers the whole dataset for every split while ``self.stems`` is narrowed per
    split. ``class_distribution``'s COCO branch must honor ``self.stems`` rather than iterating
    ``self._coco["annotations"]`` directly, or train and val report the identical, unsplit
    whole-dataset distribution instead of their own."""
    from tcip_mcp.pipelines.data.datasets import build_dataset
    from tcip_mcp.pipelines.data.label_queries import assemble_coco, resolve_registry_id_map

    images, labels = tmp_path / "images", tmp_path / "annotations"
    stems = [f"img{i}" for i in range(4)]
    _make_images(images, stems)
    labels.mkdir(parents=True)
    for i, stem in enumerate(stems):
        json_io.write_annotations(labels / f"{stem}.json", [_box(10, 10, 30, 30)] * (i + 1), 100, 100)

    _reg, id_map = resolve_registry_id_map(labels, BUD, None)
    shared_coco = assemble_coco(labels, images, subject=BUD, date=None, id_map=id_map)  # over all 4 stems

    train_ds = build_dataset("detection", images_dir=str(images), labels_dir=str(labels),
                             subject=BUD, coco_data=shared_coco, label_format="coco",
                             stems=stems[:2])
    val_ds = build_dataset("detection", images_dir=str(images), labels_dir=str(labels),
                           subject=BUD, coco_data=shared_coco, label_format="coco",
                           stems=stems[2:])

    assert train_ds._coco is val_ds._coco is shared_coco  # the actual sharing this bug depends on
    # img0 (1 box) + img1 (2 boxes) = 3; img2 (3 boxes) + img3 (4 boxes) = 7 -- not the shared
    # dict's own total of 10, and not identical between the two splits.
    assert train_ds.class_distribution == {0: 3}
    assert val_ds.class_distribution == {0: 7}


def test_auto_train_val_refuses_a_dataset_level_coco_misrouted_as_labels_dir(tmp_path, caplog):
    """split_construction.auto_train_val's own COCO-assembly branch must refuse the same shape
    build_dataset does, rather than silently training without validation on it: the refusal
    reaches the caller directly, never as a fallback build's own second raise behind a misleading
    "training without validation" warning."""
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    images = tmp_path / "images"
    labels = tmp_path / "detect"
    labels.mkdir()
    _make_images(images, ["img0", "img1"])
    (labels / "dataset.json").write_text(json.dumps(
        {"images": [{"id": 1, "file_name": "img0.jpg"}], "annotations": [], "categories": []}))

    with pytest.raises(ValueError, match="data.labels_dir="):
        auto_train_val("detection", {"images_dir": str(images), "labels_dir": str(labels),
                                      "subject": BUD}, None)
    assert "training without validation" not in caplog.text


def test_auto_train_val_raises_on_a_corrupt_explicit_validation_label(tmp_path):
    """A present, unreadable label under val_labels_dir must abort the run rather than silently
    training without validation over a document nobody can read."""
    from tcip_annotation.json_io import UnreadableLabelDocument, write_annotations
    from tcip_mcp.pipelines.data.split_construction import auto_train_val

    train_images = tmp_path / "images"
    train_labels = tmp_path / "labels"
    val_images = tmp_path / "val_images"
    val_labels = tmp_path / "val_labels"
    train_labels.mkdir()
    val_labels.mkdir()
    _make_images(train_images, ["img0"])
    _make_images(val_images, ["img1"])
    write_annotations(train_labels / "img0.json", [_box(1, 1, 10, 10)], 100, 100)
    (val_labels / "img1.json").write_bytes(b"{not json")

    with pytest.raises(UnreadableLabelDocument):
        auto_train_val("detection", {
            "images_dir": str(train_images), "labels_dir": str(train_labels),
            "val_images_dir": str(val_images), "val_labels_dir": str(val_labels),
            "subject": BUD,
        }, None)


# ── build_dataset auto-routes per-image JSON onto the COCO path ──────────────

def test_build_dataset_detection_autoresolves_json(tmp_path):
    from tcip_mcp.pipelines.data.datasets import build_dataset
    images = tmp_path / "images"
    labels = tmp_path / "detect"
    labels.mkdir()
    _make_images(images, ["img0"])
    json_io.write_annotations(labels / "img0.json", [_box(10, 10, 50, 50)], 100, 100)

    ds = build_dataset("detection", images_dir=str(images), labels_dir=str(labels), subject=BUD)
    assert ds.label_format == "coco"  # auto-routed
    _, target = ds[0]
    assert target["boxes"].shape == (1, 4)
    assert target["boxes"].tolist()[0] == pytest.approx([10, 10, 50, 50])
    assert target["labels"].tolist() == [1]  # 0-indexed cid 0 -> 1-indexed
    assert ds.class_distribution == {0: 1}


def test_build_dataset_refuses_a_dataset_level_coco_misrouted_as_labels_dir(tmp_path):
    """A dataset-level COCO file sitting in labels_dir must not be assembled from per-image files
    that are not there; the refusal names the offending file and both remedies (move it out, or
    point data.coco_json at it), since only the breeder knows which is this dataset's real label
    source."""
    from tcip_mcp.pipelines.data.datasets import build_dataset
    images = tmp_path / "images"
    labels = tmp_path / "detect"
    labels.mkdir()
    _make_images(images, ["img0"])
    (labels / "dataset.json").write_text(json.dumps(
        {"images": [{"id": 1, "file_name": "img0.jpg"}], "annotations": [], "categories": []}))

    with pytest.raises(ValueError, match="coco_json") as excinfo:
        build_dataset("detection", images_dir=str(images), labels_dir=str(labels), subject=BUD)
    assert "dataset.json" in str(excinfo.value)
    assert "move it out" in str(excinfo.value)


def test_autoresolve_json_labels_no_ops_without_an_images_dir(tmp_path):
    """The json branch already required images_dir before assembling; the coco-detection raise
    must be guarded the same way, not fire against a labels_dir-only call that has nothing to
    assemble against either way."""
    from tcip_mcp.pipelines.data.datasets import _autoresolve_json_labels

    labels = tmp_path / "detect"
    labels.mkdir()
    (labels / "dataset.json").write_text(json.dumps(
        {"images": [{"id": 1, "file_name": "img0.jpg"}], "annotations": [], "categories": []}))

    kwargs = {"labels_dir": str(labels), "images_dir": ""}
    _autoresolve_json_labels(kwargs, subject=BUD, attribute=None, id_map={BUD: 0}, date=None)
    assert "coco_data" not in kwargs and "label_format" not in kwargs


def test_build_dataset_respects_explicit_format(tmp_path):
    """An explicit label_format is never overridden by auto-resolve."""
    from tcip_mcp.pipelines.data.datasets import build_dataset
    images = tmp_path / "images"
    labels = tmp_path / "detect"
    labels.mkdir()
    _make_images(images, ["img0"])
    json_io.write_annotations(labels / "img0.json", [_box(10, 10, 50, 50)], 100, 100)

    ds = build_dataset("detection", images_dir=str(images), labels_dir=str(labels),
                       subject=BUD, label_format="yolo")
    assert ds.label_format == "yolo"  # honored, even though .json is present


def test_build_dataset_instance_seg_autoresolves_json(tmp_path):
    from tcip_mcp.pipelines.data.datasets import build_dataset
    images = tmp_path / "images"
    labels = tmp_path / "segment"
    labels.mkdir()
    _make_images(images, ["img0"])
    json_io.write_annotations(
        labels / "img0.json",
        [_poly([(10, 10), (50, 10), (50, 50), (10, 50)])], 100, 100)

    ds = build_dataset("instance_seg", images_dir=str(images), labels_dir=str(labels), subject=BUD)
    assert ds.label_format == "coco"
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


@pytest.mark.parametrize("via", ["coco", "json"])
def test_instance_seg_rasterizes_a_two_ring_instance_into_one_mask(tmp_path, via):
    """An occlusion-split instance (a bud behind a branch) is one object with two regions. Both
    label paths must rasterize every ring into that instance's single mask."""
    from tcip_mcp.pipelines.data.datasets import InstanceSegDataset
    from tcip_mcp.pipelines.data.label_queries import assemble_coco

    images, labels = tmp_path / "images", tmp_path / "annotations"
    labels.mkdir(parents=True)
    _make_images(images, ["img0"])
    json_io.write_annotations(labels / "img0.json", [_multi_poly([LOBE_A, LOBE_B])], 100, 100)

    kwargs = {"subject": BUD}
    if via == "coco":
        _reg, id_map = _reg_id_map()
        kwargs["coco_data"] = assemble_coco(labels, images, subject=BUD, date=None, id_map=id_map)
        kwargs["id_map"] = id_map

    ds = InstanceSegDataset(str(images), str(labels), **kwargs)
    _, target = ds[0]
    _assert_one_mask_over_both_lobes(target)


def test_instance_seg_two_single_ring_instances_stay_two_masks(tmp_path):
    """The rail admits the ordinary case too: the same two lobes authored as separate annotations are
    two instances with two masks; multi-ring support must not merge distinct objects."""
    from tcip_mcp.pipelines.data.datasets import InstanceSegDataset

    images, labels = tmp_path / "images", tmp_path / "annotations"
    labels.mkdir(parents=True)
    _make_images(images, ["img0"])
    json_io.write_annotations(
        labels / "img0.json", [_poly(LOBE_A), _poly(LOBE_B)], 100, 100)

    ds = InstanceSegDataset(str(images), str(labels), subject=BUD)
    _, target = ds[0]
    assert target["masks"].shape[0] == 2
    assert target["boxes"].tolist() == [[10.0, 10.0, 30.0, 30.0], [60.0, 10.0, 80.0, 30.0]]


def test_assemble_coco_keeps_both_rings_of_an_instance(tmp_path):
    """Training assembly carries the whole instance into COCO: one annotation, two segmentation
    rings, and a box over their union."""
    from tcip_mcp.pipelines.data.label_queries import assemble_coco

    images, labels = tmp_path / "images", tmp_path / "annotations"
    labels.mkdir(parents=True)
    _make_images(images, ["img0"])
    json_io.write_annotations(labels / "img0.json", [_multi_poly([LOBE_A, LOBE_B])], 100, 100)

    _reg, id_map = _reg_id_map()
    coco = assemble_coco(labels, images, subject=BUD, date=None, id_map=id_map)
    (ann,) = coco["annotations"]
    assert ann["segmentation"] == [
        [10.0, 10.0, 30.0, 10.0, 30.0, 30.0, 10.0, 30.0],
        [60.0, 10.0, 80.0, 10.0, 80.0, 30.0, 60.0, 30.0],
    ]
    assert ann["bbox"] == [10.0, 10.0, 70.0, 20.0]


def test_count_label_lines_reads_json_objects(tmp_path):
    from tcip_mcp.pipelines.data.splits import count_label_lines
    labels = tmp_path / "detect"
    labels.mkdir()
    json_io.write_annotations(labels / "a.json", [_box(0, 0, 10, 10), _box(0, 0, 20, 20)], 100, 100)
    json_io.write_annotations(labels / "neg.json", [], 100, 100, keep_empty=True)
    assert count_label_lines(labels, "a") == 2
    assert count_label_lines(labels, "neg") == 0
    assert count_label_lines(labels, "missing") == 0


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


@pytest.mark.parametrize("label_format", [None, "json", "coco"])
def test_only_annotated_and_confirmed_negatives_train(tmp_path, label_format):
    """The rail is a property of the data, not of which kwargs the caller passed.

    Enumerating samples from images_dir served unannotated images as zero-box samples, so a
    project where the breeder labelled 30 of 400 trained on 370 images asserted to be empty.
    """
    from tcip_mcp.pipelines.data.datasets import build_dataset
    from tcip_mcp.pipelines.data.label_queries import assemble_coco

    images, labels = _rail_fixture(tmp_path)
    kwargs = {"images_dir": str(images), "labels_dir": str(labels), "subject": BUD}
    if label_format == "coco":
        _reg, id_map = _reg_id_map()
        kwargs["coco_data"] = assemble_coco(labels, images, subject=BUD, date=None, id_map=id_map)
        kwargs["label_format"] = "coco"
    elif label_format:
        kwargs["label_format"] = label_format

    ds = build_dataset("detection", **kwargs)
    assert sorted(ds.stems) == ["ann", "neg"], ds.stems
    assert ds.sample_counts["annotated"] == 1
    assert ds.sample_counts["confirmed_negative"] == 1
    assert ds.sample_counts["skipped_unannotated"] >= 1


def test_a_corrupt_confirmed_negative_refuses_the_direct_json_loader(tmp_path):
    """A stored 'negative' whose label document is corrupt must never train as a zero-object
    negative: the direct-JSON path's own stem scan raises rather than reading the file as empty."""
    from tcip_annotation.json_io import UnreadableLabelDocument
    from tcip_mcp.pipelines.data.datasets import build_dataset

    images, labels = _rail_fixture(tmp_path)
    (labels / "neg.json").write_text("not json {][", encoding="utf-8")

    with pytest.raises(UnreadableLabelDocument):
        build_dataset("detection", images_dir=str(images), labels_dir=str(labels),
                     subject=BUD, label_format="json")


def test_caller_supplied_stems_are_filtered_too(tmp_path):
    """Split stems go through the same gate, otherwise the split reintroduces the fabrications."""
    from tcip_mcp.pipelines.data.datasets import build_dataset

    images, labels = _rail_fixture(tmp_path)
    ds = build_dataset("detection", images_dir=str(images), labels_dir=str(labels),
                       subject=BUD, stems=["ann", "empty", "nolabel", "neg"])
    assert sorted(ds.stems) == ["ann", "neg"]


def test_no_trainable_samples_raises_rather_than_training_on_nothing(tmp_path):
    from tcip_mcp.pipelines.data.datasets import build_dataset

    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    labels.mkdir(parents=True)
    _make_images(images, ["a", "b"])
    json_io.write_annotations(labels / "a.json", [], 100, 100, keep_empty=True)  # unconfirmed empty

    with pytest.raises(ValueError, match="no trainable samples"):
        build_dataset("detection", images_dir=str(images), labels_dir=str(labels), subject=BUD)


def test_instance_seg_applies_the_same_rail(tmp_path):
    from tcip_mcp.pipelines.data.datasets import build_dataset

    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    labels.mkdir(parents=True)
    _make_images(images, ["ann", "nolabel"])
    json_io.write_annotations(labels / "ann.json",
                              [_poly([(4, 4), (12, 4), (12, 12), (4, 12)])], 100, 100,
                              keep_empty=True)

    ds = build_dataset("instance_seg", images_dir=str(images), labels_dir=str(labels),
                       subject=BUD)
    assert ds.stems == ["ann"]


def test_instance_seg_dataset_excludes_partially_labeled_stem_from_training(tmp_path):
    """InstanceSegDataset's trainable_stems call must thread attribute/id_map through, the same as
    DetectionDataset's identical call, or the direct-JSON instance_seg path has no
    attribute-completeness rail: an image with any instance never assessed for `attribute` would
    train on its labeled subset instead of being held out whole."""
    from tcip_mcp.pipelines.data.datasets import InstanceSegDataset

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    _make_images(images_dir, ["complete", "partial"])
    labels_dir.mkdir()
    json_io.write_annotations(labels_dir / "complete.json", [
        _poly([(4, 4), (12, 4), (12, 12), (4, 12)], opening="open"),
    ], 100, 100)
    json_io.write_annotations(labels_dir / "partial.json", [
        _poly([(4, 4), (12, 4), (12, 12), (4, 12)], opening="closed"),
        _poly([(40, 40), (60, 40), (60, 60), (40, 60)]),  # unlabeled -- no opening attribute
    ], 100, 100)
    _reg, id_map = _reg_id_map(attribute="opening", values=("open", "closed"))

    ds = InstanceSegDataset(str(images_dir), str(labels_dir), subject=BUD,
                            attribute="opening", id_map=id_map)

    assert ds.stems == ["complete"]
    assert ds.sample_counts["skipped_incomplete_attribute"] == 1
    assert ds.sample_counts["annotated"] == 1


@pytest.mark.parametrize("ext", [".jpg", ".JPG"])
def test_confirmed_negative_survives_an_uppercase_extension(tmp_path, ext):
    """The name compared against the status store must be the real one on disk.

    `Path.exists()` is case-insensitive on Windows and macOS, so probing constructed paths returns
    the name that was built, not the one on disk. The status store is keyed on the real filename,
    so a fabricated name matches nothing and every human-confirmed negative is silently dropped,
    including the review loop's hard negatives. `IMG_*.JPG` is this repo's canonical camera name.
    """
    from PIL import Image as _Image

    from tcip_mcp.pipelines.data.datasets import build_dataset
    from tcip_mcp.pipelines.data.label_queries import assemble_coco

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

    _reg, id_map = _reg_id_map()
    # The assembled COCO carries the real names, so to_coco_dataset can match the store.
    coco = assemble_coco(labels, images, subject=BUD, date=None, id_map=id_map)
    assert {im["file_name"] for im in coco["images"]} == {f"IMG_0001{ext}", f"IMG_0002{ext}"}

    ds = build_dataset("detection", images_dir=str(images), labels_dir=str(labels), subject=BUD)
    assert sorted(ds.stems) == ["IMG_0001", "IMG_0002"], ds.stems
    assert ds.sample_counts["confirmed_negative"] == 1


def test_semantic_seg_requires_a_mask_but_admits_an_all_background_one(tmp_path):
    """Existence is the whole rail for masks: an all-background mask is a real annotation."""
    import numpy as np
    from PIL import Image as _Image

    from tcip_mcp.pipelines.data.datasets import build_dataset

    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    for stem in ("has_mask", "all_background", "no_mask"):
        _Image.new("RGB", (32, 32)).save(images / f"{stem}.jpg")
    _Image.fromarray(np.ones((32, 32), dtype=np.uint8)).save(masks / "has_mask.png")
    _Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(masks / "all_background.png")

    ds = build_dataset("semantic_seg", images_dir=str(images), masks_dir=str(masks), num_classes=2)
    assert sorted(ds.stems) == ["all_background", "has_mask"]  # no_mask dropped, empty mask kept


def test_sample_counts_distinguish_unannotated_from_unconfirmed_empty(tmp_path):
    """"Annotate this" and "confirm this empty one" are different jobs: the count must say which."""
    from tcip_mcp.pipelines.data.datasets import build_dataset

    images, labels = _rail_fixture(tmp_path)
    ds = build_dataset("detection", images_dir=str(images), labels_dir=str(labels), subject=BUD)
    assert ds.sample_counts == {"annotated": 1, "confirmed_negative": 1, "skipped_unannotated": 1,
                                "skipped_unconfirmed_empty": 1, "skipped_incomplete_attribute": 0,
                                "quarantined_stale_definition": 0}


def test_external_coco_zero_annotation_image_still_needs_a_human_complete(tmp_path):
    """An externally supplied COCO never passed through assemble_coco, so its zero-annotation
    images are not confirmed negatives: inferring that from the file's shape alone is invalid."""
    from tcip_mcp.pipelines.data.datasets import build_dataset

    images, labels = _rail_fixture(tmp_path)
    external = {
        "images": [{"id": 1, "file_name": "ann.jpg"}, {"id": 2, "file_name": "empty.jpg"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [4, 4, 8, 8]}],
        "categories": [{"id": 1, "name": "c0"}],
    }
    ds = build_dataset("detection", images_dir=str(images), labels_dir=str(labels),
                       subject=BUD, coco_data=external, label_format="coco")
    assert ds.stems == ["ann"], "an unconfirmed zero-annotation COCO image must not train"
    assert ds.sample_counts["confirmed_negative"] == 0


def test_a_confirmation_does_not_leak_across_subjects(tmp_path):
    """A Complete is a statement about one trait. Re-applying it elsewhere trains an image full of
    bushes as containing no bushes."""
    from tcip_mcp.pipelines.data.datasets import build_dataset
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
    assert sorted(build_dataset("detection", images_dir=str(images), labels_dir=str(labels),
                                subject="bud").stems) == ["ann", "shared"]
    assert build_dataset("detection", images_dir=str(images), labels_dir=str(labels),
                         subject="bush").stems == ["ann"]


def test_unresolvable_subject_refuses_rather_than_dropping_negatives(tmp_path):
    """Silently returning nothing would discard every hard negative the review loop harvested.

    A flat ``labels/`` dir (the shape ``draw_splits(materialize=True)`` emits) can't name its
    subject from its path; the confirmations live dataset-native, a sibling of ``labels/``'s own
    resolved root, not found by walking arbitrarily far up an ancestor chain.
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


def test_split_tree_carries_its_confirmed_negatives(tmp_path):
    """draw_splits(materialize=True) emits {train,val,test}/labels, which cannot name its subject,
    so it must carry the confirmations rather than inherit them by accident."""
    from tcip_mcp.pipelines.data.label_queries import confirmed_negative_names
    from tcip_mcp.tools.data_tools import draw_splits

    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    labels.mkdir(parents=True)
    _make_images(images, [f"i{n:02d}" for n in range(10)])
    for n in range(10):
        stem = f"i{n:02d}"
        boxes = [] if n % 2 else [_box(4, 4, 12, 12)]
        json_io.write_annotations(labels / f"{stem}.json", boxes, 100, 100, keep_empty=True)
    record_image_statuses(tmp_path, status_bucket("bud", None),
                          {f"i{n:02d}.jpg": "negative" for n in range(1, 10, 2)},
                          recorded_by="user:breeder")

    out = tmp_path / "splits"
    draw_splits(str(tmp_path), output_path=str(out), materialize=True, subject="bud",
               train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)

    carried = set()
    for split in ("train", "val", "calibration"):
        d = out / split / "labels"
        if d.is_dir():
            carried |= confirmed_negative_names(d, subject="bud", date=None)
    assert carried, "the split tree lost every human-confirmed negative"


def test_split_tree_carries_a_quarantine_capable_stamp(tmp_path):
    """A split's carried negatives must get their own classes.json + digest stamp too; without it
    quarantine can never fire on a split tree."""
    import tcip_store as ts
    from tcip_mcp import class_registry
    from tcip_mcp.class_registry import write_registry
    from tcip_mcp.dataset_layout import image_status_digest_key
    from tcip_mcp.tools.data_tools import draw_splits

    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    labels.mkdir(parents=True)
    registry = ClassRegistry(subjects=(
        Subject(name="bud", attributes=(
            Attribute(name="opening", type="categorical", values=("closed", "open")),
        )),
    ))
    write_registry(tmp_path / "classes.json", registry)
    expected_digest = class_registry.attribute_schema_digest(registry, "bud")

    _make_images(images, [f"i{n:02d}" for n in range(10)])
    for n in range(10):
        stem = f"i{n:02d}"
        boxes = [] if n % 2 else [_box(4, 4, 12, 12)]
        json_io.write_annotations(labels / f"{stem}.json", boxes, 100, 100, keep_empty=True)
    neg_names = {f"i{n:02d}.jpg" for n in range(1, 10, 2)}
    record_image_statuses(tmp_path, status_bucket("bud", None),
                          dict.fromkeys(neg_names, "negative"), recorded_by="user:breeder")
    stamp_image_status_digests(tmp_path, status_bucket("bud", None), neg_names, expected_digest)

    out = tmp_path / "splits"
    draw_splits(str(tmp_path), output_path=str(out), materialize=True, subject="bud",
               train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)

    found = False
    for split in ("train", "val", "calibration"):
        split_root = out / split
        digest_key = image_status_digest_key(split_root)
        if not ts.exists(digest_key):
            continue
        stamps = ts.read(digest_key).get(status_bucket("bud", None), {})
        carried_here = set(stamps) & neg_names
        if carried_here:
            assert (split_root / "classes.json").is_file()
            assert all(stamps[n] == expected_digest for n in carried_here)
            found = True
    assert found, "no split carried both a negative and its schema stamp"


def test_quarantined_negative_reads_the_same_reason_on_both_label_paths(tmp_path):
    """The COCO-assembly branch of ``trainable_stems`` must check quarantine for an image
    ``assemble_coco`` had already dropped from ``images``, not only ``has_record``; otherwise a
    human-confirmed-but-schema-stale negative reads as ``skipped_unconfirmed_empty``
    ("nobody ever looked") there, while the direct-JSON branch on the identical fixture correctly
    reads ``quarantined_stale_definition`` ("looked, but the schema changed since"). Same image,
    same real reason, two different label paths must not disagree on which it was."""
    from tcip_mcp import class_registry
    from tcip_mcp.class_registry import write_registry
    from tcip_mcp.pipelines.data.label_queries import assemble_coco, trainable_stems

    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    labels.mkdir(parents=True)
    registry = ClassRegistry(subjects=(
        Subject(name=BUD, attributes=(
            Attribute(name="opening", type="categorical", values=("closed", "open")),
        )),
    ))
    write_registry(tmp_path / "classes.json", registry)
    current_digest = class_registry.attribute_schema_digest(registry, BUD)

    _make_images(images, ["a"])
    json_io.write_annotations(labels / "a.json", [], 100, 100, keep_empty=True)

    record_image_statuses(tmp_path, status_bucket(BUD, None), {"a.jpg": "negative"},
                          recorded_by="user:breeder")
    # Stamped with a digest that does not match the current schema -> quarantined, not trusted.
    assert current_digest != "stale-digest"
    stamp_image_status_digests(tmp_path, status_bucket(BUD, None), ["a.jpg"], "stale-digest")

    _, id_map = _reg_id_map()
    _, counts_json = trainable_stems(str(labels), str(images), subject=BUD, date=None)
    coco = assemble_coco(labels, images, subject=BUD, date=None, id_map=id_map)
    _, counts_coco = trainable_stems(str(labels), str(images), subject=BUD, date=None, coco=coco)

    assert counts_json["quarantined_stale_definition"] == 1
    assert counts_json == counts_coco


def test_a_stale_complete_confirmation_is_quarantined_on_both_label_paths(tmp_path):
    """A complete confirmation, unlike a negative, trained by its label file's real content alone
    before this quarantine: a bud image finished under a two-value attribute vocabulary that grew
    to three must be held out exactly as a stale negative already is, on both label paths, never
    admitted as ``annotated`` by the boxes it happens to carry."""
    from tcip_mcp import class_registry
    from tcip_mcp.class_registry import write_registry
    from tcip_mcp.pipelines.data.label_queries import (
        assemble_coco, require_samples, trainable_stems,
    )

    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    labels.mkdir(parents=True)
    registry = ClassRegistry(subjects=(
        Subject(name=BUD, attributes=(
            Attribute(name="opening", type="categorical", values=("closed", "open")),
        )),
    ))
    write_registry(tmp_path / "classes.json", registry)
    current_digest = class_registry.attribute_schema_digest(registry, BUD)

    _make_images(images, ["a"])
    json_io.write_annotations(labels / "a.json", [_box(4, 4, 12, 12)], 100, 100)

    record_image_statuses(tmp_path, status_bucket(BUD, None), {"a.jpg": "complete"},
                          recorded_by="user:breeder")
    assert current_digest != "stale-digest"
    stamp_image_status_digests(tmp_path, status_bucket(BUD, None), ["a.jpg"], "stale-digest")

    _, id_map = _reg_id_map()
    keep_json, counts_json = trainable_stems(str(labels), str(images), subject=BUD, date=None)
    coco = assemble_coco(labels, images, subject=BUD, date=None, id_map=id_map)
    keep_coco, counts_coco = trainable_stems(str(labels), str(images), subject=BUD, date=None,
                                             coco=coco)

    assert keep_json == [] and keep_coco == []
    assert counts_json["quarantined_stale_definition"] == 1
    assert counts_json["annotated"] == 0
    assert counts_json == counts_coco
    with pytest.raises(ValueError, match="quarantined"):
        require_samples(keep_json, counts_json, str(labels))


def test_a_reconfirmed_complete_trains_again_after_the_schema_change(tmp_path):
    """Re-confirming restamps the current digest, so the same image trains once a human has
    looked again under the vocabulary now in effect."""
    from tcip_mcp import class_registry
    from tcip_mcp.class_registry import write_registry
    from tcip_mcp.pipelines.data.label_queries import trainable_stems

    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    labels.mkdir(parents=True)
    registry = ClassRegistry(subjects=(
        Subject(name=BUD, attributes=(
            Attribute(name="opening", type="categorical", values=("closed", "open")),
        )),
    ))
    write_registry(tmp_path / "classes.json", registry)
    current_digest = class_registry.attribute_schema_digest(registry, BUD)

    _make_images(images, ["a"])
    json_io.write_annotations(labels / "a.json", [_box(4, 4, 12, 12)], 100, 100)
    record_image_statuses(tmp_path, status_bucket(BUD, None), {"a.jpg": "complete"},
                          recorded_by="user:breeder")
    stamp_image_status_digests(tmp_path, status_bucket(BUD, None), ["a.jpg"], current_digest)

    keep, counts = trainable_stems(str(labels), str(images), subject=BUD, date=None)
    assert keep == ["a"]
    assert counts["annotated"] == 1
    assert counts["quarantined_stale_definition"] == 0


def test_an_unstamped_complete_trains(tmp_path):
    """A complete confirmation the stamp transaction never reached is admitted, not quarantined:
    absence of a stamp is never evidence of staleness."""
    from tcip_mcp.class_registry import write_registry
    from tcip_mcp.pipelines.data.label_queries import trainable_stems

    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    labels.mkdir(parents=True)
    registry = ClassRegistry(subjects=(
        Subject(name=BUD, attributes=(
            Attribute(name="opening", type="categorical", values=("closed", "open")),
        )),
    ))
    write_registry(tmp_path / "classes.json", registry)

    _make_images(images, ["a"])
    json_io.write_annotations(labels / "a.json", [_box(4, 4, 12, 12)], 100, 100)
    record_image_statuses(tmp_path, status_bucket(BUD, None), {"a.jpg": "complete"},
                          recorded_by="user:breeder")

    keep, counts = trainable_stems(str(labels), str(images), subject=BUD, date=None)
    assert keep == ["a"]
    assert counts["quarantined_stale_definition"] == 0


def test_a_complete_under_an_unchanged_subject_trains(tmp_path):
    """Another subject's own schema change never quarantines a bucket the change had no part in:
    only bud's digest moves, so bush's complete, stamped under its own still-current digest,
    admits."""
    from tcip_mcp import class_registry
    from tcip_mcp.class_registry import write_registry
    from tcip_mcp.pipelines.data.label_queries import trainable_stems

    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    labels.mkdir(parents=True)
    registry = ClassRegistry(subjects=(
        Subject(name=BUD, attributes=(
            Attribute(name="opening", type="categorical", values=("closed", "open")),
        )),
        Subject(name="bush"),
    ))
    write_registry(tmp_path / "classes.json", registry)
    bush_digest = class_registry.attribute_schema_digest(registry, "bush")

    _make_images(images, ["a"])
    json_io.write_annotations(labels / "a.json", [_box(4, 4, 12, 12, subject="bush")], 100, 100)
    record_image_statuses(tmp_path, status_bucket("bush", None), {"a.jpg": "complete"},
                          recorded_by="user:breeder")
    stamp_image_status_digests(tmp_path, status_bucket("bush", None), ["a.jpg"], bush_digest)
    # bud's own schema changes; bush's bucket, and its stamp, must be untouched by it.
    stamp_image_status_digests(tmp_path, status_bucket(BUD, None), ["b.jpg"], "stale-digest")

    keep, counts = trainable_stems(str(labels), str(images), subject="bush", date=None)
    assert keep == ["a"]
    assert counts["quarantined_stale_definition"] == 0


def test_a_partial_carrying_a_stale_stamp_still_trains(tmp_path):
    """A partial is not a human's assertion (it carries no Complete), so a stamp on it, however
    stale, never quarantines: the quarantine is over finished statuses only."""
    from tcip_mcp.class_registry import write_registry
    from tcip_mcp.pipelines.data.label_queries import trainable_stems

    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    labels.mkdir(parents=True)
    registry = ClassRegistry(subjects=(
        Subject(name=BUD, attributes=(
            Attribute(name="opening", type="categorical", values=("closed", "open")),
        )),
    ))
    write_registry(tmp_path / "classes.json", registry)

    _make_images(images, ["a"])
    json_io.write_annotations(labels / "a.json", [_box(4, 4, 12, 12)], 100, 100)
    record_image_statuses(tmp_path, status_bucket(BUD, None), {"a.jpg": "partial"},
                          recorded_by="user:breeder")
    stamped = stamp_image_status_digests(
        tmp_path, status_bucket(BUD, None), ["a.jpg"], "stale-digest")
    assert stamped == ["a.jpg"]

    keep, counts = trainable_stems(str(labels), str(images), subject=BUD, date=None)
    assert keep == ["a"]
    assert counts["quarantined_stale_definition"] == 0


def test_a_stale_and_contradicted_negative_still_trains_by_content(tmp_path):
    """Real content contradicts a stored negative outright; staleness never overrides that, and
    the contradiction is still named for the caller to surface."""
    from tcip_mcp.class_registry import write_registry
    from tcip_mcp.pipelines.data.label_queries import trainable_stems

    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    labels.mkdir(parents=True)
    registry = ClassRegistry(subjects=(
        Subject(name=BUD, attributes=(
            Attribute(name="opening", type="categorical", values=("closed", "open")),
        )),
    ))
    write_registry(tmp_path / "classes.json", registry)

    _make_images(images, ["a"])
    # Recorded negative, but the label file now carries real content: a contradiction.
    json_io.write_annotations(labels / "a.json", [_box(4, 4, 12, 12)], 100, 100)
    record_image_statuses(tmp_path, status_bucket(BUD, None), {"a.jpg": "negative"},
                          recorded_by="user:breeder")
    stamp_image_status_digests(tmp_path, status_bucket(BUD, None), ["a.jpg"], "stale-digest")

    contradicted: set[str] = set()
    keep, counts = trainable_stems(str(labels), str(images), subject=BUD, date=None,
                                   contradicted_out=contradicted)
    assert keep == ["a"]
    assert counts["annotated"] == 1
    assert counts["quarantined_stale_definition"] == 0
    assert contradicted == {"a.jpg"}


def test_trainable_stems_with_subject_none_over_only_complete_statuses_does_not_refuse(tmp_path):
    """A tree holding only complete confirmations has no negative to lose, so an unthreaded
    subject does not refuse the way it would with a confirmed negative present (coverage)."""
    from tcip_mcp.pipelines.data.label_queries import trainable_stems

    images = tmp_path / "images"
    labels = tmp_path / "annotations"
    labels.mkdir(parents=True)
    _make_images(images, ["a"])
    json_io.write_annotations(labels / "a.json", [_box(4, 4, 12, 12)], 100, 100)
    record_image_statuses(tmp_path, status_bucket(BUD, None), {"a.jpg": "complete"},
                          recorded_by="user:breeder")

    keep, counts = trainable_stems(str(labels), str(images), subject=None, date=None)
    assert keep == ["a"]
    assert counts["annotated"] == 1


def test_json_det_targets_skips_unlabeled_instead_of_raising(tmp_path):
    """The loader's own per-image target reader must accept the same partially-attributed data
    to_coco_dataset does: an unlabeled instance is excluded, not a hard abort, while an
    undecodable value still raises."""
    from tcip_mcp.pipelines.data.label_queries import json_det_targets

    path = tmp_path / "IMG_A.json"
    json_io.write_annotations(path, [
        Annotation(subject="bud", geometry=BBox(10, 10, 30, 30),
                  attributes={"opening": "closed"}),
        Annotation(subject="bud", geometry=BBox(40, 40, 60, 60), attributes={}),  # unlabeled
    ], 100, 100)

    id_map = {"open": 0, "closed": 1}
    boxes, labels, n_unlabeled = json_det_targets(str(path), "bud", "opening", id_map)
    assert len(boxes) == 1 and labels == [2]  # 0-indexed 1 ("closed") + 1 for background
    assert n_unlabeled == 1  # the second instance, disclosed rather than silently dropped

    undecodable = tmp_path / "IMG_B.json"
    json_io.write_annotations(undecodable, [
        Annotation(subject="bud", geometry=BBox(10, 10, 30, 30),
                  attributes={"opening": "not-a-real-value"}),
    ], 100, 100)
    with pytest.raises(ValueError):
        json_det_targets(str(undecodable), "bud", "opening", id_map)


def test_detection_dataset_excludes_partially_labeled_stem_from_training(tmp_path):
    """DetectionDataset's fixed-length self.stems must not include a stem with any instance
    unlabeled for `attribute` -- __getitem__ can't act on this per-call (the dataset length is
    fixed at construction), so the exclusion has to happen here, matching the delivery-gating
    paths (run_full_frame_evaluation, operating-point calibration) that already exclude the whole
    image rather than silently training on its labeled subset."""
    from tcip_mcp.pipelines.data.datasets import DetectionDataset

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    _make_images(images_dir, ["complete", "partial"])
    labels_dir.mkdir()
    json_io.write_annotations(labels_dir / "complete.json", [
        _box(10, 10, 30, 30, opening="open"),
    ], 100, 100)
    json_io.write_annotations(labels_dir / "partial.json", [
        _box(10, 10, 30, 30, opening="closed"),
        _box(40, 40, 60, 60),  # unlabeled -- no opening attribute at all
    ], 100, 100)
    _reg, id_map = _reg_id_map(attribute="opening", values=("open", "closed"))

    ds = DetectionDataset(str(images_dir), str(labels_dir), subject=BUD,
                          attribute="opening", id_map=id_map)

    assert ds.stems == ["complete"]
    # The drop must be recorded in the partition's own counts, under its real reason: filtering
    # this category-keyed dict by stem name would wipe it to {} and make the all-excluded case
    # die on a bare KeyError.
    assert ds.sample_counts["skipped_incomplete_attribute"] == 1
    assert ds.sample_counts["annotated"] == 1
    assert ds.sample_counts["skipped_unconfirmed_empty"] == 0  # not a false reason
    # The surviving stem's own target read is unaffected -- one real box, fully labeled.
    boxes, labels = ds._det_targets("complete", "")
    assert len(boxes) == 1


def test_detection_dataset_excludes_incomplete_attribute_on_the_real_build_dataset_path(tmp_path):
    """The exclusion above must also fire on the path build_dataset actually takes: build_dataset
    assembles an in-memory COCO and passes it as coco_data, which forces label_format='coco'. A
    check guarded only by label_format == 'json' is inert here, and the dropped image would be
    reported under the false reason 'skipped_unconfirmed_empty'."""
    from tcip_mcp.class_registry import Attribute, ClassRegistry, Subject, write_registry
    from tcip_mcp.pipelines.data.datasets import build_dataset

    root = tmp_path / "ds"
    images_dir, labels_dir = root / "images", root / "annotations"
    _make_images(images_dir, ["complete_a", "complete_b", "partial"])
    labels_dir.mkdir(parents=True)
    write_registry(root / "classes.json", ClassRegistry(subjects=(
        Subject(name=BUD, attributes=(
            Attribute(name="opening", type="categorical", values=("open", "closed")),)),)))
    for stem in ("complete_a", "complete_b"):
        json_io.write_annotations(labels_dir / f"{stem}.json", [
            _box(10, 10, 30, 30, opening="open")], 100, 100)
    json_io.write_annotations(labels_dir / "partial.json", [
        _box(10, 10, 30, 30, opening="closed"),
        _box(40, 40, 60, 60),  # unlabeled
    ], 100, 100)

    built = build_dataset(task="detection", images_dir=str(images_dir), labels_dir=str(labels_dir),
                          subject=BUD, attribute="opening")
    ds = built.dataset if hasattr(built, "dataset") else built

    assert ds.label_format == "coco"  # build_dataset forces the coco path, not the json-guarded one
    assert "partial" not in ds.stems
    assert ds.sample_counts["skipped_incomplete_attribute"] == 1
    assert ds.sample_counts["skipped_unconfirmed_empty"] == 0


def test_tiled_detection_indexes_no_tile_from_an_attribute_incomplete_image(tmp_path):
    """The tiler expands the base dataset's admitted stems into one training sample per tile, so an
    image the attribute-completeness rail held out must contribute no tile at all. Asserting on the
    tile index, not on the base's stems, is what pins that: a tile carrying the image's real but
    unlabeled objects would train them as background, one tile at a time."""
    from tcip_mcp.class_registry import write_registry
    from tcip_mcp.pipelines.data.datasets import build_dataset

    root = tmp_path / "ds"
    images_dir, labels_dir = root / "images", root / "annotations"
    _make_images(images_dir, ["complete", "partial"])
    labels_dir.mkdir(parents=True)
    reg, _id_map = _reg_id_map(attribute="opening", values=("open", "closed"))
    write_registry(root / "classes.json", reg)
    json_io.write_annotations(labels_dir / "complete.json", [
        _box(10, 10, 30, 30, opening="open")], 100, 100)
    json_io.write_annotations(labels_dir / "partial.json", [
        _box(10, 10, 30, 30, opening="closed"),
        _box(40, 40, 60, 60),  # unlabeled
    ], 100, 100)

    tiled = build_dataset(task="detection", images_dir=str(images_dir), labels_dir=str(labels_dir),
                          subject=BUD, attribute="opening",
                          tiling={"enabled": True, "tile_size": 64, "overlap": 0.0})

    assert set(tiled.stems) == {"complete"}  # tiles are per-index, so this is every tile's source
    assert len(tiled) > 0  # the rail admits the fully-attributed image's tiles
