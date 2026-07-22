"""COCO-training-assembly: training/eval reads the canonical per-image JSON label store by
assembling a dataset-level COCO on the fly (``build_dataset`` auto-routes a JSON label dir onto
the ``label_format='coco'`` path). Covers detection + instance-seg, format detection, and the
stratification line-count that must see JSON objects (not only YOLO ``.txt``)."""

import json

import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import BBox, Polygon  # noqa: E402


def _make_images(images_dir, stems):
    images_dir.mkdir(parents=True, exist_ok=True)
    for s in stems:
        Image.new("RGB", (100, 100)).save(images_dir / f"{s}.jpg")


# ── format detection ────────────────────────────────────────────────────────

def test_dir_label_format_detects_json(tmp_path):
    from tcip_mcp.pipelines.data.datasets import dir_label_format
    d = tmp_path / "detect"
    d.mkdir()
    json_io.write_detect(d / "a.json", [BBox(10, 10, 50, 50, 0)], 100, 100)
    assert dir_label_format(d) == "json"


def test_dir_label_format_ignores_non_canonical_json(tmp_path):
    """A LabelMe-style .json (``shapes``, no ``objects``) is not the canonical store — don't claim
    it, so it never gets silently assembled as though it were per-image JSON."""
    from tcip_mcp.pipelines.data.datasets import dir_label_format
    d = tmp_path / "labels"
    d.mkdir()
    (d / "a.json").write_text(json.dumps({"shapes": [{"label": "x", "points": [[1, 1], [2, 2]]}]}))
    assert dir_label_format(d) is None


def test_dir_label_format_empty(tmp_path):
    from tcip_mcp.pipelines.data.datasets import dir_label_format
    assert dir_label_format(tmp_path / "nope") is None


# ── assemble_coco ───────────────────────────────────────────────────────────

def test_assemble_coco_pairs_labels_with_images(tmp_path):
    from tcip_mcp.pipelines.data.datasets import assemble_coco
    images = tmp_path / "images"
    labels = tmp_path / "detect"
    labels.mkdir()
    _make_images(images, ["img0", "img1"])
    json_io.write_detect(labels / "img0.json", [BBox(10, 10, 50, 50, 0), BBox(0, 0, 20, 20, 1)], 100, 100)
    json_io.write_detect(labels / "img1.json", [], 100, 100, keep_empty=True)  # confirmed negative

    coco = assemble_coco(labels, images)
    # img1's empty file is NOT human-confirmed negative -> excluded (treated as unannotated)
    assert {im["file_name"] for im in coco["images"]} == {"img0.jpg"}
    assert len(coco["annotations"]) == 2
    assert {a["category_id"] for a in coco["annotations"]} == {0, 1}


def test_assemble_coco_includes_only_confirmed_negatives(tmp_path):
    import json as _json

    from tcip_mcp.pipelines.data.datasets import assemble_coco
    images = tmp_path / "images"
    labels = tmp_path / "annotations" / "default" / "detect"
    labels.mkdir(parents=True)
    _make_images(images, ["img0", "img1"])
    json_io.write_detect(labels / "img0.json", [], 100, 100, keep_empty=True)  # confirmed below
    json_io.write_detect(labels / "img1.json", [], 100, 100, keep_empty=True)  # emptied mid-work
    state = tmp_path / ".tcip" / "state"
    state.mkdir(parents=True)
    (state / "image_status.json").write_text(_json.dumps(
        {"default": {"img0.jpg": "negative", "img1.jpg": "partial"}}))
    coco = assemble_coco(labels, images)
    assert {im["file_name"] for im in coco["images"]} == {"img0.jpg"}  # human-confirmed only


def test_assemble_coco_skips_stem_without_image(tmp_path):
    from tcip_mcp.pipelines.data.datasets import assemble_coco
    images = tmp_path / "images"
    labels = tmp_path / "detect"
    labels.mkdir()
    _make_images(images, ["img0"])
    json_io.write_detect(labels / "img0.json", [BBox(10, 10, 50, 50, 0)], 100, 100)
    json_io.write_detect(labels / "orphan.json", [BBox(5, 5, 9, 9, 0)], 100, 100)  # no image

    coco = assemble_coco(labels, images)
    assert [im["file_name"] for im in coco["images"]] == ["img0.jpg"]


# ── build_dataset auto-routes per-image JSON onto the COCO path ──────────────

def test_build_dataset_detection_autoresolves_json(tmp_path):
    from tcip_mcp.pipelines.data.datasets import build_dataset
    images = tmp_path / "images"
    labels = tmp_path / "detect"
    labels.mkdir()
    _make_images(images, ["img0"])
    json_io.write_detect(labels / "img0.json", [BBox(10, 10, 50, 50, 0)], 100, 100)

    ds = build_dataset("detection", images_dir=str(images), labels_dir=str(labels), num_classes=1)
    assert ds.label_format == "coco"  # auto-routed
    _, target = ds[0]
    assert target["boxes"].shape == (1, 4)
    assert target["boxes"].tolist()[0] == pytest.approx([10, 10, 50, 50])
    assert target["labels"].tolist() == [1]  # 0-indexed cid 0 -> 1-indexed
    assert ds.class_distribution == {0: 1}


def test_build_dataset_respects_explicit_format(tmp_path):
    """An explicit label_format is never overridden by auto-resolve."""
    from tcip_mcp.pipelines.data.datasets import build_dataset
    images = tmp_path / "images"
    labels = tmp_path / "detect"
    labels.mkdir()
    _make_images(images, ["img0"])
    json_io.write_detect(labels / "img0.json", [BBox(10, 10, 50, 50, 0)], 100, 100)

    ds = build_dataset("detection", images_dir=str(images), labels_dir=str(labels),
                       num_classes=1, label_format="yolo")
    assert ds.label_format == "yolo"  # honored, even though .json is present


def test_build_dataset_instance_seg_autoresolves_json(tmp_path):
    from tcip_mcp.pipelines.data.datasets import build_dataset
    images = tmp_path / "images"
    labels = tmp_path / "segment"
    labels.mkdir()
    _make_images(images, ["img0"])
    json_io.write_segment(
        labels / "img0.json",
        [Polygon([(10, 10), (50, 10), (50, 50), (10, 50)], 0)], 100, 100)

    ds = build_dataset("instance_seg", images_dir=str(images), labels_dir=str(labels), num_classes=1)
    assert ds.label_format == "coco"
    _, target = ds[0]
    assert target["boxes"].shape == (1, 4)
    assert target["masks"].shape[0] == 1
    assert target["labels"].tolist() == [1]
    assert int(target["masks"].sum()) > 0  # polygon rasterized to a non-empty mask


def test_count_label_lines_reads_json_objects(tmp_path):
    from tcip_mcp.pipelines.data.splits import count_label_lines
    labels = tmp_path / "detect"
    labels.mkdir()
    json_io.write_detect(labels / "a.json", [BBox(0, 0, 10, 10, 0), BBox(0, 0, 20, 20, 0)], 100, 100)
    json_io.write_detect(labels / "neg.json", [], 100, 100, keep_empty=True)
    assert count_label_lines(labels, "a") == 2
    assert count_label_lines(labels, "neg") == 0
    assert count_label_lines(labels, "missing") == 0


def _rail_fixture(tmp_path):
    """One annotated image, one empty-unconfirmed, one with no label file, one confirmed negative."""
    images = tmp_path / "images"
    labels = tmp_path / "annotations" / "default" / "detect"
    labels.mkdir(parents=True)
    _make_images(images, ["ann", "empty", "nolabel", "neg"])
    json_io.write_detect(labels / "ann.json", [BBox(4, 4, 12, 12, 0)], 100, 100, keep_empty=True)
    json_io.write_detect(labels / "empty.json", [], 100, 100, keep_empty=True)
    json_io.write_detect(labels / "neg.json", [], 100, 100, keep_empty=True)
    state = tmp_path / ".tcip" / "state"
    state.mkdir(parents=True)
    (state / "image_status.json").write_text(
        json.dumps({"default": {"neg.jpg": "negative"}}))
    return images, labels


@pytest.mark.parametrize("label_format", [None, "json", "coco"])
def test_only_annotated_and_confirmed_negatives_train(tmp_path, label_format):
    """The rail is a property of the data, not of which kwargs the caller passed.

    Enumerating samples from images_dir served unannotated images as zero-box samples, so a
    project where the breeder labelled 30 of 400 trained on 370 images asserted to be empty.
    """
    from tcip_mcp.pipelines.data.datasets import assemble_coco, build_dataset

    images, labels = _rail_fixture(tmp_path)
    kwargs = {"images_dir": str(images), "labels_dir": str(labels), "num_classes": 1}
    if label_format == "coco":
        kwargs["coco_data"] = assemble_coco(labels, images)
        kwargs["label_format"] = "coco"
    elif label_format:
        kwargs["label_format"] = label_format

    ds = build_dataset("detection", **kwargs)
    assert sorted(ds.stems) == ["ann", "neg"], ds.stems
    assert ds.sample_counts["annotated"] == 1
    assert ds.sample_counts["confirmed_negative"] == 1
    assert ds.sample_counts["skipped_unannotated"] >= 1


def test_caller_supplied_stems_are_filtered_too(tmp_path):
    """Split stems go through the same gate — otherwise the split reintroduces the fabrications."""
    from tcip_mcp.pipelines.data.datasets import build_dataset

    images, labels = _rail_fixture(tmp_path)
    ds = build_dataset("detection", images_dir=str(images), labels_dir=str(labels),
                       num_classes=1, stems=["ann", "empty", "nolabel", "neg"])
    assert sorted(ds.stems) == ["ann", "neg"]


def test_no_trainable_samples_raises_rather_than_training_on_nothing(tmp_path):
    from tcip_mcp.pipelines.data.datasets import build_dataset

    images = tmp_path / "images"
    labels = tmp_path / "annotations" / "default" / "detect"
    labels.mkdir(parents=True)
    _make_images(images, ["a", "b"])
    json_io.write_detect(labels / "a.json", [], 100, 100, keep_empty=True)  # unconfirmed empty

    with pytest.raises(ValueError, match="no trainable samples"):
        build_dataset("detection", images_dir=str(images), labels_dir=str(labels), num_classes=1)


def test_instance_seg_applies_the_same_rail(tmp_path):
    from tcip_mcp.pipelines.data.datasets import build_dataset

    images = tmp_path / "images"
    labels = tmp_path / "annotations" / "default" / "segment"
    labels.mkdir(parents=True)
    _make_images(images, ["ann", "nolabel"])
    json_io.write_segment(labels / "ann.json",
                          [Polygon([(4, 4), (12, 4), (12, 12), (4, 12)], 0)], 100, 100,
                          keep_empty=True)

    ds = build_dataset("instance_seg", images_dir=str(images), labels_dir=str(labels),
                       num_classes=1)
    assert ds.stems == ["ann"]


@pytest.mark.parametrize("ext", [".jpg", ".JPG"])
def test_confirmed_negative_survives_an_uppercase_extension(tmp_path, ext):
    """The name compared against the status store must be the real one on disk.

    `Path.exists()` is case-insensitive on Windows and macOS, so probing constructed paths returns
    the name that was built, not the one on disk. The status store is keyed on the real filename,
    so a fabricated name matches nothing and every human-confirmed negative is silently dropped —
    including the review loop's hard negatives. `IMG_*.JPG` is this repo's canonical camera name.
    """
    from PIL import Image as _Image

    from tcip_mcp.pipelines.data.datasets import assemble_coco, build_dataset

    images = tmp_path / "images"
    labels = tmp_path / "annotations" / "default" / "detect"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for stem in ("IMG_0001", "IMG_0002"):
        _Image.new("RGB", (100, 100)).save(images / f"{stem}{ext}")
    json_io.write_detect(labels / "IMG_0001.json", [BBox(4, 4, 12, 12, 0)], 100, 100,
                         keep_empty=True)
    json_io.write_detect(labels / "IMG_0002.json", [], 100, 100, keep_empty=True)
    state = tmp_path / ".tcip" / "state"
    state.mkdir(parents=True)
    (state / "image_status.json").write_text(
        json.dumps({"default": {f"IMG_0002{ext}": "negative"}}))

    # The assembled COCO carries the real names, so to_coco_dataset can match the store.
    coco = assemble_coco(labels, images)
    assert {im["file_name"] for im in coco["images"]} == {f"IMG_0001{ext}", f"IMG_0002{ext}"}

    ds = build_dataset("detection", images_dir=str(images), labels_dir=str(labels), num_classes=1)
    assert sorted(ds.stems) == ["IMG_0001", "IMG_0002"], ds.stems
    assert ds.sample_counts["confirmed_negative"] == 1


def test_semantic_seg_requires_a_mask_but_admits_an_all_background_one(tmp_path):
    """Existence is the whole rail for masks — an all-background mask is a real annotation."""
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
    """"Annotate this" and "confirm this empty one" are different jobs — the count must say which."""
    from tcip_mcp.pipelines.data.datasets import build_dataset

    images, labels = _rail_fixture(tmp_path)
    ds = build_dataset("detection", images_dir=str(images), labels_dir=str(labels), num_classes=1)
    assert ds.sample_counts == {"annotated": 1, "confirmed_negative": 1,
                                "skipped_unannotated": 1, "skipped_unconfirmed_empty": 1}


def test_external_coco_zero_annotation_image_still_needs_a_human_complete(tmp_path):
    """An externally supplied COCO never passed through assemble_coco, so its zero-annotation
    images are not confirmed negatives — inferring that from the file's shape is the K13 bug."""
    from tcip_mcp.pipelines.data.datasets import build_dataset

    images, labels = _rail_fixture(tmp_path)
    external = {
        "images": [{"id": 1, "file_name": "ann.jpg"}, {"id": 2, "file_name": "empty.jpg"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [4, 4, 8, 8]}],
        "categories": [{"id": 1, "name": "c0"}],
    }
    ds = build_dataset("detection", images_dir=str(images), labels_dir=str(labels),
                       num_classes=1, coco_data=external, label_format="coco")
    assert ds.stems == ["ann"], "an unconfirmed zero-annotation COCO image must not train"
    assert ds.sample_counts["confirmed_negative"] == 0


def test_a_confirmation_does_not_leak_across_trait_campaigns(tmp_path):
    """A Complete is a statement about one trait. Re-applying it elsewhere trains an image full of
    bushes as containing no bushes."""
    from tcip_mcp.pipelines.data.datasets import build_dataset, confirmed_negative_names

    images = tmp_path / "images"
    catkin = tmp_path / "annotations" / "catkin" / "detect"
    bush = tmp_path / "annotations" / "bush" / "detect"
    catkin.mkdir(parents=True)
    bush.mkdir(parents=True)
    _make_images(images, ["ann", "shared"])
    for d in (catkin, bush):
        json_io.write_detect(d / "ann.json", [BBox(4, 4, 12, 12, 0)], 100, 100, keep_empty=True)
        json_io.write_detect(d / "shared.json", [], 100, 100, keep_empty=True)
    state = tmp_path / ".tcip" / "state"
    state.mkdir(parents=True)
    # Confirmed negative for catkin only — the breeder never judged it for bush.
    (state / "image_status.json").write_text(json.dumps({"catkin": {"shared.jpg": "negative"}}))

    assert confirmed_negative_names(catkin) == {"shared.jpg"}
    assert confirmed_negative_names(bush) == set()
    assert sorted(build_dataset("detection", images_dir=str(images), labels_dir=str(catkin),
                                num_classes=1).stems) == ["ann", "shared"]
    assert build_dataset("detection", images_dir=str(images), labels_dir=str(bush),
                         num_classes=1).stems == ["ann"]


def test_unresolvable_campaign_refuses_rather_than_dropping_negatives(tmp_path):
    """Silently returning nothing would discard every hard negative the review loop harvested."""
    from tcip_mcp.pipelines.data.datasets import confirmed_negative_names

    labels = tmp_path / "some" / "flat" / "labels"
    labels.mkdir(parents=True)
    state = tmp_path / ".tcip" / "state"
    state.mkdir(parents=True)
    (state / "image_status.json").write_text(json.dumps({"catkin": {"a.jpg": "negative"}}))

    with pytest.raises(ValueError, match="cannot tell which subject"):
        confirmed_negative_names(labels)


def test_a_derived_tree_without_negatives_does_not_refuse(tmp_path):
    """Refuse only when there is something to lose.

    A split or curated export cannot name its subject. Raising there would block the platform's
    own documented split -> train path; with no confirmed negatives in the project there is nothing
    to drop, so it must proceed.
    """
    from tcip_mcp.pipelines.data.datasets import confirmed_negative_names

    labels = tmp_path / "some" / "flat" / "labels"
    labels.mkdir(parents=True)
    state = tmp_path / ".tcip" / "state"
    state.mkdir(parents=True)
    (state / "image_status.json").write_text(json.dumps({"catkin": {"a.jpg": "complete"}}))

    assert confirmed_negative_names(labels) == set()


def test_split_tree_carries_its_confirmed_negatives(tmp_path):
    """make_splits(materialize=True) emits {train,val,test}/labels, which cannot name its subject,
    so it must carry the confirmations rather than inherit them by accident."""
    from tcip_mcp.pipelines.data.datasets import confirmed_negative_names
    from tcip_mcp.tools.data_tools import make_splits

    images = tmp_path / "images"
    labels = tmp_path / "annotations" / "catkin" / "detect"
    labels.mkdir(parents=True)
    _make_images(images, [f"i{n:02d}" for n in range(10)])
    for n in range(10):
        stem = f"i{n:02d}"
        boxes = [] if n % 2 else [BBox(4, 4, 12, 12, 0)]
        json_io.write_detect(labels / f"{stem}.json", boxes, 100, 100, keep_empty=True)
    state = tmp_path / ".tcip" / "state"
    state.mkdir(parents=True)
    (state / "image_status.json").write_text(json.dumps(
        {"catkin": {f"i{n:02d}.jpg": "negative" for n in range(1, 10, 2)}}))

    out = tmp_path / "splits"
    make_splits(str(tmp_path), output_path=str(out), materialize=True)

    carried = set()
    for split in ("train", "val", "test"):
        d = out / split / "labels"
        if d.is_dir():
            carried |= confirmed_negative_names(d)
    assert carried, "the split tree lost every human-confirmed negative"
