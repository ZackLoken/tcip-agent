"""Which images a detection run may train on: the scope of a label record and of a human status.

One per-image label file holds every subject's records and one status store holds every status a
human can leave, so both rails are scoped rather than global. A record of another subject is not
this run's annotation, a geometry-less whole-image note is not a detection target, and "complete"
is the token for a finished image that has content, the opposite of "negative", never a weaker
form of it.
"""

import pytest

torch = pytest.importorskip("torch")

from PIL import Image  # noqa: E402

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox  # noqa: E402
from tcip_mcp.dataset_layout import record_image_statuses, status_bucket  # noqa: E402

BUD = "bud"
BUSH = "bush"
IMAGE_W, IMAGE_H = 160, 90


def _make_images(images_dir, stems):
    """Non-square frames, so a width/height confusion anywhere downstream cannot stay hidden."""
    images_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        Image.new("RGB", (IMAGE_W, IMAGE_H)).save(images_dir / f"{stem}.jpg")


def _write(labels_dir, stem, annotations):
    json_io.write_annotations(labels_dir / f"{stem}.json", annotations, IMAGE_W, IMAGE_H,
                              keep_empty=True)


def _box(x1, y1, x2, y2, *, subject=BUD):
    return Annotation(subject=subject, geometry=BBox(x1, y1, x2, y2))


def test_a_record_of_another_subject_is_not_this_subjects_annotation(tmp_path):
    """Admission is per subject: an image annotated only for bushes has nothing a bud run can
    learn from and no human statement that it holds no buds, so it is held out rather than
    trained as a zero-box negative. Each subject's own run still admits both of its images.
    """
    from tcip_mcp.pipelines.data.datasets import DetectionDataset

    images, labels = tmp_path / "images", tmp_path / "annotations"
    labels.mkdir(parents=True)
    _make_images(images, ["buds_only", "bushes_only", "both"])
    _write(labels, "buds_only", [_box(12, 8, 60, 30)])
    _write(labels, "bushes_only", [_box(5, 5, 150, 85, subject=BUSH)])
    _write(labels, "both", [_box(20, 10, 44, 70), _box(0, 0, 159, 89, subject=BUSH)])

    bud_ds = DetectionDataset(str(images), str(labels), subject=BUD)
    assert sorted(bud_ds.stems) == ["both", "buds_only"]
    assert bud_ds.sample_counts["annotated"] == 2
    assert bud_ds.sample_counts["skipped_unconfirmed_empty"] == 1
    for idx, stem in enumerate(bud_ds.stems):
        _img, target = bud_ds[idx]
        assert target["boxes"].shape[0] > 0, (
            f"{stem} was admitted as annotated for {BUD} but carries no target of it")

    bush_ds = DetectionDataset(str(images), str(labels), subject=BUSH)
    assert sorted(bush_ds.stems) == ["both", "bushes_only"]
    assert bush_ds.sample_counts["annotated"] == 2
    assert bush_ds.sample_counts["skipped_unconfirmed_empty"] == 1


def test_a_whole_image_note_does_not_make_an_image_trainable(tmp_path):
    """A record with no geometry is a note about the whole image, not a detection target. An image
    carrying only those has no box to learn from and nobody has marked it empty, so it is held out
    with the unconfirmed-empty images rather than trained as a negative.
    """
    from tcip_mcp.pipelines.data.datasets import DetectionDataset

    images, labels = tmp_path / "images", tmp_path / "annotations"
    labels.mkdir(parents=True)
    _make_images(images, ["boxed", "rated_only"])
    _write(labels, "boxed", [_box(12, 8, 60, 30)])
    _write(labels, "rated_only",
           [Annotation(subject=BUD, geometry=None, attributes={"vigor": "high"})])

    ds = DetectionDataset(str(images), str(labels), subject=BUD)
    assert ds.stems == ["boxed"]
    assert ds.sample_counts["annotated"] == 1
    assert ds.sample_counts["skipped_unconfirmed_empty"] == 1


def test_the_only_status_that_confirms_a_negative_is_the_negative_one(tmp_path):
    """"complete" is what the review flow records for an image the human finished with content on
    it. Reading it as a confirmation of emptiness turns a populated image into a negative nobody
    stated, so only the negative token confirms one, and it still confirms it.
    """
    from tcip_mcp.pipelines.data.datasets import DetectionDataset
    from tcip_mcp.pipelines.data.label_queries import confirmed_negative_names

    images, labels = tmp_path / "images", tmp_path / "annotations"
    labels.mkdir(parents=True)
    _make_images(images, ["worked", "emptied", "confirmed_empty"])
    _write(labels, "worked", [_box(12, 8, 60, 30), _box(70, 20, 96, 84)])
    _write(labels, "emptied", [])
    _write(labels, "confirmed_empty", [])
    record_image_statuses(tmp_path, status_bucket(BUD, None), {
        "worked.jpg": "complete",
        "emptied.jpg": "complete",
        "confirmed_empty.jpg": "negative",
    }, recorded_by="user:breeder")

    assert confirmed_negative_names(labels, subject=BUD, date=None) == {"confirmed_empty.jpg"}

    ds = DetectionDataset(str(images), str(labels), subject=BUD)
    assert sorted(ds.stems) == ["confirmed_empty", "worked"]
    assert ds.sample_counts["confirmed_negative"] == 1
    assert ds.sample_counts["skipped_unconfirmed_empty"] == 1
