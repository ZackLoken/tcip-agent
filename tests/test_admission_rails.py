"""The admission rails over a place holding ground truth.

Which samples the place admits, and why each one that is dropped was dropped.

Covers the one shape read over a run's ground truth (``ground_truth_shape``), the label
documents' own subject-scoped admission and their completion marks, and the targets the loaders
read off each admitted sample's own document, an unassessed attribute among them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from tcip_annotation.json_io import CompletionMark
    from tcip_annotation.state import Ring
    from tcip_store import Version

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

import tcip_store as ts  # noqa: E402
from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox, Polygon  # noqa: E402
from tcip_mcp.dataset_layout import (  # noqa: E402
    UNDATED_BUCKET,
    image_dir,
    label_key,
)
from tcip_mcp.pipelines.data.selection import ClassScope  # noqa: E402
from tcip_mcp.subject_registry import Attribute  # noqa: E402
from tests._producer_fixtures import (  # noqa: E402
    admit_over,
    box_annotation,
    dataset_over,
    image_label_key,
    label_image,
    mark_complete,
    registry_of,
    registry_over,
    scope_over,
    write_image,
    write_jpegs,
)
from tests._training_values import tiling_block  # noqa: E402

BUD = "bud"
REGISTRY = registry_of(
    BUD,
    Attribute(name="opening", type="categorical", values=("open", "closed")),
)
"""The registry declaring :data:`BUD` alone, with ``opening`` over ``open`` and ``closed``."""


SIZE = (100, 100)
"""The frame size every image here is written and labeled at."""


def _label(
    image: Path,
    annotations: list[Annotation],
    **kwargs: bool | Version | Mapping[str, list[CompletionMark]] | None,
) -> None:
    """Write ``annotations`` as the label document of the :data:`SIZE` ``image``."""
    label_image(image, annotations, *SIZE, **kwargs)


def _poly(
    points: list[tuple[float, float]], *, subject: str = BUD, **attrs: str
) -> Annotation:
    """Return a one-ring polygon annotation of ``subject`` over ``points`` (:func:`_multi_poly`)."""
    return _multi_poly([points], subject=subject, **attrs)


def _multi_poly(
    rings: list[list[tuple[float, float]]],
    *,
    subject: str = BUD,
    **attrs: str,
) -> Annotation:
    """Return one annotation of ``subject`` whose polygon holds each of ``rings``."""
    return Annotation(
        subject=subject, geometry=Polygon(list(rings)), attributes=dict(attrs)
    )


# ── the one shape read ──────────────────────────────────────────────────────


def test_ground_truth_shape_reads_no_place_as_the_images_own_documents() -> (
    None
):
    from tcip_mcp.pipelines.data.label_queries import ground_truth_shape

    assert ground_truth_shape(None) == "document"


def test_ground_truth_shape_reads_masks_and_tables(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.data.label_queries import ground_truth_shape

    masks = tmp_path / "masks"
    masks.mkdir()
    Image.new("L", (8, 8)).save(masks / "a.png")
    assert ground_truth_shape(masks) == "mask"

    table = tmp_path / "gt.csv"
    table.write_text("stem,value\na,1\n", encoding="utf-8")
    assert ground_truth_shape(table) == "table"


def test_ground_truth_shape_refuses_a_directory_holding_no_mask(
    tmp_path: Path,
) -> None:
    """``ground_truth_shape`` refuses a directory holding no ``<stem>.png``.

    Such a directory names no ground truth a run reads: label documents are records of the
    images' own dataset, never files a run is pointed at.
    """
    from tcip_mcp.pipelines.data.label_queries import ground_truth_shape

    d = tmp_path / "detect"
    d.mkdir()
    (d / "a.json").write_text('{"annotations": []}', encoding="utf-8")
    with pytest.raises(
        ValueError, match="neither a .csv table nor a directory of <stem>.png"
    ):
        ground_truth_shape(d)


def test_a_document_with_no_image_is_not_admitted(tmp_path: Path) -> None:
    """A label document whose image is absent is not admitted.

    Membership is the images the place holds ground truth for; a document whose image is gone
    names no member.
    """
    images = image_dir(tmp_path, UNDATED_BUCKET)
    (kept,) = write_jpegs(images, ["kept"], SIZE)
    _label(kept, [box_annotation(10, 10, 50, 50)])
    json_io.write_label_document(
        label_key(tmp_path, UNDATED_BUCKET, "orphan"),
        [box_annotation(10, 10, 50, 50)],
        100,
        100,
    )

    admitted = admit_over(images, subject=BUD)
    assert [r.member for r in admitted.records] == ["kept"]


def test_a_noncanonical_labelme_record_refuses_rather_than_reading_as_empty(
    tmp_path: Path,
) -> None:
    """A record in another tool's schema is refused as unreadable ground truth."""
    images = image_dir(tmp_path, UNDATED_BUCKET)
    (image,) = write_jpegs(images, ["a"], SIZE)
    ts.replace(
        image_label_key(image),
        {
            "version": "5.0.1",
            "imageWidth": 100,
            "imageHeight": 100,
            "shapes": [
                {
                    "label": BUD,
                    "shape_type": "rectangle",
                    "points": [[10, 10], [50, 50]],
                }
            ],
        },
    )

    with pytest.raises(json_io.UnreadableLabelDocumentError):
        admit_over(images, subject=BUD)


# ── targets read off each sample's own document ─────────────────────────────


def test_detection_reads_its_samples_own_documents(tmp_path: Path) -> None:
    images = image_dir(tmp_path, UNDATED_BUCKET)
    (image,) = write_jpegs(images, ["img0"], SIZE)
    _label(image, [box_annotation(10, 10, 50, 50)])

    ds = dataset_over("detection", images, subject=BUD)
    _, target = ds[0]
    assert target["boxes"].shape == (1, 4)
    assert target["boxes"].tolist()[0] == pytest.approx([10, 10, 50, 50])
    assert target["labels"].tolist() == [1]  # 0-indexed cid 0 -> 1-indexed
    assert ds.class_distribution == {0: 1}


def test_a_point_document_is_admitted_and_trains_through_the_builder_that_reads_points(
    tmp_path: Path,
) -> None:
    """A document carrying the subject as a point is admitted and trains through a point builder.

    Two documents carrying the subject as points are admitted; the detection loader refuses them
    naming geometries it does not read, while a builder that reads points is handed both samples
    and reads the coordinates their own documents record, its loss falling over three steps.
    """
    import math

    from tcip_annotation.state import Point

    images = image_dir(tmp_path, UNDATED_BUCKET)
    for index, image in enumerate(write_jpegs(images, ["p0", "p1"], SIZE)):
        _label(
            image,
            [Annotation(subject=BUD, geometry=Point(20 + index, 30 + index))],
        )

    admitted = admit_over(images, subject=BUD)
    assert sorted(r.member for r in admitted.records) == ["p0", "p1"]

    with pytest.raises(
        ValueError, match="only in geometries a detection loader does not read"
    ):
        dataset_over("detection", images, subject=BUD)

    # Admits valid work: a builder that reads points is handed the same admitted samples, reads
    # each document's own coordinates, and trains over them.
    from tests.test_dataset_source_seam import BESPOKE_DS_FILE, _bespoke

    built = dataset_over(
        "keypoints",
        images,
        subject=BUD,
        dataset_source=_bespoke(
            {
                "builder": "test_dataset_source_seam:build_point_ds",
                "source_files": [BESPOKE_DS_FILE],
            }
        ),
    )
    assert sorted(s.member for s in built.samples) == ["p0", "p1"]
    assert built.points == [(20.0, 30.0), (21.0, 31.0)]

    pixels = torch.stack(
        [built[i][0].mean(dim=(1, 2)) for i in range(len(built))]
    )
    targets = torch.stack([built[i][1]["point"] for i in range(len(built))])
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


def test_class_distribution_counts_only_this_loaders_own_samples(
    tmp_path: Path,
) -> None:
    """Each loader's class distribution counts only its own samples' documents.

    Two loaders over two halves of one dataset's samples each report the box count of its own
    half, not the dataset's total.
    """
    from tcip_mcp.pipelines.data.datasets import build_dataset, resolve_sizes

    images = image_dir(tmp_path, UNDATED_BUCKET)
    stems = [f"img{i}" for i in range(4)]
    for i, image in enumerate(write_jpegs(images, stems, SIZE)):
        _label(image, [box_annotation(10, 10, 30, 30)] * (i + 1))

    admitted = admit_over(images, subject=BUD)
    samples = admitted.samples({stem: "train" for stem in stems}, lambda s: s)
    by_stem = {sample.member: sample for sample in samples}
    # The run's own sizes, resolved once over both sides, as a run builds its loaders.
    sizes = resolve_sizes("detection", {}, samples)
    train_ds = build_dataset(
        "detection",
        samples=[by_stem[s] for s in stems[:2]],
        scope=admitted.scope,
        sizes=sizes,
    )
    val_ds = build_dataset(
        "detection",
        samples=[by_stem[s] for s in stems[2:]],
        scope=admitted.scope,
        sizes=sizes,
    )

    # img0 (1 box) + img1 (2 boxes) = 3; img2 (3 boxes) + img3 (4 boxes) = 7.
    assert train_ds.class_distribution == {0: 3}
    assert val_ds.class_distribution == {0: 7}


def test_instance_seg_reads_its_samples_own_documents(tmp_path: Path) -> None:
    images = image_dir(tmp_path, UNDATED_BUCKET)
    (image,) = write_jpegs(images, ["img0"], SIZE)
    _label(image, [_poly([(10, 10), (50, 10), (50, 50), (10, 50)])])

    ds = dataset_over("instance_seg", images, subject=BUD)
    _, target = ds[0]
    assert target["boxes"].shape == (1, 4)
    assert target["masks"].shape[0] == 1
    assert target["labels"].tolist() == [1]
    assert (
        int(target["masks"].sum()) > 0
    )  # polygon rasterized to a non-empty mask


# Two disjoint lobes of one instance, with a clear gap between them.
LOBE_A: Ring = [(10, 10), (30, 10), (30, 30), (10, 30)]
LOBE_B: Ring = [(60, 10), (80, 10), (80, 30), (60, 30)]


def _assert_one_mask_over_both_lobes(target: dict) -> None:
    """Assert a 2-ring instance is one mask covering both lobes, not two instances, not one lobe."""
    assert target["masks"].shape[0] == 1, (
        "a multi-ring instance must not split into several"
    )
    assert target["labels"].tolist() == [1]
    # The box spans the union of the rings.
    assert target["boxes"].tolist() == [[10.0, 10.0, 80.0, 30.0]]
    mask = target["masks"][0].numpy()
    assert mask[10:31, 10:31].sum() > 0, (
        "the first lobe is missing from the mask"
    )
    assert mask[10:31, 60:81].sum() > 0, (
        "the second lobe is missing from the mask"
    )
    # The occluded gap between the lobes stays background: the union of rings, not their hull.
    assert mask[:, 35:55].sum() == 0


def test_instance_seg_rasterizes_a_two_ring_instance_into_one_mask(
    tmp_path: Path,
) -> None:
    """A two-ring instance rasterizes into one mask covering both rings.

    An occlusion-split instance (a bud behind a branch) is one object with two regions, and both
    rings rasterize into that instance's single mask.
    """
    images = image_dir(tmp_path, UNDATED_BUCKET)
    (image,) = write_jpegs(images, ["img0"], SIZE)
    _label(image, [_multi_poly([LOBE_A, LOBE_B])])

    ds = dataset_over("instance_seg", images, subject=BUD)
    _, target = ds[0]
    _assert_one_mask_over_both_lobes(target)


def test_instance_seg_two_single_ring_instances_stay_two_masks(
    tmp_path: Path,
) -> None:
    """Two single-ring annotations stay two instances with two masks.

    Two lobes authored as separate annotations give two masks, their boxes the two lobes'
    extents.
    """
    images = image_dir(tmp_path, UNDATED_BUCKET)
    (image,) = write_jpegs(images, ["img0"], SIZE)
    _label(image, [_poly(LOBE_A), _poly(LOBE_B)])

    ds = dataset_over("instance_seg", images, subject=BUD)
    _, target = ds[0]
    assert target["masks"].shape[0] == 2
    assert target["boxes"].tolist() == [
        [10.0, 10.0, 30.0, 30.0],
        [60.0, 10.0, 80.0, 30.0],
    ]


def test_count_label_lines_reads_a_documents_instances(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.data.splits import count_label_lines

    images = image_dir(tmp_path, UNDATED_BUCKET)
    a, neg = write_jpegs(images, ["a", "neg"], SIZE)
    _label(a, [box_annotation(0, 0, 10, 10), box_annotation(0, 0, 20, 20)])
    _label(neg, [], keep_empty=True)
    assert (
        count_label_lines(
            json_io.read_label_document(image_label_key(a)), ClassScope()
        )
        == 2
    )
    assert (
        count_label_lines(
            json_io.read_label_document(image_label_key(neg)), ClassScope()
        )
        == 0
    )


# ── the label documents' own rails ──────────────────────────────────────────


def _rail_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Write an annotated, an empty-unconfirmed, an unlabeled and a confirmed-negative image.

    The unlabeled image has no label document. Returns their directory and the confirmed
    negative.
    """
    images = image_dir(tmp_path, UNDATED_BUCKET)
    ann, empty, _nolabel, neg = write_jpegs(
        images, ["ann", "empty", "nolabel", "neg"], SIZE
    )
    _label(ann, [box_annotation(4, 4, 12, 12)], keep_empty=True)
    _label(empty, [], keep_empty=True)
    _label(neg, [], keep_empty=True)
    mark_complete(neg, BUD, project=tmp_path)
    return images, neg


def test_only_annotated_and_confirmed_negatives_train(tmp_path: Path) -> None:
    """Only the annotated image and the confirmed negative are admitted.

    Samples come from the annotated set, never from an image list: the empty unconfirmed image
    and the image with no label document are not admitted.
    """
    images, _neg = _rail_fixture(tmp_path)

    admitted = admit_over(images, subject=BUD)
    assert sorted(r.member for r in admitted.records) == ["ann", "neg"]
    assert admitted.tallies["partial"] == 1
    assert admitted.tallies["negative"] == 1
    assert admitted.tallies["absent"] >= 1


def test_a_corrupt_confirmed_negative_refuses_the_admission(
    tmp_path: Path,
) -> None:
    """A confirmed negative whose label document is not a document refuses the admission.

    The admission raises ``UnreadableLabelDocumentError`` on a stored 'negative' whose label
    document is not a document.
    """
    images, neg = _rail_fixture(tmp_path)
    ts.replace(image_label_key(neg), ["not", "a", "document"])

    with pytest.raises(json_io.UnreadableLabelDocumentError):
        admit_over(images, subject=BUD)


def test_caller_supplied_members_are_filtered_too(tmp_path: Path) -> None:
    """Caller-supplied members are filtered to the annotated image and the confirmed negative.

    A narrowed candidate set naming the empty unconfirmed and the unlabeled images goes through
    the admission's checks, which leave both out.
    """
    images, _neg = _rail_fixture(tmp_path)
    admitted = admit_over(
        images, subject=BUD, members=["ann", "empty", "nolabel", "neg"]
    )
    assert sorted(r.member for r in admitted.records) == ["ann", "neg"]


def test_no_trainable_samples_raises_rather_than_training_on_nothing(
    tmp_path: Path,
) -> None:
    images = image_dir(tmp_path, UNDATED_BUCKET)
    a, _b = write_jpegs(images, ["a", "b"], SIZE)
    _label(a, [], keep_empty=True)  # unconfirmed empty

    with pytest.raises(ValueError, match="no trainable samples"):
        admit_over(images, subject=BUD)


def test_instance_seg_applies_the_same_rail(tmp_path: Path) -> None:
    images = image_dir(tmp_path, UNDATED_BUCKET)
    ann, _nolabel = write_jpegs(images, ["ann", "nolabel"], SIZE)
    _label(ann, [_poly([(4, 4), (12, 4), (12, 12), (4, 12)])], keep_empty=True)

    ds = dataset_over("instance_seg", images, subject=BUD)
    assert [ds.sample_of(k).member for k in ds.stems] == ["ann"]


def test_instance_seg_admits_a_partially_assessed_stem_on_its_subject_marks(
    tmp_path: Path,
) -> None:
    """A partially assessed image is admitted on its subject's marks alone.

    Beside a fully assessed image, an image holding an instance with no value for the
    registry's attribute is admitted, and the scope carries the registry's attributes.
    """
    root = tmp_path / "ds"
    images_dir = image_dir(root, UNDATED_BUCKET)
    complete, partial = write_jpegs(images_dir, ["complete", "partial"], SIZE)
    _label(
        complete,
        [_poly([(4, 4), (12, 4), (12, 12), (4, 12)], opening="open")],
    )
    _label(
        partial,
        [
            _poly([(4, 4), (12, 4), (12, 12), (4, 12)], opening="closed"),
            _poly(
                [(40, 40), (60, 40), (60, 60), (40, 60)]
            ),  # unassessed: no opening value
        ],
    )
    registry_over(root, REGISTRY)

    admitted = admit_over(images_dir, subject=BUD)

    assert [r.member for r in admitted.records] == ["complete", "partial"]
    assert admitted.scope.attributes == REGISTRY.subjects[0].attributes
    assert admitted.tallies["partial"] == 2


def test_semantic_seg_requires_a_mask_but_admits_an_all_background_one(
    tmp_path: Path,
) -> None:
    """A semantic segmentation loader admits an all-background mask and drops a maskless image."""
    import numpy as np
    from PIL import Image as _Image

    images, masks = image_dir(tmp_path, UNDATED_BUCKET), tmp_path / "masks"
    images.mkdir(parents=True)
    masks.mkdir()
    for stem in ("has_mask", "all_background", "no_mask"):
        write_image(images / f"{stem}.jpg", (32, 32))
    _Image.fromarray(np.ones((32, 32), dtype=np.uint8)).save(
        masks / "has_mask.png"
    )
    _Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(
        masks / "all_background.png"
    )

    ds = dataset_over("semantic_seg", images, masks)
    assert sorted(ds.sample_of(k).member for k in ds.stems) == [
        "all_background",
        "has_mask",
    ]


def test_sample_counts_distinguish_unannotated_from_unconfirmed_empty(
    tmp_path: Path,
) -> None:
    """The admission's tallies tell an unannotated image from an unconfirmed empty one.

    Over the rail fixture the tallies count one partial, one negative, one absent and one
    unannotated image.
    """
    images, _neg = _rail_fixture(tmp_path)
    admitted = admit_over(images, subject=BUD)
    assert admitted.tallies == {
        "partial": 1,
        "negative": 1,
        "absent": 1,
        "unannotated": 1,
    }


def test_a_confirmation_does_not_leak_across_subjects(tmp_path: Path) -> None:
    """A completion mark for one subject does not admit the image as a negative of another."""
    images = image_dir(tmp_path, UNDATED_BUCKET)
    ann, shared = write_jpegs(images, ["ann", "shared"], SIZE)
    # Every subject's annotation records share one per-image document; subject is a field in it.
    _label(
        ann,
        [
            box_annotation(4, 4, 12, 12, subject="bud"),
            box_annotation(4, 4, 12, 12, subject="bush"),
        ],
        keep_empty=True,
    )
    _label(shared, [], keep_empty=True)
    # Confirmed negative for bud only; the breeder never judged it for bush.
    mark_complete(shared, "bud", project=tmp_path)

    assert sorted(
        r.member for r in admit_over(images, subject="bud").records
    ) == ["ann", "shared"]
    assert [r.member for r in admit_over(images, subject="bush").records] == [
        "ann"
    ]


def test_a_negative_mark_dies_with_an_edit_of_its_subject(
    tmp_path: Path,
) -> None:
    """An edit of a marked negative's subject clears the mark, and the image tallies as annotated.

    Writing a box over the marked negative's document leaves it no marks, and the admission
    tallies two partial images and no negative.
    """
    images, negative = _rail_fixture(tmp_path)
    neg = image_label_key(negative)
    marks = json_io.read_label_document(neg).marks
    json_io.write_label_document(
        neg, [box_annotation(4, 4, 12, 12)], 100, 100, marks=marks
    )

    assert json_io.read_label_document(neg).marks == {}
    tallies = admit_over(images, subject=BUD).tallies
    assert (tallies["partial"], tallies.get("negative", 0)) == (2, 0)


def test_json_det_targets_marks_an_unassessed_row_and_refuses_an_undeclared_value(
    tmp_path: Path,
) -> None:
    """``json_det_targets`` keeps an unassessed row and refuses an undeclared value.

    The loader's own per-image target reader keeps an unassessed instance as a row whose
    attribute column carries the unassessed mark, while a value its attribute does not declare
    refuses.
    """
    from tcip_mcp.pipelines.data.label_queries import json_det_targets

    root = tmp_path / "ds"
    images = image_dir(root, UNDATED_BUCKET)
    image_a, image_b = write_jpegs(images, ["IMG_A", "IMG_B"], SIZE)
    scope = scope_over(root, REGISTRY, BUD)
    _label(
        image_a,
        [
            Annotation(
                subject="bud",
                geometry=BBox(10, 10, 30, 30),
                attributes={"opening": "closed"},
            ),
            Annotation(
                subject="bud", geometry=BBox(40, 40, 60, 60), attributes={}
            ),  # unassessed
        ],
    )

    target = json_det_targets(
        json_io.read_label_document(image_label_key(image_a)).annotations,
        scope,
    )
    assert len(target["boxes"]) == 2 and target["labels"] == [1, 1]
    assert target["attributes"].tolist() == [[1], [json_io.UNASSESSED]]

    _label(
        image_b,
        [
            Annotation(
                subject="bud",
                geometry=BBox(10, 10, 30, 30),
                attributes={"opening": "not-a-real-value"},
            ),
        ],
    )
    with pytest.raises(json_io.UndeclaredValueError):
        json_det_targets(
            json_io.read_label_document(image_label_key(image_b)).annotations,
            scope,
        )


def test_detection_and_its_tiles_keep_a_partially_assessed_stem(
    tmp_path: Path,
) -> None:
    """A partially assessed image reaches the detection loader and the tiler whole.

    The loader's target keeps the unassessed instance's row beside its box, and each tile carries
    one attribute row per box.
    """
    root = tmp_path / "ds"
    images_dir = image_dir(root, UNDATED_BUCKET)
    complete, partial = write_jpegs(images_dir, ["complete", "partial"], SIZE)
    registry_over(root, REGISTRY)
    _label(complete, [box_annotation(10, 10, 30, 30, opening="open")])
    _label(
        partial,
        [
            box_annotation(10, 10, 30, 30, opening="closed"),
            box_annotation(
                40, 40, 60, 60
            ),  # unassessed: no opening value at all
        ],
    )

    ds = dataset_over("detection", images_dir, subject=BUD)
    _image, target = ds[
        ds.stems.index(
            next(k for k in ds.stems if ds.sample_of(k).member == "partial")
        )
    ]
    assert target["attributes"].tolist() == [[1], [json_io.UNASSESSED]]

    tiled = dataset_over(
        "detection",
        images_dir,
        subject=BUD,
        tiling={
            **tiling_block(),
            "tile_size": 64,
            "overlap": 0.0,
            "sliver_frac": 0.5,
        },
    )  # stated: one box derives no spread
    assert {tiled.sample_of(k).member for k in tiled.stems} == {
        "complete",
        "partial",
    }
    for index in range(len(tiled)):
        _tile, rows = tiled[index]
        assert rows["attributes"].shape == (len(rows["boxes"]), 1)
