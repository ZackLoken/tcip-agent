"""The name-based annotation schema: subjects, never integer class ids, in a record or in memory.

The measurement-critical invariants of name-based labels: the registry decodes its own labels, a
geometry-less annotation round-trips without collapsing to a negative, every attribute id rests
on the registry's declared order, negatives key through a threaded subject, the loader filters by
subject and geometry, decode inverts the recorded scope, and authoring refuses a subjectless
label. Each test builds its own dataset, sharing no fixture with another.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

import tcip_store
from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox, Polygon
from tcip_mcp.dataset_layout import UNDATED_BUCKET
from tcip_mcp.subject_registry import SubjectRegistry, Subject
from tcip_mcp.pipelines.data.label_queries import registry_scope
from tests._producer_fixtures import (  # noqa: E402
    dataset_over, image_label_key, label_image, registry_over,
)


def _write_image(images_dir: Path, stem: str, size=(640, 480)) -> Path:
    images_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(128, 128, 128)).save(images_dir / f"{stem}.jpg")
    return images_dir / f"{stem}.jpg"


def _write_registry(root: Path, *subjects: Subject) -> SubjectRegistry:
    registry = SubjectRegistry(subjects=tuple(subjects))
    registry_over(root, registry)
    return registry


def _saved(image: Path) -> list[Annotation]:
    """The annotations of ``image``'s label document as the save door left them."""
    return json_io.read_label_document(image_label_key(image)).annotations


# (a) a registry decodes its own labels after the flip.
def test_registry_decodes_its_own_labels(tmp_path):
    registry = _write_registry(tmp_path, Subject(name="bud"))
    images_dir = tmp_path / "images" / UNDATED_BUCKET
    image = _write_image(images_dir, "img_001")
    label_image(image, [Annotation(subject="bud", geometry=BBox(10, 10, 40, 40))], 640, 480)

    ds = dataset_over("detection", str(images_dir), subject="bud")
    _img, target = ds[0]

    # The loader's scope is the registry's own subject, and every row of the target is that one
    # subject: the registry reads its own labels without guessing.
    assert ds.scope.subject == registry.subjects[0].name == "bud"
    assert ds.scope.attributes == ()
    assert target["labels"].tolist() == [1], "the labeled image produced no target"


# (b) a geometry-less annotation round-trips and its image is not collapsed to empty/negative.
def test_geometryless_annotation_roundtrips_and_marks_image_annotated(tmp_path):
    from tcip_mcp.pipelines.data.label_queries import admit

    _write_registry(tmp_path, Subject(name="bud"))
    images_dir = tmp_path / "images" / UNDATED_BUCKET
    image = _write_image(images_dir, "img_001")
    label_image(image, [Annotation(subject="bud")], 640, 480)

    # Round-trips losslessly: subject preserved, geometry None.
    back = _saved(image)
    assert len(back) == 1 and back[0].subject == "bud" and back[0].geometry is None

    # The image carries a subject annotation, so the admission counts it as annotated rather than
    # as an empty one nobody confirmed; which geometries answer for a measurement is the loader's.
    admitted = admit(images_dir, scope=registry_scope(images_dir, "bud"))
    assert [record.member for record in admitted.records] == ["img_001"]
    assert admitted.tallies == {"partial": 1}


# (c) the admitted scope carries every attribute the registry declares, in declared order.
def test_the_admitted_scope_carries_the_registrys_attributes(tmp_path):
    from tcip_mcp.subject_registry import Attribute

    registry = _write_registry(tmp_path, Subject(name="bud", attributes=(
        Attribute("color", "categorical", ("red", "blue")),
        Attribute("grade", "ordinal", ("low", "high")))))
    images_dir = tmp_path / "images" / UNDATED_BUCKET
    for stem in ("a", "b"):
        label_image(_write_image(images_dir, stem),
                    [Annotation(subject="bud", geometry=BBox(10, 10, 40, 40))], 640, 480)

    ds = dataset_over("detection", str(images_dir), subject="bud")

    assert ds.scope.attributes == registry.subjects[0].attributes


# (d) a confirmed negative is scoped to the subject its mark names.
def test_a_confirmed_negative_is_its_own_subjects_alone(tmp_path):
    from tests._producer_fixtures import mark_complete

    image = _write_image(tmp_path / "images" / "2-11-26", "img_009")
    label_image(image, [], 640, 480, keep_empty=True)
    # A human confirmed img_009 an empty negative for bud on this date.
    mark_complete(image, "bud", project=tmp_path)

    doc = json_io.read_label_document(image_label_key(image))
    assert doc.state("bud") == "negative"
    # A different subject's mark is not this subject's negative (scoping holds).
    assert doc.state("bush") == "unannotated"


# (e) geometry-less + wrong-subject annotations are excluded from a detection run's targets.
def test_loader_filters_by_subject_and_geometry(tmp_path):
    import torch

    _write_registry(tmp_path, Subject(name="bud"), Subject(name="bush"))
    images_dir = tmp_path / "images" / UNDATED_BUCKET
    label_image(_write_image(images_dir, "img_001"), [
        Annotation(subject="bud", geometry=BBox(10, 10, 40, 40)),   # a legitimate target
        Annotation(subject="bush", geometry=BBox(50, 50, 90, 90)),  # wrong subject -> drop
        Annotation(subject="bud"),                                  # geometry-less -> drop
    ], 640, 480)

    ds = dataset_over("detection", str(images_dir), subject="bud")
    assert ds.scope.subject == "bud"
    _img, target = ds[0]
    # Only the one legitimate bud box survives; the wrong-subject and geometry-less rows are gone.
    assert target["boxes"].shape[0] == 1
    assert torch.equal(target["labels"], torch.tensor([1], dtype=torch.int64))  # 0-idx bud +1 bg


# (f) the encoder decodes a prediction's attribute ids through the scope the loader read under.
def test_decode_inverts_the_recorded_scope(tmp_path):
    from tcip_mcp.pipelines.postprocessing.export import encode_predictions
    from tcip_mcp.subject_registry import Attribute

    _write_registry(tmp_path, Subject(name="bud", attributes=(
        Attribute("color", "categorical", ("red", "blue")),)))
    scope = registry_scope(tmp_path / "images", "bud")

    data, _dropped = encode_predictions(
        {"image": "pred.jpg", "boxes": [[10, 10, 40, 40]], "scores": [0.9], "labels": [1],
         "attributes": [[1]], "width": 640, "height": 480},
        created_by="model:x", scope=scope)
    preds = json_io.label_document(data).annotations
    assert [(p.subject, p.attributes) for p in preds] == [("bud", {"color": "blue"})]


# (g) save_annotations refuses a missing subject.
def test_save_annotations_refuses_missing_subject(tmp_path):
    from tcip_mcp.tools.annotation_tools import save_annotations

    image = _write_image(tmp_path / "images" / UNDATED_BUCKET,"img_001")

    res = save_annotations(tmp_path, tmp_path.parent, str(image),
                           annotations=[{"bbox": [10, 10, 40, 40]}])
    assert "error" in res and "subject" in res["error"]
    assert not tcip_store.exists(image_label_key(image))

    # A real subject still writes (the rail admits valid work).
    ok = save_annotations(tmp_path, tmp_path.parent, str(image),
                          annotations=[{"subject": "bud", "bbox": [10, 10, 40, 40]}])
    assert ok.get("count") == 1 and tcip_store.exists(image_label_key(image))


def test_save_annotations_prefers_points_over_bbox(tmp_path):
    """save_annotations prefers points over bbox (aligned with the web converters), so a payload
    carrying both geometries writes the polygon, never collapsing it to a box-only record."""
    from tcip_mcp.tools.annotation_tools import save_annotations

    image = _write_image(tmp_path / "images" / UNDATED_BUCKET,"img_001")

    res = save_annotations(
        tmp_path, tmp_path.parent, str(image),
        annotations=[{
            "subject": "bud",
            "points": [[10, 20], [110, 20], [110, 220]],
            "bbox": [10, 20, 110, 220],
        }],
    )
    assert res.get("count") == 1

    (ann,) = _saved(image)
    assert isinstance(ann.geometry, Polygon)  # the polygon won; not collapsed to a box
    # "points" is the single-ring input key, wrapped as the one ring it is.
    assert ann.geometry.rings == [[(10.0, 20.0), (110.0, 20.0), (110.0, 220.0)]]
    obj = tcip_store.read(image_label_key(image))["annotations"][0]
    assert "segmentation" in obj  # written as a polygon


@pytest.mark.parametrize("geometry", [
    {"rings": []}, {"points": []}, {"points": [[10, 20]]},
    {"points": [], "bbox": [10, 20, 110, 220]},
], ids=["no_rings", "no_points", "one_vertex", "no_points_beside_a_box"])
def test_save_annotations_refuses_a_polygon_that_is_no_shape(tmp_path, geometry):
    """A supplied polygon that is empty or too short is a malformed value, refused by name before
    anything is written, never read as no geometry nor saved and then dropped on write."""
    from tcip_mcp.tools.annotation_tools import save_annotations

    image = _write_image(tmp_path / "images" / UNDATED_BUCKET,"img_001")

    res = save_annotations(tmp_path, tmp_path.parent, str(image),
                           annotations=[{"subject": "bud", **geometry}])

    assert "polygon" in res.get("error", "")
    assert not tcip_store.exists(image_label_key(image))


def test_save_annotations_accepts_rings(tmp_path):
    """An occlusion-split instance in either ring-vertex shape (``{x, y}`` mappings or ``[x, y]``
    pairs) saves with its geometry through the write door."""
    from tcip_mcp.tools.annotation_tools import save_annotations

    image = _write_image(tmp_path / "images" / UNDATED_BUCKET,"img_001")

    res = save_annotations(
        tmp_path, tmp_path.parent, str(image),
        annotations=[{
            "subject": "bud",
            "rings": [
                [{"x": 10, "y": 20}, {"x": 110, "y": 20}, {"x": 110, "y": 220}],
                [{"x": 300, "y": 300}, {"x": 340, "y": 300}, {"x": 340, "y": 340}],
            ],
        }],
    )
    assert res.get("count") == 1
    (ann,) = _saved(image)
    assert isinstance(ann.geometry, Polygon)
    assert len(ann.geometry.rings) == 2  # both occlusion-split regions survived, not just the first
    assert ann.geometry.rings[0] == [(10.0, 20.0), (110.0, 20.0), (110.0, 220.0)]
    assert ann.geometry.rings[1] == [(300.0, 300.0), (340.0, 300.0), (340.0, 340.0)]

    # Round-trip shape ([x,y] pairs, as the client projection's "rings" uses) also works.
    res2 = save_annotations(
        tmp_path, tmp_path.parent, str(image),
        annotations=[{"subject": "bud", "rings": [[[1, 2], [3, 2], [3, 4]]]}],
    )
    assert res2.get("count") == 1
    (ann2,) = _saved(image)
    assert ann2.geometry.rings == [[(1.0, 2.0), (3.0, 2.0), (3.0, 4.0)]]

    # "rings" wins over "points"/"bbox" when more than one is present (never less complete).
    res3 = save_annotations(
        tmp_path, tmp_path.parent, str(image),
        annotations=[{
            "subject": "bud",
            "rings": [[{"x": 1, "y": 2}, {"x": 3, "y": 2}, {"x": 3, "y": 4}]],
            "points": [[10, 20], [110, 20], [110, 220]],
            "bbox": [10, 20, 110, 220],
        }],
    )
    assert res3.get("count") == 1
    (ann3,) = _saved(image)
    assert ann3.geometry.rings == [[(1.0, 2.0), (3.0, 2.0), (3.0, 4.0)]]


# (h) a geometry-less-only image carries the subject and is admitted; the detection loader reads
# no target from it and refuses it by name.
def test_geometryless_only_image_is_refused_by_the_loader_that_reads_no_target_from_it(tmp_path):
    """Never trained as a fabricated zero-object negative: admission asks whether the document
    carries the subject, and the loader owns which geometries answer for its measurement."""
    from tcip_mcp.pipelines.data.label_queries import admit

    _write_registry(tmp_path, Subject(name="bud"))
    images_dir = tmp_path / "images" / UNDATED_BUCKET
    label_image(_write_image(images_dir, "boxed"),
                [Annotation(subject="bud", geometry=BBox(10, 10, 40, 40))], 640, 480)
    # geomless: a bud annotation with NO geometry (an image-level label, not a box).
    label_image(_write_image(images_dir, "geomless"), [Annotation(subject="bud")], 640, 480)

    admitted = admit(images_dir, scope=registry_scope(images_dir, "bud"))
    assert [record.member for record in admitted.records] == ["boxed", "geomless"]

    with pytest.raises(ValueError, match="only in geometries a detection loader does not read"):
        dataset_over("detection", images_dir, subject="bud")

    # Admits valid work: the image whose document carries a box still trains.
    ds = dataset_over("detection", images_dir, subject="bud", members=["boxed"])
    assert [Path(s).stem for s in ds.stems] == ["boxed"]


# (i) eval accumulates every per-image record into one score, so a subject must carry the same
# category id across images: records_from_annotation must honor a passed global name_id.
def test_records_from_annotation_honors_a_global_name_id():
    from tcip_mcp.pipelines.training.evaluation import records_from_annotation
    from tcip_annotation.state import Annotation as Ann
    from tcip_annotation.state import BBox as B

    name_id = {"bush": 1, "bud": 2}  # one global map
    # An image whose only subject is bud: a per-image-local map would give bud id 1; the global
    # map must keep it 2, matching every other image's bud.
    rec = records_from_annotation(
        [Ann(subject="bud", geometry=B(0, 0, 10, 10))],
        [Ann(subject="bud", geometry=B(0, 0, 10, 10), score=0.9)],
        width=100, height=100, name_id=name_id)
    assert {r["category_id"] for r in rec["gt"]} == {2}
    assert {r["category_id"] for r in rec["dt"]} == {2}


def test_uppercase_extension_image_still_yields_boxes(tmp_path):
    """The detection loader matches images by their real on-disk name, not a case-normalized one:
    real drone frames use an uppercase .JPG, and a fabricated ".jpg" match would silently yield
    zero boxes."""
    _write_registry(tmp_path, Subject(name="bud"))
    images_dir = tmp_path / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    Image.new("RGB", (640, 480), color=(128, 128, 128)).save(images_dir / "IMG_1.JPG")
    label_image(images_dir / "IMG_1.JPG",
                [Annotation(subject="bud", geometry=BBox(10, 10, 40, 40))], 640, 480)

    ds = dataset_over("detection", str(images_dir), subject="bud")
    assert ds.num_samples == 1
    _img, target = ds[0]
    assert target["boxes"].shape[0] == 1  # the bud box survived the name match
    assert len(target["labels"]) == 1
