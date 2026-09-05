"""The name-based annotation schema: subjects, never integer class ids, on disk or in memory.

Fail-before tests that pin the measurement-critical invariants of the flip to name-based labels: the
registry decodes its own labels, a geometry-less annotation round-trips without collapsing to a
negative, every id consumer rests on one ``assign_class_ids`` map, negatives key through a threaded
subject, the loader filters by subject + geometry, decode inverts the recorded map, and authoring
refuses a subjectless label. Each builds its own dataset (no shared fixture) so the pre-change
baseline harness runs clean.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox, Polygon
from tcip_mcp import class_registry
from tcip_mcp.class_registry import ClassRegistry, Subject
from tcip_mcp.dataset_layout import record_image_statuses, status_bucket


def _write_image(images_dir: Path, stem: str, size=(640, 480)) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(128, 128, 128)).save(images_dir / f"{stem}.jpg")


def _write_registry(root: Path, *subjects: Subject) -> ClassRegistry:
    registry = ClassRegistry(subjects=tuple(subjects))
    class_registry.write_registry(root / "classes.json", registry)
    return registry


# (a) a registry decodes its own labels after the flip.
def test_registry_decodes_its_own_labels(tmp_path):
    from tcip_mcp.pipelines.data.label_queries import assemble_coco

    registry = _write_registry(tmp_path, Subject(name="bud"))
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "annotations"
    _write_image(images_dir, "img_001")
    labels_dir.mkdir(parents=True)
    json_io.write_annotations(
        labels_dir / "img_001.json",
        [Annotation(subject="bud", geometry=BBox(10, 10, 40, 40))], 640, 480)

    id_map = class_registry.assign_class_ids(registry, "bud")
    coco = assemble_coco(labels_dir, images_dir, subject="bud", date=None, id_map=id_map)

    # The COCO categories are the assign_class_ids map, and every emitted annotation decodes back to
    # the name its label carried: the registry reads its own labels without guessing.
    assert {c["name"]: c["id"] for c in coco["categories"]} == id_map
    inv = class_registry.decode_class_ids(id_map)
    assert coco["annotations"], "the labeled image produced no COCO annotation"
    assert all(inv[a["category_id"]] == "bud" for a in coco["annotations"])


# (b) a geometry-less annotation round-trips and its image is not collapsed to empty/negative.
def test_geometryless_annotation_roundtrips_and_marks_image_annotated(tmp_path):
    from tcip_mcp.pipelines.data.label_queries import assemble_coco

    registry = _write_registry(tmp_path, Subject(name="bud"))
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "annotations"
    _write_image(images_dir, "img_001")
    labels_dir.mkdir(parents=True)
    json_io.write_annotations(
        labels_dir / "img_001.json", [Annotation(subject="bud")], 640, 480)

    # Round-trips losslessly: subject preserved, geometry None.
    back = json_io.read_annotations(str(labels_dir / "img_001.json"))
    assert len(back) == 1 and back[0].subject == "bud" and back[0].geometry is None

    id_map = class_registry.assign_class_ids(registry, "bud")
    coco = assemble_coco(labels_dir, images_dir, subject="bud", date=None, id_map=id_map)
    # The image is annotated (it carries a subject annotation), so it is present as an image and is
    # not collapsed to an empty negative; the geometry-less label just has no detection target.
    assert [im["file_name"] for im in coco["images"]] == ["img_001.jpg"]
    assert coco["annotations"] == []


# (c) loader.num_classes == class_registry.num_classes == len(assemble_coco categories), one map.
def test_num_classes_agree_on_one_assign_class_ids_map(tmp_path):
    from tcip_mcp.pipelines.data.datasets import build_dataset
    from tcip_mcp.pipelines.data.label_queries import assemble_coco

    registry = _write_registry(tmp_path, Subject(name="bud"))
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "annotations"
    for stem in ("a", "b"):
        _write_image(images_dir, stem)
    labels_dir.mkdir(parents=True)
    for stem in ("a", "b"):
        json_io.write_annotations(
            labels_dir / f"{stem}.json",
            [Annotation(subject="bud", geometry=BBox(10, 10, 40, 40))], 640, 480)

    id_map = class_registry.assign_class_ids(registry, "bud")
    coco = assemble_coco(labels_dir, images_dir, subject="bud", date=None, id_map=id_map)
    ds = build_dataset("detection", images_dir=str(images_dir), labels_dir=str(labels_dir),
                       subject="bud")

    assert ds.num_classes == class_registry.num_classes(registry, "bud") == len(coco["categories"])
    assert len(id_map) == ds.num_classes == 1


# (d) confirmed_negative_names recovers negatives and refuses (not silent-empty) with no subject.
def test_confirmed_negatives_thread_subject_and_refuse_when_unthreaded(tmp_path):
    from tcip_mcp.pipelines.data.label_queries import confirmed_negative_names

    labels_dir = tmp_path / "annotations" / "2-11-26"
    labels_dir.mkdir(parents=True)
    # A human confirmed img_009 an empty negative for bud on this date.
    record_image_statuses(tmp_path, status_bucket("bud", "2-11-26"),
                          {"img_009.jpg": "negative"}, recorded_by="user:breeder")

    got = confirmed_negative_names(labels_dir, subject="bud", date="2-11-26")
    assert got == {"img_009.jpg"}

    # A different subject's bucket is not this subject's negative (scoping holds).
    assert confirmed_negative_names(labels_dir, subject="bush", date="2-11-26") == set()

    # Unthreaded subject with negatives present: refuse loudly rather than drop the human's work.
    with pytest.raises(ValueError):
        confirmed_negative_names(labels_dir, subject=None, date="2-11-26")


# (e) geometry-less + wrong-subject annotations are excluded from a detection run's targets.
def test_loader_filters_by_subject_and_geometry(tmp_path):
    import torch

    from tcip_mcp.pipelines.data.datasets import build_dataset

    _write_registry(tmp_path, Subject(name="bud"), Subject(name="bush"))
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "annotations"
    _write_image(images_dir, "img_001")
    labels_dir.mkdir(parents=True)
    json_io.write_annotations(
        labels_dir / "img_001.json",
        [
            Annotation(subject="bud", geometry=BBox(10, 10, 40, 40)),   # a legitimate target
            Annotation(subject="bush", geometry=BBox(50, 50, 90, 90)),      # wrong subject -> drop
            Annotation(subject="bud"),                                   # geometry-less -> drop
        ],
        640, 480)

    ds = build_dataset("detection", images_dir=str(images_dir), labels_dir=str(labels_dir),
                       subject="bud")
    assert ds.num_classes == 1
    _img, target = ds[0]
    # Only the one legitimate bud box survives; the wrong-subject and geometry-less rows are gone.
    assert target["boxes"].shape[0] == 1
    assert torch.equal(target["labels"], torch.tensor([1], dtype=torch.int64))  # 0-idx bud +1 bg


# (f) the loader's assign_class_ids map == decode_class_ids of the recorded operating_point map.
def test_decode_inverts_the_recorded_map(tmp_path):
    from tcip_mcp.pipelines.postprocessing.export import write_predictions_json

    registry = _write_registry(tmp_path, Subject(name="bud"))
    id_map = class_registry.assign_class_ids(registry, "bud")  # the run's single map

    # A prediction with a 1-indexed detector label decodes to its name through the recorded map.
    out = tmp_path / "pred.json"
    write_predictions_json(
        out, {"boxes": [[10, 10, 40, 40]], "scores": [0.9], "labels": [1], "width": 640, "height": 480},
        created_by="model:x", subject="bud", attribute=None, id_map=id_map)
    preds = json_io.read_annotations(str(out))
    assert len(preds) == 1
    inv = class_registry.decode_class_ids(id_map)
    # loader-side map (id_map) inverted == the name the recorded-map decode wrote on disk.
    assert preds[0].subject == inv[0] == "bud"


# (g) save_annotations refuses a missing subject.
def test_save_annotations_refuses_missing_subject(tmp_path):
    from tcip_mcp.tools.annotation_tools import save_annotations

    images_dir = tmp_path / "images"
    _write_image(images_dir, "img_001")
    img = str(images_dir / "img_001.jpg")

    res = save_annotations(img, annotations=[{"bbox": [10, 10, 40, 40]}], path=str(tmp_path / "x.json"))
    assert "error" in res and "subject" in res["error"]
    assert not (tmp_path / "x.json").exists()

    # A real subject still writes (the rail admits valid work).
    ok = save_annotations(img, annotations=[{"subject": "bud", "bbox": [10, 10, 40, 40]}],
                          path=str(tmp_path / "y.json"))
    assert ok.get("count") == 1 and (tmp_path / "y.json").is_file()


def test_save_annotations_refuses_an_unrecognized_fmt(tmp_path):
    """An fmt outside {json, coco} is a named refusal, not an uncaught ValueError from the writer
    this tool calls into."""
    from tcip_mcp.tools.annotation_tools import save_annotations

    images_dir = tmp_path / "images"
    _write_image(images_dir, "img_001")
    img = str(images_dir / "img_001.jpg")

    res = save_annotations(
        img, annotations=[{"subject": "bud", "bbox": [10, 10, 40, 40]}],
        fmt="xml", path=str(tmp_path / "x.json"))
    assert "error" in res and "fmt" in res["error"]
    assert not (tmp_path / "x.json").exists()


# (g2) save_annotations prefers points over bbox (aligned with the web converters), so a payload
# carrying both geometries writes the polygon, never collapsing it to a box-only record (which
# would double-count against the polygon's own derived box on the next load).
def test_save_annotations_prefers_points_over_bbox(tmp_path):
    import json

    from tcip_mcp.tools.annotation_tools import save_annotations

    images_dir = tmp_path / "images"
    _write_image(images_dir, "img_001")
    img = str(images_dir / "img_001.jpg")
    out = tmp_path / "both.json"

    res = save_annotations(
        img,
        annotations=[{
            "subject": "bud",
            "points": [[10, 20], [110, 20], [110, 220]],
            "bbox": [10, 20, 110, 220],
        }],
        path=str(out),
    )
    assert res.get("count") == 1

    (ann,) = json_io.read_annotations(str(out))
    assert isinstance(ann.geometry, Polygon)  # the polygon won; not collapsed to a box
    # "points" is the single-ring input key, wrapped as the one ring it is. A caller with more
    # than one ring (occlusion-split) uses "rings" instead (see test_save_annotations_accepts_rings).
    assert ann.geometry.rings == [[(10.0, 20.0), (110.0, 20.0), (110.0, 220.0)]]
    obj = json.loads(out.read_text())["annotations"][0]
    assert "segmentation" in obj  # written as a polygon (its derived bbox rides along)

    # An empty points list falls through to bbox (truthy check, matching the web converters) rather
    # than saving a degenerate Polygon(rings=[[]]) while the tool reports success.
    box_out = tmp_path / "emptypts.json"
    res2 = save_annotations(
        img,
        annotations=[{"subject": "bud", "points": [], "bbox": [10, 20, 110, 220]}],
        path=str(box_out),
    )
    assert res2.get("count") == 1
    (box_ann,) = json_io.read_annotations(str(box_out))
    assert isinstance(box_ann.geometry, BBox)  # empty points -> saved as the box, not dropped


# (g3) segment_prompt's own output is multi-ring ({x,y} dict vertices); an occlusion-split mask
# accepted from it must not silently save as a geometry-less annotation. Both ring-vertex shapes
# this module produces ({x,y} dicts from segment_prompt, [x,y] pairs from _ann_dict's read side)
# must round-trip through the write door.
def test_save_annotations_accepts_rings(tmp_path):
    from tcip_mcp.tools.annotation_tools import save_annotations

    images_dir = tmp_path / "images"
    _write_image(images_dir, "img_001")
    img = str(images_dir / "img_001.jpg")

    # segment_prompt's own output shape: [[{x,y}, ...], ...], one ring per connected region.
    out = tmp_path / "rings.json"
    res = save_annotations(
        img,
        annotations=[{
            "subject": "bud",
            "rings": [
                [{"x": 10, "y": 20}, {"x": 110, "y": 20}, {"x": 110, "y": 220}],
                [{"x": 300, "y": 300}, {"x": 340, "y": 300}, {"x": 340, "y": 340}],
            ],
        }],
        path=str(out),
    )
    assert res.get("count") == 1
    (ann,) = json_io.read_annotations(str(out))
    assert isinstance(ann.geometry, Polygon)
    assert len(ann.geometry.rings) == 2  # both occlusion-split regions survived, not just the first
    assert ann.geometry.rings[0] == [(10.0, 20.0), (110.0, 20.0), (110.0, 220.0)]
    assert ann.geometry.rings[1] == [(300.0, 300.0), (340.0, 300.0), (340.0, 340.0)]

    # Round-trip shape ([x,y] pairs, as _ann_dict's own "rings" reader output uses) also works.
    out2 = tmp_path / "rings_listshape.json"
    res2 = save_annotations(
        img,
        annotations=[{"subject": "bud", "rings": [[[1, 2], [3, 2], [3, 4]]]}],
        path=str(out2),
    )
    assert res2.get("count") == 1
    (ann2,) = json_io.read_annotations(str(out2))
    assert ann2.geometry.rings == [[(1.0, 2.0), (3.0, 2.0), (3.0, 4.0)]]

    # "rings" wins over "points"/"bbox" when more than one is present (never less complete).
    out3 = tmp_path / "rings_precedence.json"
    res3 = save_annotations(
        img,
        annotations=[{
            "subject": "bud",
            "rings": [[{"x": 1, "y": 2}, {"x": 3, "y": 2}, {"x": 3, "y": 4}]],
            "points": [[10, 20], [110, 20], [110, 220]],
            "bbox": [10, 20, 110, 220],
        }],
        path=str(out3),
    )
    assert res3.get("count") == 1
    (ann3,) = json_io.read_annotations(str(out3))
    assert ann3.geometry.rings == [[(1.0, 2.0), (3.0, 2.0), (3.0, 4.0)]]


# (h) the direct-json and COCO loader paths agree: a geometry-less-only image is a target on neither,
# so it is never trained as a fabricated zero-object negative (the two-paths-disagree measurement bug).
def test_geometryless_only_image_is_not_a_trainable_stem_on_either_path(tmp_path):
    from tcip_mcp.pipelines.data.label_queries import assemble_coco, trainable_stems

    registry = _write_registry(tmp_path, Subject(name="bud"))
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "annotations"
    for stem in ("boxed", "geomless"):
        _write_image(images_dir, stem)
    labels_dir.mkdir(parents=True)
    json_io.write_annotations(labels_dir / "boxed.json",
                              [Annotation(subject="bud", geometry=BBox(10, 10, 40, 40))], 640, 480)
    # geomless: a bud annotation with NO geometry (an image-level label, not a box).
    json_io.write_annotations(labels_dir / "geomless.json", [Annotation(subject="bud")], 640, 480)

    id_map = class_registry.assign_class_ids(registry, "bud")
    coco = assemble_coco(labels_dir, images_dir, subject="bud", date=None, id_map=id_map)
    stems_direct, _ = trainable_stems(labels_dir, images_dir, subject="bud", date=None)          # coco=None path
    stems_coco, _ = trainable_stems(labels_dir, images_dir, subject="bud", date=None, coco=coco)  # COCO path

    assert stems_direct == stems_coco == ["boxed"]  # the two paths agree; geomless-only is dropped


# (i) eval accumulates every per-image record into one COCOeval, so a subject must carry the same
# category id across images: records_from_annotation must honor a passed global name_id, not a
# per-image-local one (a local map pools distinct subjects into one class and corrupts per-class AP).
def test_records_from_annotation_honors_a_global_name_id():
    from tcip_mcp.pipelines.training.evaluation import records_from_annotation
    from tcip_annotation.state import Annotation as Ann
    from tcip_annotation.state import BBox as B

    name_id = {"bush": 1, "bud": 2}  # one global map
    # An image whose only subject is bud: a per-image-local map would give bud id 1; the global
    # map must keep it 2, matching every other image's bud.
    _iou, rec = records_from_annotation(
        [Ann(subject="bud", geometry=B(0, 0, 10, 10))],
        [Ann(subject="bud", geometry=B(0, 0, 10, 10), score=0.9)],
        width=100, height=100, name_id=name_id)
    assert {r["category_id"] for r in rec["gt"]} == {2}
    assert {r["category_id"] for r in rec["dt"]} == {2}


# (j) the COCO-assembled loader must match images by their real on-disk name, not a case-normalized
# one: real drone frames use an uppercase .JPG, and matching against a fabricated ".jpg" silently
# yields zero boxes (an all-empty training set). (Fails pre-fix only on a case-insensitive FS, where
# the miscased probe "succeeds"; the real-name match is correct on every platform.)
def test_uppercase_extension_image_still_yields_boxes(tmp_path):
    from tcip_mcp.pipelines.data.datasets import build_dataset

    _write_registry(tmp_path, Subject(name="bud"))
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "annotations"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (640, 480), color=(128, 128, 128)).save(images_dir / "IMG_1.JPG")  # uppercase ext
    labels_dir.mkdir(parents=True)
    json_io.write_annotations(
        labels_dir / "IMG_1.json",
        [Annotation(subject="bud", geometry=BBox(10, 10, 40, 40))], 640, 480)

    ds = build_dataset("detection", images_dir=str(images_dir), labels_dir=str(labels_dir),
                       subject="bud")
    assert ds.num_samples == 1
    _img, target = ds[0]
    assert target["boxes"].shape[0] == 1  # the bud box survived the COCO name match
    assert len(target["labels"]) == 1
