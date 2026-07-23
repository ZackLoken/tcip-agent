"""K13.5 slice 2c — the atomic name-based format flip.

Fail-before tests that pin the measurement-critical invariants of the flip: the registry decodes its
own labels, a geometry-less annotation round-trips without collapsing to a negative, every id
consumer rests on one ``assign_class_ids`` map, negatives key through a threaded subject, the loader
filters by subject + geometry, decode inverts the recorded map, and authoring refuses a subjectless
label. Each builds its own dataset (no shared fixture) so the pre-change baseline harness runs clean.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox
from tcip_mcp import class_registry
from tcip_mcp.class_registry import ClassRegistry, Subject


def _write_image(images_dir: Path, stem: str, size=(640, 480)) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(128, 128, 128)).save(images_dir / f"{stem}.jpg")


def _write_registry(root: Path, *subjects: Subject) -> ClassRegistry:
    registry = ClassRegistry(subjects=tuple(subjects))
    class_registry.write_registry(root / "classes.json", registry)
    return registry


# (a) a registry decodes its OWN labels after the flip.
def test_registry_decodes_its_own_labels(tmp_path):
    from tcip_mcp.pipelines.data.datasets import assemble_coco

    registry = _write_registry(tmp_path, Subject(name="catkin"))
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "annotations"
    _write_image(images_dir, "img_001")
    labels_dir.mkdir(parents=True)
    json_io.write_annotations(
        labels_dir / "img_001.json",
        [Annotation(subject="catkin", geometry=BBox(10, 10, 40, 40))], 640, 480)

    id_map = class_registry.assign_class_ids(registry, "catkin")
    coco = assemble_coco(labels_dir, images_dir, subject="catkin", id_map=id_map)

    # The COCO categories ARE the assign_class_ids map, and every emitted annotation decodes back to
    # the name its label carried — the registry reads its own labels without guessing.
    assert {c["name"]: c["id"] for c in coco["categories"]} == id_map
    inv = class_registry.decode_class_ids(id_map)
    assert coco["annotations"], "the labeled image produced no COCO annotation"
    assert all(inv[a["category_id"]] == "catkin" for a in coco["annotations"])


# (b) a geometry-less annotation round-trips and its image is NOT collapsed to empty/negative.
def test_geometryless_annotation_roundtrips_and_marks_image_annotated(tmp_path):
    from tcip_mcp.pipelines.data.datasets import assemble_coco

    registry = _write_registry(tmp_path, Subject(name="catkin"))
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "annotations"
    _write_image(images_dir, "img_001")
    labels_dir.mkdir(parents=True)
    json_io.write_annotations(
        labels_dir / "img_001.json", [Annotation(subject="catkin")], 640, 480)

    # Round-trips losslessly: subject preserved, geometry None.
    back = json_io.read_annotations(str(labels_dir / "img_001.json"))
    assert len(back) == 1 and back[0].subject == "catkin" and back[0].geometry is None

    id_map = class_registry.assign_class_ids(registry, "catkin")
    coco = assemble_coco(labels_dir, images_dir, subject="catkin", id_map=id_map)
    # The image is annotated (it carries a subject annotation), so it is present as an image and is
    # NOT collapsed to an empty negative; the geometry-less label just has no detection target.
    assert [im["file_name"] for im in coco["images"]] == ["img_001.jpg"]
    assert coco["annotations"] == []


# (c) loader.num_classes == class_registry.num_classes == len(assemble_coco categories), one map.
def test_num_classes_agree_on_one_assign_class_ids_map(tmp_path):
    from tcip_mcp.pipelines.data.datasets import assemble_coco, build_dataset

    registry = _write_registry(tmp_path, Subject(name="catkin"))
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "annotations"
    for stem in ("a", "b"):
        _write_image(images_dir, stem)
    labels_dir.mkdir(parents=True)
    for stem in ("a", "b"):
        json_io.write_annotations(
            labels_dir / f"{stem}.json",
            [Annotation(subject="catkin", geometry=BBox(10, 10, 40, 40))], 640, 480)

    id_map = class_registry.assign_class_ids(registry, "catkin")
    coco = assemble_coco(labels_dir, images_dir, subject="catkin", id_map=id_map)
    ds = build_dataset("detection", images_dir=str(images_dir), labels_dir=str(labels_dir),
                       subject="catkin")

    assert ds.num_classes == class_registry.num_classes(registry, "catkin") == len(coco["categories"])
    assert len(id_map) == ds.num_classes == 1


# (d) confirmed_negative_names recovers negatives, AND refuses (not silent-empty) with no subject.
def test_confirmed_negatives_thread_subject_and_refuse_when_unthreaded(tmp_path):
    import json

    from tcip_mcp.pipelines.data.datasets import confirmed_negative_names

    labels_dir = tmp_path / "annotations" / "2-11-26"
    labels_dir.mkdir(parents=True)
    status = tmp_path / ".tcip" / "state" / "image_status.json"
    status.parent.mkdir(parents=True)
    # A human confirmed img_009 an empty negative for catkin on this date.
    status.write_text(json.dumps({"catkin/2-11-26": {"img_009.jpg": "negative"}}))

    got = confirmed_negative_names(labels_dir, subject="catkin", date="2-11-26")
    assert got == {"img_009.jpg"}

    # A different subject's bucket is not this subject's negative (scoping holds).
    assert confirmed_negative_names(labels_dir, subject="bush", date="2-11-26") == set()

    # Unthreaded subject with negatives present: REFUSE loudly rather than drop the human's work.
    with pytest.raises(ValueError):
        confirmed_negative_names(labels_dir, subject=None, date="2-11-26")


# (e) geometry-less + wrong-subject annotations are excluded from a detection run's targets.
def test_loader_filters_by_subject_and_geometry(tmp_path):
    import torch

    from tcip_mcp.pipelines.data.datasets import build_dataset

    _write_registry(tmp_path, Subject(name="catkin"), Subject(name="bush"))
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "annotations"
    _write_image(images_dir, "img_001")
    labels_dir.mkdir(parents=True)
    json_io.write_annotations(
        labels_dir / "img_001.json",
        [
            Annotation(subject="catkin", geometry=BBox(10, 10, 40, 40)),   # a legitimate target
            Annotation(subject="bush", geometry=BBox(50, 50, 90, 90)),      # wrong subject -> drop
            Annotation(subject="catkin"),                                   # geometry-less -> drop
        ],
        640, 480)

    ds = build_dataset("detection", images_dir=str(images_dir), labels_dir=str(labels_dir),
                       subject="catkin")
    assert ds.num_classes == 1
    _img, target = ds[0]
    # Only the one legitimate catkin box survives; the wrong-subject and geometry-less rows are gone.
    assert target["boxes"].shape[0] == 1
    assert torch.equal(target["labels"], torch.tensor([1], dtype=torch.int64))  # 0-idx catkin +1 bg


# (f) the loader's assign_class_ids map == decode_class_ids of the recorded operating_point map.
def test_decode_inverts_the_recorded_map(tmp_path):
    from tcip_mcp.pipelines.postprocessing.export import write_predictions_json

    registry = _write_registry(tmp_path, Subject(name="catkin"))
    id_map = class_registry.assign_class_ids(registry, "catkin")  # the run's single map

    # A prediction with a 1-indexed detector label decodes to its NAME through the RECORDED map.
    out = tmp_path / "pred.json"
    write_predictions_json(
        out, {"boxes": [[10, 10, 40, 40]], "scores": [0.9], "labels": [1], "width": 640, "height": 480},
        created_by="model:x", id_map=id_map)
    preds = json_io.read_annotations(str(out))
    assert len(preds) == 1
    inv = class_registry.decode_class_ids(id_map)
    # loader-side map (id_map) inverted == the name the recorded-map decode wrote on disk.
    assert preds[0].subject == inv[0] == "catkin"


# (g) save_annotations REFUSES a missing subject.
def test_save_annotations_refuses_missing_subject(tmp_path):
    from tcip_mcp.tools.annotation_tools import save_annotations

    images_dir = tmp_path / "images"
    _write_image(images_dir, "img_001")
    img = str(images_dir / "img_001.jpg")

    res = save_annotations(img, annotations=[{"bbox": [10, 10, 40, 40]}], path=str(tmp_path / "x.json"))
    assert "error" in res and "subject" in res["error"]
    assert not (tmp_path / "x.json").exists()

    # A real subject still writes (the rail admits valid work).
    ok = save_annotations(img, annotations=[{"subject": "catkin", "bbox": [10, 10, 40, 40]}],
                          path=str(tmp_path / "y.json"))
    assert ok.get("count") == 1 and (tmp_path / "y.json").is_file()


# (h) the direct-json and COCO loader paths agree: a geometry-less-only image is a target on neither,
# so it is never trained as a fabricated zero-object negative (the two-paths-disagree measurement bug).
def test_geometryless_only_image_is_not_a_trainable_stem_on_either_path(tmp_path):
    from tcip_mcp.pipelines.data.datasets import assemble_coco, trainable_stems

    registry = _write_registry(tmp_path, Subject(name="catkin"))
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "annotations"
    for stem in ("boxed", "geomless"):
        _write_image(images_dir, stem)
    labels_dir.mkdir(parents=True)
    json_io.write_annotations(labels_dir / "boxed.json",
                              [Annotation(subject="catkin", geometry=BBox(10, 10, 40, 40))], 640, 480)
    # geomless: a catkin annotation with NO geometry (an image-level label, not a box).
    json_io.write_annotations(labels_dir / "geomless.json", [Annotation(subject="catkin")], 640, 480)

    id_map = class_registry.assign_class_ids(registry, "catkin")
    coco = assemble_coco(labels_dir, images_dir, subject="catkin", id_map=id_map)
    stems_direct, _ = trainable_stems(labels_dir, images_dir, subject="catkin")          # coco=None path
    stems_coco, _ = trainable_stems(labels_dir, images_dir, subject="catkin", coco=coco)  # COCO path

    assert stems_direct == stems_coco == ["boxed"]  # the two paths agree; geomless-only is dropped


# (i) eval accumulates every per-image record into one COCOeval, so a subject must carry the SAME
# category id across images — records_from_annotation must honor a passed global name_id, not a
# per-image-local one (a local map pools distinct subjects into one class and corrupts per-class AP).
def test_records_from_annotation_honors_a_global_name_id():
    from tcip_mcp.pipelines.training.evaluation import records_from_annotation
    from tcip_annotation.state import Annotation as Ann
    from tcip_annotation.state import BBox as B

    name_id = {"bush": 1, "catkin": 2}  # one global map
    # An image whose only subject is catkin: a per-image-local map would give catkin id 1; the global
    # map must keep it 2, matching every other image's catkin.
    _iou, rec = records_from_annotation(
        [Ann(subject="catkin", geometry=B(0, 0, 10, 10))],
        [Ann(subject="catkin", geometry=B(0, 0, 10, 10), score=0.9)],
        width=100, height=100, name_id=name_id)
    assert {r["category_id"] for r in rec["gt"]} == {2}
    assert {r["category_id"] for r in rec["dt"]} == {2}
