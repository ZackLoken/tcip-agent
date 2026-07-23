"""Phase 2.1c — the web inference job runs through the tcip pipeline GenericPredictor
(one detector code path), not a separate ultralytics+SAHI stack."""

import pytest


def test_write_predictions_json_roundtrip_and_negative(tmp_path):
    import json

    from tcip_annotation import json_io
    from tcip_mcp import class_registry
    from tcip_mcp.class_registry import ClassRegistry, Subject
    from tcip_mcp.pipelines.postprocessing.export import write_predictions_json

    id_map = class_registry.assign_class_ids(
        ClassRegistry(subjects=(Subject(name="catkin"),)), "catkin")  # {catkin: 0}
    p = tmp_path / "img.json"
    write_predictions_json(p, {
        "width": 100, "height": 100,
        "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9], "labels": [1], "count": 1,
    }, id_map=id_map)
    data = json.loads(p.read_text())
    ann = data["annotations"][0]
    assert ann["subject"] == "catkin"                       # 1-indexed label 1 -> id 0 -> "catkin"
    assert ann["bbox"] == [10.0, 10.0, 20.0, 20.0]
    assert ann["score"] == pytest.approx(0.9)
    preds = json_io.read_annotations(p)                     # symmetric read
    assert len(preds) == 1 and preds[0].score == pytest.approx(0.9)

    # Negative invariant: a zero-detection image still yields an {"annotations": []} record.
    neg = tmp_path / "empty.json"
    write_predictions_json(neg, {"width": 100, "height": 100,
                                 "boxes": [], "scores": [], "labels": [], "count": 0})
    assert json.loads(neg.read_text())["annotations"] == []


def test_web_worker_uses_generic_predictor_and_writes_json(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from PIL import Image

    from tcip_web.routes.inference import InferenceJob, _worker

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "img.jpg")
    out_dir = tmp_path / "out"
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")

    captured = {}

    class FakePredictor:
        def __init__(self, checkpoint_path=None, **kwargs):
            captured["checkpoint"] = checkpoint_path
            captured["kwargs"] = kwargs

        def predict_batch(self, paths, tile=False, tile_size=224, overlap=0.2, **kw):
            captured["tile"] = tile
            captured["postprocess"] = kw.get("postprocess")
            return [{"image": p, "width": 100, "height": 100,
                     "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9], "labels": [1], "count": 1}
                    for p in paths]

    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", FakePredictor)

    job = InferenceJob(
        job_id="t", checkpoint_path=str(ckpt), images_dir=str(images_dir),
        output_dir=str(out_dir), sahi=True, conf=0.25, iou=0.7,
        slice_hw=(640, 640), overlap=0.2, postprocess="nmm",
    )
    _worker(job)

    assert job.status == "completed"
    assert job.done == 1 and job.total == 1
    assert captured["checkpoint"] == str(ckpt)
    assert captured["tile"] is True                 # sahi=True -> pipeline tiling
    assert captured["postprocess"] == "nmm"         # the GUI's tile-merge choice reaches inference
    import json
    obj = json.loads((out_dir / "img.json").read_text())["annotations"][0]
    assert obj["subject"] == "0"                     # no recorded id_map -> id 0 stringified honestly
    assert obj["score"] == pytest.approx(0.9)        # per-object confidence preserved
    assert obj["bbox"] == [10.0, 10.0, 20.0, 20.0]   # pixel COCO xywh from xyxy [10,10,30,30]
