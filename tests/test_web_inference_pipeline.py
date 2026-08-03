"""The web inference job runs through the tcip pipeline GenericPredictor, the only detector
code path; there is no separate ultralytics+SAHI-specific one."""

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
        output_dir=str(out_dir), tile=True, conf=0.25, iou=0.7,
        slice_hw=(640, 640), overlap=0.2, postprocess="nmm",
    )
    _worker(job)

    assert job.status == "completed"
    assert job.done == 1 and job.total == 1
    assert captured["checkpoint"] == str(ckpt)
    assert captured["tile"] is True                 # tile=True -> pipeline tiling
    assert captured["postprocess"] == "nmm"         # the GUI's tile-merge choice reaches inference
    import json
    obj = json.loads((out_dir / "img.json").read_text())["annotations"][0]
    assert obj["subject"] == "0"                     # no recorded id_map -> id 0 stringified honestly
    assert obj["score"] == pytest.approx(0.9)        # per-object confidence preserved
    assert obj["bbox"] == [10.0, 10.0, 20.0, 20.0]   # pixel COCO xywh from xyxy [10,10,30,30]


def test_web_worker_resolves_id_map_from_predictor_config(tmp_path, monkeypatch):
    """The GUI door reads subject/attribute off predictor.config["data"] the same way
    run_inference already does, and decodes predictions through the resolved id_map: never a raw
    index string."""
    pytest.importorskip("fastapi")
    from PIL import Image

    from tcip_web.routes.inference import InferenceJob, _worker

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "img.jpg")
    out_dir = tmp_path / "out"
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")

    class FakePredictor:
        # A single-subject detector's config, the same shape run_inference reads
        # (predictor.config["data"]["subject"]), no classes.json needed, _resolve_registry_id_map
        # synthesizes {subject: 0} for a plain single-class run.
        config = {"data": {"subject": "catkin"}}

        def __init__(self, checkpoint_path=None, **kwargs):
            pass

        def predict_batch(self, paths, tile=False, tile_size=224, overlap=0.2, **kw):
            return [{"image": p, "width": 100, "height": 100,
                     "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9], "labels": [1], "count": 1}
                    for p in paths]

    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", FakePredictor)

    job = InferenceJob(
        job_id="t2", checkpoint_path=str(ckpt), images_dir=str(images_dir),
        output_dir=str(out_dir), tile=False, conf=0.25, iou=0.7,
        slice_hw=(640, 640), overlap=0.2, postprocess="nms",
    )
    _worker(job)

    assert job.status == "completed"
    import json
    obj = json.loads((out_dir / "img.json").read_text())["annotations"][0]
    assert obj["subject"] == "catkin"  # resolved via id_map, not the raw index "0"


def test_web_worker_prefers_the_checkpoints_own_recorded_id_map(tmp_path, monkeypatch):
    """The GUI inference worker must not re-derive its id_map locally from the live registry,
    independently of run_inference's own resolution. Both doors call the same
    tcip_mcp.tools.inference_tools.resolve_decode_id_map: this proves the GUI door genuinely
    prefers a recorded map too, not just falls through to live-registry derivation."""
    pytest.importorskip("fastapi")
    from PIL import Image

    from tcip_web.routes.inference import InferenceJob, _worker

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "img.jpg")
    out_dir = tmp_path / "out"
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")

    class FakePredictor:
        # A recorded id_map naming class 1 "elongated", deliberately not what a live registry at
        # images_dir would derive (there is no classes.json under images_dir at all), so a pass
        # here can only mean the recorded map was used, never a registry fallback.
        config = {"data": {"subject": "catkin", "attribute": "elongation",
                           "id_map": {"dormant": 0, "elongated": 1}}}

        def __init__(self, checkpoint_path=None, **kwargs):
            pass

        def predict_batch(self, paths, tile=False, tile_size=224, overlap=0.2, **kw):
            return [{"image": p, "width": 100, "height": 100,
                     "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9], "labels": [2], "count": 1}
                    for p in paths]

    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", FakePredictor)

    job = InferenceJob(
        job_id="t3", checkpoint_path=str(ckpt), images_dir=str(images_dir),
        output_dir=str(out_dir), tile=False, conf=0.25, iou=0.7,
        slice_hw=(640, 640), overlap=0.2, postprocess="nms",
    )
    _worker(job)

    assert job.status == "completed"
    import json
    obj = json.loads((out_dir / "img.json").read_text())["annotations"][0]
    assert obj["subject"] == "elongated"  # label 2 -> 0-indexed 1 -> the recorded map's "elongated"
    sidecar = json.loads((out_dir / "operating_point.json").read_text())
    assert sidecar["id_map"] == {"dormant": 0, "elongated": 1}


def test_web_worker_runs_tiled_instance_seg_without_forcing_untiled(tmp_path, monkeypatch):
    """An instance_seg checkpoint launched with the GUI's tile checkbox checked runs tiled as
    requested: tiled inference now threads masks through the cross-tile reconstruction/merge, so
    this door no longer needs to silently override the breeder's own checkbox choice."""
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

    class FakeInstanceSegPredictor:
        task = "instance_seg"

        def __init__(self, checkpoint_path=None, **kwargs):
            pass

        def predict_batch(self, paths, tile=False, tile_size=224, overlap=0.2, **kw):
            captured["tile"] = tile
            return [{"image": p, "width": 100, "height": 100,
                     "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9], "labels": [1], "count": 1}
                    for p in paths]

    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", FakeInstanceSegPredictor)

    job = InferenceJob(
        job_id="t3", checkpoint_path=str(ckpt), images_dir=str(images_dir),
        output_dir=str(out_dir), tile=True, tile_source="explicit", conf=0.25, iou=0.7,
        slice_hw=(640, 640), overlap=0.2, postprocess="nms",
    )
    _worker(job)

    assert job.status == "completed"        # no crash
    assert captured["tile"] is True          # the breeder's own checkbox choice is honored
    assert job.tile is True
    assert job.tile_source == "explicit"     # never silently overridden to "default" anymore
