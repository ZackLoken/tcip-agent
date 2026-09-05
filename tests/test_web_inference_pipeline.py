"""The web inference job runs through the tcip pipeline GenericPredictor, the only detector
code path; there is no separate ultralytics+SAHI-specific one."""

import pytest


@pytest.fixture(autouse=True)
def _stub_checkpoint_verification(monkeypatch):
    """Every test in this module drives a stubbed predictor, not a real registered checkpoint;
    load_registered_checkpoint is stubbed to admit whatever path it is given, carrying the real
    file's own digest when one exists (some assertions here check that hash) and a fixed stand-in
    otherwise.
    """
    from pathlib import Path

    import tcip_mcp.model_registry as model_registry_mod

    from tests._verified_checkpoint_fixtures import stub_verified_checkpoint

    def _stub(path, *a, **kw):
        p = Path(path)
        sha = model_registry_mod._sha256_of_bytes(p.read_bytes()) if p.is_file() else "stub-sha256"
        return stub_verified_checkpoint(str(path), sha256=sha)

    monkeypatch.setattr(model_registry_mod, "load_registered_checkpoint", _stub)


def test_write_predictions_json_roundtrip_and_negative(tmp_path):
    import json

    from tcip_annotation import json_io
    from tcip_mcp import class_registry
    from tcip_mcp.class_registry import ClassRegistry, Subject
    from tcip_mcp.pipelines.postprocessing.export import write_predictions_json

    id_map = class_registry.assign_class_ids(
        ClassRegistry(subjects=(Subject(name="bud"),)), "bud")  # {bud: 0}
    p = tmp_path / "img.json"
    write_predictions_json(p, {
        "width": 100, "height": 100,
        "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9], "labels": [1], "count": 1,
    }, subject="bud", attribute=None, id_map=id_map)
    data = json.loads(p.read_text())
    ann = data["annotations"][0]
    assert ann["subject"] == "bud"                       # 1-indexed label 1 -> id 0 -> "bud"
    assert ann["bbox"] == [10.0, 10.0, 20.0, 20.0]
    assert ann["score"] == pytest.approx(0.9)
    preds = json_io.read_annotations(p)                     # symmetric read
    assert len(preds) == 1 and preds[0].score == pytest.approx(0.9)

    # Negative invariant: a zero-detection image still yields an {"annotations": []} record.
    neg = tmp_path / "empty.json"
    write_predictions_json(neg, {"width": 100, "height": 100,
                                 "boxes": [], "scores": [], "labels": [], "count": 0},
                           subject="bud", attribute=None)
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
        # A tiled run clears the worker's tile-geometry delivery gate only on a real basis for the
        # scale; this checkpoint persisted its own training geometry.
        train_tile_size = 640

        def __init__(self, checkpoint_path=None, **kwargs):
            captured["checkpoint"] = getattr(checkpoint_path, "path", checkpoint_path)
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
    import hashlib
    import json

    obj = json.loads((out_dir / "img.json").read_text())["annotations"][0]
    assert obj["subject"] == "0"                     # no recorded id_map -> id 0 stringified honestly
    assert obj["score"] == pytest.approx(0.9)        # per-object confidence preserved
    assert obj["bbox"] == [10.0, 10.0, 20.0, 20.0]   # pixel COCO xywh from xyxy [10,10,30,30]
    assert obj["created_by"] == f"model:m@{hashlib.sha256(ckpt.read_bytes()).hexdigest()[:12]}"


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
        # (predictor.config["data"]["subject"]), no classes.json needed, resolve_registry_id_map
        # synthesizes {subject: 0} for a plain single-class run.
        config = {"data": {"subject": "bud"}}

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
    assert obj["subject"] == "bud"  # resolved via id_map, not the raw index "0"


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
        # A recorded id_map naming class 1 "open", deliberately not what a live registry at
        # images_dir would derive (there is no classes.json under images_dir at all), so a pass
        # here can only mean the recorded map was used, never a registry fallback.
        config = {"data": {"subject": "bud", "attribute": "opening",
                           "id_map": {"closed": 0, "open": 1}}}

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

    import tcip_store
    from tcip_mcp.pipelines.resolution import sidecar_key

    obj = json.loads((out_dir / "img.json").read_text())["annotations"][0]
    # label 2 -> 0-indexed 1 -> the recorded map's "open"; a classified run's decoded name
    # lands under attributes[attribute], with subject carrying the object class.
    assert obj["subject"] == "bud"
    assert obj["attributes"] == {"opening": "open"}
    sidecar = tcip_store.read(sidecar_key(out_dir, "operating_point"))
    assert sidecar["id_map"] == {"closed": 0, "open": 1}
    assert sidecar["subject"] == "bud"
    assert sidecar["attribute"] == "opening"


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
        train_tile_size = 640  # a real basis for the tile scale, or the delivery gate refuses

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


def test_web_worker_runs_a_native_frame_tile_scale_and_forwards_its_recorded_resize(
        tmp_path, monkeypatch):
    """A checkpoint whose only geometry is its own uniform untiled training frame does justify a
    tile edge, and (after promotion) a real geometry reference of its own, so this door runs
    rather than refusing: the stamp carries the tier's own reference and the prediction call
    receives the resolved tile edge plus the checkpoint's recorded train-time resize."""
    pytest.importorskip("fastapi")
    from PIL import Image

    from tcip_web.routes.inference import InferenceJob, _worker

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "img.jpg")
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")
    captured = {}

    class FakeNativeFramePredictor:
        task = "detection"
        train_tile_size = None
        train_native_size = [64, 64]
        train_augmentation = {"resize": [128, 128]}

        def __init__(self, checkpoint_path=None, **kwargs):
            pass

        def predict_batch(self, paths, **kw):
            captured.update(kw)
            return [{"count": 0, "width": 100, "height": 100, "boxes": [], "scores": [], "labels": []}
                   for _ in paths]

    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", FakeNativeFramePredictor)

    job = InferenceJob(
        job_id="t4", checkpoint_path=str(ckpt), images_dir=str(images_dir),
        output_dir=str(tmp_path / "out"), tile=True, tile_source="explicit", conf=0.25, iou=0.7,
        slice_hw=(640, 640), overlap=0.2, postprocess="nms",
    )
    _worker(job)

    assert job.status == "completed"
    assert job.error is None
    assert captured.get("tile_size") == 64
    assert captured.get("tile_resize") == (128, 128)

    import tcip_store
    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE, sidecar_key

    sidecar = tcip_store.read(sidecar_key(job.output_dir, "operating_point"))
    tile_ref = sidecar["operating_point"]["tile_size"]["validated_against"]
    assert tile_ref not in (VALIDATED_FALSE, None)
    from tcip_mcp.pipelines.resolution import VALIDATED_NATIVE_FRAME_GEOMETRY

    assert tile_ref == VALIDATED_NATIVE_FRAME_GEOMETRY


def _stub_predictor_for_conf_source(monkeypatch, tmp_path):
    from PIL import Image

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "img.jpg")
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")

    class FakePredictor:
        config = {"data": {"subject": "bud"}}

        def __init__(self, checkpoint_path=None, **kwargs):
            pass

        def predict_batch(self, paths, tile=False, tile_size=224, overlap=0.2, **kw):
            return [{"image": p, "width": 100, "height": 100,
                     "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9], "labels": [1], "count": 1}
                    for p in paths]

    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", FakePredictor)
    return str(ckpt), str(images_dir)


def test_web_worker_stamps_explicit_conf_and_max_dets_source_at_the_platform_default(
    tmp_path, monkeypatch,
):
    """A caller-stated conf/max_dets equal to the platform default is stamped 'explicit', the same
    distinction tile/tile_size already carry, never silently read back as an untouched default."""
    from tcip_web.routes.inference import InferenceJob, _worker

    from tcip_mcp.pipelines.resolution import (
        DEFAULT_CONF, DEFAULT_MAX_DETS, read_operating_point_sidecar,
    )

    ckpt, images_dir = _stub_predictor_for_conf_source(monkeypatch, tmp_path)
    out_dir = tmp_path / "out"

    job = InferenceJob(
        job_id="conf-explicit", checkpoint_path=ckpt, images_dir=images_dir,
        output_dir=str(out_dir), tile=False, conf=DEFAULT_CONF, conf_stated=True,
        iou=0.7, slice_hw=(0, 0), overlap=0.2, max_dets=DEFAULT_MAX_DETS,
        max_dets_stated=True,
    )
    _worker(job)

    assert job.status == "completed"
    stamp = read_operating_point_sidecar(out_dir)
    assert stamp["operating_point"]["conf"]["source"] == "explicit"
    assert stamp["operating_point"]["max_dets"]["source"] == "explicit"


def test_web_worker_stamps_default_conf_and_max_dets_source_when_unstated(tmp_path, monkeypatch):
    """The rail must admit the ordinary, unstated launch: an omitted conf/max_dets still runs the
    pass at the platform default, unchanged from the explicit-at-default case, and its provenance
    says 'default' rather than 'explicit'."""
    from tcip_web.routes.inference import InferenceJob, _worker

    from tcip_mcp.pipelines.resolution import (
        DEFAULT_CONF, DEFAULT_MAX_DETS, read_operating_point_sidecar,
    )

    import json

    ckpt, images_dir = _stub_predictor_for_conf_source(monkeypatch, tmp_path)
    out_dir = tmp_path / "out"

    job = InferenceJob(
        job_id="conf-default", checkpoint_path=ckpt, images_dir=images_dir,
        output_dir=str(out_dir), tile=False, conf=DEFAULT_CONF, iou=0.7, slice_hw=(0, 0),
        overlap=0.2, max_dets=DEFAULT_MAX_DETS,
    )
    _worker(job)

    assert job.status == "completed"
    # The pass ran unchanged at the platform default: the one image's one detection landed.
    persisted = json.loads((out_dir / "img.json").read_text())["annotations"]
    assert len(persisted) == 1
    stamp = read_operating_point_sidecar(out_dir)
    assert stamp["operating_point"]["conf"]["source"] == "default"
    assert stamp["operating_point"]["max_dets"]["source"] == "default"


def test_web_worker_n_detections_agrees_with_the_persisted_document_on_a_degenerate_box(
    tmp_path, monkeypatch,
):
    """A box that collapses to zero width is dropped when the prediction file is written, so the
    job's own dropped-box counter and the persisted document must agree: one of the model's two
    raw detections dropped, exactly one kept on disk, never the model's raw output count."""
    pytest.importorskip("fastapi")
    import json

    from PIL import Image

    from tcip_web.routes.inference import InferenceJob, _worker

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "img.jpg")
    out_dir = tmp_path / "out"
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")

    class FakePredictor:
        train_tile_size = 640

        def __init__(self, checkpoint_path=None, **kwargs):
            pass

        def predict_batch(self, paths, tile=False, tile_size=224, overlap=0.2, **kw):
            return [{"image": p, "width": 100, "height": 100,
                     "boxes": [[10.0, 10.0, 30.0, 30.0], [50.0, 50.0, 50.0, 60.0]],
                     "scores": [0.9, 0.4], "labels": [1, 1], "count": 2}
                    for p in paths]

    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", FakePredictor)

    job = InferenceJob(
        job_id="degenerate", checkpoint_path=str(ckpt), images_dir=str(images_dir),
        output_dir=str(out_dir), tile=True, conf=0.25, iou=0.7,
        slice_hw=(640, 640), overlap=0.2, postprocess="nmm",
    )
    _worker(job)

    assert job.status == "completed"
    assert job.dropped_boxes == 1
    persisted = json.loads((out_dir / "img.json").read_text())["annotations"]
    assert len(persisted) == 1


def test_web_worker_fails_the_job_on_a_stem_collision(tmp_path):
    """``_list_images`` enumerates through ``image_utils.list_logical_images``, the same
    enumeration every reader shares: a bucket already holding two logical identities under one
    case-folded stem fails the job with the refusal's own message, through the worker's own
    ``except Exception`` handler, rather than crashing the worker thread."""
    from PIL import Image

    from tcip_web.routes.inference import InferenceJob, _worker

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "foo.jpg")
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "foo.png")
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")

    job = InferenceJob(
        job_id="collision", checkpoint_path=str(ckpt), images_dir=str(images_dir),
        output_dir=str(tmp_path / "out"), tile=False, conf=0.25, iou=0.7,
        slice_hw=(0, 0), overlap=0.2, postprocess="nms",
    )
    _worker(job)

    assert job.status == "failed"
    assert "logical image" in job.error
