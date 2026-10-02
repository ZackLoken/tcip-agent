"""The web inference job runs through the tcip pipeline GenericPredictor, the only detector
code path; there is no separate ultralytics+SAHI-specific one."""

from pathlib import Path

import pytest

from tcip_mcp.pipelines.execution import Stated


def _checkpoint(project: Path, **kwargs) -> Path:
    """A checkpoint registered in ``project``, completed by a real run (``kwargs`` are its own)."""
    from tests._verified_checkpoint_fixtures import project_checkpoint

    return Path(project_checkpoint(project, **kwargs))


def test_encode_predictions_roundtrip_and_negative(tmp_path):
    import json

    from tcip_annotation import json_io
    from tcip_mcp.pipelines.data.label_queries import registry_scope
    from tcip_mcp.pipelines.postprocessing.export import encode_predictions

    scope = registry_scope(tmp_path, "bud")
    encoded, _dropped = encode_predictions({
        "image": "img.jpg", "width": 100, "height": 100,
        "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9], "labels": [1], "count": 1,
    }, scope=scope)
    ann = json.loads(encoded)["annotations"][0]
    assert ann["subject"] == "bud"                       # the scope's one subject
    assert ann["bbox"] == [10.0, 10.0, 20.0, 20.0]
    assert ann["score"] == pytest.approx(0.9)
    preds = json_io.annotations_from_bytes(encoded, source="img.json")  # symmetric read
    assert len(preds) == 1 and preds[0].score == pytest.approx(0.9)

    # Negative invariant: a zero-detection image still yields an {"annotations": []} record.
    empty, _dropped = encode_predictions({"image": "empty.jpg", "width": 100, "height": 100,
                                          "boxes": [], "scores": [], "labels": [], "count": 0},
                                         scope=scope)
    assert json.loads(empty)["annotations"] == []


def test_web_worker_uses_generic_predictor_and_writes_json(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from PIL import Image

    from tcip_web.routes.inference import InferenceJob, _worker

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "img.jpg")
    out_dir = tmp_path / "out"
    ckpt = _checkpoint(tmp_path)

    captured = {}

    class FakePredictor:
        # A tiled run clears the worker's tile-geometry delivery gate only on a real basis for the
        # scale; this checkpoint persisted its own training geometry.
        train_tile_size = 640

        def __init__(self, checkpoint_path=None, **kwargs):
            captured["checkpoint"] = getattr(checkpoint_path, "path", checkpoint_path)
            captured["kwargs"] = kwargs

        def predict_batch(self, paths, execution=None, **kw):
            captured["execution"] = execution
            return [{"image": p, "width": 100, "height": 100,
                     "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9], "labels": [1],
                     "count": 1, "cap_hit": False} for p in paths]

    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", FakePredictor)

    job = InferenceJob(
        project=str(tmp_path), job_id="t", checkpoint_path=str(ckpt), images_dir=str(images_dir),
        output_dir=str(out_dir), stated=Stated(
            tile=True, conf=0.25, cross_tile_nms=0.7, overlap=0.2, postprocess="nmm"),
    )
    _worker(job)

    assert job.status == "completed"
    assert job.done == 1 and job.total == 1
    assert captured["checkpoint"] == str(ckpt)
    assert captured["execution"].tile_size == 640    # tile=True -> pipeline tiling
    assert captured["execution"].postprocess == "nmm"  # the GUI's merge choice reaches inference
    import hashlib
    import json

    obj = json.loads((out_dir / "img.json").read_text())["annotations"][0]
    assert obj["subject"] == "bud"                   # decoded through the checkpoint's recorded map
    assert obj["score"] == pytest.approx(0.9)        # per-object confidence preserved
    assert obj["bbox"] == [10.0, 10.0, 20.0, 20.0]   # pixel COCO xywh from xyxy [10,10,30,30]
    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()[:12]
    assert obj["created_by"] == f"model:{ckpt.stem}@{digest}"


def test_web_worker_prefers_the_checkpoints_own_recorded_scope(tmp_path, monkeypatch):
    """The GUI inference worker decodes through the checkpoint's own recorded scope, never an
    order re-read from a live registry: no subjects.json exists under images_dir at all."""
    pytest.importorskip("fastapi")
    from PIL import Image

    from tcip_mcp import subject_registry as cr
    from tcip_web.routes.inference import InferenceJob, _worker

    opening = cr.Attribute("opening", "categorical", ("closed", "open"))
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "img.jpg")
    out_dir = tmp_path / "out"
    ckpt = _checkpoint(tmp_path, data={"num_channels": 3, "scope": {"subject": "bud"}},
                       registry=cr.SubjectRegistry(subjects=(
                           cr.Subject(name="bud", attributes=(opening,)),)))

    class FakePredictor:
        def __init__(self, checkpoint_path=None, **kwargs):
            pass

        def predict_batch(self, paths, execution=None, **kw):
            return [{"image": p, "width": 100, "height": 100,
                     "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9], "labels": [1],
                     "attributes": [[1]], "count": 1, "cap_hit": False} for p in paths]

    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", FakePredictor)

    job = InferenceJob(
        project=str(tmp_path), job_id="t3", checkpoint_path=str(ckpt), images_dir=str(images_dir),
        output_dir=str(out_dir), stated=Stated(
            tile=False, conf=0.25, cross_tile_nms=0.7, overlap=0.2, postprocess="nms"),
    )
    _worker(job)

    assert job.status == "completed"
    import json

    from tcip_mcp.buckets import read_bucket

    obj = json.loads((out_dir / "img.json").read_text())["annotations"][0]
    # id 1 of the recorded "opening" is "open", under attributes; subject carries the object.
    assert obj["subject"] == "bud"
    assert obj["attributes"] == {"opening": "open"}
    assert read_bucket(out_dir).scope.attributes == (opening,)


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
    ckpt = _checkpoint(tmp_path)

    captured = {}

    class FakeInstanceSegPredictor:
        task = "instance_seg"
        train_tile_size = 640  # a real basis for the tile scale, or the delivery gate refuses

        def __init__(self, checkpoint_path=None, **kwargs):
            pass

        def predict_batch(self, paths, execution=None, **kw):
            captured["execution"] = execution
            return [{"image": p, "width": 100, "height": 100,
                     "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9], "labels": [1],
                     "count": 1, "cap_hit": False} for p in paths]

    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", FakeInstanceSegPredictor)

    job = InferenceJob(
        project=str(tmp_path), job_id="t3", checkpoint_path=str(ckpt), images_dir=str(images_dir),
        output_dir=str(out_dir), stated=Stated(
            tile=True, conf=0.25, cross_tile_nms=0.7, overlap=0.2, postprocess="nms"),
    )
    _worker(job)

    assert job.status == "completed"        # no crash
    assert captured["execution"].tiled       # the breeder's own checkbox choice is honored
    assert job.stated.tile is True
    from tcip_mcp.buckets import read_bucket

    execution = read_bucket(out_dir).execution
    assert execution.tiled
    # The stated merge threshold is recorded as stated, never silently overridden to "default".
    assert execution.sources["cross_tile_nms"] == "explicit"


def test_web_worker_runs_a_native_frame_tile_scale_and_forwards_its_recorded_resize(
        tmp_path, monkeypatch):
    """A checkpoint whose only geometry is its own uniform untiled training frame does justify a
    tile edge, so this door runs rather than refusing: the record names that basis and the
    prediction call runs at the resolved tile edge plus the checkpoint's recorded train-time
    resize."""
    pytest.importorskip("fastapi")
    from PIL import Image

    from tcip_web.routes.inference import InferenceJob, _worker

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "img.jpg")
    ckpt = _checkpoint(tmp_path)
    captured = {}

    class FakeNativeFramePredictor:
        task = "detection"
        train_tile_size = None
        train_native_size = [64, 64]
        train_augmentation = {"resize": [128, 128]}

        def __init__(self, checkpoint_path=None, **kwargs):
            pass

        def predict_batch(self, paths, execution=None, **kw):
            captured["execution"] = execution
            return [{"image": p, "count": 0, "width": 100, "height": 100, "boxes": [],
                     "scores": [], "labels": [], "cap_hit": False} for p in paths]

    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", FakeNativeFramePredictor)

    job = InferenceJob(
        project=str(tmp_path), job_id="t4", checkpoint_path=str(ckpt), images_dir=str(images_dir),
        output_dir=str(tmp_path / "out"), stated=Stated(
            tile=True, conf=0.25, cross_tile_nms=0.7, overlap=0.2, postprocess="nms"),
    )
    _worker(job)

    assert job.status == "completed", job.error
    execution = captured["execution"]
    assert (execution.tile_size, execution.tile_resize) == (64, (128, 128))
    assert execution.sources["tile_size"] == "native_ratio"


def _stub_predictor_for_conf_source(monkeypatch, tmp_path):
    from PIL import Image

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "img.jpg")
    ckpt = _checkpoint(tmp_path)

    class FakePredictor:
        def __init__(self, checkpoint_path=None, **kwargs):
            pass

        def predict_batch(self, paths, execution=None, **kw):
            return [{"image": p, "width": 100, "height": 100,
                     "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9], "labels": [1],
                     "count": 1, "cap_hit": False} for p in paths]

    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", FakePredictor)
    return str(ckpt), str(images_dir)


def _sources(project: Path, out_dir: Path) -> dict:
    from tcip_mcp.buckets import read_bucket

    return read_bucket(out_dir).execution.sources


def test_web_worker_stamps_explicit_conf_and_max_dets_source_at_the_platform_default(
    tmp_path, monkeypatch,
):
    """A caller-stated conf/max_dets equal to the platform default is recorded 'explicit', the
    same distinction tile/tile_size already carry, never silently read back as a default."""
    from tcip_web.routes.inference import InferenceJob, _worker

    from tcip_mcp.pipelines.execution import DEFAULT_CONF, DEFAULT_MAX_DETS

    ckpt, images_dir = _stub_predictor_for_conf_source(monkeypatch, tmp_path)
    out_dir = tmp_path / "out"

    job = InferenceJob(
        project=str(tmp_path), job_id="conf-explicit", checkpoint_path=ckpt, images_dir=images_dir,
        output_dir=str(out_dir), stated=Stated(
            tile=False, conf=DEFAULT_CONF, cross_tile_nms=0.7, max_dets=DEFAULT_MAX_DETS),
    )
    _worker(job)

    assert job.status == "completed", job.error
    sources = _sources(tmp_path, out_dir)
    assert (sources["conf"], sources["max_dets"]) == ("explicit", "explicit")


def test_web_worker_stamps_default_conf_and_max_dets_source_when_unstated(tmp_path, monkeypatch):
    """The rail must admit the ordinary, unstated launch: an omitted conf/max_dets still runs the
    pass at the platform default, unchanged from the explicit-at-default case, and its provenance
    says 'default' rather than 'explicit'."""
    from tcip_web.routes.inference import InferenceJob, _worker

    import json

    ckpt, images_dir = _stub_predictor_for_conf_source(monkeypatch, tmp_path)
    out_dir = tmp_path / "out"

    job = InferenceJob(
        project=str(tmp_path), job_id="conf-default", checkpoint_path=ckpt, images_dir=images_dir,
        output_dir=str(out_dir), stated=Stated(tile=False, cross_tile_nms=0.7),
    )
    _worker(job)

    assert job.status == "completed"
    # The pass ran unchanged at the platform default: the one image's one detection landed.
    persisted = json.loads((out_dir / "img.json").read_text())["annotations"]
    assert len(persisted) == 1
    sources = _sources(tmp_path, out_dir)
    assert (sources["conf"], sources["max_dets"]) == ("default", "default")


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
    ckpt = _checkpoint(tmp_path)

    class FakePredictor:
        train_tile_size = 640

        def __init__(self, checkpoint_path=None, **kwargs):
            pass

        def predict_batch(self, paths, execution=None, **kw):
            return [{"image": p, "width": 100, "height": 100,
                     "boxes": [[10.0, 10.0, 30.0, 30.0], [50.0, 50.0, 50.0, 60.0]],
                     "scores": [0.9, 0.4], "labels": [1, 1], "count": 2, "cap_hit": False}
                    for p in paths]

    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", FakePredictor)

    job = InferenceJob(
        project=str(tmp_path), job_id="degenerate", checkpoint_path=str(ckpt), images_dir=str(images_dir),
        output_dir=str(out_dir), stated=Stated(
            tile=True, conf=0.25, cross_tile_nms=0.7, overlap=0.2, postprocess="nmm"),
    )
    _worker(job)

    assert job.status == "completed"
    assert job.dropped_boxes == 1
    persisted = json.loads((out_dir / "img.json").read_text())["annotations"]
    assert len(persisted) == 1

    # The same publication whose one audit line cannot be written still reports its count.
    import tcip_mcp.audit as audit_module

    def unwritable(*args, **kwargs):
        raise RuntimeError("audit log unwritable")

    monkeypatch.setattr(audit_module, "append", unwritable)
    gap = InferenceJob(
        project=str(tmp_path), job_id="degenerate-gap", checkpoint_path=str(ckpt),
        images_dir=str(images_dir), output_dir=str(tmp_path / "out-gap"), stated=job.stated)
    _worker(gap)

    assert gap.audit_warning is not None
    assert (gap.status, gap.dropped_boxes) == ("completed", 1)


def test_web_worker_fails_the_job_on_a_stem_collision(tmp_path):
    """The worker's pass (``inference_tools._prepare_pass``) enumerates through
    ``image_utils.list_logical_images``, the same enumeration every reader shares: a bucket already holding two logical identities under one
    case-folded stem fails the job with the refusal's own message, through the worker's own
    ``except Exception`` handler, rather than crashing the worker thread."""
    from PIL import Image

    from tcip_web.routes.inference import InferenceJob, _worker

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "foo.jpg")
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "foo.png")
    ckpt = _checkpoint(tmp_path)

    job = InferenceJob(
        project=str(tmp_path), job_id="collision", checkpoint_path=str(ckpt), images_dir=str(images_dir),
        output_dir=str(tmp_path / "out"), stated=Stated(
            tile=False, conf=0.25, cross_tile_nms=0.7, overlap=0.2, postprocess="nms"),
    )
    _worker(job)

    assert job.status == "failed"
    assert "logical image" in job.error
