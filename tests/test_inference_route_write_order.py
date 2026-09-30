"""The GUI inference worker publishes through the MCP door's own publisher: the same gates refuse
before anything is written, the documents land one image at a time as the stream is consumed, the
stamp lands last, and the publication's line, or a failed pass's line naming what it wrote, is the
library's on both doors alike."""

import json
from pathlib import Path

import pytest

from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

DATE = "2025-06-01"


def _two_images(images_dir):
    from PIL import Image

    images_dir.mkdir(parents=True)
    for stem in ("a", "b"):
        Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / f"{stem}.jpg")
    return images_dir


class _FakePredictor:
    """A detector that returns one detection per image, enough for a real prediction write."""

    def __init__(self, checkpoint_path=None, **kwargs):
        pass

    def predict_batch(self, paths, tile=False, tile_size=224, overlap=0.2, **kw):
        return [{"image": p, "width": 100, "height": 100,
                 "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9], "labels": [1], "count": 1}
                for p in paths]


def _job(job_id, images_dir, out_dir, ckpt, project):
    from tcip_mcp.dataset_layout import bucket_dataset_root
    from tcip_web.routes.inference import InferenceJob

    return InferenceJob(
        job_id=job_id, project=str(project), checkpoint_path=str(ckpt),
        images_dir=str(images_dir), output_dir=str(out_dir), tile=False, conf=0.25,
        cross_tile_nms=0.7, overlap=0.2, postprocess="nms",
        dataset_root=bucket_dataset_root(out_dir),
    )


def _launch_through_the_route(dataset, ckpt, model):
    """A GUI run launched through the route's own TestClient, joined until its worker ends."""
    from fastapi.testclient import TestClient

    from tcip_web.app import app
    from tcip_web.routes.inference import _get

    client = TestClient(app, base_url="http://127.0.0.1")
    resp = client.post("/api/inference/launch", json={
        "checkpoint_path": str(ckpt), "dataset_root": str(dataset), "model_name": model,
        "date": DATE, "tile": False})
    assert resp.status_code == 200, resp.text
    job = _get(resp.json()["job_id"])
    job.thread.join(60)
    return job


def _dataset_rows(dataset) -> list[dict]:
    """Every row in the dataset's own audit log."""
    import tcip_store as ts

    from tcip_mcp.audit import audit_log_key

    return ts.read_log(audit_log_key(dataset)).records


def _documents(bucket: Path) -> list[str]:
    """The stems of the prediction documents the bucket holds, by the platform's own enumeration."""
    from tcip_mcp.prediction_buckets import bucket_stems

    return sorted(bucket_stems(bucket)) if bucket.is_dir() else []


def test_the_gui_worker_and_the_mcp_pass_prepare_the_same_run(tmp_path, monkeypatch):
    """One checkpoint over one image directory: the run the GUI worker hands the publisher and the
    one the MCP door's verified pass returns carry the same keys and the same values, apart from
    the predictions themselves and when each was produced."""
    pytest.importorskip("fastapi")
    import tcip_mcp.tools.inference_tools as itools
    from tcip_web.routes.inference import _worker
    from tests._verified_checkpoint_fixtures import registered_checkpoint, run_inference_verified

    images_dir = _two_images(tmp_path / "images")
    ckpt = registered_checkpoint(tmp_path)
    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", _FakePredictor)
    real_publish = itools.publish_bucket
    handed: list[dict] = []

    def spy(project, result, **kwargs):
        handed.append(dict(result))
        return real_publish(project, result, **kwargs)

    monkeypatch.setattr(itools, "publish_bucket", spy)
    job = _job("prepared", images_dir, tmp_path / "out", ckpt, tmp_path)
    _worker(job)
    assert job.status == "completed", job.error
    mcp = run_inference_verified(tmp_path, ckpt, images_dir=str(images_dir), conf_threshold=job.conf,
                                 cross_tile_nms=job.cross_tile_nms, tile=job.tile, overlap=job.overlap)

    outcome = {"results", "image_count", "total_detections", "produced_at"}
    (gui,) = handed
    assert set(gui) - outcome == set(mcp) - outcome
    assert {k: gui[k] for k in set(gui) - outcome} == {k: mcp[k] for k in set(mcp) - outcome}


def test_a_pass_failing_after_its_first_document_leaves_one_failure_line_on_each_door(
    tmp_path, opened_project, monkeypatch,
):
    """A pass that dies between images keeps the document it wrote and no stamp, and the library
    records that document and the error under a failed status, the GUI's pass and the MCP door's
    alike; nothing reads the half-filled bucket as published."""
    pytest.importorskip("fastapi")
    import tcip_mcp.tools.inference_tools as itools
    from tcip_mcp.dataset_layout import image_dir, prediction_dir
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    dataset = tmp_path / "orchard"
    images_dir = _two_images(image_dir(dataset, DATE))
    ckpt = registered_checkpoint(tmp_path)
    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", _FakePredictor)

    real_write = itools.write_predictions_json
    calls: list[str] = []

    def failing_second_write(json_path, result, **kwargs):
        calls.append(str(json_path))
        if len(calls) % 2 == 0:
            raise OSError("disk full")
        return real_write(json_path, result, **kwargs)

    monkeypatch.setattr(itools, "write_predictions_json", failing_second_write)

    job = _launch_through_the_route(dataset, ckpt, "gui")
    assert job.status == "failed" and job.error == "disk full"
    gui_rows = _dataset_rows(dataset)

    mcp_out = prediction_dir(dataset, "mcp", DATE)
    with pytest.raises(OSError, match="disk full"):
        itools.run_inference(tmp_path, ckpt, str(images_dir), output_dir=str(mcp_out), tile=False)
    mcp_rows = _dataset_rows(dataset)[len(gui_rows):]

    def shape(rows: list[dict]) -> list[tuple]:
        return [(r["tool"], r["status"], sorted(r["arguments"]),
                 [Path(p).name for p in r["arguments"]["written"]], r["arguments"]["error"])
                for r in rows]

    assert shape(gui_rows) == shape(mcp_rows) == [
        ("prediction_bucket_published", "failed", ["error", "predictions_dir", "written"],
         ["a.json"], "disk full")]
    for bucket in (prediction_dir(dataset, "gui", DATE), mcp_out):
        assert _documents(bucket) == ["a"]
        assert read_operating_point_sidecar(bucket) is None


def test_a_publish_the_mcp_door_refuses_the_gui_refuses_alike_with_nothing_written(
    tmp_path, opened_project, monkeypatch,
):
    """A gate the publisher runs before its first write refuses a GUI run as it refuses the MCP
    door's: the same reason, no document, no stamp, no line."""
    pytest.importorskip("fastapi")
    import tcip_mcp.tools.inference_tools as itools
    from tcip_mcp.dataset_layout import image_dir
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    dataset = tmp_path / "orchard"
    images_dir = _two_images(image_dir(dataset, DATE))
    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", _FakePredictor)
    ckpt = registered_checkpoint(tmp_path)
    # A raw GUI pass earns no claim, so this gate's refusal is stood in for at the gate itself.
    monkeypatch.setattr(itools, "_draft_count_claim", lambda project, result, **kw: (
        None, {"error": "the count claim for trait 'leaf count' was not earned: stand-in"}))

    job = _launch_through_the_route(dataset, ckpt, "run")
    bucket = Path(job.output_dir)
    mcp = itools.run_inference(tmp_path, ckpt, str(images_dir), output_dir=str(bucket), tile=False)

    assert job.status == "failed"
    assert job.error == mcp["error"]
    assert _documents(bucket) == []
    assert read_operating_point_sidecar(bucket) is None
    assert _dataset_rows(dataset) == []


def test_a_canceled_gui_pass_publishes_what_it_wrote_through_the_one_publisher(
    tmp_path, monkeypatch,
):
    """A cancel stops the stream at the next image boundary: the documents already written are
    stamped and published, the stamp names exactly those, and the job ends canceled."""
    pytest.importorskip("fastapi")
    import tcip_store as ts

    from tcip_mcp.audit import audit_log_key
    from tcip_mcp.dataset_layout import image_dir, prediction_dir
    from tcip_mcp.pipelines.resolution import sidecar_key
    from tcip_web.routes.inference import _worker
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    dataset = tmp_path / "orchard"
    images_dir = _two_images(image_dir(dataset, DATE))
    out = prediction_dir(dataset, "run", DATE)
    ckpt = registered_checkpoint(tmp_path)
    job = _job("canceled-after-one", images_dir, out, ckpt, tmp_path)

    class CancelAfterFirstImage(_FakePredictor):
        def predict_batch(self, paths, tile=False, tile_size=224, overlap=0.2, **kw):
            job.cancel_event.set()
            return super().predict_batch(paths, tile, tile_size, overlap, **kw)

    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", CancelAfterFirstImage)

    _worker(job)

    assert (job.status, job.done, job.total, job.error) == ("canceled", 1, 2, None)
    assert _documents(out) == ["a"]
    assert ts.read(sidecar_key(out))["image_filenames"] == {"a": "a.jpg"}
    rows = ts.read_log(audit_log_key(dataset)).records
    assert [(r["tool"], r["status"]) for r in rows] == [
        ("stamp_written", "ok"), ("prediction_bucket_published", "ok")]


def test_worker_writes_every_prediction_file_and_the_sidecar_on_a_full_pass(tmp_path, monkeypatch):
    """The ordering costs a completed run nothing: every prediction file and the stamp that
    certifies them are all on disk, in the order the pass walked the images, and the job still
    reports what it reported before."""
    pytest.importorskip("fastapi")
    import tcip_mcp.tools.inference_tools as itools
    from tcip_web.routes.inference import _summary, _worker
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    images_dir = _two_images(tmp_path / "images")
    out_dir = tmp_path / "out"
    ckpt = registered_checkpoint(tmp_path)

    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", _FakePredictor)

    real_write = itools.write_predictions_json
    written = []

    def recording_write(json_path, result, **kwargs):
        written.append(json_path)
        return real_write(json_path, result, **kwargs)

    monkeypatch.setattr(itools, "write_predictions_json", recording_write)

    job = _job("full-pass", images_dir, out_dir, ckpt, tmp_path)
    _worker(job)

    assert job.status == "completed"
    assert job.done == 2 and job.total == 2
    assert job.error is None
    for stem in ("a", "b"):
        assert json.loads((out_dir / f"{stem}.json").read_text())["annotations"][0]["score"] \
            == pytest.approx(0.9)

    import tcip_store as ts
    from tcip_mcp.pipelines.resolution import sidecar_key
    from tcip_mcp.registry_paths import stored_path

    sidecar = ts.read(sidecar_key(out_dir))
    assert sidecar["operating_point"]["conf"]["value"] == pytest.approx(0.25)
    assert sidecar["checkpoint"] == Path(ckpt).stem
    assert sidecar["checkpoint_sha256"] and sidecar["produced_at"]
    assert sidecar["images_dir"] == stored_path(images_dir, tmp_path)
    assert sidecar["validated"] is False
    assert sidecar["image_filenames"] == {"a": "a.jpg", "b": "b.jpg"}

    assert set(_summary(job)) == {"job_id", "status", "done", "total", "images_dir", "output_dir",
                                 "error", "warning", "audit_warning", "dropped_nonpositive_boxes"}
    assert [Path(p).stem for p in written] == ["a", "b"]


def test_a_gui_run_and_an_mcp_run_leave_the_same_publication_records(tmp_path, monkeypatch):
    """The GUI worker publishes through the MCP door's own publisher: each run leaves the stamp's
    line and the publication's line and nothing else in its dataset's log, and each bucket's
    stamp names the run that produced its checkpoint."""
    pytest.importorskip("fastapi")
    import tcip_store as ts

    from tcip_mcp.audit import audit_log_key
    from tcip_mcp.tools.inference_tools import run_inference
    from tcip_web.routes.inference import _worker
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    images_dir = _two_images(tmp_path / "images")
    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", _FakePredictor)
    dataset = tmp_path / "orchard"
    rows: dict[str, list[dict]] = {}
    for door in ("gui", "mcp"):
        ckpt = registered_checkpoint(tmp_path)
        out = dataset / "predictions" / door / "2025-06-01"
        before = len(ts.read_log(audit_log_key(dataset)).records)
        if door == "gui":
            _worker(_job(door, images_dir, out, ckpt, tmp_path))
        else:
            result = run_inference(tmp_path, ckpt, str(images_dir), output_dir=str(out), tile=False)
            assert "error" not in result, result
        rows[door] = ts.read_log(audit_log_key(dataset)).records[before:]
        assert read_operating_point_sidecar(out)["experiment_id"] == Path(ckpt).parent.name

    def shape(records: list[dict]) -> list[tuple]:
        return [(r["tool"], sorted(r["arguments"])) for r in records]

    assert shape(rows["gui"]) == shape(rows["mcp"])
    assert shape(rows["gui"])[-1] == ("prediction_bucket_published", ["predictions_dir"])
