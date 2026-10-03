"""The GUI inference worker publishes through the MCP door's own pass and publication: the same
refusals before anything is written, the documents landing one image at a time as the stream is
consumed, the bucket record last, and the publication's line, or a failed pass's line naming what
it wrote, the library's on both doors alike."""

import json
from pathlib import Path

import pytest

from tests._audit_fixtures import audit_rows
from tests._predictor_fixtures import StubPredictor, install

DATE = "2025-06-01"


def _two_images(images_dir):
    from PIL import Image

    images_dir.mkdir(parents=True)
    for stem in ("a", "b"):
        Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / f"{stem}.jpg")
    return images_dir


def _job(job_id, images_dir, out_dir, ckpt, project):
    from tcip_mcp.pipelines.execution import Stated
    from tcip_web.routes.inference import InferenceJob

    return InferenceJob(
        job_id=job_id, actor="user:tester", project=str(project), checkpoint_path=str(ckpt),
        images_dir=str(images_dir), output_dir=str(out_dir),
        stated=Stated(tile=False, conf=0.25, postprocess="nms"))


def _launch_through_the_route(dataset, output_dir, ckpt, *, tile: bool):
    """A GUI run into ``output_dir`` launched through the route's own TestClient, joined until its
    worker ends."""
    from fastapi.testclient import TestClient

    from tcip_web.app import app
    from tcip_web.routes.inference import _get

    client = TestClient(app, base_url="http://127.0.0.1")
    resp = client.post("/api/inference/launch", json={
        "user": "tester", "checkpoint_path": str(ckpt), "dataset_root": str(dataset), "date": DATE,
        "output_dir": str(output_dir), "stated": {"tile": tile}})
    assert resp.status_code == 200, resp.text
    job = _get(resp.json()["job_id"])
    job.thread.join(60)
    return job


def _documents(bucket: Path) -> list[str]:
    """The stems of the prediction documents the directory holds."""
    from tcip_annotation.json_io import prediction_documents

    return sorted(p.stem for p in prediction_documents(bucket)) if bucket.is_dir() else []


def _record(bucket: Path) -> dict:
    from dataclasses import asdict

    from tcip_mcp.buckets import read_bucket

    return {k: v for k, v in asdict(read_bucket(bucket)).items() if k != "path"}


def test_the_gui_worker_and_the_mcp_door_publish_the_same_bucket_record(tmp_path, monkeypatch):
    """One checkpoint over one image directory with the same stated values: the two doors' own
    bucket records agree on everything but where and when each was published."""
    pytest.importorskip("fastapi")
    from tcip_mcp.tools.inference_tools import run_inference
    from tcip_web.routes.inference import _worker
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    images_dir = _two_images(tmp_path / "images")
    ckpt = registered_checkpoint(tmp_path)
    install(monkeypatch, StubPredictor())
    job = _job("prepared", images_dir, tmp_path / "gui", ckpt, tmp_path)
    _worker(job)
    assert job.status == "completed", job.error
    mcp = run_inference(tmp_path, ckpt, str(images_dir), output_dir=str(tmp_path / "mcp"),
                        stated=job.stated)
    assert "error" not in mcp, mcp

    assert _record(tmp_path / "gui") == _record(tmp_path / "mcp")


def test_a_pass_failing_after_its_first_document_leaves_one_failure_line_on_each_door(
    tmp_path, opened_project, monkeypatch,
):
    """A pass that dies between images keeps the document it wrote and no record, and the library
    records that document and the error under a failed status, the GUI's pass and the MCP door's
    alike; nothing reads the half-filled directory as published."""
    pytest.importorskip("fastapi")
    from tcip_annotation.json_io import BUCKET_RECORD

    import tcip_mcp.pipelines.postprocessing.export as export
    from tcip_mcp.dataset_layout import image_dir, prediction_root
    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.tools.inference_tools import run_inference
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    dataset = tmp_path / "orchard"
    images_dir = _two_images(image_dir(dataset, DATE))
    ckpt = registered_checkpoint(tmp_path)
    install(monkeypatch, StubPredictor())

    real_encode = export.encode_predictions
    calls: list[str] = []

    def failing_second_document(result, **kwargs):
        calls.append(str(result["image"]))
        if len(calls) % 2 == 0:
            raise OSError("disk full")
        return real_encode(result, **kwargs)

    monkeypatch.setattr(export, "encode_predictions", failing_second_document)

    gui_out = prediction_root(dataset) / "gui" / DATE
    job = _launch_through_the_route(dataset, gui_out, ckpt, tile=False)
    assert job.status == "failed" and job.error == "disk full"
    gui_rows = audit_rows(dataset)

    mcp_out = prediction_root(dataset) / "mcp" / DATE
    with pytest.raises(OSError, match="disk full"):
        run_inference(tmp_path, ckpt, str(images_dir), output_dir=str(mcp_out),
                      stated=Stated(tile=False))
    mcp_rows = audit_rows(dataset)[len(gui_rows):]

    def shape(rows: list[dict]) -> list[tuple]:
        return [(r["tool"], r["status"], sorted(r["arguments"]),
                 [Path(p).name for p in r["arguments"]["written"]], r["arguments"]["error"])
                for r in rows]

    assert shape(gui_rows) == shape(mcp_rows) == [
        ("prediction_bucket_published", "failed", ["error", "predictions_dir", "written"],
         ["a"], "disk full")]
    for bucket in (gui_out, mcp_out):
        assert _documents(bucket) == ["a"]
        assert not (bucket / BUCKET_RECORD).exists()


def test_a_pass_the_mcp_door_refuses_the_gui_refuses_alike_with_nothing_published(
    tmp_path, opened_project, monkeypatch,
):
    """A refusal the pass raises before its first write refuses a GUI run as it refuses the MCP
    door's: the same reason, no document, no record, no line."""
    pytest.importorskip("fastapi")
    from tcip_annotation.json_io import BUCKET_RECORD

    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.tools.inference_tools import run_inference
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    dataset = tmp_path / "orchard"
    images_dir = _two_images(image_dir(dataset, DATE))
    install(monkeypatch, StubPredictor())
    ckpt = registered_checkpoint(tmp_path)

    job = _launch_through_the_route(dataset, dataset / "predictions" / "run" / DATE, ckpt,
                                    tile=True)
    mcp = run_inference(tmp_path, ckpt, str(images_dir),
                        output_dir=str(dataset / "predictions" / "mcp" / DATE),
                        stated=Stated(tile=True))

    assert job.status == "failed"
    assert job.error == mcp["error"]
    assert "tile_size could not be resolved" in mcp["error"]
    bucket = Path(job.output_dir)
    assert _documents(bucket) == []
    assert not (bucket / BUCKET_RECORD).exists()
    assert audit_rows(dataset) == []


def test_a_canceled_gui_pass_publishes_what_it_wrote(tmp_path, monkeypatch):
    """A cancel stops the stream at the next image boundary: the documents already written are
    published, the record names exactly those, and the job ends canceled."""
    pytest.importorskip("fastapi")
    from tcip_mcp.dataset_layout import image_dir, prediction_root
    from tcip_web.routes.inference import _worker
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    dataset = tmp_path / "orchard"
    images_dir = _two_images(image_dir(dataset, DATE))
    out = prediction_root(dataset) / "run" / DATE
    ckpt = registered_checkpoint(tmp_path)
    job = _job("canceled-after-one", images_dir, out, ckpt, tmp_path)

    class CancelAfterFirstImage(StubPredictor):
        def predict_batch(self, paths, *args, **kw):
            job.cancel_event.set()
            return super().predict_batch(paths, *args, **kw)

    install(monkeypatch, CancelAfterFirstImage())

    _worker(job)

    assert (job.status, job.done, job.total, job.error) == ("canceled", 1, 2, None)
    assert _documents(out) == ["a"]
    assert _record(out)["documents"] == {"a": "a.jpg"}
    assert [(r["tool"], r["status"]) for r in audit_rows(dataset)] == [
        ("prediction_bucket_published", "ok")]


def test_a_full_gui_pass_writes_every_document_then_the_record(tmp_path, monkeypatch):
    """Every prediction document and the record that states who produced them are on disk, in the
    order the pass walked the images, and the job reports its summary."""
    pytest.importorskip("fastapi")
    import tcip_mcp.pipelines.postprocessing.export as export
    from tcip_web.routes.inference import _summary, _worker
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    images_dir = _two_images(tmp_path / "images")
    out_dir = tmp_path / "out"
    ckpt = registered_checkpoint(tmp_path)

    install(monkeypatch, StubPredictor())

    real_encode = export.encode_predictions
    written = []

    def recording_encode(result, **kwargs):
        written.append(result["image"])
        return real_encode(result, **kwargs)

    monkeypatch.setattr(export, "encode_predictions", recording_encode)

    job = _job("full-pass", images_dir, out_dir, ckpt, tmp_path)
    _worker(job)

    assert job.status == "completed"
    assert job.done == 2 and job.total == 2
    assert job.error is None
    for stem in ("a", "b"):
        assert json.loads((out_dir / f"{stem}.json").read_text())["annotations"][0]["score"] \
            == pytest.approx(0.9)

    record = _record(out_dir)
    assert record["execution"]["conf"] == pytest.approx(0.25)
    assert record["execution"]["sources"]["conf"] == "explicit"
    assert record["producer"]["checkpoint_sha256"]
    assert record["assessment_id"] is None
    assert record["documents"] == {"a": "a.jpg", "b": "b.jpg"}

    assert set(_summary(job)) == {"job_id", "status", "done", "total", "images_dir", "output_dir",
                                 "error", "audit_warning", "dropped_boxes"}
    assert [Path(p).stem for p in written] == ["a", "b"]


def test_a_gui_run_and_an_mcp_run_leave_the_same_publication_lines(tmp_path, monkeypatch):
    """Each run leaves the publication's one line and nothing else in its dataset's log, and each
    bucket's record names the run that produced its checkpoint."""
    pytest.importorskip("fastapi")
    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.tools.inference_tools import run_inference
    from tcip_web.routes.inference import _worker
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    images_dir = _two_images(tmp_path / "images")
    install(monkeypatch, StubPredictor())
    dataset = tmp_path / "orchard"
    rows: dict[str, list[dict]] = {}
    for door in ("gui", "mcp"):
        ckpt = registered_checkpoint(tmp_path)
        out = dataset / "predictions" / door / DATE
        before = len(audit_rows(dataset))
        if door == "gui":
            _worker(_job(door, images_dir, out, ckpt, tmp_path))
        else:
            result = run_inference(tmp_path, ckpt, str(images_dir), output_dir=str(out),
                                   stated=Stated(tile=False))
            assert "error" not in result, result
        rows[door] = audit_rows(dataset)[before:]
        assert _record(out)["producer"]["experiment_id"] == Path(ckpt).parent.name

    def shape(records: list[dict]) -> list[tuple]:
        return [(r["tool"], sorted(r["arguments"])) for r in records]

    assert shape(rows["gui"]) == shape(rows["mcp"]) == [
        ("prediction_bucket_published", ["dropped_boxes", "predictions_dir"])]
    assert rows["gui"][0]["actor"] == "user:tester"
    assert "actor" not in rows["mcp"][0]
