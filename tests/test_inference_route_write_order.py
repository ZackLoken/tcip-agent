"""The GUI inference worker publishes through the MCP door's own pass and publication: the same
refusals before anything is written, one commit holding every document, the bucket's record and
the publication's line, and nothing at all for a pass that fails, the library's on both doors
alike."""

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


def _job(job_id, images_dir, bucket, ckpt, project):
    from tcip_mcp.dataset_layout import dataset_root_of
    from tcip_mcp.pipelines.execution import Stated
    from tcip_web.routes.inference import InferenceJob

    return InferenceJob(
        job_id=job_id, actor="user:tester", project=str(project), checkpoint_path=str(ckpt),
        dataset_root=str(dataset_root_of(images_dir)), images_dir=str(images_dir), bucket=bucket,
        stated=Stated(tile=False, conf=0.25, postprocess="nms"))


def _launch_through_the_route(dataset, bucket, ckpt, *, tile: bool):
    """A GUI run publishing ``bucket`` launched through the route's own TestClient, joined until
    its worker ends."""
    from fastapi.testclient import TestClient

    from tcip_web.app import app
    from tcip_web.routes.inference import _get

    client = TestClient(app, base_url="http://127.0.0.1")
    resp = client.post("/api/inference/launch", json={
        "user": "tester", "checkpoint_path": str(ckpt), "dataset_root": str(dataset), "date": DATE,
        "bucket": bucket, "stated": {"tile": tile}})
    assert resp.status_code == 200, resp.text
    job = _get(resp.json()["job_id"])
    job.thread.join(60)
    return job


def _documents(root: Path, bucket: str) -> list[str]:
    """The stems of the prediction documents stored under ``bucket``."""
    import tcip_store

    from tcip_mcp.dataset_layout import PREDICTION_DOCUMENTS

    return [k.parts[-1] for k in tcip_store.keys(PREDICTION_DOCUMENTS, str(root), (bucket,))]


def _published(root: Path, bucket: str) -> bool:
    import tcip_store

    from tcip_mcp.dataset_layout import bucket_key

    return tcip_store.exists(bucket_key(root, bucket))


def _record(root: Path, bucket: str) -> dict:
    from dataclasses import asdict

    from tcip_mcp.buckets import read_bucket

    return {k: v for k, v in asdict(read_bucket(root, bucket)).items() if k not in ("root", "name")}


def test_the_gui_worker_and_the_mcp_door_publish_the_same_bucket_record(tmp_path, monkeypatch):
    """One checkpoint over one image directory with the same stated values: the two doors' own
    bucket records agree on everything but the name each was published under."""
    pytest.importorskip("fastapi")
    from tcip_mcp.tools.inference_tools import run_inference
    from tcip_web.routes.inference import _worker
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    images_dir = _two_images(tmp_path / "images" / DATE)
    ckpt = registered_checkpoint(tmp_path)
    install(monkeypatch, StubPredictor())
    job = _job("prepared", images_dir, f"gui/{DATE}", ckpt, tmp_path)
    _worker(job)
    assert job.status == "completed", job.error
    mcp = run_inference(tmp_path, ckpt, str(images_dir), bucket=f"mcp/{DATE}", stated=job.stated)
    assert "error" not in mcp, mcp

    assert _record(tmp_path, f"gui/{DATE}") == _record(tmp_path, f"mcp/{DATE}")


def test_a_pass_failing_after_its_first_document_publishes_nothing_on_either_door(
    tmp_path, opened_project, monkeypatch,
):
    """A pass that dies between images leaves no document, no record and no line, the GUI's pass
    and the MCP door's alike: a publication is one commit."""
    pytest.importorskip("fastapi")
    import tcip_mcp.pipelines.postprocessing.export as export
    from tcip_mcp.dataset_layout import image_dir
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

    job = _launch_through_the_route(dataset, f"gui/{DATE}", ckpt, tile=False)
    assert job.status == "failed" and job.error == "disk full"

    with pytest.raises(OSError, match="disk full"):
        run_inference(tmp_path, ckpt, str(images_dir), bucket=f"mcp/{DATE}",
                      stated=Stated(tile=False))

    for bucket in (f"gui/{DATE}", f"mcp/{DATE}"):
        assert _documents(dataset, bucket) == []
        assert not _published(dataset, bucket)
    assert audit_rows(dataset) == []


def test_a_pass_the_mcp_door_refuses_the_gui_refuses_alike_with_nothing_published(
    tmp_path, opened_project, monkeypatch,
):
    """A refusal the pass raises before its first write refuses a GUI run as it refuses the MCP
    door's: the same reason, no document, no record, no line."""
    pytest.importorskip("fastapi")
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.tools.inference_tools import run_inference
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    dataset = tmp_path / "orchard"
    images_dir = _two_images(image_dir(dataset, DATE))
    install(monkeypatch, StubPredictor())
    ckpt = registered_checkpoint(tmp_path)

    job = _launch_through_the_route(dataset, f"run/{DATE}", ckpt, tile=True)
    mcp = run_inference(tmp_path, ckpt, str(images_dir), bucket=f"mcp/{DATE}",
                        stated=Stated(tile=True))

    assert job.status == "failed"
    assert job.error == mcp["error"]
    assert "tile_size could not be resolved" in mcp["error"]
    assert _documents(dataset, job.bucket) == []
    assert not _published(dataset, job.bucket)
    assert audit_rows(dataset) == []


def test_a_canceled_gui_pass_publishes_what_it_predicted(tmp_path, monkeypatch):
    """A cancel stops the stream at the next image boundary: the documents already predicted are
    published, the record names exactly those, and the job ends canceled."""
    pytest.importorskip("fastapi")
    from tcip_mcp.dataset_layout import image_dir
    from tcip_web.routes.inference import _worker
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    dataset = tmp_path / "orchard"
    images_dir = _two_images(image_dir(dataset, DATE))
    ckpt = registered_checkpoint(tmp_path)
    job = _job("canceled-after-one", images_dir, f"run/{DATE}", ckpt, tmp_path)

    class CancelAfterFirstImage(StubPredictor):
        def predict_batch(self, paths, *args, **kw):
            job.cancel_event.set()
            return super().predict_batch(paths, *args, **kw)

    install(monkeypatch, CancelAfterFirstImage())

    _worker(job)

    assert (job.status, job.done, job.total, job.error) == ("canceled", 1, 2, None)
    assert _documents(dataset, job.bucket) == ["a"]
    assert _record(dataset, job.bucket)["documents"] == {"a": "a.jpg"}
    assert [(r["tool"], r["status"]) for r in audit_rows(dataset)] == [
        ("prediction_bucket_published", "ok")]


def test_a_full_gui_pass_publishes_every_document_and_the_record(tmp_path, monkeypatch):
    """Every prediction document and the record that states who produced them are published, the
    images predicted in the order the pass walked them, and the job reports its summary."""
    pytest.importorskip("fastapi")
    from tcip_annotation import json_io

    import tcip_mcp.pipelines.postprocessing.export as export
    from tcip_mcp.buckets import read_bucket
    from tcip_web.routes.inference import _summary, _worker
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    images_dir = _two_images(tmp_path / "images" / DATE)
    ckpt = registered_checkpoint(tmp_path)

    install(monkeypatch, StubPredictor())

    real_encode = export.encode_predictions
    written = []

    def recording_encode(result, **kwargs):
        written.append(result["image"])
        return real_encode(result, **kwargs)

    monkeypatch.setattr(export, "encode_predictions", recording_encode)

    job = _job("full-pass", images_dir, f"run/{DATE}", ckpt, tmp_path)
    _worker(job)

    assert job.status == "completed"
    assert job.done == 2 and job.total == 2
    assert job.error is None
    bucket = read_bucket(tmp_path, job.bucket)
    for stem in ("a", "b"):
        (annotation,) = json_io.read_predictions(bucket.document_key(stem))
        assert annotation.score == pytest.approx(0.9)

    record = _record(tmp_path, job.bucket)
    assert record["execution"]["conf"] == pytest.approx(0.25)
    assert record["execution"]["sources"]["conf"] == "explicit"
    assert record["producer"]["checkpoint_sha256"]
    assert record["assessment_id"] is None
    assert record["documents"] == {"a": "a.jpg", "b": "b.jpg"}

    assert set(_summary(job)) == {"job_id", "status", "done", "total", "images_dir",
                                 "dataset_root", "bucket", "error", "dropped_boxes"}
    assert [Path(p).stem for p in written] == ["a", "b"]


def test_a_gui_run_and_an_mcp_run_leave_the_same_publication_lines(tmp_path, monkeypatch):
    """Each run leaves the publication's one line and nothing else in its dataset's log, and each
    bucket's record names the run that produced its checkpoint."""
    pytest.importorskip("fastapi")
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.tools.inference_tools import run_inference
    from tcip_web.routes.inference import _worker
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    dataset = tmp_path / "orchard"
    images_dir = _two_images(image_dir(dataset, DATE))
    install(monkeypatch, StubPredictor())
    rows: dict[str, list[dict]] = {}
    for door in ("gui", "mcp"):
        ckpt = registered_checkpoint(tmp_path)
        bucket = f"{door}/{DATE}"
        before = len(audit_rows(dataset))
        if door == "gui":
            _worker(_job(door, images_dir, bucket, ckpt, tmp_path))
        else:
            result = run_inference(tmp_path, ckpt, str(images_dir), bucket=bucket,
                                   stated=Stated(tile=False))
            assert "error" not in result, result
        rows[door] = audit_rows(dataset)[before:]
        assert _record(dataset, bucket)["producer"]["experiment_id"] == Path(ckpt).parent.name

    def shape(records: list[dict]) -> list[tuple]:
        return [(r["tool"], sorted(r["arguments"])) for r in records]

    assert shape(rows["gui"]) == shape(rows["mcp"]) == [
        ("prediction_bucket_published", ["bucket", "dataset_root", "dropped_boxes"])]
    assert rows["gui"][0]["actor"] == "user:tester"
    assert "actor" not in rows["mcp"][0]
