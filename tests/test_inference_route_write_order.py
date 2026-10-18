"""The GUI inference worker publishes a pass as the MCP door does.

A pass the MCP door refuses fails a GUI run with the same reason and nothing written, a GUI pass
publishes its documents, the bucket's record and the publication's audit line, and a GUI pass
canceled while its first image is predicted publishes that image's document alone.
"""

from pathlib import Path

import pytest

from tests._audit_fixtures import audit_rows
from tests._predictor_fixtures import StubPredictor, install
from tests._training_values import RATIONALE
from tests._verified_checkpoint_fixtures import SAMPLE_DETECTOR_PASS

DATE = "2025-06-01"


def _two_images(images_dir):
    from tests._producer_fixtures import gray_frame

    for stem in ("a", "b"):
        gray_frame(images_dir, 100, f"{stem}.jpg")
    return images_dir


def _job(job_id, images_dir, bucket, ckpt, project):
    from tcip_mcp.dataset_layout import dataset_root_of
    from tcip_mcp.pipelines.execution import Stated
    from tcip_web.routes.inference import InferenceJob

    return InferenceJob(
        job_id=job_id,
        actor="user:tester",
        project=str(project),
        checkpoint_path=str(ckpt),
        dataset_root=str(dataset_root_of(images_dir)),
        images_dir=str(images_dir),
        bucket=bucket,
        stated=Stated(
            tile=False, conf=0.25, postprocess="nms", rationale=RATIONALE
        ),
    )


def _launch_through_the_route(client, dataset, bucket, ckpt, *, tile: bool):
    """Launch a GUI run publishing ``bucket`` through the route and join its worker.

    The run is launched by ``client``, and its worker thread is joined for up to 60 s.
    """
    from tcip_web.routes.inference import _get

    resp = client.post(
        "/api/inference/launch",
        json={
            "user": "tester",
            "checkpoint_path": str(ckpt),
            "dataset_root": str(dataset),
            "date": DATE,
            "bucket": bucket,
            "stated": {"tile": tile, **SAMPLE_DETECTOR_PASS},
        },
    )
    assert resp.status_code == 200, resp.text
    job = _get(resp.json()["job_id"])
    job.thread.join(60)
    return job


def _documents(root: Path, bucket: str) -> list[str]:
    """Return the stems of the prediction documents stored under ``bucket``."""
    import tcip_store
    from tcip_mcp.dataset_layout import PREDICTION_DOCUMENTS

    return [
        k.parts[-1]
        for k in tcip_store.keys(PREDICTION_DOCUMENTS, str(root), (bucket,))
    ]


def _published(root: Path, bucket: str) -> bool:
    import tcip_store
    from tcip_mcp.dataset_layout import bucket_key

    return tcip_store.exists(bucket_key(root, bucket))


def _record(root: Path, bucket: str) -> dict:
    from dataclasses import asdict

    from tcip_mcp.buckets import read_bucket

    return {
        k: v
        for k, v in asdict(read_bucket(root, bucket)).items()
        if k not in ("root", "name")
    }


def test_the_gui_worker_and_the_mcp_door_publish_the_same_bucket_record(
    tmp_path, monkeypatch
):
    """The GUI worker and the MCP door publish the same bucket record.

    One checkpoint over one image directory with the same stated values: the two doors' own
    bucket records agree on everything but the name each was published under.
    """
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
    mcp = run_inference(
        tmp_path,
        ckpt,
        str(images_dir),
        bucket=f"mcp/{DATE}",
        stated=job.stated,
    )
    assert "error" not in mcp, mcp

    assert _record(tmp_path, f"gui/{DATE}") == _record(tmp_path, f"mcp/{DATE}")


def test_a_pass_failing_after_its_first_document_publishes_nothing_on_either_door(
    tmp_path,
    opened_project,
    client,
    monkeypatch,
):
    """A pass failing after its first document publishes nothing on either door.

    A pass whose second document's encoding raises leaves no document, no record and no audit
    row, the GUI's pass and the MCP door's alike.
    """
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

    job = _launch_through_the_route(
        client, dataset, f"gui/{DATE}", ckpt, tile=False
    )
    assert job.status == "failed" and job.error == "disk full"

    with pytest.raises(OSError, match="disk full"):
        run_inference(
            tmp_path,
            ckpt,
            str(images_dir),
            bucket=f"mcp/{DATE}",
            stated=Stated(tile=False, **SAMPLE_DETECTOR_PASS),
        )

    for bucket in (f"gui/{DATE}", f"mcp/{DATE}"):
        assert _documents(dataset, bucket) == []
        assert not _published(dataset, bucket)
    assert audit_rows(dataset) == []


def test_a_pass_the_mcp_door_refuses_the_gui_refuses_alike_with_nothing_published(
    tmp_path,
    opened_project,
    client,
    monkeypatch,
):
    """A pass the MCP door refuses, the GUI refuses alike, with nothing published.

    A tiled pass stating no tile size fails the GUI job with the MCP door's reason and leaves no
    document, no record and no audit row.
    """
    pytest.importorskip("fastapi")
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.tools.inference_tools import run_inference
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    dataset = tmp_path / "orchard"
    images_dir = _two_images(image_dir(dataset, DATE))
    install(monkeypatch, StubPredictor())
    ckpt = registered_checkpoint(tmp_path)

    job = _launch_through_the_route(
        client, dataset, f"run/{DATE}", ckpt, tile=True
    )
    mcp = run_inference(
        tmp_path,
        ckpt,
        str(images_dir),
        bucket=f"mcp/{DATE}",
        stated=Stated(tile=True, **SAMPLE_DETECTOR_PASS),
    )

    assert job.status == "failed"
    assert job.error == mcp["error"]
    assert "has no tile_size" in mcp["error"]
    assert _documents(dataset, job.bucket) == []
    assert not _published(dataset, job.bucket)
    assert audit_rows(dataset) == []


def test_a_canceled_gui_pass_publishes_what_it_predicted(
    tmp_path, monkeypatch
):
    """A canceled GUI pass publishes the image it predicted.

    A cancel set while the first of two images is predicted leaves one image done: its document
    is published, the record names that image alone, and the job ends canceled.
    """
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

    assert (job.status, job.done, job.total, job.error) == (
        "canceled",
        1,
        2,
        None,
    )
    assert _documents(dataset, job.bucket) == ["a"]
    assert _record(dataset, job.bucket)["documents"] == {"a": "a.jpg"}
    assert [(r["tool"], r["status"]) for r in audit_rows(dataset)] == [
        ("prediction_bucket_published", "ok")
    ]


def test_a_full_gui_pass_publishes_every_document_and_the_record(
    tmp_path, monkeypatch
):
    """A full GUI pass over two images publishes both documents and the bucket's record.

    Both prediction documents, encoded in image order, and the record naming their producer
    and execution are published, and the job reports its summary.
    """
    pytest.importorskip("fastapi")
    import tcip_mcp.pipelines.postprocessing.export as export
    from tcip_annotation import json_io
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
        (annotation,) = json_io.read_predictions(
            bucket.document_key(stem)
        ).annotations
        assert annotation.score == pytest.approx(0.9)

    record = _record(tmp_path, job.bucket)
    assert record["execution"]["conf"] == pytest.approx(0.25)
    assert record["execution"]["sources"]["conf"] == "explicit"
    assert record["producer"]["checkpoint_sha256"]
    assert record["assessment_id"] is None
    assert record["documents"] == {"a": "a.jpg", "b": "b.jpg"}

    assert set(_summary(job)) == {
        "job_id",
        "status",
        "done",
        "total",
        "images_dir",
        "dataset_root",
        "bucket",
        "error",
        "dropped_boxes",
    }
    assert [Path(p).stem for p in written] == ["a", "b"]


def test_a_gui_run_and_an_mcp_run_leave_the_same_publication_lines(
    tmp_path, monkeypatch
):
    """A GUI run and an MCP run leave the same publication lines.

    Each run leaves the publication's one line and nothing else in its dataset's log, and each
    bucket's record names the run that produced its checkpoint.
    """
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
            result = run_inference(
                tmp_path,
                ckpt,
                str(images_dir),
                bucket=bucket,
                stated=Stated(tile=False, **SAMPLE_DETECTOR_PASS),
            )
            assert "error" not in result, result
        rows[door] = audit_rows(dataset)[before:]
        assert (
            _record(dataset, bucket)["producer"]["experiment_id"]
            == Path(ckpt).parent.name
        )

    def shape(records: list[dict]) -> list[tuple]:
        return [(r["tool"], sorted(r["arguments"])) for r in records]

    assert (
        shape(rows["gui"])
        == shape(rows["mcp"])
        == [
            (
                "prediction_bucket_published",
                ["bucket", "dataset_root", "dropped_boxes"],
            )
        ]
    )
    assert rows["gui"][0]["actor"] == "user:tester"
    assert "actor" not in rows["mcp"][0]
