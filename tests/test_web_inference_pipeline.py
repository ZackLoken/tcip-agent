"""The web inference worker, with the pipeline's ``GenericPredictor`` building a stub."""

from pathlib import Path

import pytest

from tcip_mcp.pipelines.execution import Stated
from tcip_mcp.pipelines.schemas import AugmentationSpec
from tests._predictor_fixtures import BOX, StubPredictor, install
from tests._training_values import RATIONALE, augmentation

BUCKET = "out/2026-01-01"


def _checkpoint(project: Path, **kwargs) -> Path:
    """Return the checkpoint ``project_checkpoint`` registers in ``project``, passing ``kwargs``."""
    from tests._verified_checkpoint_fixtures import project_checkpoint

    return Path(project_checkpoint(project, **kwargs))


def _one_image(project: Path, *stems: str) -> Path:
    """Write a black 100x100 ``<stem>.jpg`` per stem (``img`` by default) in the undated capture.

    Returns the capture's directory.
    """
    from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir
    from tests._producer_fixtures import write_image

    images_dir = image_dir(project, UNDATED_BUCKET)
    for stem in stems or ("img",):
        write_image(images_dir / f"{stem}.jpg", (100, 100))
    return images_dir


def _job(
    job_id: str,
    project: Path,
    ckpt,
    images_dir: Path,
    stated: Stated,
    bucket: str = BUCKET,
):
    from tcip_web.routes.inference import InferenceJob

    return InferenceJob(
        job_id=job_id,
        actor="user:tester",
        project=str(project),
        checkpoint_path=str(ckpt),
        dataset_root=str(images_dir.parent.parent),
        images_dir=str(images_dir),
        bucket=bucket,
        stated=stated,
    )


def _annotations(
    project: Path, stem: str = "img", bucket: str = BUCKET
) -> list[dict]:
    """Return the records of ``stem``'s published prediction document under ``bucket``."""
    import tcip_store
    from tcip_mcp.dataset_layout import prediction_key

    return tcip_store.read(prediction_key(project, bucket, stem))[
        "annotations"
    ]


def test_encode_predictions_roundtrip_and_negative(tmp_path):
    from tcip_annotation import json_io
    from tcip_mcp.pipelines.data.label_queries import registry_scope
    from tcip_mcp.pipelines.postprocessing.export import encode_predictions

    scope = registry_scope(tmp_path, "bud")
    encoded, _dropped = encode_predictions(
        {
            "image": "img.jpg",
            "width": 100,
            "height": 100,
            "boxes": [[10.0, 10.0, 30.0, 30.0]],
            "scores": [0.9],
            "labels": [1],
            "count": 1,
            "cap": 2,
        },
        "model:fixture",
        scope=scope,
    )
    ann = encoded["annotations"][0]
    assert ann["subject"] == "bud"  # the scope's one subject
    assert ann["bbox"] == [10.0, 10.0, 20.0, 20.0]
    assert ann["score"] == pytest.approx(0.9)
    preds = json_io.label_document(encoded).annotations  # symmetric read
    assert len(preds) == 1 and preds[0].score == pytest.approx(0.9)

    # Negative invariant: a zero-detection image still yields an {"annotations": []} record.
    empty, _dropped = encode_predictions(
        {
            "image": "empty.jpg",
            "width": 100,
            "height": 100,
            "boxes": [],
            "scores": [],
            "labels": [],
            "count": 0,
            "cap": 1,
        },
        "model:fixture",
        scope=scope,
    )
    assert empty["annotations"] == []


def test_web_worker_uses_generic_predictor_and_publishes_its_documents(
    tmp_path, monkeypatch
):
    pytest.importorskip("fastapi")
    import hashlib

    from tcip_web.routes.inference import _worker

    images_dir = _one_image(tmp_path)
    ckpt = _checkpoint(tmp_path)

    # A tiled run clears the worker's tile-geometry delivery gate only on a real basis for the
    # scale; this checkpoint persisted its own training geometry.
    predictor = install(monkeypatch, StubPredictor(train_tile_size=640))

    job = _job(
        "t",
        tmp_path,
        ckpt,
        images_dir,
        Stated(
            tile=True,
            conf=0.25,
            cross_tile_nms=0.7,
            overlap=0.2,
            postprocess="nmm",
            rationale=RATIONALE,
        ),
    )
    _worker(job)

    assert job.status == "completed"
    assert job.done == 1 and job.total == 1
    assert predictor.checkpoint == str(ckpt)
    (execution,) = predictor.executions
    assert execution.tile_size == 640  # tile=True -> pipeline tiling
    assert (
        execution.postprocess == "nmm"
    )  # the GUI's merge choice reaches inference

    (obj,) = _annotations(tmp_path)
    assert (
        obj["subject"] == "bud"
    )  # decoded through the checkpoint's recorded map
    assert obj["score"] == pytest.approx(
        0.9
    )  # per-object confidence preserved
    assert obj["bbox"] == [
        10.0,
        10.0,
        20.0,
        20.0,
    ]  # pixel COCO xywh from xyxy [10,10,30,30]
    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()[:12]
    assert obj["created_by"] == f"model:{ckpt.stem}@{digest}"


def test_web_worker_prefers_the_checkpoints_own_recorded_scope(
    tmp_path, monkeypatch
):
    """The GUI inference worker decodes through the checkpoint's own recorded scope.

    No subjects.json exists over the images; the object's attribute decodes to ``open``, and
    the bucket's scope carries the recorded attribute.
    """
    pytest.importorskip("fastapi")
    from tcip_mcp import subject_registry as cr
    from tcip_mcp.buckets import read_bucket
    from tcip_web.routes.inference import _worker

    opening = cr.Attribute("opening", "categorical", ("closed", "open"))
    images_dir = _one_image(tmp_path)
    ckpt = _checkpoint(
        tmp_path,
        data={"num_channels": 3, "scope": {"subject": "bud"}},
        registry=cr.SubjectRegistry(
            subjects=(cr.Subject(name="bud", attributes=(opening,)),)
        ),
    )

    install(monkeypatch, StubPredictor(attributes=[[1]]))

    job = _job(
        "t3",
        tmp_path,
        ckpt,
        images_dir,
        Stated(
            tile=False,
            conf=0.25,
            cross_tile_nms=0.7,
            overlap=0.2,
            postprocess="nms",
            rationale=RATIONALE,
        ),
    )
    _worker(job)

    assert job.status == "completed"
    (obj,) = _annotations(tmp_path)
    # id 1 of the recorded "opening" is "open", under attributes; subject carries the object.
    assert obj["subject"] == "bud"
    assert obj["attributes"] == {"opening": "open"}
    assert read_bucket(tmp_path, BUCKET).scope.attributes == (opening,)


def test_web_worker_runs_tiled_instance_seg_without_forcing_untiled(
    tmp_path, monkeypatch
):
    """An instance_seg checkpoint launched tiled from the GUI hands its predictor a tiled pass.

    The job completes stating tile, the bucket's execution records a tiled pass, and the
    recorded merge threshold's source is ``explicit``.
    """
    pytest.importorskip("fastapi")
    from tcip_mcp.buckets import read_bucket
    from tcip_web.routes.inference import _worker

    images_dir = _one_image(tmp_path)
    ckpt = _checkpoint(tmp_path)

    # A real basis for the tile scale, or the delivery gate refuses.
    predictor = install(
        monkeypatch, StubPredictor(task="instance_seg", train_tile_size=640)
    )

    job = _job(
        "t3",
        tmp_path,
        ckpt,
        images_dir,
        Stated(
            tile=True,
            conf=0.25,
            cross_tile_nms=0.7,
            overlap=0.2,
            postprocess="nms",
            rationale=RATIONALE,
        ),
    )
    _worker(job)

    assert job.status == "completed"  # no crash
    assert predictor.executions[
        0
    ].tiled  # the breeder's own checkbox choice is honored
    assert job.stated.tile is True
    execution = read_bucket(tmp_path, BUCKET).execution
    assert execution.tiled
    # The stated merge threshold is recorded as the caller's own statement.
    assert execution.sources["cross_tile_nms"] == "explicit"


def test_web_worker_runs_a_native_frame_tile_scale_and_forwards_its_recorded_resize(
    tmp_path, monkeypatch
):
    """The GUI worker hands its predictor a native-frame tile scale and the recorded resize.

    For a checkpoint whose geometry is its untiled 64x64 training frame, the predictor's one
    execution carries tile edge 64, sourced ``native_ratio``, and the recorded 128x128 resize.
    """
    pytest.importorskip("fastapi")
    from tcip_web.routes.inference import _worker

    images_dir = _one_image(tmp_path)
    ckpt = _checkpoint(tmp_path)
    predictor = install(
        monkeypatch,
        StubPredictor(
            boxes=(),
            scores=(),
            task="detection",
            train_tile_size=None,
            train_native_size=(64, 64),
            train_augmentation=AugmentationSpec.model_validate(
                augmentation(resize={"size": [128, 128]})
            ),
        ),
    )

    job = _job(
        "t4",
        tmp_path,
        ckpt,
        images_dir,
        Stated(
            tile=True,
            conf=0.25,
            cross_tile_nms=0.7,
            overlap=0.2,
            postprocess="nms",
            rationale=RATIONALE,
        ),
    )
    _worker(job)

    assert job.status == "completed", job.error
    (execution,) = predictor.executions
    assert (execution.tile_size, execution.tile_resize) == (64, (128, 128))
    assert execution.sources["tile_size"] == "native_ratio"


def _sources(project: Path) -> dict:
    from tcip_mcp.buckets import read_bucket

    return read_bucket(project, BUCKET).execution.sources


def test_web_worker_stamps_the_stated_conf_explicit_and_the_checkpoints_density(
    tmp_path, monkeypatch
):
    """The GUI worker records a stated conf's source as explicit and the density's as derived.

    The job completes with one published annotation, and the recorded sources read
    ``explicit`` for conf and ``derived`` for density.
    """
    from tcip_web.routes.inference import _worker
    from tests._verified_checkpoint_fixtures import SAMPLE_CONF

    images_dir = _one_image(tmp_path)
    ckpt = _checkpoint(tmp_path)
    install(monkeypatch, StubPredictor())

    job = _job(
        "conf-explicit",
        tmp_path,
        ckpt,
        images_dir,
        Stated(
            tile=False,
            conf=SAMPLE_CONF,
            cross_tile_nms=0.7,
            rationale=RATIONALE,
        ),
    )
    _worker(job)

    assert job.status == "completed", job.error
    # The pass ran at the stated values: the one image's one detection landed.
    assert len(_annotations(tmp_path)) == 1
    sources = _sources(tmp_path)
    assert (sources["conf"], sources["density"]) == ("explicit", "derived")


def test_web_worker_fails_a_job_stating_no_conf_naming_it(
    tmp_path, monkeypatch
):
    """A GUI inference job stating no conf fails, naming it.

    The job's status reads failed, and its error names conf.
    """
    from tcip_web.routes.inference import _worker

    images_dir = _one_image(tmp_path)
    ckpt = _checkpoint(tmp_path)
    install(monkeypatch, StubPredictor())

    job = _job(
        "conf-unstated",
        tmp_path,
        ckpt,
        images_dir,
        Stated(tile=False, cross_tile_nms=0.7, rationale=RATIONALE),
    )
    _worker(job)

    assert job.status == "failed"
    assert "conf" in (job.error or ""), job.error


def test_web_worker_dropped_boxes_agree_with_the_published_document_on_a_degenerate_box(
    tmp_path,
    monkeypatch,
):
    """The job's dropped-box count agrees with the published document on a degenerate box.

    Of the model's two raw detections, one has zero width: the job counts one dropped box, and
    the published document holds one object.
    """
    pytest.importorskip("fastapi")
    from tcip_web.routes.inference import _worker

    images_dir = _one_image(tmp_path)
    ckpt = _checkpoint(tmp_path)

    install(
        monkeypatch,
        StubPredictor(
            boxes=(BOX, (50.0, 50.0, 50.0, 60.0)),
            scores=(0.9, 0.4),
            train_tile_size=640,
        ),
    )

    job = _job(
        "degenerate",
        tmp_path,
        ckpt,
        images_dir,
        Stated(
            tile=True,
            conf=0.25,
            cross_tile_nms=0.7,
            overlap=0.2,
            postprocess="nmm",
            rationale=RATIONALE,
        ),
    )
    _worker(job)

    assert job.status == "completed"
    assert job.dropped_boxes == 1
    assert len(_annotations(tmp_path)) == 1


def test_web_worker_fails_the_job_on_a_stem_collision(tmp_path):
    """The GUI worker fails the job on a stem collision.

    The worker's pass enumerates through ``image_utils.list_logical_images``: a capture holding
    two logical identities under one case-folded stem fails the job with an error naming the
    logical image.
    """
    from tcip_web.routes.inference import _worker

    images_dir = _one_image(tmp_path, "foo")
    from PIL import Image

    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "foo.png")
    ckpt = _checkpoint(tmp_path)

    job = _job(
        "collision",
        tmp_path,
        ckpt,
        images_dir,
        Stated(
            tile=False,
            conf=0.25,
            cross_tile_nms=0.7,
            overlap=0.2,
            postprocess="nms",
            rationale=RATIONALE,
        ),
    )
    _worker(job)

    assert job.status == "failed"
    assert "logical image" in job.error
