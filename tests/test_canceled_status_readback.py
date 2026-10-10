"""A canceled run or job reads back as ``canceled`` through each reader of its status: the web job
registry and the run directory's derived state. Each status is written by its own producer, never
spelled into a fixture."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_a_canceled_inference_job_reads_as_canceled(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    monkeypatch.chdir(tmp_path)
    from PIL import Image

    from tcip_mcp.pipelines.execution import Stated
    from tcip_web.routes import inference
    from tests._verified_checkpoint_fixtures import SAMPLE_MAX_DETS, registered_checkpoint

    images_dir = tmp_path / "images" / "2026-01-01"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(images_dir / "img.jpg")
    ckpt = registered_checkpoint(tmp_path)

    job = inference.InferenceJob(job_id="canceled-job", actor="user:tester", checkpoint_path=ckpt,
                                 images_dir=str(images_dir), dataset_root=str(tmp_path),
                                 bucket="out", project=str(tmp_path), stated=Stated(
                                     tile=False, conf=0.25, max_dets=SAMPLE_MAX_DETS,
                                     cross_tile_nms=0.7, overlap=0.2))
    inference._register(job)
    job.cancel_event.set()
    try:
        inference._worker(job)
        assert inference._registry.get("canceled-job").status == "canceled"
    finally:
        inference._registry.jobs.clear()


def _train_stops_on_cancel(ctx):
    """A body that receives a cancellation request before training anything."""
    from tcip_mcp.experiments import request_cancel

    request_cancel(ctx.run_dir)


def test_a_canceled_training_run_derives_as_canceled_from_its_directory(tmp_path):
    pytest.importorskip("torch")
    from tcip_mcp.experiments import observe
    from tests._verified_checkpoint_fixtures import detector_declaring, finished_run

    run_dir = finished_run(tmp_path, model_source=detector_declaring(__file__),
                           training_source=f"{Path(__file__).stem}:_train_stops_on_cancel")

    assert observe(run_dir).state == "canceled"
