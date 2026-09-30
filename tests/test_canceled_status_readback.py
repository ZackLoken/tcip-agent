"""A canceled run or job reads back as ``canceled`` through each reader of its status: the web job
registry, the run directory's derived state, and the frontend's generated union. Each status is
written by its own producer, never spelled into a fixture."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

GENERATED_TYPES = (Path(__file__).resolve().parents[1] / "packages" / "tcip-web" / "frontend"
                   / "src" / "api" / "types.generated.ts")


def test_a_canceled_inference_job_reads_as_canceled(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    monkeypatch.chdir(tmp_path)
    from PIL import Image

    from tcip_web.routes import inference
    from tests._verified_checkpoint_fixtures import foreign_checkpoint

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (16, 16)).save(images_dir / "img.jpg")
    ckpt = foreign_checkpoint(tmp_path)

    job = inference.InferenceJob(job_id="canceled-job", checkpoint_path=ckpt,
                                 images_dir=str(images_dir), output_dir=str(tmp_path / "out"),
                                 tile=False, conf=0.25, cross_tile_nms=0.7, overlap=0.2,
                                 project=str(tmp_path))
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
    from tests._verified_checkpoint_fixtures import finished_run

    run_dir = finished_run(tmp_path, training_source=f"{__name__}:_train_stops_on_cancel")

    assert observe(run_dir).state == "canceled"


def test_the_generated_job_status_union_is_the_backends_own():
    from tcip_web.jobstore import JOB_STATES

    declared = re.search(r"export type JobStatus = ([^;]+);",
                         GENERATED_TYPES.read_text(encoding="utf-8"))
    assert declared is not None
    members = set(re.findall(r'"([^"]+)"', declared.group(1)))
    assert members == set(JOB_STATES)
    assert "canceled" in members
