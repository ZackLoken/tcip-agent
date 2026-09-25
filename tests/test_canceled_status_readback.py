"""A canceled run or job reads back as ``canceled`` through each reader of its status: the web job
registry's rehydrate, the experiment record's derived state, and the frontend's generated union.
Each status is written by its own producer, never spelled into a fixture."""

from __future__ import annotations

import re
import typing
from pathlib import Path

import pytest

GENERATED_TYPES = (Path(__file__).resolve().parents[1] / "packages" / "tcip-web" / "frontend"
                   / "src" / "api" / "types.generated.ts")


def test_a_canceled_inference_job_rehydrates_as_canceled(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    monkeypatch.chdir(tmp_path)
    from PIL import Image

    from tcip_web.routes import inference
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (16, 16)).save(images_dir / "img.jpg")
    ckpt = registered_checkpoint(tmp_path, project_root=tmp_path)

    job = inference.InferenceJob(job_id="canceled-job", checkpoint_path=ckpt,
                                 images_dir=str(images_dir), output_dir=str(tmp_path / "out"),
                                 tile=False, conf=0.25, iou=0.7, overlap=0.2)
    inference._register(job)
    job.cancel_event.set()
    try:
        inference._worker(job)
        inference._registry.jobs.clear()
        inference.rehydrate_for_current_root()
        assert inference._registry.jobs["canceled-job"].status == "canceled"
    finally:
        inference._registry.jobs.clear()


def _train_stops_on_cancel(ctx):
    """A body that honors a cancellation request before training anything."""
    ctx.run.cancel_event.set()


def test_a_canceled_training_run_derives_as_canceled_from_its_experiment_record(tmp_path):
    pytest.importorskip("torch")
    from tcip_mcp.experiments import (
        create_experiment, derived_state, read_member, status_key, update_status,
    )
    from tcip_mcp.pipelines.training.envelope import TrainContext, run_training_envelope
    from tcip_mcp.pipelines.training.run_registry import create_run, draw_seed_if_unset

    config = {
        "model_source": {"builder": "x:y", "task": "detection", "in_chans": 3},
        "training_source": f"{__name__}:_train_stops_on_cancel",
        "device": "cpu",
    }
    create_experiment("exp-canceled", config, data_source="imgs")
    update_status("exp-canceled", "running")
    draw_seed_if_unset(config)
    run = create_run(config, str(tmp_path / "out"), id="canceled-run")
    run_training_envelope(TrainContext(run=run, train_loader=None, val_loader=None,
                                       task="detection", experiment_id="exp-canceled"))

    status = read_member(status_key("exp-canceled"), None)
    assert derived_state(status, 600.0) == "canceled"


def test_the_generated_job_status_union_is_the_backends_own():
    from tcip_web.jobstore import JobStatus

    declared = re.search(r"export type JobStatus = ([^;]+);",
                         GENERATED_TYPES.read_text(encoding="utf-8"))
    assert declared is not None
    members = set(re.findall(r'"([^"]+)"', declared.group(1)))
    assert members == set(typing.get_args(JobStatus))
    assert "canceled" in members
