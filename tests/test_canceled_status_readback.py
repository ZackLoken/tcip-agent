"""A canceled run or job reads back as ``canceled`` through each reader of its status.

The readers are the web job registry and the run directory's derived state. Each status is
written by its own producer: the inference worker, and a training body requesting its own
cancellation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tcip_mcp.pipelines.training.envelope import TrainContext


def test_a_canceled_inference_job_reads_as_canceled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fastapi")
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.pipelines.execution import Stated
    from tcip_web.routes import inference
    from tests._producer_fixtures import write_image
    from tests._training_values import RATIONALE
    from tests._verified_checkpoint_fixtures import registered_checkpoint
    from tests._web_fixtures import inference_job

    images_dir = image_dir(tmp_path, "2026-01-01")
    write_image(images_dir / "img.jpg", (16, 16))
    ckpt = registered_checkpoint(tmp_path)

    job = inference_job(
        images_dir,
        tmp_path,
        job_id="canceled-job",
        checkpoint_path=ckpt,
        stated=Stated(
            tile=False,
            conf=0.25,
            cross_tile_nms=0.7,
            overlap=0.2,
            rationale=RATIONALE,
        ),
    )
    job.cancel_event.set()
    try:
        inference._registry.launch(job, inference._worker)
        assert job.thread is not None
        job.thread.join(timeout=60)
        assert inference._registry.get("canceled-job").status == "canceled"
    finally:
        inference._registry.jobs.clear()


def _train_stops_on_cancel(ctx: TrainContext) -> None:
    """Request the run's cancellation and train nothing."""
    from tcip_mcp.experiments import request_cancel

    request_cancel(ctx.run_dir)


def test_a_canceled_training_run_derives_as_canceled_from_its_directory(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from tcip_mcp.experiments import observe
    from tests._verified_checkpoint_fixtures import (
        detector_declaring,
        finished_run,
    )

    run_dir = finished_run(
        tmp_path,
        model_source=detector_declaring(__file__),
        training_source=f"{Path(__file__).stem}:_train_stops_on_cancel",
    )

    assert observe(run_dir).state == "canceled"
