"""Training-run cancellation (cancel_training; graceful stop)."""

import pytest
from tcip_mcp.dataset_layout import UNDATED_BUCKET
from tests._chain_fixtures import CLASSIFIER_SOURCE
from tests._training_values import schedule

torch = pytest.importorskip("torch")


def test_cancel_training_reaches_the_runs_own_poll(tmp_path):
    """A cancel requested by id is the one record the run's own poll reads; an id naming no run
    refuses."""
    from tcip_mcp.experiments import observe
    from tcip_mcp.pipelines.training.run_registry import observed_run
    from tcip_mcp.tools.training_tools import cancel_training
    from tests._verified_checkpoint_fixtures import (
        detection_config,
        opened_run,
    )

    run_dir = opened_run(
        tmp_path,
        detection_config(tmp_path / "data"),
        experiment_id="cancel-run-1",
    )
    run = observed_run(observe(run_dir))
    assert not run.should_cancel()

    res = cancel_training(tmp_path, run.id, actor=None)
    assert res["cancel_requested"] is True and res["experiment_id"] == run.id
    assert run.should_cancel()
    assert "error" in cancel_training(tmp_path, "missing-run", actor=None)


def test_cancel_before_training_yields_canceled(tmp_path):
    pytest.importorskip("torchvision")
    from PIL import Image

    from tcip_mcp.pipelines.training.generic_trainer import train
    from tcip_mcp.experiments import request_cancel
    from tests._chain_fixtures import training_config
    from tests.tiny_trainer_fixtures import run_context

    images_dir = tmp_path / "ds" / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    rows = ["stem,label"]
    for i in range(4):
        Image.new("RGB", (32, 32), (20 * i, 30, 40)).save(
            images_dir / f"img{i}.png"
        )
        rows.append(f"img{i},{i % 2}")
    labels = tmp_path / "ds" / "labels.csv"
    labels.write_text("\n".join(rows) + "\n")

    cfg = training_config(
        CLASSIFIER_SOURCE,
        {
            "images_dir": str(images_dir),
            "labels_dir": str(labels),
            "auto_val": False,
        },
        stages=[{"freeze_to": -1}],
        scheduler=schedule("cosine", horizon_epochs=3),
    )
    ctx = run_context(tmp_path, cfg)
    request_cancel(ctx.run_dir)  # request cancellation before any epoch runs
    run = train(ctx.run, ctx.train_loader)

    assert run.status == "canceled"
    assert run.current_epoch == 0  # stopped before training
    assert (
        ctx.run_dir / "model_final.pt"
    ).is_file()  # partial progress still saved
