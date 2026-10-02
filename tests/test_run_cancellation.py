"""Training-run cancellation (cancel_training; graceful stop)."""

import pytest
from tests._producer_fixtures import run_over  # noqa: E402

torch = pytest.importorskip("torch")


def test_cancel_training_reaches_the_runs_own_poll(tmp_path):
    """A cancel requested by id is the one record the run's own poll reads; an id naming no run
    refuses."""
    from tcip_mcp.experiments import RUN_FILE, read_record
    from tcip_mcp.pipelines.training.run_registry import TrainRun, trained_config
    from tcip_mcp.tools.training_tools import cancel_training
    from tests._verified_checkpoint_fixtures import detection_config, opened_run

    run_dir = opened_run(tmp_path, detection_config(tmp_path / "data"), experiment_id="cancel-run-1")
    record = read_record(run_dir / RUN_FILE)
    run = TrainRun(id=run_dir.name, config=trained_config(record),
                   objective=record["resolved"]["objective"], project=tmp_path,
                   output_dir=str(run_dir))
    assert not run.should_cancel()

    res = cancel_training(tmp_path, run.id)
    assert res["cancel_requested"] is True and res["experiment_id"] == run.id
    assert run.should_cancel()
    assert "error" in cancel_training(tmp_path, "missing-run")


def test_cancel_before_training_yields_canceled(tmp_path):
    pytest.importorskip("torchvision")
    from PIL import Image
    from torch.utils.data import DataLoader

    from tcip_mcp.pipelines.training.generic_trainer import train
    from tcip_mcp.pipelines.training.collation import task_collate
    from tcip_mcp.experiments import request_cancel
    from tests.tiny_trainer_fixtures import trainer_run

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    rows = ["stem,label"]
    for i in range(4):
        Image.new("RGB", (32, 32), (20 * i, 30, 40)).save(images_dir / f"img{i}.png")
        rows.append(f"img{i},{i % 2}")
    (tmp_path / "labels.csv").write_text("\n".join(rows) + "\n")

    ds, data = run_over("classification", str(images_dir), str(tmp_path / "labels.csv"))
    loader = DataLoader(ds, batch_size=2, collate_fn=task_collate("classification"))
    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_classifier",
                         "task": "classification"},
        "data": data,
        "device": "cpu", "stages": [{"freeze_to": -1, "epochs": 3}],
        "mixed_precision": False, "early_stopping": {"enabled": False},
    }
    run = trainer_run(cfg, tmp_path / "out", project=tmp_path, has_val_loader=False,
                      id="cancel-run-2")
    (tmp_path / "out").mkdir()
    request_cancel(tmp_path / "out")  # request cancellation before any epoch runs
    run = train(run, loader)

    assert run.status == "canceled"
    assert run.current_epoch == 0                                  # stopped before training
    assert (tmp_path / "out" / "model_final.pt").is_file()         # partial progress still saved
