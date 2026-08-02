"""Training-run cancellation (cancel_run / cancel_training; graceful stop)."""

import pytest

torch = pytest.importorskip("torch")


def test_cancel_run_helper_and_tool():
    from tcip_mcp.pipelines.training.generic_trainer import cancel_run, create_run
    from tcip_mcp.tools.training_tools import cancel_training

    run = create_run({"model_source": {"builder": "x:y"}}, "out")
    assert cancel_run(run.run_id) is True
    assert run.cancel_event.is_set()
    assert cancel_run("no-such-run") is False

    res = cancel_training(run.run_id)
    assert res["cancel_requested"] is True and res["run_id"] == run.run_id
    assert "error" in cancel_training("missing-run")


def test_cancel_before_training_yields_cancelled(tmp_path):
    pytest.importorskip("torchvision")
    from PIL import Image
    from torch.utils.data import DataLoader

    from tcip_mcp.pipelines.data.datasets import build_dataset
    from tcip_mcp.pipelines.training.generic_trainer import create_run, task_collate, train

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    rows = ["stem,label"]
    for i in range(4):
        Image.new("RGB", (32, 32), (20 * i, 30, 40)).save(images_dir / f"img{i}.png")
        rows.append(f"img{i},{i % 2}")
    (tmp_path / "labels.csv").write_text("\n".join(rows) + "\n")

    ds = build_dataset("classification", images_dir=str(images_dir),
                       csv_path=str(tmp_path / "labels.csv"), num_classes=2)
    loader = DataLoader(ds, batch_size=2, collate_fn=task_collate("classification"))
    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_classifier",
                         "builder_kwargs": {"num_classes": 2}, "task": "classification"},
        "device": "cpu", "stages": [{"freeze_to": -1, "epochs": 3}],
        "mixed_precision": False, "early_stopping": {"enabled": False},
    }
    run = create_run(cfg, str(tmp_path / "out"))
    run.cancel_event.set()  # request cancellation before any epoch runs
    run = train(run, loader, task="classification")

    assert run.status == "cancelled"
    assert run.current_epoch == 0                                  # stopped before training
    assert (tmp_path / "out" / "model_final.pt").is_file()         # partial progress still saved
