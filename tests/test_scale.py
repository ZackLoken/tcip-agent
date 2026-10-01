"""GPU-batched detection inference + paginated experiment metrics."""

import pytest


def test_predict_batch_detection_uses_one_forward_per_batch(tmp_path):
    pytest.importorskip("torch")
    import torch
    from PIL import Image

    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor

    paths = []
    for i in range(5):
        p = tmp_path / f"img{i}.png"
        Image.new("RGB", (16, 16)).save(p)
        paths.append(str(p))

    # Construct without a real checkpoint; we only exercise the batching path.
    pred = GenericPredictor.__new__(GenericPredictor)
    pred.device = torch.device("cpu")
    pred.in_chans = 3
    pred.task = "detection"

    calls = {"n": 0, "sizes": []}

    class FakeDet(torch.nn.Module):
        def forward(self, images):
            calls["n"] += 1
            calls["sizes"].append(len(images))
            return [{"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0),
                     "labels": torch.zeros(0, dtype=torch.int64)} for _ in images]

    pred.model = FakeDet()

    from types import SimpleNamespace

    from tcip_mcp.pipelines.execution import untiled_execution

    detector = SimpleNamespace(task="detection", path="detector.pt")
    results = pred.predict_batch(paths, untiled_execution(detector, conf=0.0, max_dets=None),
                                 batch_size=2)
    assert [r["count"] for r in results] == [0] * 5  # one result per image
    assert calls["n"] == 3              # ceil(5/2) forwards, not 5
    assert calls["sizes"] == [2, 2, 1]


def test_get_experiment_metrics_pagination(tmp_path):
    from tcip_mcp.experiments import get_experiment
    from tests._verified_checkpoint_fixtures import detection_config, log_epoch, opened_run

    run_dir = opened_run(tmp_path, detection_config(tmp_path / "run-data"), experiment_id="exp1")
    for e in range(10):
        log_epoch(run_dir, e, {"loss": float(e)})

    full = get_experiment("exp1", project=tmp_path)
    assert full["n_epochs"] == 10 and len(full["metrics"]) == 10

    page = get_experiment("exp1", metrics_offset=3, metrics_limit=4, project=tmp_path)
    assert page["n_epochs"] == 10       # true total preserved even when paginated
    assert len(page["metrics"]) == 4
    assert page["metrics"][0]["epoch"] == 3
    assert page["metrics_offset"] == 3


def test_get_experiment_n_epochs_counts_distinct_values_not_rows(tmp_path):
    """n_epochs is the count of distinct epoch values, not the row count: a bespoke loop logging
    train and val as separate rows under the same epoch still counts as one epoch. n_rows is the
    row count, and is what metrics_offset/metrics_limit actually page against."""
    from tcip_mcp.experiments import get_experiment
    from tests._verified_checkpoint_fixtures import detection_config, log_epoch, opened_run

    run_dir = opened_run(tmp_path, detection_config(tmp_path / "run-data"), experiment_id="exp2")
    log_epoch(run_dir, 3, {"loss_train": 0.5})
    log_epoch(run_dir, 3, {"loss_val": 0.4})

    result = get_experiment("exp2", project=tmp_path)
    assert result["n_epochs"] == 1
    assert result["n_rows"] == 2
