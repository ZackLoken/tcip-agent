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
    pred.score_threshold = 0.0
    pred._format_detection = lambda out, path, w, h: {"image": path, "count": 0}

    calls = {"n": 0, "sizes": []}

    class FakeDet(torch.nn.Module):
        def forward(self, images):
            calls["n"] += 1
            calls["sizes"].append(len(images))
            return [{} for _ in images]

    pred.model = FakeDet()

    results = pred.predict_batch(paths, batch_size=2)
    assert len(results) == 5            # one result per image
    assert calls["n"] == 3              # ceil(5/2) forwards, not 5
    assert calls["sizes"] == [2, 2, 1]


def test_get_experiment_metrics_pagination(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment, get_experiment, log_metrics

    create_experiment("exp1", {"model_source": {"builder": "x:y"}})
    for e in range(10):
        log_metrics("exp1", e, {"loss": float(e)})

    full = get_experiment("exp1")
    assert full["n_epochs"] == 10 and len(full["metrics"]) == 10

    page = get_experiment("exp1", metrics_offset=3, metrics_limit=4)
    assert page["n_epochs"] == 10       # true total preserved even when paginated
    assert len(page["metrics"]) == 4
    assert page["metrics"][0]["epoch"] == 3
    assert page["metrics_offset"] == 3


def test_get_experiment_n_epochs_counts_distinct_values_not_rows(tmp_path, monkeypatch):
    """n_epochs is the count of distinct epoch values, not the row count: a bespoke loop logging
    train and val as separate rows under the same epoch still counts as one epoch. n_rows is the
    row count, and is what metrics_offset/metrics_limit actually page against."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment, get_experiment, log_metrics

    create_experiment("exp2", {"model_source": {"builder": "x:y"}})
    log_metrics("exp2", 3, {"loss_train": 0.5})
    log_metrics("exp2", 3, {"loss_val": 0.4})

    result = get_experiment("exp2")
    assert result["n_rows"] == 2
    assert result["n_epochs"] == 1
