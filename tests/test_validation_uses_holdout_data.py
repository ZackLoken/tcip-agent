"""The per-epoch validation pass scores the held-out loader, not the training loader.

These runs give the train and val sides loss landscapes that pull in opposite directions, so
the recorded ``val_`` metrics, the selection value, the best checkpoint and early stopping each
show which side was scored.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from tcip_mcp.pipelines.model_build import METRICS_KEY
from tcip_mcp.pipelines.training.evaluation import evaluate
from tcip_mcp.pipelines.training.generic_trainer import train
from tests._training_values import IOU_THRESHOLD
from tests.tiny_trainer_fixtures import (
    TRAIN_INTENSITIES,
    VAL_INTENSITIES,
    capture_model,
    opposed_regression_data,
    regressor_config,
    run_context,
)


def _context(tmp_path, horizon: int, **overrides):
    """Return the context of a regression run over opposed frames under ``tmp_path``.

    The run is opened by the launcher's own producer.
    """
    data = opposed_regression_data(
        tmp_path / "ds", TRAIN_INTENSITIES, VAL_INTENSITIES
    )
    return run_context(
        tmp_path,
        regressor_config(
            horizon,
            data=data,
            project=tmp_path,
            builder_kwargs={"init_weight": 0.0},
            **overrides,
        ),
    )


def test_the_run_a_test_trains_is_the_one_its_directory_records(tmp_path):
    """The context ``_context`` builds holds the run its directory records.

    Its run is the one ``observed_run`` reads back from the directory the launcher's own producer
    opened, its loaders hold the recorded partition's sides, and the validation side holds the
    validation frames the fixture wrote and no other.
    """
    from tcip_mcp.experiments import observe
    from tcip_mcp.pipelines.training.run_registry import observed_run
    from tests._verified_checkpoint_fixtures import partition_side

    ctx = _context(tmp_path, 1)
    observation = observe(ctx.run_dir)
    recorded = observed_run(observation)
    partition = observation.record["resolved"]["partition"]

    assert (ctx.run.id, ctx.run.spec.record(), ctx.run.trait_revision) == (
        recorded.id,
        recorded.spec.record(),
        recorded.trait_revision,
    )
    for side, loader in (("train", ctx.train_loader), ("val", ctx.val_loader)):
        dataset = loader.dataset
        assert sorted(
            dataset.sample_of(key).member for key in dataset.stems
        ) == partition_side(partition, side)
    assert partition_side(partition, "val") == sorted(
        f"img{len(TRAIN_INTENSITIES) + i}" for i in range(len(VAL_INTENSITIES))
    )


def test_recorded_val_metrics_match_an_evaluation_of_the_holdout_loader(
    tmp_path, monkeypatch
):
    """An epoch's recorded ``val_`` metrics match an evaluation of the holdout loader.

    They agree, within 1e-6, with a real evaluation of that epoch's model on the holdout
    loader, and differ from the evaluation run over the training loader.
    """
    models: list = []
    capture_model(monkeypatch, models)

    from tcip_mcp.experiments import METRICS_FILE, partition_rows, read_rows

    ctx = _context(tmp_path, 1)
    # The production wiring: the trainer hands each row to the envelope's sink, which logs it
    # to the run's own metrics log.
    run = ctx.default_train()

    assert run.status == "completed", run.status_error
    assert len(models) == 1
    model = models[0]

    device = torch.device("cpu")
    on_holdout = evaluate(
        model,
        ctx.val_loader,
        device,
        "regression",
        dims={"in_chans": 3},
        conf_threshold=None,
        iou_threshold=IOU_THRESHOLD,
        score_weights=None,
        trait=None,
        density=None,
        mask_cut=None,
    )
    on_training = evaluate(
        model,
        ctx.train_loader,
        device,
        "regression",
        dims={"in_chans": 3},
        conf_threshold=None,
        iou_threshold=IOU_THRESHOLD,
        score_weights=None,
        trait=None,
        density=None,
        mask_cut=None,
    )
    # The two loaders must be distinguishable at all, or nothing below can discriminate.
    assert on_holdout["loss"] > 3.0 * on_training["loss"] > 0.0
    assert on_holdout["mae"] != pytest.approx(on_training["mae"], rel=0.1)

    record = run.metrics_history[-1]
    assert record["val_loss"] == pytest.approx(on_holdout["loss"], abs=1e-6)
    assert record["val_mae"] == pytest.approx(on_holdout["mae"], abs=1e-6)
    assert record["val_rmse"] == pytest.approx(on_holdout["rmse"], abs=1e-6)
    # A regression run selects by val_loss, so the checkpoint objective is the holdout number too.
    assert run.best_metric == pytest.approx(on_holdout["loss"], abs=1e-6)

    persisted = partition_rows(read_rows(ctx.run_dir / METRICS_FILE)[0])[0]
    assert len(persisted) == 1
    assert persisted[0]["val_loss"] == pytest.approx(
        on_holdout["loss"], abs=1e-6
    )


def test_best_checkpoint_and_early_stopping_follow_the_holdout_loader(
    tmp_path, monkeypatch
):
    """The best checkpoint and early stopping follow the holdout loader.

    With holdout loss worsening while training loss improves, the run stops after two of its
    four epochs, and its best metric and ``model_best`` checkpoint are epoch 1's holdout loss.
    """
    models: list = []
    capture_model(monkeypatch, models)

    from tests._training_values import stop_rule

    ctx = _context(tmp_path, 4, early_stopping=stop_rule(1, 1e-4))
    run = train(ctx.run, ctx.train_loader, val_loader=ctx.val_loader)

    assert run.status == "completed", run.status_error
    history = run.metrics_history
    assert len(history) == 2, history
    # The two directions disagree: training improves epoch over epoch while holdout degrades.
    assert history[1]["train_loss"] < history[0]["train_loss"]
    assert history[1]["val_loss"] > history[0]["val_loss"]

    assert run.best_metric == pytest.approx(history[0]["val_loss"], abs=1e-6)
    best = torch.load(ctx.run_dir / "model_best.pt", weights_only=False)
    assert best["epoch"] == 1
    assert best[METRICS_KEY]["val_loss"] == pytest.approx(
        history[0]["val_loss"], abs=1e-6
    )
