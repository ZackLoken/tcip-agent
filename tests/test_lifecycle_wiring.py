"""Training to registry lifecycle wiring: a run's completion is its registration.

The registration carries the metrics the run's best checkpoint recorded, and a failed run
registers nothing.
"""

from pathlib import Path

from tcip_mcp.model_registry import Listed
from tcip_mcp.pipelines.model_build import METRICS_KEY


def _stock_run(
    root: Path,
    builder: str,
    horizon: int,
    experiment_id: str,
    builder_kwargs: dict | None = None,
) -> Path:
    """Run the regression ``builder`` at ``builder_kwargs`` and return the run directory.

    The run's stage ends at its schedule's ``horizon``, under ``root`` over a regression dataset
    of its own on disk (four training and two validation frames, each valued twice its
    intensity), opened by the launcher's own writer and run in-process through the envelope's
    stock trainer over its own loaders.
    """
    from tcip_mcp.pipelines.training.envelope import run_training_envelope
    from tests.tiny_trainer_fixtures import (
        drawn_data,
        regressor_config,
        run_context,
    )

    data = drawn_data(
        root / f"{experiment_id}-data",
        ([0.1, 0.3, 0.5, 0.7], [0.2, 0.6, 1.0, 1.4]),
        ([0.2, 0.6], [0.4, 1.2]),
    )
    config = regressor_config(
        horizon,
        builder=builder,
        builder_kwargs=builder_kwargs,
        data=data,
        project=root,
    )
    ctx = run_context(root, config, experiment_id=experiment_id)
    run_training_envelope(ctx)
    return ctx.run_dir


def _entry(root: Path, name: str) -> Listed | None:
    """Return the registry entry ``root``'s listing names ``name``, or ``None``."""
    from tcip_mcp.model_registry import ModelRegistry

    return next(
        (
            m
            for m in ModelRegistry(str(root)).list_models()
            if m.entry.name == name
        ),
        None,
    )


def test_a_stock_trainer_run_registers_with_trainer_source_and_the_best_epochs_metrics(
    tmp_path: Path,
) -> None:
    """A stock-trainer run registers with the trainer source and its best epoch's metrics.

    A completed stock-trainer run (no training_source in config) registers
    metrics_source='trainer', carrying model_best.pt's own best-epoch metrics; model_final.pt's
    metrics is one dict.
    """
    import torch

    from tcip_mcp.experiments import observe
    from tests.tiny_trainer_fixtures import MEAN_INTENSITY_REGRESSOR

    run_dir = _stock_run(
        tmp_path,
        MEAN_INTENSITY_REGRESSOR,
        2,
        "exp-trainer-source",
        builder_kwargs={"init_weight": 0.0},
    )

    assert observe(run_dir).state == "completed", observe(run_dir).final
    entry = _entry(tmp_path, "exp-trainer-source")
    assert entry is not None
    assert entry.entry.experiment_id == "exp-trainer-source"
    assert entry.facts.metrics_source == "trainer"

    best = torch.load(run_dir / "model_best.pt", weights_only=False)
    assert entry.facts.metrics == best[METRICS_KEY]

    final = torch.load(run_dir / "model_final.pt", weights_only=False)
    assert isinstance(
        final[METRICS_KEY], dict
    )  # one mapping, never a per-epoch list


def test_a_diverged_stock_run_ends_failed_and_registers_nothing(
    tmp_path: Path,
) -> None:
    """A diverged stock run ends failed and registers nothing.

    Two full passes with no finite training loss end the run failed with the reason on its
    final status; the verdict leaves no checkpoint on disk and nothing reaches the registry.
    """
    from tcip_mcp.experiments import observe
    from tests.tiny_trainer_fixtures import ALWAYS_DIVERGED_MODEL

    run_dir = _stock_run(tmp_path, ALWAYS_DIVERGED_MODEL, 3, "exp-diverged")

    final = observe(run_dir).final
    assert final is not None and final.state == "failed"
    assert final.status_error is not None
    assert "2 consecutive full training passes" in final.status_error
    assert final.checkpoint is None
    assert not (run_dir / "model_best.pt").exists()
    assert not (run_dir / "model_final.pt").exists()
    assert _entry(tmp_path, "exp-diverged") is None


def test_a_completed_run_with_a_diverged_val_metric_registers_it_as_null(
    tmp_path: Path,
) -> None:
    """A completed run registers a diverged validation metric as null.

    A healthy training loss completes the run even when a validation metric goes
    non-finite; the final checkpoint carries that metric as null beside a ``_state`` key naming
    the value, and the registry entry carries it as null with metrics_source='trainer'.
    """
    import torch

    from tcip_mcp.experiments import observe
    from tests.tiny_trainer_fixtures import NAN_EVAL_REGRESSOR

    run_dir = _stock_run(tmp_path, NAN_EVAL_REGRESSOR, 3, "exp-nan-val")

    assert observe(run_dir).state == "completed", observe(run_dir).final

    final = torch.load(run_dir / "model_final.pt", weights_only=False)
    assert final[METRICS_KEY]["train_loss"] is not None
    assert final[METRICS_KEY]["val_mae"] is None
    assert final[METRICS_KEY]["val_mae_state"] == "nan"

    entry = _entry(tmp_path, "exp-nan-val")
    assert entry is not None
    assert entry.facts.metrics_source == "trainer"
    assert entry.facts.metrics["val_mae"] is None
    assert entry.entry.checkpoint_path.endswith("model_best.pt")
