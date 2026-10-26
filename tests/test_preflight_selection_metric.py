"""Preflight's smoke reads a run's selection metric from the row its evaluator reports.

For a run trained by the default body, the smoke calls the trainer's own
``generic_trainer.selection_value`` over the validation row it admitted, or the empty row of a run
with no validation side, so a metric the run's evaluator does not report blocks the launch and the
sweep.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._chain_fixtures import training_config
from tests._training_values import drawn_split, evaluation_block

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tests.test_correctness_loop_wired import (  # noqa: E402
    _admitted_tree,
    _semantic_seg_config,
)


def _selecting(config: dict, metric: str, **stated: object) -> dict:
    """Return ``config`` with an evaluation block selecting on ``metric``, ``stated`` over it."""
    return {
        **config,
        "evaluation": evaluation_block(selection_metric=metric, **stated),
    }


def _epoch_row_issues(report: dict) -> list[str]:
    """Return the issues of a preflight ``report`` its smoke's epoch row raised."""
    return [i for i in report["issues"] if i.startswith("epoch row:")]


def test_a_metric_the_evaluator_does_not_report_refuses_at_the_smoke_as_at_the_first_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A semantic segmentation run selecting ``objective`` refuses at the smoke and at epoch one.

    With ``evaluation.score_weights`` stated, the smoke refuses naming the metric, a run selecting
    ``mIoU`` passes, and the refused config trained fails at its first epoch with the sentence the
    smoke's issue carries.
    """
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import observe
    from tcip_mcp.tools.training_tools import preflight_config
    from tests._verified_checkpoint_fixtures import worker_run

    config = _semantic_seg_config(tmp_path)
    objective = _selecting(config, "objective")
    assert objective["evaluation"]["score_weights"] is not None

    refused = preflight_config(tmp_path, objective, smoke=True)
    passed = preflight_config(tmp_path, _selecting(config, "mIoU"), smoke=True)

    assert refused["valid"] is False
    (issue,) = _epoch_row_issues(refused)
    assert "'objective'" in issue and "val_mIoU" in issue, issue
    assert passed["valid"] is True, passed["issues"]

    final = observe(worker_run(tmp_path, objective)).final
    assert final is not None and final.state == "failed", final
    assert issue == f"epoch row: {final.status_error}"


REPORTING_BODY = """\
import torch

from tcip_mcp.pipelines.model_build import STATE_DICT_KEY


def train(ctx):
    model = ctx.build_model().to(ctx.device)
    optimizer = ctx.build_optimizer(model)
    model.train()
    total = 0.0
    for images, targets in ctx.train_loader:
        optimizer.zero_grad()
        out = model(
            images.to(ctx.device),
            {k: v.to(ctx.device) if torch.is_tensor(v) else v for k, v in targets.items()},
        )
        loss = sum(out.values()) if isinstance(out, dict) else out
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
    ctx.log_metrics(1, {"train_loss": total})
    ctx.report_objective(total)
    ctx.save_checkpoint({STATE_DICT_KEY: model.state_dict()}, "model_final")
"""
"""A semantic segmentation ``train(ctx)`` body: one pass, its summed loss reported as the
selection through ``ctx.report_objective``."""

DELEGATING_BODY = "def train(ctx):\n    ctx.default_train()\n"
"""A ``train(ctx)`` body handing the run to the default trainer."""


def _with_body(project: Path, config: dict, name: str, text: str) -> dict:
    """Return ``config`` trained by the body ``text``, written as ``name.py`` under ``project``.

    The body's file is declared among the model source's files, where ``training_source``
    imports from.
    """
    body = project / f"{name}.py"
    body.write_text(text, encoding="utf-8")
    source = config["model_source"]
    return {
        **config,
        "model_source": {
            **source,
            "source_files": [*source["source_files"], str(body)],
        },
        "training_source": f"{name}:train",
    }


def test_a_body_reporting_its_own_selection_is_admitted_and_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``training_source`` body reporting its own selection passes the smoke and completes.

    A semantic segmentation run selecting ``objective``, which the default evaluator does not
    report, passes the smoke with a warning naming the metric; launched, its answer carries that
    warning, and the run completes with its ``selection`` row under ``objective``.
    """
    monkeypatch.chdir(tmp_path)
    from tcip_mcp import experiments
    from tcip_mcp.tools.training_tools import (
        RunLaunched,
        launch_training,
        preflight_config,
    )
    from tests._verified_checkpoint_fixtures import run_to_end

    config = _with_body(
        tmp_path,
        _selecting(_semantic_seg_config(tmp_path), "objective"),
        "reporting_body",
        REPORTING_BODY,
    )

    report = preflight_config(tmp_path, config, smoke=True)
    assert report["valid"] is True, report["issues"]
    (warning,) = [w for w in report["warnings"] if "training_source" in w]
    assert "'objective'" in warning, warning

    launched = launch_training(tmp_path, config, actor=None)
    assert isinstance(launched, RunLaunched), launched
    assert warning in launched.warnings, launched
    assert run_to_end(tmp_path, launched.experiment_id)["state"] == (
        "completed"
    )
    run_dir = experiments.experiment_dir(
        launched.experiment_id, project=tmp_path
    )
    rows = experiments.epoch_rows(
        experiments.read_rows(run_dir / experiments.METRICS_FILE)[0]
    )
    assert any(
        row.get("selection_metric") == "objective" and "selection" in row
        for row in rows
    ), rows


def test_a_sweeps_answer_carries_its_base_preflights_warnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sweep over a base config whose preflight warns answers that warning.

    The base config is a reporting ``training_source`` body selecting ``objective``; the search
    itself is replaced by one that runs no trial.
    """
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import run_hyperparameter_search
    from tests._training_values import fifo_search, sweep_space

    config = _with_body(
        tmp_path,
        _selecting(_semantic_seg_config(tmp_path), "objective"),
        "reporting_body",
        REPORTING_BODY,
    )
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search", lambda **_kw: None
    )

    answer = run_hyperparameter_search(
        tmp_path,
        base_config=config,
        param_space=sweep_space(),
        n_trials=1,
        **fifo_search(),
        search_seed=0,
        auto_tensorboard=False,
        actor=None,
    )

    assert "error" not in answer, answer
    assert any(
        "'objective'" in w and "training_source" in w
        for w in answer["warnings"]
    ), answer


def test_a_body_delegating_to_the_default_trainer_is_read_at_its_first_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``training_source`` delegating to ``ctx.default_train()`` is read at its first epoch.

    Selecting ``objective``, which the default evaluator does not report for semantic
    segmentation, it passes the smoke with the warning, and trained it fails at its first epoch
    with ``selection_value``'s refusal naming the metric.
    """
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import observe
    from tcip_mcp.tools.training_tools import preflight_config
    from tests._verified_checkpoint_fixtures import worker_run

    delegating = _with_body(
        tmp_path,
        _selecting(_semantic_seg_config(tmp_path), "objective"),
        "delegating_body",
        DELEGATING_BODY,
    )

    report = preflight_config(tmp_path, delegating, smoke=True)
    assert report["valid"] is True, report["issues"]
    assert any("ctx.default_train()" in w for w in report["warnings"]), report

    final = observe(worker_run(tmp_path, delegating)).final
    assert final is not None and final.state == "failed", final
    assert final.status_error is not None
    assert "'objective'" in final.status_error, final.status_error
    assert "is not in task 'semantic_seg'" in final.status_error


def test_a_detector_selecting_an_unweighted_objective_refuses_at_the_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A detector run selecting ``objective`` with no score weights refuses at the smoke.

    The issue names ``evaluation.score_weights``; the same run stating them passes.
    """
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import preflight_config
    from tests._verified_checkpoint_fixtures import BUILT_DETECTOR

    config = training_config(
        tmp_path,
        BUILT_DETECTOR,
        {
            "images_dir": str(_admitted_tree(tmp_path)),
            "scope": {"subject": "leaf"},
            "split": drawn_split(val_ratio=0.15, grouping="tile_prefix"),
        },
        batch_size=1,
    )

    refused = preflight_config(
        tmp_path,
        _selecting(config, "objective", score_weights=None),
        smoke=True,
    )
    weighted = preflight_config(
        tmp_path, _selecting(config, "objective"), smoke=True
    )

    (issue,) = _epoch_row_issues(refused)
    assert "evaluation.score_weights" in issue, issue
    assert weighted["valid"] is True, weighted["issues"]


def test_a_run_with_no_validation_side_selects_on_the_loss_alone_at_the_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run with no validation side refuses any selection metric but ``loss`` at the smoke.

    Selecting ``mIoU`` refuses naming the metric; selecting ``loss`` passes.
    """
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import preflight_config

    config = _semantic_seg_config(tmp_path)
    config["data"]["split"] = drawn_split(val_ratio=0.0, seed=3)

    refused = preflight_config(
        tmp_path, _selecting(config, "mIoU"), smoke=True
    )
    passed = preflight_config(tmp_path, config, smoke=True)

    (issue,) = _epoch_row_issues(refused)
    assert "'mIoU'" in issue and "no validation side" in issue, issue
    assert passed["valid"] is True, passed["issues"]
    assert passed["smoke"]["evaluation_not_run"] is not None


def test_a_sweep_whose_base_config_selects_an_unreported_metric_opens_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sweep over a base config selecting a metric its evaluator does not report refuses.

    The refusal comes before the sweep's directory is created, naming the metric.
    """
    monkeypatch.chdir(tmp_path)
    from tcip_mcp import experiments
    from tcip_mcp.tools.training_tools import open_sweep
    from tests._verified_checkpoint_fixtures import SWEEP_ARGUMENTS

    arguments = dict(SWEEP_ARGUMENTS)
    opened = open_sweep(
        tmp_path,
        _selecting(_semantic_seg_config(tmp_path), "objective"),
        arguments.pop("param_space"),
        actor=None,
        **arguments,
    )

    assert isinstance(opened, dict), opened
    (issue,) = [i for i in opened["issues"] if i.startswith("epoch row:")]
    assert "'objective'" in issue, issue
    assert experiments.sweep_dirs(tmp_path) == []
