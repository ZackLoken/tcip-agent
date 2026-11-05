"""A learning-rate range test derives a head-rate range as its own run, and the range gates.

``derive_lr_range`` runs the test, its reading derives ``[low, high)``, and a launch, a preflight or
a sweep whose config names the run through ``optimizer.lr_range_run`` is held to that range, to
the freeze state probed and to the ratio ramped under.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402
from torch import nn  # noqa: E402

from tests._training_values import sgd_optimizer, stage  # noqa: E402
from tests.tiny_trainer_fixtures import (  # noqa: E402
    BACKBONE_REGRESSOR,
    QUADRATIC_BOWL,
    TRAIN_INTENSITIES,
    WHOLLY_FROZEN_REGRESSOR,
    frames_data,
    regressor_config,
)

if TYPE_CHECKING:
    from pydantic import JsonValue

    from tcip_mcp.workspace import BoundProject

RAMP = {"ramp_low": 1e-3, "ramp_high": 10.0, "batches": 100}
"""A sample ramp: a thousandth to ten over a hundred steps."""

CURVATURE = 1.0
"""The sample bowl's curvature, whose gradient descent stops converging at a rate of 2."""


def _grid_step(ramp_low: float, ramp_high: float, batches: int) -> float:
    """Return the factor a ramp from ``ramp_low`` to ``ramp_high`` over ``batches`` steps by."""
    return (ramp_high / ramp_low) ** (1 / (batches - 1))


def _derived(bound: BoundProject, config: dict, **ramp: object) -> JsonValue:
    """Return ``derive_lr_range``'s answer for stage 0 of ``config``, called as a client calls it.

    ``ramp`` states the ramp's parameters; those it leaves out take the tool's defaults.
    """
    from tests._server_calls import called_through_the_server

    return called_through_the_server(
        "derive_lr_range", {"config": config, "stage": 0, **ramp}, bound=bound
    )


def _completed(answer: JsonValue) -> dict:
    """Return ``answer``, asserting it is a range test's completed answer."""
    assert isinstance(answer, dict) and "error" not in answer, answer
    return answer


def _frames(project: Path) -> dict:
    """Return the data block of the sample training frames under ``project``, valued 2x."""
    return frames_data(
        project / "ds", TRAIN_INTENSITIES, [2.0 * c for c in TRAIN_INTENSITIES]
    )


def _bowl_config(project: Path, **overrides: object) -> dict:
    """Return a config of the bowl at :data:`CURVATURE` from weight 1 under plain SGD.

    ``overrides`` replace their keys.
    """
    return regressor_config(
        1,
        data=_frames(project),
        project=project,
        builder=QUADRATIC_BOWL,
        builder_kwargs={"init_weight": 1.0, "curvature": CURVATURE},
        optimizer=sgd_optimizer(backbone_lr=1e-2, head_lr=1e-2, momentum=0.0),
        **overrides,
    )


def _backbone_config(project: Path, **overrides: object) -> dict:
    """Return a backbone regressor's config under plain SGD, its backbone at a tenth of its head.

    Every frame is one batch; ``overrides`` replace their keys.
    """
    return regressor_config(
        1,
        data=_frames(project),
        project=project,
        builder=BACKBONE_REGRESSOR,
        optimizer=sgd_optimizer(backbone_lr=1e-3, head_lr=1e-2, momentum=0.0),
        batch_size=len(TRAIN_INTENSITIES),
        **overrides,
    )


def _ranged(bound: BoundProject, project: Path) -> tuple[dict, dict]:
    """Return a backbone regressor's config and the range test of it ``derive_lr_range`` ran.

    The ramp runs from a thousandth to a thousand over a hundred steps, past where the
    regressor's descent diverges.
    """
    config = _backbone_config(project)
    return config, _completed(
        _derived(bound, config, ramp_low=1e-3, ramp_high=1e3, batches=100)
    )


def _naming(
    config: dict, answer: dict, head_lr: float, **stated: object
) -> dict:
    """Return ``config`` naming ``answer``'s range test at ``head_lr`` and the ramped ratio.

    ``stated`` replace their keys.
    """
    return {
        **config,
        "optimizer": {
            **config["optimizer"],
            "lr_range_run": answer["experiment_id"],
            "head_lr": head_lr,
            "backbone_lr": head_lr * answer["result"]["ratio"],
        },
        **stated,
    }


def _refusal(answer: JsonValue, project: Path) -> str:
    """Return the error of a range test that ended with no reading, its final status the same.

    The run it names ended ``failed``, its final status naming the error.
    """
    from tcip_mcp import experiments

    assert isinstance(answer, dict) and "experiment_id" in answer, answer
    observation = experiments.observe_run(
        experiments.experiment_dir(answer["experiment_id"], project=project)
    )
    assert observation.state == "failed", observation.final
    assert observation.status_error is not None
    assert observation.status_error in answer["error"]
    return observation.status_error


def test_a_range_test_over_a_bowl_turns_where_gradient_descent_stops_converging(
    bound: BoundProject, project: Path
) -> None:
    """The rate of least loss over a bowl is the first ramp rate at or past ``2 / curvature``.

    Plain gradient descent scales the bowl's loss by ``(1 - curvature * r) ** 2`` a step, so the
    unsmoothed loss is least at the first step whose rate reaches ``2 / curvature``: the derived
    ``high`` lies within one step of the ramp's grid above it.
    """
    answer = _completed(
        _derived(bound, _bowl_config(project), smoothing=1.0, **RAMP)
    )

    turn = 2 / CURVATURE
    assert turn <= answer["high"] < turn * _grid_step(**RAMP)
    assert answer["high"] == answer["result"]["min_loss_rate"]
    assert answer["low"] == min(
        answer["high"] / 10, answer["result"]["steepest_rate"]
    )
    assert answer["result"]["stop"] is not None, answer


class _Inputs(nn.Module):
    """A model torch-lr-finder calls on a batch's images, which it hands its criterion as is.

    It holds the platform's model, so the finder's optimizer, mode switches and state cache reach
    the platform model's own parameters.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images


def test_the_steepest_descent_agrees_with_torch_lr_finders_suggestion(
    bound: BoundProject, project: Path
) -> None:
    """The range test's rate of steepest descent is torch-lr-finder's suggestion, within a step.

    Over the same regressor, loader, optimizer and ramp, ``LRFinder.range_test`` and its
    steepest-gradient suggestion (``plot``, nothing trimmed) land within one step of the ramp's
    grid of the rate the platform's reading records.
    """
    from tcip_mcp import experiments
    from tcip_mcp.pipelines.training.optimizer_factory import build_optimizer
    from tcip_mcp.pipelines.training.subprocess_worker import (
        prepare_run_context,
    )

    ramp = {
        "ramp_low": 1e-4,
        "ramp_high": 100.0,
        "batches": 60,
        "smoothing": 0.05,
        "divergence": 5.0,
    }
    config = regressor_config(
        1,
        data=_frames(project),
        project=project,
        optimizer=sgd_optimizer(backbone_lr=0.05, head_lr=0.05, momentum=0.0),
        batch_size=len(TRAIN_INTENSITIES),
    )
    answer = _completed(_derived(bound, config, **ramp))

    from matplotlib.figure import Figure
    from torch_lr_finder import LRFinder

    ctx = prepare_run_context(
        experiments.observe_run(
            experiments.experiment_dir(
                answer["experiment_id"], project=project
            )
        )
    )
    model = ctx.build_model()
    block = ctx.spec.optimizer
    assert block is not None
    finder = LRFinder(
        _Inputs(model),
        build_optimizer(
            block, model, backbone_lr=block.backbone_lr, head_lr=block.head_lr
        ),
        lambda images, targets: sum(model(images, targets).values()),
        device="cpu",
    )
    finder.range_test(
        ctx.train_loader,
        start_lr=ramp["ramp_low"],
        end_lr=ramp["ramp_high"],
        num_iter=ramp["batches"],
        step_mode="exp",
        smooth_f=ramp["smoothing"],
        diverge_th=ramp["divergence"],
    )
    _, suggested = finder.plot(
        skip_start=0, skip_end=0, ax=Figure().subplots()
    )

    step = _grid_step(ramp["ramp_low"], ramp["ramp_high"], ramp["batches"])
    assert (
        suggested / step
        <= answer["result"]["steepest_rate"]
        <= suggested * step
    ), (suggested, answer)


def test_a_range_test_of_the_chain_detector_is_its_own_listed_run(
    bound: BoundProject, project: Path
) -> None:
    """A range test of the chain fixtures' detector at stage 0 is a run of its own kind.

    Its record is launched as a range test at the ramp it ran, each of its per-batch rows carries
    its loss, smoothed loss and head rate, its ``lr_range.json`` reads back as the answer's
    result, ``list_experiments`` lists it as a range test, and ``get_experiment`` returns its rows
    and its result.
    """
    from tcip_mcp import experiments
    from tcip_mcp.tools.experiment_tools import (
        get_experiment,
        list_experiments,
    )
    from tests._chain_fixtures import (
        DETECTION_SOURCE,
        draw_reference_selection,
        run_config,
        synthetic_capture,
    )

    root = project / "ds"
    synthetic_capture(root)
    selection_dir = project / "selection"
    draw_reference_selection(project, root, selection_dir)
    ramp = {"ramp_low": 1e-5, "ramp_high": 10.0, "batches": 20}
    answer = _completed(
        _derived(
            bound, run_config(project, selection_dir, DETECTION_SOURCE), **ramp
        )
    )

    from tcip_mcp.pipelines.training.lr_range import range_result

    run_dir = experiments.experiment_dir(
        answer["experiment_id"], project=project
    )
    observation = experiments.observe_run(run_dir)
    assert observation.record.launched_as == experiments.RangeTest(
        stage=0, divergence=4.0, smoothing=0.05, **ramp
    )
    rows = experiments.partition_rows(
        experiments.read_rows(observation.metrics_log)[0]
    )[1]
    assert rows
    for row in rows:
        assert {experiments.BATCH_LOSS_KEY, "smoothed_loss", "lr"} <= set(
            row.model_dump()
        )
    assert (
        range_result(answer["experiment_id"], project).model_dump(mode="json")
        == answer["result"]
    )

    listed = list_experiments(project)["runs"]
    assert [(row["experiment_id"], row["kind"]) for row in listed] == [
        (answer["experiment_id"], "range_test")
    ]
    detail = get_experiment(project, answer["experiment_id"])["lr_range"]
    assert detail["result"] == answer["result"]
    assert (detail["low"], detail["high"]) == (answer["low"], answer["high"])
    assert len(detail["rows"]) == len(rows)


def test_a_launch_at_the_ranges_high_end_is_refused_naming_the_rate_range_and_run(
    bound: BoundProject, project: Path
) -> None:
    """A launch naming a range test at the range's high end is refused, naming all three.

    The range is half-open, so a constant rate at ``high`` itself is refused; the refusal names
    the stated rate, both ends of the range and the run that derived it.
    """
    from tcip_mcp.tools.training_tools import launch_training

    config, answer = _ranged(bound, project)
    refused = launch_training(
        project, _naming(config, answer, answer["high"]), actor=None
    )

    assert isinstance(refused, dict), refused
    (issue,) = [i for i in refused["issues"] if "head_lr" in i]
    for named in (
        f"{answer['high']:g}",
        f"{answer['low']:g}",
        answer["experiment_id"],
    ):
        assert named in issue, issue


def test_a_preflight_below_the_ranges_low_end_is_refused_naming_the_rate_range_and_run(
    bound: BoundProject, project: Path
) -> None:
    """A preflight naming a range test at half the range's low end is refused, naming all three.

    The refusal names the stated rate, both ends of the range and the run that derived it.
    """
    from tcip_mcp.tools.training_tools import preflight_config

    config, answer = _ranged(bound, project)
    below = answer["low"] / 2
    report = preflight_config(project, _naming(config, answer, below))

    assert report["valid"] is False
    (issue,) = [i for i in report["issues"] if "head_lr" in i]
    for named in (
        f"{below:g}",
        f"{answer['high']:g}",
        f"{answer['low']:g}",
        answer["experiment_id"],
    ):
        assert named in issue, issue


def test_a_preflight_whose_stage_freezes_otherwise_than_the_probe_is_refused(
    bound: BoundProject, project: Path
) -> None:
    """A preflight naming a range test probed at ``freeze_to`` 0 refuses a stage freezing 1.

    The range governs the freeze state it was probed at alone; the refusal names the stage, the
    probed state and the run.
    """
    from tcip_mcp.tools.training_tools import preflight_config

    config, answer = _ranged(bound, project)
    report = preflight_config(
        project,
        _naming(config, answer, answer["low"], stages=[stage(0), stage(1)]),
    )

    assert report["valid"] is False
    (issue,) = [i for i in report["issues"] if "stages[1]" in i]
    assert "freeze_to=0" in issue and answer["experiment_id"] in issue, issue


def test_a_sweep_point_at_a_ratio_unlike_the_ramped_one_is_refused(
    bound: BoundProject, project: Path
) -> None:
    """A sweep whose point states its own backbone rate off the ramped ratio is refused.

    The point holds ``backbone_lr`` at the head rate it samples, a ratio of one against the
    tenth the range test ramped under; the refusal names the ratio and the run.
    """
    from tcip_mcp.tools.training_tools import run_hyperparameter_search

    config, answer = _ranged(bound, project)
    refused = run_hyperparameter_search(
        project,
        _naming(config, answer, answer["low"]),
        {
            "optimizer.backbone_lr": {
                "type": "categorical",
                "choices": [answer["low"]],
            }
        },
        auto_tensorboard=False,
        n_trials=1,
        search_alg="random",
        scheduler={"name": "fifo"},
        search_seed=0,
        actor=None,
    )

    assert "error" in refused, refused
    (issue,) = [i for i in refused["issues"] if "backbone_lr / head_lr" in i]
    assert answer["experiment_id"] in issue, issue


def test_a_sweeps_own_head_rate_axis_beside_a_range_test_is_refused(
    bound: BoundProject, project: Path
) -> None:
    """A sweep stating its own ``optimizer.head_lr`` axis beside a range test is refused.

    The range test's range is the axis, so a second one is refused naming both before anything
    is opened.
    """
    from tcip_mcp import experiments
    from tcip_mcp.tools.training_tools import run_hyperparameter_search

    config, answer = _ranged(bound, project)
    refused = run_hyperparameter_search(
        project,
        _naming(config, answer, answer["low"]),
        {
            "optimizer.head_lr": {
                "type": "loguniform",
                "low": answer["low"],
                "high": answer["high"],
            }
        },
        auto_tensorboard=False,
        n_trials=1,
        search_alg="random",
        scheduler={"name": "fifo"},
        search_seed=0,
        actor=None,
    )

    assert "optimizer.head_lr" in refused["error"], refused
    assert answer["experiment_id"] in refused["error"], refused
    assert experiments.sweep_dirs(project) == []


def test_a_launch_inside_the_range_trains_to_completion(
    bound: BoundProject, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A launch naming a range test at a rate inside its range and the ramped ratio completes."""
    from tcip_mcp.tools.training_tools import RunLaunched, launch_training
    from tests._producer_fixtures import no_tensorboard
    from tests._verified_checkpoint_fixtures import run_to_end

    no_tensorboard(monkeypatch)
    config, answer = _ranged(bound, project)
    launched = launch_training(
        project, _naming(config, answer, answer["low"]), actor=None
    )

    assert isinstance(launched, RunLaunched), launched
    assert run_to_end(project, launched.experiment_id)["state"] == "completed"


def test_a_sweep_naming_a_range_test_samples_its_range_at_its_ratio_and_relaunches(
    bound: BoundProject, project: Path
) -> None:
    """A sweep naming a range test samples the derived axis, each trial at the ramped ratio.

    The sweep records the range as its ``optimizer.head_lr`` axis, every trial's head rate lies
    in ``[low, high)`` with its backbone rate at the ratio the test ramped under, and
    ``launch_sweep`` relaunches it from its record to completion.
    """
    from tcip_mcp import experiments
    from tcip_mcp.tools.training_tools import (
        SweepLaunched,
        launch_sweep,
        monitor_training,
        run_hyperparameter_search,
    )

    config, answer = _ranged(bound, project)
    swept = run_hyperparameter_search(
        project,
        _naming(config, answer, answer["low"]),
        {},
        auto_tensorboard=False,
        n_trials=2,
        search_alg="random",
        scheduler={"name": "fifo"},
        search_seed=0,
        actor=None,
    )

    sweep = swept["sweep"]
    assert sweep["state"] == "completed", swept
    assert sweep["input"]["param_space"] == {
        "optimizer.head_lr": {
            "type": "loguniform",
            "low": answer["low"],
            "high": answer["high"],
        }
    }
    assert len(sweep["trials"]) == 2
    for trial in sweep["trials"]:
        assert trial["state"] == "completed", trial
        block = experiments.read_run(Path(trial["output_dir"])).spec.optimizer
        assert block is not None
        assert answer["low"] <= block.head_lr < answer["high"]
        assert block.backbone_lr == pytest.approx(
            block.head_lr * answer["result"]["ratio"]
        )

    relaunched = launch_sweep(
        project,
        experiments.experiment_dir(sweep["sweep_id"], project=project),
        actor=None,
    )
    assert isinstance(relaunched, SweepLaunched), relaunched
    deadline = time.monotonic() + 300
    state = "running"
    while state == "running" and time.monotonic() < deadline:
        time.sleep(1)
        state = monitor_training(project, relaunched.sweep_id)["sweep"][
            "state"
        ]
    assert state == "completed", state


@pytest.mark.parametrize(
    ("ramp", "named"),
    [
        (
            {"ramp_low": 1e3, "ramp_high": 1e4, "batches": 10},
            "1 finite smoothed row",
        ),
        (
            {
                "ramp_low": 3.0,
                "ramp_high": 10.0,
                "batches": 10,
                "divergence": 1e30,
            },
            "least at the first row",
        ),
        (
            {"ramp_low": 1e-3, "ramp_high": 0.5, "batches": 20},
            "least at the last row",
        ),
    ],
    ids=["one-row", "no-descent", "no-turning-point"],
)
def test_a_curve_the_reading_refuses_ends_the_run_failed_naming_why(
    bound: BoundProject, project: Path, ramp: dict, named: str
) -> None:
    """A range test whose curve gives no reading ends failed, its final status naming why.

    Over the bowl, a ramp far past ``2 / curvature`` diverges at its second step and logs one
    row; a ramp wholly past it, held from stopping, rises from its first row; and a ramp wholly
    below it ends still falling, with no turning point.
    """
    status_error = _refusal(
        _derived(bound, _bowl_config(project), smoothing=1.0, **ramp), project
    )
    assert named in status_error, status_error


def test_a_config_naming_its_own_training_loop_is_refused(
    bound: BoundProject, project: Path
) -> None:
    """A range test of a config naming a ``training_source`` is refused naming it, nothing run."""
    from tcip_mcp import experiments
    from tests._chain_fixtures import BESPOKE_MODELS, SAVE_BUILT_WEIGHTS

    config = _bowl_config(project, training_source=SAVE_BUILT_WEIGHTS)
    config["model_source"]["source_files"].append(BESPOKE_MODELS)
    answer = _derived(bound, config, **RAMP)

    assert isinstance(answer, dict) and "training_source" in answer["error"]
    assert experiments.run_dirs(project) == []


def test_a_stage_that_trains_nothing_ends_the_run_failed_naming_it(
    bound: BoundProject, project: Path
) -> None:
    """A range test of a stage whose freeze leaves no parameter training ends failed, naming it."""
    config = regressor_config(
        1,
        data=_frames(project),
        project=project,
        builder=WHOLLY_FROZEN_REGRESSOR,
        stages=[stage(1)],
    )
    status_error = _refusal(_derived(bound, config, **RAMP), project)
    assert "stage 0 (freeze_to=1) trains no parameter" in status_error
