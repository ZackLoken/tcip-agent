"""The classification, agreement and regression metrics agree with scikit-learn's own figures.

Each test trains a model through the launcher's own producer, runs the checkpoint it completed in
eval mode over the loader of the data it trained on, and hands the platform's metric and
scikit-learn's the same predictions and targets. The two are compared after the platform's own
six-place rounding. An undefined score is the platform's ``None`` where scikit-learn answers nan
(``zero_division=np.nan``): a class no label and no prediction carries is ``None`` on every row
and out of the macro mean, and so is an undefined agreement or R-squared.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

torch = pytest.importorskip("torch")
metrics = pytest.importorskip("sklearn.metrics")

from tcip_mcp.pipelines.training.evaluation import (  # noqa: E402
    classification_metrics,
    ordinal_metrics,
    quadratic_weighted_kappa,
    r_squared,
    regression_metrics,
)
from tests._chain_fixtures import (  # noqa: E402
    BESPOKE_CLASSIFIER,
    BESPOKE_ORDINAL,
    BESPOKE_REGRESSOR,
)
from tests._training_values import adamw_optimizer  # noqa: E402
from tests._verified_checkpoint_fixtures import (  # noqa: E402
    bespoke_source,
    scored_tensors,
    trained_run,
)
from tests.tiny_trainer_fixtures import write_regression_dataset  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

    from tcip_mcp.pipelines.schemas import ModelDims

FRAMES = 24
GRADES = (0, 2, 3)
"""The labels the classification and ordinal frames take, one band of frames each in order of
brightness. The run's class and rank count is the largest plus one, so grade 1 is a class no
frame carries."""
LEARNING = adamw_optimizer(backbone_lr=1e-2, head_lr=1e-1)
"""A rate high enough that one epoch over these frames moves the head off its initial output."""


def _scored(
    tmp_path: Path,
    builder: str,
    task: str,
    values: list[float],
    **overrides: object,
) -> tuple[Tensor, Tensor, ModelDims]:
    """Return a trained ``task`` run's predictions, targets and dims over ``FRAMES`` frames.

    The frames brighten in order and carry ``values``; ``overrides`` are the run config's own keys
    over :func:`trained_run`'s.
    """
    images_dir, table = write_regression_dataset(
        tmp_path,
        [i / FRAMES for i in range(FRAMES)],
        values,
        height=48,
        width=64,
    )
    observation = trained_run(
        bespoke_source(builder, task),
        {"images_dir": str(images_dir), "labels_dir": str(table)},
        tmp_path,
        batch_size=2,
        **overrides,
    )
    return scored_tensors(tmp_path, observation)


def _grades() -> list[float]:
    return [GRADES[i * len(GRADES) // FRAMES] for i in range(FRAMES)]


def _six(value: float | None) -> float | None:
    """Return ``value`` to six places, ``None`` for ``None`` and for the library's nan."""
    return None if value is None or math.isnan(value) else round(value, 6)


def _classified(tmp_path: Path) -> tuple[Tensor, Tensor, int]:
    """Return a trained classifier's predictions, targets and one class count past its own.

    The class past the run's own is in no label and no prediction, so it is absent from both
    sides whatever the run learned; the run's grade 1 is absent from the labels alone.
    """
    pred, gt, dims = _scored(
        tmp_path,
        BESPOKE_CLASSIFIER,
        "classification",
        _grades(),
        optimizer=LEARNING,
    )
    return pred, gt, dims["num_classes"] + 1


def _ranked(tmp_path: Path) -> tuple[Tensor, Tensor, int]:
    pred, gt, dims = _scored(
        tmp_path, BESPOKE_ORDINAL, "ordinal", _grades(), optimizer=LEARNING
    )
    return pred, gt, dims["num_ranks"]


@pytest.mark.parametrize("statistic", ["precision", "recall", "f1"])
def test_per_class_precision_recall_and_f1_agree_with_scikit_learn(
    tmp_path: Path, statistic: str
) -> None:
    pred, gt, num_classes = _classified(tmp_path)
    ours = classification_metrics(pred, gt, num_classes)["per_class"]

    precision, recall, f1, support = metrics.precision_recall_fscore_support(
        gt.numpy(),
        pred.numpy(),
        labels=range(num_classes),
        zero_division=np.nan,
    )
    expected = {"precision": precision, "recall": recall, "f1": f1}[statistic]
    assert [_six(ours[str(c)][statistic]) for c in range(num_classes)] == [
        _six(float(v)) for v in expected
    ]
    assert [ours[str(c)]["support"] for c in range(num_classes)] == (
        support.tolist()
    )
    assert ours[str(num_classes - 1)][statistic] is None


def test_accuracy_agrees_with_scikit_learn(tmp_path: Path) -> None:
    pred, gt, num_classes = _classified(tmp_path)
    ours = classification_metrics(pred, gt, num_classes)["accuracy"]
    assert _six(ours) == _six(
        float(metrics.accuracy_score(gt.numpy(), pred.numpy()))
    )


def test_macro_f1_agrees_with_scikit_learn_over_the_defined_classes(
    tmp_path: Path,
) -> None:
    pred, gt, num_classes = _classified(tmp_path)
    ours = classification_metrics(pred, gt, num_classes)["f1"]
    assert _six(ours) == _six(
        float(
            metrics.f1_score(
                gt.numpy(),
                pred.numpy(),
                labels=range(num_classes),
                average="macro",
                zero_division=np.nan,
            )
        )
    )


def test_rank_accuracy_agrees_with_scikit_learn(tmp_path: Path) -> None:
    pred, gt, num_ranks = _ranked(tmp_path)
    ours = ordinal_metrics(pred, gt, num_ranks)["rank_acc"]
    assert _six(ours) == _six(
        float(
            metrics.accuracy_score(
                gt.round().long().numpy(), pred.round().long().numpy()
            )
        )
    )


def test_quadratic_weighted_kappa_agrees_with_scikit_learn(
    tmp_path: Path,
) -> None:
    pred, gt, num_ranks = _ranked(tmp_path)
    ours = quadratic_weighted_kappa(pred, gt, num_ranks)
    assert ours is not None
    assert _six(ours) == _six(
        float(
            metrics.cohen_kappa_score(
                gt.round().long().numpy(),
                pred.round().long().numpy(),
                weights="quadratic",
                labels=range(num_ranks),
            )
        )
    )


def test_a_rank_outside_the_run_rank_count_refuses_naming_it(
    tmp_path: Path,
) -> None:
    pred, gt, num_ranks = _ranked(tmp_path)
    assert quadratic_weighted_kappa(pred, gt, num_ranks) is not None

    beyond = pred.clone()
    beyond[0] = num_ranks + 0.4
    below = gt.clone()
    below[1] = -1
    with pytest.raises(
        ValueError, match=rf"\[{num_ranks}\].*{num_ranks} ranks"
    ):
        quadratic_weighted_kappa(beyond, gt, num_ranks)
    with pytest.raises(ValueError, match=r"\[-1\].*ranks"):
        quadratic_weighted_kappa(pred, below, num_ranks)


def test_ordinal_mae_agrees_with_scikit_learn(tmp_path: Path) -> None:
    pred, gt, num_ranks = _ranked(tmp_path)
    ours = ordinal_metrics(pred, gt, num_ranks)["mae"]
    assert _six(ours) == _six(
        float(metrics.mean_absolute_error(gt.numpy(), pred.numpy()))
    )


def _regressed(tmp_path: Path) -> tuple[Tensor, Tensor]:
    pred, gt, _dims = _scored(
        tmp_path,
        BESPOKE_REGRESSOR,
        "regression",
        [round(0.1 + 0.8 * i / FRAMES, 3) for i in range(FRAMES)],
    )
    return pred, gt


def test_regression_mae_agrees_with_scikit_learn(tmp_path: Path) -> None:
    pred, gt = _regressed(tmp_path)
    ours = regression_metrics(pred, gt)["mae"]
    assert _six(ours) == _six(
        float(metrics.mean_absolute_error(gt.numpy(), pred.numpy()))
    )


def test_regression_rmse_agrees_with_scikit_learn(tmp_path: Path) -> None:
    pred, gt = _regressed(tmp_path)
    ours = regression_metrics(pred, gt)["rmse"]
    assert _six(ours) == _six(
        float(metrics.root_mean_squared_error(gt.numpy(), pred.numpy()))
    )


def test_r_squared_agrees_with_scikit_learn(tmp_path: Path) -> None:
    pred, gt = _regressed(tmp_path)
    ours = r_squared(pred, gt)
    assert ours is not None
    assert _six(ours) == _six(
        float(metrics.r2_score(gt.numpy(), pred.numpy()))
    )
