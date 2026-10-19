"""Tiny deterministic models, the frames written for them, and run configs over those frames.

The module holds the models, their ``model_source`` builders, the frame writer, and the run
configs and contexts over the frames. Each model here initializes its parameters from
constants, and each builder accepts ``in_chans`` and, for the classifier, ``num_classes``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.nn import functional

from tests import REPO_ROOT
from tests._training_values import (
    adamw_optimizer,
    drawn_split,
    evaluation_block,
    schedule,
    stage,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    import pytest
    from pydantic import BaseModel

    from tcip_mcp.pipelines.model_build import SourceLayout
    from tcip_mcp.pipelines.schemas import ModelSourceSchema, TrainGeometry
    from tcip_mcp.pipelines.training.envelope import TrainContext
    from tcip_mcp.pipelines.training.generic_trainer import EpochCallback


def _phantom_nan() -> torch.Tensor:
    """Return a non-finite loss on a leaf disconnected from any weight's graph."""
    return torch.zeros((), requires_grad=True) + float("nan")


class MeanIntensityRegressor(nn.Module):
    """A regressor predicting ``weight * mean(image)``, with a squared-error loss in train mode.

    It holds one parameter.

    With ``cancel_at_call`` and ``cancel_output_dir`` given, the training forward with that
    one-based call count requests the run's own cancellation (``experiments.request_cancel``)
    at that output directory.
    """

    def __init__(
        self,
        init_weight: float = 0.0,
        cancel_at_call: int | None = None,
        cancel_output_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([float(init_weight)]))
        self.cancel_at_call = cancel_at_call
        self.cancel_output_dir = cancel_output_dir
        self._forwards = 0

    def _count_training_forward(self) -> None:
        """Count one training forward, requesting the run's cancellation at ``cancel_at_call``."""
        self._forwards += 1
        if (
            self._forwards == self.cancel_at_call
            and self.cancel_output_dir is not None
        ):
            from pathlib import Path

            from tcip_mcp.experiments import request_cancel

            path = Path(self.cancel_output_dir)
            path.mkdir(parents=True, exist_ok=True)
            request_cancel(path)

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        return self.weight * images.mean(dim=(1, 2, 3))

    def fit_loss(self, images: torch.Tensor, targets: dict) -> dict:
        return {
            "mse": (
                (self.predict(images) - targets["values"].float()) ** 2
            ).mean()
        }

    def forward(
        self, images: torch.Tensor, targets: dict | None = None
    ) -> dict:
        if self.training and targets is not None:
            self._count_training_forward()
            return self.fit_loss(images, targets)
        return {"head0_values": self.predict(images)}


def build_mean_intensity_regressor(
    *,
    in_chans: int = 1,
    init_weight: float = 0.0,
    cancel_at_call: int | None = None,
    cancel_output_dir: str | None = None,
    **_statistics: object,
) -> MeanIntensityRegressor:
    """Build a :class:`MeanIntensityRegressor` as a ``model_source`` builder.

    The weight and cancellation arguments pass through; the statistics it is handed are unused.
    """
    return MeanIntensityRegressor(
        init_weight=init_weight,
        cancel_at_call=cancel_at_call,
        cancel_output_dir=cancel_output_dir,
    )


class CountingRegressor(MeanIntensityRegressor):
    """A :class:`MeanIntensityRegressor` with a forward-count buffer and a tensor group setting.

    Its state also holds a registered buffer, ``forwards``, counting its loss calls, and its
    one param group (``get_param_groups``, at the head rate) carries a tensor-valued setting,
    ``rates``, holding the backbone and head rates it was handed.
    """

    def __init__(self, init_weight: float = 0.0) -> None:
        super().__init__(init_weight)
        self.register_buffer("forwards", torch.zeros(()))

    def fit_loss(self, images: torch.Tensor, targets: dict) -> dict:
        self.forwards += 1
        return super().fit_loss(images, targets)

    def get_param_groups(
        self, backbone_lr: float, head_lr: float
    ) -> list[dict]:
        return [
            {
                "params": [self.weight],
                "lr": head_lr,
                "rates": torch.tensor([backbone_lr, head_lr]),
            }
        ]


def build_counting_regressor(
    *, in_chans: int = 1, init_weight: float = 0.0, **_statistics: object
) -> CountingRegressor:
    """Build a :class:`CountingRegressor` as a ``model_source`` builder."""
    return CountingRegressor(init_weight=init_weight)


class TwoRateRegressor(MeanIntensityRegressor):
    """A regressor predicting ``weight * mean(image) + bias`` with one param group per rate.

    Its weight sits in a param group at the backbone rate and its bias in one at the head rate
    (``get_param_groups``), the two groups in reverse order when ``reverse_groups``.
    """

    def __init__(self, reverse_groups: bool = False) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.reverse_groups = reverse_groups

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        return self.weight * images.mean(dim=(1, 2, 3)) + self.bias

    def get_param_groups(
        self, backbone_lr: float, head_lr: float
    ) -> list[dict]:
        groups = [
            {"params": [self.weight], "lr": backbone_lr},
            {"params": [self.bias], "lr": head_lr},
        ]
        return groups[::-1] if self.reverse_groups else groups


def build_two_rate_regressor(
    *, in_chans: int = 1, reverse_groups: bool = False, **_statistics: object
) -> TwoRateRegressor:
    """Build a :class:`TwoRateRegressor` as a ``model_source`` builder."""
    return TwoRateRegressor(reverse_groups=reverse_groups)


class NanEvalRegressor(MeanIntensityRegressor):
    """A regressor answering its parent's loss in train mode and ``predict + nan`` in eval."""

    def forward(
        self, images: torch.Tensor, targets: dict | None = None
    ) -> dict:
        if self.training and targets is not None:
            return self.fit_loss(images, targets)
        return {"head0_values": self.predict(images) + float("nan")}


def build_nan_eval_regressor(
    *, in_chans: int = 1, init_weight: float = 0.0, **_statistics: object
) -> NanEvalRegressor:
    """Build a :class:`NanEvalRegressor` as a ``model_source`` builder."""
    return NanEvalRegressor(init_weight=init_weight)


class MeanIntensityClassifier(nn.Module):
    """A two-class classifier with logit ``[0, weight * (mean(image) - 0.5)]``.

    Its loss is cross-entropy in train mode, and in eval it answers the argmax label. It holds
    one parameter.
    """

    num_classes = 2

    def __init__(self, init_weight: float = 0.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([float(init_weight)]))
        self.heads = [self]  # evaluate() reads num_classes off model.heads[0]

    def forward(
        self, images: torch.Tensor, targets: dict | None = None
    ) -> dict:
        logit1 = self.weight * (images.mean(dim=(1, 2, 3)) - 0.5)
        logits = torch.stack([torch.zeros_like(logit1), logit1], dim=1)
        if self.training and targets is not None:
            return {"ce": functional.cross_entropy(logits, targets["labels"])}
        return {"head0_labels": logits.argmax(dim=1)}


def build_mean_intensity_classifier(
    *,
    in_chans: int = 1,
    num_classes: int = 2,
    init_weight: float = 0.0,
    **_statistics: object,
) -> MeanIntensityClassifier:
    """Build a :class:`MeanIntensityClassifier` as a ``model_source`` builder.

    ``init_weight`` passes through; the statistics it is handed are unused.
    """
    return MeanIntensityClassifier(init_weight=init_weight)


class DataScaledGradientModel(MeanIntensityRegressor):
    """A model whose loss is ``weight * sum(batch values)``."""

    def fit_loss(self, images: torch.Tensor, targets: dict) -> dict:
        return {
            "linear": (self.weight * targets["values"].float().sum()).squeeze()
        }


def build_data_scaled_gradient_model(
    *, in_chans: int = 1, **_statistics: object
) -> DataScaledGradientModel:
    """Build a :class:`DataScaledGradientModel` as a ``model_source`` builder."""
    return DataScaledGradientModel()


class AlwaysDivergedModel(MeanIntensityRegressor):
    """A :class:`MeanIntensityRegressor` whose ``fit_loss`` answers ``weight * nan``."""

    def fit_loss(self, images: torch.Tensor, targets: dict) -> dict:
        return {"nan_loss": self.weight * float("nan")}


def build_always_diverged_model(
    *,
    in_chans: int = 1,
    cancel_at_call: int | None = None,
    cancel_output_dir: str | None = None,
    **_statistics: object,
) -> AlwaysDivergedModel:
    """Build an :class:`AlwaysDivergedModel` as a ``model_source`` builder."""
    return AlwaysDivergedModel(
        cancel_at_call=cancel_at_call, cancel_output_dir=cancel_output_dir
    )


def run_context(
    root: str | Path,
    config: dict,
    *,
    epoch_hook: EpochCallback | None = None,
    experiment_id: str | None = None,
) -> TrainContext:
    """Return the context the child's own entry builds for a run opened over ``config``.

    The context is built by ``subprocess_worker.prepare_run_context`` for the run
    :func:`~tests._verified_checkpoint_fixtures.opened_run` opens under the project ``root``:
    ``run_registry.observed_run``'s run over that directory and the loaders its recorded
    partition builds, ``epoch_hook`` the context's own.
    """
    from tcip_mcp.experiments import observe
    from tcip_mcp.pipelines.training.subprocess_worker import (
        prepare_run_context,
    )
    from tests._verified_checkpoint_fixtures import opened_run

    return prepare_run_context(
        observe(opened_run(root, config, experiment_id=experiment_id)),
        epoch_hook=epoch_hook,
    )


def frames_data(
    root: str | Path, intensities: Sequence[float], values: Sequence[float]
) -> dict:
    """Return a run's data block over every frame written under ``root``.

    :func:`write_regression_dataset` writes the frames at ``intensities`` and valued
    ``values``; the block's split is ``drawn_split(val_ratio=0.0)``.
    """
    images_dir, csv_path = write_regression_dataset(root, intensities, values)
    return {
        "images_dir": str(images_dir),
        "labels_dir": str(csv_path),
        "split": drawn_split(val_ratio=0.0),
    }


def drawn_data(
    root: str | Path, train: tuple[list, list], val: tuple[list, list]
) -> dict:
    """Return the data block of a run drawing its own split over frames written under ``root``.

    :func:`write_regression_dataset` writes ``train``'s and then ``val``'s frames, each
    ``(intensities, values)``. The split's grouping map names each frame's member identity and
    gives it a group: ``train-first`` for ``train``'s first frame, ``train`` for its others and
    ``val`` for ``val``'s, at a val_ratio of ``val``'s share of the frames. Raises ValueError
    unless ``val`` holds more than one frame and fewer than ``train``'s count less one.
    """
    from tcip_mcp.dataset_layout import UNDATED_BUCKET

    n_train, n_val = len(train[0]), len(val[0])
    if not 1 < n_val < n_train - 1:
        raise ValueError(
            f"{n_val} validation frames beside {n_train} training frames: the draw "
            "places these groups only for more than one and fewer than "
            f"{n_train - 1}"
        )
    images_dir, csv_path = write_regression_dataset(
        root, [*train[0], *val[0]], [*train[1], *val[1]]
    )
    groups = ["train-first"] + ["train"] * (n_train - 1) + ["val"] * n_val
    return {
        "images_dir": str(images_dir),
        "labels_dir": str(csv_path),
        "split": drawn_split(
            val_ratio=n_val / (n_train + n_val),
            grouping={
                f"{UNDATED_BUCKET}/img{i}": group
                for i, group in enumerate(groups)
            },
        ),
    }


FIT, HOLDOUT_FIT = 2.0, -5.0
"""The weights the frames of :func:`opposed_regression_data` are fit by: training frames valued
at ``FIT`` times their intensity, validation frames at ``HOLDOUT_FIT``."""


def opposed_regression_data(
    root: str | Path,
    train_intensities: list[float],
    val_intensities: list[float],
) -> dict:
    """Return :func:`drawn_data` over training and validation frames written under ``root``.

    The training frames sit at ``train_intensities``, each valued :data:`FIT` times its
    intensity, and the validation frames at ``val_intensities``, each valued
    :data:`HOLDOUT_FIT` times its intensity.
    """
    return drawn_data(
        root,
        (train_intensities, [FIT * c for c in train_intensities]),
        (val_intensities, [HOLDOUT_FIT * c for c in val_intensities]),
    )


def distinct_intensities(n: int) -> list[float]:
    """Return ``(i + 1) / (n + 1)`` for each ``i`` in ``range(n)``."""
    return [(i + 1) / (n + 1) for i in range(n)]


TRAIN_INTENSITIES = [0.10, 0.25, 0.40, 0.55, 0.70, 0.85]
VAL_INTENSITIES = [0.15, 0.35, 0.60, 0.90]
"""Sample frame intensities of a regression run's training and validation sides, the two sets
apart so no validation frame repeats a training one."""

SEPARABLE_TRAIN = ([0.10, 0.20, 0.30, 0.70, 0.80, 0.90], [0, 0, 0, 1, 1, 1])
SEPARABLE_VAL = ([0.15, 0.40, 0.60, 0.85], [0, 0, 1, 1])
"""Classification frames labeled 0 below mid-gray and 1 above it, as ``(intensities, labels)``:
six training and four validation frames, which :class:`MeanIntensityClassifier` separates."""


def separable_classifier_data(root: str | Path) -> dict:
    """Return :func:`drawn_data` of :data:`SEPARABLE_TRAIN` and :data:`SEPARABLE_VAL`."""
    return drawn_data(root, SEPARABLE_TRAIN, SEPARABLE_VAL)


def count_validations(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Return a list each explicit config or model-source validation appends its class name to.

    From here on, each ``model_validate`` of a whole training config
    (``schemas.TrainConfigSchema``) or of a model source alone (``schemas.ModelSourceSchema``)
    appends to it.
    """
    from tcip_mcp.pipelines.schemas import ModelSourceSchema, TrainConfigSchema

    seen: list[str] = []
    for schema in (TrainConfigSchema, ModelSourceSchema):
        real = schema.model_validate.__func__

        def counted(
            cls: type[BaseModel],
            obj: object,
            *args: object,
            _real: Callable[..., BaseModel] = real,
            **kwargs: object,
        ) -> BaseModel:
            seen.append(cls.__name__)
            return _real(cls, obj, *args, **kwargs)

        monkeypatch.setattr(schema, "model_validate", classmethod(counted))
    return seen


MEAN_INTENSITY_REGRESSOR = (
    "tiny_trainer_fixtures:build_mean_intensity_regressor"
)
MEAN_INTENSITY_CLASSIFIER = (
    "tiny_trainer_fixtures:build_mean_intensity_classifier"
)
ALWAYS_DIVERGED_MODEL = "tiny_trainer_fixtures:build_always_diverged_model"
NAN_EVAL_REGRESSOR = "tiny_trainer_fixtures:build_nan_eval_regressor"
TWO_RATE_REGRESSOR = "tiny_trainer_fixtures:build_two_rate_regressor"
COUNTING_REGRESSOR = "tiny_trainer_fixtures:build_counting_regressor"
TINY_TRAINER_FILE = str(REPO_ROOT / "tests" / "tiny_trainer_fixtures.py")
"""The file of ``tests.tiny_trainer_fixtures`` in this repository, which a config naming one of
its builders declares under ``model_source.source_files``; never this module's own
``__file__``, which names a run's snapshot copy once a run has bound the module there."""

REGRESSOR_ADAMW = adamw_optimizer(backbone_lr=0.05, head_lr=0.05)
"""The AdamW section the one-weight regressors here train at."""


def regressor_config(
    horizon: int = 1,
    *,
    data: dict,
    project: Path,
    builder: str = MEAN_INTENSITY_REGRESSOR,
    builder_kwargs: dict | None = None,
    normalization: dict | None = None,
    **overrides: object,
) -> dict:
    """Return a training config of ``project`` for the regression ``builder`` over ``data``.

    The config is a :func:`~tests._chain_fixtures.training_config` of the regression
    ``builder`` (by default the mean-intensity regressor) at ``builder_kwargs`` over the data
    block ``data``: one unfrozen stage at :data:`REGRESSOR_ADAMW` under a cosine schedule of
    ``horizon`` epochs, selecting on the loss, no epoch checkpoints, with ``overrides`` in place
    of their keys. Its normalization is ``normalization``, or ``FRAME_STATISTICS`` when none is
    given.
    """
    from tests._chain_fixtures import FRAME_STATISTICS, training_config

    source: dict = {
        "builder": builder,
        "source_files": [TINY_TRAINER_FILE],
        "task": "regression",
        "normalization": normalization or FRAME_STATISTICS,
    }
    if builder_kwargs is not None:
        source["builder_kwargs"] = builder_kwargs
    return training_config(
        project,
        source,
        data,
        **{
            "stages": [stage(0)],
            "optimizer": REGRESSOR_ADAMW,
            "scheduler": schedule("cosine", horizon_epochs=horizon),
            "evaluation": evaluation_block(selection_metric="loss"),
            "checkpoint_every_n_epochs": 0,
            **overrides,
        },
    )


def classifier_config(
    horizon: int,
    data: dict,
    *,
    project: Path,
    task: str = "classification",
    normalization: dict | None = None,
    **overrides: object,
) -> dict:
    """Return a training config of ``project`` for the mean-intensity classifier over ``data``.

    The config is a :func:`~tests._chain_fixtures.training_config` of the mean-intensity
    classifier (weight -1) as a ``task`` model over the data block ``data``: one unfrozen stage
    in batches of three under a cosine schedule of ``horizon`` epochs, AdamW at 0.2, selecting
    on the loss, no epoch checkpoints, with ``overrides`` in place of their keys. Its
    normalization is ``normalization``, or ``FRAME_STATISTICS`` when none is given.
    """
    from tests._chain_fixtures import FRAME_STATISTICS, training_config

    return training_config(
        project,
        {
            "builder": MEAN_INTENSITY_CLASSIFIER,
            "builder_kwargs": {"init_weight": -1.0},
            "source_files": [TINY_TRAINER_FILE],
            "task": task,
            "normalization": normalization or FRAME_STATISTICS,
        },
        data,
        **{
            "batch_size": 3,
            "stages": [stage(0)],
            "scheduler": schedule("cosine", horizon_epochs=horizon),
            "optimizer": adamw_optimizer(backbone_lr=0.2, head_lr=0.2),
            "evaluation": evaluation_block(selection_metric="loss"),
            "checkpoint_every_n_epochs": 0,
            **overrides,
        },
    )


def capture_model(monkeypatch: pytest.MonkeyPatch, sink: list) -> None:
    """Append every model ``generic_trainer.build_from_model_source`` builds to ``sink``.

    Each model is appended as it is built.
    """
    from tcip_mcp.pipelines.training import generic_trainer as gt

    real_build = gt.build_from_model_source

    def build(
        source: ModelSourceSchema,
        plan: SourceLayout,
        dims: Mapping[str, object],
        geometry: TrainGeometry,
    ) -> object:
        model = real_build(source, plan, dims, geometry)
        sink.append(model)
        return model

    monkeypatch.setattr(gt, "build_from_model_source", build)


class _CallScheduledModel(MeanIntensityRegressor):
    """A :class:`MeanIntensityRegressor` whose ``fit_loss`` is gated by ``finite``.

    ``fit_loss`` answers the parent's weight-fit loss when ``finite`` admits the count of
    training forwards so far, and :func:`_phantom_nan` when it does not.
    """

    def finite(self, call: int) -> bool:
        raise NotImplementedError

    def fit_loss(self, images: torch.Tensor, targets: dict) -> dict:
        if self.finite(self._forwards):
            return super().fit_loss(images, targets)
        return {"nan_loss": _phantom_nan()}


class TransientlyDivergedModel(_CallScheduledModel):
    """A :class:`_CallScheduledModel` whose ``finite(call)`` is ``call > bad_batches``."""

    def __init__(self, bad_batches: int = 2, init_weight: float = 0.0) -> None:
        super().__init__(init_weight)
        self.bad_batches = int(bad_batches)

    def finite(self, call: int) -> bool:
        return call > self.bad_batches


def build_transiently_diverged_model(
    *,
    in_chans: int = 1,
    bad_batches: int = 2,
    init_weight: float = 0.0,
    **_statistics: object,
) -> TransientlyDivergedModel:
    """Build a :class:`TransientlyDivergedModel` as a ``model_source`` builder."""
    return TransientlyDivergedModel(
        bad_batches=bad_batches, init_weight=init_weight
    )


class StepCountedDivergenceModel(_CallScheduledModel):
    """A :class:`_CallScheduledModel` whose ``finite(call)`` is ``call in set(finite_at)``."""

    def __init__(
        self, finite_at: Iterable[int] = (), init_weight: float = 0.0
    ) -> None:
        super().__init__(init_weight)
        self.finite_at = set(finite_at)

    def finite(self, call: int) -> bool:
        return call in self.finite_at


def build_step_counted_divergence_model(
    *,
    in_chans: int = 1,
    finite_at: Iterable[int] = (),
    init_weight: float = 0.0,
    **_statistics: object,
) -> StepCountedDivergenceModel:
    """Build a :class:`StepCountedDivergenceModel` as a ``model_source`` builder."""
    return StepCountedDivergenceModel(
        finite_at=finite_at, init_weight=init_weight
    )


class DivergesAfterModel(_CallScheduledModel):
    """A :class:`_CallScheduledModel` whose ``finite(call)`` is ``call <= good_calls``."""

    def __init__(self, good_calls: int, init_weight: float = 0.0) -> None:
        super().__init__(init_weight)
        self.good_calls = int(good_calls)

    def finite(self, call: int) -> bool:
        return call <= self.good_calls


def build_diverges_after_model(
    *,
    good_calls: int,
    in_chans: int = 1,
    init_weight: float = 0.0,
    **_statistics: object,
) -> DivergesAfterModel:
    """Build a :class:`DivergesAfterModel` as a ``model_source`` builder."""
    return DivergesAfterModel(good_calls=good_calls, init_weight=init_weight)


class PixelSpreadDivideModel(MeanIntensityRegressor):
    """A regressor dividing its prediction by each sample's spread across its bands.

    The spread is the bands' standard deviation at each pixel, averaged over the pixels.
    """

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        return (self.weight + images.mean(dim=(1, 2, 3))) / images.std(
            dim=1
        ).mean(dim=(1, 2))


def build_pixel_spread_divide_model(
    *, in_chans: int = 1, **_statistics: object
) -> PixelSpreadDivideModel:
    """Build a :class:`PixelSpreadDivideModel` as a ``model_source`` builder."""
    return PixelSpreadDivideModel()


def write_regression_dataset(
    root: str | Path,
    intensities: Sequence[float],
    values: Sequence[float],
    *,
    height: int = 6,
    width: int = 10,
) -> tuple[Path, Path]:
    """Write a small on-disk table-labeled dataset of PNG frames and a CSV of values.

    The frames are uint8 RGB PNGs ``img<i>.png`` under ``root/images/<undated bucket>``, and
    ``root/values.csv`` holds ``stem,value`` rows, a regression's values or a classification's
    labels.

    Each frame is a ``height`` by ``width`` checkerboard, 6 by 10 by default, of ``low`` and
    ``high`` cast to uint8 and stacked into three bands, where ``px = round(255 * fraction)``
    and ``(low, high)`` is (0, 1) at 0, (254, 255) at 255 and (px - 1, px + 1) at other values.
    Returns ``(images_dir, csv_path)``.
    """
    import numpy as np
    from PIL import Image

    if len(intensities) != len(values):
        raise ValueError("intensities and values must be the same length")
    from tcip_mcp.dataset_layout import UNDATED_BUCKET

    images_dir = Path(root) / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(root) / "values.csv"
    rows = ["stem,value"]
    checker = np.indices((height, width)).sum(axis=0) % 2
    for i, (frac, value) in enumerate(zip(intensities, values)):
        stem = f"img{i}"
        px = round(255 * float(frac))
        low, high = {0: (0, 1), 255: (254, 255)}.get(px, (px - 1, px + 1))
        plane = np.where(checker == 1, high, low).astype(np.uint8)
        Image.fromarray(np.stack([plane] * 3, axis=-1)).save(
            images_dir / f"{stem}.png"
        )
        rows.append(f"{stem},{value}")
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    return images_dir, csv_path
