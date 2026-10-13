"""Tiny deterministic models, their ``model_source`` builders, the frames they train on, and the
runs those frames are trained in.

Every model here holds parameters initialized from constants (one, except
:class:`TwoRateRegressor`), so a run's trajectory is decided by the data it is fed and never by
random init. Each builder accepts the width
(``in_chans``) and, for the classifier, the class count a builder is handed; a model reads a
frame's mean intensity whatever its width. A frame holds its intensity to within half a level
(:func:`write_regression_dataset`), so the loss landscape a run presents is fixed by its frames'
(intensity, value) pairs alone.
"""

from __future__ import annotations

from pathlib import Path

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


def _phantom_nan() -> torch.Tensor:
    """A non-finite loss on a leaf disconnected from any weight's graph, so a bad call's optimizer
    step never poisons the weight a later good call relies on.
    """
    return torch.zeros((), requires_grad=True) + float("nan")


class MeanIntensityRegressor(nn.Module):
    """Predicts ``weight * mean(image)``, with a squared-error loss in train mode.

    One parameter, so which data a pass is run over is the only thing that can move its loss. A
    subclass changes only which training calls it answers with that loss.

    With ``cancel_at_call`` and ``cancel_output_dir`` given, the training forward with that
    one-based call count requests the run's own cancellation (``experiments.request_cancel``)
    at that output directory, so a test can place a cancel at an exact point in the batch
    stream; both are record values, so the builder's kwargs stay a config.
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

    def predict(self, images):
        return self.weight * images.mean(dim=(1, 2, 3))

    def fit_loss(self, images, targets) -> dict:
        return {
            "mse": (
                (self.predict(images) - targets["values"].float()) ** 2
            ).mean()
        }

    def forward(self, images, targets=None):
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
    **_statistics,
) -> MeanIntensityRegressor:
    """``model_source`` builder for :class:`MeanIntensityRegressor`, which reads raw
    intensities, not the statistics it is handed.
    """
    return MeanIntensityRegressor(
        init_weight=init_weight,
        cancel_at_call=cancel_at_call,
        cancel_output_dir=cancel_output_dir,
    )


class CountingRegressor(MeanIntensityRegressor):
    """A :class:`MeanIntensityRegressor` whose state also holds a registered buffer, the count of
    its training forwards, and whose one param group (``get_param_groups``, at the head rate)
    carries a tensor-valued setting, ``rates``, the two rates it was built at: a capture of its
    training state has a buffer and a tensor setting to hold.
    """

    def __init__(self, init_weight: float = 0.0) -> None:
        super().__init__(init_weight)
        self.register_buffer("forwards", torch.zeros(()))

    def fit_loss(self, images, targets) -> dict:
        self.forwards += 1
        return super().fit_loss(images, targets)

    def get_param_groups(self, backbone_lr, head_lr):
        return [
            {
                "params": [self.weight],
                "lr": head_lr,
                "rates": torch.tensor([backbone_lr, head_lr]),
            }
        ]


def build_counting_regressor(
    *, in_chans: int = 1, init_weight: float = 0.0, **_statistics
) -> CountingRegressor:
    """``model_source`` builder for :class:`CountingRegressor`."""
    return CountingRegressor(init_weight=init_weight)


class TwoRateRegressor(MeanIntensityRegressor):
    """Predicts ``weight * mean(image) + bias``, its weight in a param group at the backbone rate
    and its bias in one at the head rate (``get_param_groups``), the two groups in reverse order
    when ``reverse_groups``.
    """

    def __init__(self, reverse_groups: bool = False) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.reverse_groups = reverse_groups

    def predict(self, images):
        return self.weight * images.mean(dim=(1, 2, 3)) + self.bias

    def get_param_groups(self, backbone_lr, head_lr):
        groups = [
            {"params": [self.weight], "lr": backbone_lr},
            {"params": [self.bias], "lr": head_lr},
        ]
        return groups[::-1] if self.reverse_groups else groups


def build_two_rate_regressor(
    *, in_chans: int = 1, reverse_groups: bool = False, **_statistics
) -> TwoRateRegressor:
    """``model_source`` builder for :class:`TwoRateRegressor`."""
    return TwoRateRegressor(reverse_groups=reverse_groups)


class NanEvalRegressor(MeanIntensityRegressor):
    """Finite squared-error loss in train mode, all-nan predictions in eval mode.

    The training loss stays finite while every prediction-derived validation metric goes
    non-finite: a run selecting on the loss completes with those metrics normalized, and one
    selecting on a prediction-derived metric has no selectable epoch.
    """

    def forward(self, images, targets=None):
        if self.training and targets is not None:
            return self.fit_loss(images, targets)
        return {"head0_values": self.predict(images) + float("nan")}


def build_nan_eval_regressor(
    *, in_chans: int = 1, init_weight: float = 0.0, **_statistics
) -> NanEvalRegressor:
    """``model_source`` builder for :class:`NanEvalRegressor`."""
    return NanEvalRegressor(init_weight=init_weight)


class MeanIntensityClassifier(nn.Module):
    """Two-class logit ``[0, weight * (mean(image) - 0.5)]``, cross-entropy loss in train mode:
    frames darker and brighter than mid-gray fall either side of its boundary.

    One parameter, so a run's trajectory is decided by the data it is fed alone.
    """

    num_classes = 2

    def __init__(self, init_weight: float = 0.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([float(init_weight)]))
        self.heads = [self]  # evaluate() reads num_classes off model.heads[0]

    def forward(self, images, targets=None):
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
    **_statistics,
) -> MeanIntensityClassifier:
    """``model_source`` builder for :class:`MeanIntensityClassifier`, which reads raw
    intensities, not the statistics it is handed.
    """
    return MeanIntensityClassifier(init_weight=init_weight)


class DataScaledGradientModel(MeanIntensityRegressor):
    """Loss linear in the parameter: ``weight * sum(batch values)``.

    A batch's gradient is that batch's summed target value and nothing else, independent of the
    weights the backward pass runs at, which makes each batch's contribution to an optimizer
    step separable and checkable on its own.
    """

    def fit_loss(self, images, targets) -> dict:
        return {
            "linear": (self.weight * targets["values"].float().sum()).squeeze()
        }


def build_data_scaled_gradient_model(
    *, in_chans: int = 1, **_statistics
) -> DataScaledGradientModel:
    """``model_source`` builder for :class:`DataScaledGradientModel`."""
    return DataScaledGradientModel()


class AlwaysDivergedModel(MeanIntensityRegressor):
    """Reports a non-finite loss unconditionally, for exercising a diverged run end to end."""

    def fit_loss(self, images, targets) -> dict:
        return {"nan_loss": self.weight * float("nan")}


def build_always_diverged_model(
    *,
    in_chans: int = 1,
    cancel_at_call: int | None = None,
    cancel_output_dir: str | None = None,
    **_statistics,
) -> AlwaysDivergedModel:
    """``model_source`` builder for :class:`AlwaysDivergedModel`."""
    return AlwaysDivergedModel(
        cancel_at_call=cancel_at_call, cancel_output_dir=cancel_output_dir
    )


def run_context(
    root, config: dict, *, epoch_hook=None, experiment_id: str | None = None
):
    """The context the child's own entry builds (``subprocess_worker.prepare_run_context``) for
    the run :func:`~tests._verified_checkpoint_fixtures.opened_run` opens under the project
    ``root`` over ``config``: ``run_registry.observed_run``'s run over that directory and the
    loaders its recorded partition builds, ``epoch_hook`` the context's own.
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


def frames_data(root, intensities, values) -> dict:
    """The data block of a run training on every frame :func:`write_regression_dataset` writes
    under ``root``, at ``intensities`` and valued ``values``, with no validation side (every
    share zero).
    """
    images_dir, csv_path = write_regression_dataset(root, intensities, values)
    return {
        "images_dir": str(images_dir),
        "labels_dir": str(csv_path),
        "split": drawn_split(val_ratio=0.0),
    }


def drawn_data(root, train: tuple[list, list], val: tuple[list, list]) -> dict:
    """The data block of a run drawing its own split over the frames
    :func:`write_regression_dataset` writes under ``root``: ``train``'s and then ``val``'s, each
    ``(intensities, values)``. The frames are grouped so the draw keeps ``val``'s whole on the
    validation side and ``train``'s on the training side: ``train``'s first frame one group, its
    others a second, ``val``'s a third, ``val``'s share its own; the draw meets each side's one
    group smallest first, so ``val`` holds more than one frame and fewer than ``train`` less one.
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


def opposed_regression_data(root, train_intensities, val_intensities) -> dict:
    """:func:`drawn_data` of training frames at ``train_intensities`` valued by :data:`FIT` and
    validation frames at ``val_intensities`` valued by :data:`HOLDOUT_FIT`: the two sides are fit
    by opposite-signed weights, so training progress on one is a regression on the other.
    """
    return drawn_data(
        root,
        (train_intensities, [FIT * c for c in train_intensities]),
        (val_intensities, [HOLDOUT_FIT * c for c in val_intensities]),
    )


def distinct_intensities(n: int) -> list[float]:
    """``n`` frame intensities spread evenly inside ``(0, 1)``, no two alike."""
    return [(i + 1) / (n + 1) for i in range(n)]


TRAIN_INTENSITIES = [0.10, 0.25, 0.40, 0.55, 0.70, 0.85]
VAL_INTENSITIES = [0.15, 0.35, 0.60, 0.90]
"""Sample frame intensities of a regression run's training and validation sides, the two sets
apart so no validation frame repeats a training one."""

SEPARABLE_TRAIN = ([0.10, 0.20, 0.30, 0.70, 0.80, 0.90], [0, 0, 0, 1, 1, 1])
SEPARABLE_VAL = ([0.15, 0.40, 0.60, 0.85], [0, 0, 1, 1])
"""Classification frames labeled 0 below mid-gray and 1 above it, as ``(intensities, labels)``:
six training and four validation frames, which :class:`MeanIntensityClassifier` separates."""


def separable_classifier_data(root) -> dict:
    """:func:`drawn_data` of :data:`SEPARABLE_TRAIN` and :data:`SEPARABLE_VAL`."""
    return drawn_data(root, SEPARABLE_TRAIN, SEPARABLE_VAL)


def count_validations(monkeypatch) -> list[str]:
    """A list each explicit ``model_validate`` of a whole training config
    (``schemas.TrainConfigSchema``) or of a model source alone (``schemas.ModelSourceSchema``)
    appends its class name to from here on.
    """
    from tcip_mcp.pipelines.schemas import ModelSourceSchema, TrainConfigSchema

    seen: list[str] = []
    for schema in (TrainConfigSchema, ModelSourceSchema):
        real = schema.model_validate.__func__

        def counted(cls, obj, *args, _real=real, **kwargs):
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
    **overrides,
) -> dict:
    """A :func:`~tests._chain_fixtures.training_config` of ``project`` of the regression
    ``builder`` (by default the mean-intensity regressor) at ``builder_kwargs`` over the data
    block ``data``: one unfrozen stage at :data:`REGRESSOR_ADAMW` under a cosine schedule of
    ``horizon`` epochs, selecting on the loss, no epoch checkpoints, with ``overrides`` in place
    of their keys. It normalizes by ``normalization``, or with none stated by the statistics of
    ``data``'s own frames.
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
    **overrides,
) -> dict:
    """A :func:`~tests._chain_fixtures.training_config` of ``project`` of the mean-intensity
    classifier (weight -1) as a ``task`` model over the data block ``data``: one unfrozen stage
    in batches of three under a cosine schedule of ``horizon`` epochs, AdamW at 0.2, selecting
    on the loss, no epoch checkpoints, with ``overrides`` in place of their keys. It normalizes
    by ``normalization``, or with none stated by the statistics of ``data``'s own frames.
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


def capture_model(monkeypatch, sink: list) -> None:
    """Every model ``generic_trainer.build_from_model_source`` builds appended to ``sink`` as it
    is built.
    """
    from tcip_mcp.pipelines.training import generic_trainer as gt

    real_build = gt.build_from_model_source

    def build(source, plan, dims, geometry):
        model = real_build(source, plan, dims, geometry)
        sink.append(model)
        return model

    monkeypatch.setattr(gt, "build_from_model_source", build)


class _CallScheduledModel(MeanIntensityRegressor):
    """A :class:`MeanIntensityRegressor` answering its weight-fit loss on the one-based
    training-forward calls ``finite(call)`` admits and a non-finite loss on every other.
    """

    def finite(self, call: int) -> bool:
        raise NotImplementedError

    def fit_loss(self, images, targets) -> dict:
        if self.finite(self._forwards):
            return super().fit_loss(images, targets)
        return {"nan_loss": _phantom_nan()}


class TransientlyDivergedModel(_CallScheduledModel):
    """Reports a non-finite loss for its first ``bad_batches`` forward calls, then a normal
    weight-fit loss for every call after, for proving one diverged epoch short of the trainer's
    own two-pass divergence rule does not kill a run.
    """

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
    **_statistics,
) -> TransientlyDivergedModel:
    """``model_source`` builder for :class:`TransientlyDivergedModel`."""
    return TransientlyDivergedModel(
        bad_batches=bad_batches, init_weight=init_weight
    )


class StepCountedDivergenceModel(_CallScheduledModel):
    """Reports a normal weight-fit loss on the one-based training-forward calls named in
    ``finite_at``, and a non-finite loss on every other call: an exact call-by-call divergence
    pattern.
    """

    def __init__(self, finite_at=(), init_weight: float = 0.0) -> None:
        super().__init__(init_weight)
        self.finite_at = set(finite_at)

    def finite(self, call: int) -> bool:
        return call in self.finite_at


def build_step_counted_divergence_model(
    *,
    in_chans: int = 1,
    finite_at=(),
    init_weight: float = 0.0,
    **_statistics,
) -> StepCountedDivergenceModel:
    """``model_source`` builder for :class:`StepCountedDivergenceModel`."""
    return StepCountedDivergenceModel(
        finite_at=finite_at, init_weight=init_weight
    )


class DivergesAfterModel(_CallScheduledModel):
    """Reports a normal weight-fit loss for its first ``good_calls`` forward calls, then a
    non-finite loss for every call after: the reverse of :class:`TransientlyDivergedModel`, for
    proving a run that trains one real epoch and then dies must not let that epoch's real score
    win a comparison against a config that only ever scored worse.
    """

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
    **_statistics,
) -> DivergesAfterModel:
    """``model_source`` builder for :class:`DivergesAfterModel`."""
    return DivergesAfterModel(good_calls=good_calls, init_weight=init_weight)


class PixelSpreadDivideModel(MeanIntensityRegressor):
    """Divides its prediction by each sample's spread across its bands (their standard deviation
    at each pixel, averaged): a real fp32 division-by-zero divergence on a batch of gray frames
    normalized by statistics alike in every band, while a random synthetic smoke batch (never
    alike across bands) passes the measurement-boundary contract cleanly.
    """

    def predict(self, images):
        return (self.weight + images.mean(dim=(1, 2, 3))) / images.std(
            dim=1
        ).mean(dim=(1, 2))


def build_pixel_spread_divide_model(
    *, in_chans: int = 1, **_statistics
) -> PixelSpreadDivideModel:
    """``model_source`` builder for :class:`PixelSpreadDivideModel`."""
    return PixelSpreadDivideModel()


def write_regression_dataset(
    root, intensities, values, *, height: int = 6, width: int = 10
):
    """Write a small on-disk table-labeled dataset (uint8 RGB PNGs ``img<i>.png`` + a CSV of
    ``stem,value`` rows), a regression's values or a classification's labels.

    Every frame is a checkerboard one uint8 level either side of its intensity
    (``round(255 * fraction)``), toward the inside of the range at either end, so each band
    varies within the frame while the frame's mean stays within half a level of its intensity
    and frames keep their intensities' order. Returns ``(images_dir, csv_path)``.
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
