"""Tiny deterministic models, their ``model_source`` builders, and datasets.

Every model here holds parameters initialized from constants (one, except
:class:`TwoRateRegressor`), so a run's trajectory is decided by the data it is fed and never by
random init. Each builder accepts the width
(``in_chans``) and, for the classifier, the class count a builder is handed; a model reads a
frame's mean intensity whatever its width.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional
from torch.utils.data import Dataset


class ConstantImageDataset(Dataset):
    """Non-square single-channel frames, one intensity per frame, each paired with a target under
    ``key`` (``"values"``, a regression value, by default; ``"labels"`` with ``cast=int`` for a
    class label).

    A frame is filled with a single value, so a sample's mean intensity is exactly its
    ``intensity`` and the loss landscape a loader presents is fixed by its (intensity, target)
    pairs alone.
    """

    def __init__(self, intensities, targets, height: int = 6, width: int = 10, *,
                 key: str = "values", cast=float) -> None:
        if len(intensities) != len(targets):
            raise ValueError("intensities and targets must be the same length")
        self.intensities = [float(i) for i in intensities]
        self.targets = [cast(v) for v in targets]
        self.key = key
        self.height = int(height)
        self.width = int(width)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        image = torch.full((1, self.height, self.width), self.intensities[idx])
        return image, {self.key: self.targets[idx]}


def _phantom_nan() -> torch.Tensor:
    """A non-finite loss on a leaf disconnected from any weight's graph, so a bad call's optimizer
    step never poisons the weight a later good call relies on."""
    return torch.zeros((), requires_grad=True) + float("nan")


class MeanIntensityRegressor(nn.Module):
    """Predicts ``weight * mean(image)``, with a squared-error loss in train mode.

    One parameter, so which data a pass is run over is the only thing that can move its loss. A
    subclass changes only which training calls it answers with that loss.
    """

    def __init__(self, init_weight: float = 0.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([float(init_weight)]))

    def predict(self, images):
        return self.weight * images.mean(dim=(1, 2, 3))

    def fit_loss(self, images, targets) -> dict:
        return {"mse": ((self.predict(images) - targets["values"].float()) ** 2).mean()}

    def forward(self, images, targets=None):
        if self.training and targets is not None:
            return self.fit_loss(images, targets)
        return {"head0_values": self.predict(images)}


def build_mean_intensity_regressor(
    *, in_chans: int = 1, init_weight: float = 0.0,
) -> MeanIntensityRegressor:
    """``model_source`` builder for :class:`MeanIntensityRegressor`."""
    return MeanIntensityRegressor(init_weight=init_weight)


class TwoRateRegressor(MeanIntensityRegressor):
    """Predicts ``weight * mean(image) + bias``, its weight in a param group at the backbone rate
    and its bias in one at the head rate (``get_param_groups``), the two groups in reverse order
    when ``reverse_groups``."""

    def __init__(self, reverse_groups: bool = False) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.reverse_groups = reverse_groups

    def predict(self, images):
        return self.weight * images.mean(dim=(1, 2, 3)) + self.bias

    def get_param_groups(self, backbone_lr, head_lr):
        groups = [{"params": [self.weight], "lr": backbone_lr},
                  {"params": [self.bias], "lr": head_lr}]
        return groups[::-1] if self.reverse_groups else groups


def build_two_rate_regressor(*, in_chans: int = 1, reverse_groups: bool = False
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


def build_nan_eval_regressor(*, in_chans: int = 1, init_weight: float = 0.0) -> NanEvalRegressor:
    """``model_source`` builder for :class:`NanEvalRegressor`."""
    return NanEvalRegressor(init_weight=init_weight)


class MeanIntensityClassifier(nn.Module):
    """Two-class logit ``[0, weight * mean(image)]``, cross-entropy loss in train mode.

    One parameter, so a run's trajectory is decided by the data it is fed alone.
    """

    num_classes = 2

    def __init__(self, init_weight: float = 0.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([float(init_weight)]))
        self.heads = [self]  # evaluate() reads num_classes off model.heads[0]

    def forward(self, images, targets=None):
        logit1 = self.weight * images.mean(dim=(1, 2, 3))
        logits = torch.stack([torch.zeros_like(logit1), logit1], dim=1)
        if self.training and targets is not None:
            return {"ce": functional.cross_entropy(logits, targets["labels"])}
        return {"head0_labels": logits.argmax(dim=1)}


def build_mean_intensity_classifier(
    *, in_chans: int = 1, num_classes: int = 2, init_weight: float = 0.0,
) -> MeanIntensityClassifier:
    """``model_source`` builder for :class:`MeanIntensityClassifier`."""
    return MeanIntensityClassifier(init_weight=init_weight)


class DataScaledGradientModel(MeanIntensityRegressor):
    """Loss linear in the parameter: ``weight * sum(batch values)``.

    A batch's gradient is that batch's summed target value and nothing else, independent of the
    weights the backward pass runs at, which makes each batch's contribution to an optimizer
    step separable and checkable on its own.
    """

    def fit_loss(self, images, targets) -> dict:
        return {"linear": (self.weight * targets["values"].float().sum()).squeeze()}


def build_data_scaled_gradient_model(*, in_chans: int = 1) -> DataScaledGradientModel:
    """``model_source`` builder for :class:`DataScaledGradientModel`."""
    return DataScaledGradientModel()


class AlwaysDivergedModel(MeanIntensityRegressor):
    """Reports a non-finite loss unconditionally, for exercising a diverged run end to end.

    ``on_forward``, when given, is called with the one-based training-forward-call count after
    each training-mode forward, so a caller can trigger a side effect (e.g. requesting
    cancellation) at an exact point in the batch stream without threading a real clock through
    the trainer.
    """

    def __init__(self, on_forward=None) -> None:
        super().__init__()
        self.on_forward = on_forward
        self._calls = 0

    def fit_loss(self, images, targets) -> dict:
        self._calls += 1
        if self.on_forward is not None:
            self.on_forward(self._calls)
        return {"nan_loss": self.weight * float("nan")}


def build_always_diverged_model(*, in_chans: int = 1, on_forward=None) -> AlwaysDivergedModel:
    """``model_source`` builder for :class:`AlwaysDivergedModel`."""
    return AlwaysDivergedModel(on_forward=on_forward)


class CancelSentinelAtCall:
    """An ``AlwaysDivergedModel``-style ``on_forward`` callback that requests the run's own
    cancellation (``experiments.request_cancel``, what ``TrainRun.should_cancel()`` reads) on one
    named forward-call count. Holds only ``output_dir`` (a plain string) and the call count, never
    the run object itself, so it stays picklable through a checkpoint write."""

    def __init__(self, output_dir, at_call: int) -> None:
        self.output_dir = str(output_dir)
        self.at_call = int(at_call)

    def __call__(self, call_count: int) -> None:
        if call_count == self.at_call:
            from pathlib import Path

            from tcip_mcp.experiments import request_cancel

            path = Path(self.output_dir)
            path.mkdir(parents=True, exist_ok=True)
            request_cancel(path)


def trainer_run(config: dict, output_dir, *, project, has_val_loader: bool, id: str = "run"):
    """A ``TrainRun`` of ``project`` over ``config`` writing into ``output_dir`` at the objective
    the launcher's own producer resolves for it (``generic_trainer.resolve_objective``)."""
    from pathlib import Path

    from tcip_mcp.pipelines.model_build import run_task
    from tcip_mcp.pipelines.schemas import train_config
    from tcip_mcp.pipelines.training.generic_trainer import resolve_objective
    from tcip_mcp.pipelines.training.run_registry import TrainRun

    spec = train_config(config)
    return TrainRun(id=id, config=config, spec=spec,
                    objective=resolve_objective(spec, run_task(config),
                                                project=Path(project),
                                                has_val_loader=has_val_loader),
                    project=Path(project), output_dir=str(output_dir))


def count_validations(monkeypatch) -> list[str]:
    """A list each explicit ``model_validate`` of a whole training config
    (``schemas.TrainConfigSchema``) or of a model source alone (``schemas.ModelSourceSchema``)
    appends its class name to from here on."""
    from tcip_mcp.pipelines.schemas import ModelSourceSchema, TrainConfigSchema

    seen: list[str] = []
    for schema in (TrainConfigSchema, ModelSourceSchema):
        real = schema.model_validate.__func__

        def counted(cls, obj, *args, _real=real, **kwargs):
            seen.append(cls.__name__)
            return _real(cls, obj, *args, **kwargs)

        monkeypatch.setattr(schema, "model_validate", classmethod(counted))
    return seen


MEAN_INTENSITY_REGRESSOR = "tests.tiny_trainer_fixtures:build_mean_intensity_regressor"
MEAN_INTENSITY_CLASSIFIER = "tests.tiny_trainer_fixtures:build_mean_intensity_classifier"
ALWAYS_DIVERGED_MODEL = "tests.tiny_trainer_fixtures:build_always_diverged_model"
NAN_EVAL_REGRESSOR = "tests.tiny_trainer_fixtures:build_nan_eval_regressor"
TWO_RATE_REGRESSOR = "tests.tiny_trainer_fixtures:build_two_rate_regressor"

REGRESSOR_ADAMW = {"name": "adamw", "backbone_lr": 0.05, "head_lr": 0.05, "weight_decay": 0.0}
"""The AdamW section the one-weight regressors here train at."""


def regressor_config(epochs: int = 1, *, builder: str = MEAN_INTENSITY_REGRESSOR,
                     builder_kwargs: dict | None = None, data: dict | None = None,
                     **overrides) -> dict:
    """A :func:`~tests._chain_fixtures.training_config` of the regression ``builder`` (by
    default the mean-intensity regressor) at ``builder_kwargs`` over ``data`` (by default
    one-band frames with no scope): ``epochs`` epochs of one unfrozen stage at
    :data:`REGRESSOR_ADAMW`, no epoch checkpoints, with ``overrides`` in place of their keys."""
    from tests._chain_fixtures import training_config

    source: dict = {"builder": builder, "task": "regression"}
    if builder_kwargs is not None:
        source["builder_kwargs"] = builder_kwargs
    return training_config(
        source, data if data is not None else {"num_channels": 1, "scope": {}},
        **{"stages": [{"freeze_to": 0, "epochs": epochs}], "optimizer": REGRESSOR_ADAMW,
           "checkpoint_every_n_epochs": 0, **overrides})


def classifier_config(epochs: int, **overrides) -> dict:
    """A :func:`~tests._chain_fixtures.training_config` of the mean-intensity classifier
    (weight -1) over one-band, two-class frames: ``epochs`` epochs of one unfrozen stage, AdamW
    at 0.2, no epoch checkpoints, with ``overrides`` in place of their keys."""
    from tests._chain_fixtures import training_config

    return training_config(
        {"builder": MEAN_INTENSITY_CLASSIFIER, "builder_kwargs": {"init_weight": -1.0},
         "task": "classification"},
        {"num_channels": 1, "num_classes": 2, "scope": {}},
        **{"stages": [{"freeze_to": 0, "epochs": epochs}],
           "optimizer": {"name": "adamw", "backbone_lr": 0.2, "head_lr": 0.2,
                         "weight_decay": 0.0},
           "checkpoint_every_n_epochs": 0, **overrides})


def separable_classifier_loaders():
    """A classification training loader of six frames (three of negative intensity, three of
    positive, batches of three) and a validation loader of four (two each, one batch), labeled 0
    for negative intensity and 1 for positive."""
    from torch.utils.data import DataLoader

    from tcip_mcp.pipelines.training.collation import task_collate

    collate = task_collate("classification")
    train_ds = ConstantImageDataset(
        [-2.0, -1.5, -1.0, 1.0, 1.5, 2.0], [0, 0, 0, 1, 1, 1], key="labels", cast=int)
    val_ds = ConstantImageDataset([-1.8, -0.4, 0.4, 1.8], [0, 0, 1, 1], key="labels", cast=int)
    return (DataLoader(train_ds, batch_size=3, collate_fn=collate),
            DataLoader(val_ds, batch_size=4, collate_fn=collate))


def opposed_regression_loaders(train_intensities, val_intensities, *, shuffle_seed=None):
    """A regression training loader over ``train_intensities`` fit by weight +2 and a holdout
    loader over ``val_intensities`` fit by weight -5, batches of two. ``shuffle_seed`` makes the
    training loader shuffle from a generator of its own seeded with it; ``None`` keeps its order
    fixed."""
    from torch.utils.data import DataLoader

    from tcip_mcp.pipelines.training.collation import task_collate

    train_ds = ConstantImageDataset(train_intensities, [2.0 * c for c in train_intensities])
    val_ds = ConstantImageDataset(
        val_intensities, [-5.0 * c for c in val_intensities], height=8, width=12)
    collate = task_collate("regression")
    shuffling: dict = {}
    if shuffle_seed is not None:
        shuffling = {"shuffle": True, "generator": torch.Generator().manual_seed(shuffle_seed)}
    return (DataLoader(train_ds, batch_size=2, collate_fn=collate, **shuffling),
            DataLoader(val_ds, batch_size=2, collate_fn=collate))


def capture_model(monkeypatch, sink: list) -> None:
    """Every model ``generic_trainer.build_from_model_source`` builds appended to ``sink`` as it
    is built."""
    from tcip_mcp.pipelines.training import generic_trainer as gt

    real_build = gt.build_from_model_source

    def build(source, dims):
        model = real_build(source, dims)
        sink.append(model)
        return model

    monkeypatch.setattr(gt, "build_from_model_source", build)


class _CallScheduledModel(MeanIntensityRegressor):
    """A :class:`MeanIntensityRegressor` answering its weight-fit loss on the one-based
    training-forward calls ``finite(call)`` admits and a non-finite loss on every other."""

    def __init__(self, init_weight: float = 0.0) -> None:
        super().__init__(init_weight)
        self._calls = 0

    def finite(self, call: int) -> bool:
        raise NotImplementedError

    def fit_loss(self, images, targets) -> dict:
        self._calls += 1
        if self.finite(self._calls):
            return super().fit_loss(images, targets)
        return {"nan_loss": _phantom_nan()}


class TransientlyDivergedModel(_CallScheduledModel):
    """Reports a non-finite loss for its first ``bad_batches`` forward calls, then a normal
    weight-fit loss for every call after, for proving one diverged epoch short of the trainer's
    own two-pass divergence rule does not kill a run."""

    def __init__(self, bad_batches: int = 2, init_weight: float = 0.0) -> None:
        super().__init__(init_weight)
        self.bad_batches = int(bad_batches)

    def finite(self, call: int) -> bool:
        return call > self.bad_batches


def build_transiently_diverged_model(
    *, in_chans: int = 1, bad_batches: int = 2, init_weight: float = 0.0,
) -> TransientlyDivergedModel:
    """``model_source`` builder for :class:`TransientlyDivergedModel`."""
    return TransientlyDivergedModel(bad_batches=bad_batches, init_weight=init_weight)


class StepCountedDivergenceModel(_CallScheduledModel):
    """Reports a normal weight-fit loss on the one-based training-forward calls named in
    ``finite_at``, and a non-finite loss on every other call: an exact call-by-call divergence
    pattern."""

    def __init__(self, finite_at=(), init_weight: float = 0.0) -> None:
        super().__init__(init_weight)
        self.finite_at = set(finite_at)

    def finite(self, call: int) -> bool:
        return call in self.finite_at


def build_step_counted_divergence_model(
    *, in_chans: int = 1, finite_at=(), init_weight: float = 0.0,
) -> StepCountedDivergenceModel:
    """``model_source`` builder for :class:`StepCountedDivergenceModel`."""
    return StepCountedDivergenceModel(finite_at=finite_at, init_weight=init_weight)


class DivergesAfterModel(_CallScheduledModel):
    """Reports a normal weight-fit loss for its first ``good_calls`` forward calls, then a
    non-finite loss for every call after: the reverse of :class:`TransientlyDivergedModel`, for
    proving a run that trains one real epoch and then dies must not let that epoch's real score
    win a comparison against a config that only ever scored worse."""

    def __init__(self, good_calls: int, init_weight: float = 0.0) -> None:
        super().__init__(init_weight)
        self.good_calls = int(good_calls)

    def finite(self, call: int) -> bool:
        return call <= self.good_calls


def build_diverges_after_model(
    *, good_calls: int, in_chans: int = 1, init_weight: float = 0.0,
) -> DivergesAfterModel:
    """``model_source`` builder for :class:`DivergesAfterModel`."""
    return DivergesAfterModel(good_calls=good_calls, init_weight=init_weight)


class PixelSumDivideModel(MeanIntensityRegressor):
    """Divides its prediction by the batch's own per-sample pixel sum: a real fp32
    division-by-zero divergence on a batch of zero-intensity images, while a random synthetic
    smoke batch (never exactly zero) passes the measurement-boundary contract cleanly."""

    def predict(self, images):
        return (self.weight + images.mean(dim=(1, 2, 3))) / images.sum(dim=(1, 2, 3))


def build_pixel_sum_divide_model(*, in_chans: int = 1) -> PixelSumDivideModel:
    """``model_source`` builder for :class:`PixelSumDivideModel`."""
    return PixelSumDivideModel()


def write_regression_dataset(root, intensities, values, *, height: int = 6, width: int = 10):
    """Write a small on-disk regression dataset (uint8 RGB PNGs + a CSV of ``stem,value`` rows),
    the real-file counterpart to :class:`ConstantImageDataset`.

    Every frame is filled with one uint8 intensity (``round(255 * fraction)``), so
    ``intensities=[0.0, ...]`` decodes to exactly zero-valued pixels (``pil_to_tensor`` scales a
    uint8 frame by 255). Returns ``(images_dir, csv_path)``.
    """
    from pathlib import Path

    from PIL import Image

    if len(intensities) != len(values):
        raise ValueError("intensities and values must be the same length")
    from tcip_mcp.dataset_layout import UNDATED_BUCKET

    images_dir = Path(root) / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(root) / "values.csv"
    rows = ["stem,value"]
    for i, (frac, value) in enumerate(zip(intensities, values)):
        stem = f"img{i}"
        px = round(255 * float(frac))
        Image.new("RGB", (width, height), (px, px, px)).save(images_dir / f"{stem}.png")
        rows.append(f"{stem},{value}")
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    return images_dir, csv_path
