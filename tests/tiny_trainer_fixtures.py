"""Tiny deterministic models and datasets for driving ``generic_trainer.train`` end to end.

Every model here holds a single parameter initialized from a constant, so a run's trajectory is
decided by the data it is fed and never by random init: two runs on the same batches take the
same path, and two runs on different batches take visibly different ones.

Not a ``test_*`` module: the trainer imports these builders by dotted name through
``model_source``, the same seam a real bespoke model comes through.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class ConstantImageDataset(Dataset):
    """Non-square single-channel frames, one intensity per frame, paired with a regression value.

    A frame is filled with a single value, so a sample's mean intensity is exactly its
    ``intensity`` and the loss landscape a loader presents is fixed by its (intensity, value)
    pairs alone.
    """

    def __init__(self, intensities, values, height: int = 6, width: int = 10) -> None:
        if len(intensities) != len(values):
            raise ValueError("intensities and values must be the same length")
        self.intensities = [float(i) for i in intensities]
        self.values = [float(v) for v in values]
        self.height = int(height)
        self.width = int(width)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, idx: int):
        image = torch.full((1, self.height, self.width), self.intensities[idx])
        return image, {"values": self.values[idx]}


class MeanIntensityRegressor(nn.Module):
    """Predicts ``weight * mean(image)``, with a squared-error loss in train mode.

    One parameter, so which data a pass is run over is the only thing that can move its loss.
    """

    def __init__(self, init_weight: float = 0.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([float(init_weight)]))

    def forward(self, images, targets=None):
        pred = self.weight * images.mean(dim=(1, 2, 3))
        if self.training and targets is not None:
            return {"mse": ((pred - targets["values"].float()) ** 2).mean()}
        return {"head0_values": pred}


def build_mean_intensity_regressor(*, init_weight: float = 0.0) -> MeanIntensityRegressor:
    """``model_source`` builder for :class:`MeanIntensityRegressor`."""
    return MeanIntensityRegressor(init_weight=init_weight)


class ConstantImageClassDataset(Dataset):
    """Non-square single-channel frames, one intensity per frame, paired with a class label."""

    def __init__(self, intensities, labels, height: int = 6, width: int = 10) -> None:
        if len(intensities) != len(labels):
            raise ValueError("intensities and labels must be the same length")
        self.intensities = [float(i) for i in intensities]
        self.labels = [int(v) for v in labels]
        self.height = int(height)
        self.width = int(width)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        image = torch.full((1, self.height, self.width), self.intensities[idx])
        return image, {"labels": self.labels[idx]}


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
            return {"ce": F.cross_entropy(logits, targets["labels"])}
        return {"head0_labels": logits.argmax(dim=1)}


def build_mean_intensity_classifier(*, init_weight: float = 0.0) -> MeanIntensityClassifier:
    """``model_source`` builder for :class:`MeanIntensityClassifier`."""
    return MeanIntensityClassifier(init_weight=init_weight)


class DataScaledGradientModel(nn.Module):
    """Loss linear in the parameter: ``weight * sum(batch values)``.

    A batch's gradient is that batch's summed target value and nothing else, independent of the
    weights the backward pass runs at, which makes each batch's contribution to an optimizer
    step separable and checkable on its own.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, images, targets=None):
        if self.training and targets is not None:
            return {"linear": (self.weight * targets["values"].float().sum()).squeeze()}
        return {"head0_values": self.weight * images.mean(dim=(1, 2, 3))}


def build_data_scaled_gradient_model() -> DataScaledGradientModel:
    """``model_source`` builder for :class:`DataScaledGradientModel`."""
    return DataScaledGradientModel()


class AlwaysDivergedModel(nn.Module):
    """Reports a non-finite loss unconditionally, for exercising a diverged run end to end.

    ``on_forward``, when given, is called with the one-based training-forward-call count after
    each training-mode forward, so a caller can trigger a side effect (e.g. requesting
    cancellation) at an exact point in the batch stream without threading a real clock through
    the trainer.
    """

    def __init__(self, on_forward=None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.on_forward = on_forward
        self._calls = 0

    def forward(self, images, targets=None):
        if self.training and targets is not None:
            self._calls += 1
            if self.on_forward is not None:
                self.on_forward(self._calls)
            return {"nan_loss": self.weight * float("nan")}
        return {"head0_values": self.weight * images.mean(dim=(1, 2, 3))}


def build_always_diverged_model(*, on_forward=None) -> AlwaysDivergedModel:
    """``model_source`` builder for :class:`AlwaysDivergedModel`."""
    return AlwaysDivergedModel(on_forward=on_forward)


class CancelSentinelAtCall:
    """An ``AlwaysDivergedModel``-style ``on_forward`` callback that touches the run's own
    ``.cancel_requested`` sentinel (the same file ``TrainRun.should_cancel()`` polls) on one named
    forward-call count. Holds only ``output_dir`` (a plain string) and the call count, never the
    run object itself, so it stays picklable through a checkpoint write."""

    def __init__(self, output_dir, at_call: int) -> None:
        self.output_dir = str(output_dir)
        self.at_call = int(at_call)

    def __call__(self, call_count: int) -> None:
        if call_count == self.at_call:
            from pathlib import Path

            path = Path(self.output_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / ".cancel_requested").touch()


class TransientlyDivergedModel(nn.Module):
    """Reports a non-finite loss for its first ``bad_batches`` forward calls, then a normal
    weight-fit loss for every call after, for proving one diverged epoch short of the trainer's
    own two-pass divergence rule does not kill a run."""

    def __init__(self, bad_batches: int = 2, init_weight: float = 0.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([float(init_weight)]))
        self.bad_batches = int(bad_batches)
        self._calls = 0

    def forward(self, images, targets=None):
        if self.training and targets is not None:
            self._calls += 1
            if self._calls <= self.bad_batches:
                # A leaf disconnected from `weight`'s graph, so a bad batch's optimizer step
                # never poisons the weight the later recovery relies on.
                phantom = torch.zeros((), requires_grad=True) + float("nan")
                return {"nan_loss": phantom}
            pred = self.weight * images.mean(dim=(1, 2, 3))
            return {"mse": ((pred - targets["values"].float()) ** 2).mean()}
        pred = self.weight * images.mean(dim=(1, 2, 3))
        return {"head0_values": pred}


def build_transiently_diverged_model(
    *, bad_batches: int = 2, init_weight: float = 0.0,
) -> TransientlyDivergedModel:
    """``model_source`` builder for :class:`TransientlyDivergedModel`."""
    return TransientlyDivergedModel(bad_batches=bad_batches, init_weight=init_weight)


class StepCountedDivergenceModel(nn.Module):
    """Reports a normal weight-fit loss on the one-based training-forward calls named in
    ``finite_at``, and a non-finite loss on every other call, for constructing an exact
    call-by-call (and so, given a known batches-per-epoch count, epoch-by-epoch) divergence
    pattern up front rather than inferring one from a bad-batch prefix."""

    def __init__(self, finite_at=(), init_weight: float = 0.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([float(init_weight)]))
        self.finite_at = set(finite_at)
        self._calls = 0

    def forward(self, images, targets=None):
        if self.training and targets is not None:
            self._calls += 1
            if self._calls in self.finite_at:
                pred = self.weight * images.mean(dim=(1, 2, 3))
                return {"mse": ((pred - targets["values"].float()) ** 2).mean()}
            # A leaf disconnected from `weight`'s graph, so a bad call's optimizer step never
            # poisons the weight a later good call relies on.
            phantom = torch.zeros((), requires_grad=True) + float("nan")
            return {"nan_loss": phantom}
        pred = self.weight * images.mean(dim=(1, 2, 3))
        return {"head0_values": pred}


def build_step_counted_divergence_model(
    *, finite_at=(), init_weight: float = 0.0,
) -> StepCountedDivergenceModel:
    """``model_source`` builder for :class:`StepCountedDivergenceModel`."""
    return StepCountedDivergenceModel(finite_at=finite_at, init_weight=init_weight)


class DivergesAfterModel(nn.Module):
    """Reports a normal weight-fit loss for its first ``good_calls`` forward calls, then a
    non-finite loss for every call after: the reverse of :class:`TransientlyDivergedModel`, for
    proving a run that trains one real epoch and then dies must not let that epoch's real score
    win a comparison against a config that only ever scored worse."""

    def __init__(self, good_calls: int, init_weight: float = 0.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([float(init_weight)]))
        self.good_calls = int(good_calls)
        self._calls = 0

    def forward(self, images, targets=None):
        if self.training and targets is not None:
            self._calls += 1
            if self._calls <= self.good_calls:
                pred = self.weight * images.mean(dim=(1, 2, 3))
                return {"mse": ((pred - targets["values"].float()) ** 2).mean()}
            phantom = torch.zeros((), requires_grad=True) + float("nan")
            return {"nan_loss": phantom}
        pred = self.weight * images.mean(dim=(1, 2, 3))
        return {"head0_values": pred}


def build_diverges_after_model(*, good_calls: int, init_weight: float = 0.0) -> DivergesAfterModel:
    """``model_source`` builder for :class:`DivergesAfterModel`."""
    return DivergesAfterModel(good_calls=good_calls, init_weight=init_weight)


class PixelSumDivideModel(nn.Module):
    """Divides its prediction by the batch's own per-sample pixel sum: a real fp32
    division-by-zero divergence on a batch of zero-intensity images, rather than a hand-authored
    nan, while a random synthetic smoke batch (never exactly zero) passes the measurement-boundary
    contract cleanly."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, images, targets=None):
        pixel_sum = images.sum(dim=(1, 2, 3))
        pred = (self.weight + images.mean(dim=(1, 2, 3))) / pixel_sum
        if self.training and targets is not None:
            return {"mse": ((pred - targets["values"].float()) ** 2).mean()}
        return {"head0_values": pred}


def build_pixel_sum_divide_model() -> PixelSumDivideModel:
    """``model_source`` builder for :class:`PixelSumDivideModel`."""
    return PixelSumDivideModel()


def write_regression_dataset(root, intensities, values, *, height: int = 6, width: int = 10):
    """Write a small on-disk regression dataset (uint8 RGB PNGs + a CSV of ``stem,value`` rows),
    the real-file counterpart to :class:`ConstantImageDataset` for a run that must go through the
    known ``RegressionDataset`` loader (a real subprocess launch, or ``_run_hpo_trial``'s own
    ``auto_train_val`` build) rather than a tensor dataset passed straight to ``train()``.

    Every frame is filled with one uint8 intensity (``round(255 * fraction)``), so
    ``intensities=[0.0, ...]`` decodes to exactly zero-valued pixels (``pil_to_tensor`` scales a
    uint8 frame by 255). Returns ``(images_dir, csv_path)``.
    """
    from pathlib import Path

    from PIL import Image

    if len(intensities) != len(values):
        raise ValueError("intensities and values must be the same length")
    images_dir = Path(root) / "images"
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
