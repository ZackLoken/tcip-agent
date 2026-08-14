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
