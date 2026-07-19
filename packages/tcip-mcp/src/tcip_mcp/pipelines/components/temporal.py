"""Temporal modeling heads for phenology prediction.

TemporalHead takes a sequence of (date, embedding) pairs per plant
and predicts milestone dates (5%, 50%, 95% thresholds).

Also provides TemporalTransformerHead as an alternative to LSTM.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalLSTMHead(nn.Module):
    """LSTM over per-date embeddings → phenology milestone predictions.

    Input: [B, T, D] — B plants, T dates, D embedding dim
    Output: [B, num_milestones] — predicted dates for each milestone
    """

    task_type = "temporal"
    default_loss = "smooth_l1"

    def __init__(
        self,
        in_channels: int = 512,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_milestones: int = 3,  # 5%, 50%, 95%
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_milestones),
        )
        self.num_milestones = num_milestones

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] temporal embeddings

        Returns:
            [B, num_milestones] predicted milestone values (e.g., day-of-year)
        """
        output, (hn, _) = self.lstm(x)
        # Use last hidden state from both directions
        last = output[:, -1, :]  # [B, hidden*2]
        return self.head(last)

    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """SmoothL1 loss on milestone predictions."""
        return nn.functional.smooth_l1_loss(predictions, targets)

    def decode(self, predictions: torch.Tensor) -> dict:
        return {"milestones": predictions}


class TemporalTransformerHead(nn.Module):
    """Transformer encoder over temporal embeddings → milestone predictions.

    Better than LSTM for long sequences or when positional encoding matters.
    """

    task_type = "temporal"
    default_loss = "smooth_l1"

    def __init__(
        self,
        in_channels: int = 512,
        num_milestones: int = 3,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pos_encoding = nn.Parameter(torch.randn(1, 365, in_channels) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, num_milestones),
        )
        self.num_milestones = num_milestones

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] temporal embeddings

        Returns:
            [B, num_milestones]
        """
        T = x.shape[1]
        x = x + self.pos_encoding[:, :T, :]
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)  # global average over time
        return self.head(pooled)

    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.smooth_l1_loss(predictions, targets)

    def decode(self, predictions: torch.Tensor) -> dict:
        return {"milestones": predictions}


def _build_temporal_lstm(in_channels: int = 512, num_milestones: int = 3, **kwargs):
    return TemporalLSTMHead(in_channels=in_channels, num_milestones=num_milestones, **kwargs)


def _build_temporal_transformer(in_channels: int = 512, num_milestones: int = 3, **kwargs):
    return TemporalTransformerHead(in_channels=in_channels, num_milestones=num_milestones, **kwargs)
