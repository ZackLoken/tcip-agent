"""Neck modules: adapt backbone features for downstream heads.

Necks sit between the backbone and head(s), transforming multi-scale
feature maps into the format each head expects.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """Feature Pyramid Network.

    Takes multi-scale features from backbone and produces uniform-channel
    multi-scale features via lateral connections and top-down pathway.
    """

    def __init__(self, in_channels_list: list[int], out_channels: int = 256, add_p2: bool = False) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.add_p2 = add_p2
        self.num_levels = len(in_channels_list) + int(add_p2)

        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        for in_ch in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_ch, out_channels, 1))
            self.output_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        # Opt-in extra finer level (yolo11-p2 analog): 2x upsample of the finest lateral.
        if add_p2:
            self.p2_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        keys = sorted(features.keys())
        feat_list = [features[k] for k in keys]

        # Lateral connections
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, feat_list)]

        # Top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            up = F.interpolate(laterals[i + 1], size=laterals[i].shape[-2:], mode="nearest")
            laterals[i] = laterals[i] + up

        # 3x3 convs to remove aliasing, finest -> coarsest.
        outs = [conv(laterals[i]) for i, conv in enumerate(self.output_convs)]
        if self.add_p2:
            extra = self.p2_conv(F.interpolate(laterals[0], scale_factor=2, mode="nearest"))
            outs.insert(0, extra)  # new finest level at index 0
        return {f"p{i}": f for i, f in enumerate(outs)}


class PAN(nn.Module):
    """Path Aggregation Network: bidirectional FPN.

    Adds bottom-up pathway after FPN top-down for better low-level features.
    """

    def __init__(self, in_channels_list: list[int], out_channels: int = 256, add_p2: bool = False) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.fpn = FPN(in_channels_list, out_channels, add_p2=add_p2)
        self.bottom_up_convs = nn.ModuleList()
        for _ in range(len(in_channels_list) + int(add_p2) - 1):
            self.bottom_up_convs.append(
                nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
            )

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        fpn_out = self.fpn(features)
        keys = sorted(fpn_out.keys())
        feat_list = [fpn_out[k] for k in keys]

        # Bottom-up pathway
        for i in range(len(self.bottom_up_convs)):
            feat_list[i + 1] = feat_list[i + 1] + self.bottom_up_convs[i](feat_list[i])

        return {f"p{i}": f for i, f in enumerate(feat_list)}


class IdentityNeck(nn.Module):
    """Pass-through: returns features unchanged.

    Use when the head directly consumes backbone features (e.g. when
    only the final feature map is needed).
    """

    def __init__(self, in_channels_list: list[int], **_: Any) -> None:
        super().__init__()
        self.out_channels = in_channels_list[-1] if in_channels_list else 0
        self._in_channels_list = in_channels_list

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return features


class GlobalAvgPoolNeck(nn.Module):
    """Global average pooling: produces a flat feature vector.

    Takes the last-stage feature map from backbone and pools it to
    [B, C].  Used for classification, ordinal, and regression heads.
    """

    def __init__(self, in_channels_list: list[int], **_: Any) -> None:
        super().__init__()
        self.out_channels = in_channels_list[-1] if in_channels_list else 0

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        # Take highest-level features
        keys = sorted(features.keys())
        x = features[keys[-1]]
        return F.adaptive_avg_pool2d(x, 1).flatten(1)


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def _build_fpn(in_channels_list: list[int], out_channels: int = 256, add_p2: bool = False, **_: Any) -> FPN:
    return FPN(in_channels_list, out_channels, add_p2=add_p2)


def _build_pan(in_channels_list: list[int], out_channels: int = 256, add_p2: bool = False, **_: Any) -> PAN:
    return PAN(in_channels_list, out_channels, add_p2=add_p2)


def _build_identity(in_channels_list: list[int], **kw: Any) -> IdentityNeck:
    return IdentityNeck(in_channels_list, **kw)


def _build_gap(in_channels_list: list[int], **kw: Any) -> GlobalAvgPoolNeck:
    return GlobalAvgPoolNeck(in_channels_list, **kw)
