"""3D point cloud backbones for LiDAR data.

PointNet++ set abstraction layers for tree structure analysis.
Registered into BACKBONES registry (via ``tcip_mcp.tools.pipeline_tools``)
alongside 2D CNN/ViT backbones.

EXPERIMENTAL — component only, not usable end-to-end. There is no point-cloud
dataset/loader in ``pipelines.data`` and ``build_dataset`` has no point-cloud
task type, so this backbone cannot be trained through the normal pipeline yet.
Wiring a 3D data path is future work; see the Roadmap in the repo README.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tcip_mcp.pipelines.registry import BACKBONES


class SetAbstraction(nn.Module):
    """PointNet++ set abstraction layer (simplified).

    Groups points into local regions, applies shared-MLP, and max-pools
    to produce subsampled points with richer features.
    """

    def __init__(self, n_points: int, radius: float, n_samples: int, in_channels: int, mlp_channels: list[int]) -> None:
        super().__init__()
        self.n_points = n_points
        self.radius = radius
        self.n_samples = n_samples

        layers = []
        prev = in_channels + 3  # +3 for relative xyz
        for ch in mlp_channels:
            layers.append(nn.Conv1d(prev, ch, 1))
            layers.append(nn.BatchNorm1d(ch))
            layers.append(nn.ReLU(inplace=True))
            prev = ch
        self.mlp = nn.Sequential(*layers)
        self.out_ch = mlp_channels[-1]

    def forward(self, xyz: torch.Tensor, features: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: [B, N, 3] point coordinates
            features: [B, N, C] point features (or None for first layer)

        Returns:
            new_xyz: [B, n_points, 3] subsampled centroids
            new_features: [B, n_points, out_ch] aggregated features
        """
        B, N, _ = xyz.shape

        # Farthest point sampling (simplified: random for now)
        idx = torch.randint(0, N, (B, self.n_points), device=xyz.device)
        new_xyz = torch.gather(xyz, 1, idx.unsqueeze(-1).expand(-1, -1, 3))

        # Ball query + grouping (simplified: kNN)
        dists = torch.cdist(new_xyz, xyz)  # [B, n_points, N]
        _, knn_idx = dists.topk(self.n_samples, dim=-1, largest=False)  # [B, n_points, n_samples]

        # Gather grouped points
        grouped_xyz = torch.gather(xyz.unsqueeze(1).expand(-1, self.n_points, -1, -1), 2,
                                   knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3))
        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)  # relative coords

        if features is not None:
            C = features.shape[-1]
            grouped_feat = torch.gather(features.unsqueeze(1).expand(-1, self.n_points, -1, -1), 2,
                                        knn_idx.unsqueeze(-1).expand(-1, -1, -1, C))
            grouped = torch.cat([grouped_xyz, grouped_feat], dim=-1)  # [B, n_points, n_samples, 3+C]
        else:
            grouped = grouped_xyz  # [B, n_points, n_samples, 3]

        # shared MLP: reshape to [B*n_points, 3+C, n_samples] for Conv1d
        grouped = grouped.reshape(B * self.n_points, self.n_samples, -1).permute(0, 2, 1)
        out = self.mlp(grouped)  # [B*n_points, out_ch, n_samples]
        out = out.max(dim=-1)[0]  # [B*n_points, out_ch]
        new_features = out.reshape(B, self.n_points, -1)

        return new_xyz, new_features


class PointNetPPBackbone(nn.Module):
    """PointNet++ backbone with 3 set abstraction layers.

    Input: [B, N, 3+C] point cloud (xyz + optional features)
    Output: dict of multi-scale features
    """

    def __init__(self, in_channels: int = 0) -> None:
        super().__init__()
        self.sa1 = SetAbstraction(1024, 0.1, 32, in_channels, [64, 64, 128])
        self.sa2 = SetAbstraction(256, 0.2, 64, 128, [128, 128, 256])
        self.sa3 = SetAbstraction(64, 0.4, 128, 256, [256, 256, 512])
        self.out_channels = [128, 256, 512]

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            points: [B, N, 3+C] where first 3 are xyz

        Returns:
            dict with "sa1", "sa2", "sa3" feature tensors
        """
        xyz = points[:, :, :3]
        features = points[:, :, 3:] if points.shape[-1] > 3 else None

        xyz1, feat1 = self.sa1(xyz, features)
        xyz2, feat2 = self.sa2(xyz1, feat1)
        xyz3, feat3 = self.sa3(xyz2, feat2)

        return {
            "sa1": feat1,  # [B, 1024, 128]
            "sa2": feat2,  # [B, 256, 256]
            "sa3": feat3,  # [B, 64, 512]
        }

    def freeze_to(self, stage: int) -> None:
        """Freeze early set abstraction layers."""
        layers = [self.sa1, self.sa2, self.sa3]
        for i, layer in enumerate(layers):
            for p in layer.parameters():
                p.requires_grad = i >= stage


def _build_pointnetpp(in_channels: int = 0, **kwargs):
    return PointNetPPBackbone(in_channels=in_channels)


BACKBONES.register_factory("pointnet++", _build_pointnetpp, category="3d", metadata={
    "description": "PointNet++ with 3 set abstraction layers",
    "valid_tasks": ["point_cloud_seg", "point_cloud_regression"],
    "input_format": "point_cloud",
    "output_format": "multi_scale_features",
    "params_M": 1.7,
})
