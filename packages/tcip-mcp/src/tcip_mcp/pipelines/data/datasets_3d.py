"""Point cloud dataset for LiDAR .las/.laz files.

Loads point clouds, applies optional preprocessing (ground classification,
height normalization, voxel downsampling), and returns (points, targets).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class PointCloudDataset(Dataset):
    """Load .las point clouds for 3D analysis tasks.

    Targets depend on task:
      - point_cloud_seg: per-point labels
      - point_cloud_regression: per-cloud scalar (height, volume, biomass)
    """

    def __init__(
        self,
        las_dir: str,
        stems: list[str] | None = None,
        labels_csv: str | None = None,
        task: str = "point_cloud_regression",
        max_points: int = 8192,
        normalize_height: bool = True,
        voxel_size: float | None = None,
    ) -> None:
        self.las_dir = Path(las_dir)
        self.task = task
        self.max_points = max_points
        self.normalize_height = normalize_height
        self.voxel_size = voxel_size

        self.stems = stems or sorted(
            f.stem for f in self.las_dir.iterdir()
            if f.suffix.lower() in {".las", ".laz"}
        )

        self._labels: dict[str, float] = {}
        if labels_csv:
            import csv
            with open(labels_csv) as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) >= 2:
                        self._labels[row[0].strip()] = float(row[1].strip())

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        stem = self.stems[idx]
        points = self._load_las(stem)

        if self.voxel_size is not None:
            points = _voxel_downsample(points, self.voxel_size)

        if self.normalize_height:
            points = _height_normalize_simple(points)

        # Subsample or pad to max_points
        points = _fixed_size(points, self.max_points)

        target: dict[str, Any] = {}
        if self.task == "point_cloud_regression" and stem in self._labels:
            target["value"] = self._labels[stem]
        elif self.task == "point_cloud_seg":
            # Per-point labels would come from classified .las
            target["point_labels"] = torch.zeros(len(points), dtype=torch.int64)

        return torch.tensor(points, dtype=torch.float32), target

    def _load_las(self, stem: str) -> np.ndarray:
        """Load .las file and return [N, 3+] array (xyz + features)."""
        try:
            import laspy
        except ImportError:
            raise ImportError("laspy is required for point cloud data. Install: pip install laspy")

        for ext in (".las", ".laz"):
            p = self.las_dir / f"{stem}{ext}"
            if p.exists():
                las = laspy.read(str(p))
                xyz = np.stack([las.x, las.y, las.z], axis=-1).astype(np.float32)
                # Include intensity if available
                features = [xyz]
                if hasattr(las, "intensity"):
                    intensity = np.array(las.intensity, dtype=np.float32).reshape(-1, 1)
                    intensity /= max(intensity.max(), 1.0)
                    features.append(intensity)
                return np.concatenate(features, axis=-1)

        raise FileNotFoundError(f"No .las/.laz for stem: {stem}")


def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Simple voxel downsampling — one point per voxel."""
    xyz = points[:, :3]
    voxel_ids = np.floor(xyz / voxel_size).astype(np.int32)
    _, unique_idx = np.unique(voxel_ids, axis=0, return_index=True)
    return points[unique_idx]


def _height_normalize_simple(points: np.ndarray) -> np.ndarray:
    """Normalize Z to height above minimum (simple ground approx)."""
    out = points.copy()
    out[:, 2] -= out[:, 2].min()
    return out


def _fixed_size(points: np.ndarray, n: int) -> np.ndarray:
    """Subsample or pad points to exactly n points."""
    if len(points) >= n:
        idx = np.random.choice(len(points), n, replace=False)
        return points[idx]
    else:
        pad = np.zeros((n - len(points), points.shape[1]), dtype=points.dtype)
        return np.concatenate([points, pad])
