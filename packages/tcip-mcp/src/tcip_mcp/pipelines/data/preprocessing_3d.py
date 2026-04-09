"""3D point cloud preprocessing utilities.

- Ground classification (cloth simulation filter concept)
- Height normalization
- Voxel downsampling
- Canopy height model (CHM) rasterization
"""

from __future__ import annotations

import numpy as np


def ground_classify(points: np.ndarray, method: str = "simple", **kwargs) -> np.ndarray:
    """Classify ground points.

    Args:
        points: [N, 3+] array (xyz + features)
        method: "simple" (z-percentile) or "csf" (cloth simulation, requires CSF lib)

    Returns:
        Boolean mask [N] — True for ground points
    """
    z = points[:, 2]

    if method == "csf":
        try:
            import CSF
            csf = CSF.CSF()
            csf.params.bSloopSmooth = kwargs.get("smooth", False)
            csf.params.cloth_resolution = kwargs.get("resolution", 0.5)
            csf.setPointCloud(points[:, :3])
            ground_idx = CSF.VecInt()
            non_ground_idx = CSF.VecInt()
            csf.do_filtering(ground_idx, non_ground_idx)
            mask = np.zeros(len(points), dtype=bool)
            mask[list(ground_idx)] = True
            return mask
        except ImportError:
            pass  # fall through to simple

    # Simple: lowest Z percentile
    threshold = np.percentile(z, kwargs.get("percentile", 5))
    return z <= threshold + kwargs.get("tolerance", 0.3)


def height_normalize(points: np.ndarray, ground_mask: np.ndarray) -> np.ndarray:
    """Normalize Z to height above ground surface.

    Uses ground points to create a local ground model, then subtracts
    interpolated ground height from each point.
    """
    out = points.copy()
    ground_z = points[ground_mask, 2]

    if len(ground_z) == 0:
        out[:, 2] -= out[:, 2].min()
        return out

    # Simple: subtract mean ground height (for flat terrain)
    # For complex terrain, use scipy.interpolate.griddata
    mean_ground = ground_z.mean()
    out[:, 2] -= mean_ground
    return out


def voxel_downsample(points: np.ndarray, voxel_size: float = 0.05) -> np.ndarray:
    """Reduce point density by keeping one representative per voxel."""
    xyz = points[:, :3]
    voxel_ids = np.floor(xyz / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(voxel_ids, axis=0, return_index=True)
    return points[unique_idx]


def compute_chm(
    points: np.ndarray,
    resolution: float = 0.1,
    ground_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Compute canopy height model (CHM) raster from point cloud.

    Args:
        points: [N, 3+] (xyz+)
        resolution: pixel size in same units as point cloud
        ground_mask: if provided, normalize heights first

    Returns:
        (chm_array, metadata_dict) where metadata has origin, resolution, shape
    """
    if ground_mask is not None:
        points = height_normalize(points, ground_mask)

    xyz = points[:, :3]
    x_min, y_min = xyz[:, 0].min(), xyz[:, 1].min()
    x_max, y_max = xyz[:, 0].max(), xyz[:, 1].max()

    cols = int(np.ceil((x_max - x_min) / resolution)) + 1
    rows = int(np.ceil((y_max - y_min) / resolution)) + 1

    chm = np.full((rows, cols), np.nan, dtype=np.float32)

    col_idx = np.floor((xyz[:, 0] - x_min) / resolution).astype(int)
    row_idx = np.floor((xyz[:, 1] - y_min) / resolution).astype(int)
    col_idx = np.clip(col_idx, 0, cols - 1)
    row_idx = np.clip(row_idx, 0, rows - 1)

    # Max height per cell
    for r, c, z in zip(row_idx, col_idx, xyz[:, 2]):
        if np.isnan(chm[r, c]) or z > chm[r, c]:
            chm[r, c] = z

    metadata = {
        "origin": (x_min, y_min),
        "resolution": resolution,
        "shape": (rows, cols),
    }
    return chm, metadata
