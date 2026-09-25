"""Measurement primitives: morphology on a validated mask (area / perimeter / centroid / PCA axis
extents). An axis extent is a chord of the mask's footprint; see ``mask_geometry``'s module
docstring.
"""

from __future__ import annotations

from tcip_mcp.pipelines.measurement.mask_geometry import instance_geometries, mask_geometry

__all__ = ["mask_geometry", "instance_geometries"]
