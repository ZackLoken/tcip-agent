"""Measurement primitives — morphology on a *validated* mask (a first-class toolkit primitive).

``geometry on a validated mask is a valid measurement``: once a mask has been validated against
expert-scored ground truth, computing its area / length / width / perimeter / centroid is a real
phenotype measurement, not a proxy. The agent composes these for dimensional traits (organ area,
length, width). They never substitute for the CV step or manufacture a number from an unvalidated
mask — the validation invariant in CLAUDE.md still governs.
"""

from __future__ import annotations

from tcip_mcp.pipelines.measurement.mask_geometry import instance_geometries, mask_geometry

__all__ = ["mask_geometry", "instance_geometries"]
