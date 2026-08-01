"""Measurement primitives: morphology on a *validated* mask (a first-class toolkit primitive).

``geometry on a validated mask is a valid measurement``: once a mask has been validated against
expert-scored ground truth, computing its area / perimeter / centroid / PCA axis extents is a real
phenotype measurement, not a proxy. The agent composes these for dimensional traits. They never
substitute for the CV step or manufacture a number from an unvalidated mask; the validation
invariant in CLAUDE.md still governs. An axis extent is a chord of the mask's footprint, so whether
it is the dimension a trait's definition names is the agent's call to make per trait, not a rename
away (see ``mask_geometry``'s module docstring).
"""

from __future__ import annotations

from tcip_mcp.pipelines.measurement.mask_geometry import instance_geometries, mask_geometry

__all__ = ["mask_geometry", "instance_geometries"]
