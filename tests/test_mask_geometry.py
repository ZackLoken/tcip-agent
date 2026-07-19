"""S4a — mask-geometry measurement primitive: exact values on synthetic masks.

Locks that geometry on a validated mask is a real, correct measurement: a known rectangle / ellipse
yields the right area / length / width / perimeter / centroid in pixels, a physical scale converts
px -> mm correctly (area by the square), and degenerate / empty masks are handled without inventing
a measurement.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tcip_mcp.pipelines.measurement import instance_geometries, mask_geometry


# --------------------------------------------------------------------------
# Rectangle — every pixel quantity is exact
# --------------------------------------------------------------------------

def _rect_mask(h=64, w=64, r0=5, r1=24, c0=10, c1=49):
    """Solid rectangle: rows r0..r1 (inclusive), cols c0..c1 -> (r1-r0+1) x (c1-c0+1)."""
    m = np.zeros((h, w), dtype=np.uint8)
    m[r0:r1 + 1, c0:c1 + 1] = 1
    return m


def test_rectangle_pixel_measurements_exact():
    m = _rect_mask()  # 40 wide (cols 10..49), 20 tall (rows 5..24)
    g = mask_geometry(m)
    assert g["empty"] is False
    assert g["area_px"] == 40 * 20
    assert g["length_px"] == 40.0        # major axis == the longer side
    assert g["width_px"] == 20.0         # minor axis == the shorter side
    assert g["perimeter_px"] == 2 * (40 + 20)
    assert g["centroid_px"] == pytest.approx((29.5, 14.5))
    assert g["angle_deg"] == pytest.approx(0.0, abs=1e-6)  # horizontal major axis


def test_rectangle_physical_scale_converts_px_to_mm():
    m = _rect_mask()
    g = mask_geometry(m, mm_per_px=0.5)
    assert g["mm_per_px"] == 0.5
    assert g["area_mm2"] == pytest.approx(800 * 0.25)   # area scales by the square of the linear scale
    assert g["length_mm"] == pytest.approx(20.0)
    assert g["width_mm"] == pytest.approx(10.0)
    assert g["perimeter_mm"] == pytest.approx(60.0)
    assert g["centroid_mm"] == pytest.approx((14.75, 7.25))


def test_gsd_alias_equals_mm_per_px():
    m = _rect_mask()
    assert mask_geometry(m, gsd=0.5)["area_mm2"] == mask_geometry(m, mm_per_px=0.5)["area_mm2"]
    with pytest.raises(ValueError):
        mask_geometry(m, mm_per_px=0.5, gsd=0.5)   # ambiguous scale is rejected


# --------------------------------------------------------------------------
# Ellipse — area/axes correct within a pixel-discretization tolerance
# --------------------------------------------------------------------------

def test_ellipse_area_and_axes():
    h = w = 128
    cx, cy, a, b = 64.0, 64.0, 30.0, 15.0   # semi-axes: 30 along x (major), 15 along y (minor)
    yy, xx = np.ogrid[:h, :w]
    m = (((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1.0).astype(np.uint8)
    g = mask_geometry(m)
    assert g["area_px"] == pytest.approx(math.pi * a * b, rel=0.05)
    assert g["length_px"] == pytest.approx(2 * a, abs=2.0)
    assert g["width_px"] == pytest.approx(2 * b, abs=2.0)
    assert g["centroid_px"] == pytest.approx((cx, cy), abs=1.0)


# --------------------------------------------------------------------------
# Degenerate / empty masks
# --------------------------------------------------------------------------

def test_empty_mask_is_handled_without_inventing_a_measurement():
    g = mask_geometry(np.zeros((32, 32), dtype=np.uint8), mm_per_px=2.0)
    assert g["empty"] is True
    assert g["area_px"] == 0.0 and g["length_px"] == 0.0 and g["width_px"] == 0.0
    assert g["centroid_px"] is None and g["centroid_mm"] is None
    assert g["area_mm2"] == 0.0        # scale still applied, still zero


def test_single_row_line_is_1px_wide():
    m = np.zeros((16, 16), dtype=np.uint8)
    m[8, 3:13] = 1                      # a 10 px horizontal line
    g = mask_geometry(m)
    assert g["area_px"] == 10.0
    assert g["length_px"] == 10.0 and g["width_px"] == 1.0
    assert g["perimeter_px"] == 2 * (10 + 1)


# --------------------------------------------------------------------------
# Input shapes: [1,H,W], torch tensor, and [N,H,W] instance stacks
# --------------------------------------------------------------------------

def test_accepts_chw_and_soft_masks():
    m = _rect_mask().astype(np.float32)[None]         # [1, H, W], float
    soft = m * 0.9                                     # soft probabilities, still >= 0.5 in-shape
    assert mask_geometry(soft)["area_px"] == 40 * 20
    assert mask_geometry(m * 0.4)["empty"] is True     # all below threshold -> empty


def test_accepts_torch_tensor():
    torch = pytest.importorskip("torch")
    g = mask_geometry(torch.from_numpy(_rect_mask()))
    assert g["area_px"] == 40 * 20 and g["length_px"] == 40.0


def test_instance_geometries_over_a_stack():
    stack = np.stack([_rect_mask(), np.zeros((64, 64), dtype=np.uint8)])  # one real, one empty
    out = instance_geometries(stack, mm_per_px=0.5)
    assert len(out) == 2
    assert out[0]["area_mm2"] == pytest.approx(200.0)
    assert out[1]["empty"] is True
