"""mask-geometry measurement primitive: exact values on synthetic masks.

A known rectangle or ellipse yields the right area, axis extents, perimeter and centroid in
pixels, a physical scale converts pixels to the stated unit (area by the square), and
degenerate and empty masks yield no invented measurement.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tcip_mcp.pipelines.measurement import instance_geometries, mask_geometry
from tests._training_values import MASK_BINARIZE

# ── rectangle: every pixel quantity is exact ─────────────────────────────


def _rect_mask(h=64, w=64, r0=5, r1=24, c0=10, c1=49):
    """Return a solid rectangle mask over rows r0..r1 and cols c0..c1, both inclusive."""
    from tests._producer_fixtures import painted_array

    return painted_array(w, h, [((c0, r0, c1 + 1, r1 + 1), 1)])


def test_rectangle_pixel_measurements_exact():
    m = _rect_mask()  # 40 wide (cols 10..49), 20 tall (rows 5..24)
    g = mask_geometry(m, unit="mm", threshold=MASK_BINARIZE)
    assert g["empty"] is False
    assert g["area_px"] == 40 * 20
    assert (
        g["principal_axis_extent_px"] == 40.0
    )  # major axis == the longer side
    assert (
        g["secondary_axis_extent_px"] == 20.0
    )  # minor axis == the shorter side
    assert g["perimeter_px"] == 2 * (40 + 20)
    assert g["centroid_px"] == pytest.approx((29.5, 14.5))
    assert g["angle_deg"] == pytest.approx(
        0.0, abs=1e-6
    )  # horizontal major axis


def test_rectangle_physical_scale_converts_px_to_mm():
    m = _rect_mask()
    g = mask_geometry(m, scale=0.5, unit="mm", threshold=MASK_BINARIZE)
    assert g["mm_per_px"] == 0.5
    # area scales by the square of the linear scale
    assert g["area_mm2"] == pytest.approx(800 * 0.25)
    assert g["principal_axis_extent_mm"] == pytest.approx(20.0)
    assert g["secondary_axis_extent_mm"] == pytest.approx(10.0)
    assert g["perimeter_mm"] == pytest.approx(60.0)
    assert g["centroid_mm"] == pytest.approx((14.75, 7.25))


def test_scale_unit_is_the_callers_fact_never_assumed_to_be_mm():
    """A cm/px scale reports area in cm² and axis extents, perimeter and centroid in cm.

    The result carries no mm-labeled field.
    """
    m = _rect_mask()
    g = mask_geometry(m, scale=0.5, unit="cm", threshold=MASK_BINARIZE)
    assert g["cm_per_px"] == 0.5
    assert g["area_cm2"] == pytest.approx(800 * 0.25)
    assert g["principal_axis_extent_cm"] == pytest.approx(20.0)
    assert g["secondary_axis_extent_cm"] == pytest.approx(10.0)
    assert g["perimeter_cm"] == pytest.approx(60.0)
    assert g["centroid_cm"] == pytest.approx((14.75, 7.25))
    # no mm-labeled field is invented for a cm scale, and no implicit mm default is left anywhere
    assert not any(k.endswith(("_mm", "_mm2")) or k == "mm_per_px" for k in g)


def test_no_gsd_parameter_survives_anywhere_in_the_module():
    """No callable in the mask geometry module takes a ``gsd`` or ``mm_per_px`` parameter."""
    import importlib
    import inspect

    mg = importlib.import_module(
        "tcip_mcp.pipelines.measurement.mask_geometry"
    )
    for name, fn in vars(mg).items():
        if not inspect.isfunction(fn) or fn.__module__ != mg.__name__:
            continue
        params = inspect.signature(fn).parameters
        assert "gsd" not in params, f"{name} still takes a gsd parameter"
        assert "mm_per_px" not in params, (
            f"{name} still takes an mm_per_px parameter"
        )


def test_no_length_or_width_key_survives_under_any_alias():
    """The geometry carries no length or width key, and unit lookup reads such keys by vocabulary.

    Over two geometries no key starts with ``length`` or ``width``; ``unit_from_value_key`` reads
    mm or cm off ``length_mm``, ``width_cm`` and the two axis-extent keys, and no unit off a px
    key.
    """
    from tcip_mcp.pipelines.measurement.mask_geometry import (
        unit_from_value_key,
    )

    for g in (
        mask_geometry(_rect_mask(), unit="mm", threshold=MASK_BINARIZE),
        mask_geometry(
            _rect_mask(), scale=0.5, unit="mm", threshold=MASK_BINARIZE
        ),
    ):
        assert not [k for k in g if k.startswith(("length", "width"))], sorted(
            g
        )
    assert unit_from_value_key("length_mm") == ("mm", "mm")
    assert unit_from_value_key("width_cm") == ("cm", "cm")
    assert unit_from_value_key("principal_axis_extent_mm") == ("mm", "mm")
    assert unit_from_value_key("secondary_axis_extent_cm") == ("cm", "cm")
    assert unit_from_value_key("principal_axis_extent_px") is None


# ── ellipse: area/axes correct within a pixel-discretization tolerance ─────


def test_ellipse_area_and_axes():
    h = w = 128
    cx, cy, a, b = (
        64.0,
        64.0,
        30.0,
        15.0,
    )  # semi-axes: 30 along x (major), 15 along y (minor)
    yy, xx = np.ogrid[:h, :w]
    m = (((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1.0).astype(np.uint8)
    g = mask_geometry(m, unit="mm", threshold=MASK_BINARIZE)
    assert g["area_px"] == pytest.approx(math.pi * a * b, rel=0.05)
    assert g["principal_axis_extent_px"] == pytest.approx(2 * a, abs=2.0)
    assert g["secondary_axis_extent_px"] == pytest.approx(2 * b, abs=2.0)
    assert g["centroid_px"] == pytest.approx((cx, cy), abs=1.0)


# ── degenerate / empty masks ──────────────────────────────────────────────


def test_empty_mask_is_handled_without_inventing_a_measurement():
    g = mask_geometry(
        np.zeros((32, 32), dtype=np.uint8),
        scale=2.0,
        unit="mm",
        threshold=MASK_BINARIZE,
    )
    assert g["empty"] is True
    assert g["area_px"] == 0.0
    assert (
        g["principal_axis_extent_px"] == 0.0
        and g["secondary_axis_extent_px"] == 0.0
    )
    assert g["centroid_px"] is None and g["centroid_mm"] is None
    assert g["area_mm2"] == 0.0  # scale still applied, still zero


def test_single_row_line_is_1px_wide():
    m = _rect_mask(
        h=16, w=16, r0=8, r1=8, c0=3, c1=12
    )  # a 10 px horizontal line
    g = mask_geometry(m, unit="mm", threshold=MASK_BINARIZE)
    assert g["area_px"] == 10.0
    assert (
        g["principal_axis_extent_px"] == 10.0
        and g["secondary_axis_extent_px"] == 1.0
    )
    assert g["perimeter_px"] == 2 * (10 + 1)


# ── input shapes: [1,H,W], torch tensor, and [N,H,W] instance stacks ──────


def test_accepts_chw_and_soft_masks_cut_at_the_stated_threshold():
    soft = _rect_mask().astype(np.float32)[None] * 0.6
    assert mask_geometry(soft, unit="mm", threshold=0.6)["area_px"] == 40 * 20
    assert mask_geometry(soft, unit="mm", threshold=0.7)["empty"] is True


def test_accepts_torch_tensor():
    torch = pytest.importorskip("torch")
    g = mask_geometry(
        torch.from_numpy(_rect_mask()), unit="mm", threshold=MASK_BINARIZE
    )
    assert g["area_px"] == 40 * 20 and g["principal_axis_extent_px"] == 40.0


def test_instance_geometries_over_a_stack():
    stack = np.stack(
        [_rect_mask(), np.zeros((64, 64), dtype=np.uint8)]
    )  # one real, one empty
    out = instance_geometries(
        stack, scale=0.5, unit="mm", threshold=MASK_BINARIZE
    )
    assert len(out) == 2
    assert out[0]["area_mm2"] == pytest.approx(200.0)
    assert out[1]["empty"] is True


def test_instance_geometries_carries_the_callers_unit():
    out = instance_geometries(
        _rect_mask()[None], scale=0.5, unit="cm", threshold=MASK_BINARIZE
    )
    assert out[0]["area_cm2"] == pytest.approx(200.0)
    assert "area_mm2" not in out[0]


# ── the binarize threshold the caller states ──────────────────────────────


def test_an_unstated_threshold_is_refused_by_the_signatures():
    for measure in (mask_geometry, instance_geometries):
        with pytest.raises(TypeError, match="threshold"):
            measure(_rect_mask(), unit="mm")  # type: ignore[call-arg]
