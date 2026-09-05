"""mask-geometry measurement primitive: exact values on synthetic masks.

Locks that geometry on a validated mask is a real, correct measurement: a known rectangle / ellipse
yields the right area / axis extents / perimeter / centroid in pixels, a physical scale converts
px -> the caller's own unit correctly (area by the square), and degenerate / empty masks are handled
without inventing a measurement. Also locks the two firewalled resolvers this module owns
(``resolve_binarize_threshold`` / ``resolve_scale``): each is un-shippable until validated against a
reference of its own kind, so an annotations-kind stamp can never clear a physical scale.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import tcip_store as ts
from tcip_mcp.pipelines.measurement import instance_geometries, mask_geometry


# --------------------------------------------------------------------------
# Rectangle: every pixel quantity is exact
# --------------------------------------------------------------------------

def _rect_mask(h=64, w=64, r0=5, r1=24, c0=10, c1=49):
    """Solid rectangle: rows r0..r1 (inclusive), cols c0..c1 -> (r1-r0+1) x (c1-c0+1)."""
    m = np.zeros((h, w), dtype=np.uint8)
    m[r0:r1 + 1, c0:c1 + 1] = 1
    return m


def test_rectangle_pixel_measurements_exact():
    m = _rect_mask()  # 40 wide (cols 10..49), 20 tall (rows 5..24)
    g = mask_geometry(m, unit="mm")
    assert g["empty"] is False
    assert g["area_px"] == 40 * 20
    assert g["principal_axis_extent_px"] == 40.0   # major axis == the longer side
    assert g["secondary_axis_extent_px"] == 20.0   # minor axis == the shorter side
    assert g["perimeter_px"] == 2 * (40 + 20)
    assert g["centroid_px"] == pytest.approx((29.5, 14.5))
    assert g["angle_deg"] == pytest.approx(0.0, abs=1e-6)  # horizontal major axis


def test_rectangle_physical_scale_converts_px_to_mm():
    m = _rect_mask()
    g = mask_geometry(m, scale=0.5, unit="mm")
    assert g["mm_per_px"] == 0.5
    assert g["area_mm2"] == pytest.approx(800 * 0.25)   # area scales by the square of the linear scale
    assert g["principal_axis_extent_mm"] == pytest.approx(20.0)
    assert g["secondary_axis_extent_mm"] == pytest.approx(10.0)
    assert g["perimeter_mm"] == pytest.approx(60.0)
    assert g["centroid_mm"] == pytest.approx((14.75, 7.25))


def test_scale_unit_is_the_callers_fact_never_assumed_to_be_mm():
    """The unit is real data the caller states, so a cm/px scale reports cm, never a mislabeled mm.

    Replaces the former ``gsd`` alias, which silently treated a field-standard GSD (cm/px) as an
    mm/px synonym: every dimensional number came out 10x wrong under an ``_mm`` label.
    """
    m = _rect_mask()
    g = mask_geometry(m, scale=0.5, unit="cm")
    assert g["cm_per_px"] == 0.5
    assert g["area_cm2"] == pytest.approx(800 * 0.25)
    assert g["principal_axis_extent_cm"] == pytest.approx(20.0)
    assert g["secondary_axis_extent_cm"] == pytest.approx(10.0)
    assert g["perimeter_cm"] == pytest.approx(60.0)
    assert g["centroid_cm"] == pytest.approx((14.75, 7.25))
    # no mm-labeled field is invented for a cm scale, and no implicit mm default is left anywhere
    assert not any(k.endswith(("_mm", "_mm2")) or k == "mm_per_px" for k in g)


def test_no_gsd_parameter_survives_anywhere_in_the_module():
    """The naming trap is gone: no callable in the module still accepts a ``gsd``/``mm_per_px`` knob."""
    import importlib
    import inspect

    mg = importlib.import_module("tcip_mcp.pipelines.measurement.mask_geometry")
    for name, fn in vars(mg).items():
        if not inspect.isfunction(fn) or fn.__module__ != mg.__name__:
            continue
        params = inspect.signature(fn).parameters
        assert "gsd" not in params, f"{name} still takes a gsd parameter"
        assert "mm_per_px" not in params, f"{name} still takes an mm_per_px parameter"


def test_no_length_or_width_key_survives_under_any_alias():
    """A PCA-chord extent is not an anatomical length/width, and no alias keeps that claim alive.

    An alias would defeat the rename: code reading ``length_px`` would keep treating the principal-axis
    chord as the organ's real length, which is exactly the reading the axis-named keys refuse to offer.
    """
    from tcip_mcp.pipelines.measurement.mask_geometry import unit_from_value_key

    for g in (mask_geometry(_rect_mask(), unit="mm"), mask_geometry(_rect_mask(), scale=0.5, unit="mm")):
        assert not [k for k in g if k.startswith(("length", "width"))], sorted(g)
    # unit_from_value_key is vocabulary-driven (crops.yml's real declared units), not a field-name
    # whitelist: a bespoke ``length_mm``/``width_cm`` from measurement code outside this module is
    # recognized the same way mask_geometry's own fields are, so it is not the length/width alias
    # this test guards against: this module still never emits a length/width key itself (asserted
    # above), it just no longer refuses to recognize the unit on someone else's.
    assert unit_from_value_key("length_mm") == ("mm", "mm")
    assert unit_from_value_key("width_cm") == ("cm", "cm")
    assert unit_from_value_key("principal_axis_extent_mm") == ("mm", "mm")
    assert unit_from_value_key("secondary_axis_extent_cm") == ("cm", "cm")
    assert unit_from_value_key("principal_axis_extent_px") is None


# --------------------------------------------------------------------------
# Ellipse: area/axes correct within a pixel-discretization tolerance
# --------------------------------------------------------------------------

def test_ellipse_area_and_axes():
    h = w = 128
    cx, cy, a, b = 64.0, 64.0, 30.0, 15.0   # semi-axes: 30 along x (major), 15 along y (minor)
    yy, xx = np.ogrid[:h, :w]
    m = (((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1.0).astype(np.uint8)
    g = mask_geometry(m, unit="mm")
    assert g["area_px"] == pytest.approx(math.pi * a * b, rel=0.05)
    assert g["principal_axis_extent_px"] == pytest.approx(2 * a, abs=2.0)
    assert g["secondary_axis_extent_px"] == pytest.approx(2 * b, abs=2.0)
    assert g["centroid_px"] == pytest.approx((cx, cy), abs=1.0)


# --------------------------------------------------------------------------
# Degenerate / empty masks
# --------------------------------------------------------------------------

def test_empty_mask_is_handled_without_inventing_a_measurement():
    g = mask_geometry(np.zeros((32, 32), dtype=np.uint8), scale=2.0, unit="mm")
    assert g["empty"] is True
    assert g["area_px"] == 0.0
    assert g["principal_axis_extent_px"] == 0.0 and g["secondary_axis_extent_px"] == 0.0
    assert g["centroid_px"] is None and g["centroid_mm"] is None
    assert g["area_mm2"] == 0.0        # scale still applied, still zero


def test_single_row_line_is_1px_wide():
    m = np.zeros((16, 16), dtype=np.uint8)
    m[8, 3:13] = 1                      # a 10 px horizontal line
    g = mask_geometry(m, unit="mm")
    assert g["area_px"] == 10.0
    assert g["principal_axis_extent_px"] == 10.0 and g["secondary_axis_extent_px"] == 1.0
    assert g["perimeter_px"] == 2 * (10 + 1)


# --------------------------------------------------------------------------
# Input shapes: [1,H,W], torch tensor, and [N,H,W] instance stacks
# --------------------------------------------------------------------------

def test_accepts_chw_and_soft_masks():
    m = _rect_mask().astype(np.float32)[None]         # [1, H, W], float
    soft = m * 0.9                                     # soft probabilities, still >= 0.5 in-shape
    assert mask_geometry(soft, unit="mm")["area_px"] == 40 * 20
    assert mask_geometry(m * 0.4, unit="mm")["empty"] is True     # all below threshold -> empty


def test_accepts_torch_tensor():
    torch = pytest.importorskip("torch")
    g = mask_geometry(torch.from_numpy(_rect_mask()), unit="mm")
    assert g["area_px"] == 40 * 20 and g["principal_axis_extent_px"] == 40.0


def test_instance_geometries_over_a_stack():
    stack = np.stack([_rect_mask(), np.zeros((64, 64), dtype=np.uint8)])  # one real, one empty
    out = instance_geometries(stack, scale=0.5, unit="mm")
    assert len(out) == 2
    assert out[0]["area_mm2"] == pytest.approx(200.0)
    assert out[1]["empty"] is True


def test_instance_geometries_carries_the_callers_unit():
    out = instance_geometries(_rect_mask()[None], scale=0.5, unit="cm")
    assert out[0]["area_cm2"] == pytest.approx(200.0)
    assert "area_mm2" not in out[0]


# --------------------------------------------------------------------------
# The firewalled resolvers: binarize threshold (annotations) and scale (physical)
# --------------------------------------------------------------------------

def test_binarize_threshold_is_unvalidated_and_annotations_kind():
    from tcip_mcp.pipelines.measurement.mask_geometry import resolve_binarize_threshold
    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE, UnvalidatedOperatingPointError

    p = resolve_binarize_threshold()
    assert p.requires_validation is True
    assert p.validation_kind == "annotations"
    assert p.validated_against == VALIDATED_FALSE
    assert p.is_shippable is False
    with pytest.raises(UnvalidatedOperatingPointError):
        p.value
    assert p.unvalidated_value(acknowledge_unvalidated=True) == 0.5
    assert resolve_binarize_threshold(0.3).unvalidated_value(acknowledge_unvalidated=True) == 0.3


def test_resolve_scale_is_unvalidated_by_default():
    from tcip_mcp.pipelines.measurement.mask_geometry import resolve_scale
    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE, UnvalidatedOperatingPointError

    p = resolve_scale(0.5, unit="mm")
    assert p.name == "scale_mm_per_px"
    assert p.source == "explicit"
    assert p.requires_validation is True
    assert p.validation_kind == "physical"
    assert p.validated_against == VALIDATED_FALSE
    assert p.is_shippable is False
    with pytest.raises(UnvalidatedOperatingPointError):
        p.value
    assert p.unvalidated_value(acknowledge_unvalidated=True) == 0.5
    assert resolve_scale(2.0, unit="cm").name == "scale_cm_per_px"
    assert resolve_scale(unit="mm").source == "default"


def test_resolve_scale_capture_scoping_is_the_callers_fact():
    from tcip_mcp.pipelines.measurement.mask_geometry import resolve_scale

    unscoped = resolve_scale(0.5, unit="mm")
    assert unscoped.capture_scoped is False and unscoped.capture_id is None
    scoped = resolve_scale(0.5, unit="mm", capture_id="2026-02-10_plot7")
    assert scoped.capture_scoped is True and scoped.capture_id == "2026-02-10_plot7"


def test_an_annotations_reference_can_never_validate_a_physical_scale():
    """The point of the validation_kind split: a held-out-GT stamp does not make a scale shippable.

    A scale is a physical fact; only a physical-measurement reference clears it. Stamping it with an
    annotations-kind reference (the reference a conf threshold earns) leaves it un-shippable and
    ``.value`` still raising, so a wrong scale can never launder itself into a dimensional phenotype
    through the count operating point's paperwork.
    """
    import dataclasses

    from tcip_mcp.pipelines.measurement.mask_geometry import resolve_scale
    from tcip_mcp.pipelines.resolution import (
        VALIDATED_HELD_OUT,
        VALIDATED_PHYSICAL_MEASUREMENT,
        VALIDATED_REVIEW_CONFIRMED,
        UnvalidatedOperatingPointError,
        accepted_references,
    )

    base = resolve_scale(0.5, unit="mm")
    for wrong_kind_ref in (VALIDATED_HELD_OUT, VALIDATED_REVIEW_CONFIRMED):
        stamped = dataclasses.replace(base, validated_against=wrong_kind_ref)
        assert wrong_kind_ref not in accepted_references("physical")
        assert stamped.is_shippable is False
        with pytest.raises(UnvalidatedOperatingPointError):
            stamped.value

    # ...and the rail still admits the legitimate case: a real physical reference ships.
    validated = dataclasses.replace(base, validated_against=VALIDATED_PHYSICAL_MEASUREMENT)
    assert validated.is_shippable is True
    assert validated.value == 0.5
    # symmetrically, a physical reference cannot clear the annotations-kind binarize threshold
    from tcip_mcp.pipelines.measurement.mask_geometry import resolve_binarize_threshold

    thr = dataclasses.replace(resolve_binarize_threshold(),
                              validated_against=VALIDATED_PHYSICAL_MEASUREMENT)
    assert thr.is_shippable is False


# resolve_scale.json: the sidecar a delivery door reconciles a physical scale from.

def _write_scale_sidecar(path, *, validated_against, value=0.05, unit="mm", capture_id=None):
    """Plant a resolve_scale.json stamp through the seam, bypassing the writer-side claim rail."""
    from tcip_mcp.pipelines.resolution import sidecar_key

    path.mkdir(parents=True, exist_ok=True)
    stamp = {
        "validated": validated_against == "physical_measurement",
        "operating_point": {
            "scale": {
                "value": value, "unit": unit, "capture_id": capture_id,
                "requires_validation": True, "validation_kind": "physical",
                "validated_against": validated_against,
            },
        },
    }
    key = sidecar_key(path, "resolve_scale")
    with ts.transaction(key) as txn:
        txn.write(key, stamp)
    return str(path)


def _write_bound_scale_sidecar(path, dataset_root, *, value=0.05, unit="mm", capture_id=None,
                               experiment_id="exp-scale", trait="bud_opening"):
    """A resolve_scale.json sidecar genuinely answered for by a validation record.

    ``covered_buckets`` keys a bucket by its dataset-relative path, resolved via
    ``dataset_layout.dataset_root_of``, which only recognizes a path holding one of
    ``annotations``/``predictions``/``images``; the bucket is nested under ``predictions`` here so
    that resolution (and the caller's own ``path`` naming) actually agree. The bucket itself carries
    no prediction stems, so ``images_dir`` names a directory with nothing to resolve against; the
    imagery digest is over an empty stem set regardless.
    """
    from tcip_mcp.pipelines.resolution import VALIDATED_PHYSICAL_MEASUREMENT
    from tests._binding_fixtures import write_bound_sidecar

    bucket = Path(dataset_root) / "predictions" / Path(path).name
    stamp = {
        "validated": True, "trait": trait,
        "operating_point": {
            "scale": {
                "value": value, "unit": unit, "capture_id": capture_id,
                "requires_validation": True, "validation_kind": "physical",
                "validated_against": VALIDATED_PHYSICAL_MEASUREMENT,
            },
        },
    }
    bucket.mkdir(parents=True, exist_ok=True)
    write_bound_sidecar(bucket, stamp, document="resolve_scale", dataset_root=dataset_root,
                        images_dir=Path(dataset_root) / "images", experiment_id=experiment_id)
    return str(bucket)


def test_read_scale_sidecar_round_trips_a_validated_fixture(tmp_path):
    from tcip_mcp.pipelines.resolution import VALIDATED_PHYSICAL_MEASUREMENT, read_scale_sidecar

    d = _write_scale_sidecar(tmp_path / "preds", validated_against=VALIDATED_PHYSICAL_MEASUREMENT)
    sc = read_scale_sidecar(d)
    assert sc["validated"] is True
    assert sc["operating_point"]["scale"]["validated_against"] == VALIDATED_PHYSICAL_MEASUREMENT


def test_read_scale_sidecar_missing_file_returns_none(tmp_path):
    from tcip_mcp.pipelines.resolution import read_scale_sidecar

    (tmp_path / "preds").mkdir()
    assert read_scale_sidecar(str(tmp_path / "preds")) is None


def test_reconcile_scale_validity_ships_when_validated(tmp_path):
    from tcip_mcp.pipelines.resolution import (
        VALIDATED_PHYSICAL_MEASUREMENT,
        reconcile_scale_validity,
    )

    d = _write_bound_scale_sidecar(tmp_path / "preds", tmp_path)
    recon = reconcile_scale_validity([d], unit="mm", trait="bud_opening", images_dir=tmp_path / "images")
    assert recon["operative"] is True
    assert recon["validated"] == VALIDATED_PHYSICAL_MEASUREMENT
    assert recon["unvalidated_buckets"] == []


def test_reconcile_scale_validity_missing_sidecar_floors(tmp_path):
    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE, reconcile_scale_validity

    (tmp_path / "preds").mkdir()
    recon = reconcile_scale_validity([str(tmp_path / "preds")], unit="mm", trait="bud_opening",
                                    images_dir=tmp_path / "images")
    assert recon["operative"] is True
    assert recon["validated"] == VALIDATED_FALSE
    assert recon["unvalidated_buckets"] == [str(tmp_path / "preds")]


def test_reconcile_scale_validity_no_pred_dirs_is_not_operative():
    from tcip_mcp.pipelines.resolution import reconcile_scale_validity

    recon = reconcile_scale_validity([], unit="mm", trait="bud_opening", images_dir="unused")
    assert recon["operative"] is False
    assert recon["validated"] is None


def test_reconcile_scale_validity_an_annotations_reference_never_clears_it(tmp_path):
    from tcip_mcp.pipelines.resolution import (
        VALIDATED_FALSE,
        VALIDATED_HELD_OUT,
        reconcile_scale_validity,
    )

    d = _write_scale_sidecar(tmp_path / "preds", validated_against=VALIDATED_HELD_OUT)
    recon = reconcile_scale_validity([d], unit="mm", trait="bud_opening", images_dir=tmp_path / "images")
    assert recon["validated"] == VALIDATED_FALSE


def test_reconcile_scale_validity_capture_id_mismatch_floors(tmp_path):
    from tcip_mcp.pipelines.resolution import (
        VALIDATED_FALSE,
        reconcile_scale_validity,
    )

    d = _write_bound_scale_sidecar(tmp_path / "preds", tmp_path, capture_id="2026-02-10_plot7")
    recon = reconcile_scale_validity([d], unit="mm", trait="bud_opening", images_dir=tmp_path / "images",
                                    capture_id="2026-02-10_plot9")
    assert recon["validated"] == VALIDATED_FALSE
    assert recon["unvalidated_buckets"] == [d]


def test_reconcile_scale_validity_capture_id_match_ships(tmp_path):
    from tcip_mcp.pipelines.resolution import (
        VALIDATED_PHYSICAL_MEASUREMENT,
        reconcile_scale_validity,
    )

    d = _write_bound_scale_sidecar(tmp_path / "preds", tmp_path, capture_id="2026-02-10_plot7")
    recon = reconcile_scale_validity([d], unit="mm", trait="bud_opening", images_dir=tmp_path / "images",
                                    capture_id="2026-02-10_plot7")
    assert recon["validated"] == VALIDATED_PHYSICAL_MEASUREMENT


def test_reconcile_scale_validity_unscoped_sidecar_applies_to_any_capture(tmp_path):
    """A scale never resolved with a capture_id makes no capture-specific claim, so it is not floored
    just because the caller happens to be asking on behalf of a particular capture."""
    from tcip_mcp.pipelines.resolution import (
        VALIDATED_PHYSICAL_MEASUREMENT,
        reconcile_scale_validity,
    )

    d = _write_bound_scale_sidecar(tmp_path / "preds", tmp_path, capture_id=None)
    recon = reconcile_scale_validity([d], unit="mm", trait="bud_opening", images_dir=tmp_path / "images",
                                    capture_id="2026-02-10_plot7")
    assert recon["validated"] == VALIDATED_PHYSICAL_MEASUREMENT


def test_reconcile_scale_validity_asserted_can_only_lower(tmp_path):
    from tcip_mcp.pipelines.resolution import (
        VALIDATED_FALSE,
        reconcile_scale_validity,
    )

    d = _write_bound_scale_sidecar(tmp_path / "preds", tmp_path)
    recon = reconcile_scale_validity([d], unit="mm", trait="bud_opening", images_dir=tmp_path / "images", asserted=VALIDATED_FALSE)
    assert recon["validated"] == VALIDATED_FALSE


def test_reconcile_scale_validity_unit_mismatch_floors(tmp_path):
    """A scale stamped in one linear unit cannot clear a delivery stated in another
    (count-delivery-door design section 2, rule 4): centimetres cannot answer for millimetres."""
    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE, reconcile_scale_validity

    d = _write_bound_scale_sidecar(tmp_path / "preds", tmp_path, unit="cm")
    recon = reconcile_scale_validity([d], unit="mm", trait="bud_opening", images_dir=tmp_path / "images")
    assert recon["validated"] == VALIDATED_FALSE
    assert recon["unvalidated_buckets"] == [d]


def test_reconcile_scale_validity_trait_mismatch_floors(tmp_path):
    """A scale earned for one trait does not answer for a delivery of another (the same trait
    binding P4-86 gives the count/ordinal/regression reconcilers, extended to the scale document)."""
    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE, reconcile_scale_validity

    d = _write_bound_scale_sidecar(tmp_path / "preds", tmp_path)
    recon = reconcile_scale_validity([d], unit="mm", trait="a_different_trait", images_dir=tmp_path / "images")
    assert recon["validated"] == VALIDATED_FALSE
    assert recon["unvalidated_buckets"] == [d]
