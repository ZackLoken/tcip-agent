"""The parameter-resolution currency + firewall (derive-don't-pin foundation)."""

from __future__ import annotations

import pytest

from tcip_mcp.pipelines.resolution import (
    ResolvedBundle,
    ResolvedParam,
    UnvalidatedOperatingPointError,
    VALIDATED_FALSE,
    VALIDATED_HELD_OUT,
    dataset_hash,
    derived,
    validate_resolved_bundle,
)
from tcip_mcp.traits import CATKIN, TraitUnknownError, get_trait, registered_traits


# --- the firewall: an unvalidated calibration op-point is un-consumable ---

def test_unvalidated_calibration_value_raises():
    p = derived("conf", 0.4, derivation_class="calibration", derived_from="sweep",
                validated_vs_gt=VALIDATED_FALSE, sweep={"curve": []})
    assert not p.is_shippable
    with pytest.raises(UnvalidatedOperatingPointError):
        _ = p.value


def test_validated_heldout_calibration_value_ok():
    p = derived("conf", 0.4, derivation_class="calibration", derived_from="sweep",
                validated_vs_gt=VALIDATED_HELD_OUT)
    assert p.is_shippable
    assert p.value == 0.4


def test_non_calibration_value_always_ok():
    # facts (deterministic) and knobs (engineering) ship regardless of validated_vs_gt
    assert derived("num_classes", 2, derivation_class="deterministic", derived_from="labels").value == 2
    from tcip_mcp.pipelines.resolution import default
    assert default("lr", 1e-3).value == 1e-3


def test_unvalidated_value_requires_acknowledgement():
    p = derived("conf", 0.4, derivation_class="calibration", derived_from="sweep",
                validated_vs_gt=VALIDATED_FALSE)
    with pytest.raises(UnvalidatedOperatingPointError):
        p.unvalidated_value(acknowledge_unvalidated=False)
    assert p.unvalidated_value(acknowledge_unvalidated=True) == 0.4


def test_resolvedparam_rejects_bad_vocab():
    with pytest.raises(ValueError):
        ResolvedParam(name="x", _raw=1, source="bogus", derivation_class="calibration")
    with pytest.raises(ValueError):
        ResolvedParam(name="x", _raw=1, source="derived", derivation_class="bogus")


# --- bundle shippability + provenance ---

def test_bundle_shippable_only_when_all_calibration_validated():
    unval = derived("conf", 0.4, derivation_class="calibration", derived_from="sweep",
                    validated_vs_gt=VALIDATED_FALSE)
    val = derived("conf", 0.4, derivation_class="calibration", derived_from="sweep",
                  validated_vs_gt=VALIDATED_HELD_OUT)
    fact = derived("max_dets", 300, derivation_class="distribution", derived_from="p99")
    assert not ResolvedBundle("catkin", "h1", {"conf": unval, "max_dets": fact}).is_shippable
    assert ResolvedBundle("catkin", "h1", {"conf": val, "max_dets": fact}).is_shippable


def test_dataset_scoped_calibration_not_inherited_across_hash():
    p = derived("conf", 0.4, derivation_class="calibration", derived_from="sweep",
                validated_vs_gt=VALIDATED_HELD_OUT, dataset_scoped=True, dataset_hash="AAAA")
    b = ResolvedBundle("catkin", "AAAA", {"conf": p})
    assert b.shippable_issues(target_dataset_hash="AAAA") == []
    issues = b.shippable_issues(target_dataset_hash="BBBB")
    assert any("never inherit" in s for s in issues)


def test_provenance_roundtrip_is_serializable():
    import json
    b = ResolvedBundle("catkin", "h1", {
        "conf": derived("conf", 0.4, derivation_class="calibration", derived_from="sweep",
                        validated_vs_gt=VALIDATED_HELD_OUT),
    })
    json.dumps(b.to_provenance())  # must not raise


# --- dataset identity ---

def test_dataset_hash_content_addressed(tmp_path):
    d = tmp_path / "labels"
    d.mkdir()
    (d / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (d / "b.txt").write_text("")  # empty = valid negative, still contributes to identity
    h1 = dataset_hash(d)
    h2 = dataset_hash(d)
    assert h1 == h2
    (d / "a.txt").write_text("1 0.5 0.5 0.1 0.1\n")  # change GT content
    assert dataset_hash(d) != h1


# --- validate_resolved_bundle live checks ---

def test_validate_in_chans_vs_probed_bands():
    b = ResolvedBundle("catkin", "h1", {
        "in_chans": derived("in_chans", 3, derivation_class="deterministic", derived_from="raster"),
    })
    assert validate_resolved_bundle(b, probed_channels=3) == []
    issues = validate_resolved_bundle(b, probed_channels=4)
    assert any("in_chans" in s for s in issues)


def test_validate_eval_vs_inference_operating_point_mismatch():
    def bundle(conf):
        return ResolvedBundle("catkin", "h1", {
            "conf": derived("conf", conf, derivation_class="calibration", derived_from="sweep",
                            validated_vs_gt=VALIDATED_HELD_OUT),
        })
    ev, inf = bundle(0.25), bundle(0.5)
    issues = validate_resolved_bundle(ev, inference_bundle=inf)
    assert any("operating point" in s for s in issues)
    assert validate_resolved_bundle(bundle(0.4), inference_bundle=bundle(0.4)) == []


def test_validate_export_refuses_unvalidated():
    b = ResolvedBundle("catkin", "h1", {
        "conf": derived("conf", 0.4, derivation_class="calibration", derived_from="sweep",
                        validated_vs_gt=VALIDATED_FALSE),
    })
    issues = validate_resolved_bundle(b, for_export=True)
    assert any("shippable" in s or "not validated" in s for s in issues)


# --- trait knowledge ---

def test_catkin_trait_semantics():
    t = get_trait("catkin")
    assert t is CATKIN
    assert t.count_objective == "count_unbiased"
    assert t.localization == "center_match"
    assert t.positive_class_id == 1
    assert t.positive_is_texture is True
    assert t.milestone_fractions == (0.05, 0.50, 0.95)
    assert t.milestone_on == "positive_fraction"


def test_unknown_trait_lists_available():
    with pytest.raises(TraitUnknownError) as exc:
        get_trait("banana")
    assert "catkin" in str(exc.value)
    assert "catkin" in registered_traits()
