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


def test_review_confirmed_calibration_is_shippable():
    from tcip_mcp.pipelines.resolution import VALIDATED_REVIEW_CONFIRMED
    p = derived("conf", 0.4, derivation_class="calibration",
                derived_from="count-unbiased center-match sweep over review verdicts",
                validated_vs_gt=VALIDATED_REVIEW_CONFIRMED)
    assert p.is_shippable  # a review-confirmed reference ships (distinct flag, same gate)
    assert p.value == 0.4


# --- W1-R3: reconcile the delivery gate against on-disk operating_point.json (T5-3) ---

def _bucket(tmp_path, name, *, validated, ref=VALIDATED_HELD_OUT, conf=0.6):
    import json
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "operating_point.json").write_text(json.dumps({
        "validated": validated,
        "operating_point": {"conf": {"value": conf, "validated_vs_gt": ref if validated else "false"}},
    }), encoding="utf-8")
    return str(d)


def test_reconcile_missing_sidecar_floors_to_false(tmp_path):
    from tcip_mcp.pipelines.resolution import reconcile_operating_point_validity
    # A caller asserting validated cannot open the gate when no sidecar backs it (the T5-3 hole).
    r = reconcile_operating_point_validity([str(tmp_path / "nope")], asserted=VALIDATED_HELD_OUT)
    assert r["validated"] == VALIDATED_FALSE
    assert r["missing_sidecars"] == [str(tmp_path / "nope")]


def test_reconcile_all_validated_on_disk(tmp_path):
    from tcip_mcp.pipelines.resolution import reconcile_operating_point_validity
    dirs = [_bucket(tmp_path, "d1", validated=True), _bucket(tmp_path, "d2", validated=True)]
    r = reconcile_operating_point_validity(dirs)  # no caller assertion needed
    assert r["validated"] == VALIDATED_HELD_OUT
    assert r["on_disk_validated"] is True
    assert r["conf"] == 0.6


def test_reconcile_one_unvalidated_bucket_floors_whole_curve(tmp_path):
    from tcip_mcp.pipelines.resolution import reconcile_operating_point_validity
    dirs = [_bucket(tmp_path, "d1", validated=True), _bucket(tmp_path, "d2", validated=False)]
    r = reconcile_operating_point_validity(dirs, asserted=VALIDATED_HELD_OUT)
    assert r["validated"] == VALIDATED_FALSE
    assert r["unvalidated_buckets"] == [str(tmp_path / "d2")]


def test_reconcile_asserted_false_lowers_on_disk_validated(tmp_path):
    from tcip_mcp.pipelines.resolution import reconcile_operating_point_validity
    dirs = [_bucket(tmp_path, "d1", validated=True)]
    # The floor: an explicit asserted='false' lowers even a validated-on-disk bucket.
    r = reconcile_operating_point_validity(dirs, asserted=VALIDATED_FALSE)
    assert r["validated"] == VALIDATED_FALSE


def test_reconcile_review_confirmed_reference_preserved(tmp_path):
    from tcip_mcp.pipelines.resolution import (
        VALIDATED_REVIEW_CONFIRMED, reconcile_operating_point_validity,
    )
    dirs = [_bucket(tmp_path, "d1", validated=True, ref=VALIDATED_REVIEW_CONFIRMED)]
    r = reconcile_operating_point_validity(dirs)
    assert r["validated"] == VALIDATED_REVIEW_CONFIRMED  # provenance records which reference
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
    from tcip_annotation import json_io
    from tcip_annotation.state import BBox

    d = tmp_path / "labels"
    d.mkdir()
    json_io.write_detect(d / "a.json", [BBox(10, 10, 30, 30, 0)], 100, 100)
    json_io.write_detect(d / "b.json", [], 100, 100, keep_empty=True)  # negative still contributes
    h1 = dataset_hash(d)
    h2 = dataset_hash(d)
    assert h1 == h2
    # JSON-only: a stray legacy .txt is not part of the canonical GT identity and is ignored.
    (d / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    assert dataset_hash(d) == h1
    json_io.write_detect(d / "a.json", [BBox(10, 10, 30, 30, 1)], 100, 100)  # change GT content
    assert dataset_hash(d) != h1  # canonical JSON content changes the identity (not read as empty)


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


def test_validate_bundle_surfaces_all_firewall_issues_at_export():
    # The realistic delivery-time call: a dataset-scoped calibration inherited across a different
    # dataset while exporting, plus an in_chans mismatch. All three firewall checks must fire, so a
    # regression dropping any one (e.g. the shippable_issues(target_dataset_hash=...) fold-in)
    # can't slip past — each check is exercised in isolation elsewhere but never together.
    b = ResolvedBundle("catkin", "AAAA", {
        "in_chans": derived("in_chans", 3, derivation_class="deterministic", derived_from="raster"),
        "conf": derived("conf", 0.4, derivation_class="calibration", derived_from="sweep",
                        validated_vs_gt=VALIDATED_FALSE, dataset_scoped=True, dataset_hash="AAAA"),
    })
    issues = validate_resolved_bundle(b, probed_channels=4, target_dataset_hash="BBBB", for_export=True)
    assert any("in_chans" in s for s in issues)       # channel mismatch
    assert any("never inherit" in s for s in issues)  # dataset-scoped inherited across a hash
    assert any("shippable" in s for s in issues)      # export refuses an unvalidated operating point


# --- trait knowledge ---

def test_catkin_trait_semantics():
    t = get_trait("catkin")
    assert t is CATKIN
    assert t.count_objective == "count_unbiased"
    assert t.localization == "center_match"
    assert t.localization_tolerance == "half_class_avg_size"
    assert t.positive_is_texture is True
    assert t.milestone_fractions == (0.05, 0.50, 0.95)
    assert t.milestone_on == "positive_fraction"
    assert t.sliver_policy == "class_avg_size"
    assert t.count_bias_tolerance == 1.0


def test_unknown_trait_lists_available():
    with pytest.raises(TraitUnknownError) as exc:
        get_trait("banana")
    assert "catkin" in str(exc.value)
    assert "catkin" in registered_traits()
