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
    default,
    derived,
    validate_resolved_bundle,
)
from tcip_mcp.traits import TraitUnknownError, get_trait, registered_traits
from tests._trait_fixtures import BUD_OPENING

# No built-in traits: seed_bud_trait_spec (conftest.py) writes a real bud_opening.yml into this
# test's pinned platform state root so get_trait("bud_opening") keeps resolving by default.
pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")


# --- the firewall: an unvalidated param that requires validation is un-consumable ---

def test_unvalidated_calibration_value_raises():
    p = derived("conf", 0.4, requires_validation=True, validation_kind="annotations", derived_from="sweep",
                validated_against=VALIDATED_FALSE, gate_evidence={"curve": []})
    assert not p.is_shippable
    with pytest.raises(UnvalidatedOperatingPointError):
        _ = p.value


def test_validated_heldout_calibration_value_ok():
    p = derived("conf", 0.4, requires_validation=True, validation_kind="annotations", derived_from="sweep",
                validated_against=VALIDATED_HELD_OUT)
    assert p.is_shippable
    assert p.value == 0.4


def test_param_that_needs_no_validation_always_ships():
    # a fact read from the data needs no validation, so it ships with no validated_against at all
    p = derived("num_classes", 2, derived_from="labels")
    assert p.requires_validation is False and p.validation_kind is None
    assert p.value == 2


def test_review_confirmed_calibration_is_shippable():
    from tcip_mcp.pipelines.resolution import VALIDATED_REVIEW_CONFIRMED
    p = derived("conf", 0.4, requires_validation=True, validation_kind="annotations",
                derived_from="count-unbiased center-match curve over review verdicts",
                validated_against=VALIDATED_REVIEW_CONFIRMED)
    assert p.is_shippable  # a review-confirmed reference ships (distinct flag, same gate)
    assert p.value == 0.4


# --- reconcile the delivery gate against on-disk operating_point.json ---

def _bucket(tmp_path, name, *, validated, ref=VALIDATED_HELD_OUT, conf=0.6):
    from tcip_mcp.pipelines.resolution import write_sidecar
    from tests._binding_fixtures import write_bound_sidecar, write_prediction

    root = tmp_path / "ds"
    d = root / "predictions" / name
    stamp = {
        "validated": validated, "trait": "bud_opening",
        "operating_point": {"conf": {"value": conf, "validated_against": ref if validated else "false"}},
        "subject": "bud", "attribute": None,
    }
    if validated:
        write_prediction(d, "img_a")
        write_bound_sidecar(d, stamp, dataset_root=root, experiment_id=f"exp-{name}")
    else:
        d.mkdir(parents=True, exist_ok=True)
        write_sidecar(d, stamp)
    return str(d)


def test_reconcile_missing_sidecar_floors_to_false(tmp_path):
    from tcip_mcp.pipelines.resolution import reconcile_operating_point_validity
    # A caller asserting validated cannot open the gate when no sidecar backs it.
    r = reconcile_operating_point_validity(
        [str(tmp_path / "nope")], trait="bud_opening", asserted=VALIDATED_HELD_OUT)
    assert r["validated"] == VALIDATED_FALSE
    assert r["missing_sidecars"] == [str(tmp_path / "nope")]


def test_reconcile_all_validated_on_disk(tmp_path):
    from tcip_mcp.pipelines.resolution import reconcile_operating_point_validity
    dirs = [_bucket(tmp_path, "d1", validated=True), _bucket(tmp_path, "d2", validated=True)]
    r = reconcile_operating_point_validity(dirs, trait="bud_opening")  # no caller assertion needed
    assert r["validated"] == VALIDATED_HELD_OUT
    assert r["on_disk_validated"] is True
    assert r["conf"] == 0.6


def test_reconcile_one_unvalidated_bucket_floors_whole_curve(tmp_path):
    from tcip_mcp.pipelines.resolution import reconcile_operating_point_validity
    d1 = _bucket(tmp_path, "d1", validated=True)
    d2 = _bucket(tmp_path, "d2", validated=False)
    r = reconcile_operating_point_validity([d1, d2], trait="bud_opening", asserted=VALIDATED_HELD_OUT)
    assert r["validated"] == VALIDATED_FALSE
    assert r["unvalidated_buckets"] == [d2]


def test_reconcile_asserted_false_lowers_on_disk_validated(tmp_path):
    from tcip_mcp.pipelines.resolution import reconcile_operating_point_validity
    dirs = [_bucket(tmp_path, "d1", validated=True)]
    # The floor: an explicit asserted='false' lowers even a validated-on-disk bucket.
    r = reconcile_operating_point_validity(dirs, trait="bud_opening", asserted=VALIDATED_FALSE)
    assert r["validated"] == VALIDATED_FALSE


def test_reconcile_review_confirmed_reference_preserved(tmp_path):
    from tcip_mcp.pipelines.resolution import (
        VALIDATED_REVIEW_CONFIRMED, reconcile_operating_point_validity,
    )
    dirs = [_bucket(tmp_path, "d1", validated=True, ref=VALIDATED_REVIEW_CONFIRMED)]
    r = reconcile_operating_point_validity(dirs, trait="bud_opening")
    assert r["validated"] == VALIDATED_REVIEW_CONFIRMED  # provenance records which reference
    from tcip_mcp.pipelines.resolution import default
    assert default("lr", 1e-3).value == 1e-3


def test_unvalidated_value_requires_acknowledgement():
    p = derived("conf", 0.4, requires_validation=True, validation_kind="annotations", derived_from="sweep",
                validated_against=VALIDATED_FALSE)
    with pytest.raises(UnvalidatedOperatingPointError):
        p.unvalidated_value(acknowledge_unvalidated=False)
    assert p.unvalidated_value(acknowledge_unvalidated=True) == 0.4


def test_resolvedparam_rejects_bad_vocab():
    with pytest.raises(ValueError):
        ResolvedParam(name="x", _raw=1, source="bogus", requires_validation=True, validation_kind="annotations")
    with pytest.raises(ValueError):
        # requires_validation with no kind: a param needing validation must say which reference can
        # give it, never left to a silent default
        ResolvedParam(name="x", _raw=1, source="derived", requires_validation=True)
    with pytest.raises(ValueError):
        ResolvedParam(name="x", _raw=1, source="derived", requires_validation=True, validation_kind="bogus")
    with pytest.raises(ValueError):
        # a kind with nothing to validate is the mirror contradiction
        ResolvedParam(name="x", _raw=1, source="derived", validation_kind="annotations")
    with pytest.raises(ValueError):
        ResolvedParam(name="x", _raw=1, source="derived", validated_against="sort-of")


def test_validated_against_must_be_the_right_kind_for_the_param():
    from tcip_mcp.pipelines.resolution import (
        VALIDATED_PHYSICAL_MEASUREMENT, accepted_references,
    )

    # an annotations-kind param (conf) is not cleared by a physical-measurement reference...
    conf = derived("conf", 0.4, derived_from="sweep", requires_validation=True,
                   validation_kind="annotations", validated_against=VALIDATED_PHYSICAL_MEASUREMENT)
    assert conf.is_shippable is False
    # ...and a physical-kind param is not cleared by held-out annotations
    scale = derived("mm_per_px", 0.5, derived_from="reference object", requires_validation=True,
                    validation_kind="physical", validated_against=VALIDATED_HELD_OUT)
    assert scale.is_shippable is False
    assert VALIDATED_HELD_OUT not in accepted_references("physical")
    assert VALIDATED_PHYSICAL_MEASUREMENT not in accepted_references("annotations")
    ok = derived("mm_per_px", 0.5, derived_from="reference object", requires_validation=True,
                 validation_kind="physical", validated_against=VALIDATED_PHYSICAL_MEASUREMENT)
    assert ok.is_shippable is True and ok.value == 0.5


def test_capture_scoped_param_is_not_comparable_without_a_capture_id():
    p = derived("mm_per_px", 0.5, derived_from="reference object", capture_scoped=True)
    issues = ResolvedBundle("bud_opening", "h1", {"mm_per_px": p}).shippable_issues()
    assert any("capture" in s for s in issues)
    scoped = derived("mm_per_px", 0.5, derived_from="reference object",
                     capture_scoped=True, capture_id="cap1")
    assert ResolvedBundle("bud_opening", "h1", {"mm_per_px": scoped}).shippable_issues(
        target_capture_id="cap1") == []
    assert any("never inherit" in s for s in ResolvedBundle("bud_opening", "h1", {"mm_per_px": scoped})
               .shippable_issues(target_capture_id="cap2"))


# --- tile_size gates the same shape conf already does ----------------

def test_tile_size_untiled_is_never_gating():
    # An untiled run's count never depends on tile_size, so it stays a plain non-gating fact
    # (mirrors in_chans), never manufacturing a refusal over a dimension that was never operative.
    from tcip_mcp.pipelines.resolution import resolve_tile_size_param

    p = resolve_tile_size_param(640, tiled=False, tile_size_source="default",
                                tile_size_derived_from=None)
    assert p.requires_validation is False
    assert p.validation_kind is None
    assert p._raw is None
    assert p.is_shippable is True


def test_tile_size_derived_from_checkpoint_geometry_is_shippable():
    from tcip_mcp.pipelines.resolution import VALIDATED_PERSISTED_GEOMETRY, resolve_tile_size_param

    p = resolve_tile_size_param(224, tiled=True, tile_size_source="derived",
                                tile_size_derived_from=None)
    assert p.requires_validation is True and p.validation_kind == "geometry"
    assert p.validated_against == VALIDATED_PERSISTED_GEOMETRY
    assert p.is_shippable is True
    assert p.value == 224


def test_tile_size_explicit_caller_override_is_shippable():
    # Accepted on the same terms run_full_frame_evaluation already accepts an explicit value on: a
    # stated decision, checked for contradiction against the checkpoint upstream of this function.
    from tcip_mcp.pipelines.resolution import VALIDATED_EXPLICIT_GEOMETRY, resolve_tile_size_param

    p = resolve_tile_size_param(512, tiled=True, tile_size_source="explicit",
                                tile_size_derived_from="stated on a checkpoint that records no "
                                                       "tile geometry")
    assert p.requires_validation is True and p.validation_kind == "geometry"
    assert p.validated_against == VALIDATED_EXPLICIT_GEOMETRY
    assert p.is_shippable is True
    assert p.value == 512
    assert p.derived_from == "stated on a checkpoint that records no tile geometry"


def test_tile_size_no_basis_is_not_shippable():
    # A checkpoint with no persisted geometry and no explicit override has no real basis for a tile
    # scale at all: no fallback number is fabricated, and the dimension is firewalled like conf is.
    from tcip_mcp.pipelines.resolution import resolve_tile_size_param

    p = resolve_tile_size_param(None, tiled=True, tile_size_source="unavailable",
                                tile_size_derived_from=None)
    assert p.requires_validation is True and p.validation_kind == "geometry"
    assert p.validated_against == VALIDATED_FALSE
    assert p.is_shippable is False
    with pytest.raises(UnvalidatedOperatingPointError):
        _ = p.value
    assert p.unvalidated_value(acknowledge_unvalidated=True) is None


def test_tile_size_native_ratio_is_a_real_basis_and_shippable_under_its_own_reference():
    # A real basis to tile at all (a checkpoint's own uniform untiled training size), and (after
    # promotion) a real geometry reference in its own right, distinct from a persisted one.
    import tcip_mcp.pipelines.resolution as resolution_mod
    from tcip_mcp.pipelines.resolution import VALIDATED_PERSISTED_GEOMETRY, resolve_tile_size_param

    native_ref = getattr(resolution_mod, "VALIDATED_NATIVE_FRAME_GEOMETRY", None)
    p = resolve_tile_size_param(300, tiled=True, tile_size_source="native_ratio",
                                tile_size_derived_from=None)
    assert p.requires_validation is True and p.validation_kind == "geometry"
    assert p.validated_against == native_ref
    assert p.validated_against != VALIDATED_PERSISTED_GEOMETRY
    assert p.is_shippable is True
    assert p.value == 300


def test_raw_operating_point_no_basis_tiled_default_surfaces_its_own_shippable_issue():
    # tile_size must participate in shippable_issues() regardless of source: a no-basis fallback
    # (the caller's raw 640 is discarded either way) must not be silently shippable trivia.
    from tcip_mcp.pipelines.resolution import raw_operating_point

    b = raw_operating_point(conf=0.9, cross_tile_nms=0.3, tiled=True, tile_size=640, max_dets=1000)
    issues = b.shippable_issues()
    assert any(i.startswith("tile_size:") for i in issues)
    assert any(i.startswith("conf:") for i in issues)  # both dimensions gate independently


def test_raw_operating_point_untiled_never_gates_tile_size():
    # The rail must admit valid work, not only reject invalid work: an untiled call must never be
    # refused over a tile_size that was never operative for it. Asserted as a contrast against the
    # same fabricated tile_size value actually gating once tiled=True: an untiled-only assertion
    # alone would pass just as well against a broken build where tile_size never gates at all, so
    # this pins the "only when operative" boundary, not merely "untiled is fine" in isolation.
    from tcip_mcp.pipelines.resolution import raw_operating_point

    tiled = raw_operating_point(conf=0.9, cross_tile_nms=0.3, tiled=True, tile_size=640, max_dets=1000)
    assert any(i.startswith("tile_size:") for i in tiled.shippable_issues())

    untiled = raw_operating_point(conf=0.9, cross_tile_nms=None, tiled=False, tile_size=None, max_dets=1000)
    assert not any(i.startswith("tile_size:") for i in untiled.shippable_issues())
    assert untiled.get("tile_size").requires_validation is False


def test_raw_operating_point_max_dets_none_is_a_real_uncapped_value():
    # A deliberate value (uncapped), not an unset caller, whatever source it carries: never
    # coerced into DEFAULT_MAX_DETS or refused for being falsy.
    from tcip_mcp.pipelines.resolution import raw_operating_point

    b = raw_operating_point(conf=0.9, cross_tile_nms=0.3, tiled=True, tile_size=64, max_dets=None)
    md = b.get("max_dets")
    assert md._raw is None
    assert md.requires_validation is False  # never gates a delivery on its own


# --- bundle shippability + provenance ---

def test_bundle_shippable_only_when_all_calibration_validated():
    unval = derived("conf", 0.4, requires_validation=True, validation_kind="annotations", derived_from="sweep",
                    validated_against=VALIDATED_FALSE)
    val = derived("conf", 0.4, requires_validation=True, validation_kind="annotations", derived_from="sweep",
                  validated_against=VALIDATED_HELD_OUT)
    fact = derived("max_dets", 300, derived_from="p99")
    assert not ResolvedBundle("bud_opening", "h1", {"conf": unval, "max_dets": fact}).is_shippable
    assert ResolvedBundle("bud_opening", "h1", {"conf": val, "max_dets": fact}).is_shippable


def test_dataset_scoped_calibration_not_inherited_across_hash():
    p = derived("conf", 0.4, requires_validation=True, validation_kind="annotations", derived_from="sweep",
                validated_against=VALIDATED_HELD_OUT, dataset_scoped=True, dataset_hash="AAAA")
    b = ResolvedBundle("bud_opening", "AAAA", {"conf": p})
    assert b.shippable_issues(target_dataset_hash="AAAA") == []
    issues = b.shippable_issues(target_dataset_hash="BBBB")
    assert any("never inherit" in s for s in issues)


def test_provenance_roundtrip_is_serializable():
    import json
    b = ResolvedBundle("bud_opening", "h1", {
        "conf": derived("conf", 0.4, requires_validation=True, validation_kind="annotations", derived_from="sweep",
                        validated_against=VALIDATED_HELD_OUT),
    })
    json.dumps(b.to_provenance())  # must not raise


# --- dataset identity ---

def test_dataset_hash_content_addressed(tmp_path):
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    d = tmp_path / "labels"
    d.mkdir()
    json_io.write_annotations(d / "a.json", [Annotation(subject="bud", geometry=BBox(10, 10, 30, 30))],
                              100, 100)
    json_io.write_annotations(d / "b.json", [], 100, 100, keep_empty=True)  # negative still contributes
    h1 = dataset_hash(d)
    h2 = dataset_hash(d)
    assert h1 == h2
    # JSON-only: a stray legacy .txt is not part of the canonical GT identity and is ignored.
    (d / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    assert dataset_hash(d) == h1
    json_io.write_annotations(d / "a.json", [Annotation(subject="bud", geometry=BBox(10, 10, 40, 40))],
                              100, 100)  # change GT content
    assert dataset_hash(d) != h1  # canonical JSON content changes the identity (not read as empty)


def test_dataset_hash_with_no_stems_excludes_a_bucket_sidecar(tmp_path):
    """A bucket's own provenance stamp is not a label, and with no explicit stems dataset_hash
    walks the directory itself: its bytes must never enter a ground-truth identity."""
    from tcip_annotation import json_io
    from tcip_annotation.json_io import SIDECAR_FILENAMES
    from tcip_annotation.state import Annotation, BBox

    d = tmp_path / "labels"
    d.mkdir()
    json_io.write_annotations(d / "a.json", [Annotation(subject="bud", geometry=BBox(10, 10, 30, 30))],
                              100, 100)
    before = dataset_hash(d)
    for name in SIDECAR_FILENAMES:
        (d / name).write_text('{"conf": {"value": 0.5}}', encoding="utf-8")
    assert dataset_hash(d) == before


# --- validate_resolved_bundle live checks ---

def test_validate_in_chans_vs_probed_bands():
    b = ResolvedBundle("bud_opening", "h1", {
        "in_chans": derived("in_chans", 3, derived_from="raster"),
    })
    assert validate_resolved_bundle(b, probed_channels=3) == []
    issues = validate_resolved_bundle(b, probed_channels=4)
    assert any("in_chans" in s for s in issues)


def test_validate_eval_vs_inference_operating_point_mismatch():
    def bundle(conf):
        return ResolvedBundle("bud_opening", "h1", {
            "conf": derived("conf", conf, requires_validation=True, validation_kind="annotations", derived_from="sweep",
                            validated_against=VALIDATED_HELD_OUT),
        })
    ev, inf = bundle(0.25), bundle(0.5)
    issues = validate_resolved_bundle(ev, inference_bundle=inf)
    assert any("operating point" in s for s in issues)
    assert validate_resolved_bundle(bundle(0.4), inference_bundle=bundle(0.4)) == []


def test_validate_max_dets_divergence_is_a_named_exemption_not_a_silent_gap():
    # A block bundle's max_dets and its export bundle's max_dets diverge by design (see this
    # function's own docstring); a caller comparing exactly those two must exclude "max_dets".
    block = ResolvedBundle("bud_opening", "h1", {"max_dets": derived("max_dets", 42, derived_from="p99")})
    export = ResolvedBundle("bud_opening", "h1", {
        "max_dets": default("max_dets", None, derived_from="block calibration: uncapped"),
    })
    issues = validate_resolved_bundle(block, inference_bundle=export)
    assert any(i.startswith("max_dets:") for i in issues)


def test_validate_export_refuses_unvalidated():
    b = ResolvedBundle("bud_opening", "h1", {
        "conf": derived("conf", 0.4, requires_validation=True, validation_kind="annotations", derived_from="sweep",
                        validated_against=VALIDATED_FALSE),
    })
    issues = validate_resolved_bundle(b, for_export=True)
    assert any("shippable" in s or "not validated" in s for s in issues)


def test_validate_bundle_surfaces_all_firewall_issues_at_export():
    # The realistic delivery-time call: a dataset-scoped calibration inherited across a different
    # dataset while exporting, plus an in_chans mismatch. All three firewall checks must fire, so a
    # regression dropping any one (e.g. the shippable_issues(target_dataset_hash=...) fold-in)
    # can't slip past: each check is exercised in isolation elsewhere but never together.
    b = ResolvedBundle("bud_opening", "AAAA", {
        "in_chans": derived("in_chans", 3, derived_from="raster"),
        "conf": derived("conf", 0.4, requires_validation=True, validation_kind="annotations", derived_from="sweep",
                        validated_against=VALIDATED_FALSE, dataset_scoped=True, dataset_hash="AAAA"),
    })
    issues = validate_resolved_bundle(b, probed_channels=4, target_dataset_hash="BBBB", for_export=True)
    assert any("in_chans" in s for s in issues)       # channel mismatch
    assert any("never inherit" in s for s in issues)  # dataset-scoped inherited across a hash
    assert any("shippable" in s for s in issues)      # export refuses an unvalidated operating point


# --- trait knowledge ---

def test_bud_opening_trait_semantics():
    # config-loaded specs are rebuilt fresh per call (traits.py), never module-load singletons, so
    # value equality against the same-valued local fixture, not identity.
    t = get_trait("bud_opening")
    assert t == BUD_OPENING
    assert t.count_objective == "count_unbiased"
    assert t.localization == "center_match"
    assert t.localization_tolerance == "half_class_avg_size"
    assert t.milestone_fractions == (0.05, 0.50, 0.95)
    assert t.milestone_on == "positive_fraction"
    assert t.sliver_policy == "class_avg_size"
    assert t.count_bias_tolerance_frac is None  # not yet authored by the domain expert


def test_unknown_trait_lists_available():
    with pytest.raises(TraitUnknownError) as exc:
        get_trait("banana")
    assert "bud_opening" in str(exc.value)
    assert "bud_opening" in registered_traits()


# --- raw_operating_point: a stated conf/max_dets is never laundered into a default ---

def test_raw_operating_point_stamps_explicit_when_the_caller_states_a_value():
    """A caller-stated value is stamped 'explicit' even when it happens to equal the platform
    default, the same distinction tile_size_source/tiled_source already carry."""
    from tcip_mcp.pipelines.resolution import DEFAULT_CONF, DEFAULT_MAX_DETS, raw_operating_point

    bundle = raw_operating_point(
        conf=DEFAULT_CONF, cross_tile_nms=None, tiled=False, tile_size=None,
        max_dets=DEFAULT_MAX_DETS, conf_stated=True, max_dets_stated=True,
    )
    assert bundle.get("conf").source == "explicit"
    assert bundle.get("conf").derived_from == "caller override"
    assert bundle.get("max_dets").source == "explicit"
    assert bundle.get("max_dets").derived_from == "caller override"


def test_raw_operating_point_stamps_default_when_the_caller_states_nothing():
    """The rail must admit the ordinary, unstated call: an omitted conf/max_dets is never
    laundered into 'explicit'."""
    from tcip_mcp.pipelines.resolution import DEFAULT_CONF, DEFAULT_MAX_DETS, raw_operating_point

    bundle = raw_operating_point(
        conf=DEFAULT_CONF, cross_tile_nms=None, tiled=False, tile_size=None,
        max_dets=DEFAULT_MAX_DETS,
    )
    assert bundle.get("conf").source == "default"
    assert bundle.get("max_dets").source == "default"


# --- _reconcile_validity: a stamp earned for one trait does not answer for another ---

def test_reconcile_operating_point_validity_floors_a_trait_mismatch(tmp_path):
    """A validated count stamp whose trait differs from the delivery's registry trait floors, with
    a note naming both."""
    from tcip_mcp.pipelines.resolution import reconcile_operating_point_validity

    d = _bucket(tmp_path, "d1", validated=True)  # stamped trait="bud_opening"

    mismatched = reconcile_operating_point_validity([d], trait="second_trait")
    assert mismatched["validated"] == VALIDATED_FALSE
    note = mismatched["binding_notes"][d]
    assert "bud_opening" in note and "second_trait" in note


def test_reconcile_operating_point_validity_admits_a_matching_trait(tmp_path):
    """The rail must admit valid work: a delivery whose trait matches the stamp's own still
    validates."""
    from tcip_mcp.pipelines.resolution import reconcile_operating_point_validity

    d = _bucket(tmp_path, "d1", validated=True)  # stamped trait="bud_opening"

    matched = reconcile_operating_point_validity([d], trait="bud_opening")
    assert matched["validated"] == VALIDATED_HELD_OUT


def test_reconcile_operating_point_validity_still_floors_an_unbacked_trait_none_bucket(tmp_path):
    """A raw web-door bucket (validated=false, trait=None) still floors as unvalidated through the
    pre-existing short-circuit, never erroring when the delivery states a real trait the bucket
    states none of."""
    from tcip_mcp.pipelines.resolution import reconcile_operating_point_validity, write_sidecar

    d = tmp_path / "raw_bucket"
    write_sidecar(d, {"validated": False, "trait": None,
                      "operating_point": {"conf": {"validated_against": None}},
                      "subject": None, "attribute": None})

    r = reconcile_operating_point_validity([str(d)], trait="bud_opening")
    assert r["validated"] == VALIDATED_FALSE


# --- resolve_tile_size_param: an explicit source states its own derivation text ---

def test_tile_size_explicit_with_no_derived_from_text_refuses():
    from tcip_mcp.pipelines.resolution import resolve_tile_size_param

    with pytest.raises(ValueError, match="tile_size_derived_from"):
        resolve_tile_size_param(512, tiled=True, tile_size_source="explicit",
                                tile_size_derived_from=None)


def test_resolve_operating_point_explicit_tile_size_with_no_text_refuses():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    with pytest.raises(ValueError, match="tile_size_derived_from"):
        resolve_operating_point("bud_opening", tiled=True, dataset_hash=None, tile_size=512,
                                tile_size_source="explicit")


def test_resolve_operating_point_explicit_tile_size_with_text_ships():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    bundle = resolve_operating_point(
        "bud_opening", tiled=True, dataset_hash=None, tile_size=512, tile_size_source="explicit",
        tile_size_derived_from="stated on a checkpoint that records no tile geometry")
    param = bundle.get("tile_size")
    assert param.source == "explicit"
    assert param.derived_from == "stated on a checkpoint that records no tile geometry"


# --- resolver_train_disjointness: one branch per declared document ---

def test_resolver_train_disjointness_refuses_an_undeclared_document():
    from tcip_mcp.pipelines.resolution import resolver_train_disjointness

    with pytest.raises(ValueError, match="not a document"):
        resolver_train_disjointness({"gate_evidence": {"train_disjointness": {"checked": True}}},
                                    "made_up_document")


def test_resolver_train_disjointness_reads_a_declared_documents_result():
    from tcip_mcp.pipelines.resolution import resolver_train_disjointness

    result = {"gate_evidence": {"train_disjointness": {"checked": True, "group_check": None}}}
    assert resolver_train_disjointness(result, "classifier_operating_point") == (
        {"checked": True, "group_check": None})


# --- resolver_selection_disjointness: the same shape, plus applicable/reason ---

def test_resolver_selection_disjointness_refuses_an_undeclared_document():
    from tcip_mcp.pipelines.resolution import resolver_selection_disjointness

    with pytest.raises(ValueError, match="not a document"):
        resolver_selection_disjointness(
            {"gate_evidence": {"selection_disjointness": {"applicable": False}}}, "made_up_document")


def test_resolver_selection_disjointness_reads_a_declared_documents_result():
    from tcip_mcp.pipelines.resolution import resolver_selection_disjointness

    result = {"gate_evidence": {"selection_disjointness": {
        "applicable": True, "reason": None, "checked": True, "unresolvable": False,
        "leaked_groups": [], "leaked_stems": [], "group_check": "performed"}}}
    assert resolver_selection_disjointness(result, "classifier_operating_point") == {
        "applicable": True, "reason": None, "checked": True, "unresolvable": False,
        "leaked_groups": [], "leaked_stems": [], "group_check": "performed",
        "labels_moved_draw_to_run": None, "labels_moved_run_to_now": None,
        "calibration_labels_moved": None, "manifest_redrawn": None,
        "calibration_labels_dir": None}


def test_resolver_selection_disjointness_carries_the_leak_fields():
    """The row's field carries the same twelve keys the live gate evidence does, unresolvable/
    leaked_groups/leaked_stems and the label-movement keys included, not only the pass/fail
    booleans a caller cannot floor a leaking or a moved-label row from."""
    from tcip_mcp.pipelines.resolution import resolver_selection_disjointness

    result = {"gate_evidence": {"selection_disjointness": {
        "applicable": True, "reason": None, "checked": True, "unresolvable": False,
        "leaked_groups": ["g1"], "leaked_stems": ["s1"], "group_check": "performed"}}}
    assert resolver_selection_disjointness(result, "classifier_operating_point") == {
        "applicable": True, "reason": None, "checked": True, "unresolvable": False,
        "leaked_groups": ["g1"], "leaked_stems": ["s1"], "group_check": "performed",
        "labels_moved_draw_to_run": None, "labels_moved_run_to_now": None,
        "calibration_labels_moved": None, "manifest_redrawn": None,
        "calibration_labels_dir": None}


def test_resolver_selection_disjointness_is_none_for_resolve_scale():
    from tcip_mcp.pipelines.resolution import resolver_selection_disjointness

    assert resolver_selection_disjointness({"gate_evidence": {}}, "resolve_scale") is None


# --- every real reference is reachable through the delivery gate's own dimension table ---

def test_every_validated_shippable_reference_clears_some_gate_dimension():
    """Coverage of an invariant that already holds at HEAD: VALIDATED_SHIPPABLE is "every real
    (non-false) reference across every kind", and _DIMENSION_REFERENCES is what check_delivery_gate
    actually reads to decide whether a caller-recorded reference clears a dimension. A reference
    could be added to one table and not the other by mistake; this pins them to agreeing, so a
    real, accepted reference is never invisible to the gate that is supposed to recognize it.
    """
    from tcip_mcp.pipelines.resolution import VALIDATED_SHIPPABLE, _DIMENSION_REFERENCES

    assert VALIDATED_SHIPPABLE
    reachable = {reference for refs in _DIMENSION_REFERENCES.values() for reference in refs}
    missing = [name for name in VALIDATED_SHIPPABLE if name not in reachable]
    assert not missing, (
        f"{missing} are in VALIDATED_SHIPPABLE but no _DIMENSION_REFERENCES dimension accepts "
        "them, so check_delivery_gate can never clear a dimension with this reference"
    )
