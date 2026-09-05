"""What the operating-point record says about itself has to match what actually happened.

Every field here is read by someone reconstructing a delivered number: which split policy drew the
reference, which derivation produced the conf, what basis a tile edge rests on, what the reference's
own scores looked like against the floor the caller asserted, and which floor a calibration gate
was actually held to. A record that describes a different derivation, a different source, or a
different bar than the one that ran is a silent loss of auditability, with no wrong number visible
anywhere to signal it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import tcip_mcp.pipelines.operating_point as OP  # noqa: E402
from tcip_mcp.pipelines.operating_point import (  # noqa: E402
    attach_spatial_split_kind_provenance,
    attach_split_policy_provenance,
    resolve_operating_point,
)
from tcip_mcp.pipelines.resolution import (  # noqa: E402
    VALIDATED_FALSE,
    VALIDATED_HELD_OUT,
    VALIDATED_REVIEW_CONFIRMED,
)

pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")

N_IMAGES = 4
OBJECTS_PER_IMAGE = 8


def _records(prefix: str, offset: float, n_images: int = N_IMAGES) -> list[dict]:
    """One record per image, every object matched exactly by one detection at score 0.9. Boxes sit
    100 px apart, well outside the tolerance this GT derives.
    """
    recs = []
    for i in range(n_images):
        gt, dt = [], []
        for k in range(OBJECTS_PER_IMAGE):
            box = [offset + 100.0 * k, 50.0 + 10.0 * i, 40.0, 40.0]
            gt.append({"bbox": box, "category_id": 1})
            dt.append({"bbox": box, "category_id": 1, "score": 0.9})
        recs.append({"image_id": f"{prefix}{i}", "width": 4000, "height": 1000, "gt": gt, "dt": dt})
    return recs


# -- the split policy a reference was drawn under ----------------------------------------------

def test_split_policy_provenance_carries_each_locked_field_under_its_own_name():
    """Each locked field is stamped with a value distinct from every other, so a cross-wired read
    is visible rather than hidden behind two fields that happen to agree. The enrichment must also
    leave the resolved value and its validation stamp exactly as the gate produced them.
    """
    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h",
                                calibration_records=_records("c", 0.0))
    conf = b.params["conf"]
    value_before, stamp_before = conf._raw, conf.validated_against

    locked = {"group_by": "tile_prefix", "group_key_map": {"a_0_0": "a"}, "seed": 7,
              "holdout_ratio": 0.25, "identity_hash": "abc123",
              "policy_divergence": {"requested": {"seed": 9}, "locked": {"seed": 7}},
              "unlocked_stems": ["b_0_0"]}
    attach_split_policy_provenance(b, locked)

    policy = conf.gate_evidence["split_policy"]
    assert policy["group_by"] == "tile_prefix"
    assert policy["group_key_map"] == {"a_0_0": "a"}
    assert policy["seed"] == 7
    assert policy["holdout_ratio"] == 0.25
    assert policy["identity_hash"] == "abc123"
    assert conf.gate_evidence["split_policy_divergence"] == locked["policy_divergence"]
    assert conf.gate_evidence["split_unlocked_stems"] == ["b_0_0"]

    assert conf._raw == value_before
    assert conf.validated_against == stamp_before


def test_spatial_split_kind_provenance_names_the_split_and_its_own_geometry():
    """A block-calibrated bundle has no locked draw to read a policy off, only the mosaic's own
    recorded geometry, and each of those fields likewise has to land under its own name.
    """
    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h",
                                calibration_records=_records("c", 0.0))
    conf = b.params["conf"]
    attach_spatial_split_kind_provenance(b, {"seed": 3, "tile_size": 512, "overlap": 0.25})

    assert conf.gate_evidence["split_policy"] == {"group_by": "spatial_strip", "seed": 3,
                                          "tile_size": 512, "overlap": 0.25}


# -- the derivation label stamped on conf ------------------------------------------------------

def test_every_conf_label_the_registered_pickers_can_stamp_has_a_registered_implementation(
        monkeypatch):
    """The conf label is built at runtime from whichever picker ran, so a static scan of the stamp
    site's source cannot see it. Drive every registered count objective, under both accepted
    reference kinds, and check the labels those runs actually produced against the derivation
    registry, which exists so no data-sounding provenance string can name a derivation nothing
    implements.
    """
    from tcip_mcp.pipelines.derivations import DERIVATION_IMPLEMENTATIONS
    from tcip_mcp.pipelines.operating_point import COUNT_OBJECTIVE_PICKERS
    from tcip_mcp.traits import TraitSpec

    recs = _records("c", 0.0)
    labels = set()
    for objective in sorted(COUNT_OBJECTIVE_PICKERS):
        spec = TraitSpec(name="bud_opening", count_objective=objective,
                         delivers=("leaf_out_50per_date",))
        monkeypatch.setattr(OP, "get_trait", lambda name, s=spec: s)
        for reference in (VALIDATED_HELD_OUT, VALIDATED_REVIEW_CONFIRMED):
            b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h",
                                        calibration_records=recs, validated_reference=reference)
            labels.add(b.params["conf"].derived_from)

    assert sorted(lbl for lbl in labels if lbl not in DERIVATION_IMPLEMENTATIONS) == []
    assert len(labels) == 4  # two pickers, each with and without the review-verdict qualifier


# -- the basis a tile edge rests on ------------------------------------------------------------

def test_a_native_ratio_tile_source_reaches_the_resolver_intact():
    """``native_ratio`` is the fourth tile-size source and the one whose behavior differs from the
    no-basis case: it keeps a real tile edge, stamps a derived source, and clears its own geometry
    reference. Collapsing it into the no-basis case would discard a caller's edge silently.
    """
    import tcip_mcp.pipelines.resolution as resolution_mod

    native_ref = getattr(resolution_mod, "VALIDATED_NATIVE_FRAME_GEOMETRY", None)
    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h",
                                tile_size=300, tile_size_source="native_ratio")
    p = b.params["tile_size"]
    assert p._raw == 300
    assert p.source == "derived"
    assert p.validated_against == native_ref
    assert p.is_shippable is True

    no_basis = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h",
                                       tile_size=300, tile_size_source="unavailable")
    assert no_basis.params["tile_size"]._raw is None  # the case native_ratio is not


# -- the reference's own scores against the asserted floor -------------------------------------

def test_a_floor_mismatch_on_either_side_of_the_reference_is_surfaced():
    """The asserted floor is reconciled against both halves of the reference. Here calibration's
    lowest score sits far above the asserted floor while the holdout's sits right at it, so only a
    check that reads the calibration side too can see the gap. The signal is provenance, never a
    gate, so the reference still validates.
    """
    cal = _records("c", 0.0)
    hold = _records("h", 100000.0)
    hold[0]["dt"].append({"bbox": [900000.0, 50.0, 40.0, 40.0], "category_id": 1, "score": 0.05})

    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h", staged_conf_floor=0.05,
                                calibration_records=cal, holdout_records=hold)
    sweep = b.params["conf"].gate_evidence

    assert sweep["calibration_observed_min_score"] == pytest.approx(0.9)
    assert sweep["holdout_observed_min_score"] == pytest.approx(0.05)
    assert sweep["conf_floor_mismatch"] is True
    assert "conf_floor_mismatch" not in sweep["failures"]
    assert sweep["failures"] == []
    assert b.params["conf"].validated_against == VALIDATED_HELD_OUT


# -- what counts as a cloned holdout -----------------------------------------------------------

def test_a_holdout_sharing_one_image_of_content_with_calibration_refuses():
    """A holdout sharing even one image's content with calibration is not independent for that
    image, so the gate refuses rather than merely reporting the overlap; a holdout sharing no
    image's content still passes, with the overlap (zero) reported.
    """
    cal = _records("c", 0.0, n_images=3)
    hold = _records("h", 100000.0, n_images=3)
    hold[0]["gt"] = [dict(a) for a in cal[0]["gt"]]  # one image of genuinely shared content
    hold[0]["dt"] = [dict(d) for d in cal[0]["dt"]]

    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h", staged_conf_floor=0.05,
                                calibration_records=cal, holdout_records=hold)
    sweep = b.params["conf"].gate_evidence

    assert sweep["content_overlap_frac"] == pytest.approx(1.0 / 3.0)  # the overlap is real
    assert sweep["content_shared_with_calibration"] is True
    assert "content_shared_with_calibration" in sweep["failures"]
    assert b.params["conf"].validated_against == VALIDATED_FALSE


# -- which floor a scalar calibration gate was held to -----------------------------------------

_RANKS = [0, 1, 2, 3, 4] * 4


def _ordinal_items(prefix: str, true_ranks: list[int], pred_ranks: list[int]) -> list[dict]:
    return [{"image_id": f"{prefix}{i}", "true_rank": t, "predicted_rank": p}
            for i, (t, p) in enumerate(zip(true_ranks, pred_ranks))]


def _regression_items(prefix: str, true_values: list[float],
                      pred_values: list[float]) -> list[dict]:
    return [{"image_id": f"{prefix}{i}", "true_value": t, "predicted_value": p}
            for i, (t, p) in enumerate(zip(true_values, pred_values))]


def test_an_authored_ordinal_agreement_floor_governs_instead_of_the_platform_placeholder(
        monkeypatch):
    """A trait that authors its own agreement bar is held to that bar. The platform's placeholder
    exists only for a trait that has not authored one, and quietly substituting it would admit a
    calibration the breeder's own stated bar refuses, while the record still names the trait as the
    source of the floor.
    """
    from tcip_mcp.pipelines.operating_point import resolve_ordinal_operating_point
    from tcip_mcp.traits import TraitSpec

    pred = list(_RANKS)
    for i in range(6):
        pred[i] = min(4, pred[i] + 2)

    authored = TraitSpec(name="bud_opening", ordinal_agreement_floor=0.9,
                         delivers=("leaf_out_50per_date",))
    monkeypatch.setattr(OP, "get_trait", lambda name: authored)

    res = resolve_ordinal_operating_point(
        "bud_opening", criterion="quadratic_weighted_kappa",
        calibration_items=_ordinal_items("c", _RANKS, pred),
        holdout_items=_ordinal_items("h", _RANKS, pred), experiment_id=None)

    score = res["gate_evidence"]["score"]
    # Real agreement, above the platform placeholder and below the authored bar: exactly the band
    # in which the two floors disagree about the verdict.
    assert 0.41 < score < 0.9
    assert res["gate_evidence"]["floor"] == pytest.approx(0.9)
    assert res["gate_evidence"]["floor_source"] == "trait"
    assert res["failures"] == ["compensating_error_floor_failed"]
    assert res["passed"] is False
    assert res["validated_against"] == VALIDATED_FALSE


def test_an_authored_regression_skill_floor_governs_instead_of_the_platform_placeholder(
        monkeypatch):
    """The regression counterpart, on a criterion whose scale is its own: the authored bar is what
    the calibration is judged against.
    """
    from tcip_mcp.pipelines.operating_point import resolve_regression_operating_point
    from tcip_mcp.traits import TraitSpec

    true_values = [float(r) for r in _RANKS]
    pred_values = [t * 0.5 + 1.0 for t in true_values]

    authored = TraitSpec(name="bud_opening", regression_skill_floor=0.9,
                         delivers=("leaf_out_50per_date",))
    monkeypatch.setattr(OP, "get_trait", lambda name: authored)

    res = resolve_regression_operating_point(
        "bud_opening", criterion="r_squared",
        calibration_items=_regression_items("c", true_values, pred_values),
        holdout_items=_regression_items("h", true_values, pred_values), experiment_id=None)

    score = res["gate_evidence"]["score"]
    assert 0.5 < score < 0.9
    assert res["gate_evidence"]["floor"] == pytest.approx(0.9)
    assert res["gate_evidence"]["floor_source"] == "trait"
    assert res["failures"] == ["compensating_error_floor_failed"]
    assert res["passed"] is False
    assert res["validated_against"] == VALIDATED_FALSE


def test_a_trait_that_authors_no_scalar_floor_still_calibrates_against_the_placeholder(
        monkeypatch):
    """The companion obligation: the placeholder path must keep admitting a calibration that
    clears it, so authoring a floor stays optional rather than a precondition for calibrating at
    all.
    """
    from tcip_mcp.pipelines.operating_point import resolve_ordinal_operating_point
    from tcip_mcp.traits import TraitSpec

    pred = list(_RANKS)
    for i in range(6):
        pred[i] = min(4, pred[i] + 2)

    unauthored = TraitSpec(name="bud_opening", delivers=("leaf_out_50per_date",))
    monkeypatch.setattr(OP, "get_trait", lambda name: unauthored)

    res = resolve_ordinal_operating_point(
        "bud_opening", criterion="quadratic_weighted_kappa",
        calibration_items=_ordinal_items("c", _RANKS, pred),
        holdout_items=_ordinal_items("h", _RANKS, pred), experiment_id=None)

    assert res["gate_evidence"]["floor_source"] == "default"
    assert res["failures"] == []
    assert res["passed"] is True
