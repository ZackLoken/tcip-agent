"""What a resolved operating point can and cannot be turned into once it exists.

Two boundaries meet here. A resolved param's own fields are fixed at construction, so a validity
label cannot be rewritten in place after the firewall was computed over it, while the sanctioned
post-resolve provenance enrichment (rebinding a param with richer free text) still works. And the
raw inference operating point carries no reference for its conf whatever number produced it, so a
caller's own pick is as un-consumable as the documented default.
"""

from __future__ import annotations

import dataclasses

import pytest

from tcip_mcp.pipelines.resolution import (
    DEFAULT_CONF,
    ResolvedBundle,
    UnvalidatedOperatingPointError,
    VALIDATED_FALSE,
    VALIDATED_HELD_OUT,
    check_delivery_gate,
    derived,
    raw_operating_point,
)


def test_a_resolved_param_cannot_be_relabelled_in_place():
    """Rewriting the validity fields of a param that already exists would let an unvalidated
    threshold be laundered into a validated one without producing a new resolution at all, so the
    fields refuse assignment and the firewall keeps holding afterwards."""
    p = derived("conf", 0.62, derived_from="count-vs-conf sweep", requires_validation=True,
                validation_kind="annotations", validated_against=VALIDATED_FALSE)
    with pytest.raises(dataclasses.FrozenInstanceError, match="validated_against"):
        p.validated_against = VALIDATED_HELD_OUT
    with pytest.raises(dataclasses.FrozenInstanceError, match="requires_validation"):
        p.requires_validation = False
    assert p.is_shippable is False
    with pytest.raises(UnvalidatedOperatingPointError, match="requires validation"):
        _ = p.value


def test_enriching_a_resolved_param_with_richer_provenance_keeps_its_value_and_reference():
    """The rail must admit valid work: a bundle is enriched with provenance after it resolves, and
    rebinding a param to carry a fuller description must change neither the value nor the reference
    the delivery gate reads."""
    bundle = ResolvedBundle("bud_opening", "AAAA", {"conf": derived(
        "conf", 0.62, derived_from="count-vs-conf sweep", requires_validation=True,
        validation_kind="annotations", validated_against=VALIDATED_HELD_OUT)})
    bundle.params["conf"] = dataclasses.replace(
        bundle.params["conf"],
        derived_from="count-vs-conf sweep over a disjoint held-out split")
    assert bundle.value("conf") == 0.62
    assert bundle.is_shippable is True
    prov = bundle.to_provenance()["operating_point"]["conf"]
    assert prov["validated_against"] == VALIDATED_HELD_OUT
    assert prov["derived_from"] == "count-vs-conf sweep over a disjoint held-out split"


@pytest.mark.parametrize("conf", [DEFAULT_CONF, 0.87, 0.13])
def test_a_caller_picked_conf_is_firewalled_exactly_like_the_documented_default(conf):
    """A raw inference run's threshold has no per-dataset reference behind it whichever number it
    is: a caller who picked one deliberately has not thereby checked it against anything, so the
    value stays readable only through the acknowledging accessor and the delivery gate refuses."""
    bundle = raw_operating_point(conf=conf, cross_tile_nms=0.35, tiled=False, tile_size=None,
                                 max_dets=750)
    param = bundle.get("conf")
    assert param.requires_validation is True
    assert param.validated_against == VALIDATED_FALSE
    assert param.is_shippable is False
    with pytest.raises(UnvalidatedOperatingPointError, match="cannot be consumed"):
        bundle.value("conf")
    assert param.unvalidated_value(acknowledge_unvalidated=True) == conf

    gate = check_delivery_gate({"operating_point": param.validated_against})
    assert gate.ok is False
    assert gate.unvalidated == ("operating_point",)
    assert gate.stamp == {"operating_point": VALIDATED_FALSE}
