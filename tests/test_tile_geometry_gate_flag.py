"""The tile-geometry dimension is decided by the tile scale's recorded reference, not its source.

A tile edge derived from a checkpoint's own uniform untiled training size is a real basis to tile at
all, so it records ``source="derived"`` and keeps a concrete edge, while its reference stays
``false``: nothing independently confirmed that scale. That is the one tier where the source label
and the reference disagree, so a gate that reads the source label reports a fabricated scale as a
persisted training geometry.
"""

from __future__ import annotations

from pathlib import Path

import tcip_store as ts

from tcip_mcp.pipelines.resolution import (
    VALIDATED_EXPLICIT_GEOMETRY,
    VALIDATED_FALSE,
    VALIDATED_PERSISTED_GEOMETRY,
    raw_operating_point,
    reconcile_tile_size_validity,
    sidecar_key,
    tile_size_gate_flag,
)


def _provenance(*, tile_size: int | None, tile_size_source: str) -> dict:
    """The operating-point mapping a tiled run stamps, built through the real resolver."""
    derived_from = (
        "stated on a checkpoint that records no tile geometry"
        if tile_size_source == "explicit" else None)
    bundle = raw_operating_point(conf=0.62, cross_tile_nms=0.35, tiled=True, tile_size=tile_size,
                                 max_dets=750, tile_size_source=tile_size_source,
                                 tile_size_derived_from=derived_from)
    return bundle.to_provenance()["operating_point"]


def _bucket(path: Path, provenance: dict) -> str:
    """A written prediction bucket carrying that run's own operating-point provenance.

    Written through the seam, bypassing the writer-side claim rail: this stamp claims
    ``validated`` with no ``validated_by`` pointer on purpose, since the dimension under test
    here is the tile-geometry gate flag, not the validation-record binding.
    """
    path.mkdir(parents=True, exist_ok=True)
    key = sidecar_key(path, "operating_point")
    with ts.transaction(key) as txn:
        txn.write(key, {"validated": True, "operating_point": provenance})
    return str(path)


def test_a_native_ratio_tile_edge_never_reads_back_as_a_persisted_geometry():
    import tcip_mcp.pipelines.resolution as resolution_mod

    native_ref = getattr(resolution_mod, "VALIDATED_NATIVE_FRAME_GEOMETRY", None)
    prov = _provenance(tile_size=300, tile_size_source="native_ratio")
    assert prov["tile_size"]["source"] == "derived"
    assert prov["tile_size"]["value"] == 300
    flag = tile_size_gate_flag(prov)
    assert flag == native_ref
    assert flag != VALIDATED_PERSISTED_GEOMETRY


def test_a_run_with_no_basis_for_a_scale_carries_no_tile_edge_at_all():
    """The other unvalidated tier, and the contrast that makes the native-ratio one distinguishable:
    with nothing to justify a scale the edge itself is dropped rather than kept at the caller's
    number."""
    prov = _provenance(tile_size=640, tile_size_source="unavailable")
    assert prov["tile_size"]["value"] is None
    assert tile_size_gate_flag(prov) == VALIDATED_FALSE


def test_a_persisted_training_geometry_clears_the_tile_gate():
    prov = _provenance(tile_size=224, tile_size_source="derived")
    assert prov["tile_size"]["value"] == 224
    assert tile_size_gate_flag(prov) == VALIDATED_PERSISTED_GEOMETRY


def test_a_caller_stated_tile_edge_clears_the_tile_gate():
    prov = _provenance(tile_size=512, tile_size_source="explicit")
    assert prov["tile_size"]["value"] == 512
    assert tile_size_gate_flag(prov) == VALIDATED_EXPLICIT_GEOMETRY


def test_a_native_ratio_bucket_ranks_against_the_other_tiers_and_floors_beside_no_basis(tmp_path):
    """A delivery spanning tiers travels under the weakest one present (never a stronger one some
    other bucket earned); a bucket with no basis at all still floors the whole dimension."""
    import tcip_mcp.pipelines.resolution as resolution_mod

    native_ref = getattr(resolution_mod, "VALIDATED_NATIVE_FRAME_GEOMETRY", None)
    persisted = _bucket(tmp_path / "b1", _provenance(tile_size=224, tile_size_source="derived"))
    native = _bucket(tmp_path / "b2", _provenance(tile_size=300, tile_size_source="native_ratio"))
    explicit = _bucket(tmp_path / "b3", _provenance(tile_size=512, tile_size_source="explicit"))
    no_basis = _bucket(tmp_path / "b4", _provenance(tile_size=640, tile_size_source="unavailable"))

    persisted_native = reconcile_tile_size_validity([persisted, native])
    assert persisted_native["validated"] == native_ref

    native_explicit = reconcile_tile_size_validity([native, explicit])
    assert native_explicit["validated"] == VALIDATED_EXPLICIT_GEOMETRY

    native_no_basis = reconcile_tile_size_validity([native, no_basis])
    assert native_no_basis["operative"] is True
    assert native_no_basis["validated"] == VALIDATED_FALSE
    assert native_no_basis["unvalidated_buckets"] == [no_basis]


def test_a_delivery_of_only_native_frame_buckets_reads_back_as_native_never_persisted(tmp_path):
    """The weakest-present pick must not default to the strongest reference when nothing stronger
    is in the set: an all-native-frame delivery reads back as the native reference, never silently
    upgraded to persisted_training_geometry the way a naive two-branch (explicit/else) reconciler
    would once the native tier became a real reference."""
    import tcip_mcp.pipelines.resolution as resolution_mod

    native_ref = getattr(resolution_mod, "VALIDATED_NATIVE_FRAME_GEOMETRY", None)
    a = _bucket(tmp_path / "a", _provenance(tile_size=64, tile_size_source="native_ratio"))
    b = _bucket(tmp_path / "b", _provenance(tile_size=64, tile_size_source="native_ratio"))

    recon = reconcile_tile_size_validity([a, b])

    assert recon["validated"] == native_ref
    assert recon["validated"] != VALIDATED_PERSISTED_GEOMETRY
