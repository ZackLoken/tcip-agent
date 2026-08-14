"""The tile-geometry dimension is decided by the tile scale's recorded reference, not its source.

A tile edge derived from a checkpoint's own uniform untiled training size is a real basis to tile at
all, so it records ``source="derived"`` and keeps a concrete edge, while its reference stays
``false``: nothing independently confirmed that scale. That is the one tier where the source label
and the reference disagree, so a gate that reads the source label reports a fabricated scale as a
persisted training geometry.
"""

from __future__ import annotations

import json
from pathlib import Path

from tcip_mcp.pipelines.resolution import (
    VALIDATED_EXPLICIT_GEOMETRY,
    VALIDATED_FALSE,
    VALIDATED_PERSISTED_GEOMETRY,
    raw_operating_point,
    reconcile_tile_size_validity,
    tile_size_gate_flag,
)


def _provenance(*, tile_size: int | None, tile_size_source: str) -> dict:
    """The operating-point mapping a tiled run stamps, built through the real resolver."""
    bundle = raw_operating_point(conf=0.62, cross_tile_nms=0.35, tiled=True, tile_size=tile_size,
                                 max_dets=750, tile_size_source=tile_size_source)
    return bundle.to_provenance()["operating_point"]


def _bucket(path: Path, provenance: dict) -> str:
    """A written prediction bucket carrying that run's own operating-point provenance."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "operating_point.json").write_text(
        json.dumps({"validated": True, "operating_point": provenance}), encoding="utf-8")
    return str(path)


def test_a_native_ratio_tile_edge_never_reads_back_as_a_persisted_geometry():
    prov = _provenance(tile_size=300, tile_size_source="native_ratio")
    assert prov["tile_size"]["source"] == "derived"
    assert prov["tile_size"]["value"] == 300
    assert tile_size_gate_flag(prov) == VALIDATED_FALSE


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


def test_a_native_ratio_bucket_floors_a_delivery_assembled_beside_a_persisted_one(tmp_path):
    """A delivery spanning both tiers is only as grounded as the weaker bucket, and the refusal
    names the bucket whose scale nothing confirmed."""
    persisted = _bucket(tmp_path / "b1", _provenance(tile_size=224, tile_size_source="derived"))
    native = _bucket(tmp_path / "b2", _provenance(tile_size=300, tile_size_source="native_ratio"))
    recon = reconcile_tile_size_validity([persisted, native])
    assert recon["operative"] is True
    assert recon["validated"] == VALIDATED_FALSE
    assert recon["unvalidated_buckets"] == [native]
    assert recon["per_bucket"] == {persisted: VALIDATED_PERSISTED_GEOMETRY,
                                   native: VALIDATED_FALSE}
