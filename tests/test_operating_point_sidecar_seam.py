"""The prediction bucket's provenance stamps: one store family, one stamp shape, one gate column.

A stamp records what stood behind a bucket's counts. Every producer writes the same shape through
the same locked store, and the delivery gate owns what a deliverable's validation column carries.
"""

from __future__ import annotations

import pytest
import tcip_store

from tcip_annotation.json_io import SIDECAR_FILENAMES
from tcip_mcp.pipelines.resolution import (
    VALIDATED_EXPLICIT_GEOMETRY,
    VALIDATED_FALSE,
    VALIDATED_HELD_OUT,
    VALIDATED_PERSISTED_GEOMETRY,
    Acknowledgement,
    ResolvedBundle,
    ResolvedParam,
    accepted_references,
    block_calibrated_export_operating_point,
    check_delivery_gate,
    default,
    operating_point_stamp,
    read_operating_point_sidecar,
    resolve_tile_size_param,
    sidecar_key,
    tile_size_source_of,
    update_sidecar,
    write_sidecar,
)


def _stamp(*, validated_by=None, **overrides) -> dict:
    fields = dict(
        validated=True,
        tile_size_validated=None,
        shippable_issues=[],
        id_map={"bud": 0},
        subject="bud",
        attribute=None,
        trait="bud_opening",
        dataset_hash="abc123",
        checkpoint="best",
        checkpoint_sha256="f" * 64,
        experiment_id="exp_001",
        images_dir="/data/images",
        raster_path=None,
        produced_at="2026-01-01T00:00:00+00:00",
    )
    fields.update(overrides)
    return operating_point_stamp({"conf": {"value": 0.5}}, validated_by=validated_by, **fields)


def _validated_stamp(tmp_path, **overrides) -> dict:
    """A validated stamp genuinely answered for by a validation record, for tests whose subject is
    the sidecar store's own locking/CAS/codec mechanics rather than the binding check itself."""
    from tests._binding_fixtures import file_validation_record

    return file_validation_record(_stamp(**overrides), dataset_root=tmp_path)


# --- the stamp shape is one shape, whatever the producer had in hand ---

def test_stamp_carries_the_same_keys_whatever_the_producer_supplies():
    from_images = _stamp()
    from_raster = _stamp(images_dir=None, raster_path="/data/mosaic.tif")
    assert set(from_images) == set(from_raster)


def test_stamp_floors_validated_when_the_tile_scale_has_no_basis():
    """A bucket whose tile geometry rests on nothing produced its counts at a scale nothing
    justifies, so it is not a validated bucket however the conf dimension was earned."""
    assert _stamp(validated=True, tile_size_validated=VALIDATED_FALSE)["validated"] is False
    assert _stamp(validated=True,
                  tile_size_validated=VALIDATED_PERSISTED_GEOMETRY)["validated"] is True


def test_stamp_admits_a_producer_specific_field():
    """The one shape is a floor, not a ceiling: a producer with an extra fact still records it."""
    assert (_stamp(calibration_curve_path="/artifacts/curve.json")["calibration_curve_path"]
            == "/artifacts/curve.json")


# --- the declared key set: one union, checked at the two writers ---

def test_stamp_keys_matches_the_constructors_own_returned_keys():
    """STAMP_KEYS is declared by hand, not derived from the signature (a parameter name matching
    its returned key is this constructor's own convention, never a guarantee); this pins the two
    against each other so they cannot drift apart unnoticed."""
    from tcip_mcp.pipelines.resolution import STAMP_KEYS

    assert STAMP_KEYS == set(_stamp())


def test_the_stamp_constructor_marks_its_own_writing_vintage():
    """Every stamp the constructor returns carries schema_version 2, the value assertion the
    key-set pin above cannot make, so the vintage marker cannot silently regress to absence."""
    assert _stamp(validated=False)["schema_version"] == 2


def test_write_sidecar_refuses_an_undeclared_top_level_key(tmp_path):
    """A producer inventing a top-level key nobody declared is refused at the writer, independent
    of the validated_by rail (validated=False here, so only the key-set check is in play)."""
    bucket = tmp_path / "bucket"
    with pytest.raises(ValueError, match="mystery_field"):
        write_sidecar(bucket, _stamp(validated=False, mystery_field="anything"))


def test_write_sidecar_admits_every_declared_extension_key(tmp_path):
    """The rail must admit valid work: every real producer's own addition (the raster export's
    claim_scope_validated/block_calibration/raster_content_identity, the web worker's overlap/
    overlap_source, the review promotion's five, the calibrated run's calibration_curve_path/
    gate_evidence_summary, mask_binarize) writes cleanly through the declared union."""
    from tcip_mcp.pipelines.resolution import STAMP_EXTENSION_KEYS

    bucket = tmp_path / "bucket"
    stamp = _stamp(validated=False, **{key: "x" for key in STAMP_EXTENSION_KEYS})
    write_sidecar(bucket, stamp)
    assert read_operating_point_sidecar(bucket) == stamp


def test_update_sidecar_refuses_an_undeclared_top_level_key(tmp_path):
    bucket = tmp_path / "bucket"
    write_sidecar(bucket, _stamp(validated=False))
    with pytest.raises(ValueError, match="mystery_field"):
        update_sidecar(bucket, lambda stored: {**stored, "mystery_field": "x"})


def test_update_sidecar_admits_a_promotion_over_a_stamp_carrying_a_pre_existing_foreign_key(
    tmp_path,
):
    """The rail must admit valid work: a direct store write or a hand-authored stamp may already
    carry a foreign top-level key (the pre-existing conform item the grounding record names), and a
    later promotion that introduces only declared keys is not refused for a key it did not itself
    write, only for one it introduces."""
    import tcip_store

    from tcip_mcp.pipelines.resolution import sidecar_key

    bucket = tmp_path / "bucket"
    stamp = {**_stamp(validated=False), "hand_authored_field": "legacy"}
    key = sidecar_key(bucket)
    with tcip_store.transaction(key) as txn:
        txn.write(key, stamp)

    ok = update_sidecar(bucket, lambda stored: {**stored, "calibration_curve_path": "/artifacts/curve.json"})

    assert ok is True
    assert read_operating_point_sidecar(bucket)["hand_authored_field"] == "legacy"


def test_the_key_set_rail_is_scoped_to_the_operating_point_document_only(tmp_path):
    """Other sidecar documents (classifier/ordinal/regression) carry no declared shape and are not
    checked here: their own producers already write fields (failures, gate_evidence) this
    constructor never declared."""
    bucket = tmp_path / "bucket"
    write_sidecar(
        bucket, {"validated": False, "trait": "bud_opening", "failures": [], "gate_evidence": {}},
        "ordinal_operating_point",
    )


def test_an_old_vintage_sweep_data_keyed_sidecar_still_reads(tmp_path):
    """Coverage, not a guard: a record an older producer wrote under the retired ``sweep_data``
    key (never rewritten by this migration) reads through the real reader with nothing raising,
    since these unshaped documents carry no key-set rail at all to reject it on."""
    bucket = tmp_path / "bucket"
    write_sidecar(
        bucket, {"validated": False, "trait": "bud_opening", "failures": [], "sweep_data": {"kappa": 0.5}},
        "classifier_operating_point",
    )
    from tcip_mcp.pipelines.resolution import read_classifier_operating_point_sidecar

    stamp = read_classifier_operating_point_sidecar(bucket)
    assert stamp["sweep_data"] == {"kappa": 0.5}


# --- the store: locked, compare-and-set, byte-compatible with what readers expect ---

def test_sidecar_write_and_read_round_trip(tmp_path):
    bucket = tmp_path / "predictions" / "best" / "2026-03-04"
    stamp = _validated_stamp(tmp_path)
    write_sidecar(bucket, stamp)
    assert read_operating_point_sidecar(bucket) == stamp


def test_sidecar_store_refuses_an_unconditional_replace(tmp_path):
    """The stamps are read-modify-written by more than one process (a run writes one, a review
    promotion merges into it), so the unconditional write form is not available on them."""
    from tcip_store.errors import PolicyViolation

    bucket = tmp_path / "bucket"
    stamp = _validated_stamp(tmp_path)
    write_sidecar(bucket, stamp)
    with pytest.raises(PolicyViolation):
        tcip_store.replace(sidecar_key(bucket), stamp)


def test_sidecar_update_merges_against_what_is_stored(tmp_path):
    """The compare-and-set form decides against the stored stamp rather than a copy read earlier,
    so fields the producing run wrote survive a promotion."""
    bucket = tmp_path / "bucket"
    write_sidecar(bucket, _stamp(validated=False))
    wrote = update_sidecar(
        bucket, lambda stored: {**stored, "validated_reference": VALIDATED_HELD_OUT})
    stored = read_operating_point_sidecar(bucket)

    assert wrote is True
    assert stored["validated_reference"] == VALIDATED_HELD_OUT
    assert stored["checkpoint_sha256"] == "f" * 64


def test_sidecar_update_can_decline_to_write(tmp_path):
    bucket = tmp_path / "bucket"
    write_sidecar(bucket, _validated_stamp(tmp_path, validated=True))

    assert update_sidecar(bucket, lambda stored: None) is False
    assert read_operating_point_sidecar(bucket)["validated"] is True


def test_a_stamp_field_that_json_cannot_hold_is_refused_at_the_sidecar_writer(tmp_path):
    """A stamp is assembled by its producer and carries a producer's own extra fields, so the
    check sits where every sidecar write passes rather than at each producer.

    A stamp is what a reviewer reconstructs an operating point from, so a field that turned
    into its repr would be provenance that reads as recorded fact.
    """
    from datetime import datetime, timezone
    from pathlib import Path

    bucket = tmp_path / "bucket"

    with pytest.raises(TypeError) as refused:
        write_sidecar(bucket, _stamp(produced_at=datetime.now(timezone.utc)))
    assert "stamp.produced_at" in str(refused.value)
    assert read_operating_point_sidecar(bucket) is None

    write_sidecar(bucket, _validated_stamp(tmp_path))
    with pytest.raises(TypeError) as update_refused:
        update_sidecar(bucket, lambda stored: {**stored, "raster_path": Path("ortho.tif")})
    assert "stamp.raster_path" in str(update_refused.value)
    assert read_operating_point_sidecar(bucket)["raster_path"] is None


def test_an_ordinary_stamp_is_still_written_and_merged(tmp_path):
    """The refusal above must not cost a real calibration its stamp or its promotion."""
    bucket = tmp_path / "bucket"

    write_sidecar(bucket, _validated_stamp(tmp_path))
    wrote = update_sidecar(bucket, lambda stored: {**stored, "raster_path": "ortho.tif"})

    assert wrote is True
    stored = read_operating_point_sidecar(bucket)
    assert stored["raster_path"] == "ortho.tif"
    assert stored["checkpoint_sha256"] == "f" * 64


def test_sidecar_bytes_are_the_canonical_record_spelling(tmp_path):
    """A stamp is spelled the way every other record is, so a breeder opening one and a
    reader parsing one meet the same document.

    Bound to the file backend: the sidecar is a record, so this is a claim about the bytes
    the store lands on disk, which only the file backend answers directly.
    """
    from tcip_store import RECORD_JSON
    from tcip_store.file_backend import FileBackend

    tcip_store.bind(FileBackend())
    bucket = tmp_path / "bucket"
    stamp = _validated_stamp(tmp_path)
    write_sidecar(bucket, stamp)

    assert (bucket / "operating_point.json").read_bytes() == RECORD_JSON.encode(stamp)


def test_unreadable_stamp_reads_as_absent_rather_than_raising(tmp_path):
    """An unreadable stamp floors its dimension to unvalidated at every reconciler, the safe
    direction; raising would take down a delivery gate that has a well-defined answer."""
    bucket = tmp_path / "bucket"
    bucket.mkdir()
    (bucket / "operating_point.json").write_text("{ not json", encoding="utf-8")

    assert read_operating_point_sidecar(bucket) is None


def test_sidecar_key_names_the_declared_documents(tmp_path):
    """A stamp this platform does not declare is refused rather than silently addressed, and every
    declared one still resolves."""
    for document in ("operating_point", "classifier_operating_point", "ordinal_operating_point",
                     "regression_operating_point", "resolve_scale"):
        assert sidecar_key(tmp_path, document).parts == (document,)
    with pytest.raises(ValueError, match="not a prediction-bucket stamp"):
        sidecar_key(tmp_path, "made_up_operating_point")


def test_declared_stamp_filenames_cover_every_declared_document(tmp_path):
    assert SIDECAR_FILENAMES == frozenset({
        "operating_point.json", "classifier_operating_point.json",
        "ordinal_operating_point.json", "regression_operating_point.json",
        "resolve_scale.json"})


# --- the delivery gate owns the floored column value ---

def test_column_stamp_floors_on_a_dimension_with_no_column_of_its_own():
    gate = check_delivery_gate({"operating_point": VALIDATED_HELD_OUT, "tile_size": VALIDATED_FALSE},
                               allow_unvalidated_staging=True)

    assert gate.ok is True
    assert gate.stamp["operating_point"] == VALIDATED_HELD_OUT
    assert gate.column_stamp("operating_point") == VALIDATED_FALSE


def test_column_stamp_is_not_floored_by_a_separately_stamped_dimension():
    """A dimension the deliverable stamps into a column of its own reports itself there, so it must
    not also drag down the column beside it."""
    gate = check_delivery_gate(
        {"operating_point": VALIDATED_HELD_OUT, "classifier": VALIDATED_FALSE},
        acknowledgement=Acknowledgement(acknowledged_by="user:tester", reason="known unvalidated"))

    assert gate.column_stamp("operating_point", own_column=("classifier",)) == VALIDATED_HELD_OUT
    assert gate.stamp["classifier"] == VALIDATED_FALSE


def test_column_stamp_reports_the_cleared_reference_when_everything_cleared():
    gate = check_delivery_gate({"operating_point": VALIDATED_HELD_OUT,
                                "tile_size": VALIDATED_PERSISTED_GEOMETRY})

    assert gate.column_stamp("operating_point") == VALIDATED_HELD_OUT


# --- the geometry reference and its source are one mapping, in both directions ---

@pytest.mark.parametrize("source", ["derived", "explicit"])
def test_tile_size_source_round_trips_through_its_reference(source):
    derived_from = (
        "stated on a checkpoint that records no tile geometry" if source == "explicit" else None)
    param = resolve_tile_size_param(512, tiled=True, tile_size_source=source,
                                    tile_size_derived_from=derived_from)

    assert tile_size_source_of(param.validated_against, tile_size=512) == source


def test_tile_size_source_of_recovers_each_geometry_tier_from_its_own_reference():
    native_param = resolve_tile_size_param(512, tiled=True, tile_size_source="native_ratio",
                                           tile_size_derived_from=None)
    assert tile_size_source_of(native_param.validated_against, tile_size=512) == "native_ratio"

    persisted_param = resolve_tile_size_param(512, tiled=True, tile_size_source="derived",
                                              tile_size_derived_from=None)
    assert tile_size_source_of(persisted_param.validated_against, tile_size=512) == "derived"

    # A reference nothing in the current vocabulary answers for: recorded (the edge is kept), never
    # laundered back into the tier a bare source label might otherwise suggest.
    assert tile_size_source_of(VALIDATED_FALSE, tile_size=512) == "recorded"


def test_no_tile_size_reads_back_as_the_documented_default():
    assert tile_size_source_of(None, tile_size=None) == "default"


def test_accepted_geometry_references_are_the_three_the_resolver_stamps_in_strength_order():
    import tcip_mcp.pipelines.resolution as resolution_mod

    native_ref = getattr(resolution_mod, "VALIDATED_NATIVE_FRAME_GEOMETRY", None)
    assert accepted_references("geometry") == (
        VALIDATED_PERSISTED_GEOMETRY, native_ref, VALIDATED_EXPLICIT_GEOMETRY)


def test_geometry_reference_strength_matches_the_mapping_it_is_defined_beside():
    import tcip_mcp.pipelines.resolution as resolution_mod

    strength = getattr(resolution_mod, "GEOMETRY_REFERENCE_STRENGTH", None)
    native_ref = getattr(resolution_mod, "VALIDATED_NATIVE_FRAME_GEOMETRY", None)
    assert strength is not None
    assert set(strength) == {VALIDATED_PERSISTED_GEOMETRY, native_ref, VALIDATED_EXPLICIT_GEOMETRY}
    # A member added to the lookup mapping and not to the strength order, or the reverse, fails.
    assert set(strength) == set(resolution_mod._GEOMETRY_REFERENCE_BY_SOURCE.values())
    assert len(strength) == len(resolution_mod._GEOMETRY_REFERENCE_BY_SOURCE)


# --- the block-calibrated whole-mosaic regime ---

def _block_bundle() -> ResolvedBundle:
    return ResolvedBundle(trait="bud_opening", dataset_hash="mosaic1", params={
        "conf": ResolvedParam("conf", 0.42, source="derived", derived_from="count-unbiased",
                              requires_validation=True, validation_kind="annotations",
                              validated_against=VALIDATED_HELD_OUT),
        "cross_tile_nms": default("cross_tile_nms", 0.25),
        "tiled": default("tiled", True),
        "tile_size": default("tile_size", 640),
        "max_dets": ResolvedParam("max_dets", 37, source="derived", derived_from="band density"),
    })


def test_block_calibrated_export_ships_at_what_the_reserved_bands_measured():
    block = _block_bundle()
    export = block_calibrated_export_operating_point(
        block, trait="bud_opening", tile_size=640, tile_size_source="derived")

    assert export.get("conf") is block.get("conf")
    assert export.get("cross_tile_nms") is block.get("cross_tile_nms")
    assert export.dataset_hash == "mosaic1"


def test_block_calibrated_export_does_not_inherit_the_band_scoped_detection_cap():
    """The block bundle's cap is one reserved band's density; adopting it would truncate the count
    over the whole mosaic, which is the phenotype."""
    export = block_calibrated_export_operating_point(
        _block_bundle(), trait="bud_opening", tile_size=640, tile_size_source="derived")

    assert export.get("max_dets").value is None
    assert "not transferred" in export.get("max_dets").derived_from


def test_block_calibrated_export_gates_its_tile_scale_like_every_other_door():
    export = block_calibrated_export_operating_point(
        _block_bundle(), trait="bud_opening", tile_size=640, tile_size_source="derived")
    tile = export.get("tile_size")

    assert tile.requires_validation is True
    assert tile.validated_against == VALIDATED_PERSISTED_GEOMETRY
    assert export.get("tiled").value is True
