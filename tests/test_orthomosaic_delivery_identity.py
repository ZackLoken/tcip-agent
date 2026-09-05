"""A per-plant orthomosaic delivery resolves detections only through the raster its bucket was
produced on.

The counts in the delivered CSV are attributed to plants by the caller-supplied raster's own
georeferencing, so the raster is part of the measurement, not a convenience argument: a
pixel-identical copy at a moved tiepoint re-attributes every count to a neighbouring plant, and a
far-shifted one reads every plant as an explicit zero. Covers the delivery-time identity check and
the claim-scope dimension the shared delivery gate reconciles from a bucket's own sidecar.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tcip_mcp.pipelines.resolution import (  # noqa: E402
    VALIDATED_FALSE, VALIDATED_HELD_OUT, VALIDATED_SAME_MOSAIC_IDENTITY,
)
from tests.test_orthomosaic_tools import (  # noqa: E402
    _PLANT_PIXELS, TIEPOINT_NATIVE_X, TILE, _bespoke_detection_checkpoint, _plant_grid_csv,
    _plant_registry, _replace_boxes, _write_geo_raster,
)

from tcip_mcp import operationalization as op  # noqa: E402
from tests import _operationalization_fixtures as fx  # noqa: E402

# One detection sitting on the first plant of the grid (pixel 10, 10).
_ON_FIRST_PLANT = [(8.0, 8.0, 12.0, 12.0)]



@pytest.fixture(autouse=True)
def _recorded_meaning(tmp_path):
    """Every per-plant delivery below ships under a trait whose meaning is confirmed.

    Seeded into the project these tests pin as well as the one the autouse pin names, so a
    delivery reads the same registry whichever of the two it resolves against.
    """
    for project_root in (tmp_path, tmp_path / "proj"):
        fx.seed_delivery_traits(project_root)
        fx.confirm_aggregate(project_root, fx.COUNT_TRAIT, op.PER_PLANT_COUNT_AGGREGATE,
                             delivered_phenotype="stem_count", value_keys=["count"])


def _project(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / ".tcip" / "state").mkdir(parents=True, exist_ok=True)


def _raster_bucket(tmp_path, raster_path: Path, boxes) -> Path:
    """A bucket from the real producer: run_inference's whole-raster regime."""
    from tcip_mcp.tools.inference_tools import run_inference

    out_dir = tmp_path / "preds"
    result = run_inference(
        _bespoke_detection_checkpoint(tmp_path), output_dir=str(out_dir),
        raster_path=str(raster_path), conf_threshold=0.0, tile_size=TILE)
    assert "error" not in result, result
    _replace_boxes(Path(result["files"][0]), boxes)
    return out_dir


def _hand_written_bucket(tmp_path, name: str, stamp: dict) -> Path:
    """A bucket whose sidecar is written directly, for the stamps a live run cannot produce.

    A claiming stamp earns a real record over the prediction file below, so a refusal these tests
    assert comes from the claim scope they are about rather than from an unanswered-for claim. The
    bucket sits in the dataset's own predictions layout, since a count claim's covered set is keyed
    relative to a dataset root.
    """
    root = tmp_path / "ds"
    d = root / "predictions" / name
    d.mkdir(parents=True, exist_ok=True)
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    from tests._binding_fixtures import write_bound_sidecar

    json_io.write_annotations(
        str(d / "mosaic.json"),
        [Annotation(subject="0", geometry=BBox(8.0, 8.0, 12.0, 12.0), score=0.9)], 64, 64,
        keep_empty=True)
    write_bound_sidecar(d, stamp, dataset_root=root, experiment_id=f"exp-{name}")
    return d


def _validated_count_stamp(*, claim_scope: str | None = None) -> dict:
    stamp = {"validated": True, "trait": fx.COUNT_TRAIT,
             "operating_point": {"conf": {"value": 0.5, "validated_against": VALIDATED_HELD_OUT}},
             "subject": fx.COUNT_SUBJECT, "attribute": None}
    if claim_scope is not None:
        stamp["claim_scope_validated"] = claim_scope
    return stamp


# ── the delivery-time identity check ─────────────────────────────────────


def test_delivery_refuses_a_pixel_identical_raster_whose_tiepoint_moved(tmp_path, monkeypatch):
    """Identical pixels at a moved tiepoint resolve every detection onto a different plant, so the
    raster is refused rather than silently believed: the content half alone cannot see this."""
    _project(tmp_path, monkeypatch)
    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    moved = tmp_path / "mosaic_moved.tif"
    _write_geo_raster(moved, tiepoint_x=TIEPOINT_NATIVE_X + 20.0)

    bucket = _raster_bucket(tmp_path, raster_path, _ON_FIRST_PLANT)
    plant_csv = _plant_grid_csv(tmp_path, raster_path, _PLANT_PIXELS)

    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    out_csv = tmp_path / "counts.csv"
    refused = deliver_orthomosaic_plant_counts(
        str(bucket), str(moved), _plant_registry(plant_csv), str(out_csv), delivered_phenotype="stem_count")

    assert "error" in refused
    assert "georeferencing mismatch" in refused["error"]
    assert "tiepoint_native_x" in refused["error"]
    assert not out_csv.exists()


def test_delivery_refuses_a_raster_of_different_content(tmp_path, monkeypatch):
    """The content half still refuses on its own: a different mosaic at the same tiepoint."""
    _project(tmp_path, monkeypatch)
    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    other = tmp_path / "other.tif"
    _write_geo_raster(other, seed=7)

    bucket = _raster_bucket(tmp_path, raster_path, _ON_FIRST_PLANT)
    plant_csv = _plant_grid_csv(tmp_path, raster_path, _PLANT_PIXELS)

    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    out_csv = tmp_path / "counts.csv"
    refused = deliver_orthomosaic_plant_counts(
        str(bucket), str(other), _plant_registry(plant_csv), str(out_csv), delivered_phenotype="stem_count")

    assert "error" in refused
    assert "content mismatch" in refused["error"]
    assert not out_csv.exists()


def test_delivery_refuses_a_bucket_that_records_no_raster_identity(tmp_path, monkeypatch):
    """No recorded identity means nothing to check the supplied raster against, so the delivery
    refuses and names the regime that records one rather than trusting whatever it was handed."""
    _project(tmp_path, monkeypatch)
    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    bucket = _hand_written_bucket(tmp_path, "preds_no_identity", _validated_count_stamp())
    plant_csv = _plant_grid_csv(tmp_path, raster_path, _PLANT_PIXELS)

    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    out_csv = tmp_path / "counts.csv"
    refused = deliver_orthomosaic_plant_counts(
        str(bucket), str(raster_path), _plant_registry(plant_csv), str(out_csv), delivered_phenotype="stem_count")

    assert "error" in refused
    assert "run_inference" in refused["error"]
    assert not out_csv.exists()


def test_delivery_resolves_the_raster_it_was_produced_on_then_refuses_on_the_uncalibrated_count(
    tmp_path, monkeypatch,
):
    """The raster the bucket was produced on is admitted by the identity check, with the identity
    recorded by the producer itself rather than written into the fixture by hand: the mapping
    resolves and attributes the detection to its plant. A bare run_inference pass reserved no
    calibration region, so the count operating point never earned a reference and this door takes
    no acknowledgement; the refusal named is the gate's, never the identity check's."""
    _project(tmp_path, monkeypatch)
    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    bucket = _raster_bucket(tmp_path, raster_path, _ON_FIRST_PLANT)

    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

    recorded = read_operating_point_sidecar(bucket)["raster_content_identity"]
    assert recorded["width"] == 64 and recorded["pixel_checksum"]
    assert recorded["geotransform"]["tiepoint_native_x"] == TIEPOINT_NATIVE_X

    plant_csv = _plant_grid_csv(tmp_path, raster_path, _PLANT_PIXELS)

    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    out_csv = tmp_path / "counts.csv"
    refused = deliver_orthomosaic_plant_counts(
        str(bucket), str(raster_path), _plant_registry(plant_csv), str(out_csv), delivered_phenotype="stem_count")

    assert "georeferencing mismatch" not in refused["error"]
    assert "content mismatch" not in refused["error"]
    assert "unvalidated dimension" in refused["error"]
    assert refused["n_mapped"] == 1
    assert not out_csv.exists()


# ── the claim-scope dimension of the shared delivery gate ────────────────


def _aggregated(tmp_path, bucket: Path, *, name: str = "counts.csv"):
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    path, _tail, _event_recorded = export_aggregated_csv(
        [{"plant_id": "plot0", "value": 3, "observations": 1, "value_key": "count",
          "measurement_document": "operating_point", "plant_attribution": "detection"}],
        str(tmp_path / name),
        delivered_phenotype="stem_count", operating_point_validated=VALIDATED_HELD_OUT,
        pred_dirs=[str(bucket)])
    return path


def test_delivery_gate_ships_a_bucket_scoped_to_the_mosaic_it_was_produced_on(tmp_path):
    bucket = _hand_written_bucket(
        tmp_path, "preds", _validated_count_stamp(claim_scope=VALIDATED_SAME_MOSAIC_IDENTITY))
    out = _aggregated(tmp_path, bucket)
    rows = list(csv.DictReader(Path(out).open(newline="")))
    assert rows[0]["operating_point_validated"] == VALIDATED_HELD_OUT


def test_delivery_gate_refuses_a_bucket_whose_claim_scope_cleared_nothing(tmp_path):
    bucket = _hand_written_bucket(
        tmp_path, "preds", _validated_count_stamp(claim_scope=VALIDATED_FALSE))
    with pytest.raises(ValueError, match="claim_scope"):
        _aggregated(tmp_path, bucket)


def test_delivery_gate_refuses_a_claim_scope_borrowed_from_another_dimension(tmp_path):
    """A held-out annotation reference says nothing about which raster produced the bucket, so it
    clears nothing here even though the gate treats it as shippable for the dimension it belongs
    to."""
    bucket = _hand_written_bucket(
        tmp_path, "preds", _validated_count_stamp(claim_scope=VALIDATED_HELD_OUT))
    with pytest.raises(ValueError, match="claim_scope"):
        _aggregated(tmp_path, bucket)


def test_delivery_gate_never_gates_a_bucket_that_records_no_claim_scope(tmp_path):
    """The dimension is operative only for a bucket that records one, so every other delivery is
    unaffected."""
    bucket = _hand_written_bucket(tmp_path, "preds", _validated_count_stamp())
    out = _aggregated(tmp_path, bucket)
    rows = list(csv.DictReader(Path(out).open(newline="")))
    assert rows[0]["operating_point_validated"] == VALIDATED_HELD_OUT


def test_orthomosaic_precondition_precedes_the_raster_identity_refusal(tmp_path, monkeypatch):
    """Two refusals apply to one delivery and the earlier one reports by itself.

    A bucket that cannot vouch for the raster it was produced on says how a number would be
    attributed, which is nothing to answer for while nobody has said what the number means. The
    door runs its own check first for exactly this, rather than inheriting one from its writer.
    """
    _project(tmp_path, monkeypatch)
    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    bucket = _hand_written_bucket(tmp_path, "preds_no_identity", _validated_count_stamp())
    plant_csv = _plant_grid_csv(tmp_path, raster_path, _PLANT_PIXELS)

    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    out_csv = tmp_path / "counts.csv"
    refused = deliver_orthomosaic_plant_counts(
        str(bucket), str(raster_path), _plant_registry(plant_csv), str(out_csv),
        delivered_phenotype="bark_thickness")

    assert "no operationalization is recorded" in refused["error"]
    assert "run_inference" not in refused["error"]
    assert "raster content identity" not in refused["error"]
    assert not out_csv.exists()
