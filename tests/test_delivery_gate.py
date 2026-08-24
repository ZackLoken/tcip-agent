"""The generic delivery doors are gated by one shared refuse-or-stamp check.

Covers the single ``check_delivery_gate`` helper and its retrofit onto the previously-ungated
writers/tools: ``export_detection_csv`` / ``export_aggregated_csv`` (writer-level, no MCP wrapper)
and ``tabulate_counts`` (reads the run's resolved validity, not a caller string). The phenology
doors' gate behavior is pinned in the Phase-0 measurement goldens; here we pin the doors newly
gated, plus the escape hatch (acknowledge_unvalidated) that ships an honestly-flagged provisional CSV.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tcip_mcp.pipelines import resolution as res
from tcip_mcp.pipelines.resolution import (
    VALIDATED_FALSE,
    VALIDATED_HELD_OUT,
    VALIDATED_REVIEW_CONFIRMED,
    check_delivery_gate,
)
from tcip_mcp import operationalization as op
from tests import _operationalization_fixtures as fx
from tests._binding_fixtures import write_bound_sidecar, write_prediction


@pytest.fixture(autouse=True)
def _recorded_meaning(tmp_path):
    """Every delivery below ships under a trait whose delivered number has a confirmed meaning.

    The doors refuse a number nobody defined, which is a different question from the one this
    module is about, so the meaning is on record here and the evidence gate stays the subject.
    """
    fx.seed_delivery_traits(tmp_path)
    fx.seed_confirmed_count(tmp_path)
    for trait, keys in (("stem_count", ["count"]), ("plant_surface_area", ["area_mm2"]),
                        ("fruit_diameter", ["fruit_diameter"])):
        fx.confirm_aggregate(tmp_path, fx.DELIVERY_TRAIT_BY_PHENOTYPE[trait],
                             op.PER_PLANT_COUNT_AGGREGATE,
                             delivered_phenotype=trait, value_keys=keys)
    fx.confirm_aggregate(tmp_path, "fruit_diameter", op.PER_PLANT_REGRESSION_AGGREGATE,
                         delivered_phenotype="fruit_diameter", value_keys=["fruit_diameter"])
    fx.confirm_aggregate(tmp_path, "astringency", op.PER_PLANT_ORDINAL_AGGREGATE,
                         delivered_phenotype="astringency", value_keys=["astringency"])


# ── the shared helper ─────────────────────────────────────────────────────

def test_gate_passes_when_every_dimension_validated():
    g = check_delivery_gate({"classifier": VALIDATED_HELD_OUT,
                             "operating_point": VALIDATED_REVIEW_CONFIRMED})
    assert g.ok is True
    assert g.unvalidated == ()
    assert g.stamp == {"classifier": VALIDATED_HELD_OUT,
                       "operating_point": VALIDATED_REVIEW_CONFIRMED}


def test_gate_refuses_a_bare_unvalidated_dimension():
    g = check_delivery_gate({"operating_point": None})
    assert g.ok is False
    assert g.unvalidated == ("operating_point",)
    assert g.stamp == {"operating_point": VALIDATED_FALSE}
    assert "acknowledge_unvalidated" in g.reason


def test_gate_acknowledge_ships_but_stamps_false():
    g = check_delivery_gate({"operating_point": "false"}, acknowledge_unvalidated=True)
    assert g.ok is True
    # the acknowledged dimension still travels stamped false, never silently upgraded
    assert g.stamp == {"operating_point": VALIDATED_FALSE}


@pytest.mark.parametrize("dimension, reference", [
    ("measurement", res.VALIDATED_SAME_MOSAIC_IDENTITY),
    ("operating_point", res.VALIDATED_PERSISTED_GEOMETRY),
    ("classifier", res.VALIDATED_PHYSICAL_MEASUREMENT),
    ("tile_size", res.VALIDATED_PHYSICAL_MEASUREMENT),
    ("scale", VALIDATED_HELD_OUT),
    ("claim_scope", VALIDATED_HELD_OUT),
])
def test_a_wrong_kind_reference_floors_the_dimension_it_cannot_clear(dimension, reference):
    """Each dimension is cleared only by references of its own kind: a raster-scope identity says
    nothing about a count, a training geometry nothing about an operating point, an annotation
    reference nothing about which raster a bucket's predictions were produced on."""
    g = check_delivery_gate({dimension: reference})
    assert g.ok is False
    assert g.unvalidated == (dimension,)
    assert g.stamp == {dimension: VALIDATED_FALSE}


@pytest.mark.parametrize("dimension, reference", [
    ("operating_point", VALIDATED_HELD_OUT),
    ("operating_point", VALIDATED_REVIEW_CONFIRMED),
    ("measurement", VALIDATED_HELD_OUT),
    ("measurement", VALIDATED_REVIEW_CONFIRMED),
    ("classifier", VALIDATED_HELD_OUT),
    ("classifier", VALIDATED_REVIEW_CONFIRMED),
    ("tile_size", res.VALIDATED_PERSISTED_GEOMETRY),
    # getattr: absent at the baseline (pre-promotion), present after; a baseline run then floors
    # this case on assertion (None clears nothing) rather than erroring at collection.
    ("tile_size", getattr(res, "VALIDATED_NATIVE_FRAME_GEOMETRY", None)),
    ("tile_size", res.VALIDATED_EXPLICIT_GEOMETRY),
    ("scale", res.VALIDATED_PHYSICAL_MEASUREMENT),
    ("claim_scope", res.VALIDATED_SAME_MOSAIC_IDENTITY),
])
def test_every_dimension_still_clears_with_its_own_kind(dimension, reference):
    """The kind-aware gate admits every legitimate pairing a real delivery door produces."""
    g = check_delivery_gate({dimension: reference})
    assert g.ok is True
    assert g.stamp == {dimension: reference}


def test_acknowledged_wrong_kind_reference_ships_stamped_false():
    g = check_delivery_gate({"measurement": res.VALIDATED_SAME_MOSAIC_IDENTITY},
                            acknowledge_unvalidated=True)
    assert g.ok is True
    assert g.stamp == {"measurement": VALIDATED_FALSE}


def test_an_unknown_dimension_name_refuses_loudly():
    """A dimension the gate has no reference vocabulary for is a new door under construction, not
    a delivery to wave through under the any-shippable-reference union."""
    with pytest.raises(ValueError, match="provenance"):
        check_delivery_gate({"provenance": VALIDATED_HELD_OUT})


def test_the_refusal_names_the_failed_dimensions_own_references():
    """The refusal must point at what actually clears the failed dimension: telling a tile-size
    failure to collect annotations sends the caller after a reference that clears nothing."""
    native_ref = getattr(res, "VALIDATED_NATIVE_FRAME_GEOMETRY", "<no-such-reference>")
    g = check_delivery_gate({"tile_size": VALIDATED_FALSE})
    assert res.VALIDATED_PERSISTED_GEOMETRY in g.reason
    assert native_ref in g.reason
    assert res.VALIDATED_EXPLICIT_GEOMETRY in g.reason
    assert VALIDATED_HELD_OUT not in g.reason


# ── export_detection_csv (writer) refuses a bare write ─────────────────────

def test_export_detection_csv_refuses_bare_write(tmp_path):
    from tcip_mcp.pipelines.postprocessing.export import export_detection_csv

    with pytest.raises(ValueError, match="unvalidated measurement"):
        export_detection_csv([{"image": "a.jpg", "count": 3}], str(tmp_path / "o.csv"),
                             trait=fx.COUNT_TRAIT)


def test_export_detection_csv_acknowledge_stamps_false(tmp_path):
    from tcip_mcp.pipelines.postprocessing.export import export_detection_csv

    out = tmp_path / "o.csv"
    export_detection_csv([{"image": "a.jpg", "count": 3}], str(out),
                         trait=fx.COUNT_TRAIT, acknowledge_unvalidated=True)
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["measurement_validated"] == VALIDATED_FALSE


# ── export_detection_csv reconciles pred_dirs against on-disk sidecars ─────

def _detection_bucket(tmp_path, name, *, validated, ref=VALIDATED_HELD_OUT, conf=0.6,
                      tile_size_prov=None):
    root = tmp_path / "ds"
    d = root / "predictions" / name
    write_prediction(d, "img_a")
    op = {"conf": {"value": conf, "validated_against": ref if validated else VALIDATED_FALSE}}
    if tile_size_prov is not None:
        op["tile_size"] = tile_size_prov
    stamp = {"validated": validated, "trait": fx.COUNT_TRAIT, "operating_point": op}
    if validated:
        write_bound_sidecar(d, stamp, dataset_root=root, experiment_id=f"exp-{name}")
    else:
        (d / "operating_point.json").write_text(json.dumps(stamp), encoding="utf-8")
    return str(d)


def test_export_detection_csv_reconciles_sidecar_floor(tmp_path):
    # A caller-asserted measurement_validated cannot open the gate when the bucket it names has no
    # readable/validated sidecar backing it: pred_dirs reconciles from disk, never trusts the string.
    from tcip_mcp.pipelines.postprocessing.export import export_detection_csv

    bucket = _detection_bucket(tmp_path, "preds", validated=False)
    with pytest.raises(ValueError, match="unvalidated measurement"):
        export_detection_csv([{"image": "a.jpg", "count": 3}], str(tmp_path / "o.csv"),
                             trait=fx.COUNT_TRAIT,
                             measurement_validated=VALIDATED_HELD_OUT, pred_dirs=[bucket])


def test_export_detection_csv_pred_dirs_ships_when_bucket_validated(tmp_path):
    from tcip_mcp.pipelines.postprocessing.export import export_detection_csv

    bucket = _detection_bucket(tmp_path, "preds", validated=True)
    out = tmp_path / "o.csv"
    export_detection_csv([{"image": "a.jpg", "count": 3, "scores": [0.9]}], str(out),
                         trait=fx.COUNT_TRAIT,
                         measurement_validated=VALIDATED_HELD_OUT, pred_dirs=[bucket])
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["measurement_validated"] == VALIDATED_HELD_OUT


def test_export_detection_csv_floors_a_stamp_earned_for_a_different_trait(tmp_path):
    """A count stamp validated for one trait must not answer for a delivery under a different
    trait: the refusal names the sidecar and both traits."""
    from tcip_mcp.pipelines.postprocessing.export import export_detection_csv

    other_trait = "astringency"
    record = op.state_operationalization(
        tmp_path, other_trait, op.PER_IMAGE_COUNT,
        statement="how many astringent structures the model finds in one frame",
        mechanism="the calibrated detector over whole frames at the derived operating point",
        measured_subject=fx.COUNT_SUBJECT, delivered_phenotypes=[],
    )
    fx.confirm(tmp_path, other_trait, op.PER_IMAGE_COUNT, record)

    bucket = _detection_bucket(tmp_path, "preds", validated=True)  # stamped trait=fx.COUNT_TRAIT
    with pytest.raises(ValueError) as exc:
        export_detection_csv([{"image": "a.jpg", "count": 3}], str(tmp_path / "o.csv"),
                             trait=other_trait,
                             measurement_validated=VALIDATED_HELD_OUT, pred_dirs=[bucket])
    message = str(exc.value)
    assert bucket in message
    assert fx.COUNT_TRAIT in message and other_trait in message


def test_export_detection_csv_pred_dirs_gates_fabricated_tile_size(tmp_path):
    # A tiled bucket with no persisted training geometry must gate the delivery even though the
    # conf operating point itself cleared, mirroring export_aggregated_csv's tile_size dimension.
    from tcip_mcp.pipelines.postprocessing.export import export_detection_csv

    bucket = _detection_bucket(
        tmp_path, "preds", validated=True,
        tile_size_prov={"value": 640, "requires_validation": True,
                        "validation_kind": "geometry", "validated_against": VALIDATED_FALSE},
    )
    with pytest.raises(ValueError, match="unvalidated measurement"):
        export_detection_csv([{"image": "a.jpg", "count": 3}], str(tmp_path / "o.csv"),
                             trait=fx.COUNT_TRAIT,
                             measurement_validated=VALIDATED_HELD_OUT, pred_dirs=[bucket])


def test_a_wrong_kind_assertion_floors_a_valid_bucket(tmp_path):
    """The asserted string may only lower the on-disk result, and a wrong-kind assertion is not a
    real assertion about this dimension: it floors rather than being silently ignored while the
    on-disk reference ships as if the caller had asserted nothing."""
    from tcip_mcp.pipelines.postprocessing.export import export_detection_csv

    bucket = _detection_bucket(tmp_path, "preds", validated=True)
    recon = res.reconcile_operating_point_validity(
        [bucket], trait=fx.COUNT_TRAIT, asserted=res.VALIDATED_SAME_MOSAIC_IDENTITY)
    assert recon["validated"] == VALIDATED_FALSE

    with pytest.raises(ValueError, match="unvalidated measurement"):
        export_detection_csv([{"image": "a.jpg", "count": 3}], str(tmp_path / "o.csv"),
                             trait=fx.COUNT_TRAIT,
                             measurement_validated=res.VALIDATED_SAME_MOSAIC_IDENTITY,
                             pred_dirs=[bucket])


def test_export_detection_csv_omitted_pred_dirs_floors_to_unvalidated(tmp_path):
    """No buckets to reconcile from: nothing on disk backs the caller's string, so the measurement
    dimension floors and refuses, mirroring export_aggregated_csv's no-pred_dirs path. The
    explicit acknowledge is the only delivery route, and it stamps false."""
    from tcip_mcp.pipelines.postprocessing.export import export_detection_csv

    out = tmp_path / "o.csv"
    with pytest.raises(ValueError, match="unvalidated measurement"):
        export_detection_csv([{"image": "a.jpg", "count": 3, "scores": [0.9]}], str(out),
                             trait=fx.COUNT_TRAIT, measurement_validated=VALIDATED_HELD_OUT)
    assert not out.exists()

    export_detection_csv([{"image": "a.jpg", "count": 3, "scores": [0.9]}], str(out),
                         trait=fx.COUNT_TRAIT, measurement_validated=VALIDATED_HELD_OUT,
                         acknowledge_unvalidated=True)
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["measurement_validated"] == VALIDATED_FALSE


# ── export_aggregated_csv (writer) refuses a bare write ────────────────────

def test_export_aggregated_csv_refuses_bare_write(tmp_path):
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    with pytest.raises(ValueError, match="unvalidated measurement"):
        export_aggregated_csv(
            [{"plant_id": "p1", "value": 5, "observations": 2, "value_key": "count",
             "measurement_document": "operating_point"}],
            str(tmp_path / "o.csv"), trait_name="stem_count")


def test_export_aggregated_csv_reconciles_sidecar_floor(tmp_path):
    # A count trait reconciles its measurement validity from the prediction buckets' sidecars; a
    # bucket with no operating_point.json floors to false and refuses (never trusts a caller string).
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    bucket = tmp_path / "preds"
    bucket.mkdir()
    with pytest.raises(ValueError, match="unvalidated measurement"):
        export_aggregated_csv(
            [{"plant_id": "p1", "value": 5, "observations": 2, "value_key": "count",
             "measurement_document": "operating_point"}],
            str(tmp_path / "o.csv"), trait_name="stem_count",
            measurement_validated=VALIDATED_HELD_OUT, pred_dirs=[str(bucket)])


def test_export_aggregated_csv_continuous_trait_bare_string_never_trusted(tmp_path):
    # A continuous/ordinal trait has no on-disk measurement-validity producer, so a bare
    # caller-asserted measurement_validated string, with no pred_dirs to reconcile against, is
    # never trusted directly: it refuses without an explicit acknowledge.
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    out = tmp_path / "o.csv"
    with pytest.raises(ValueError):
        export_aggregated_csv(
            [{"plant_id": "p1", "value": 4.2, "observations": 3,
              "value_key": "fruit_diameter", "measurement_document": "regression_operating_point"}],
            str(out), trait_name="fruit_diameter", measurement_validated=VALIDATED_HELD_OUT)


def test_export_aggregated_csv_continuous_trait_ships_provisional_when_acknowledged(tmp_path):
    # The honest provisional path still ships (stamped false); it just can't masquerade as
    # validated on a bare string.
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    out = tmp_path / "o.csv"
    export_aggregated_csv(
        [{"plant_id": "p1", "value": 4.2, "observations": 3, "value_key": "fruit_diameter",
         "measurement_document": "regression_operating_point"}],
        str(out), trait_name="fruit_diameter", measurement_validated=VALIDATED_HELD_OUT,
        acknowledge_unvalidated=True)
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["measurement_validated"] == VALIDATED_FALSE


# ── export_aggregated_csv wired to the ordinal/regression sidecar producer ────

def _scalar_bucket(tmp_path, name, task, *, validated, ref=VALIDATED_HELD_OUT, criterion="r_squared",
                   trait="catkin"):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    document = f"{task}_operating_point"
    stamp = {
        "validated": validated, "trait": trait,
        "operating_point": {task: {"validated_against": ref if validated else VALIDATED_FALSE,
                                   "criterion": criterion}},
    }
    if validated:
        write_bound_sidecar(d, stamp, document=document, dataset_root=tmp_path,
                            experiment_id=f"exp-{name}-{task}")
    else:
        (d / f"{document}.json").write_text(json.dumps(stamp), encoding="utf-8")
    return str(d)


def test_export_aggregated_csv_ordinal_trait_ships_when_sidecar_validated(tmp_path):
    # An ordinal trait with a genuinely validated sidecar (calibrate_ordinal_regression_operating_point's
    # producer) ships as validated, not floored to VALIDATED_FALSE by the unconditional no-producer path.
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    bucket = _scalar_bucket(tmp_path, "preds", "ordinal", validated=True, trait="astringency")
    out = tmp_path / "o.csv"
    export_aggregated_csv(
        [{"plant_id": "p1", "value": 2, "observations": 3, "value_key": "astringency",
         "measurement_document": "ordinal_operating_point"}],
        str(out), trait_name="astringency", measurement_validated=VALIDATED_HELD_OUT,
        pred_dirs=[bucket])
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["measurement_validated"] == VALIDATED_HELD_OUT


def test_export_aggregated_csv_regression_trait_ships_when_sidecar_validated(tmp_path):
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    bucket = _scalar_bucket(tmp_path, "preds", "regression", validated=True, trait="fruit_diameter")
    out = tmp_path / "o.csv"
    export_aggregated_csv(
        [{"plant_id": "p1", "value": 4.2, "observations": 3, "value_key": "fruit_diameter",
         "measurement_document": "regression_operating_point"}],
        str(out), trait_name="fruit_diameter", measurement_validated=VALIDATED_HELD_OUT,
        pred_dirs=[bucket])
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["measurement_validated"] == VALIDATED_HELD_OUT


def test_export_aggregated_csv_ordinal_trait_floors_on_missing_sidecar(tmp_path):
    # A bucket with no ordinal_operating_point.json floors to false and refuses, the same
    # reconcile-from-disk discipline the count operating point already has.
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    bucket = tmp_path / "preds"
    bucket.mkdir()
    with pytest.raises(ValueError, match="unvalidated measurement"):
        export_aggregated_csv(
            [{"plant_id": "p1", "value": 2, "observations": 3, "value_key": "astringency",
             "measurement_document": "ordinal_operating_point"}],
            str(tmp_path / "o.csv"), trait_name="astringency",
            measurement_validated=VALIDATED_HELD_OUT, pred_dirs=[str(bucket)])


def test_export_aggregated_csv_regression_trait_floors_on_a_failed_sidecar(tmp_path):
    # A sidecar that exists but is stamped unvalidated must also refuse, not just an entirely
    # missing sidecar.
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    bucket = _scalar_bucket(tmp_path, "preds", "regression", validated=False)
    with pytest.raises(ValueError, match="unvalidated measurement"):
        export_aggregated_csv(
            [{"plant_id": "p1", "value": 4.2, "observations": 3,
              "value_key": "fruit_diameter", "measurement_document": "regression_operating_point"}],
            str(tmp_path / "o.csv"), trait_name="fruit_diameter",
            measurement_validated=VALIDATED_HELD_OUT, pred_dirs=[str(bucket)])


def test_export_aggregated_csv_rejects_an_unrecognized_measurement_document(tmp_path):
    # A typo'd measurement_document must raise rather than reconciling against the wrong
    # dimension: the statement rail replacing the old task-typo guard.
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    bucket = _scalar_bucket(tmp_path, "preds", "ordinal", validated=True)
    with pytest.raises(ValueError, match="measurement_document"):
        export_aggregated_csv(
            [{"plant_id": "p1", "value": 2, "observations": 3, "value_key": "astringency",
             "measurement_document": "oridnal_operating_point"}],
            str(tmp_path / "o.csv"), trait_name="astringency",
            measurement_validated=VALIDATED_HELD_OUT, pred_dirs=[str(bucket)])


# ── tabulate_counts reads the run's resolved validity, not a caller string ─

def test_tabulate_counts_refuses_unvalidated_run(tmp_path, monkeypatch):
    import tcip_mcp.tools.inference_tools as itools

    def _fake_run_inference(**kw):
        return {"results": [{"image": "a.png", "count": 3}], "image_count": 1,
                "total_detections": 3, "operating_point": {"conf": {"value": 0.5}},
                "validated": False, "conf_source": "default"}

    monkeypatch.setattr(itools, "run_inference", _fake_run_inference)
    r = itools.tabulate_counts("m.pt", str(tmp_path), str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT)
    assert "error" in r
    assert r["operating_point_validated"] == VALIDATED_FALSE
    assert not (tmp_path / "o.csv").exists()


def test_tabulate_counts_acknowledge_writes_flagged(tmp_path, monkeypatch):
    import tcip_mcp.tools.inference_tools as itools

    captured = {}

    def _fake_run_inference(**kw):
        return {"results": [{"image": "a.png", "count": 3, "scores": [0.9]}], "image_count": 1,
                "total_detections": 3, "operating_point": {"conf": {"value": 0.5}},
                "validated": False, "conf_source": "default"}

    def _fake_export(results, path, provenance=None, *, trait, measurement_validated=None,
                     pred_dirs=None, acknowledge_unvalidated=False):
        captured["measurement_validated"] = measurement_validated
        captured["acknowledge_unvalidated"] = acknowledge_unvalidated
        return str(path)

    monkeypatch.setattr(itools, "run_inference", _fake_run_inference)
    monkeypatch.setattr(itools, "export_detection_csv", _fake_export)
    r = itools.tabulate_counts("m.pt", str(tmp_path), str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT, acknowledge_unvalidated=True)
    assert "error" not in r
    assert r["operating_point_validated"] == VALIDATED_FALSE
    assert captured["acknowledge_unvalidated"] is True


# ── tile_size gates the same way, closing the asymmetry with conf ─────

def _fake_run_inference_with(*, conf_ref, tile_size_prov=None):
    def _fake(**kw):
        op = {"conf": {"value": 0.6, "validated_against": conf_ref}}
        if tile_size_prov is not None:
            op["tile_size"] = tile_size_prov
        return {"results": [{"image": "a.png", "count": 3, "scores": [0.9]}], "image_count": 1,
                "total_detections": 3, "operating_point": op,
                "validated": conf_ref == VALIDATED_HELD_OUT, "conf_source": "calibration"}
    return _fake


def test_tabulate_counts_refuses_fabricated_tile_size_even_with_validated_conf(tmp_path, monkeypatch):
    """A fabricated tile_size must gate the same way an unvalidated conf does: a checkpoint with
    no persisted training geometry must not ship a real count here while run_full_frame_evaluation
    refuses to even measure that regime. A cleanly-validated conf must not paper over an
    ungrounded tile scale."""
    import tcip_mcp.tools.inference_tools as itools

    monkeypatch.setattr(itools, "run_inference", _fake_run_inference_with(
        conf_ref=VALIDATED_HELD_OUT,
        tile_size_prov={"value": 640, "requires_validation": True,
                        "validation_kind": "geometry", "validated_against": VALIDATED_FALSE}))
    r = itools.tabulate_counts("m.pt", str(tmp_path), str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT, calibration_labels_dir=str(tmp_path))
    assert "error" in r
    assert r["operating_point_validated"] == VALIDATED_HELD_OUT  # conf itself is fine...
    assert r["tile_size_validated"] == VALIDATED_FALSE           # ...tile_size is what refuses
    assert not (tmp_path / "o.csv").exists()


def _capture_csv_stamp(monkeypatch, itools, captured):
    """Stand in for the CSV writer, keeping the validity stamp and buckets the door handed it."""
    monkeypatch.setattr(
        itools, "export_detection_csv",
        lambda results, path, provenance=None, *, trait, measurement_validated=None,
        pred_dirs=None, acknowledge_unvalidated=False: (
            captured.update(mv=measurement_validated, pred_dirs=pred_dirs) or str(path)))


def test_tabulate_counts_ships_when_tile_size_has_a_real_basis(tmp_path, monkeypatch):
    """The rail must admit valid work, not only reject invalid work: a tile_size genuinely derived
    from the checkpoint's persisted training geometry ships cleanly, same as a validated conf."""
    import tcip_mcp.tools.inference_tools as itools

    from tcip_mcp.pipelines.resolution import VALIDATED_PERSISTED_GEOMETRY

    captured = {}
    monkeypatch.setattr(itools, "run_inference", lambda **kw: _earned_run_inference_result(
        tmp_path, trait=fx.COUNT_TRAIT, tiled=True, tile_size=224, tile_size_source="derived"))
    _capture_csv_stamp(monkeypatch, itools, captured)
    bucket = tmp_path / "ds" / "predictions" / "baseline" / "2026-01-01"
    r = itools.tabulate_counts("m.pt", str(tmp_path), str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT, calibration_labels_dir=str(tmp_path),
                               predictions_dir=str(bucket))
    assert "error" not in r, r
    assert r["tile_size_validated"] == VALIDATED_PERSISTED_GEOMETRY
    assert captured["mv"] == VALIDATED_HELD_OUT  # the CSV stamp reflects the fully-cleared gate
    assert captured["pred_dirs"] == [str(bucket)]  # and is reconciled against the bucket it wrote


def test_tabulate_counts_never_gates_tile_size_when_untiled(tmp_path, monkeypatch):
    """An untiled run's tile_size is never operative: it must not manufacture a refusal just
    because the run's own bundle happens to carry a non-gating tile_size entry."""
    import tcip_mcp.tools.inference_tools as itools

    captured = {}
    monkeypatch.setattr(itools, "run_inference",
                        lambda **kw: _earned_run_inference_result(
                            tmp_path, trait=fx.COUNT_TRAIT, tiled=False))
    _capture_csv_stamp(monkeypatch, itools, captured)
    r = itools.tabulate_counts(
        "m.pt", str(tmp_path), str(tmp_path / "o.csv"), trait=fx.COUNT_TRAIT,
        calibration_labels_dir=str(tmp_path),
        predictions_dir=str(tmp_path / "ds" / "predictions" / "baseline" / "2026-01-01"))
    assert "error" not in r, r
    assert r["tile_size_validated"] is None  # never entered the gate at all


def test_tabulate_counts_without_a_persisted_bucket_cannot_deliver_a_validated_csv(
    tmp_path, monkeypatch,
):
    """A count read off one in-memory pass rests on nothing a reviewer can re-read, so the CSV is
    refused unless it is acknowledged as provisional, and the acknowledged one is stamped false
    however clean the operating point behind it was."""
    import tcip_mcp.tools.inference_tools as itools

    captured = {}
    monkeypatch.setattr(itools, "run_inference",
                        lambda **kw: _earned_run_inference_result(
                            tmp_path, trait=fx.COUNT_TRAIT, tiled=False))
    _capture_csv_stamp(monkeypatch, itools, captured)
    refused = itools.tabulate_counts("m.pt", str(tmp_path), str(tmp_path / "o.csv"),
                                     trait=fx.COUNT_TRAIT,
                                     calibration_labels_dir=str(tmp_path))
    assert "predictions_dir" in refused["error"]
    assert refused["operating_point_validated"] == VALIDATED_HELD_OUT  # conf itself was fine

    provisional = itools.tabulate_counts("m.pt", str(tmp_path), str(tmp_path / "o.csv"),
                                         trait=fx.COUNT_TRAIT,
                                         calibration_labels_dir=str(tmp_path),
                                         acknowledge_unvalidated=True)
    assert "error" not in provisional, provisional
    assert provisional["predictions_dir"] is None
    assert captured["mv"] == VALIDATED_FALSE


def test_tabulate_counts_acknowledge_unvalidated_tile_size_floors_csv_stamp_despite_valid_conf(
    tmp_path, monkeypatch,
):
    """A CSV whose conf is genuinely validated but whose tile_size only shipped via
    acknowledge_unvalidated must not stamp measurement_validated as if the whole delivery were
    trustworthy: the single CSV column must reflect the floor across every gated dimension, not
    just conf's own (possibly-real) reference."""
    import tcip_mcp.tools.inference_tools as itools

    captured = {}
    monkeypatch.setattr(itools, "run_inference", _fake_run_inference_with(
        conf_ref=VALIDATED_HELD_OUT,
        tile_size_prov={"value": 640, "requires_validation": True,
                        "validation_kind": "geometry", "validated_against": VALIDATED_FALSE}))
    _capture_csv_stamp(monkeypatch, itools, captured)
    r = itools.tabulate_counts("m.pt", str(tmp_path), str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT, calibration_labels_dir=str(tmp_path),
                               acknowledge_unvalidated=True)
    assert "error" not in r
    assert captured["mv"] == VALIDATED_FALSE  # floored, not laundered into conf's clean reference


# ── export_predictions gates tile_size too: it is the door that actually persists a bucket ──

def _fake_run_inference_result(*, conf_ref, tile_size_prov=None):
    op = {"conf": {"value": 0.6, "validated_against": conf_ref}}
    if tile_size_prov is not None:
        op["tile_size"] = tile_size_prov
    return {
        "results": [{"image": "a.png", "width": 100, "height": 100,
                     "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9], "labels": [1], "count": 1}],
        "image_count": 1, "total_detections": 1, "operating_point": op, "id_map": None,
        "validated": conf_ref == VALIDATED_HELD_OUT, "conf_source": "calibration",
        "checkpoint_sha256": "deadbeef", "experiment_id": "exp1", "produced_at": "2026-01-01T00:00:00Z",
    }


def _earned_run_inference_result(tmp_path, *, trait="catkin", **calibration):
    """A stand-in run that left behind the evidence a door earns its validation record from.

    A door cannot stamp a validated bucket from a bare assertion that the run validated, so a test
    exercising the validated path resolves a real operating point and files its evidence.
    """
    from tests._binding_fixtures import calibrated_run_fields

    return {
        "results": [{"image": "a.png", "width": 100, "height": 100,
                     "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9], "labels": [1],
                     "count": 1}],
        "image_count": 1, "total_detections": 1, "id_map": None,
        "checkpoint_sha256": "deadbeef", "produced_at": "2026-01-01T00:00:00Z",
        **calibrated_run_fields(trait, labels_dir=tmp_path, **calibration),
    }


def test_export_predictions_refuses_fabricated_tile_size_even_with_validated_conf(tmp_path, monkeypatch):
    """The delivery door that actually persists a prediction bucket must refuse a fabricated tile
    scale the same way tabulate_counts/compute_phenology/export_aggregated_csv already do:
    run_inference itself never refuses (it is the shared, honestly-stamped raw substrate every
    door builds on, same contract as an uncalibrated conf), so the refusal belongs here."""
    import tcip_mcp.tools.inference_tools as itools

    monkeypatch.setattr(itools, "run_inference", lambda **kw: _fake_run_inference_result(
        conf_ref=VALIDATED_HELD_OUT,
        tile_size_prov={"value": 640, "requires_validation": True,
                        "validation_kind": "geometry", "validated_against": VALIDATED_FALSE}))
    out = tmp_path / "preds"
    r = itools.export_predictions("m.pt", str(tmp_path), str(out))
    assert "error" in r
    assert r["tile_size_validated"] == VALIDATED_FALSE
    assert not out.exists()


def test_export_predictions_ships_when_tile_size_has_a_real_basis(
    tmp_path, monkeypatch, seed_catkin_trait_spec,
):
    """The rail must admit valid work, not only reject invalid work."""
    import tcip_mcp.tools.inference_tools as itools

    from tcip_mcp.pipelines.resolution import VALIDATED_PERSISTED_GEOMETRY

    monkeypatch.setattr(itools, "run_inference", lambda **kw: _earned_run_inference_result(
        tmp_path, tiled=True, tile_size=224, tile_size_source="derived"))
    out = tmp_path / "ds" / "predictions" / "baseline" / "2026-01-01"
    r = itools.export_predictions("m.pt", str(tmp_path), str(out), trait="catkin")
    assert "error" not in r, r
    assert r["tile_size_validated"] == VALIDATED_PERSISTED_GEOMETRY
    assert r["validated"] is True
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

    sidecar = read_operating_point_sidecar(out)
    assert sidecar["tile_size_validated"] == VALIDATED_PERSISTED_GEOMETRY
    assert sidecar["validated"] is True


def test_export_predictions_never_gates_tile_size_when_untiled(
    tmp_path, monkeypatch, seed_catkin_trait_spec,
):
    """An untiled run's tile_size is never operative: it must not manufacture a refusal just
    because the run's own bundle happens to carry a non-gating tile_size entry."""
    import tcip_mcp.tools.inference_tools as itools

    monkeypatch.setattr(itools, "run_inference",
                        lambda **kw: _earned_run_inference_result(tmp_path, tiled=False))
    out = tmp_path / "ds" / "predictions" / "baseline" / "2026-01-01"
    r = itools.export_predictions("m.pt", str(tmp_path), str(out), trait="catkin")
    assert "error" not in r, r
    assert r["tile_size_validated"] is None
    assert r["validated"] is True


def test_export_predictions_acknowledge_writes_and_floors_the_sidecar_stamp(tmp_path, monkeypatch):
    """A bucket whose conf is genuinely validated but whose tile_size only shipped via
    acknowledge_unvalidated must not stamp validated=true on the sidecar, or a downstream door
    reading it would treat a fabricated tile scale as trustworthy."""
    import tcip_mcp.tools.inference_tools as itools

    monkeypatch.setattr(itools, "run_inference", lambda **kw: _fake_run_inference_result(
        conf_ref=VALIDATED_HELD_OUT,
        tile_size_prov={"value": 640, "requires_validation": True,
                        "validation_kind": "geometry", "validated_against": VALIDATED_FALSE}))
    out = tmp_path / "preds"
    r = itools.export_predictions("m.pt", str(tmp_path), str(out), acknowledge_unvalidated=True)
    assert "error" not in r
    assert r["tile_size_validated"] == VALIDATED_FALSE
    assert r["validated"] is False  # floored despite conf's own clean reference
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

    sidecar = read_operating_point_sidecar(out)
    assert sidecar["validated"] is False
    assert (out / "a.json").exists()  # the honestly-flagged provisional bucket still wrote


def test_export_predictions_images_dir_gates_before_the_pass_not_after(tmp_path, monkeypatch):
    """DECIDED #1: the images_dir regime's gate runs before the (expensive) pass, the same
    ordering the raster_path regime already had, not only after run_inference already ran it.
    A real checkpoint with no tile geometry at all (no persisted tile size, no untiled training
    frame to derive a native-ratio edge from, no explicit override) must refuse without ever
    reaching the model's own forward pass; GenericPredictor's predict_batch is monkeypatched to
    raise if called at all, so this proves the skip, not just that no bucket got written."""
    import numpy as np
    import torch
    from PIL import Image

    from tcip_mcp.pipelines.inference import generic_predictor as gp_mod
    from tcip_mcp.pipelines.model_build import build_model
    from tcip_mcp.tools import inference_tools as itools

    def _never_called(*a, **kw):
        raise AssertionError("predict_batch must not run: the pre-pass gate should have refused")

    monkeypatch.setattr(gp_mod.GenericPredictor, "predict_batch", _never_called)

    model_source = {"builder": "tests.bespoke_models:build_bespoke_detection",
                    "builder_kwargs": {"num_classes": 1, "min_size": 64, "max_size": 128},
                    "task": "detection"}
    model = build_model({"model_source": model_source})
    ckpt = tmp_path / "m.pt"
    torch.save({
        "model_source": model_source, "model_state_dict": model.state_dict(),
        "config": {"data": {}, "augmentation": {}},
    }, str(ckpt))

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    Image.fromarray(arr).save(images_dir / "a.png")

    out = tmp_path / "preds"
    r = itools.export_predictions(str(ckpt), str(images_dir), str(out), conf_threshold=0.0,
                                  tile=True)
    assert "error" in r
    assert not out.exists()


# ── the GUI inference worker gates the bucket it persists, same as export_predictions ──

def _run_gui_inference_worker(tmp_path, monkeypatch, *, tile, train_tile_size=None,
                              slice_source="default", tile_source="explicit"):
    """Run the web Inference tab's own worker over one image and return ``(job, output_dir)``.

    ``train_tile_size`` is the checkpoint's own persisted training geometry, absent when the
    checkpoint recorded none; ``slice_source="explicit"`` is a caller-stated tile edge.
    ``tile_source`` defaults to ``"explicit"`` (every prior caller here passes a concrete
    ``tile`` bool); pass ``"default"`` alongside ``tile=None`` to exercise the GUI launch route's
    own "no tile field" case, where the worker derives the bool from the checkpoint itself.
    """
    pytest.importorskip("fastapi")
    from PIL import Image

    from tcip_web.routes.inference import InferenceJob, _worker

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "img.jpg")
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")

    class FakePredictor:
        task = "detection"

        def __init__(self, checkpoint_path=None, **kwargs):
            pass

        def predict_batch(self, paths, **kw):
            return [{"image": p, "width": 100, "height": 100,
                     "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9], "labels": [1], "count": 1}
                    for p in paths]

    if train_tile_size is not None:
        FakePredictor.train_tile_size = train_tile_size
    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", FakePredictor)

    out_dir = tmp_path / "out"
    job = InferenceJob(
        job_id="gate", checkpoint_path=str(ckpt), images_dir=str(images_dir),
        output_dir=str(out_dir), tile=tile, tile_source=tile_source, conf=0.25, iou=0.7,
        slice_hw=(512, 512), overlap=0.2, slice_source=slice_source,
    )
    _worker(job)
    return job, out_dir


def _sidecar_tile_reference(out_dir):
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

    op = read_operating_point_sidecar(out_dir)["operating_point"]
    return op["tile_size"]["validated_against"]


def test_gui_inference_worker_refuses_a_fabricated_tile_scale(tmp_path, monkeypatch):
    """The breeder's own door must be gated like every other door that persists a bucket: a tiled
    run off a checkpoint with no persisted training geometry writes counts at a scale nothing
    justifies, which export_predictions already refuses. The refusal has to reach the breeder as a
    failed job carrying the reason, never a silent bucket plus an unvalidated sidecar."""
    job, out_dir = _run_gui_inference_worker(tmp_path, monkeypatch, tile=True)
    assert job.status == "failed"
    assert "tile_size" in job.error
    assert "training tile geometry" in job.error   # names what is missing, not just that it failed
    assert not (out_dir / "operating_point.json").exists()
    assert not (out_dir / "img.json").exists()     # nothing of the bucket was written
    assert job.done == 0


def test_gui_inference_worker_ships_when_the_tile_scale_has_a_real_basis(tmp_path, monkeypatch):
    """The rail must admit valid work, not only reject invalid work: a tiled run whose tile edge
    came from the checkpoint's own persisted training geometry writes its bucket and stamps the
    real reference the scale cleared."""
    from tcip_mcp.pipelines.resolution import VALIDATED_PERSISTED_GEOMETRY

    job, out_dir = _run_gui_inference_worker(tmp_path, monkeypatch, tile=True, train_tile_size=224)
    assert job.status == "completed"
    assert job.error is None
    assert (out_dir / "img.json").exists()
    assert _sidecar_tile_reference(out_dir) == VALIDATED_PERSISTED_GEOMETRY


def test_gui_inference_worker_ships_a_caller_stated_tile_geometry(tmp_path, monkeypatch):
    """The launch payload's own tile-size override is the other real basis for the scale, and it
    must clear the gate on the same terms the MCP door accepts an explicit tile_size on."""
    from tcip_mcp.pipelines.resolution import VALIDATED_EXPLICIT_GEOMETRY

    job, out_dir = _run_gui_inference_worker(tmp_path, monkeypatch, tile=True,
                                             slice_source="explicit")
    assert job.status == "completed"
    assert _sidecar_tile_reference(out_dir) == VALIDATED_EXPLICIT_GEOMETRY


def test_gui_inference_worker_never_gates_an_untiled_run_on_tile_size(tmp_path, monkeypatch):
    """An untiled run's tile_size was never operative, so a checkpoint with no persisted geometry
    must still run: gating it would refuse work that was always fine."""
    job, out_dir = _run_gui_inference_worker(tmp_path, monkeypatch, tile=False)
    assert job.status == "completed"
    assert (out_dir / "img.json").exists()
    assert _sidecar_tile_reference(out_dir) is None  # never entered the gate at all


def test_gui_launch_with_no_tile_field_derives_from_the_checkpoint_not_a_default(
        tmp_path, monkeypatch):
    """The GUI's launch payload omits ``tile`` on a real launch with the checkbox retired
    (routes/inference.py's ``LaunchInferencePayload.tile`` stays ``None``); the worker must derive
    the bool from the checkpoint's own persisted training geometry at that point, the same as the
    MCP door's ``run_inference``, never fall back to always-tiled. A checkpoint that trained tiled
    must still tile when the field is unset."""
    from tcip_mcp.pipelines.resolution import VALIDATED_PERSISTED_GEOMETRY

    job, out_dir = _run_gui_inference_worker(
        tmp_path, monkeypatch, tile=None, tile_source="default", train_tile_size=224)
    assert job.status == "completed"
    assert (out_dir / "img.json").exists()
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

    op = read_operating_point_sidecar(out_dir)["operating_point"]
    assert op["tiled"]["value"] is True
    assert _sidecar_tile_reference(out_dir) == VALIDATED_PERSISTED_GEOMETRY


def test_gui_launch_with_no_tile_field_and_no_checkpoint_geometry_stays_untiled(
        tmp_path, monkeypatch):
    """The mirror case, and the one a fixed ``DEFAULT_TILED=True`` used to get silently wrong: a
    checkpoint with no persisted training geometry, launched with the tile field unset, must run
    untiled rather than tiling at a scale nothing justifies."""
    job, out_dir = _run_gui_inference_worker(
        tmp_path, monkeypatch, tile=None, tile_source="default")
    assert job.status == "completed"
    assert (out_dir / "img.json").exists()
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

    op = read_operating_point_sidecar(out_dir)["operating_point"]
    assert op["tiled"]["value"] is False
    assert _sidecar_tile_reference(out_dir) is None  # never entered the gate at all


# ── the same tile-geometry dimension, read from a written bucket's sidecar ──

def _write_bucket(tmp_path, name, *, conf_ref, tile_size_prov=None, validated=None,
                  trait=fx.COUNT_TRAIT):
    """A prediction bucket's operating_point.json, the shape export_predictions writes."""
    root = tmp_path / "ds"
    d = root / "predictions" / name
    write_prediction(d, "img_a")
    op = {"conf": {"value": 0.4, "requires_validation": True, "validation_kind": "annotations",
                   "validated_against": conf_ref}}
    if tile_size_prov is not None:
        op["tile_size"] = tile_size_prov
    is_validated = (conf_ref == VALIDATED_HELD_OUT) if validated is None else validated
    stamp = {"validated": is_validated, "trait": trait, "operating_point": op}
    if is_validated:
        write_bound_sidecar(d, stamp, dataset_root=root, experiment_id=f"exp-{name}")
    else:
        from tcip_mcp.pipelines.resolution import write_sidecar

        write_sidecar(d, stamp)
    return str(d)


def _tile(ref, value=640):
    return {"value": value, "requires_validation": True, "validation_kind": "geometry",
            "validated_against": ref}


def test_untiled_buckets_leave_the_tile_dimension_out_of_the_gate(tmp_path):
    """A delivery assembled from untiled buckets must not acquire a tile-geometry dimension: the
    scale was never operative, so gating on it would refuse work that was always fine."""
    from tcip_mcp.pipelines.resolution import reconcile_tile_size_validity

    d = _write_bucket(tmp_path, "b1", conf_ref=VALIDATED_HELD_OUT,
                      tile_size_prov={"value": None, "requires_validation": False,
                                      "validation_kind": None, "validated_against": None})
    recon = reconcile_tile_size_validity([d])
    assert recon["operative"] is False
    assert recon["validated"] is None


def test_a_persisted_tile_geometry_is_not_floored_by_an_uncalibrated_conf(tmp_path):
    """The tile dimension reads the tile_size param's own recorded reference, never the sidecar's
    top-level bundle flag: a genuinely persisted training geometry stays persisted geometry even
    when the conf beside it is what failed, so a refusal names the dimension that actually broke."""
    from tcip_mcp.pipelines.resolution import (
        VALIDATED_PERSISTED_GEOMETRY,
        reconcile_tile_size_validity,
    )

    d = _write_bucket(tmp_path, "b1", conf_ref=VALIDATED_FALSE,
                      tile_size_prov=_tile(VALIDATED_PERSISTED_GEOMETRY, 224))
    recon = reconcile_tile_size_validity([d])
    assert recon["operative"] is True
    assert recon["validated"] == VALIDATED_PERSISTED_GEOMETRY
    assert recon["unvalidated_buckets"] == []


def test_one_ungrounded_tiled_bucket_floors_the_whole_delivery(tmp_path):
    """A delivery is only as grounded as its least-grounded tiled bucket: one fabricated tile edge
    among several persisted ones floors the dimension and names the bucket that caused it."""
    from tcip_mcp.pipelines.resolution import (
        VALIDATED_PERSISTED_GEOMETRY,
        reconcile_tile_size_validity,
    )

    good = _write_bucket(tmp_path, "b1", conf_ref=VALIDATED_HELD_OUT,
                         tile_size_prov=_tile(VALIDATED_PERSISTED_GEOMETRY, 224))
    bad = _write_bucket(tmp_path, "b2", conf_ref=VALIDATED_HELD_OUT,
                        tile_size_prov=_tile(VALIDATED_FALSE, 640))
    recon = reconcile_tile_size_validity([good, bad])
    assert recon["validated"] == VALIDATED_FALSE
    assert recon["unvalidated_buckets"] == [bad]


def test_a_stated_override_beside_a_persisted_geometry_reports_the_weaker_basis(tmp_path):
    """Both bases ship, but the delivery's recorded basis is the weaker one present: a caller's
    stated tile edge was never cross-checked against a checkpoint's real training scale, and one
    bucket that was must not lend its stronger basis to the whole delivery's stamp."""
    from tcip_mcp.pipelines.resolution import (
        VALIDATED_EXPLICIT_GEOMETRY,
        VALIDATED_PERSISTED_GEOMETRY,
        reconcile_tile_size_validity,
    )

    a = _write_bucket(tmp_path, "b1", conf_ref=VALIDATED_HELD_OUT,
                      tile_size_prov=_tile(VALIDATED_PERSISTED_GEOMETRY, 224))
    b = _write_bucket(tmp_path, "b2", conf_ref=VALIDATED_HELD_OUT,
                      tile_size_prov=_tile(VALIDATED_EXPLICIT_GEOMETRY, 512))
    recon = reconcile_tile_size_validity([a, b])
    assert recon["validated"] == VALIDATED_EXPLICIT_GEOMETRY


def test_export_aggregated_csv_refuses_a_fabricated_tile_size_with_a_validated_conf(tmp_path):
    """The per-plant CSV aggregates per-image counts, and a tile edge with no persisted training
    geometry and no explicit caller override moves those counts. A cleanly-calibrated conf must not
    paper over it, exactly as it does not at the count-CSV door."""
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    d = _write_bucket(tmp_path, "preds", conf_ref=VALIDATED_HELD_OUT,
                      tile_size_prov=_tile(VALIDATED_FALSE, 640))
    with pytest.raises(ValueError, match="unvalidated measurement"):
        export_aggregated_csv(
            [{"plant_id": "p1", "value": 5, "observations": 2, "value_key": "count",
             "measurement_document": "operating_point"}],
            str(tmp_path / "o.csv"), trait_name="stem_count", pred_dirs=[d])


def test_export_aggregated_csv_ships_when_the_tile_scale_has_a_real_basis(tmp_path):
    """The rail must admit valid work: a tiled bucket whose tile edge came from the checkpoint's
    own persisted training geometry delivers cleanly and stamps its real reference."""
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv
    from tcip_mcp.pipelines.resolution import VALIDATED_PERSISTED_GEOMETRY

    d = _write_bucket(tmp_path, "preds", conf_ref=VALIDATED_HELD_OUT,
                      tile_size_prov=_tile(VALIDATED_PERSISTED_GEOMETRY, 224))
    out = tmp_path / "o.csv"
    export_aggregated_csv(
        [{"plant_id": "p1", "value": 5, "observations": 2, "value_key": "count",
         "measurement_document": "operating_point"}], str(out),
        trait_name="stem_count", pred_dirs=[d])
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["measurement_validated"] == VALIDATED_HELD_OUT


def test_export_aggregated_csv_never_gates_an_untiled_bucket_on_tile_size(tmp_path):
    """A bucket from an untiled run carries a non-gating tile_size entry; the per-plant door must
    not acquire a tile-geometry dimension from it and refuse work that was always fine."""
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    d = _write_bucket(tmp_path, "preds", conf_ref=VALIDATED_HELD_OUT,
                      tile_size_prov={"value": None, "requires_validation": False,
                                      "validation_kind": None, "validated_against": None})
    out = tmp_path / "o.csv"
    export_aggregated_csv(
        [{"plant_id": "p1", "value": 5, "observations": 2, "value_key": "count",
         "measurement_document": "operating_point"}], str(out),
        trait_name="stem_count", pred_dirs=[d])
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["measurement_validated"] == VALIDATED_HELD_OUT


def test_export_aggregated_csv_acknowledged_tile_size_floors_the_row_stamp(tmp_path):
    """A per-plant CSV whose conf is genuinely validated but whose tile scale only shipped through
    acknowledge_unvalidated must stamp its one measurement column false, not conf's clean
    reference."""
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    d = _write_bucket(tmp_path, "preds", conf_ref=VALIDATED_HELD_OUT,
                      tile_size_prov=_tile(VALIDATED_FALSE, 640))
    out = tmp_path / "o.csv"
    export_aggregated_csv(
        [{"plant_id": "p1", "value": 5, "observations": 2, "value_key": "count",
         "measurement_document": "operating_point"}], str(out),
        trait_name="stem_count", pred_dirs=[d], acknowledge_unvalidated=True)
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["measurement_validated"] == VALIDATED_FALSE


# ── export_aggregated_csv gates a dimensional value_key on its physical scale too ──

def _write_scale_sidecar(path, *, validated_against, capture_id=None, value=0.05, unit="mm",
                         trait="plant_surface_area"):
    """A bucket's resolve_scale.json, the shape reconcile_scale_validity reads, alongside its
    operating_point.json in the same directory."""
    from tcip_mcp.pipelines.resolution import VALIDATED_PHYSICAL_MEASUREMENT

    path.mkdir(parents=True, exist_ok=True)
    is_validated = validated_against == VALIDATED_PHYSICAL_MEASUREMENT
    stamp = {
        "validated": is_validated, "trait": trait,
        "operating_point": {
            "scale": {
                "value": value, "unit": unit, "capture_id": capture_id,
                "requires_validation": True, "validation_kind": "physical",
                "validated_against": validated_against,
            },
        },
    }
    if is_validated:
        write_bound_sidecar(path, stamp, document="resolve_scale", dataset_root=path.parent.parent,
                            experiment_id=f"exp-scale-{path.name}")
    else:
        (path / "resolve_scale.json").write_text(json.dumps(stamp), encoding="utf-8")
    return str(path)


_DIM_RESULTS = [{"plant_id": "p1", "value": 12.5, "observations": 1, "value_key": "area_mm2",
                "measurement_document": "operating_point", "scale_document": "resolve_scale"}]


def test_export_aggregated_csv_ships_dimensional_value_with_a_validated_scale(tmp_path):
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv
    from tcip_mcp.pipelines.resolution import VALIDATED_PHYSICAL_MEASUREMENT

    d = _write_bucket(tmp_path, "preds", conf_ref=VALIDATED_HELD_OUT, trait="plant_surface_area")
    _write_scale_sidecar(Path(d), validated_against=VALIDATED_PHYSICAL_MEASUREMENT)
    out = tmp_path / "o.csv"
    export_aggregated_csv(_DIM_RESULTS, str(out), trait_name="plant_surface_area",
                          pred_dirs=[d])
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["measurement_validated"] == VALIDATED_HELD_OUT
    assert rows[0]["units"] == "mm2"
    assert rows[0]["scale_document"] == "resolve_scale"


def test_export_aggregated_csv_refuses_a_dimensional_delivery_with_no_scale_sidecar(tmp_path):
    """A dimensional CSV must not ship stamped validated when its physical scale was never checked
    against anything, even though the count operating point beside it is genuinely validated: this
    is exactly the gap the scale gate closes."""
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    d = _write_bucket(tmp_path, "preds", conf_ref=VALIDATED_HELD_OUT, trait="plant_surface_area")
    with pytest.raises(ValueError, match="unvalidated measurement"):
        export_aggregated_csv(_DIM_RESULTS, str(tmp_path / "o.csv"),
                              trait_name="plant_surface_area", pred_dirs=[d])


def test_export_aggregated_csv_count_trait_never_gates_on_scale(tmp_path):
    """A count trait's value_key implies no physical unit, so the scale dimension never becomes
    operative even though pred_dirs is given: nothing dimensional to protect, nothing to refuse
    over."""
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    d = _write_bucket(tmp_path, "preds", conf_ref=VALIDATED_HELD_OUT)
    out = tmp_path / "o.csv"
    export_aggregated_csv(
        [{"plant_id": "p1", "value": 5, "observations": 2, "value_key": "count",
         "measurement_document": "operating_point"}],
        str(out), trait_name="stem_count", pred_dirs=[d])
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["measurement_validated"] == VALIDATED_HELD_OUT


def test_export_aggregated_csv_scale_capture_id_mismatch_floors(tmp_path):
    """A handheld standoff's scale can vary capture to capture: a sidecar validated for a different
    capture than the one this delivery names must not silently clear this delivery's scale."""
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv
    from tcip_mcp.pipelines.resolution import VALIDATED_PHYSICAL_MEASUREMENT

    d = _write_bucket(tmp_path, "preds", conf_ref=VALIDATED_HELD_OUT, trait="plant_surface_area")
    _write_scale_sidecar(Path(d), validated_against=VALIDATED_PHYSICAL_MEASUREMENT,
                         capture_id="2026-02-10_plot7")
    with pytest.raises(ValueError, match="unvalidated measurement"):
        export_aggregated_csv(_DIM_RESULTS, str(tmp_path / "o.csv"),
                              trait_name="plant_surface_area", pred_dirs=[d],
                              scale_capture_id="2026-02-10_plot9")


def test_export_aggregated_csv_scale_capture_id_match_ships(tmp_path):
    """The rail must admit valid work: the scale's own recorded capture matches the one this
    delivery is scoped to, so it ships cleanly."""
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv
    from tcip_mcp.pipelines.resolution import VALIDATED_PHYSICAL_MEASUREMENT

    d = _write_bucket(tmp_path, "preds", conf_ref=VALIDATED_HELD_OUT, trait="plant_surface_area")
    _write_scale_sidecar(Path(d), validated_against=VALIDATED_PHYSICAL_MEASUREMENT,
                         capture_id="2026-02-10_plot7")
    out = tmp_path / "o.csv"
    export_aggregated_csv(_DIM_RESULTS, str(out), trait_name="plant_surface_area",
                          pred_dirs=[d], scale_capture_id="2026-02-10_plot7")
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["measurement_validated"] == VALIDATED_HELD_OUT


def test_export_aggregated_csv_acknowledged_unvalidated_scale_floors_the_row_stamp(tmp_path):
    """A dimensional CSV whose conf is genuinely validated but whose scale never cleared must stamp
    its one measurement column false when shipped through acknowledge_unvalidated, not conf's clean
    reference."""
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    d = _write_bucket(tmp_path, "preds", conf_ref=VALIDATED_HELD_OUT, trait="plant_surface_area")
    out = tmp_path / "o.csv"
    export_aggregated_csv(_DIM_RESULTS, str(out), trait_name="plant_surface_area",
                          pred_dirs=[d], acknowledge_unvalidated=True)
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["measurement_validated"] == VALIDATED_FALSE


def test_export_aggregated_csv_refuses_a_stated_scale_with_no_physical_unit(tmp_path):
    """A stated scale_document with a value_key implying no physical unit is refused: a physical
    scale cannot answer for a non-dimensional value (count-delivery-door design section 2, rule 4)."""
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    d = _write_bucket(tmp_path, "preds", conf_ref=VALIDATED_HELD_OUT)
    with pytest.raises(ValueError, match="scale_document"):
        export_aggregated_csv(
            [{"plant_id": "p1", "value": 5, "observations": 2, "value_key": "count",
             "measurement_document": "operating_point", "scale_document": "resolve_scale"}],
            str(tmp_path / "o.csv"), trait_name="stem_count", pred_dirs=[d])


def test_export_aggregated_csv_refuses_a_dimensional_operating_point_delivery_with_no_stated_scale(
    tmp_path,
):
    """A value_key implying a physical unit under operating_point with no stated scale_document
    refuses outright: a dimensional number from a detection/segmentation bucket has nothing
    answering for its unit without one (rule 4), distinct from the gate floor above, which is for a
    delivery that at least states the scale it rests on."""
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    d = _write_bucket(tmp_path, "preds", conf_ref=VALIDATED_HELD_OUT, trait="plant_surface_area")
    with pytest.raises(ValueError, match="scale_document"):
        export_aggregated_csv(
            [{"plant_id": "p1", "value": 12.5, "observations": 1, "value_key": "area_mm2",
             "measurement_document": "operating_point"}],
            str(tmp_path / "o.csv"), trait_name="plant_surface_area", pred_dirs=[d])


def test_export_aggregated_csv_regression_head_delivers_a_dimensional_value_with_no_scale(tmp_path):
    """The rail must admit valid work: a regression head's prediction is in the trait's declared
    unit by construction, so a dimensional delivery under regression_operating_point with no stated
    scale_document ships cleanly, unlike the same shape under operating_point above."""
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    fx.confirm_aggregate(tmp_path, "fruit_diameter", op.PER_PLANT_REGRESSION_AGGREGATE,
                         delivered_phenotype="fruit_diameter", value_keys=["fruit_diameter_mm"])
    bucket = _scalar_bucket(tmp_path, "preds", "regression", validated=True, trait="fruit_diameter")
    out = tmp_path / "o.csv"
    export_aggregated_csv(
        [{"plant_id": "p1", "value": 4.2, "observations": 3, "value_key": "fruit_diameter_mm",
         "measurement_document": "regression_operating_point"}],
        str(out), trait_name="fruit_diameter", measurement_validated=VALIDATED_HELD_OUT,
        pred_dirs=[bucket])
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["measurement_validated"] == VALIDATED_HELD_OUT
    assert rows[0]["units"] == "mm"
    assert rows[0]["scale_document"] == ""


def test_export_aggregated_csv_refuses_a_declared_unit_trait_with_a_pixel_space_key(tmp_path):
    """A trait declaring a physical unit (fruit_diameter, mm in crops.yml) whose delivered
    value_key implies none refuses under operating_point: a value with no unit suffix delivered
    under a unit-declared trait is not that trait's number (P4-45). The value_key itself
    ('fruit_diameter', no unit suffix) is confirmed for stem_count-style count aggregation in the
    fixture above, so only the document differs from the passing regression scenario elsewhere in
    this file."""
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    d = _write_bucket(tmp_path, "preds", conf_ref=VALIDATED_HELD_OUT, trait="fruit_diameter")
    with pytest.raises(ValueError, match="declared units"):
        export_aggregated_csv(
            [{"plant_id": "p1", "value": 4.2, "observations": 1,
             "value_key": "fruit_diameter", "measurement_document": "operating_point"}],
            str(tmp_path / "o.csv"), trait_name="fruit_diameter", pred_dirs=[d])


def test_export_aggregated_csv_refuses_classifier_operating_point_as_a_measurement_document(
    tmp_path,
):
    """No per-plant aggregate this door delivers rests on a classifier alone; a record naming
    classifier_operating_point as its measurement_document refuses."""
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    with pytest.raises(ValueError, match="classifier_operating_point"):
        export_aggregated_csv(
            [{"plant_id": "p1", "value": 5, "observations": 2, "value_key": "count",
             "measurement_document": "classifier_operating_point"}],
            str(tmp_path / "o.csv"), trait_name="stem_count", acknowledge_unvalidated=True)


def test_export_aggregated_csv_refuses_resolve_scale_as_a_measurement_document(tmp_path):
    """A physical scale is never itself the measurement it states the unit of; naming resolve_scale
    as measurement_document refuses."""
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    with pytest.raises(ValueError, match="resolve_scale"):
        export_aggregated_csv(
            [{"plant_id": "p1", "value": 5, "observations": 2, "value_key": "count",
             "measurement_document": "resolve_scale"}],
            str(tmp_path / "o.csv"), trait_name="stem_count", acknowledge_unvalidated=True)


def test_aggregate_per_plant_refuses_a_plant_whose_images_disagree_on_the_statement(tmp_path):
    """A statement that disagrees with itself is not a statement: aggregate_per_plant refuses
    rather than collapsing to 'mixed' the way plant_id_source does."""
    from tcip_mcp.pipelines.postprocessing.aggregation import aggregate_per_plant

    records = [
        {"image": "a1", "plant_id": "PLANT_A", "count": 2,
         "measurement_document": "operating_point"},
        {"image": "a2", "plant_id": "PLANT_A", "count": 4,
         "measurement_document": "regression_operating_point"},
    ]
    with pytest.raises(ValueError, match="measurement_document"):
        aggregate_per_plant(records, strategy="count", value_key="count")


def test_a_plant_with_no_value_at_all_refuses_naming_the_plant(tmp_path):
    """A per-plant row whose value is None (no observation carried the value_key) refuses at the
    door, naming the plant, rather than writing an empty cell beside a validated stamp (P4-46)."""
    from tcip_mcp.pipelines.postprocessing.aggregation import (
        aggregate_per_plant,
        export_aggregated_csv,
    )

    records = [
        {"image": "a1", "plant_id": "PLANT_A", "count": 5,
         "measurement_document": "operating_point"},
        {"image": "b1", "plant_id": "PLANT_B", "measurement_document": "operating_point"},
    ]
    summaries = aggregate_per_plant(records, strategy="count", value_key="count")
    with pytest.raises(ValueError, match="PLANT_B"):
        export_aggregated_csv(summaries, str(tmp_path / "o.csv"), trait_name="stem_count",
                              acknowledge_unvalidated=True)


def test_a_plant_with_a_real_zero_ships_beside_one_with_a_value(tmp_path):
    """The rail must admit valid work: a plant with a genuine 0 observation is a real measured
    absence, never confused with the missing-observation case above."""
    from tcip_mcp.pipelines.postprocessing.aggregation import (
        aggregate_per_plant,
        export_aggregated_csv,
    )

    records = [
        {"image": "a1", "plant_id": "PLANT_A", "count": 5,
         "measurement_document": "operating_point"},
        {"image": "b1", "plant_id": "PLANT_B", "count": 0,
         "measurement_document": "operating_point"},
    ]
    summaries = aggregate_per_plant(records, strategy="count", value_key="count")
    out = tmp_path / "o.csv"
    export_aggregated_csv(summaries, str(out), trait_name="stem_count",
                          acknowledge_unvalidated=True)
    rows = {r["plant_id"]: r for r in csv.DictReader(out.open())}
    assert rows["PLANT_B"]["value"] == "0"


# ── the provenance columns a delivery may carry ────────────────────────────

_COUNT_RESULTS = [{"plant_id": "p1", "value": 5, "observations": 2, "value_key": "count",
                  "measurement_document": "operating_point"}]


def _delivered_row(out_path):
    return next(csv.DictReader(Path(out_path).open(newline="")))


def test_a_bespoke_producer_keeps_its_checkpoint_hash_in_a_provisional_delivery(tmp_path):
    """A bucket produced by a real checkpoint that belongs to no experiment, delivered honestly
    unvalidated, still names the checkpoint behind its numbers: validity and producer identity rest
    on different evidence, and a hash resolved from the checkpoint file answers for itself."""
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    d = _write_bucket(tmp_path, "bespoke", conf_ref=VALIDATED_FALSE)
    out = tmp_path / "o.csv"
    export_aggregated_csv(
        _COUNT_RESULTS, str(out), trait_name="stem_count", pred_dirs=[d],
        provenance={"producer_model_sha256": "a" * 64, "experiment_id": None,
                    "produced_at": "2026-03-04T12:00:00+00:00"},
        acknowledge_unvalidated=True)

    row = _delivered_row(out)
    assert row["producer_model_sha256"] == "a" * 64
    assert row["experiment_id"] == ""
    assert row["validation_record"] == ""
    assert row["measurement_validated"] == VALIDATED_FALSE


def test_a_bucket_naming_an_experiment_that_never_ran_delivers_no_producer_identity(tmp_path):
    """The provisional half of the recorded route: a stamp may assert any checkpoint and any run,
    so a delivery repeats neither unless something outside the bucket answers for the run. The CSV
    says the producer is unknown rather than naming an experiment the store never held."""
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    d = _write_bucket(tmp_path, "forged", conf_ref=VALIDATED_FALSE)
    out = tmp_path / "o.csv"
    export_aggregated_csv(
        _COUNT_RESULTS, str(out), trait_name="stem_count", pred_dirs=[d],
        provenance={"producer_model_sha256": "0" * 64, "experiment_id": "exp_that_never_ran",
                    "produced_at": "2026-03-04T12:00:00+00:00"},
        acknowledge_unvalidated=True)

    row = _delivered_row(out)
    assert row["producer_model_sha256"] == ""
    assert row["experiment_id"] == ""
    assert row["validation_record"] == ""


def test_a_validated_delivery_names_the_record_its_numbers_rest_on(tmp_path):
    """The column a reviewer opens: the delivered validation_record is the experiment and row the
    stamp's own pointer names, which is the pair verification confirmed rather than a second
    reading of the stamp."""
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    d = _write_bucket(tmp_path, "preds", conf_ref=VALIDATED_HELD_OUT)
    out = tmp_path / "o.csv"
    export_aggregated_csv(_COUNT_RESULTS, str(out), trait_name="stem_count", pred_dirs=[d],
                          provenance={"produced_at": "2026-03-04T12:00:00+00:00"})

    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

    pointer = read_operating_point_sidecar(d)["validated_by"]
    row = _delivered_row(out)
    assert row["measurement_validated"] == VALIDATED_HELD_OUT
    assert row["validation_record"] == f"{pointer['experiment_id']}:{pointer['record_digest']}"
    assert row["experiment_id"] == "exp-preds"


def test_the_detection_csv_carries_the_same_provenance_the_aggregate_does(tmp_path):
    """Two deliverables, two column lists, one builder: the per-image CSV's producer cells are
    decided the same way the per-plant CSV's are, so they cannot disagree about one bucket."""
    from tcip_mcp.pipelines.postprocessing.export import export_detection_csv

    d = _write_bucket(tmp_path, "preds", conf_ref=VALIDATED_HELD_OUT)
    out = tmp_path / "o.csv"
    export_detection_csv([{"image": "img_a.jpg", "count": 5}], str(out), pred_dirs=[d],
                         trait=fx.COUNT_TRAIT,
                         provenance={"producer_model_sha256": "b" * 64,
                                     "experiment_id": "exp_that_never_ran",
                                     "operating_point_conf": 0.4})

    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

    pointer = read_operating_point_sidecar(d)["validated_by"]
    row = _delivered_row(out)
    assert row["validation_record"] == f"{pointer['experiment_id']}:{pointer['record_digest']}"
    # The record's own producing run wins over the asserted one, so the forged name never ships.
    assert row["experiment_id"] == "exp-preds"
    assert row["producer_model_sha256"] == ""
    assert row["operating_point_conf"] == "0.4"
    assert len(_audit_rows(tmp_path / "ds", "export_detection_csv")) == 1


def _audit_rows(root, tool):
    import tcip_store

    from tcip_mcp.audit import audit_log_key

    page = tcip_store.read_log(audit_log_key(root))
    return [r for r in page.records if r["tool"] == tool]


def test_the_delivery_records_what_it_verified_in_the_dataset_own_log(tmp_path):
    """What stood behind a delivered number is a fact about the dataset, so it travels with it. The
    audited decorator records arguments and status only, which is why the door emits this itself."""
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    d = _write_bucket(tmp_path, "preds", conf_ref=VALIDATED_HELD_OUT)
    export_aggregated_csv(_COUNT_RESULTS, str(tmp_path / "o.csv"), trait_name="stem_count",
                          pred_dirs=[d])

    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

    pointer = read_operating_point_sidecar(d)["validated_by"]
    rows = _audit_rows(tmp_path / "ds", "export_aggregated_csv")
    assert len(rows) == 1, rows
    assert rows[0]["record_digests"] == [pointer["record_digest"]]
    assert rows[0]["verified_buckets"][d]["verified"] is True
    assert rows[0]["verified_buckets"][d]["record"] == (
        f"{pointer['experiment_id']}:{pointer['record_digest']}")
    assert _audit_rows(tmp_path, "export_aggregated_csv") == []


def test_an_unbound_bucket_records_why_it_was_not_verified(tmp_path):
    """The same event on the failing side: a reader of the log sees which bucket floored the
    delivery and the reason, not only that a CSV was written."""
    import tcip_store

    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv
    from tcip_mcp.pipelines.resolution import sidecar_key

    d = _write_bucket(tmp_path, "unbound", conf_ref=VALIDATED_HELD_OUT, validated=False)
    # Forges a stamp claiming validated=True with no real validated_by, exactly the shape
    # write_sidecar's own claim check refuses, so this writes under it, through the seam.
    key = sidecar_key(d)
    with tcip_store.transaction(key) as txn:
        stamp = txn.read(key)
        txn.write(key, {**stamp, "validated": True})

    export_aggregated_csv(_COUNT_RESULTS, str(tmp_path / "o.csv"), trait_name="stem_count",
                          pred_dirs=[d], acknowledge_unvalidated=True)

    rows = _audit_rows(tmp_path / "ds", "export_aggregated_csv")
    assert len(rows) == 1, rows
    assert rows[0]["verified_buckets"][d]["verified"] is False
    assert "validated_by" in rows[0]["verified_buckets"][d]["note"]
    assert rows[0]["record_digests"] == []


def test_the_count_tool_records_what_it_verified_in_the_bucket_own_dataset_log(tmp_path, monkeypatch):
    """The tool delivers through the same writer, so its persisted path emits the same event.

    What stood behind a delivered number is a fact about the dataset the buckets sit in, and the
    tool that persisted them must not be the one path where that record goes missing.
    """
    import tcip_mcp.tools.inference_tools as itools

    monkeypatch.setattr(itools, "run_inference", lambda **kw: _earned_run_inference_result(
        tmp_path, trait=fx.COUNT_TRAIT, tiled=False))
    bucket = tmp_path / "ds" / "predictions" / "baseline" / "2026-01-01"

    r = itools.tabulate_counts("m.pt", str(tmp_path), str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT, calibration_labels_dir=str(tmp_path),
                               predictions_dir=str(bucket))

    assert "error" not in r, r
    rows = _audit_rows(tmp_path / "ds", "export_detection_csv")
    assert len(rows) == 1, rows
    assert rows[0]["arguments"]["pred_dirs"] == [str(bucket)]
    assert _audit_rows(tmp_path, "export_detection_csv") == []


# ── calibrate_physical_scale: the producer for resolve_scale.json ─────────

_SCALE_PX_PER_MM = 10.0  # a fixed 0.1 mm/px reference scale, chosen for round test numbers


def _rect_points(length_px, width_px, angle_deg=0.0, center=(500.0, 500.0)):
    """Four corners of a length x width rectangle, rotated ``angle_deg`` about its own centre."""
    import math

    hl, hw = length_px / 2.0, width_px / 2.0
    local = [(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)]
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    cx, cy = center
    return [(cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a) for x, y in local]


def _write_reference_annotation(labels_dir, stem, *, subject, points_by_annotation,
                                width=1000, height=1000):
    """One reference image's per-image JSON, carrying one Polygon annotation per entry in
    ``points_by_annotation`` (each entry a ring's own point list), all under ``subject``."""
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, Polygon

    labels_dir.mkdir(parents=True, exist_ok=True)
    anns = [Annotation(subject=subject, geometry=Polygon(rings=[points]))
           for points in points_by_annotation]
    json_io.write_annotations(str(labels_dir / f"{stem}.json"), anns, width, height)


def _write_reference_bbox(labels_dir, stem, *, subject, points, width=1000, height=1000):
    """A reference image annotated with a BBox instead of a Polygon (the refused geometry kind)."""
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    labels_dir.mkdir(parents=True, exist_ok=True)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ann = Annotation(subject=subject, geometry=BBox(min(xs), min(ys), max(xs), max(ys)))
    json_io.write_annotations(str(labels_dir / f"{stem}.json"), [ann], width, height)


def _write_reference_csv(path, rows):
    """``rows`` is ``[(stem, physical_extent, unit), ...]``."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_stem", "physical_extent", "unit"])
        for row in rows:
            writer.writerow(row)


def _author_scale_tolerance(tmp_path, trait, tolerance_frac=0.1):
    from tcip_mcp.traits import write_trait_spec_fields

    write_trait_spec_fields(trait, {"scale_tolerance_frac": tolerance_frac}, project_root=tmp_path)


def _calibration_setup(tmp_path, *, lengths_px, unit="mm", angle_deg_by_index=None):
    """A bucket carrying one real prediction plus one reference-object image per entry in
    ``lengths_px``, and the reference labels/CSV a calibrate_physical_scale call reads, each
    reference's physical_extent derived from the same fixed scale. Returns (pred_dir, labels_dir,
    reference_csv, stems, group_key_map)."""
    root = tmp_path / "ds"
    pred_dir = root / "predictions" / "preds"
    write_prediction(pred_dir, "img_a")
    labels_dir = tmp_path / "reference_labels"
    csv_path = tmp_path / "reference.csv"
    stems = [f"r{i}" for i in range(1, len(lengths_px) + 1)]
    rows = []
    for i, (stem, length_px) in enumerate(zip(stems, lengths_px)):
        angle = (angle_deg_by_index or {}).get(i, 0.0)
        points = _rect_points(length_px, 10.0, angle)
        _write_reference_annotation(labels_dir, stem, subject="cal_bar",
                                    points_by_annotation=[points])
        write_prediction(pred_dir, stem)
        physical = round(length_px / _SCALE_PX_PER_MM, 6)
        rows.append((stem, physical, unit))
    _write_reference_csv(csv_path, rows)
    group_key_map = {s: s for s in stems}
    return str(pred_dir), str(labels_dir), str(csv_path), stems, group_key_map


def test_calibrate_physical_scale_whole_chain_delivers_a_validated_mm2_area(tmp_path):
    """The whole chain: the tool's own gate, seal, write, the reconciler that reads it back, and a
    mask-geometry area delivered under it, proving the producer and the door agree. The scale is
    stamped into the same bucket a real detection run produced, the shape a delivery actually
    reads: one bucket carrying both operating_point.json and resolve_scale.json."""
    from tcip_mcp.tools.scale_tools import calibrate_physical_scale

    _author_scale_tolerance(tmp_path, "plant_surface_area")
    # References are written into the bucket before it seals, so the count operating point's
    # digest covers the final file set (adding files after sealing would invalidate that claim).
    bucket_dir = tmp_path / "ds" / "predictions" / "preds"
    labels_dir = tmp_path / "reference_labels"
    ref_csv = tmp_path / "reference.csv"
    stems = ["r1", "r2", "r3", "r4"]
    rows = []
    for stem in stems:
        points = _rect_points(100.0, 10.0)
        _write_reference_annotation(labels_dir, stem, subject="cal_bar",
                                    points_by_annotation=[points])
        write_prediction(bucket_dir, stem)
        rows.append((stem, round(100.0 / _SCALE_PX_PER_MM, 6), "mm"))
    _write_reference_csv(ref_csv, rows)
    group_key_map = {s: s for s in stems}

    bucket = _write_bucket(tmp_path, "preds", conf_ref=VALIDATED_HELD_OUT,
                           trait="plant_surface_area")

    result = calibrate_physical_scale(
        trait="plant_surface_area", pred_dir=bucket, dataset_root=str(tmp_path / "ds"),
        unit="mm", reference_subject="cal_bar", labels_dir=str(labels_dir),
        reference_csv=str(ref_csv), group_key_map=group_key_map)

    assert result["passed"] is True, result
    assert result["value"] == pytest.approx(0.1)  # 10 mm implied over a 100 px reference length
    assert result["validated_by"] is not None

    from tcip_mcp.pipelines.resolution import VALIDATED_PHYSICAL_MEASUREMENT, reconcile_scale_validity

    recon = reconcile_scale_validity([bucket], unit="mm", trait="plant_surface_area")
    assert recon["validated"] == VALIDATED_PHYSICAL_MEASUREMENT

    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    out = tmp_path / "o.csv"
    export_aggregated_csv(
        [{"plant_id": "p1", "value": 12.5, "observations": 1, "value_key": "area_mm2",
         "measurement_document": "operating_point", "scale_document": "resolve_scale"}],
        str(out), trait_name="plant_surface_area", pred_dirs=[bucket])
    out_rows = list(csv.DictReader(out.open()))
    assert out_rows[0]["measurement_validated"] == VALIDATED_HELD_OUT
    assert out_rows[0]["units"] == "mm2"


def test_calibrate_physical_scale_survives_a_prediction_re_export(tmp_path):
    """A scale claim binds to the bucket's own image stems, not its prediction bytes: re-exporting
    predictions over the same images must leave a validated scale standing."""
    from tcip_mcp.tools.scale_tools import calibrate_physical_scale

    _author_scale_tolerance(tmp_path, "plant_surface_area")
    pred_dir, labels_dir, ref_csv, _stems, group_key_map = _calibration_setup(
        tmp_path, lengths_px=[100.0, 100.0, 100.0, 100.0])
    result = calibrate_physical_scale(
        trait="plant_surface_area", pred_dir=pred_dir, dataset_root=str(tmp_path / "ds"),
        unit="mm", reference_subject="cal_bar", labels_dir=labels_dir, reference_csv=ref_csv,
        group_key_map=group_key_map)
    assert result["passed"] is True, result

    write_prediction(Path(pred_dir), "img_a", count=9)  # re-export: same images, new bytes

    from tcip_mcp.pipelines.resolution import VALIDATED_PHYSICAL_MEASUREMENT, reconcile_scale_validity

    recon = reconcile_scale_validity([pred_dir], unit="mm", trait="plant_surface_area")
    assert recon["validated"] == VALIDATED_PHYSICAL_MEASUREMENT


def test_calibrate_physical_scale_copied_sidecar_refuses(tmp_path):
    """A resolve_scale.json copied into another bucket fails on membership: the covered set names
    only the bucket the claim was actually earned against."""
    from tcip_mcp.tools.scale_tools import calibrate_physical_scale

    _author_scale_tolerance(tmp_path, "plant_surface_area")
    pred_dir, labels_dir, ref_csv, _stems, group_key_map = _calibration_setup(
        tmp_path, lengths_px=[100.0, 100.0, 100.0, 100.0])
    result = calibrate_physical_scale(
        trait="plant_surface_area", pred_dir=pred_dir, dataset_root=str(tmp_path / "ds"),
        unit="mm", reference_subject="cal_bar", labels_dir=labels_dir, reference_csv=ref_csv,
        group_key_map=group_key_map)
    assert result["passed"] is True, result

    from tcip_mcp.pipelines.resolution import (
        VALIDATED_FALSE,
        read_scale_sidecar,
        reconcile_scale_validity,
        write_sidecar,
    )

    other = Path(tmp_path) / "ds" / "predictions" / "other"
    write_prediction(other, "img_z")
    write_sidecar(other, read_scale_sidecar(pred_dir), "resolve_scale")

    recon = reconcile_scale_validity([str(other)], unit="mm", trait="plant_surface_area")
    assert recon["validated"] == VALIDATED_FALSE
    assert recon["unvalidated_buckets"] == [str(other)]


def test_calibrate_physical_scale_refuses_a_box_reference_geometry(tmp_path):
    """A bounding box's long side is the object's projected extent, orientation-dependent in both
    directions; the tool refuses it outright rather than deriving a scale from it."""
    from tcip_mcp.tools.scale_tools import calibrate_physical_scale

    _author_scale_tolerance(tmp_path, "plant_surface_area")
    pred_dir, labels_dir, ref_csv, stems, group_key_map = _calibration_setup(
        tmp_path, lengths_px=[100.0, 100.0])
    # Overwrite one reference's annotation with a BBox instead of a Polygon.
    _write_reference_bbox(Path(labels_dir), stems[0], subject="cal_bar",
                          points=_rect_points(100.0, 10.0))

    result = calibrate_physical_scale(
        trait="plant_surface_area", pred_dir=pred_dir, dataset_root=str(tmp_path / "ds"),
        unit="mm", reference_subject="cal_bar", labels_dir=labels_dir, reference_csv=ref_csv,
        group_key_map=group_key_map)
    assert "error" in result
    assert "Polygon" in result["error"] or "polygon" in result["error"]


def test_calibrate_physical_scale_refuses_a_reference_stem_outside_the_bucket(tmp_path):
    """A reference object photographed in some other capture says nothing about this bucket's
    scale; the tool refuses naming the stems outside the bucket rather than trusting the CSV."""
    from tcip_mcp.tools.scale_tools import calibrate_physical_scale

    _author_scale_tolerance(tmp_path, "plant_surface_area")
    pred_dir, labels_dir, ref_csv, stems, group_key_map = _calibration_setup(
        tmp_path, lengths_px=[100.0, 100.0])
    # Add a reference row/annotation for a stem never written into the bucket.
    _write_reference_annotation(Path(labels_dir), "outsider", subject="cal_bar",
                                points_by_annotation=[_rect_points(100.0, 10.0)])
    with open(ref_csv, "a", newline="") as f:
        csv.writer(f).writerow(["outsider", 10.0, "mm"])

    result = calibrate_physical_scale(
        trait="plant_surface_area", pred_dir=pred_dir, dataset_root=str(tmp_path / "ds"),
        unit="mm", reference_subject="cal_bar", labels_dir=labels_dir, reference_csv=ref_csv,
        group_key_map={**group_key_map, "outsider": "outsider"})
    assert "error" in result
    assert "outsider" in result["error"]


def test_calibrate_physical_scale_refuses_an_image_with_two_reference_annotations(tmp_path):
    """An image carrying more than one annotation of the reference subject is ambiguous: the tool
    refuses rather than picking one."""
    from tcip_mcp.tools.scale_tools import calibrate_physical_scale

    _author_scale_tolerance(tmp_path, "plant_surface_area")
    pred_dir, labels_dir, ref_csv, stems, group_key_map = _calibration_setup(
        tmp_path, lengths_px=[100.0, 100.0])
    _write_reference_annotation(
        Path(labels_dir), stems[0], subject="cal_bar",
        points_by_annotation=[_rect_points(100.0, 10.0), _rect_points(80.0, 8.0, center=(200, 200))])

    result = calibrate_physical_scale(
        trait="plant_surface_area", pred_dir=pred_dir, dataset_root=str(tmp_path / "ds"),
        unit="mm", reference_subject="cal_bar", labels_dir=labels_dir, reference_csv=ref_csv,
        group_key_map=group_key_map)
    assert "error" in result


def test_calibrate_physical_scale_refuses_with_no_authored_tolerance(tmp_path):
    """TraitSpec.scale_tolerance_frac unset has no platform-provisional fallback: the tool refuses
    and names the field, never validating against a platform-invented number."""
    from tcip_mcp.tools.scale_tools import calibrate_physical_scale

    pred_dir, labels_dir, ref_csv, _stems, group_key_map = _calibration_setup(
        tmp_path, lengths_px=[100.0, 100.0, 100.0, 100.0])

    result = calibrate_physical_scale(
        trait="plant_surface_area", pred_dir=pred_dir, dataset_root=str(tmp_path / "ds"),
        unit="mm", reference_subject="cal_bar", labels_dir=labels_dir, reference_csv=ref_csv,
        group_key_map=group_key_map)
    assert "error" in result
    assert "scale_tolerance_frac" in result["error"]


def test_resolve_physical_scale_refuses_too_few_references_per_half(tmp_path):
    """Either half with fewer than two references refuses, naming the count: an unreplicated
    measurement validates nothing."""
    from tcip_mcp.tools.scale_tools import calibrate_physical_scale

    _author_scale_tolerance(tmp_path, "plant_surface_area")
    pred_dir, labels_dir, ref_csv, _stems, group_key_map = _calibration_setup(
        tmp_path, lengths_px=[100.0, 100.0])

    result = calibrate_physical_scale(
        trait="plant_surface_area", pred_dir=pred_dir, dataset_root=str(tmp_path / "ds"),
        unit="mm", reference_subject="cal_bar", labels_dir=labels_dir, reference_csv=ref_csv,
        group_key_map=group_key_map)
    assert result["passed"] is False
    assert any(f.startswith("insufficient_") for f in result["failures"]), result["failures"]


def test_resolve_physical_scale_refuses_a_wildly_inconsistent_reference_set(tmp_path):
    """A holdout that disagrees with itself, or with the calibration half, by more than the
    authored tolerance cannot validate to it: the reference set's own precision is the floor."""
    from tcip_mcp.tools.scale_tools import calibrate_physical_scale

    _author_scale_tolerance(tmp_path, "plant_surface_area", tolerance_frac=0.05)
    root = tmp_path / "ds"
    pred_dir = root / "predictions" / "preds"
    write_prediction(pred_dir, "img_a")
    labels_dir = tmp_path / "reference_labels"
    csv_path = tmp_path / "reference.csv"
    # Four references, each at a genuinely different implied scale (0.05, 0.1, 0.15, 0.5 mm/px):
    # any 2-2 split disagrees with itself or across halves by far more than 5%.
    scales = [0.05, 0.1, 0.15, 0.5]
    stems = [f"r{i}" for i in range(1, 5)]
    rows = []
    for stem, scale in zip(stems, scales):
        points = _rect_points(100.0, 10.0)
        _write_reference_annotation(labels_dir, stem, subject="cal_bar",
                                    points_by_annotation=[points])
        write_prediction(pred_dir, stem)
        rows.append((stem, round(100.0 * scale, 4), "mm"))
    _write_reference_csv(csv_path, rows)
    group_key_map = {s: s for s in stems}

    result = calibrate_physical_scale(
        trait="plant_surface_area", pred_dir=str(pred_dir), dataset_root=str(root),
        unit="mm", reference_subject="cal_bar", labels_dir=str(labels_dir),
        reference_csv=str(csv_path), group_key_map=group_key_map)
    assert result["passed"] is False, result


def test_calibrate_physical_scale_a_45_degree_bar_validates_the_same_as_axis_aligned(tmp_path):
    """The reference's pixel extent is the principal-axis extent of its own geometry, orientation-
    independent: a bar annotated at 45 degrees implies the same scale an axis-aligned one does."""
    from tcip_mcp.tools.scale_tools import calibrate_physical_scale

    _author_scale_tolerance(tmp_path, "plant_surface_area")
    pred_dir, labels_dir, ref_csv, _stems, group_key_map = _calibration_setup(
        tmp_path, lengths_px=[100.0, 100.0, 100.0, 100.0], angle_deg_by_index={0: 45.0})

    result = calibrate_physical_scale(
        trait="plant_surface_area", pred_dir=pred_dir, dataset_root=str(tmp_path / "ds"),
        unit="mm", reference_subject="cal_bar", labels_dir=labels_dir, reference_csv=ref_csv,
        group_key_map=group_key_map)
    assert result["passed"] is True, result
    assert result["value"] == pytest.approx(0.1)  # 10 mm implied over a 100 px reference length
