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

from tcip_mcp.pipelines.resolution import (
    VALIDATED_FALSE,
    VALIDATED_HELD_OUT,
    VALIDATED_REVIEW_CONFIRMED,
    check_delivery_gate,
)
from tests._binding_fixtures import write_bound_sidecar, write_prediction


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


# ── export_detection_csv (writer) refuses a bare write ─────────────────────

def test_export_detection_csv_refuses_bare_write(tmp_path):
    from tcip_mcp.pipelines.postprocessing.export import export_detection_csv

    with pytest.raises(ValueError, match="unvalidated measurement"):
        export_detection_csv([{"image": "a.jpg", "count": 3}], str(tmp_path / "o.csv"))


def test_export_detection_csv_ships_validated_and_stamps_column(tmp_path):
    from tcip_mcp.pipelines.postprocessing.export import export_detection_csv

    out = tmp_path / "o.csv"
    export_detection_csv([{"image": "a.jpg", "count": 3, "scores": [0.9]}], str(out),
                         measurement_validated=VALIDATED_HELD_OUT)
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["measurement_validated"] == VALIDATED_HELD_OUT


def test_export_detection_csv_acknowledge_stamps_false(tmp_path):
    from tcip_mcp.pipelines.postprocessing.export import export_detection_csv

    out = tmp_path / "o.csv"
    export_detection_csv([{"image": "a.jpg", "count": 3}], str(out),
                         acknowledge_unvalidated=True)
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
    stamp = {"validated": validated, "trait": "catkin", "operating_point": op}
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
                             measurement_validated=VALIDATED_HELD_OUT, pred_dirs=[bucket])


def test_export_detection_csv_pred_dirs_ships_when_bucket_validated(tmp_path):
    from tcip_mcp.pipelines.postprocessing.export import export_detection_csv

    bucket = _detection_bucket(tmp_path, "preds", validated=True)
    out = tmp_path / "o.csv"
    export_detection_csv([{"image": "a.jpg", "count": 3, "scores": [0.9]}], str(out),
                         measurement_validated=VALIDATED_HELD_OUT, pred_dirs=[bucket])
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["measurement_validated"] == VALIDATED_HELD_OUT


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
                             measurement_validated=VALIDATED_HELD_OUT, pred_dirs=[bucket])


def test_export_detection_csv_omitted_pred_dirs_trusts_bare_string(tmp_path):
    # No buckets to reconcile from: measurement_validated is taken as-is, unchanged from before
    # pred_dirs existed (a caller that already resolved the gate against a live run's own bundle).
    from tcip_mcp.pipelines.postprocessing.export import export_detection_csv

    out = tmp_path / "o.csv"
    export_detection_csv([{"image": "a.jpg", "count": 3, "scores": [0.9]}], str(out),
                         measurement_validated=VALIDATED_HELD_OUT)
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["measurement_validated"] == VALIDATED_HELD_OUT


# ── export_aggregated_csv (writer) refuses a bare write ────────────────────

def test_export_aggregated_csv_refuses_bare_write(tmp_path):
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    with pytest.raises(ValueError, match="unvalidated measurement"):
        export_aggregated_csv([{"plant_id": "p1", "value": 5, "observations": 2}],
                              str(tmp_path / "o.csv"), trait_name="count")


def test_export_aggregated_csv_reconciles_sidecar_floor(tmp_path):
    # A count trait reconciles its measurement validity from the prediction buckets' sidecars; a
    # bucket with no operating_point.json floors to false and refuses (never trusts a caller string).
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    bucket = tmp_path / "preds"
    bucket.mkdir()
    with pytest.raises(ValueError, match="unvalidated measurement"):
        export_aggregated_csv([{"plant_id": "p1", "value": 5, "observations": 2}],
                              str(tmp_path / "o.csv"), trait_name="count",
                              measurement_validated=VALIDATED_HELD_OUT, pred_dirs=[str(bucket)])


def test_export_aggregated_csv_continuous_trait_bare_string_never_trusted(tmp_path):
    # A continuous/ordinal trait has no on-disk measurement-validity producer, so a bare
    # caller-asserted measurement_validated string, with no pred_dirs to reconcile against, is
    # never trusted directly: it refuses without an explicit acknowledge.
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    out = tmp_path / "o.csv"
    with pytest.raises(ValueError):
        export_aggregated_csv([{"plant_id": "p1", "value": 4.2, "observations": 3}], str(out),
                              trait_name="fruit_diameter", measurement_validated=VALIDATED_HELD_OUT)


def test_export_aggregated_csv_continuous_trait_ships_provisional_when_acknowledged(tmp_path):
    # The honest provisional path still ships (stamped false); it just can't masquerade as
    # validated on a bare string.
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    out = tmp_path / "o.csv"
    export_aggregated_csv([{"plant_id": "p1", "value": 4.2, "observations": 3}], str(out),
                          trait_name="fruit_diameter", measurement_validated=VALIDATED_HELD_OUT,
                          acknowledge_unvalidated=True)
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["measurement_validated"] == VALIDATED_FALSE


# ── export_aggregated_csv wired to the ordinal/regression sidecar producer ────

def _scalar_bucket(tmp_path, name, task, *, validated, ref=VALIDATED_HELD_OUT, criterion="r_squared"):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    document = f"{task}_operating_point"
    stamp = {
        "validated": validated, "trait": "catkin",
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

    bucket = _scalar_bucket(tmp_path, "preds", "ordinal", validated=True)
    out = tmp_path / "o.csv"
    export_aggregated_csv([{"plant_id": "p1", "value": 2, "observations": 3}], str(out),
                          trait_name="ripening_stage", measurement_validated=VALIDATED_HELD_OUT,
                          pred_dirs=[bucket], task="ordinal")
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["measurement_validated"] == VALIDATED_HELD_OUT


def test_export_aggregated_csv_regression_trait_ships_when_sidecar_validated(tmp_path):
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    bucket = _scalar_bucket(tmp_path, "preds", "regression", validated=True)
    out = tmp_path / "o.csv"
    export_aggregated_csv([{"plant_id": "p1", "value": 4.2, "observations": 3}], str(out),
                          trait_name="fruit_diameter", measurement_validated=VALIDATED_HELD_OUT,
                          pred_dirs=[bucket], task="regression")
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["measurement_validated"] == VALIDATED_HELD_OUT


def test_export_aggregated_csv_ordinal_trait_floors_on_missing_sidecar(tmp_path):
    # A bucket with no ordinal_operating_point.json floors to false and refuses (never trusts a
    # caller-asserted measurement_validated string), the same reconcile-from-disk discipline the
    # count operating point already has.
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    bucket = tmp_path / "preds"
    bucket.mkdir()
    with pytest.raises(ValueError, match="unvalidated measurement"):
        export_aggregated_csv([{"plant_id": "p1", "value": 2, "observations": 3}],
                              str(tmp_path / "o.csv"), trait_name="ripening_stage",
                              measurement_validated=VALIDATED_HELD_OUT, pred_dirs=[str(bucket)],
                              task="ordinal")


def test_export_aggregated_csv_regression_trait_floors_on_a_failed_sidecar(tmp_path):
    # A sidecar that exists but is stamped unvalidated (the calibration ran and refused) must also
    # refuse, not just an entirely-missing sidecar.
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    bucket = _scalar_bucket(tmp_path, "preds", "regression", validated=False)
    with pytest.raises(ValueError, match="unvalidated measurement"):
        export_aggregated_csv([{"plant_id": "p1", "value": 4.2, "observations": 3}],
                              str(tmp_path / "o.csv"), trait_name="fruit_diameter",
                              measurement_validated=VALIDATED_HELD_OUT, pred_dirs=[str(bucket)],
                              task="regression")


def test_export_aggregated_csv_rejects_an_unrecognized_task(tmp_path):
    # A typo'd task must raise, not silently fall through to reconciling pred_dirs against the
    # count operating point's own sidecar (a different dimension, wrong validity determination).
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    bucket = _scalar_bucket(tmp_path, "preds", "ordinal", validated=True)
    with pytest.raises(ValueError, match="task must be"):
        export_aggregated_csv([{"plant_id": "p1", "value": 2, "observations": 3}],
                              str(tmp_path / "o.csv"), trait_name="ripening_stage",
                              measurement_validated=VALIDATED_HELD_OUT, pred_dirs=[str(bucket)],
                              task="oridnal")


# ── tabulate_counts reads the run's resolved validity, not a caller string ─

def test_tabulate_counts_refuses_unvalidated_run(tmp_path, monkeypatch):
    import tcip_mcp.tools.inference_tools as itools

    def _fake_run_inference(**kw):
        return {"results": [{"image": "a.png", "count": 3}], "image_count": 1,
                "total_detections": 3, "operating_point": {"conf": {"value": 0.5}},
                "validated": False, "conf_source": "default"}

    monkeypatch.setattr(itools, "run_inference", _fake_run_inference)
    r = itools.tabulate_counts("m.pt", str(tmp_path), str(tmp_path / "o.csv"))
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

    def _fake_export(results, path, provenance=None, measurement_validated=None,
                     acknowledge_unvalidated=False):
        captured["measurement_validated"] = measurement_validated
        captured["acknowledge_unvalidated"] = acknowledge_unvalidated
        return str(path)

    monkeypatch.setattr(itools, "run_inference", _fake_run_inference)
    monkeypatch.setattr(itools, "export_detection_csv", _fake_export)
    r = itools.tabulate_counts("m.pt", str(tmp_path), str(tmp_path / "o.csv"),
                               acknowledge_unvalidated=True)
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
                               trait="catkin", calibration_labels_dir=str(tmp_path))
    assert "error" in r
    assert r["operating_point_validated"] == VALIDATED_HELD_OUT  # conf itself is fine...
    assert r["tile_size_validated"] == VALIDATED_FALSE           # ...tile_size is what refuses
    assert not (tmp_path / "o.csv").exists()


def test_tabulate_counts_ships_when_tile_size_has_a_real_basis(tmp_path, monkeypatch):
    """The rail must admit valid work, not only reject invalid work: a tile_size genuinely derived
    from the checkpoint's persisted training geometry ships cleanly, same as a validated conf."""
    import tcip_mcp.tools.inference_tools as itools

    from tcip_mcp.pipelines.resolution import VALIDATED_PERSISTED_GEOMETRY

    captured = {}
    monkeypatch.setattr(itools, "run_inference", _fake_run_inference_with(
        conf_ref=VALIDATED_HELD_OUT,
        tile_size_prov={"value": 224, "requires_validation": True,
                        "validation_kind": "geometry",
                        "validated_against": VALIDATED_PERSISTED_GEOMETRY}))
    monkeypatch.setattr(
        itools, "export_detection_csv",
        lambda results, path, provenance=None, measurement_validated=None,
        acknowledge_unvalidated=False: (captured.update(mv=measurement_validated) or str(path)))
    r = itools.tabulate_counts("m.pt", str(tmp_path), str(tmp_path / "o.csv"),
                               trait="catkin", calibration_labels_dir=str(tmp_path))
    assert "error" not in r
    assert r["tile_size_validated"] == VALIDATED_PERSISTED_GEOMETRY
    assert captured["mv"] == VALIDATED_HELD_OUT  # the CSV stamp reflects the fully-cleared gate


def test_tabulate_counts_never_gates_tile_size_when_untiled(tmp_path, monkeypatch):
    """An untiled run's tile_size is never operative: it must not manufacture a refusal just
    because the run's own bundle happens to carry a non-gating tile_size entry."""
    import tcip_mcp.tools.inference_tools as itools

    monkeypatch.setattr(itools, "run_inference", _fake_run_inference_with(
        conf_ref=VALIDATED_HELD_OUT,
        tile_size_prov={"value": None, "requires_validation": False,
                        "validation_kind": None, "validated_against": None}))
    r = itools.tabulate_counts("m.pt", str(tmp_path), str(tmp_path / "o.csv"),
                               trait="catkin", calibration_labels_dir=str(tmp_path))
    assert "error" not in r
    assert r["tile_size_validated"] is None  # never entered the gate at all


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
    monkeypatch.setattr(
        itools, "export_detection_csv",
        lambda results, path, provenance=None, measurement_validated=None,
        acknowledge_unvalidated=False: (captured.update(mv=measurement_validated) or str(path)))
    r = itools.tabulate_counts("m.pt", str(tmp_path), str(tmp_path / "o.csv"),
                               trait="catkin", calibration_labels_dir=str(tmp_path),
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


def test_export_predictions_ships_when_tile_size_has_a_real_basis(tmp_path, monkeypatch):
    """The rail must admit valid work, not only reject invalid work."""
    import tcip_mcp.tools.inference_tools as itools

    from tcip_mcp.pipelines.resolution import VALIDATED_PERSISTED_GEOMETRY

    monkeypatch.setattr(itools, "run_inference", lambda **kw: _fake_run_inference_result(
        conf_ref=VALIDATED_HELD_OUT,
        tile_size_prov={"value": 224, "requires_validation": True,
                        "validation_kind": "geometry",
                        "validated_against": VALIDATED_PERSISTED_GEOMETRY}))
    out = tmp_path / "preds"
    r = itools.export_predictions("m.pt", str(tmp_path), str(out))
    assert "error" not in r
    assert r["tile_size_validated"] == VALIDATED_PERSISTED_GEOMETRY
    assert r["validated"] is True
    sidecar = json.loads((out / "operating_point.json").read_text())
    assert sidecar["tile_size_validated"] == VALIDATED_PERSISTED_GEOMETRY
    assert sidecar["validated"] is True


def test_export_predictions_never_gates_tile_size_when_untiled(tmp_path, monkeypatch):
    """An untiled run's tile_size is never operative: it must not manufacture a refusal just
    because the run's own bundle happens to carry a non-gating tile_size entry."""
    import tcip_mcp.tools.inference_tools as itools

    monkeypatch.setattr(itools, "run_inference", lambda **kw: _fake_run_inference_result(
        conf_ref=VALIDATED_HELD_OUT,
        tile_size_prov={"value": None, "requires_validation": False,
                        "validation_kind": None, "validated_against": None}))
    out = tmp_path / "preds"
    r = itools.export_predictions("m.pt", str(tmp_path), str(out))
    assert "error" not in r
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
    sidecar = json.loads((out / "operating_point.json").read_text())
    assert sidecar["validated"] is False
    assert (out / "a.json").exists()  # the honestly-flagged provisional bucket still wrote


def test_export_predictions_images_dir_gates_before_the_pass_not_after(tmp_path, monkeypatch):
    """DECIDED #1: the images_dir regime's gate runs before the (expensive) pass, the same
    ordering the raster_path regime already had, not only after run_inference already ran it.
    A real checkpoint whose only basis is the native-ratio tier (a real but never-shippable
    basis) must refuse without ever reaching the model's own forward pass; GenericPredictor's
    predict_batch is monkeypatched to raise if called at all, so this proves the skip, not just
    that no bucket got written."""
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
        "config": {"data": {"train_native_size": [64, 64]}, "augmentation": {}},
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
    op = json.loads((out_dir / "operating_point.json").read_text())["operating_point"]
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
    op = json.loads((out_dir / "operating_point.json").read_text())["operating_point"]
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
    op = json.loads((out_dir / "operating_point.json").read_text())["operating_point"]
    assert op["tiled"]["value"] is False
    assert _sidecar_tile_reference(out_dir) is None  # never entered the gate at all


# ── the same tile-geometry dimension, read from a written bucket's sidecar ──

def _write_bucket(tmp_path, name, *, conf_ref, tile_size_prov=None, validated=None):
    """A prediction bucket's operating_point.json, the shape export_predictions writes."""
    root = tmp_path / "ds"
    d = root / "predictions" / name
    write_prediction(d, "img_a")
    op = {"conf": {"value": 0.4, "requires_validation": True, "validation_kind": "annotations",
                   "validated_against": conf_ref}}
    if tile_size_prov is not None:
        op["tile_size"] = tile_size_prov
    is_validated = (conf_ref == VALIDATED_HELD_OUT) if validated is None else validated
    stamp = {"validated": is_validated, "trait": "catkin", "operating_point": op}
    if is_validated:
        write_bound_sidecar(d, stamp, dataset_root=root, experiment_id=f"exp-{name}")
    else:
        (d / "operating_point.json").write_text(json.dumps(stamp), encoding="utf-8")
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
        export_aggregated_csv([{"plant_id": "p1", "value": 5, "observations": 2}],
                              str(tmp_path / "o.csv"), trait_name="count", pred_dirs=[d])


def test_export_aggregated_csv_ships_when_the_tile_scale_has_a_real_basis(tmp_path):
    """The rail must admit valid work: a tiled bucket whose tile edge came from the checkpoint's
    own persisted training geometry delivers cleanly and stamps its real reference."""
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv
    from tcip_mcp.pipelines.resolution import VALIDATED_PERSISTED_GEOMETRY

    d = _write_bucket(tmp_path, "preds", conf_ref=VALIDATED_HELD_OUT,
                      tile_size_prov=_tile(VALIDATED_PERSISTED_GEOMETRY, 224))
    out = tmp_path / "o.csv"
    export_aggregated_csv([{"plant_id": "p1", "value": 5, "observations": 2}], str(out),
                          trait_name="count", pred_dirs=[d])
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
    export_aggregated_csv([{"plant_id": "p1", "value": 5, "observations": 2}], str(out),
                          trait_name="count", pred_dirs=[d])
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
    export_aggregated_csv([{"plant_id": "p1", "value": 5, "observations": 2}], str(out),
                          trait_name="count", pred_dirs=[d], acknowledge_unvalidated=True)
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["measurement_validated"] == VALIDATED_FALSE


# ── export_aggregated_csv gates a dimensional value_key on its physical scale too ──

def _write_scale_sidecar(path, *, validated_against, capture_id=None, value=0.05, unit="mm"):
    """A bucket's resolve_scale.json, the shape reconcile_scale_validity reads, alongside its
    operating_point.json in the same directory."""
    from tcip_mcp.pipelines.resolution import VALIDATED_PHYSICAL_MEASUREMENT

    path.mkdir(parents=True, exist_ok=True)
    is_validated = validated_against == VALIDATED_PHYSICAL_MEASUREMENT
    stamp = {
        "validated": is_validated, "trait": "catkin",
        "operating_point": {
            "scale": {
                "value": value, "unit": unit, "capture_id": capture_id,
                "requires_validation": True, "validation_kind": "physical",
                "validated_against": validated_against,
            },
        },
    }
    if is_validated:
        write_bound_sidecar(path, stamp, document="resolve_scale", dataset_root=path,
                            experiment_id=f"exp-scale-{path.name}")
    else:
        (path / "resolve_scale.json").write_text(json.dumps(stamp), encoding="utf-8")
    return str(path)


_DIM_RESULTS = [{"plant_id": "p1", "value": 12.5, "observations": 1, "value_key": "area_mm2"}]


def test_export_aggregated_csv_ships_dimensional_value_with_a_validated_scale(tmp_path):
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv
    from tcip_mcp.pipelines.resolution import VALIDATED_PHYSICAL_MEASUREMENT

    d = _write_bucket(tmp_path, "preds", conf_ref=VALIDATED_HELD_OUT)
    _write_scale_sidecar(Path(d), validated_against=VALIDATED_PHYSICAL_MEASUREMENT)
    out = tmp_path / "o.csv"
    export_aggregated_csv(_DIM_RESULTS, str(out), trait_name="not_a_real_crops_yml_trait", pred_dirs=[d])
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["measurement_validated"] == VALIDATED_HELD_OUT
    assert rows[0]["units"] == "mm2"


def test_export_aggregated_csv_refuses_a_dimensional_delivery_with_no_scale_sidecar(tmp_path):
    """A dimensional CSV must not ship stamped validated when its physical scale was never checked
    against anything, even though the count operating point beside it is genuinely validated: this
    is exactly the gap the scale gate closes."""
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    d = _write_bucket(tmp_path, "preds", conf_ref=VALIDATED_HELD_OUT)
    with pytest.raises(ValueError, match="unvalidated measurement"):
        export_aggregated_csv(_DIM_RESULTS, str(tmp_path / "o.csv"),
                              trait_name="not_a_real_crops_yml_trait", pred_dirs=[d])


def test_export_aggregated_csv_count_trait_never_gates_on_scale(tmp_path):
    """A count trait's value_key implies no physical unit, so the scale dimension never becomes
    operative even though pred_dirs is given: nothing dimensional to protect, nothing to refuse
    over."""
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    d = _write_bucket(tmp_path, "preds", conf_ref=VALIDATED_HELD_OUT)
    out = tmp_path / "o.csv"
    export_aggregated_csv([{"plant_id": "p1", "value": 5, "observations": 2, "value_key": "count"}],
                          str(out), trait_name="count", pred_dirs=[d])
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["measurement_validated"] == VALIDATED_HELD_OUT


def test_export_aggregated_csv_scale_capture_id_mismatch_floors(tmp_path):
    """A handheld standoff's scale can vary capture to capture: a sidecar validated for a different
    capture than the one this delivery names must not silently clear this delivery's scale."""
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv
    from tcip_mcp.pipelines.resolution import VALIDATED_PHYSICAL_MEASUREMENT

    d = _write_bucket(tmp_path, "preds", conf_ref=VALIDATED_HELD_OUT)
    _write_scale_sidecar(Path(d), validated_against=VALIDATED_PHYSICAL_MEASUREMENT,
                         capture_id="2026-02-10_plot7")
    with pytest.raises(ValueError, match="unvalidated measurement"):
        export_aggregated_csv(_DIM_RESULTS, str(tmp_path / "o.csv"),
                              trait_name="not_a_real_crops_yml_trait", pred_dirs=[d],
                              scale_capture_id="2026-02-10_plot9")


def test_export_aggregated_csv_scale_capture_id_match_ships(tmp_path):
    """The rail must admit valid work: the scale's own recorded capture matches the one this
    delivery is scoped to, so it ships cleanly."""
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv
    from tcip_mcp.pipelines.resolution import VALIDATED_PHYSICAL_MEASUREMENT

    d = _write_bucket(tmp_path, "preds", conf_ref=VALIDATED_HELD_OUT)
    _write_scale_sidecar(Path(d), validated_against=VALIDATED_PHYSICAL_MEASUREMENT,
                         capture_id="2026-02-10_plot7")
    out = tmp_path / "o.csv"
    export_aggregated_csv(_DIM_RESULTS, str(out), trait_name="not_a_real_crops_yml_trait",
                          pred_dirs=[d], scale_capture_id="2026-02-10_plot7")
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["measurement_validated"] == VALIDATED_HELD_OUT


def test_export_aggregated_csv_acknowledged_unvalidated_scale_floors_the_row_stamp(tmp_path):
    """A dimensional CSV whose conf is genuinely validated but whose scale never cleared must stamp
    its one measurement column false when shipped through acknowledge_unvalidated, not conf's clean
    reference."""
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    d = _write_bucket(tmp_path, "preds", conf_ref=VALIDATED_HELD_OUT)
    out = tmp_path / "o.csv"
    export_aggregated_csv(_DIM_RESULTS, str(out), trait_name="not_a_real_crops_yml_trait",
                          pred_dirs=[d], acknowledge_unvalidated=True)
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["measurement_validated"] == VALIDATED_FALSE
