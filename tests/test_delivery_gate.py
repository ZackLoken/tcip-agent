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

import pytest

from tcip_mcp.pipelines.resolution import (
    VALIDATED_FALSE,
    VALIDATED_HELD_OUT,
    VALIDATED_REVIEW_CONFIRMED,
    check_delivery_gate,
)


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
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    op = {"conf": {"value": conf, "validated_against": ref if validated else VALIDATED_FALSE}}
    if tile_size_prov is not None:
        op["tile_size"] = tile_size_prov
    (d / "operating_point.json").write_text(json.dumps({
        "validated": validated, "operating_point": op,
    }), encoding="utf-8")
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


# ── the same tile-geometry dimension, read from a written bucket's sidecar ──

def _write_bucket(path, *, conf_ref, tile_size_prov=None, validated=None):
    """A prediction bucket's operating_point.json, the shape export_predictions writes."""
    path.mkdir(parents=True, exist_ok=True)
    op = {"conf": {"value": 0.4, "requires_validation": True, "validation_kind": "annotations",
                   "validated_against": conf_ref}}
    if tile_size_prov is not None:
        op["tile_size"] = tile_size_prov
    (path / "operating_point.json").write_text(json.dumps({
        "validated": conf_ref == VALIDATED_HELD_OUT if validated is None else validated,
        "operating_point": op,
    }), encoding="utf-8")
    return str(path)


def _tile(ref, value=640):
    return {"value": value, "requires_validation": True, "validation_kind": "geometry",
            "validated_against": ref}


def test_untiled_buckets_leave_the_tile_dimension_out_of_the_gate(tmp_path):
    """A delivery assembled from untiled buckets must not acquire a tile-geometry dimension: the
    scale was never operative, so gating on it would refuse work that was always fine."""
    from tcip_mcp.pipelines.resolution import reconcile_tile_size_validity

    d = _write_bucket(tmp_path / "b1", conf_ref=VALIDATED_HELD_OUT,
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

    d = _write_bucket(tmp_path / "b1", conf_ref=VALIDATED_FALSE,
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

    good = _write_bucket(tmp_path / "b1", conf_ref=VALIDATED_HELD_OUT,
                         tile_size_prov=_tile(VALIDATED_PERSISTED_GEOMETRY, 224))
    bad = _write_bucket(tmp_path / "b2", conf_ref=VALIDATED_HELD_OUT,
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

    a = _write_bucket(tmp_path / "b1", conf_ref=VALIDATED_HELD_OUT,
                      tile_size_prov=_tile(VALIDATED_PERSISTED_GEOMETRY, 224))
    b = _write_bucket(tmp_path / "b2", conf_ref=VALIDATED_HELD_OUT,
                      tile_size_prov=_tile(VALIDATED_EXPLICIT_GEOMETRY, 512))
    recon = reconcile_tile_size_validity([a, b])
    assert recon["validated"] == VALIDATED_EXPLICIT_GEOMETRY


def test_export_aggregated_csv_refuses_a_fabricated_tile_size_with_a_validated_conf(tmp_path):
    """The per-plant CSV aggregates per-image counts, and a tile edge with no persisted training
    geometry and no explicit caller override moves those counts. A cleanly-calibrated conf must not
    paper over it, exactly as it does not at the count-CSV door."""
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    d = _write_bucket(tmp_path / "preds", conf_ref=VALIDATED_HELD_OUT,
                      tile_size_prov=_tile(VALIDATED_FALSE, 640))
    with pytest.raises(ValueError, match="unvalidated measurement"):
        export_aggregated_csv([{"plant_id": "p1", "value": 5, "observations": 2}],
                              str(tmp_path / "o.csv"), trait_name="count", pred_dirs=[d])


def test_export_aggregated_csv_ships_when_the_tile_scale_has_a_real_basis(tmp_path):
    """The rail must admit valid work: a tiled bucket whose tile edge came from the checkpoint's
    own persisted training geometry delivers cleanly and stamps its real reference."""
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv
    from tcip_mcp.pipelines.resolution import VALIDATED_PERSISTED_GEOMETRY

    d = _write_bucket(tmp_path / "preds", conf_ref=VALIDATED_HELD_OUT,
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

    d = _write_bucket(tmp_path / "preds", conf_ref=VALIDATED_HELD_OUT,
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

    d = _write_bucket(tmp_path / "preds", conf_ref=VALIDATED_HELD_OUT,
                      tile_size_prov=_tile(VALIDATED_FALSE, 640))
    out = tmp_path / "o.csv"
    export_aggregated_csv([{"plant_id": "p1", "value": 5, "observations": 2}], str(out),
                          trait_name="count", pred_dirs=[d], acknowledge_unvalidated=True)
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["measurement_validated"] == VALIDATED_FALSE
