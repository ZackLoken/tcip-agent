"""W7 — the generic delivery doors are gated by one shared refuse-or-stamp check.

Covers the single ``check_delivery_gate`` helper and its retrofit onto the previously-ungated
writers/tools: ``export_detection_csv`` / ``export_aggregated_csv`` (writer-level, no MCP wrapper)
and ``tabulate_counts`` (reads the run's resolved validity, not a caller string). The phenology
doors' gate behavior is pinned in the Phase-0 measurement goldens; here we pin the doors W7 newly
gated, plus the escape hatch (acknowledge_unvalidated) that ships an honestly-flagged provisional CSV.
"""

from __future__ import annotations

import csv

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
    # K3 finding #3: a continuous/ordinal trait has no on-disk measurement-validity producer today —
    # a bare caller-asserted measurement_validated string, with no pred_dirs to reconcile against,
    # must never be trusted directly. Refuses without an explicit acknowledge.
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    out = tmp_path / "o.csv"
    with pytest.raises(ValueError):
        export_aggregated_csv([{"plant_id": "p1", "value": 4.2, "observations": 3}], str(out),
                              trait_name="fruit_diameter", measurement_validated=VALIDATED_HELD_OUT)


def test_export_aggregated_csv_continuous_trait_ships_provisional_when_acknowledged(tmp_path):
    # rails-admit-valid-work: the honest provisional path still ships (stamped false), it just
    # can't masquerade as validated on a bare string.
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
