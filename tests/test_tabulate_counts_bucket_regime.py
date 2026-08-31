"""tabulate_counts' bucket regime: a persisted, reviewed prediction bucket in, no GPU re-run.

Covers the source-regime discrimination, the bucket's positive-claim checks (a readable stamp,
the per-image mandatory shape, the trait binding), the publish bracket the live regime shares with
export_predictions, and the refusal-channel split between a meaning-door raise and a gate refusal.
"""

from __future__ import annotations

import csv
import subprocess
import sys

import pytest

from tcip_mcp.pipelines.resolution import VALIDATED_FALSE, VALIDATED_HELD_OUT
from tests import _operationalization_fixtures as fx
from tests._binding_fixtures import calibrated_run_fields, write_bound_sidecar, write_prediction


@pytest.fixture(autouse=True)
def _stub_checkpoint_verification(monkeypatch):
    """Every test here exercises the door's own logic, not a checkpoint load."""
    import tcip_mcp.model_registry as model_registry_mod
    from tests._verified_checkpoint_fixtures import stub_verified_checkpoint

    monkeypatch.setattr(model_registry_mod, "load_registered_checkpoint",
                        lambda path, *a, **kw: stub_verified_checkpoint(str(path)))


@pytest.fixture(autouse=True)
def _recorded_meaning(tmp_path):
    fx.seed_delivery_traits(tmp_path)
    fx.seed_confirmed_count(tmp_path)


def _dummy_checkpoint(tmp_path) -> str:
    p = tmp_path / "m.pt"
    if not p.exists():
        p.write_bytes(b"x")
    return str(p)


def _write_real_prediction(bucket, stem: str, *, score: float = 0.9) -> None:
    """One real per-image prediction document, through the platform's own writer, decoded to
    fx.COUNT_SUBJECT via a recorded id_map, so the bucket regime's own counting reader has a real
    detection (not the bare content-hashing stand-in ``write_prediction`` writes) to count."""
    from pathlib import Path

    from tcip_mcp.pipelines.postprocessing.export import write_predictions_json

    Path(bucket).mkdir(parents=True, exist_ok=True)
    result = {"image": f"{stem}.png", "width": 100, "height": 100,
             "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [score], "labels": [1], "count": 1}
    write_predictions_json(Path(bucket) / f"{stem}.json", result, created_by="test-producer",
                           id_map={fx.COUNT_SUBJECT: 0})


def _unvalidated_run_result(*, experiment_id=None):
    """A stand-in live-run result honestly stamped unvalidated: bypasses the earned-evidence path
    (``_draft_count_claim`` returns nothing to open) so a test about the publish bracket's own
    tile/lineage checks is not entangled with the calibration-evidence machinery."""
    return {"results": [{"image": "a.png", "width": 100, "height": 100,
                        "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9], "labels": [1],
                        "count": 1}],
           "image_count": 1, "total_detections": 1, "id_map": None,
           "operating_point": {"conf": {"value": 0.5}}, "validated": False,
           "conf_source": "default", "experiment_id": experiment_id,
           "checkpoint_sha256": "deadbeef"}


def _earned_run_result(tmp_path, *, trait=fx.COUNT_TRAIT, tiled=False, tile_size=None,
                       tile_size_source="default"):
    """A stand-in live-run result that left behind real evidence, so a bucket published from it
    earns a genuine validation record (the same shape test_delivery_gate.py's own helper builds)."""
    return {
        "results": [{"image": "a.png", "width": 100, "height": 100,
                    "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9], "labels": [1],
                    "count": 1}],
        "image_count": 1, "total_detections": 1, "id_map": None,
        "produced_at": "2026-01-01T00:00:00Z",
        **calibrated_run_fields(trait, labels_dir=tmp_path, checkpoint_sha256="deadbeef",
                                tiled=tiled, tile_size=tile_size, tile_size_source=tile_size_source),
    }


# ── exactly one source, or refuse naming both regimes ──────────────────────

def test_neither_source_stated_refuses_naming_both_regimes(tmp_path):
    import tcip_mcp.tools.inference_tools as itools

    r = itools.tabulate_counts(output_path=str(tmp_path / "o.csv"), trait=fx.COUNT_TRAIT)
    assert "error" in r
    assert "checkpoint_path" in r["error"] and "predictions_dir" in r["error"]


def test_images_dir_with_no_checkpoint_path_refuses(tmp_path):
    import tcip_mcp.tools.inference_tools as itools

    r = itools.tabulate_counts(images_dir=str(tmp_path), output_path=str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT)
    assert "error" in r
    assert "checkpoint_path" in r["error"]


def test_no_output_path_refuses(tmp_path):
    import tcip_mcp.tools.inference_tools as itools

    r = itools.tabulate_counts(predictions_dir=str(tmp_path / "preds"), trait=fx.COUNT_TRAIT)
    assert "error" in r
    assert "output_path" in r["error"]


# ── live-only parameters refuse in the bucket regime, including at their own default ──

@pytest.mark.parametrize("name, value", [
    ("conf_threshold", 0.5), ("device", "cpu"), ("tile", True), ("tile_size", 320),
    ("overlap", 0.2), ("global_nms_iou", 0.4), ("max_dets", 50),
    ("calibration_labels_dir", "labels"), ("calibration_images_dir", "images"),
    ("split_manifest_dir", "manifest"), ("experiment_id", "exp-live-only"),
    ("postprocess", "nmm"), ("tile_batch_size", 32),
])
def test_each_live_only_parameter_refuses_in_the_bucket_regime(tmp_path, name, value):
    """Every parameter design section 3 names live-only refuses by name in the bucket regime,
    including the two non-None-defaulted ones (postprocess, tile_batch_size) away from their own
    documented default."""
    import tcip_mcp.tools.inference_tools as itools

    bucket = tmp_path / "preds"
    write_prediction(bucket, "a")
    stamp = {"trait": fx.COUNT_TRAIT, "images_dir": str(tmp_path), "raster_path": None,
             "operating_point": {"conf": {"value": 0.5, "validated_against": VALIDATED_FALSE}}}
    write_bound_sidecar(bucket, stamp, dataset_root=tmp_path)

    if name in ("calibration_labels_dir", "calibration_images_dir", "split_manifest_dir"):
        value = str(tmp_path / value)
    r = itools.tabulate_counts(predictions_dir=str(bucket), output_path=str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT, **{name: value})
    assert "error" in r
    assert name in r["error"]


def test_a_live_only_parameter_stated_at_its_own_default_is_silently_admitted(tmp_path):
    """The rail must admit valid work: tile_batch_size's non-None default (96) makes
    stated-at-default indistinguishable from stating nothing, so a bucket regime call naming it
    at that default is honestly admitted rather than refused for a statement it cannot detect."""
    import tcip_mcp.tools.inference_tools as itools

    bucket = tmp_path / "preds"
    _write_real_prediction(bucket, "a")
    stamp = {"trait": fx.COUNT_TRAIT, "images_dir": str(tmp_path), "raster_path": None,
             "operating_point": {"conf": {"value": 0.5, "validated_against": VALIDATED_FALSE}}}
    write_bound_sidecar(bucket, stamp, dataset_root=tmp_path)

    r = itools.tabulate_counts(predictions_dir=str(bucket), output_path=str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT, tile_batch_size=96,
                               acknowledge_unvalidated=True)
    assert "error" not in r, r


# ── the bucket's positive stamp shape ───────────────────────────────────────

def test_bucket_regime_refuses_a_directory_with_no_stamp(tmp_path):
    """A directory of label JSON with no stamp (a GT annotations tree) is refused, never counted,
    whatever acknowledge_unvalidated says: the platform's own label writer produces the tree."""
    import tcip_mcp.tools.inference_tools as itools
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    labels_dir = tmp_path / "annotations" / "2026-01-01"
    json_io.write_annotations(
        str(labels_dir / "a.json"), [Annotation(subject=fx.COUNT_SUBJECT, geometry=BBox(0, 0, 5, 5))],
        100, 100)

    r = itools.tabulate_counts(predictions_dir=str(labels_dir), output_path=str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT, acknowledge_unvalidated=True)
    assert "error" in r
    assert "operating_point.json" in r["error"]


def test_bucket_regime_refuses_a_mosaic_bucket(tmp_path):
    """A stamp recording raster_path (a whole-mosaic bucket) is refused on its own mandatory
    shape, naming deliver_orthomosaic_plant_counts: one mosaic total is not a per-image count."""
    import tcip_mcp.tools.inference_tools as itools

    bucket = tmp_path / "mosaic_preds"
    write_prediction(bucket, "tile_0")
    stamp = {"trait": fx.COUNT_TRAIT, "images_dir": None, "raster_path": "mosaic.tif",
             "operating_point": {"conf": {"value": 0.5, "validated_against": VALIDATED_FALSE}}}
    write_bound_sidecar(bucket, stamp, dataset_root=tmp_path)

    r = itools.tabulate_counts(predictions_dir=str(bucket), output_path=str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT, acknowledge_unvalidated=True)
    assert "error" in r
    assert "deliver_orthomosaic_plant_counts" in r["error"]


def test_bucket_regime_refuses_a_stamp_naming_neither_images_dir_nor_raster_path(tmp_path):
    import tcip_mcp.tools.inference_tools as itools

    bucket = tmp_path / "bare_preds"
    write_prediction(bucket, "a")
    stamp = {"trait": fx.COUNT_TRAIT, "images_dir": None, "raster_path": None,
             "operating_point": {"conf": {"value": 0.5, "validated_against": VALIDATED_FALSE}}}
    write_bound_sidecar(bucket, stamp, dataset_root=tmp_path)

    r = itools.tabulate_counts(predictions_dir=str(bucket), output_path=str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT, acknowledge_unvalidated=True)
    assert "error" in r
    assert "per-image" in r["error"]


def test_bucket_regime_refuses_a_stamp_naming_an_empty_string_images_dir(tmp_path):
    """An empty-string images_dir records nothing, the same as no images_dir at all: the per-image
    shape check is a truthy check, not merely a not-None one, so it cannot be satisfied by a key
    present with nothing behind it."""
    import tcip_mcp.tools.inference_tools as itools

    bucket = tmp_path / "empty_images_dir_preds"
    _write_real_prediction(bucket, "a")
    stamp = {"trait": fx.COUNT_TRAIT, "images_dir": "", "raster_path": None,
             "operating_point": {"conf": {"value": 0.5, "validated_against": VALIDATED_FALSE}}}
    write_bound_sidecar(bucket, stamp, dataset_root=tmp_path)

    r = itools.tabulate_counts(predictions_dir=str(bucket), output_path=str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT, acknowledge_unvalidated=True)
    assert "error" in r
    assert "per-image" in r["error"]


def test_bucket_regime_refuses_a_trait_contradiction_even_unvalidated(tmp_path):
    """An unvalidated stamp recording a different, non-None trait refuses as a positive
    contradiction: verify_stamp_binding only compares traits when a stamp claims validation, so
    the door itself has to catch the unvalidated case."""
    import tcip_mcp.tools.inference_tools as itools

    other_trait = "astringency"
    from tcip_mcp import operationalization as op

    record = op.state_operationalization(
        tmp_path, other_trait, op.PER_IMAGE_COUNT,
        statement="how many astringent structures the model finds in one frame",
        mechanism="the calibrated detector over whole frames at the derived operating point",
        measured_subject=fx.COUNT_SUBJECT, delivered_phenotypes=[],
    )
    fx.confirm(tmp_path, other_trait, op.PER_IMAGE_COUNT, record)

    bucket = tmp_path / "other_trait_preds"
    write_prediction(bucket, "a")
    stamp = {"trait": other_trait, "images_dir": str(tmp_path), "raster_path": None,
             "operating_point": {"conf": {"value": 0.5, "validated_against": VALIDATED_FALSE}}}
    write_bound_sidecar(bucket, stamp, dataset_root=tmp_path)

    r = itools.tabulate_counts(predictions_dir=str(bucket), output_path=str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT, acknowledge_unvalidated=True)
    assert "error" in r
    assert other_trait in r["error"] and fx.COUNT_TRAIT in r["error"]


def test_bucket_regime_admits_a_stamp_naming_no_trait_at_all(tmp_path):
    """The rail must admit valid work: a stamp that names no trait (trait=None) is not a
    contradiction, so a bucket-regime call over it delivers under the caller's stated trait."""
    import tcip_mcp.tools.inference_tools as itools

    bucket = tmp_path / "no_trait_preds"
    _write_real_prediction(bucket, "a")
    stamp = {"trait": None, "images_dir": str(tmp_path), "raster_path": None,
             "operating_point": {"conf": {"value": 0.5, "validated_against": VALIDATED_FALSE}}}
    write_bound_sidecar(bucket, stamp, dataset_root=tmp_path)

    r = itools.tabulate_counts(predictions_dir=str(bucket), output_path=str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT, acknowledge_unvalidated=True)
    assert "error" not in r, r


def test_bucket_regime_refuses_a_stamped_bucket_with_no_prediction_documents(tmp_path):
    """A stamped bucket holding zero prediction documents refuses naming the fact, before any CSV
    is written: an empty bucket is not a per-image count either, however honestly its stamp reads."""
    import tcip_mcp.tools.inference_tools as itools

    bucket = tmp_path / "empty_preds"
    stamp = {"trait": fx.COUNT_TRAIT, "images_dir": str(tmp_path), "raster_path": None,
             "operating_point": {"conf": {"value": 0.5, "validated_against": VALIDATED_HELD_OUT}}}
    write_bound_sidecar(bucket, stamp, dataset_root=tmp_path)

    out_csv = tmp_path / "o.csv"
    r = itools.tabulate_counts(predictions_dir=str(bucket), output_path=str(out_csv),
                               trait=fx.COUNT_TRAIT, acknowledge_unvalidated=True)
    assert "error" in r
    assert "no prediction documents" in r["error"]
    assert not out_csv.exists()


def test_bucket_regime_measured_subject_check_is_driven_by_a_recorded_id_map(tmp_path):
    """A bucket whose sidecar records an id_map drives the measured-subject check both ways: a
    stated trait whose confirmed subject is absent from every recorded id_map refuses naming it,
    and one present in it delivers."""
    from tcip_mcp import operationalization as op
    import tcip_mcp.tools.inference_tools as itools

    other_trait = "astringency"
    other_subject = "a subject no recorded id_map names"
    record = op.state_operationalization(
        tmp_path, other_trait, op.PER_IMAGE_COUNT,
        statement="how many astringent structures the model finds in one frame",
        mechanism="the calibrated detector over whole frames at the derived operating point",
        measured_subject=other_subject, delivered_phenotypes=[],
    )
    fx.confirm(tmp_path, other_trait, op.PER_IMAGE_COUNT, record)

    bucket = tmp_path / "id_mapped_preds"
    _write_real_prediction(bucket, "a")
    stamp = {"trait": None, "images_dir": str(tmp_path), "raster_path": None,
             "id_map": {fx.COUNT_SUBJECT: 0},
             "operating_point": {"conf": {"value": 0.5, "validated_against": VALIDATED_FALSE}}}
    write_bound_sidecar(bucket, stamp, dataset_root=tmp_path)

    with pytest.raises(ValueError, match="a subject no recorded id_map names"):
        itools.tabulate_counts(predictions_dir=str(bucket),
                               output_path=str(tmp_path / "mismatch.csv"),
                               trait=other_trait, acknowledge_unvalidated=True)

    match = itools.tabulate_counts(predictions_dir=str(bucket), output_path=str(tmp_path / "match.csv"),
                                   trait=fx.COUNT_TRAIT, acknowledge_unvalidated=True)
    assert "error" not in match, match


# ── the publish bracket: shared with export_predictions ────────────────────

def test_publish_bracket_refuses_a_fabricated_tile_with_the_bucket_left_absent(tmp_path, monkeypatch):
    """A tiled run whose tile scale has no real basis refuses before anything lands, gated exactly
    as export_predictions gates it: the bucket is left absent, not published unvalidated."""
    import tcip_mcp.tools.inference_tools as itools

    def _fake(*a, **kw):
        return {"results": [{"image": "a.png", "count": 1, "scores": [0.9]}], "image_count": 1,
                "total_detections": 1,
                "operating_point": {"conf": {"value": 0.6, "validated_against": VALIDATED_HELD_OUT},
                                    "tile_size": {"value": 640, "requires_validation": True,
                                                  "validation_kind": "geometry",
                                                  "validated_against": VALIDATED_FALSE}},
                "validated": True, "conf_source": "calibration", "experiment_id": None,
                "checkpoint_sha256": "deadbeef"}

    monkeypatch.setattr(itools, "_run_inference_verified", _fake)
    bucket = tmp_path / "ds" / "predictions" / "baseline" / "2026-01-01"
    r = itools.tabulate_counts(_dummy_checkpoint(tmp_path), str(tmp_path),
                               str(tmp_path / "o.csv"), trait=fx.COUNT_TRAIT,
                               predictions_dir=str(bucket))
    assert "error" in r
    assert not bucket.exists()


def test_publish_bracket_refuses_a_frozen_lineage_pointer(tmp_path, monkeypatch):
    """A second live publish for a terminal experiment whose lineage already names a different
    bucket refuses; the remedy is the bucket regime, re-delivering from the existing bucket."""
    import tcip_mcp.tools.inference_tools as itools
    from tcip_mcp.experiments import create_experiment, update_lineage, update_status

    eid = "exp-tabulate-lineage-frozen"
    create_experiment(eid, {"note": "producing run"})
    update_status(eid, "running")
    update_lineage(eid, predictions=str(tmp_path / "already-published"))
    update_status(eid, "completed")

    monkeypatch.setattr(itools, "_run_inference_verified",
                        lambda *a, **kw: _unvalidated_run_result(experiment_id=eid))
    bucket = tmp_path / "ds" / "predictions" / "baseline" / "2026-01-01"
    r = itools.tabulate_counts(_dummy_checkpoint(tmp_path), str(tmp_path), str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT, predictions_dir=str(bucket),
                               acknowledge_unvalidated=True)
    assert "error" in r
    assert eid in r["error"]
    assert not bucket.exists()


def test_publish_bracket_links_a_resolvable_experiments_bucket_into_its_lineage(tmp_path, monkeypatch):
    import tcip_store
    import tcip_mcp.tools.inference_tools as itools
    from tcip_mcp.experiments import create_experiment, lineage_key, update_status

    eid = "exp-tabulate-lineage-link"
    create_experiment(eid, {"note": "producing run"})
    update_status(eid, "running")

    monkeypatch.setattr(itools, "_run_inference_verified",
                        lambda *a, **kw: _unvalidated_run_result(experiment_id=eid))
    bucket = tmp_path / "ds" / "predictions" / "baseline" / "2026-01-01"
    r = itools.tabulate_counts(_dummy_checkpoint(tmp_path), str(tmp_path), str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT, predictions_dir=str(bucket),
                               acknowledge_unvalidated=True)
    assert "error" not in r, r
    assert r["lineage_linked"] is True
    lineage = tcip_store.read(lineage_key(eid), default={})
    assert lineage.get("predictions") == str(bucket)


# ── refusal-channel separation ──────────────────────────────────────────────

def test_a_withdrawn_operationalization_mid_flow_is_count_free_in_the_bucket_regime(
    tmp_path, monkeypatch,
):
    """The writer's own meaning-door raise (a confirmation withdrawn since the door's own first
    check) propagates bare past tabulate_counts, never composed into the counts-bearing refusal
    dict a gate refusal gets: a count-bearing response can only come from a caught DeliveryRefused."""
    from dataclasses import replace

    import tcip_mcp.tools.inference_tools as itools
    from tcip_mcp import operationalization as op

    bucket = tmp_path / "ds" / "predictions" / "baseline" / "2026-01-01"
    _write_real_prediction(bucket, "a")
    stamp = {"trait": fx.COUNT_TRAIT, "images_dir": str(tmp_path), "raster_path": None,
             "operating_point": {"conf": {"value": 0.5, "validated_against": VALIDATED_HELD_OUT}}}
    write_bound_sidecar(bucket, stamp, dataset_root=tmp_path)

    real_check = op.check_operationalization
    calls = {"n": 0}

    def _flaky_check(*a, **kw):
        calls["n"] += 1
        result = real_check(*a, **kw)
        # The door's own first check (call 1) passes; every later call (the writer's own,
        # including its post-write re-check) reads as withdrawn since.
        return result if calls["n"] == 1 else replace(
            result, state=1, message="operationalization withdrawn mid-flow")

    monkeypatch.setattr(op, "check_operationalization", _flaky_check)

    with pytest.raises(ValueError, match="withdrawn mid-flow"):
        itools.tabulate_counts(predictions_dir=str(bucket), output_path=str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT)


def test_a_withdrawn_operationalization_mid_flow_is_count_free_in_the_live_regime(
    tmp_path, monkeypatch,
):
    """The live regime's own except block catches only DeliveryRefused, so the writer's bare
    meaning-door raise propagates past a bucket the shared bracket already published and linked,
    never composed into a counts-bearing refusal dict."""
    from dataclasses import replace

    import tcip_mcp.tools.inference_tools as itools
    from tcip_mcp import operationalization as op

    monkeypatch.setattr(itools, "_run_inference_verified",
                        lambda *a, **kw: _unvalidated_run_result())
    real_check = op.check_operationalization
    calls = {"n": 0}

    def _flaky_check(*a, **kw):
        calls["n"] += 1
        result = real_check(*a, **kw)
        return result if calls["n"] == 1 else replace(
            result, state=1, message="operationalization withdrawn mid-flow")

    monkeypatch.setattr(op, "check_operationalization", _flaky_check)

    bucket = tmp_path / "ds" / "predictions" / "baseline" / "2026-01-01"
    with pytest.raises(ValueError, match="withdrawn mid-flow"):
        itools.tabulate_counts(_dummy_checkpoint(tmp_path), str(tmp_path), str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT, predictions_dir=str(bucket))
    assert bucket.exists()  # the bracket already published before the bare raise escaped


def test_a_gate_refusal_is_counts_bearing_in_the_bucket_regime(tmp_path):
    bucket = tmp_path / "ds" / "predictions" / "baseline" / "2026-01-01"
    _write_real_prediction(bucket, "a")
    stamp = {"trait": fx.COUNT_TRAIT, "images_dir": str(tmp_path), "raster_path": None,
             "operating_point": {"conf": {"value": 0.5, "validated_against": VALIDATED_FALSE}}}
    write_bound_sidecar(bucket, stamp, dataset_root=tmp_path)

    import tcip_mcp.tools.inference_tools as itools

    r = itools.tabulate_counts(predictions_dir=str(bucket), output_path=str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT)
    assert "error" in r
    assert r["image_count"] == 1
    assert r["total_detections"] == 1


def test_a_gate_refusal_names_every_disclosure_field_in_the_live_regime(tmp_path, monkeypatch):
    """Live with predictions_dir, unvalidated conf, no acknowledgement: the bucket lands honestly
    stamped false (the publish bracket only gates tile geometry) and the CSV refuses; the refusal
    is counts-bearing and names every section 5.9 disclosure field, so the review-promotion
    workflow can proceed from what landed."""
    import tcip_mcp.tools.inference_tools as itools

    monkeypatch.setattr(itools, "_run_inference_verified",
                        lambda *a, **kw: _unvalidated_run_result())
    bucket = tmp_path / "ds" / "predictions" / "baseline" / "2026-01-01"
    out_csv = tmp_path / "o.csv"
    r = itools.tabulate_counts(_dummy_checkpoint(tmp_path), str(tmp_path), str(out_csv),
                               trait=fx.COUNT_TRAIT, predictions_dir=str(bucket))
    assert "error" in r
    assert r["image_count"] == 1
    assert r["total_detections"] == 1
    assert r["bucket_published"] is True
    assert r["predictions_dir"] == str(bucket)
    assert r["bucket_redirected"] is False
    assert r["lineage_linked"] is None
    assert r["csv_delivered"] is False
    assert bucket.exists()
    assert not out_csv.exists()


# ── both regimes agree on the CSV, and the bucket regime never imports torch ──

def test_live_and_bucket_regime_produce_the_same_csv_rows(tmp_path, monkeypatch):
    """One calibrated live call with predictions_dir produces CSV A and the bucket; the bucket
    regime over that bucket produces CSV B; every row and column except produced_at agree."""
    import tcip_mcp.tools.inference_tools as itools

    monkeypatch.setattr(itools, "_run_inference_verified",
                        lambda *a, **kw: _earned_run_result(tmp_path, tiled=False))
    bucket = tmp_path / "ds" / "predictions" / "baseline" / "2026-01-01"
    csv_a = tmp_path / "a.csv"
    live = itools.tabulate_counts(_dummy_checkpoint(tmp_path), str(tmp_path), str(csv_a),
                                  trait=fx.COUNT_TRAIT, calibration_labels_dir=str(tmp_path),
                                  predictions_dir=str(bucket))
    assert "error" not in live, live

    csv_b = tmp_path / "b.csv"
    bucket_result = itools.tabulate_counts(predictions_dir=str(bucket), output_path=str(csv_b),
                                           trait=fx.COUNT_TRAIT)
    assert "error" not in bucket_result, bucket_result

    rows_a = list(csv.DictReader(csv_a.open()))
    rows_b = list(csv.DictReader(csv_b.open()))
    assert len(rows_a) == len(rows_b) == 1
    # The delivered cell carries the source extension, not the bare stem the .json document lost.
    assert rows_a[0]["image"] == "a.png"
    assert "image_note" not in live and "image_note" not in bucket_result
    for key in rows_a[0]:
        if key == "produced_at":
            from datetime import datetime

            datetime.fromisoformat(rows_a[0][key])
            datetime.fromisoformat(rows_b[0][key])
            continue
        assert rows_a[0][key] == rows_b[0][key], key


def test_bucket_regime_falls_back_to_the_stem_for_a_bucket_with_no_filename_map(tmp_path):
    """A bucket published before the image_filenames map existed carries no such key in its stamp:
    the delivered image cell falls back to the bare document stem, and the response discloses the
    fallback through image_note rather than silently reading as a filename."""
    import tcip_mcp.tools.inference_tools as itools

    bucket = tmp_path / "ds" / "predictions" / "baseline" / "2026-01-01"
    _write_real_prediction(bucket, "a")
    stamp = {"trait": fx.COUNT_TRAIT, "images_dir": str(tmp_path), "raster_path": None,
             "operating_point": {"conf": {"value": 0.5, "validated_against": VALIDATED_FALSE}}}
    write_bound_sidecar(bucket, stamp, dataset_root=tmp_path)

    out_csv = tmp_path / "o.csv"
    r = itools.tabulate_counts(predictions_dir=str(bucket), output_path=str(out_csv),
                               trait=fx.COUNT_TRAIT, acknowledge_unvalidated=True)
    assert "error" not in r, r
    assert "predates the image filename map" in r["image_note"]
    rows = list(csv.DictReader(out_csv.open()))
    assert rows[0]["image"] == "a"


def test_bucket_regime_reads_a_real_published_bucket_with_no_torch_import(tmp_path):
    """The bucket regime is no-GPU, no-predictor-import, no-checkpoint-argument at all: a real
    bucket built through the platform's own sidecar/prediction writers is read to a delivered CSV
    in a subprocess with torch blocked from importing at all."""
    pytest.importorskip("torch")
    bucket = tmp_path / "ds" / "predictions" / "baseline" / "2026-01-01"
    _write_real_prediction(bucket, "a")
    stamp = {"trait": fx.COUNT_TRAIT, "images_dir": str(tmp_path), "raster_path": None,
             "operating_point": {"conf": {"value": 0.5, "validated_against": VALIDATED_FALSE}}}
    write_bound_sidecar(bucket, stamp, dataset_root=tmp_path)
    out_csv = tmp_path / "o.csv"

    script = f"""
import sys

class _BlockTorch:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch.") or name == "torchvision" or name.startswith("torchvision."):
            raise ImportError(f"torch blocked for this check: {{name}}")
        return None

sys.meta_path.insert(0, _BlockTorch())
from tcip_store.binding import bind_default
bind_default()
import tcip_mcp.tools.inference_tools as itools
r = itools.tabulate_counts(predictions_dir={str(bucket)!r}, output_path={str(out_csv)!r},
                           trait={fx.COUNT_TRAIT!r}, acknowledge_unvalidated=True)
assert "error" not in r, r
assert "torch" not in sys.modules, "the bucket regime pulled torch into sys.modules"
print("ok")
"""
    import os
    env = {**os.environ, "TCIP_PROJECT_ROOT": str(tmp_path)}
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            timeout=60, env=env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
    assert out_csv.exists()


# ── the provisional path re-delivers identically ────────────────────────────

def test_bucket_regime_re_delivers_the_provisional_floor_identically(tmp_path, monkeypatch):
    """An uncalibrated acknowledged live call publishes a false-stamped bucket and CSV; the bucket
    regime re-delivers identical floored rows off the same bucket."""
    import tcip_mcp.tools.inference_tools as itools

    monkeypatch.setattr(itools, "_run_inference_verified",
                        lambda *a, **kw: _unvalidated_run_result())
    bucket = tmp_path / "ds" / "predictions" / "baseline" / "2026-01-01"
    csv_a = tmp_path / "a.csv"
    live = itools.tabulate_counts(_dummy_checkpoint(tmp_path), str(tmp_path), str(csv_a),
                                  trait=fx.COUNT_TRAIT, predictions_dir=str(bucket),
                                  acknowledge_unvalidated=True)
    assert "error" not in live, live
    assert live["measurement_validated"] == VALIDATED_FALSE

    csv_b = tmp_path / "b.csv"
    reread = itools.tabulate_counts(predictions_dir=str(bucket), output_path=str(csv_b),
                                    trait=fx.COUNT_TRAIT, acknowledge_unvalidated=True)
    assert "error" not in reread, reread
    assert reread["measurement_validated"] == VALIDATED_FALSE
    rows_a = list(csv.DictReader(csv_a.open()))
    rows_b = list(csv.DictReader(csv_b.open()))
    assert rows_a[0]["detection_count"] == rows_b[0]["detection_count"]
    assert rows_a[0]["avg_confidence"] == rows_b[0]["avg_confidence"]


# ── promotion: an unvalidated bucket earns a validated re-delivery with no re-run ──

def test_bucket_regime_delivers_validated_after_the_stamp_is_promoted(tmp_path):
    """A bucket published unvalidated, its sidecar promoted through the real update_sidecar merge
    with an earned record (resolution.py's own promotion primitive, the one
    routes/validation.py's review-promotion route calls), then the bucket regime delivers
    validated with validation_record naming the earned record. The merge is layered over the
    producing run's own stamp, never a wholesale replace, so a field the promotion never touches
    (images_dir) survives it."""
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar, update_sidecar
    from tests._binding_fixtures import file_validation_record

    bucket = tmp_path / "ds" / "predictions" / "baseline" / "2026-01-01"
    _write_real_prediction(bucket, "a")
    unvalidated_stamp = {"validated": False, "trait": fx.COUNT_TRAIT, "images_dir": str(tmp_path),
                        "raster_path": None,
                        "operating_point": {"conf": {"value": 0.5,
                                                     "validated_against": VALIDATED_FALSE}}}
    write_bound_sidecar(bucket, unvalidated_stamp, dataset_root=tmp_path)

    import tcip_mcp.tools.inference_tools as itools

    before = itools.tabulate_counts(predictions_dir=str(bucket), output_path=str(tmp_path / "before.csv"),
                                    trait=fx.COUNT_TRAIT, acknowledge_unvalidated=True)
    assert "error" not in before, before
    assert before["measurement_validated"] == VALIDATED_FALSE

    earned_op_point = {"conf": {"value": 0.5, "validated_against": VALIDATED_HELD_OUT}}
    bound = file_validation_record(
        {"operating_point": earned_op_point}, dataset_root=tmp_path / "ds", pred_dirs=[bucket],
        trait=fx.COUNT_TRAIT, experiment_id="exp-promotion-earned")

    def _promote(stored: dict) -> dict:
        """The merge the review-promotion route's own update_sidecar call performs: the earned
        fields layered over whatever the producing run left, never a wholesale replace."""
        merged = dict(stored)
        merged.update({"validated": True, "operating_point": earned_op_point,
                      "validated_by": bound["validated_by"]})
        return merged

    assert update_sidecar(bucket, _promote) is True
    promoted = read_operating_point_sidecar(bucket)
    assert promoted["images_dir"] == str(tmp_path)  # preserved from the producing run, not restated

    after = itools.tabulate_counts(predictions_dir=str(bucket), output_path=str(tmp_path / "after.csv"),
                                   trait=fx.COUNT_TRAIT)
    assert "error" not in after, after
    assert after["measurement_validated"] == VALIDATED_HELD_OUT
    pointer = bound["validated_by"]
    rows = list(csv.DictReader((tmp_path / "after.csv").open()))
    assert rows[0]["validation_record"] == f"{pointer['experiment_id']}:{pointer['record_digest']}"
