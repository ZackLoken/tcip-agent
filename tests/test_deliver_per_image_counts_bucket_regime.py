"""deliver_per_image_counts' bucket regime: a persisted, reviewed prediction bucket in, no GPU re-run.

Covers the source-regime discrimination, the bucket's positive-claim checks (a readable stamp,
the per-image mandatory shape, the trait binding), the publish bracket the live regime shares with
run_inference, and the refusal-channel split between a meaning-door raise and a gate refusal.
"""

from __future__ import annotations

import csv
import subprocess
import sys

import pytest

from tcip_mcp.pipelines.resolution import VALIDATED_FALSE, VALIDATED_HELD_OUT
from tests import _operationalization_fixtures as fx
from tests._binding_fixtures import calibrated_run_fields, write_bound_sidecar, write_prediction
from tests._record_damage_fixtures import damage_record


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
                           subject=fx.COUNT_SUBJECT, attribute=None, id_map={fx.COUNT_SUBJECT: 0})


def _unvalidated_run_result(*, experiment_id=None, stem="a"):
    """A stand-in live-run result honestly stamped unvalidated: bypasses the earned-evidence path
    (``_draft_count_claim`` returns nothing to open) so a test about the publish bracket's own
    tile/lineage checks is not entangled with the calibration-evidence machinery."""
    return {"results": [{"image": f"{stem}.png", "width": 100, "height": 100,
                        "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9], "labels": [1],
                        "count": 1}],
           "image_count": 1, "total_detections": 1, "id_map": None,
           "operating_point": {"conf": {"value": 0.5}}, "validated": False,
           "conf_source": "default", "experiment_id": experiment_id,
           "checkpoint_sha256": "deadbeef"}


def _earned_run_result(tmp_path, *, trait=fx.COUNT_TRAIT, tiled=False, tile_size=None,
                       tile_size_source="default", stem="a"):
    """A stand-in live-run result that left behind real evidence, so a bucket published from it
    earns a genuine validation record (the same shape test_delivery_gate.py's own helper builds)."""
    return {
        "results": [{"image": f"{stem}.png", "width": 100, "height": 100,
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

    r = itools.deliver_per_image_counts(output_path=str(tmp_path / "o.csv"), trait=fx.COUNT_TRAIT)
    assert "error" in r
    assert "checkpoint_path" in r["error"] and "predictions_dir" in r["error"]


def test_images_dir_with_no_checkpoint_path_refuses(tmp_path):
    import tcip_mcp.tools.inference_tools as itools

    r = itools.deliver_per_image_counts(images_dir=str(tmp_path), output_path=str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT)
    assert "error" in r
    assert "checkpoint_path" in r["error"]


def test_no_output_path_refuses(tmp_path):
    import tcip_mcp.tools.inference_tools as itools

    r = itools.deliver_per_image_counts(predictions_dir=str(tmp_path / "preds"), trait=fx.COUNT_TRAIT)
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
    stamp = {"subject": fx.COUNT_SUBJECT, "attribute": None, "trait": fx.COUNT_TRAIT, "images_dir": str(tmp_path), "raster_path": None,
             "operating_point": {"conf": {"value": 0.5, "validated_against": VALIDATED_FALSE}}}
    write_bound_sidecar(bucket, stamp, dataset_root=tmp_path)

    if name in ("calibration_labels_dir", "calibration_images_dir", "split_manifest_dir"):
        value = str(tmp_path / value)
    r = itools.deliver_per_image_counts(predictions_dir=str(bucket), output_path=str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT, **{name: value})
    assert "error" in r
    assert name in r["error"]


def test_a_live_only_parameter_stated_at_its_own_default_is_silently_admitted(tmp_path):
    """The rail must admit valid work: tile_batch_size's non-None default (96) makes
    stated-at-default indistinguishable from stating nothing, so a bucket regime call naming it
    at that default is honestly admitted rather than refused for a statement it cannot detect."""
    import tcip_mcp.tools.inference_tools as itools

    dataset_root = tmp_path / "ds"
    bucket = dataset_root / "predictions" / "baseline" / "2026-01-01"
    _write_real_prediction(bucket, "a")
    stamp = {"subject": fx.COUNT_SUBJECT, "attribute": None, "validated": True, "trait": fx.COUNT_TRAIT, "images_dir": str(tmp_path),
             "raster_path": None,
             "operating_point": {"conf": {"value": 0.5, "validated_against": VALIDATED_HELD_OUT}}}
    write_bound_sidecar(bucket, stamp, dataset_root=dataset_root)

    r = itools.deliver_per_image_counts(predictions_dir=str(bucket), output_path=str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT, tile_batch_size=96)
    assert "error" not in r, r


# ── the bucket's positive stamp shape ───────────────────────────────────────

def test_bucket_regime_refuses_a_directory_with_no_stamp(tmp_path):
    """A directory of label JSON with no stamp (a GT annotations tree) is refused, never counted:
    the platform's own label writer produces the tree."""
    import tcip_mcp.tools.inference_tools as itools
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    labels_dir = tmp_path / "annotations" / "2026-01-01"
    json_io.write_annotations(
        str(labels_dir / "a.json"), [Annotation(subject=fx.COUNT_SUBJECT, geometry=BBox(0, 0, 5, 5))],
        100, 100)

    r = itools.deliver_per_image_counts(predictions_dir=str(labels_dir), output_path=str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT)
    assert "error" in r
    assert "operating_point.json" in r["error"]


def test_bucket_regime_refuses_a_mosaic_bucket(tmp_path):
    """A stamp recording raster_path (a whole-mosaic bucket) is refused on its own mandatory
    shape, naming deliver_orthomosaic_plant_counts: one mosaic total is not a per-image count."""
    import tcip_mcp.tools.inference_tools as itools

    bucket = tmp_path / "mosaic_preds"
    write_prediction(bucket, "tile_0")
    stamp = {"subject": fx.COUNT_SUBJECT, "attribute": None, "trait": fx.COUNT_TRAIT, "images_dir": None, "raster_path": "mosaic.tif",
             "operating_point": {"conf": {"value": 0.5, "validated_against": VALIDATED_FALSE}}}
    write_bound_sidecar(bucket, stamp, dataset_root=tmp_path)

    r = itools.deliver_per_image_counts(predictions_dir=str(bucket), output_path=str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT)
    assert "error" in r
    assert "deliver_orthomosaic_plant_counts" in r["error"]


def test_bucket_regime_refuses_a_stamp_naming_neither_images_dir_nor_raster_path(tmp_path):
    import tcip_mcp.tools.inference_tools as itools

    bucket = tmp_path / "bare_preds"
    write_prediction(bucket, "a")
    stamp = {"subject": fx.COUNT_SUBJECT, "attribute": None, "trait": fx.COUNT_TRAIT, "images_dir": None, "raster_path": None,
             "operating_point": {"conf": {"value": 0.5, "validated_against": VALIDATED_FALSE}}}
    write_bound_sidecar(bucket, stamp, dataset_root=tmp_path)

    r = itools.deliver_per_image_counts(predictions_dir=str(bucket), output_path=str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT)
    assert "error" in r
    assert "per-image" in r["error"]


def test_bucket_regime_refuses_a_stamp_naming_an_empty_string_images_dir(tmp_path):
    """An empty-string images_dir records nothing, the same as no images_dir at all: the per-image
    shape check is a truthy check, not merely a not-None one, so it cannot be satisfied by a key
    present with nothing behind it."""
    import tcip_mcp.tools.inference_tools as itools

    bucket = tmp_path / "empty_images_dir_preds"
    _write_real_prediction(bucket, "a")
    stamp = {"subject": fx.COUNT_SUBJECT, "attribute": None, "trait": fx.COUNT_TRAIT, "images_dir": "", "raster_path": None,
             "operating_point": {"conf": {"value": 0.5, "validated_against": VALIDATED_FALSE}}}
    write_bound_sidecar(bucket, stamp, dataset_root=tmp_path)

    r = itools.deliver_per_image_counts(predictions_dir=str(bucket), output_path=str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT)
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
    stamp = {"subject": fx.COUNT_SUBJECT, "attribute": None, "trait": other_trait, "images_dir": str(tmp_path), "raster_path": None,
             "operating_point": {"conf": {"value": 0.5, "validated_against": VALIDATED_FALSE}}}
    write_bound_sidecar(bucket, stamp, dataset_root=tmp_path)

    r = itools.deliver_per_image_counts(predictions_dir=str(bucket), output_path=str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT)
    assert "error" in r
    assert other_trait in r["error"] and fx.COUNT_TRAIT in r["error"]


def test_bucket_regime_admits_a_stamp_naming_no_trait_at_all(tmp_path):
    """The rail must admit valid work: a stamp that names no trait (trait=None) is not a
    contradiction, so a bucket-regime call over it reaches the counting/gate stage under the
    caller's stated trait rather than refusing on a mismatch it never made. A trait-less stamp can
    never itself earn a validated claim (the writer refuses one with no trait to check against),
    so what this proves is that the call gets past the trait check, not that it ships a CSV."""
    import tcip_mcp.tools.inference_tools as itools
    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE

    bucket = tmp_path / "no_trait_preds"
    _write_real_prediction(bucket, "a")
    stamp = {"subject": fx.COUNT_SUBJECT, "attribute": None, "trait": None, "images_dir": str(tmp_path), "raster_path": None,
             "operating_point": {"conf": {"value": 0.5, "validated_against": VALIDATED_FALSE}}}
    write_bound_sidecar(bucket, stamp, dataset_root=tmp_path)

    r = itools.deliver_per_image_counts(predictions_dir=str(bucket), output_path=str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT)
    assert "error" in r
    assert "image_count" in r  # past the trait check, into the gate refusal
    assert r["unvalidated_dimensions"] == "operating_point"


def test_bucket_regime_refuses_a_stamped_bucket_with_no_prediction_documents(tmp_path):
    """A stamped bucket holding zero prediction documents refuses naming the fact, before any CSV
    is written: an empty bucket is not a per-image count either, however honestly its stamp reads."""
    import tcip_mcp.tools.inference_tools as itools

    bucket = tmp_path / "empty_preds"
    stamp = {"subject": fx.COUNT_SUBJECT, "attribute": None, "trait": fx.COUNT_TRAIT, "images_dir": str(tmp_path), "raster_path": None,
             "operating_point": {"conf": {"value": 0.5, "validated_against": VALIDATED_HELD_OUT}}}
    write_bound_sidecar(bucket, stamp, dataset_root=tmp_path)

    out_csv = tmp_path / "o.csv"
    r = itools.deliver_per_image_counts(predictions_dir=str(bucket), output_path=str(out_csv),
                               trait=fx.COUNT_TRAIT)
    assert "error" in r
    assert "no prediction documents" in r["error"]
    assert not out_csv.exists()


def test_bucket_regime_measured_subject_check_is_driven_by_a_recorded_id_map(tmp_path):
    """A bucket whose sidecar records an id_map drives the measured-subject check both ways: a
    stated trait whose confirmed subject is absent from every recorded id_map refuses naming it,
    and one present in it clears the same check, reaching the counting/gate stage rather than
    refusing on the subject. The stamp names no trait, which a validated claim can never do, so
    the matching case is read off the gate refusal it reaches, not a delivered CSV."""
    from tcip_mcp import operationalization as op
    import tcip_mcp.tools.inference_tools as itools
    from tcip_mcp.operationalization import OperationalizationRefused
    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE

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
    stamp = {"subject": fx.COUNT_SUBJECT, "attribute": None, "trait": None, "images_dir": str(tmp_path), "raster_path": None,
             "id_map": {fx.COUNT_SUBJECT: 0},
             "operating_point": {"conf": {"value": 0.5, "validated_against": VALIDATED_FALSE}}}
    write_bound_sidecar(bucket, stamp, dataset_root=tmp_path)

    with pytest.raises(OperationalizationRefused, match="a subject no recorded id_map names"):
        itools.per_image_counts_from_bucket(str(bucket), str(tmp_path / "mismatch.csv"),
                               trait=other_trait)

    match = itools.deliver_per_image_counts(predictions_dir=str(bucket), output_path=str(tmp_path / "match.csv"),
                                   trait=fx.COUNT_TRAIT)
    assert "error" in match
    assert "image_count" in match  # past the subject check, into the gate refusal
    assert match["unvalidated_dimensions"] == "operating_point"


def test_bucket_regime_returns_an_error_dict_for_an_unknown_trait(tmp_path):
    """No spec is registered for this trait name at all: resolve_trait_and_record's own
    TraitUnknownError is caught and converted to the tool's ordinary {"error": ...} shape, the
    same contract every other bucket-regime refusal has, never an unhandled 500 or a raise out of
    the tool."""
    import tcip_mcp.tools.inference_tools as itools

    bucket = tmp_path / "ds" / "predictions" / "baseline" / "2026-01-01"
    _write_real_prediction(bucket, "a")
    stamp = {"subject": fx.COUNT_SUBJECT, "attribute": None, "trait": "no-such-trait", "images_dir": str(tmp_path), "raster_path": None,
             "operating_point": {"conf": {"value": 0.5, "validated_against": VALIDATED_HELD_OUT}}}
    write_bound_sidecar(bucket, stamp, dataset_root=tmp_path)

    r = itools.deliver_per_image_counts(
        predictions_dir=str(bucket), output_path=str(tmp_path / "o.csv"), trait="no-such-trait")
    assert "error" in r
    assert not (tmp_path / "o.csv").exists()


def test_bucket_regime_forwards_project_root_none_unchanged_to_its_own_precheck(
    tmp_path, monkeypatch,
):
    """The tool builds no project_root of its own (always None); the core's own meaning
    pre-check (run before the bucket is even touched) must receive that None unchanged, never a
    substituted resolved root."""
    from tcip_mcp import operationalization as op
    from tcip_mcp.pipelines.resolution import CountDeliveryRefused
    import tcip_mcp.tools.inference_tools as itools

    real_resolve = op.resolve_trait_and_record
    seen_roots = []

    def _spy(*a, **kw):
        seen_roots.append(kw.get("project_root"))
        return real_resolve(*a, **kw)

    monkeypatch.setattr(op, "resolve_trait_and_record", _spy)
    with pytest.raises(CountDeliveryRefused):
        itools.per_image_counts_from_bucket(
            str(tmp_path / "no-such-bucket"), str(tmp_path / "o.csv"), trait=fx.COUNT_TRAIT)
    assert seen_roots == [None]


# ── the publish bracket: shared with run_inference ────────────────────

def test_publish_bracket_refuses_a_fabricated_tile_with_the_bucket_left_absent(tmp_path, monkeypatch):
    """A tiled run whose tile scale has no real basis refuses before anything lands, gated exactly
    as run_inference gates it: the bucket is left absent, not published unvalidated."""
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
    r = itools.deliver_per_image_counts(_dummy_checkpoint(tmp_path), str(tmp_path),
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
    r = itools.deliver_per_image_counts(_dummy_checkpoint(tmp_path), str(tmp_path), str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT, predictions_dir=str(bucket))
    assert "error" in r
    assert eid in r["error"]
    assert not bucket.exists()


def test_publish_bracket_links_a_resolvable_experiments_bucket_into_its_lineage(tmp_path, monkeypatch):
    """The bracket links a resolvable run into its lineage as it publishes, before the CSV's own
    gate ever runs: an uncalibrated conf with no acknowledgement route refuses the CSV, but the
    refusal still discloses the bucket it published and the lineage it linked."""
    import tcip_store
    import tcip_mcp.tools.inference_tools as itools
    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE
    from tcip_mcp.experiments import create_experiment, lineage_key, update_status

    eid = "exp-tabulate-lineage-link"
    create_experiment(eid, {"note": "producing run"})
    update_status(eid, "running")

    monkeypatch.setattr(itools, "_run_inference_verified",
                        lambda *a, **kw: _unvalidated_run_result(experiment_id=eid))
    bucket = tmp_path / "ds" / "predictions" / "baseline" / "2026-01-01"
    r = itools.deliver_per_image_counts(_dummy_checkpoint(tmp_path), str(tmp_path), str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT, predictions_dir=str(bucket))
    assert "error" in r
    assert r["operating_point_validated"] == VALIDATED_FALSE
    assert r["bucket_published"] is True
    assert r["lineage_linked"] is True
    lineage = tcip_store.read(lineage_key(eid), default={})
    assert lineage.get("predictions") == str(bucket)


# ── refusal-channel separation ──────────────────────────────────────────────

def test_a_withdrawn_operationalization_mid_flow_is_count_free_in_the_bucket_regime(
    tmp_path, monkeypatch,
):
    """The writer's own meaning-door raise (a confirmation withdrawn since the door's own first
    check) is the same typed OperationalizationRefused the core's own first check raises, so
    deliver_per_image_counts converts it to the same count-free {"error": ...} dict either way,
    never the counts-bearing shape a caught DeliveryRefused gets."""
    from dataclasses import replace

    import tcip_mcp.tools.inference_tools as itools
    from tcip_mcp import operationalization as op

    bucket = tmp_path / "ds" / "predictions" / "baseline" / "2026-01-01"
    _write_real_prediction(bucket, "a")
    stamp = {"subject": fx.COUNT_SUBJECT, "attribute": None, "trait": fx.COUNT_TRAIT, "images_dir": str(tmp_path), "raster_path": None,
             "operating_point": {"conf": {"value": 0.5, "validated_against": VALIDATED_HELD_OUT}}}
    write_bound_sidecar(bucket, stamp, dataset_root=tmp_path)

    real_check = op.check_operationalization
    calls = {"n": 0}

    def _flaky_check(*a, **kw):
        calls["n"] += 1
        result = real_check(*a, **kw)
        # The door's own first check (call 1) passes; every later call (the writer's own) reads
        # as withdrawn since.
        return result if calls["n"] == 1 else replace(
            result, state=1, message="operationalization withdrawn mid-flow")

    monkeypatch.setattr(op, "check_operationalization", _flaky_check)

    result = itools.deliver_per_image_counts(
        predictions_dir=str(bucket), output_path=str(tmp_path / "o.csv"), trait=fx.COUNT_TRAIT)
    assert result == {"error": "operationalization withdrawn mid-flow"}
    assert not (tmp_path / "o.csv").exists()


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
        itools.deliver_per_image_counts(_dummy_checkpoint(tmp_path), str(tmp_path), str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT, predictions_dir=str(bucket))
    assert bucket.exists()  # the bracket already published before the bare raise escaped


def test_a_gate_refusal_is_counts_bearing_in_the_bucket_regime(tmp_path):
    bucket = tmp_path / "ds" / "predictions" / "baseline" / "2026-01-01"
    _write_real_prediction(bucket, "a")
    stamp = {"subject": fx.COUNT_SUBJECT, "attribute": None, "trait": fx.COUNT_TRAIT, "images_dir": str(tmp_path), "raster_path": None,
             "operating_point": {"conf": {"value": 0.5, "validated_against": VALIDATED_FALSE}}}
    write_bound_sidecar(bucket, stamp, dataset_root=tmp_path)

    import tcip_mcp.tools.inference_tools as itools

    r = itools.deliver_per_image_counts(predictions_dir=str(bucket), output_path=str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT)
    assert "error" in r
    assert r["image_count"] == 1
    assert r["total_detections"] == 1
    assert r["unvalidated_dimensions"] == "operating_point"


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
    r = itools.deliver_per_image_counts(_dummy_checkpoint(tmp_path), str(tmp_path), str(out_csv),
                               trait=fx.COUNT_TRAIT, predictions_dir=str(bucket))
    assert "error" in r
    assert r["image_count"] == 1
    assert r["total_detections"] == 1
    assert r["bucket_published"] is True
    assert r["predictions_dir"] == str(bucket)
    assert r["bucket_redirected"] is False
    assert r["lineage_linked"] is None
    assert r["csv_delivered"] is False
    assert r["unvalidated_dimensions"] == "operating_point"
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
    live = itools.deliver_per_image_counts(_dummy_checkpoint(tmp_path), str(tmp_path), str(csv_a),
                                  trait=fx.COUNT_TRAIT, calibration_labels_dir=str(tmp_path),
                                  predictions_dir=str(bucket))
    assert "error" not in live, live

    csv_b = tmp_path / "b.csv"
    bucket_result = itools.deliver_per_image_counts(predictions_dir=str(bucket), output_path=str(csv_b),
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

    dataset_root = tmp_path / "ds"
    bucket = dataset_root / "predictions" / "baseline" / "2026-01-01"
    _write_real_prediction(bucket, "a")
    stamp = {"subject": fx.COUNT_SUBJECT, "attribute": None, "validated": True, "trait": fx.COUNT_TRAIT, "images_dir": str(tmp_path),
             "raster_path": None,
             "operating_point": {"conf": {"value": 0.5, "validated_against": VALIDATED_HELD_OUT}}}
    write_bound_sidecar(bucket, stamp, dataset_root=dataset_root)

    out_csv = tmp_path / "o.csv"
    r = itools.deliver_per_image_counts(predictions_dir=str(bucket), output_path=str(out_csv),
                               trait=fx.COUNT_TRAIT)
    assert "error" not in r, r
    assert "carries no image filename map" in r["image_note"]
    rows = list(csv.DictReader(out_csv.open()))
    assert rows[0]["image"] == "a"


def test_bucket_regime_partial_map_delivers_filenames_for_mapped_rows_and_stems_for_the_rest(
    tmp_path,
):
    """A stamp's image filename map naming some but not all of the bucket's documents' stems
    delivers the mapped rows under their filename and the rest under the bare stem, disclosing
    the unmapped stems through image_note: the fallback branch is per-row, not all-or-nothing."""
    import tcip_mcp.tools.inference_tools as itools

    dataset_root = tmp_path / "ds"
    bucket = dataset_root / "predictions" / "baseline" / "2026-01-01"
    _write_real_prediction(bucket, "a")
    _write_real_prediction(bucket, "b")
    stamp = {"subject": fx.COUNT_SUBJECT, "attribute": None, "validated": True, "trait": fx.COUNT_TRAIT, "images_dir": str(tmp_path),
             "raster_path": None, "image_filenames": {"a": "a.png"},
             "operating_point": {"conf": {"value": 0.5, "validated_against": VALIDATED_HELD_OUT}}}
    write_bound_sidecar(bucket, stamp, dataset_root=dataset_root)

    out_csv = tmp_path / "o.csv"
    r = itools.deliver_per_image_counts(predictions_dir=str(bucket), output_path=str(out_csv),
                               trait=fx.COUNT_TRAIT)
    assert "error" not in r, r
    assert "['b']" in r["image_note"]
    rows = {row["image"]: row for row in csv.DictReader(out_csv.open())}
    assert "a.png" in rows
    assert "b" in rows


def test_bucket_regime_gate_refusal_on_a_mapless_bucket_carries_the_image_note(tmp_path):
    """A gate refusal (unvalidated conf, unacknowledged) is counts-bearing already; it must not
    drop the same fallback disclosure a successful delivery off the same bucket would carry."""
    import tcip_mcp.tools.inference_tools as itools

    bucket = tmp_path / "preds"
    _write_real_prediction(bucket, "a")
    stamp = {"subject": fx.COUNT_SUBJECT, "attribute": None, "trait": fx.COUNT_TRAIT, "images_dir": str(tmp_path), "raster_path": None,
             "operating_point": {"conf": {"value": 0.5, "validated_against": VALIDATED_FALSE}}}
    write_bound_sidecar(bucket, stamp, dataset_root=tmp_path)

    r = itools.deliver_per_image_counts(predictions_dir=str(bucket), output_path=str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT)
    assert "error" in r
    assert "carries no image filename map" in r["image_note"]


def test_bucket_regime_refuses_a_non_dict_image_filenames_in_the_stamp(tmp_path):
    """A stamp whose image_filenames is not a mapping (a corrupted or hand-edited stamp) refuses
    by name before _bucket_csv_rows ever calls .get on it, rather than crashing with
    AttributeError on a value that carries no .get method."""
    import tcip_mcp.tools.inference_tools as itools

    bucket = tmp_path / "preds"
    _write_real_prediction(bucket, "a")
    stamp = {"subject": fx.COUNT_SUBJECT, "attribute": None, "trait": fx.COUNT_TRAIT, "images_dir": str(tmp_path), "raster_path": None,
             "image_filenames": ["a.png"],
             "operating_point": {"conf": {"value": 0.5, "validated_against": VALIDATED_FALSE}}}
    write_bound_sidecar(bucket, stamp, dataset_root=tmp_path)

    r = itools.deliver_per_image_counts(predictions_dir=str(bucket), output_path=str(tmp_path / "o.csv"),
                               trait=fx.COUNT_TRAIT)
    assert "error" in r
    assert "image_filenames" in r["error"]


def test_live_regime_second_publish_into_a_document_holding_bucket_refuses(tmp_path, monkeypatch):
    """A verdict-free bucket already holding a document from an earlier publish refuses a second
    live publish outright, naming the document count and a suggested fresh bucket; the first
    publish's own document and stamp are unchanged afterwards (the digest and stamp equality
    prove those two artifacts alone, not that nothing else in the bucket moved). The refused
    call's own _run_inference_verified is a spy that fails the test if it is reached, proving the
    refusal runs before any pass; the suggested bucket, once published into, admits a real run,
    the live path's own producer-fed admitting case."""
    import tcip_mcp.tools.inference_tools as itools
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar
    from tcip_mcp.prediction_buckets import bucket_content_digest

    bucket = tmp_path / "ds" / "predictions" / "baseline" / "2026-01-01"
    monkeypatch.setattr(itools, "_run_inference_verified",
                        lambda *a, **kw: _earned_run_result(tmp_path, stem="a"))
    first = itools.deliver_per_image_counts(_dummy_checkpoint(tmp_path), str(tmp_path),
                                   str(tmp_path / "first.csv"), trait=fx.COUNT_TRAIT,
                                   calibration_labels_dir=str(tmp_path), predictions_dir=str(bucket))
    assert "error" not in first, first

    digest_before = bucket_content_digest(bucket)
    stamp_before = read_operating_point_sidecar(bucket)

    def _fail_if_reached(*a, **kw):
        raise AssertionError("_run_inference_verified must not run on a refused publish")

    monkeypatch.setattr(itools, "_run_inference_verified", _fail_if_reached)
    second = itools.deliver_per_image_counts(_dummy_checkpoint(tmp_path), str(tmp_path),
                                    str(tmp_path / "second.csv"), trait=fx.COUNT_TRAIT,
                                    calibration_labels_dir=str(tmp_path),
                                    predictions_dir=str(bucket))
    assert "error" in second
    assert second["document_stem_count"] == 1
    from pathlib import Path
    assert Path(second["suggested_bucket"]).parent.name == "baseline@r2"
    assert not (tmp_path / "second.csv").exists()
    assert bucket_content_digest(bucket) == digest_before
    assert read_operating_point_sidecar(bucket) == stamp_before

    # The admitting case: the suggested bucket is free of both a verdict and a document.
    monkeypatch.setattr(itools, "_run_inference_verified",
                        lambda *a, **kw: _earned_run_result(tmp_path, stem="b"))
    third = itools.deliver_per_image_counts(_dummy_checkpoint(tmp_path), str(tmp_path),
                                   str(tmp_path / "third.csv"), trait=fx.COUNT_TRAIT,
                                   calibration_labels_dir=str(tmp_path),
                                   predictions_dir=second["suggested_bucket"])
    assert "error" not in third, third
    assert third["predictions_dir"] == second["suggested_bucket"]


def test_live_regime_second_publish_refuses_on_documents_even_toward_an_unvalidated_run(
    tmp_path, monkeypatch,
):
    """The same document refusal fires even when the second run's own operating point would not
    have cleared the CSV's delivery gate: the document check runs first, before the pass or the
    gate, so the refusal is the document one, not the gate's. The refused call's own
    _run_inference_verified is a spy that fails the test if it is reached, proving the refusal
    runs before any pass; digest equality proves only that one bucket's own content, not a wider
    fact about the dataset."""
    import tcip_mcp.tools.inference_tools as itools
    from tcip_mcp.prediction_buckets import bucket_content_digest

    bucket = tmp_path / "ds" / "predictions" / "baseline" / "2026-01-01"
    monkeypatch.setattr(itools, "_run_inference_verified",
                        lambda *a, **kw: _earned_run_result(tmp_path, stem="a"))
    first = itools.deliver_per_image_counts(_dummy_checkpoint(tmp_path), str(tmp_path),
                                   str(tmp_path / "first.csv"), trait=fx.COUNT_TRAIT,
                                   calibration_labels_dir=str(tmp_path), predictions_dir=str(bucket))
    assert "error" not in first, first
    digest_before = bucket_content_digest(bucket)

    def _fail_if_reached(*a, **kw):
        raise AssertionError("_run_inference_verified must not run on a refused publish")

    monkeypatch.setattr(itools, "_run_inference_verified", _fail_if_reached)
    r = itools.deliver_per_image_counts(_dummy_checkpoint(tmp_path), str(tmp_path),
                               str(tmp_path / "second.csv"), trait=fx.COUNT_TRAIT,
                               predictions_dir=str(bucket))
    assert "error" in r
    assert r["document_stem_count"] == 1
    assert bucket_content_digest(bucket) == digest_before


def test_live_regime_second_publish_refuses_before_the_checkpoint_is_read(tmp_path, monkeypatch):
    """The document refusal runs at bucket resolution, ahead of the checkpoint load: a second
    call into a document-holding bucket never reaches load_registered_checkpoint. The spy is
    installed before the first, admitted call too, so its own count on that call is the positive
    control proving the spy is wired to fire, before the refused call is shown to skip it."""
    import tcip_mcp.model_registry as model_registry_mod
    import tcip_mcp.tools.inference_tools as itools

    bucket = tmp_path / "ds" / "predictions" / "baseline" / "2026-01-01"

    calls = {"n": 0}
    real = model_registry_mod.load_registered_checkpoint

    def _counting(path, *a, **kw):
        calls["n"] += 1
        return real(path, *a, **kw)

    monkeypatch.setattr(model_registry_mod, "load_registered_checkpoint", _counting)

    monkeypatch.setattr(itools, "_run_inference_verified",
                        lambda *a, **kw: _earned_run_result(tmp_path, stem="a"))
    first = itools.deliver_per_image_counts(_dummy_checkpoint(tmp_path), str(tmp_path),
                                   str(tmp_path / "first.csv"), trait=fx.COUNT_TRAIT,
                                   calibration_labels_dir=str(tmp_path), predictions_dir=str(bucket))
    assert "error" not in first, first
    assert calls["n"] == 1

    second = itools.deliver_per_image_counts(_dummy_checkpoint(tmp_path), str(tmp_path),
                                    str(tmp_path / "second.csv"), trait=fx.COUNT_TRAIT,
                                    predictions_dir=str(bucket))
    assert "error" in second
    assert second["document_stem_count"] == 1
    assert calls["n"] == 1


def test_bucket_regime_reads_a_real_published_bucket_with_no_torch_import(tmp_path):
    """The bucket regime is no-GPU, no-predictor-import, no-checkpoint-argument at all: a real
    bucket built through the platform's own sidecar/prediction writers is read to a delivered CSV
    in a subprocess with torch blocked from importing at all."""
    pytest.importorskip("torch")
    dataset_root = tmp_path / "ds"
    bucket = dataset_root / "predictions" / "baseline" / "2026-01-01"
    _write_real_prediction(bucket, "a")
    stamp = {"subject": fx.COUNT_SUBJECT, "attribute": None, "validated": True, "trait": fx.COUNT_TRAIT, "images_dir": str(tmp_path),
             "raster_path": None,
             "operating_point": {"conf": {"value": 0.5, "validated_against": VALIDATED_HELD_OUT}}}
    write_bound_sidecar(bucket, stamp, dataset_root=dataset_root)
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
r = itools.deliver_per_image_counts(predictions_dir={str(bucket)!r}, output_path={str(out_csv)!r},
                           trait={fx.COUNT_TRAIT!r})
assert "error" not in r, r
assert "torch" not in sys.modules, "the bucket regime pulled torch into sys.modules"
print("ok")
"""
    import os
    env = {**os.environ, "TCIP_STATE_ROOT": str(tmp_path)}
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            timeout=60, env=env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
    assert out_csv.exists()


# ── the provisional path re-delivers identically ────────────────────────────

def test_bucket_regime_re_delivers_the_provisional_floor_identically(tmp_path, monkeypatch):
    """An uncalibrated live call publishes a false-stamped bucket even though its own CSV takes no
    acknowledgement and refuses; the bucket regime reads the same published bucket back and floors
    it identically, both refusing on the same disclosed facts rather than one silently outranking
    the other."""
    import tcip_mcp.tools.inference_tools as itools

    monkeypatch.setattr(itools, "_run_inference_verified",
                        lambda *a, **kw: _unvalidated_run_result())
    bucket = tmp_path / "ds" / "predictions" / "baseline" / "2026-01-01"
    csv_a = tmp_path / "a.csv"
    live = itools.deliver_per_image_counts(_dummy_checkpoint(tmp_path), str(tmp_path), str(csv_a),
                                  trait=fx.COUNT_TRAIT, predictions_dir=str(bucket))
    assert "error" in live
    assert live["operating_point_validated"] == VALIDATED_FALSE
    assert live["bucket_published"] is True
    assert not csv_a.exists()

    csv_b = tmp_path / "b.csv"
    reread = itools.deliver_per_image_counts(predictions_dir=str(bucket), output_path=str(csv_b),
                                    trait=fx.COUNT_TRAIT)
    assert "error" in reread
    assert reread["operating_point_validated"] == VALIDATED_FALSE
    assert reread["unvalidated_dimensions"] == live["unvalidated_dimensions"]
    assert reread["image_count"] == live["image_count"]
    assert reread["total_detections"] == live["total_detections"]
    assert not csv_b.exists()


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
    unvalidated_stamp = {"subject": fx.COUNT_SUBJECT, "attribute": None, "validated": False, "trait": fx.COUNT_TRAIT, "images_dir": str(tmp_path),
                        "raster_path": None,
                        "operating_point": {"conf": {"value": 0.5,
                                                     "validated_against": VALIDATED_FALSE}}}
    write_bound_sidecar(bucket, unvalidated_stamp, dataset_root=tmp_path)

    import tcip_mcp.tools.inference_tools as itools

    before_csv = tmp_path / "before.csv"
    before = itools.deliver_per_image_counts(predictions_dir=str(bucket), output_path=str(before_csv),
                                    trait=fx.COUNT_TRAIT)
    assert "error" in before
    assert before["operating_point_validated"] == VALIDATED_FALSE
    assert not before_csv.exists()

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

    after = itools.deliver_per_image_counts(predictions_dir=str(bucket), output_path=str(tmp_path / "after.csv"),
                                   trait=fx.COUNT_TRAIT)
    assert "error" not in after, after
    assert after["operating_point_validated"] == VALIDATED_HELD_OUT
    pointer = bound["validated_by"]
    rows = list(csv.DictReader((tmp_path / "after.csv").open()))
    assert rows[0]["validation_record"] == f"{pointer['experiment_id']}:{pointer['record_digest']}"


def _write_raw_stamp(bucket, stamp: dict) -> None:
    """A bucket's own ``operating_point.json``, written straight through the store, bypassing
    ``write_sidecar``'s rail: the only way to mint a stamp that decodes with no usable
    ``(subject, attribute)`` pair, the shape a live producer can no longer write."""
    from pathlib import Path

    import tcip_store as ts
    from tcip_mcp.pipelines.resolution import sidecar_key

    Path(bucket).mkdir(parents=True, exist_ok=True)
    ts.replace(sidecar_key(bucket, "operating_point"), stamp, expect=ts.Version.ABSENT)


def _damage_stamp_bytes(bucket) -> None:
    """Corrupt ``bucket``'s already-written ``operating_point`` stamp bytes in place, so a strict
    read raises a real ``StoreError``."""
    from tcip_mcp.pipelines.resolution import sidecar_key

    damage_record(sidecar_key(bucket, "operating_point"), b"{not json")


def _real_stamp_scope_unstated(tmp_path, name: str):
    """A real ``StampScopeUnstated``, earned from an actual ``bucket_scope`` call over a bucket
    whose stamp decodes with no pair: a raw sidecar write is the only producer of that shape."""
    from tcip_mcp.pipelines.resolution import StampScopeUnstated, bucket_scope

    scratch = tmp_path / name
    _write_raw_stamp(scratch, {"checkpoint_sha256": "f" * 64})
    with pytest.raises(StampScopeUnstated) as excinfo:
        bucket_scope(scratch)
    return excinfo.value


def _real_store_error(tmp_path, name: str):
    """A real ``StoreError``, earned from an actual ``bucket_scope`` call over a bucket whose
    stamp will not decode at all."""
    from tcip_store import StoreError
    from tcip_mcp.pipelines.resolution import bucket_scope

    scratch = tmp_path / name
    _write_raw_stamp(scratch, {"checkpoint_sha256": "f" * 64, "subject": "s", "attribute": None})
    _damage_stamp_bytes(scratch)
    with pytest.raises(StoreError) as excinfo:
        bucket_scope(scratch)
    return excinfo.value


def test_the_live_regimes_export_detection_csv_stamp_scope_unstated_becomes_the_tools_own_error(
    tmp_path, monkeypatch,
):
    """This conversion is defence in depth behind an identical earlier check: the fresh stamp
    ``_publish_bucket_bracket`` writes at this live site always carries the pair
    (``operating_point_stamp`` requires it with no default), so a no-pair stamp never survives a
    live publish and this call site's own conversion has no naturally reachable shape to test
    through. A real ``StampScopeUnstated``, earned from an actual ``bucket_scope`` call rather
    than a test-written message, is raised from the monkeypatched ``export_detection_csv``,
    pinning the type conversion this call site makes (the exception becomes a bare
    ``{"error": str(exc)}``) and that the seam's own remedy text, naming
    ``tcip repair-classified-predictions``, survives into it unchanged: this door adds no text
    of its own."""
    import tcip_mcp.tools.inference_tools as itools

    monkeypatch.setattr(itools, "_run_inference_verified",
                        lambda *a, **kw: _unvalidated_run_result())
    real_exc = _real_stamp_scope_unstated(tmp_path, "no_pair_stamp")

    def _raise(*a, **kw):
        raise real_exc

    monkeypatch.setattr(itools, "export_detection_csv", _raise)

    bucket = tmp_path / "ds" / "predictions" / "baseline" / "2026-01-01"
    r = itools.deliver_per_image_counts(_dummy_checkpoint(tmp_path), str(tmp_path),
                               str(tmp_path / "o.csv"), trait=fx.COUNT_TRAIT,
                               predictions_dir=str(bucket))

    assert r["error"] == str(real_exc)


def test_the_live_regimes_export_detection_csv_store_error_becomes_the_tools_own_error(
    tmp_path, monkeypatch,
):
    """This call site's other conversion arm, defence in depth for the same reason as the
    ``StampScopeUnstated`` arm above: the fresh stamp ``_publish_bucket_bracket`` writes at this
    live site is always readable, so an undecodable stamp never survives a live publish. A real
    ``StoreError``, earned from an actual ``bucket_scope`` call over an undecodable stamp, pins the
    same type conversion (the exception becomes ``{"error": str(exc)}``, verbatim, with no text of
    this door's own)."""
    import tcip_mcp.tools.inference_tools as itools

    monkeypatch.setattr(itools, "_run_inference_verified",
                        lambda *a, **kw: _unvalidated_run_result())
    real_exc = _real_store_error(tmp_path, "undecodable_stamp")

    def _raise(*a, **kw):
        raise real_exc

    monkeypatch.setattr(itools, "export_detection_csv", _raise)

    bucket = tmp_path / "ds" / "predictions" / "baseline" / "2026-01-01"
    r = itools.deliver_per_image_counts(_dummy_checkpoint(tmp_path), str(tmp_path),
                               str(tmp_path / "o.csv"), trait=fx.COUNT_TRAIT,
                               predictions_dir=str(bucket))

    assert r["error"] == str(real_exc)


def test_per_image_counts_from_bucket_converts_export_detection_csvs_stamp_scope_unstated(
    tmp_path, monkeypatch,
):
    """This conversion is defence in depth behind an identical earlier check:
    ``per_image_counts_from_bucket``'s own ``bucket_scope`` call at the bucket site already
    refuses a no-pair stamp before this call site is ever reached, so the bucket under test here
    carries a real, fully scoped stamp and a real ``StampScopeUnstated``, earned from a separate
    scratch bucket's actual ``bucket_scope`` call, is raised from the monkeypatched
    ``export_detection_csv`` to pin the type conversion at this call site specifically (the
    exception becomes ``CountDeliveryRefused(str(exc))``, the seam's own remedy text surviving
    into it unchanged, no text of this door's own added)."""
    import tcip_mcp.tools.inference_tools as itools
    from tcip_mcp.pipelines.resolution import CountDeliveryRefused

    dataset_root = tmp_path / "ds"
    bucket = dataset_root / "predictions" / "baseline" / "2026-01-01"
    _write_real_prediction(bucket, "a")
    stamp = {"subject": fx.COUNT_SUBJECT, "attribute": None, "validated": True,
             "trait": fx.COUNT_TRAIT, "images_dir": str(tmp_path), "raster_path": None,
             "operating_point": {"conf": {"value": 0.5, "validated_against": VALIDATED_HELD_OUT}}}
    write_bound_sidecar(bucket, stamp, dataset_root=dataset_root)
    real_exc = _real_stamp_scope_unstated(tmp_path, "no_pair_stamp")

    def _raise(*a, **kw):
        raise real_exc

    monkeypatch.setattr(itools, "export_detection_csv", _raise)

    with pytest.raises(CountDeliveryRefused) as excinfo:
        itools.per_image_counts_from_bucket(
            str(bucket), str(tmp_path / "o.csv"), trait=fx.COUNT_TRAIT)

    assert str(excinfo.value) == str(real_exc)


def test_per_image_counts_from_bucket_converts_export_detection_csvs_store_error(
    tmp_path, monkeypatch,
):
    """This call site's other conversion arm, defence in depth for the same reason as the
    ``StampScopeUnstated`` arm above: ``per_image_counts_from_bucket``'s own ``bucket_scope`` call
    at the bucket site already refuses an undecodable stamp before this call site is reached. A
    real ``StoreError``, earned from an actual ``bucket_scope`` call over an undecodable stamp on
    a separate scratch bucket, pins the same type conversion (``CountDeliveryRefused(str(exc))``,
    verbatim, no text of this door's own)."""
    import tcip_mcp.tools.inference_tools as itools
    from tcip_mcp.pipelines.resolution import CountDeliveryRefused

    dataset_root = tmp_path / "ds"
    bucket = dataset_root / "predictions" / "baseline" / "2026-01-01"
    _write_real_prediction(bucket, "a")
    stamp = {"subject": fx.COUNT_SUBJECT, "attribute": None, "validated": True,
             "trait": fx.COUNT_TRAIT, "images_dir": str(tmp_path), "raster_path": None,
             "operating_point": {"conf": {"value": 0.5, "validated_against": VALIDATED_HELD_OUT}}}
    write_bound_sidecar(bucket, stamp, dataset_root=dataset_root)
    real_exc = _real_store_error(tmp_path, "undecodable_stamp")

    def _raise(*a, **kw):
        raise real_exc

    monkeypatch.setattr(itools, "export_detection_csv", _raise)

    with pytest.raises(CountDeliveryRefused) as excinfo:
        itools.per_image_counts_from_bucket(
            str(bucket), str(tmp_path / "o.csv"), trait=fx.COUNT_TRAIT)

    assert str(excinfo.value) == str(real_exc)
