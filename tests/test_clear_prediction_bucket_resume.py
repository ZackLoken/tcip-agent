"""clear_prediction_bucket: interrupted clears finished by a second call naming cleared_bucket,
each crash point the design's Tests section names, plus the D5 refusals that only arise from an
interrupted or racing clear (an unfinished clear on record, a document that arrived mid-move, a
secondary stamp written fresh into a half-cleared source) and review_state_landed_during_clear."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tests._clear_prediction_bucket_faults import inject_store_fault, key_in_store, raise_on_nth_call
from tests._clear_prediction_bucket_fixtures import build_published_bucket, record_review_verdict


FIXED_STAMP = "20260906T120000Z"


def _fix_stamp(monkeypatch) -> None:
    import tcip_mcp.dataset_layout as dataset_layout_mod

    monkeypatch.setattr(dataset_layout_mod, "current_cleared_stamp", lambda: FIXED_STAMP)


def _expected_destination(built: dict) -> Path:
    from tcip_mcp.dataset_layout import cleared_prediction_dir

    return cleared_prediction_dir(built["dataset_root"], "m", "2026-03-02", FIXED_STAMP)


def test_fault_at_first_document_write_after_stamps_moved_is_finished_by_resume(tmp_path, monkeypatch):
    from tcip_mcp.prediction_buckets import bucket_stems
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expFaultFirstDoc")
    _fix_stamp(monkeypatch)
    destination = _expected_destination(built)

    fault = inject_store_fault(monkeypatch, method_name="put_blob")
    with pytest.raises(RuntimeError):
        clear_prediction_bucket(str(built["bucket"]), "should crash before the first document")
    assert fault.fired

    result = clear_prediction_bucket(
        str(built["bucket"]), "resume after the first-document crash",
        cleared_bucket=str(destination))
    assert "error" not in result, result
    assert result["resumed"] is True
    assert result["source_republished"] is False
    assert bucket_stems(built["bucket"]) == set()
    assert bucket_stems(destination) == {"img"}


def test_fault_between_artifact_record_and_first_stamp_write_is_finished_by_resume(tmp_path, monkeypatch):
    from tcip_mcp.prediction_buckets import bucket_stems
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expFaultNoDestination")
    _fix_stamp(monkeypatch)
    destination = _expected_destination(built)

    fault = inject_store_fault(
        monkeypatch, method_name="replace", predicate=key_in_store("operating_point_sidecar"))
    with pytest.raises(RuntimeError):
        clear_prediction_bucket(str(built["bucket"]), "should crash with no destination content")
    assert fault.fired

    result = clear_prediction_bucket(
        str(built["bucket"]), "resume after the no-destination crash",
        cleared_bucket=str(destination))
    assert "error" not in result, result
    assert result["resumed"] is True
    assert bucket_stems(built["bucket"]) == set()
    assert bucket_stems(destination) == {"img"}


def test_fault_between_op_stamp_copy_and_delete_leaves_it_at_both_finished_by_resume(tmp_path, monkeypatch):
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar
    from tcip_mcp.prediction_buckets import bucket_stems
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expFaultStampBoth")
    _fix_stamp(monkeypatch)
    destination = _expected_destination(built)

    fault = inject_store_fault(
        monkeypatch, method_name="delete", predicate=key_in_store("operating_point_sidecar"))
    with pytest.raises(RuntimeError):
        clear_prediction_bucket(str(built["bucket"]), "should crash with the stamp at both")
    assert fault.fired
    # The clock-free mark: the stamp landed at the destination while the source still holds it.
    assert read_operating_point_sidecar(destination) is not None
    assert read_operating_point_sidecar(built["bucket"]) is not None

    result = clear_prediction_bucket(
        str(built["bucket"]), "resume after the stamp-at-both crash",
        cleared_bucket=str(destination))
    assert "error" not in result, result
    assert result["resumed"] is True
    assert read_operating_point_sidecar(built["bucket"]) is None
    assert bucket_stems(built["bucket"]) == set()
    assert bucket_stems(destination) == {"img"}


def test_fault_between_document_write_and_source_delete_finished_by_resume(tmp_path, monkeypatch):
    from tcip_annotation.json_io import ANNOTATION_RECORDS_STORE
    from tcip_mcp.prediction_buckets import bucket_stems
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expFaultDocBoth")
    _fix_stamp(monkeypatch)
    destination = _expected_destination(built)

    fault = inject_store_fault(
        monkeypatch, method_name="delete", predicate=key_in_store(ANNOTATION_RECORDS_STORE))
    with pytest.raises(RuntimeError):
        clear_prediction_bucket(str(built["bucket"]), "should crash with the document at both")
    assert fault.fired
    assert bucket_stems(built["bucket"]) == {"img"}
    assert bucket_stems(destination) == {"img"}

    result = clear_prediction_bucket(
        str(built["bucket"]), "resume after the document-at-both crash",
        cleared_bucket=str(destination))
    assert "error" not in result, result
    assert result["resumed"] is True
    assert bucket_stems(built["bucket"]) == set()
    assert bucket_stems(destination) == {"img"}


def test_fault_after_last_source_delete_reports_republication_on_resume(tmp_path, monkeypatch):
    """A crash after the clear's own last write, before the body returns: a fresh publish landed
    into the emptied source before the resume runs. The resume moves nothing more and reports
    source_republished, the artifact already recorded."""
    from tcip_mcp.prediction_buckets import bucket_stems
    import tcip_mcp.prediction_buckets as prediction_buckets_mod
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket, run_inference

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expFaultLastStep")
    _fix_stamp(monkeypatch)
    destination = _expected_destination(built)

    # review_state_count runs twice per call (the D5 gate, then the final count); raising on the
    # second crashes after every write has landed, before the body returns.
    raise_on_nth_call(monkeypatch, prediction_buckets_mod, "review_state_count", 2)
    with pytest.raises(RuntimeError):
        clear_prediction_bucket(str(built["bucket"]), "should crash after the last write")
    assert bucket_stems(built["bucket"]) == set()
    assert bucket_stems(destination) == {"img"}

    # A fresh publish lands into the now-empty, still-terminal, same-pointer source.
    republish = run_inference(
        str(built["checkpoint"]), str(built["images_dir"]), output_dir=str(built["bucket"]),
        tile=False, experiment_id="expFaultLastStep")
    assert "error" not in republish, republish
    assert bucket_stems(built["bucket"]) == {"img"}

    result = clear_prediction_bucket(
        str(built["bucket"]), "resume after the last-step crash and a republication",
        cleared_bucket=str(destination))
    assert "error" not in result, result
    assert result["resumed"] is True
    assert result["source_republished"] is True
    assert result["documents_moved_this_call"] == 0
    assert result["cleared_artifact_recorded"] is True
    # The re-publication is left exactly as it landed.
    assert bucket_stems(built["bucket"]) == {"img"}
    assert bucket_stems(destination) == {"img"}


def test_unfinished_clear_no_destination_content_refuses_a_keyword_less_call(tmp_path, monkeypatch):
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expUnfinishedNoContent")
    _fix_stamp(monkeypatch)
    destination = _expected_destination(built)

    fault = inject_store_fault(monkeypatch, method_name="put_blob")
    with pytest.raises(RuntimeError):
        clear_prediction_bucket(str(built["bucket"]), "should crash before the first document")
    assert fault.fired

    result = clear_prediction_bucket(str(built["bucket"]), "keyword-less call over the wreckage")
    assert "error" in result
    assert "unfinished" in result["error"]
    assert repr(str(destination)) in result["error"]


def test_unfinished_clear_stamp_at_both_refuses_a_keyword_less_call(tmp_path, monkeypatch):
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expUnfinishedStampBoth")
    _fix_stamp(monkeypatch)
    destination = _expected_destination(built)

    fault = inject_store_fault(
        monkeypatch, method_name="delete", predicate=key_in_store("operating_point_sidecar"))
    with pytest.raises(RuntimeError):
        clear_prediction_bucket(str(built["bucket"]), "should crash with the stamp at both")
    assert fault.fired

    result = clear_prediction_bucket(str(built["bucket"]), "keyword-less call over the wreckage")
    assert "error" in result
    assert "unfinished" in result["error"]
    assert repr(str(destination)) in result["error"]


def test_a_document_arrived_during_the_move_refuses_naming_the_stem(tmp_path, monkeypatch):
    """The staging door admits a stampless bucket: a document staged into the source between the
    preflight and the loop's own re-enumeration is left where it is, and the call refuses naming
    the stem and the resume remedy."""
    from tcip_annotation import Annotation, BBox
    from tcip_mcp.prediction_buckets import stage_prediction_shapes
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket
    import tcip_mcp.tools.inference_tools as inference_tools_mod

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expArrivedDuringMove")
    _fix_stamp(monkeypatch)

    real_reconcile_document = inference_tools_mod._reconcile_document
    staged_once = {"done": False}

    def stage_arriving_document(*args, **kwargs):
        if not staged_once["done"]:
            staged_once["done"] = True
            stage_prediction_shapes(
                str(built["dataset_root"]), "m", "2026-03-02", "arrived",
                annotations=[Annotation(subject="bud", geometry=BBox(1, 1, 5, 5))],
                img_w=32, img_h=32,
            )
        return real_reconcile_document(*args, **kwargs)

    monkeypatch.setattr(inference_tools_mod, "_reconcile_document", stage_arriving_document)

    result = clear_prediction_bucket(str(built["bucket"]), "a document arrives mid-move")
    assert "error" in result
    assert "arrived" in result["error"]


def test_a_secondary_stamp_written_fresh_into_the_half_cleared_source_refuses_on_resume(
        tmp_path, monkeypatch):
    """One of the three producers that write a fresh stamp into any directory landed on the
    half-cleared source between the two calls: the resume refuses naming them, no door removes it.
    The original bucket carries its own resolve_scale stamp (so the interrupted call moves it to
    the destination before the crash); a fresh value written into the now-stampless source after
    is what the resume must catch, present at both and different."""
    from tcip_mcp.pipelines.resolution import write_sidecar
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expFreshSecondary")
    write_sidecar(built["bucket"], {"conf": {"value": 0.1}}, document="resolve_scale")
    _fix_stamp(monkeypatch)
    destination = _expected_destination(built)

    fault = inject_store_fault(monkeypatch, method_name="put_blob")
    with pytest.raises(RuntimeError):
        clear_prediction_bucket(str(built["bucket"]), "should crash before the first document")
    assert fault.fired

    write_sidecar(built["bucket"], {"conf": {"value": 0.4}}, document="resolve_scale")

    result = clear_prediction_bucket(
        str(built["bucket"]), "resume after a fresh secondary stamp landed",
        cleared_bucket=str(destination))
    assert "error" in result
    assert "resolve_scale.json differs" in result["error"]


def test_a_same_stem_document_staged_over_a_copied_one_refuses_on_resume(tmp_path, monkeypatch):
    """A same-stem document staged over one the clear had already copied but not yet deleted at
    the source: the resume refuses naming the staging door, no door removes it."""
    from tcip_annotation import Annotation, BBox
    from tcip_annotation.json_io import ANNOTATION_RECORDS_STORE, write_annotations
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expDocStagedOver")
    _fix_stamp(monkeypatch)
    destination = _expected_destination(built)

    fault = inject_store_fault(
        monkeypatch, method_name="delete", predicate=key_in_store(ANNOTATION_RECORDS_STORE))
    with pytest.raises(RuntimeError):
        clear_prediction_bucket(str(built["bucket"]), "should crash with the document at both")
    assert fault.fired

    write_annotations(
        built["bucket"] / "img.json",
        [Annotation(subject="bud", geometry=BBox(2, 2, 6, 6))], 100, 100,
    )

    result = clear_prediction_bucket(
        str(built["bucket"]), "resume after a same-stem document was staged over",
        cleared_bucket=str(destination))
    assert "error" in result
    assert "img.json differs" in result["error"]


def test_review_state_landed_during_clear_is_reported(tmp_path, monkeypatch):
    """Review state recorded on the source between the preflight and the last document's delete
    is not caught by the D5 refusal (which read the state before this call began); it is counted
    once more after the last delete and reported, never silently dropped."""
    from tcip_mcp.prediction_buckets import review_state_dir_of
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket
    import tcip_mcp.tools.inference_tools as inference_tools_mod

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expReviewLandedDuring")
    review_state_dir = review_state_dir_of(built["dataset_root"])

    real_reconcile_document = inference_tools_mod._reconcile_document
    landed_once = {"done": False}

    def land_review_mid_move(*args, **kwargs):
        if not landed_once["done"]:
            landed_once["done"] = True
            record_review_verdict(built["bucket"], review_state_dir, "img.png")
        return real_reconcile_document(*args, **kwargs)

    monkeypatch.setattr(inference_tools_mod, "_reconcile_document", land_review_mid_move)

    result = clear_prediction_bucket(str(built["bucket"]), "review lands mid-move")
    assert "error" not in result, result
    assert result["review_state_landed_during_clear"] == 1
