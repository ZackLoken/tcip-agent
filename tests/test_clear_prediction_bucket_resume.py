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


def test_a_merge_landed_on_the_source_in_the_stamp_at_both_state_is_carried_to_the_destination(
        tmp_path, monkeypatch):
    """While the source still reads as published (the stamp-at-both state), a merge writer can
    still land on it: the count calibrator's own merge (calibrate_count_operating_point, through
    resolution.update_sidecar) folds an earned conf into the stored stamp without replacing it
    wholesale, the same shape produced here directly through update_sidecar with an updater that
    keeps the stored (subject, attribute) pair and adds a calibration-shaped key the writer rail
    admits, rather than running the full calibration pass. The resume carries the merged value to
    the destination, replacing its stale copy."""
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar, update_sidecar
    from tcip_mcp.prediction_buckets import bucket_stems
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expMergeCarried")
    _fix_stamp(monkeypatch)
    destination = _expected_destination(built)

    fault = inject_store_fault(
        monkeypatch, method_name="delete", predicate=key_in_store("operating_point_sidecar"))
    with pytest.raises(RuntimeError):
        clear_prediction_bucket(str(built["bucket"]), "should crash with the stamp at both")
    assert fault.fired
    stale_at_destination = read_operating_point_sidecar(destination)
    assert stale_at_destination is not None

    def merge_a_calibration_shaped_key(stored: dict) -> dict:
        conf = dict(stored.get("operating_point", {}).get("conf", {}))
        conf["validated_against"] = "held_out_annotations"
        return {**stored, "operating_point": {**stored.get("operating_point", {}), "conf": conf}}

    assert update_sidecar(built["bucket"], merge_a_calibration_shaped_key) is True
    merged_at_source = read_operating_point_sidecar(built["bucket"])
    assert merged_at_source != stale_at_destination

    result = clear_prediction_bucket(
        str(built["bucket"]), "resume after a count calibration merged into the still-published source",
        cleared_bucket=str(destination))
    assert "error" not in result, result
    assert bucket_stems(built["bucket"]) == set()
    assert read_operating_point_sidecar(built["bucket"]) is None
    assert read_operating_point_sidecar(destination) == merged_at_source


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


def test_a_version_conflict_during_reconcile_refuses_naming_the_key_and_what_moved(tmp_path, monkeypatch):
    """A conflicting write lands under the door mid-move (monkeypatched onto the door's own store
    attribute, never the seam itself): the VersionConflict a reconcile step's own conditional write
    raises is caught and turned into a refusal naming the key that changed and what this call had
    already moved, which stands, rather than propagating out of the audited envelope."""
    from tcip_annotation.json_io import ANNOTATION_RECORDS_STORE, annotation_record_key
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar
    from tcip_mcp.prediction_buckets import bucket_stems
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket
    from tcip_store import Version, VersionConflict

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expVersionConflict")
    _fix_stamp(monkeypatch)
    destination = _expected_destination(built)

    conflicting_key = annotation_record_key(destination, "img")
    conflict = VersionConflict(conflicting_key, Version.ABSENT, Version("stale-token"))
    fault = inject_store_fault(
        monkeypatch, method_name="put_blob",
        predicate=key_in_store(ANNOTATION_RECORDS_STORE), exc=conflict)

    result = clear_prediction_bucket(str(built["bucket"]), "should refuse: version conflict")
    assert "error" in result
    assert fault.fired
    assert "img" in result["error"]
    assert "already moved this call stand" in result["error"]
    # the stamps moved before the conflicting document write still stand at the destination.
    assert read_operating_point_sidecar(destination) is not None
    assert read_operating_point_sidecar(built["bucket"]) is None
    assert bucket_stems(destination) == set()


def test_an_undecodable_stamp_at_the_destination_refuses_by_name_on_resume(tmp_path, monkeypatch):
    """A stamp already moved to the destination before a crash, then corrupted there: the resume
    preflights the destination's five stamps the way it preflights the source's, refusing by name
    instead of raising a bare StoreError out of the reconcilers."""
    from tcip_mcp.pipelines.resolution import sidecar_key, write_sidecar
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket
    from tests._record_damage_fixtures import damage_record

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expDestUndecodable")
    write_sidecar(built["bucket"], {"conf": {"value": 0.1}}, document="resolve_scale")
    _fix_stamp(monkeypatch)
    destination = _expected_destination(built)

    fault = inject_store_fault(monkeypatch, method_name="put_blob")
    with pytest.raises(RuntimeError):
        clear_prediction_bucket(str(built["bucket"]), "should crash after the stamps moved")
    assert fault.fired

    key = sidecar_key(destination, "resolve_scale")
    damage_record(key, b"{not json")

    result = clear_prediction_bucket(
        str(built["bucket"]), "resume over a corrupted destination stamp",
        cleared_bucket=str(destination))
    assert "error" in result
    assert "resolve_scale.json will not decode" in result["error"]


def test_unfinished_clear_no_destination_content_refuses_a_keyword_less_call(tmp_path, monkeypatch):
    """A crash before any stamp write leaves the destination holding no document: the artifact
    already names it, so a keyword-less call refuses on record as unfinished, naming it."""
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expUnfinishedNoContent")
    _fix_stamp(monkeypatch)
    destination = _expected_destination(built)

    fault = inject_store_fault(
        monkeypatch, method_name="replace", predicate=key_in_store("operating_point_sidecar"))
    with pytest.raises(RuntimeError):
        clear_prediction_bucket(str(built["bucket"]), "should crash before any stamp write")
    assert fault.fired

    result = clear_prediction_bucket(str(built["bucket"]), "keyword-less call over the wreckage")
    assert "error" in result
    assert "is on record and unfinished" in result["error"]
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
    assert "is on record and unfinished" in result["error"]
    assert repr(str(destination)) in result["error"]


def test_a_keyword_less_call_in_the_stamps_moved_state_refuses_naming_the_newest_archive(
        tmp_path, monkeypatch):
    """A crash after every stamp has moved but before the first document leaves the source with no
    operating_point stamp at all and its documents still in place: the door reads no experiment
    from a stampless source, so the keyword-less refusal is the archive-walking one
    (_find_cleared_candidate_with_no_source_stamp), not the artifact-record one, and it names the
    same destination as the remedy."""
    from tcip_mcp.prediction_buckets import bucket_stems
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expStampsMovedNoDoc")
    _fix_stamp(monkeypatch)
    destination = _expected_destination(built)

    fault = inject_store_fault(monkeypatch, method_name="put_blob")
    with pytest.raises(RuntimeError):
        clear_prediction_bucket(str(built["bucket"]), "should crash after the stamps moved")
    assert fault.fired
    assert bucket_stems(built["bucket"]) == {"img"}

    result = clear_prediction_bucket(str(built["bucket"]), "keyword-less call over the wreckage")
    assert "error" in result
    assert "carries no operating_point.json" in result["error"]
    assert f"cleared_bucket={str(destination)!r}" in result["error"]


def test_a_keyword_less_call_in_the_emptied_state_refuses_naming_the_newest_archive(
        tmp_path, monkeypatch):
    """A source a clear emptied entirely, whose own audit entry never appended (the crash after the
    last delete and before the body returns): the source carries neither a stamp nor a document, so
    a later keyword-less call over it still reads no experiment and is answered by the same
    archive-walking refusal, naming this already-finished archive as the remedy."""
    from tcip_mcp.prediction_buckets import bucket_stems
    import tcip_mcp.prediction_buckets as prediction_buckets_mod
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expEmptiedNoAuditLine")
    _fix_stamp(monkeypatch)
    destination = _expected_destination(built)

    raise_on_nth_call(monkeypatch, prediction_buckets_mod, "review_state_count", 2)
    with pytest.raises(RuntimeError):
        clear_prediction_bucket(str(built["bucket"]), "should crash after the last write")
    assert bucket_stems(built["bucket"]) == set()
    assert bucket_stems(destination) == {"img"}

    result = clear_prediction_bucket(str(built["bucket"]), "keyword-less call over the empty source")
    assert "error" in result
    assert "carries no operating_point.json" in result["error"]
    assert f"cleared_bucket={str(destination)!r}" in result["error"]


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


def test_a_secondary_stamp_present_at_the_source_alone_moves_or_refuses_by_the_op_stamps_state(
        tmp_path, monkeypatch):
    """A source-only secondary stamp reads differently by whether operating_point.json has
    already left source. Before it has (the crash fell before any stamp write): the secondary
    stamp is this call's own original, unmoved value, and the resume moves it normally. After it
    has (a resume over an already half-cleared source): a value present at the source alone can
    only be a fresh write one of the three producers made into the half-cleared directory since
    this door already left it stampless, and the resume refuses naming them; a value present at
    both sides and differing refuses the same way, whether or not it was ever absent."""
    from tcip_mcp.pipelines.resolution import _read_sidecar, read_operating_point_sidecar, write_sidecar
    from tcip_mcp.prediction_buckets import bucket_stems
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    built_a = build_published_bucket(
        tmp_path, monkeypatch, experiment_id="expSecondaryOpNotMoved", dataset_root=tmp_path / "ds_a")
    write_sidecar(built_a["bucket"], {"conf": {"value": 0.1}}, document="resolve_scale")
    _fix_stamp(monkeypatch)
    destination_a = _expected_destination(built_a)

    fault_a = inject_store_fault(
        monkeypatch, method_name="replace", predicate=key_in_store("operating_point_sidecar"))
    with pytest.raises(RuntimeError):
        clear_prediction_bucket(str(built_a["bucket"]), "should crash before any stamp write")
    assert fault_a.fired

    result_a = clear_prediction_bucket(
        str(built_a["bucket"]), "resume after the no-stamp-write crash",
        cleared_bucket=str(destination_a))
    assert "error" not in result_a, result_a
    assert read_operating_point_sidecar(destination_a) is not None
    assert _read_sidecar(destination_a, "resolve_scale") == {"conf": {"value": 0.1}}
    assert _read_sidecar(built_a["bucket"], "resolve_scale") is None
    assert bucket_stems(built_a["bucket"]) == set()

    built_b = build_published_bucket(
        tmp_path, monkeypatch, experiment_id="expSecondaryOpMoved", dataset_root=tmp_path / "ds_b")
    destination_b = _expected_destination(built_b)

    fault_b = inject_store_fault(monkeypatch, method_name="put_blob")
    with pytest.raises(RuntimeError):
        clear_prediction_bucket(str(built_b["bucket"]), "should crash after the stamps moved")
    assert fault_b.fired

    write_sidecar(built_b["bucket"], {"conf": {"value": 0.4}}, document="resolve_scale")

    result_b = clear_prediction_bucket(
        str(built_b["bucket"]), "resume after a fresh secondary stamp landed",
        cleared_bucket=str(destination_b))
    assert "error" in result_b
    assert "resolve_scale.json is present at" in result_b["error"]

    built_c = build_published_bucket(
        tmp_path, monkeypatch, experiment_id="expSecondaryOpMovedDiffers", dataset_root=tmp_path / "ds_c")
    write_sidecar(built_c["bucket"], {"conf": {"value": 0.1}}, document="resolve_scale")
    destination_c = _expected_destination(built_c)

    fault_c = inject_store_fault(monkeypatch, method_name="put_blob")
    with pytest.raises(RuntimeError):
        clear_prediction_bucket(str(built_c["bucket"]), "should crash after the stamps moved")
    assert fault_c.fired

    write_sidecar(built_c["bucket"], {"conf": {"value": 0.4}}, document="resolve_scale")

    result_c = clear_prediction_bucket(
        str(built_c["bucket"]), "resume after a differing secondary stamp landed",
        cleared_bucket=str(destination_c))
    assert "error" in result_c
    assert "resolve_scale.json differs" in result_c["error"]


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


def test_a_resume_naming_an_older_finished_archive_of_a_source_cleared_twice_refuses_naming_the_newest(
        tmp_path, monkeypatch):
    """A source cleared, republished and cleared again carries two cleared: artifacts on the same
    experiment. A keyword-less call after the republication proceeds as a new, second clear with
    its own second artifact (coverage: the pass-through baseline proceeds too), and a resume
    naming the older archive refuses, naming the newest instead of merging this clear's own
    reconciliation into a finished, earlier publication."""
    import tcip_mcp.dataset_layout as dataset_layout_mod
    from tcip_mcp.prediction_buckets import bucket_stems
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket, run_inference

    stamps = iter(["20260906T120000Z", "20260906T120001Z"])
    monkeypatch.setattr(dataset_layout_mod, "current_cleared_stamp", lambda: next(stamps))

    exp_id = "expClearedTwice"
    built = build_published_bucket(tmp_path, monkeypatch, experiment_id=exp_id)
    source = built["bucket"]

    first = clear_prediction_bucket(str(source), "first clear")
    assert "error" not in first, first
    first_destination = first["cleared_bucket"]
    assert bucket_stems(source) == set()

    republish = run_inference(
        str(built["checkpoint"]), str(built["images_dir"]), output_dir=str(source), tile=False,
        experiment_id=exp_id)
    assert "error" not in republish, republish
    assert bucket_stems(source) == {"img"}

    second = clear_prediction_bucket(str(source), "second clear over the re-published source")
    assert "error" not in second, second
    second_destination = second["cleared_bucket"]
    assert second_destination != first_destination
    assert bucket_stems(source) == set()

    resume_on_older = clear_prediction_bucket(
        str(source), "resume naming the older archive", cleared_bucket=first_destination)
    assert "error" in resume_on_older
    assert "is not the newest cleared bucket on record" in resume_on_older["error"]
    assert f"cleared_bucket={second_destination!r}" in resume_on_older["error"]
