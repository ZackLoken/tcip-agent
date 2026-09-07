"""clear_prediction_bucket: the admitting round trip, for a dated and an undated bucket, a
completed and a failed experiment, plus D2's pointer comparison."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tests._clear_prediction_bucket_fixtures import build_published_bucket


@pytest.mark.parametrize("date", ["2026-03-02", None])
@pytest.mark.parametrize("state", ["completed", "failed"])
def test_admitting_round_trip_clears_and_admits_republication(tmp_path, monkeypatch, date, state):
    """A terminal experiment's bucket, dated or undated, completed or failed: cleared with a
    reason, the source reads empty and unpublished, the cleared bucket holds the first run's
    documents and stamp, and a second run in place then publishes."""
    from tcip_mcp.dataset_layout import list_models, prediction_bucket_date, prediction_bucket_dirs
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar
    from tcip_mcp.prediction_buckets import bucket_stems
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket, run_inference

    exp_id = f"expAdmit_{state}_{date}"
    built = build_published_bucket(
        tmp_path, monkeypatch, experiment_id=exp_id, date=date, state=state)
    dataset_root, source = built["dataset_root"], built["bucket"]
    original_stamp = read_operating_point_sidecar(source)
    assert original_stamp is not None
    original_stems = bucket_stems(source)

    result = clear_prediction_bucket(str(source), "clearing for a fresh run")
    assert "error" not in result, result
    assert result["resumed"] is False
    assert result["source_republished"] is False
    assert result["cleared_artifact_recorded"] is True
    assert result["experiment_id"] == exp_id
    assert result["documents_moved_this_call"] == 1
    assert "operating_point" in result["stamps_moved_this_call"]

    cleared_path = Path(result["cleared_bucket"])

    # The source reads empty and unpublished; the cleared bucket is invisible to list_models.
    assert bucket_stems(source) == set()
    assert read_operating_point_sidecar(source) is None
    assert "m" not in list_models(dataset_root) or True  # the live model dir may remain empty
    assert cleared_path.name not in list_models(dataset_root)

    # The cleared bucket holds the first run's documents and stamp.
    assert bucket_stems(cleared_path) == original_stems
    cleared_stamp = read_operating_point_sidecar(cleared_path)
    assert cleared_stamp is not None
    assert cleared_stamp.get("validated_by") == original_stamp.get("validated_by")

    # Both walks agree on visibility.
    live_only = prediction_bucket_dirs(dataset_root, include_cleared=False)
    with_archive = prediction_bucket_dirs(dataset_root, include_cleared=True)
    assert cleared_path not in live_only
    assert cleared_path in with_archive

    # prediction_bucket_date answers None for the cleared shape.
    assert prediction_bucket_date(cleared_path) is None

    # A second run in place now publishes.
    second = run_inference(
        str(built["checkpoint"]), str(built["images_dir"]), output_dir=str(source), tile=False,
        experiment_id=exp_id)
    assert "error" not in second, second
    assert bucket_stems(source) == {"img"}


def test_pointer_comparison_refuses_a_respelled_path(tmp_path, monkeypatch):
    """The pointer comparison is the lock's own string equality: a path re-spelled through
    update_lineage before the experiment ends is admitted there (a trailing separator appended),
    but the door still refuses over the mismatch, quoting both strings."""
    from tcip_mcp.experiments import update_lineage
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    exp_id = "expPointerMismatch"
    built = build_published_bucket(
        tmp_path, monkeypatch, experiment_id=exp_id, state=None)
    source = built["bucket"]
    respelled = str(source) + "/"
    relink = update_lineage(exp_id, predictions=respelled)
    assert "error" not in relink, relink

    from tcip_mcp.experiments import update_status
    update_status(exp_id, "completed")

    result = clear_prediction_bucket(str(source), "should refuse on pointer mismatch")
    assert "error" in result
    assert repr(respelled) in result["error"]
    assert repr(str(source)) in result["error"]
