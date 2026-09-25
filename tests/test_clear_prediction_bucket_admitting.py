"""clear_prediction_bucket: the admitting round trip, for a dated and an undated bucket, a
completed and a failed experiment, plus the lock's own pointer comparison."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tests._clear_prediction_bucket_fixtures import (
    assert_source_stamps_absent, build_published_bucket, earn_validated_stamp,
)

pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")

_EARNED_TRAIT = "bud_opening"


def _bundle_class(accounting, path: Path) -> str:
    """Which of account_for's four membership classes path falls under: "plan", "blob",
    "bookkeeping" or "unaccounted"; "not_walked" when the tree account_for walked never named it."""
    target = path.resolve()
    for plan in accounting.plans:
        if any(entry.path.resolve() == target for entry in plan.entries):
            return "plan"
    if any(Path(p).resolve() == target for p in accounting.blobs):
        return "blob"
    if any(Path(p).resolve() == target for p in accounting.bookkeeping):
        return "bookkeeping"
    if any(Path(p).resolve() == target for p in accounting.unaccounted):
        return "unaccounted"
    return "not_walked"


@pytest.mark.parametrize("date", ["2026-03-02", None])
@pytest.mark.parametrize("state", ["completed", "failed"])
def test_admitting_round_trip_clears_and_admits_republication(tmp_path, monkeypatch, date, state):
    """A terminal experiment's bucket, dated or undated, completed or failed: cleared with a
    reason, the source reads empty and unpublished, the cleared bucket holds the first run's
    documents and stamp, and a second run in place then publishes."""
    from tcip_mcp.dataset_layout import (
        find_prediction, image_path, list_models, prediction_bucket_date, prediction_bucket_dirs,
    )
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar
    from tcip_mcp.prediction_buckets import bucket_content_digest, bucket_stems
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket, run_inference

    exp_id = f"expAdmit_{state}_{date}"
    built = build_published_bucket(
        tmp_path, monkeypatch, experiment_id=exp_id, date=date, state=state)
    dataset_root, source = built["dataset_root"], built["bucket"]
    original_stamp = earn_validated_stamp(source, dataset_root, trait=_EARNED_TRAIT)
    original_stems = bucket_stems(source)
    live_reference = str(image_path(dataset_root, date, "img", ".png"))
    assert find_prediction(live_reference) == source / "img.json"

    from tcip_mcp.tools.bundle import account_for

    live_class = _bundle_class(account_for(dataset_root), source / "img.json")
    assert live_class != "not_walked"

    result = clear_prediction_bucket(str(source), "clearing for a fresh run")
    assert "error" not in result, result
    assert result["resumed"] is False
    assert result["source_republished"] is False
    assert result["cleared_artifact_recorded"] is True
    assert result["experiment_id"] == exp_id
    assert result["documents_moved_this_call"] == 1
    assert "operating_point" in result["stamps_moved_this_call"]

    cleared_path = Path(result["cleared_bucket"])
    assert result["source_digest_before_call"] == bucket_content_digest(cleared_path)

    # The source reads empty and unpublished; the cleared bucket is invisible to list_models.
    assert bucket_stems(source) == set()
    assert read_operating_point_sidecar(source) is None
    assert_source_stamps_absent(source)
    assert ".cleared" not in list_models(dataset_root)
    assert cleared_path.name not in list_models(dataset_root)

    # The cleared bucket holds the first run's documents and stamp.
    assert bucket_stems(cleared_path) == original_stems
    cleared_stamp = read_operating_point_sidecar(cleared_path)
    assert cleared_stamp is not None
    assert cleared_stamp.get("validated_by") is not None
    assert cleared_stamp.get("validated_by") == original_stamp.get("validated_by")

    # Both walks agree on visibility.
    live_only = prediction_bucket_dirs(dataset_root, include_cleared=False)
    with_archive = prediction_bucket_dirs(dataset_root, include_cleared=True)
    assert cleared_path not in live_only
    assert cleared_path in with_archive

    # prediction_bucket_date answers None for the cleared shape.
    assert prediction_bucket_date(cleared_path) is None

    # find_prediction answers None for the stem now that it lives under the archive.
    assert find_prediction(live_reference) is None

    # project_roots names the cleared bucket's own path, once the dataset is registered.
    from tcip_mcp.store_catalog import project_roots
    from tcip_mcp.tools.project_tools import register_dataset

    registered = register_dataset(str(dataset_root), "black_locust")
    assert "error" not in registered, registered
    roots = {Path(p).resolve() for p, _layout in project_roots(dataset_root)}
    assert cleared_path.resolve() in roots

    # account_for gives the cleared files exactly the class the live bucket's files took.
    cleared_class = _bundle_class(account_for(dataset_root), cleared_path / "img.json")
    assert cleared_class == live_class

    # The experiment's artifacts carry the cleared path under its own cleared: name.
    from tcip_mcp.experiments import get_experiment

    experiment = get_experiment(exp_id)
    cleared_artifacts = {
        name: entry for name, entry in (experiment.get("artifacts") or {}).items()
        if name.startswith("cleared:")
    }
    assert any(entry.get("path") == str(cleared_path) for entry in cleared_artifacts.values())

    # The dataset audit log carries the door's own entry, naming the reason.
    from tcip_store import read_log

    from tcip_mcp.audit import audit_log_key

    door_entries = [
        e for e in read_log(audit_log_key(dataset_root)).records
        if e.get("tool") == "clear_prediction_bucket"
        and e.get("arguments", {}).get("predictions_dir") == str(source)
    ]
    assert door_entries
    assert door_entries[-1]["arguments"]["reason"] == "clearing for a fresh run"

    # A second run in place now publishes, and find_prediction answers the source path again.
    second = run_inference(
        str(built["checkpoint"]), str(built["images_dir"]), output_dir=str(source), tile=False,
        experiment_id=exp_id)
    assert "error" not in second, second
    assert bucket_stems(source) == {"img"}
    assert find_prediction(live_reference) == source / "img.json"


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
