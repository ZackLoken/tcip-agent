"""run_inference's raster_path regime resumes an interrupted tiled pass.

A pass over a small (64x64, tile 32, no overlap: exactly four tiles) geo-referenced raster
records its own identity before the first tile and one batch record per flushed tile
(tile_batch_size=1 here, so each tile is its own batch). Interruption is simulated by making the
store's own replace raise once the pass has durably recorded one batch, the same shape a real
crash mid-pass leaves: the identity record and that one batch record survive, nothing else does.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tests.test_orthomosaic_tools import (  # noqa: E402
    TILE, _bespoke_detection_checkpoint, _write_geo_raster,
)


def _instance_seg_checkpoint(tmp_path: Path) -> str:
    from tcip_mcp.pipelines.model_build import build_model
    from tcip_mcp.tools.model_tools import register_model

    model_source = {"builder": "tests.bespoke_models:build_bespoke_instance_seg",
                    "builder_kwargs": {"num_classes": 1, "min_size": TILE, "max_size": TILE * 2},
                    "task": "instance_seg"}
    model = build_model({"model_source": model_source})
    ckpt = tmp_path / "instance_seg.pt"
    torch.save({"model_source": model_source, "model_state_dict": model.state_dict()}, str(ckpt))
    result = register_model(name="instance-seg-test-model", checkpoint_path=str(ckpt), config={})
    assert "error" not in result, result
    return str(ckpt)


def _setup(tmp_path: Path, monkeypatch) -> tuple[str, Path]:
    """A registered bespoke detection checkpoint and a real, readable 64x64 geo raster (four
    32px tiles at overlap 0.0), under a pinned platform state root."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path, height=64, width=64)
    ckpt = _bespoke_detection_checkpoint(tmp_path)
    return ckpt, raster_path


def _run(ckpt: str, raster_path: Path, out: Path, **kwargs) -> dict:
    from tcip_mcp.tools.inference_tools import run_inference

    call_kwargs = {
        "raster_path": str(raster_path), "output_dir": str(out), "conf_threshold": 0.0,
        "tile_size": TILE, "overlap": 0.0, "tile_batch_size": 1, "device": "cpu",
    }
    call_kwargs.update(kwargs)
    return run_inference(ckpt, **call_kwargs)


def _progress_keys(out: Path):
    from tcip_mcp.tools.inference_tools import RASTER_PASS_PROGRESS_STORE
    from tcip_store import store

    return store.keys(RASTER_PASS_PROGRESS_STORE, str(out))


def _interrupt_after_one_batch(monkeypatch) -> None:
    """Make the store's own replace raise the moment a second raster-pass batch record would
    land, so the durable state left behind is exactly what a real crash after one flushed batch
    leaves: the identity record, and that one batch record, nothing past it."""
    import tcip_store.store as store_mod

    real_replace = store_mod.replace
    seen = {"batches": 0}

    def _flaky_replace(key, value, **kw):
        if key.store == "raster_pass_progress" and key.parts[0].startswith("batch-"):
            seen["batches"] += 1
            if seen["batches"] == 2:
                raise RuntimeError("simulated crash mid-pass")
        return real_replace(key, value, **kw)

    monkeypatch.setattr(store_mod, "replace", _flaky_replace)


def test_resume_refuses_with_images_dir(tmp_path):
    from tcip_mcp.tools.inference_tools import run_inference

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    result = run_inference(
        str(ckpt), images_dir=str(images_dir), output_dir=str(tmp_path / "out"), resume=True)

    assert "error" in result and "resume=True" in result["error"]


def test_resume_refuses_with_no_recorded_progress(tmp_path, monkeypatch):
    ckpt, raster_path = _setup(tmp_path, monkeypatch)
    out = tmp_path / "preds"

    result = _run(ckpt, raster_path, out, resume=True)

    assert "error" in result
    assert "no raster-pass identity record" in result["error"]
    assert not out.exists()


def test_resume_refuses_a_mask_bearing_pass(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path, height=64, width=64)
    ckpt = _instance_seg_checkpoint(tmp_path)
    out = tmp_path / "preds"

    result = _run(ckpt, raster_path, out, resume=True, require_masks=True)

    assert "error" in result and "mask-bearing" in result["error"]
    assert not out.exists()


def test_an_interrupted_pass_leaves_one_identity_and_one_batch_record(tmp_path, monkeypatch):
    ckpt, raster_path = _setup(tmp_path, monkeypatch)
    out = tmp_path / "preds"
    _interrupt_after_one_batch(monkeypatch)

    with pytest.raises(RuntimeError, match="simulated crash"):
        _run(ckpt, raster_path, out)

    keys = _progress_keys(out)
    assert sorted(k.parts[0] for k in keys) == ["batch-000000", "identity"]
    assert not (out / "mosaic.json").exists()  # the prediction file itself never landed


def test_a_bucket_holding_progress_enumerates_only_its_prediction_documents(tmp_path, monkeypatch):
    """``prediction_documents``' own non-recursive glob of the bucket root must never see the
    progress records sitting under ``<bucket>/.tcip/raster_pass_progress/``."""
    from tcip_annotation.json_io import prediction_documents

    ckpt, raster_path = _setup(tmp_path, monkeypatch)
    out = tmp_path / "preds"
    _interrupt_after_one_batch(monkeypatch)

    with pytest.raises(RuntimeError, match="simulated crash"):
        _run(ckpt, raster_path, out)

    assert _progress_keys(out)  # the progress really is there to be missed
    assert list(prediction_documents(out)) == []


def test_resume_completes_an_interrupted_pass_with_the_same_detections_and_clears_progress(
    tmp_path, monkeypatch,
):
    ckpt, raster_path = _setup(tmp_path, monkeypatch)
    interrupted_out = tmp_path / "interrupted"
    _interrupt_after_one_batch(monkeypatch)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run(ckpt, raster_path, interrupted_out)

    resumed = _run(ckpt, raster_path, interrupted_out, resume=True)
    assert "error" not in resumed, resumed
    assert resumed["tiles"] == 4
    assert _progress_keys(interrupted_out) == []

    # An uninterrupted pass over the identical inputs, into its own bucket, is the ground truth
    # the resumed one must match: same tile count, same detections.
    uninterrupted_out = tmp_path / "uninterrupted"
    baseline = _run(ckpt, raster_path, uninterrupted_out)
    assert "error" not in baseline, baseline
    assert baseline["tiles"] == resumed["tiles"]

    import json

    resumed_doc = json.loads((interrupted_out / "mosaic.json").read_text())
    baseline_doc = json.loads((uninterrupted_out / "mosaic.json").read_text())

    def _as_produced(doc: dict) -> list[tuple]:
        return [(tuple(a["bbox"]), a["score"], a["subject"]) for a in doc["annotations"]]

    # As produced, not sorted: a resumed pass must reconstruct the identical detection set in the
    # identical order, not merely the same boxes in some order.
    assert _as_produced(resumed_doc) == _as_produced(baseline_doc)


def test_no_resume_refuses_over_recorded_progress_naming_both_ways_out(tmp_path, monkeypatch):
    ckpt, raster_path = _setup(tmp_path, monkeypatch)
    out = tmp_path / "preds"
    _interrupt_after_one_batch(monkeypatch)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run(ckpt, raster_path, out)

    result = _run(ckpt, raster_path, out)

    assert "error" in result
    assert "resume=True" in result["error"] and "overwrite=True" in result["error"]
    assert _progress_keys(out)  # still there, untouched by the refusal


def test_overwrite_discards_recorded_progress_and_starts_over(tmp_path, monkeypatch):
    ckpt, raster_path = _setup(tmp_path, monkeypatch)
    out = tmp_path / "preds"
    _interrupt_after_one_batch(monkeypatch)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run(ckpt, raster_path, out)
    assert _progress_keys(out)

    result = _run(ckpt, raster_path, out, overwrite=True)

    assert "error" not in result, result
    assert result["tiles"] == 4
    assert _progress_keys(out) == []


def test_a_document_and_progress_record_left_coexisting_refuses_on_the_document(
    tmp_path, monkeypatch,
):
    """A crash between the document write and the progress clear leaves both a document and a
    progress record behind. The resolver's document refusal fires on any later call to this
    bucket before resume or overwrite is ever consulted (the resolver runs ahead of both), and
    the progress record stays exactly where it is, inert: only a run into this bucket reads it."""
    import tcip_mcp.tools.inference_tools as itools

    ckpt, raster_path = _setup(tmp_path, monkeypatch)
    out = tmp_path / "preds"

    def _raise_after_write(bucket):
        raise RuntimeError("simulated crash between the document write and the progress clear")

    monkeypatch.setattr(itools, "_clear_raster_pass_progress", _raise_after_write)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run(ckpt, raster_path, out)

    assert (out / "mosaic.json").is_file()
    assert _progress_keys(out)

    for kwargs in ({}, {"resume": True}, {"overwrite": True}):
        result = _run(ckpt, raster_path, out, **kwargs)
        assert "error" in result, (kwargs, result)
        assert result["document_stem_count"] == 1
    assert _progress_keys(out)  # untouched by any of the three refusals above


def test_resume_refuses_a_call_that_differs_from_the_recorded_pass(tmp_path, monkeypatch):
    ckpt, raster_path = _setup(tmp_path, monkeypatch)
    out = tmp_path / "preds"
    _interrupt_after_one_batch(monkeypatch)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run(ckpt, raster_path, out)

    result = _run(ckpt, raster_path, out, resume=True, conf_threshold=0.9)

    assert "error" in result
    assert "differs" in result["error"] and "operating_point.conf" in result["error"]
    assert _progress_keys(out)  # a refused resume leaves the recorded progress untouched


def test_resume_refuses_an_identity_schema_version_the_reader_does_not_know(tmp_path, monkeypatch):
    from tcip_mcp.tools.inference_tools import RASTER_PASS_PROGRESS_STORE
    from tcip_store import Key, store

    ckpt, raster_path = _setup(tmp_path, monkeypatch)
    out = tmp_path / "preds"
    _interrupt_after_one_batch(monkeypatch)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run(ckpt, raster_path, out)

    identity_key = Key(RASTER_PASS_PROGRESS_STORE, str(out), ("identity",))
    body = dict(store.read(identity_key))
    body["schema_version"] = 999
    with store.transaction(identity_key) as txn:
        txn.write(identity_key, body)

    result = _run(ckpt, raster_path, out, resume=True)

    assert "error" in result and "schema_version" in result["error"]


def test_a_redirected_interrupted_pass_resumes_in_its_own_bucket(tmp_path, monkeypatch):
    """A bucket redirect (a review verdict already recorded against the requested path) resolves
    the same way on the interrupted attempt and on the resume attempt, so resume=True against the
    originally-requested output_dir finds and continues the redirected bucket's own progress."""
    import json

    from tcip_annotation.review_engine import ReviewContext, ReviewDetection, ReviewEngine
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.prediction_buckets import bucket_key_of

    ckpt, raster_path = _setup(tmp_path, monkeypatch)
    dataset_root = tmp_path / "dataset"
    requested = dataset_root / "predictions" / "preds"
    requested.mkdir(parents=True)
    (requested / "mosaic.json").write_text(
        json.dumps({"image": "mosaic", "width": 64, "height": 64, "annotations": []}))
    engine = ReviewEngine(dataset_root / ".tcip" / "state")
    ctx = ReviewContext(img_name="mosaic", img_width=64, img_height=64, preds=[
        Annotation(subject="bud", geometry=BBox(1.0, 1.0, 5.0, 5.0), score=0.9)])
    det = ReviewDetection(det_type="fp", class_name="bud", conf=0.9, iou=None, gt_idx=None,
                          pred_idx=0, bbox=(1.0, 1.0, 5.0, 5.0))
    engine.record_detection_action(bucket_key_of(requested), det, ctx, action="accepted")

    _interrupt_after_one_batch(monkeypatch)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run(ckpt, raster_path, requested)

    redirected = requested.parent / "preds@r2"
    assert _progress_keys(redirected)
    assert not _progress_keys(requested)

    result = _run(ckpt, raster_path, requested, resume=True)

    assert "error" not in result, result
    assert Path(result["output_dir"]) == redirected
    assert _progress_keys(redirected) == []


def test_resume_refuses_when_the_recorded_identity_carries_an_extra_top_level_key(
    tmp_path, monkeypatch,
):
    """The top-level comparison is a key union, not a fixed field list: a field a future writer
    adds to the identity body is compared, and named, without a second list to keep in sync."""
    from tcip_mcp.tools.inference_tools import RASTER_PASS_PROGRESS_STORE
    from tcip_store import Key, store

    ckpt, raster_path = _setup(tmp_path, monkeypatch)
    out = tmp_path / "preds"
    _interrupt_after_one_batch(monkeypatch)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run(ckpt, raster_path, out)

    identity_key = Key(RASTER_PASS_PROGRESS_STORE, str(out), ("identity",))
    body = dict(store.read(identity_key))
    body["future_field"] = "a value this reader does not expect"
    with store.transaction(identity_key) as txn:
        txn.write(identity_key, body)

    result = _run(ckpt, raster_path, out, resume=True)

    assert "error" in result and "future_field" in result["error"]


def test_content_identity_failure_after_open_refuses_naming_the_raster(tmp_path, monkeypatch):
    """A raster that opens cleanly but fails partway through the sampling read refuses through
    the same ``{"error": ...}`` shape a failed open already does, rather than letting a bare
    exception escape this audited tool."""
    ckpt, raster_path = _setup(tmp_path, monkeypatch)
    out = tmp_path / "preds"

    import tcip_mcp.pipelines.raster_source as raster_source_module

    real_open_raster = raster_source_module.open_raster

    def _flaky_open_raster(source, num_channels):
        reader = real_open_raster(source, num_channels)

        def _raise(*args, **kwargs):
            raise OSError("simulated disk read failure")

        reader.read_region = _raise
        return reader

    monkeypatch.setattr(raster_source_module, "open_raster", _flaky_open_raster)

    result = _run(ckpt, raster_path, out)

    assert "error" in result
    assert str(raster_path) in result["error"]
    assert not out.exists()


def test_resume_completes_an_interrupted_block_calibrated_pass_running_the_calibration_pass_once(
    tmp_path, monkeypatch, seed_bud_trait_spec,
):
    """A block-calibrated resume applies the recorded conf/cross_tile_nms directly rather than
    re-running the reserved-band calibration a second time: the resolver that earns them runs
    exactly once, at the interrupted attempt, and the resumed pass reproduces the same detections
    an uninterrupted pass over the identical inputs would."""
    from tests.test_block_calibration import TILE as BLOCK_TILE, _attest_regions_complete, _build_experiment

    exp = _build_experiment(tmp_path)
    manifest = exp["spatial_manifest"]
    _attest_regions_complete(
        exp["root"], exp["stem"], [manifest["calibration_region"], manifest["test_region"]])

    import tcip_mcp.pipelines.block_calibration as block_calibration_module

    real_resolve = block_calibration_module.resolve_block_calibration_records
    calls = {"n": 0}

    def _counting_resolve(*args, **kwargs):
        calls["n"] += 1
        return real_resolve(*args, **kwargs)

    monkeypatch.setattr(
        block_calibration_module, "resolve_block_calibration_records", _counting_resolve)

    from tcip_mcp.tools.inference_tools import run_inference

    call_kwargs = {
        "raster_path": str(exp["raster_path"]), "conf_threshold": 0.0, "tile_size": BLOCK_TILE,
        "overlap": 0.2, "tile_batch_size": 50, "device": "cpu", "trait": "bud_opening",
        "experiment_id": exp["experiment_id"],
    }

    interrupted_out = tmp_path / "interrupted"
    _interrupt_after_one_batch(monkeypatch)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_inference(exp["checkpoint_path"], output_dir=str(interrupted_out), **call_kwargs)
    assert calls["n"] == 1

    resumed = run_inference(
        exp["checkpoint_path"], output_dir=str(interrupted_out), resume=True, **call_kwargs)
    assert "error" not in resumed, resumed
    assert calls["n"] == 1  # the resume applied the recorded operating point, never re-derived one

    uninterrupted_out = tmp_path / "uninterrupted"
    baseline = run_inference(
        exp["checkpoint_path"], output_dir=str(uninterrupted_out), **call_kwargs)
    assert "error" not in baseline, baseline
    assert calls["n"] == 2

    import json

    resumed_doc = json.loads((interrupted_out / f"{exp['stem']}.json").read_text())
    baseline_doc = json.loads((uninterrupted_out / f"{exp['stem']}.json").read_text())

    def _as_produced(doc: dict) -> list[tuple]:
        return [(tuple(a["bbox"]), a["score"], a["subject"]) for a in doc["annotations"]]

    assert _as_produced(resumed_doc) == _as_produced(baseline_doc)
