"""run_inference's raster_path regime resumes an interrupted tiled pass.

A pass over a small (64x64, tile 32, no overlap: exactly four tiles) geo-referenced raster
records its own identity under the project before the first tile and one batch record per
flushed tile (tile_batch_size=1 here, so each tile is its own batch). Interruption is simulated
by making the
store's own replace raise once the pass has durably recorded one batch, the same shape a real
crash mid-pass leaves: the identity record and that one batch record survive, nothing else does.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tcip_store.file_backend import _is_bookkeeping  # noqa: E402

from tests.test_orthomosaic_tools import (  # noqa: E402
    TILE, _bespoke_detection_checkpoint, _write_geo_raster,
)


def _instance_seg_checkpoint(tmp_path: Path) -> str:
    from tcip_mcp.pipelines.model_build import build_model, recorded_model_dims
    from tcip_mcp.tools.model_tools import register_model

    model_source = {"builder": "tests.bespoke_models:build_fixed_mask_instance_seg",
                    "task": "instance_seg"}
    config = {"model_source": model_source,
              "data": {"num_channels": 3, "scope": {"subject": "bud", "id_map": {"bud": 0}}}}
    model = build_model(config, recorded_model_dims(config))
    ckpt = tmp_path / "instance_seg.pt"
    torch.save({"config": config, "model_state_dict": model.state_dict()}, str(ckpt))
    result = register_model(tmp_path, name="instance-seg-test-model", checkpoint_path=str(ckpt),
                            config={})
    assert "error" not in result, result
    return str(ckpt)


def _setup(tmp_path: Path) -> tuple[str, Path]:
    """A registered bespoke detection checkpoint and a real, readable 64x64 geo raster (four
    32px tiles at overlap 0.0), in the project ``tmp_path``."""
    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path, height=64, width=64)
    return _bespoke_detection_checkpoint(tmp_path), raster_path


def _run(project: Path, ckpt: str, raster_path: Path, out: Path, *, conf: float = 0.0,
         **kwargs) -> dict:
    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.tools.inference_tools import run_inference

    call_kwargs = {
        "raster_path": str(raster_path), "output_dir": str(out),
        "stated": Stated(conf=conf, tile_size=TILE, overlap=0.0), "tile_batch_size": 1,
        "device": "cpu",
    }
    call_kwargs.update(kwargs)
    return run_inference(project, ckpt, **call_kwargs)


def _progress(project: Path, out: Path) -> list[str]:
    """The segments of every progress record a raster pass toward ``out`` left under
    ``project``."""
    from tcip_mcp.tools.inference_tools import _progress_keys

    return sorted(key.parts[1] for key in _progress_keys(project, out))


def _interrupt_after_one_batch(monkeypatch) -> None:
    """Make the store's own replace raise the moment a second raster-pass batch record would
    land, so the durable state left behind is exactly what a real crash after one flushed batch
    leaves: the identity record, and that one batch record, nothing past it."""
    import tcip_store.store as store_mod

    real_replace = store_mod.replace
    seen = {"batches": 0}

    def _flaky_replace(key, value, **kw):
        if key.store == "raster_pass_progress" and key.parts[1].startswith("batch-"):
            seen["batches"] += 1
            if seen["batches"] == 2:
                raise RuntimeError("simulated crash mid-pass")
        return real_replace(key, value, **kw)

    monkeypatch.setattr(store_mod, "replace", _flaky_replace)


def _interrupted(tmp_path: Path, monkeypatch, out: Path, ckpt: str, raster_path: Path,
                 **kwargs) -> None:
    _interrupt_after_one_batch(monkeypatch)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run(tmp_path, ckpt, raster_path, out, **kwargs)
    monkeypatch.undo()


def _as_produced(out: Path, stem: str = "mosaic") -> list[tuple]:
    doc = json.loads((out / f"{stem}.json").read_text())
    return [(tuple(a["bbox"]), a["score"], a["subject"]) for a in doc["annotations"]]


def test_resume_refuses_with_images_dir(tmp_path):
    from tcip_mcp.tools.inference_tools import run_inference

    images_dir = tmp_path / "images"
    images_dir.mkdir()

    result = run_inference(tmp_path, str(tmp_path / "m.pt"), images_dir=str(images_dir),
                           output_dir=str(tmp_path / "out"), resume=True)

    assert "resume applies only to a raster_path pass" in result["error"]


def test_resume_refuses_with_no_recorded_progress(tmp_path):
    ckpt, raster_path = _setup(tmp_path)
    out = tmp_path / "preds"

    result = _run(tmp_path, ckpt, raster_path, out, resume=True)

    assert "no raster-pass progress toward" in result["error"]
    assert not out.exists()


def test_an_interrupted_instance_segmentation_pass_resumes_to_the_uninterrupted_masks(
    tmp_path, monkeypatch,
):
    """The recorded progress carries each slice's shifted predictions whole, polygons included,
    so the resumed pass merges the seeded masks exactly as the uninterrupted pass does."""
    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path, height=64, width=64)
    ckpt = _instance_seg_checkpoint(tmp_path)
    interrupted_out, uninterrupted_out = tmp_path / "interrupted", tmp_path / "uninterrupted"

    baseline = _run(tmp_path, ckpt, raster_path, uninterrupted_out, require_masks=True)
    assert "error" not in baseline, baseline
    _interrupted(tmp_path, monkeypatch, interrupted_out, ckpt, raster_path, require_masks=True)
    resumed = _run(tmp_path, ckpt, raster_path, interrupted_out, resume=True, require_masks=True)
    assert "error" not in resumed, resumed

    def _annotations(out: Path) -> list[dict]:
        doc = json.loads((out / "mosaic.json").read_text())
        return [{k: v for k, v in a.items() if k != "created_at"} for a in doc["annotations"]]

    baseline_annotations = _annotations(uninterrupted_out)
    assert any(a.get("segmentation") for a in baseline_annotations)
    assert _annotations(interrupted_out) == baseline_annotations


def test_an_interrupted_pass_leaves_one_identity_and_one_batch_record_and_no_bucket(
    tmp_path, monkeypatch,
):
    """The progress sits under the project, apart from the bucket the pass will publish, and the
    bucket directory is created only once the pass is whole."""
    ckpt, raster_path = _setup(tmp_path)
    out = tmp_path / "preds"

    _interrupted(tmp_path, monkeypatch, out, ckpt, raster_path)

    assert _progress(tmp_path, out) == ["batch-000000", "identity"]
    assert not out.exists()


def test_resume_completes_an_interrupted_pass_with_the_same_detections_and_clears_progress(
    tmp_path, monkeypatch,
):
    ckpt, raster_path = _setup(tmp_path)
    interrupted_out = tmp_path / "interrupted"
    _interrupted(tmp_path, monkeypatch, interrupted_out, ckpt, raster_path)

    resumed = _run(tmp_path, ckpt, raster_path, interrupted_out, resume=True)

    assert "error" not in resumed, resumed
    assert _progress(tmp_path, interrupted_out) == []
    uninterrupted_out = tmp_path / "uninterrupted"
    baseline = _run(tmp_path, ckpt, raster_path, uninterrupted_out)
    assert "error" not in baseline, baseline
    assert resumed["execution"] == baseline["execution"]
    # As produced, not sorted: a resumed pass reconstructs the identical detection set in the
    # identical order, not merely the same boxes in some order.
    assert _as_produced(interrupted_out) == _as_produced(uninterrupted_out)


def test_a_fresh_pass_over_recorded_progress_refuses_and_leaves_it(tmp_path, monkeypatch):
    ckpt, raster_path = _setup(tmp_path)
    out = tmp_path / "preds"
    _interrupted(tmp_path, monkeypatch, out, ckpt, raster_path)

    result = _run(tmp_path, ckpt, raster_path, out)

    assert "already recorded" in result["error"]
    assert _progress(tmp_path, out) == ["batch-000000", "identity"]
    assert not out.exists()


def test_a_published_bucket_whose_progress_clear_failed_refuses_every_later_pass(
    tmp_path, monkeypatch,
):
    """A crash between publication and the progress clear leaves a whole bucket and a progress
    record beside it: a fresh pass refuses on the recorded progress, and a resume refuses on the
    bucket rather than publishing over it; every file of the bucket stays as published."""
    import tcip_mcp.tools.inference_tools as itools

    ckpt, raster_path = _setup(tmp_path)
    out = tmp_path / "preds"

    def _raise_after_publish(*_args):
        raise RuntimeError("simulated crash between publication and the progress clear")

    monkeypatch.setattr(itools, "_clear_raster_pass_progress", _raise_after_publish)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run(tmp_path, ckpt, raster_path, out)
    monkeypatch.undo()

    def files() -> dict[str, bytes]:
        return {p.name: p.read_bytes() for p in sorted(out.iterdir())
                if not _is_bookkeeping(p.name)}

    published = files()
    assert set(published) == {"bucket.json", "mosaic.json"}

    fresh = _run(tmp_path, ckpt, raster_path, out)
    resumed = _run(tmp_path, ckpt, raster_path, out, resume=True)

    assert "already recorded" in fresh["error"]
    assert "already exists" in resumed["error"]
    assert files() == published


def test_resume_refuses_a_call_that_differs_from_the_recorded_pass(tmp_path, monkeypatch):
    ckpt, raster_path = _setup(tmp_path)
    out = tmp_path / "preds"
    _interrupted(tmp_path, monkeypatch, out, ckpt, raster_path)

    result = _run(tmp_path, ckpt, raster_path, out, resume=True, conf=0.9)

    assert "conf" in result["error"]
    # A refused resume leaves the recorded progress untouched.
    assert _progress(tmp_path, out) == ["batch-000000", "identity"]


def test_resume_refuses_when_the_recorded_identity_carries_an_extra_top_level_key(
    tmp_path, monkeypatch,
):
    """The top-level comparison is a key union, not a fixed field list: a field a future writer
    adds to the identity body is compared, and named, without a second list to keep in sync."""
    from tcip_mcp.tools.inference_tools import _progress_key
    from tcip_store import store

    ckpt, raster_path = _setup(tmp_path)
    out = tmp_path / "preds"
    _interrupted(tmp_path, monkeypatch, out, ckpt, raster_path)
    identity_key = _progress_key(tmp_path, out, "identity")
    body = dict(store.read(identity_key))
    body["future_field"] = "a value this reader does not expect"
    with store.transaction(identity_key) as txn:
        txn.write(identity_key, body)

    result = _run(tmp_path, ckpt, raster_path, out, resume=True)

    assert "future_field" in result["error"]


def test_content_identity_failure_after_open_refuses_naming_the_raster(tmp_path, monkeypatch):
    """A raster that opens cleanly but fails partway through the sampling read refuses through
    the same ``{"error": ...}`` shape a failed open does, and writes nothing."""
    import tcip_mcp.pipelines.raster_source as raster_source_module

    ckpt, raster_path = _setup(tmp_path)
    out = tmp_path / "preds"
    real_open_raster = raster_source_module.open_raster

    def _flaky_open_raster(source, num_channels):
        reader = real_open_raster(source, num_channels)

        def _raise(*args, **kwargs):
            raise OSError("simulated disk read failure")

        reader.read_region = _raise
        return reader

    monkeypatch.setattr(raster_source_module, "open_raster", _flaky_open_raster)

    result = _run(tmp_path, ckpt, raster_path, out)

    assert str(raster_path) in result["error"]
    assert not out.exists()


def test_an_interrupted_pass_under_an_assessment_resumes_under_that_assessment(
    tmp_path, monkeypatch,
):
    """A pass published under a reserved-regions assessment resumes under the same assessment and
    execution record, reproducing the detections an uninterrupted pass makes, and a resume naming
    another assessment refuses by name."""
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.tools.inference_tools import run_inference
    from tests import _trait_fixtures as fx
    from tests.test_block_calibration import TILE as BLOCK_TILE, _assess, _attested

    fx.seed_confirmed_count(tmp_path, measured_subject="bud")
    exp = _attested(tmp_path)
    assessment = _assess(exp)
    call = {"raster_path": str(exp["raster_path"]), "tile_batch_size": 50, "device": "cpu",
            "assessment_id": assessment["assessment_id"]}
    interrupted_out = tmp_path / "interrupted"
    _interrupt_after_one_batch(monkeypatch)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_inference(tmp_path, exp["checkpoint_path"], output_dir=str(interrupted_out), **call)
    monkeypatch.undo()

    other = run_inference(tmp_path, exp["checkpoint_path"], output_dir=str(interrupted_out),
                          resume=True, **{**call, "assessment_id": "another"})
    resumed = run_inference(tmp_path, exp["checkpoint_path"], output_dir=str(interrupted_out),
                            resume=True, **call)
    uninterrupted_out = tmp_path / "uninterrupted"
    baseline = run_inference(tmp_path, exp["checkpoint_path"],
                             output_dir=str(uninterrupted_out), **call)

    assert "error" in other
    assert "error" not in resumed, resumed
    assert "error" not in baseline, baseline
    bucket = read_bucket(interrupted_out)
    assert bucket.assessment_id == assessment["assessment_id"]
    assert bucket.execution.record() == assessment["execution"]
    assert bucket.execution.tile_size == BLOCK_TILE
    assert _as_produced(interrupted_out, exp["stem"]) == _as_produced(uninterrupted_out,
                                                                       exp["stem"])
