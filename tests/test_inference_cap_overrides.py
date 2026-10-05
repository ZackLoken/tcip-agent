"""The merge-threshold, detection-cap and conf values of a pass: unset means derive or default,
and a stated value is honored at every value it can take.

``prepare_pass`` decides whether the caller stated a value by the ``None`` sentinel, so a stated
value is recorded as explicit even when it equals the documented default the pass would otherwise
have run at, and a stated cap reaches the predictor, not only the record.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

OVERLAPPING_GT = [[[0.0, 0.0, 20.0, 20.0], [10.0, 0.0, 20.0, 20.0], [60.0, 60.0, 20.0, 20.0]]] * 2
"""Two images of overlapping ground truth (xywh), enough for a real cross-tile merge
derivation."""


def _pass(tmp_path: Path, **stated):
    """The pass a registered checkpoint of ``tmp_path`` prepares under ``stated``."""
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.execution import Stated, prepare_pass
    from tests._verified_checkpoint_fixtures import project_checkpoint

    checkpoint = load_registered_checkpoint(project_checkpoint(tmp_path), project=tmp_path)
    return prepare_pass(checkpoint, Stated(**stated), device="cpu")


def test_values_stated_at_the_documented_defaults_are_recorded_as_explicit(tmp_path):
    """The value a caller states is the value that runs, whatever it happens to equal: reading a
    stated value back as unset would hand the caller a default or a derivation they did not ask
    for."""
    from tcip_mcp.pipelines.execution import DEFAULT_CONF, DEFAULT_MAX_DETS, DEFAULT_NMS_IOU

    p = _pass(tmp_path, tile=True, tile_size=64, cross_tile_nms=DEFAULT_NMS_IOU,
              max_dets=DEFAULT_MAX_DETS, conf=DEFAULT_CONF)

    record = p.execution.record()
    assert (record["cross_tile_nms"], record["max_dets"], record["conf"]) == (
        DEFAULT_NMS_IOU, DEFAULT_MAX_DETS, DEFAULT_CONF)
    assert {record["sources"][k] for k in ("cross_tile_nms", "max_dets", "conf")} == {"explicit"}


def test_an_unstated_merge_threshold_is_derived_and_a_stated_one_is_kept(tmp_path):
    """The derivation an unstated merge threshold gets from the reference's ground truth is the
    point of leaving it unstated; a stated one survives the same derivation untouched."""
    unstated = _pass(tmp_path, tile=True, tile_size=64)
    stated = _pass(tmp_path, tile=True, tile_size=64, cross_tile_nms=0.42)

    unstated.derive_merge(OVERLAPPING_GT)
    stated.derive_merge(OVERLAPPING_GT)

    assert unstated.execution.sources["cross_tile_nms"] not in ("default", "explicit")
    assert "neighbor-IoU" in unstated.execution.sources["cross_tile_nms"]
    assert (stated.execution.cross_tile_nms, stated.execution.sources["cross_tile_nms"]) == (
        0.42, "explicit")


def _in_model_point(p, tmp_path: Path) -> dict:
    """The operating point the pass's model holds after predicting one frame."""
    from PIL import Image

    from tcip_mcp.pipelines.operating_point import (
        OPERATING_POINT_ATTRS, detector_operating_point_holder,
    )

    Image.new("RGB", (64, 64)).save(tmp_path / "frame.png")
    p.predict([str(tmp_path / "frame.png")])
    holder, _path = detector_operating_point_holder(p.predictor.model)
    return {attr: getattr(holder, attr) for attr in OPERATING_POINT_ATTRS}


def test_unstated_values_run_the_pass_at_the_documented_defaults(tmp_path):
    """Unstated is not uncapped: the predictor runs at the documented cap and conf, recorded as
    defaults, and an untiled pass records no merge threshold at all."""
    from tcip_mcp.pipelines.execution import DEFAULT_CONF, DEFAULT_MAX_DETS

    p = _pass(tmp_path, tile=False)

    assert _in_model_point(p, tmp_path) == {"score_thresh": DEFAULT_CONF,
                                            "detections_per_img": DEFAULT_MAX_DETS}
    assert (p.execution.sources["max_dets"], p.execution.sources["conf"]) == ("default",
                                                                              "default")
    assert p.execution.cross_tile_nms is None


def test_the_record_a_prediction_is_handed_governs_the_model_it_runs(tmp_path):
    """The execution record is the one authority over a prediction: the pass's own record sets
    the model's cap and conf, and a record handed to one prediction sets them for it."""
    p = _pass(tmp_path, tile=False, max_dets=77)

    assert _in_model_point(p, tmp_path) == {"score_thresh": p.execution.conf,
                                            "detections_per_img": 77}

    staged = p.execution.with_value("conf", 0.01, "staged").with_value("max_dets", 5, "fit")
    p.predict([str(tmp_path / "frame.png")], execution=staged)
    from tcip_mcp.pipelines.operating_point import detector_operating_point_holder

    holder, _path = detector_operating_point_holder(p.predictor.model)
    assert (holder.score_thresh, holder.detections_per_img) == (0.01, 5)


def test_the_public_inference_and_assessment_tools_agree_that_an_unstated_value_is_none():
    """One sentinel across the doors: every door takes the one stated mapping, unstated by
    default, and that mapping leaves every value unstated until a caller states it."""
    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.tools.calibration_tools import assess_checkpoint, assess_reserved_regions
    from tcip_mcp.tools.inference_tools import run_inference

    for tool in (run_inference, assess_checkpoint, assess_reserved_regions):
        assert inspect.signature(tool).parameters["stated"].default is None, tool.__name__
    assert set(Stated().model_dump().values()) == {None}


def test_the_dry_run_report_shows_the_values_the_pass_will_run_at(tmp_path):
    """An omitted conf and cap are None on the wire; the dry-run report shows the values the pass
    will actually run at, over images and over a raster alike."""
    from PIL import Image

    from tcip_mcp.pipelines.execution import DEFAULT_CONF, DEFAULT_MAX_DETS, Stated
    from tcip_mcp.tools.inference_tools import run_inference
    from tests._verified_checkpoint_fixtures import project_checkpoint

    checkpoint = project_checkpoint(tmp_path)
    images_dir = tmp_path / "images" / "undated"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64)).save(images_dir / "a.png")
    raster = tmp_path / "ortho" / "images" / "undated" / "mosaic.tif"
    raster.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64)).save(raster)

    over_images = run_inference(tmp_path, checkpoint_path=checkpoint, images_dir=str(images_dir),
                                bucket="out/2026-01-01", dry_run=True)
    over_raster = run_inference(tmp_path, checkpoint_path=checkpoint, raster_path=str(raster),
                                bucket="out/2026-01-01", stated=Stated(tile_size=32),
                                dry_run=True)

    import tcip_store

    from tcip_mcp.dataset_layout import bucket_key

    for result in (over_images, over_raster):
        assert "error" not in result, result
        assert result["execution"]["conf"] == DEFAULT_CONF
        assert result["execution"]["max_dets"] == DEFAULT_MAX_DETS
    for root in (tmp_path, tmp_path / "ortho"):
        assert not tcip_store.exists(bucket_key(root, "out/2026-01-01"))
