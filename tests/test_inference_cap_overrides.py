"""The merge-threshold, detection-cap and conf values of a pass: each is the stated one, else the
one a reference the caller holds derives, else a refusal naming it; a stated value is honored at
every value it can take, and a stated cap reaches the predictor, not only the record.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

OVERLAPPING_GT = [[[0.0, 0.0, 20.0, 20.0], [10.0, 0.0, 20.0, 20.0], [60.0, 60.0, 20.0, 20.0]]] * 2
"""Two images of overlapping ground truth (xywh), enough for a real cross-tile merge
derivation."""


def _checkpoint(tmp_path: Path):
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tests._verified_checkpoint_fixtures import project_checkpoint

    return load_registered_checkpoint(project_checkpoint(tmp_path), project=tmp_path)


def _pass(tmp_path: Path, reference=None, **stated):
    """The pass a registered checkpoint of ``tmp_path`` runs under ``stated``, made runnable from
    ``reference`` when one is given."""
    from tcip_mcp.pipelines.execution import Stated, prepare

    return prepare(_checkpoint(tmp_path), Stated(**stated), device="cpu").runnable(reference)


def test_stated_values_are_recorded_as_explicit(tmp_path):
    """The value a caller states is the value that runs, recorded as stated."""
    from tests._verified_checkpoint_fixtures import SAMPLE_DETECTOR_PASS

    p = _pass(tmp_path, tile=True, tile_size=64, **SAMPLE_DETECTOR_PASS)

    record = p.execution.record()
    assert {k: record[k] for k in SAMPLE_DETECTOR_PASS} == SAMPLE_DETECTOR_PASS
    assert {record["sources"][k] for k in SAMPLE_DETECTOR_PASS} == {"explicit"}


def test_unstated_merge_threshold_and_cap_derive_from_the_reference_and_stated_ones_are_kept(
        tmp_path):
    """A reference's boxes derive an unstated merge threshold and its counted density, over the
    published frame's footprint, an unstated cap; a stated one of each survives the same
    reference untouched, and counts with no known footprint derive no cap and refuse naming it."""
    from tcip_mcp.pipelines.derivations import CROSS_TILE_NMS_DERIVATION, MAX_DETS_DERIVATION
    from tcip_mcp.pipelines.execution import ExecutionRefusedError, Reference
    from tests._verified_checkpoint_fixtures import SAMPLE_CONF

    frame = 100.0 * 100.0
    reference = Reference(boxes_per_image=OVERLAPPING_GT, counted=[(3, frame)] * 2,
                          footprint=frame)
    with pytest.raises(ExecutionRefusedError, match="max_dets"):
        _pass(tmp_path, Reference(boxes_per_image=OVERLAPPING_GT, counted=[(3, frame)] * 2,
                                  footprint=None), tile=True, tile_size=64, conf=SAMPLE_CONF)
    unstated = _pass(tmp_path, reference, tile=True, tile_size=64, conf=SAMPLE_CONF)
    stated = _pass(tmp_path, reference, tile=True, tile_size=64, conf=SAMPLE_CONF,
                   cross_tile_nms=0.42, max_dets=7)

    assert unstated.execution.sources["cross_tile_nms"] == CROSS_TILE_NMS_DERIVATION
    assert (unstated.execution.max_dets, unstated.execution.sources["max_dets"]) == (
        5, MAX_DETS_DERIVATION)
    assert (stated.execution.cross_tile_nms, stated.execution.max_dets) == (0.42, 7)
    assert {stated.execution.sources[k] for k in ("cross_tile_nms", "max_dets")} == {"explicit"}


@pytest.mark.parametrize("unstated", ["conf", "max_dets"])
def test_a_detector_pass_refuses_an_unstated_conf_or_cap_by_name(tmp_path, unstated):
    """No default stands behind a detector's conf or cap: a pass with no reference states both,
    and one left unstated refuses naming it."""
    from tcip_mcp.pipelines.execution import ExecutionRefusedError
    from tests._verified_checkpoint_fixtures import SAMPLE_CONF, SAMPLE_MAX_DETS

    stated = {"conf": SAMPLE_CONF, "max_dets": SAMPLE_MAX_DETS}
    del stated[unstated]
    with pytest.raises(ExecutionRefusedError, match=unstated):
        _pass(tmp_path, tile=False, **stated)


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


def test_the_record_a_prediction_is_handed_governs_the_model_it_runs(tmp_path):
    """The execution record is the one authority over a prediction: the pass's own record sets
    the model's cap and conf, and a record handed to one prediction sets them for it."""
    from tests._verified_checkpoint_fixtures import SAMPLE_CONF

    p = _pass(tmp_path, tile=False, conf=SAMPLE_CONF, max_dets=77)

    assert _in_model_point(p, tmp_path) == {"score_thresh": SAMPLE_CONF,
                                            "detections_per_img": 77}
    assert p.execution.cross_tile_nms is None

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


def test_the_dry_run_reports_the_stated_values_and_refuses_an_unstated_conf(tmp_path):
    """The dry-run report shows the values the pass will run at, over images and over a raster
    alike; a dry run stating no conf refuses as the pass would."""
    from PIL import Image

    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.tools.inference_tools import run_inference
    from tests._verified_checkpoint_fixtures import (
        SAMPLE_CONF, SAMPLE_DETECTOR_PASS, SAMPLE_MAX_DETS, project_checkpoint,
    )

    checkpoint = project_checkpoint(tmp_path)
    images_dir = tmp_path / "images" / "undated"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64)).save(images_dir / "a.png")
    raster = tmp_path / "ortho" / "images" / "undated" / "mosaic.tif"
    raster.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64)).save(raster)

    over_images = run_inference(tmp_path, checkpoint_path=checkpoint, images_dir=str(images_dir),
                                bucket="out/2026-01-01", dry_run=True,
                                stated=Stated(tile=False, conf=SAMPLE_CONF,
                                              max_dets=SAMPLE_MAX_DETS))
    over_raster = run_inference(tmp_path, checkpoint_path=checkpoint, raster_path=str(raster),
                                bucket="out/2026-01-01", dry_run=True,
                                stated=Stated(tile_size=32, **SAMPLE_DETECTOR_PASS))
    unstated = run_inference(tmp_path, checkpoint_path=checkpoint, images_dir=str(images_dir),
                             bucket="out/2026-01-01", dry_run=True,
                             stated=Stated(tile=False, max_dets=SAMPLE_MAX_DETS))

    import tcip_store

    from tcip_mcp.dataset_layout import bucket_key

    for result in (over_images, over_raster):
        assert "error" not in result, result
        assert (result["execution"]["conf"], result["execution"]["max_dets"]) == (
            SAMPLE_CONF, SAMPLE_MAX_DETS)
    assert "conf" in unstated.get("error", ""), unstated
    for root in (tmp_path, tmp_path / "ortho"):
        assert not tcip_store.exists(bucket_key(root, "out/2026-01-01"))
