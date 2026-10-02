"""Where a detector's operating-point knobs live: the module itself, its ``.detector.roi_heads``,
or its ``.detector``, resolved independently of what a given call actually applies.

A module exposing a knob on itself, with no ``.detector`` to route through, reaches a validated
operating point via the ``getattr`` chain falling through to the module itself. The interface is
stated once (:func:`~tcip_mcp.pipelines.operating_point.detector_operating_point_holder`), read
by both the setter and the model contract, and the two unstated-floor producers (a module with no
knob, the review route's own unknowns) share one gate name distinct from a stated floor the pick
does not clear.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")

IMG = 32
# The width and count a three-band, one-class detector run is built at.
_DIMS = {"in_chans": 3, "num_classes": 1}


def test_holder_resolves_to_the_module_itself_with_no_detector():
    from tcip_mcp.pipelines.operating_point import detector_operating_point_holder

    model = SimpleNamespace(score_thresh=0.5)
    holder, path = detector_operating_point_holder(model)
    assert holder is model and path == "self"


def test_holder_resolves_to_detector_roi_heads_for_a_two_stage_wrapper():
    from tcip_mcp.pipelines.operating_point import detector_operating_point_holder

    roi_heads = SimpleNamespace(score_thresh=0.5)
    model = SimpleNamespace(detector=SimpleNamespace(roi_heads=roi_heads))
    holder, path = detector_operating_point_holder(model)
    assert holder is roi_heads and path == "detector.roi_heads"


def test_holder_resolves_to_detector_itself_for_a_one_stage_wrapper():
    from tcip_mcp.pipelines.operating_point import detector_operating_point_holder

    detector = SimpleNamespace(score_thresh=0.5)
    model = SimpleNamespace(detector=detector)
    holder, path = detector_operating_point_holder(model)
    assert holder is detector and path == "detector"


def test_holder_is_none_when_nothing_exposes_a_knob():
    from tcip_mcp.pipelines.operating_point import detector_operating_point_holder

    model = SimpleNamespace(unrelated=1)
    holder, path = detector_operating_point_holder(model)
    assert holder is None and path is None


def test_holder_refuses_when_the_module_and_its_detectors_roi_heads_both_expose_a_knob():
    """An ambiguous module, ambiguous in the same shape a real bespoke wrapper could build: a
    torchvision two-stage detector under ``.detector`` (whose ``roi_heads`` already exposes the
    knob) plus a knob restated on the wrapper itself. The platform must not silently pick one."""
    from tcip_mcp.pipelines.model_build import build_model

    model = build_model({"model_source": {
        "builder": "tests.bespoke_models:build_bespoke_detection",
        "builder_kwargs": {"min_size": 64, "max_size": 128},
        "task": "detection"}}, _DIMS)
    assert hasattr(model.detector, "roi_heads") and hasattr(model.detector.roi_heads, "score_thresh")
    model.score_thresh = 0.5  # restated on the wrapper itself, ambiguous with .detector.roi_heads

    from tcip_mcp.pipelines.operating_point import detector_operating_point_holder

    with pytest.raises(ValueError, match="more than one location"):
        detector_operating_point_holder(model)


def test_set_detector_operating_point_returns_the_attribute_path():
    from tcip_mcp.pipelines.operating_point import set_detector_operating_point

    model = SimpleNamespace(score_thresh=0.5)
    applied, attribute_path = set_detector_operating_point(model, score_thresh=0.2)
    assert applied["score_thresh"] == 0.2
    assert attribute_path == "self"


def test_set_detector_operating_point_reports_no_path_when_nothing_matches():
    from tcip_mcp.pipelines.operating_point import set_detector_operating_point

    model = SimpleNamespace(unrelated=1)
    applied, attribute_path = set_detector_operating_point(model, score_thresh=0.2)
    assert applied.get("score_thresh") is None
    assert attribute_path is None


def _checkpoint(tmp_path, builder: str) -> str:
    from tests._verified_checkpoint_fixtures import foreign_checkpoint

    return foreign_checkpoint(
        tmp_path, name=builder,
        model_source={"builder": f"tests.bespoke_models:{builder}", "task": "detection"},
        data={"tiling": {"enabled": False}, "num_channels": 3,
              "scope": {"subject": "bud"}})


def _assessed(tmp_path: Path, builder: str) -> dict:
    """The assessment of ``builder``'s checkpoint over a drawn selection of images, each its own
    size, labeled at exactly the box ``BareScoreThreshDetector``/``BareNoKnobDetector`` always
    predict for that size, so every image matches and no two collide on content."""
    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    from tcip_mcp.tools.data_tools import draw_splits
    from tests._chain_fixtures import assess, confirm_count_trait

    root = tmp_path / "ds"
    images_dir, labels_dir = root / "images" / "2-11-26", root / "annotations" / "2-11-26"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    for i, size in enumerate(range(32, 96, 8)):
        Image.new("RGB", (size, size), (100, 100, 100)).save(images_dir / f"img{i}.png")
        box = BBox(size * 0.25, size * 0.25, size * 0.75, size * 0.75)
        json_io.write_annotations(str(labels_dir / f"img{i}.json"),
                                  [Annotation(subject="bud", geometry=box)], size, size)
    selection_dir = tmp_path / "selection"
    drawn = draw_splits(tmp_path, str(root), output_path=str(selection_dir), subject="bud", seed=2,
                        train_ratio=0.25, val_ratio=0.25, calibration_ratio=0.25,
                        holdout_ratio=0.25)
    assert "error" not in drawn, drawn
    confirm_count_trait(tmp_path)
    record = assess(tmp_path, _checkpoint(tmp_path, builder), selection_dir, device="cpu",
                    tile=False)
    assert "error" not in record, record
    return record


def test_a_bespoke_module_exposing_its_own_knob_is_assessed_at_its_stated_floor(tmp_path):
    """A hand-rolled, non-torchvision module exposing score_thresh on itself is assessed at the
    staged floor applied there, and its record names the attribute path it was applied on."""
    record = _assessed(tmp_path, "build_bare_score_thresh_detector")

    assert record["criterion"]["count"]["staged_conf_floor_attribute_path"] == "self"
    assert "conf_floor_unstated" not in record["failures"]
    assert "conf_censored" not in record["failures"]


def test_a_module_exposing_no_knob_fails_unstated_not_censored(tmp_path):
    """A module exposing no operating-point knob under any recognized name has no floor the
    platform can state, and fails with conf_floor_unstated, never conf_censored."""
    record = _assessed(tmp_path, "build_bare_no_knob_detector")

    assert record["passed"] is False
    assert record["criterion"]["count"]["staged_conf_floor_attribute_path"] is None
    assert "conf_floor_unstated" in record["failures"]
    assert "conf_censored" not in record["failures"]


def test_model_contract_records_the_holders_own_knobs():
    from tcip_mcp.pipelines.model_build import build_model
    from tcip_mcp.pipelines.model_contract import check_model_contract

    with_knob = build_model({"model_source": {
        "builder": "tests.bespoke_models:build_bare_score_thresh_detector",
        "task": "detection"}}, _DIMS)
    without_knob = build_model({"model_source": {
        "builder": "tests.bespoke_models:build_bare_no_knob_detector",
        "task": "detection"}}, _DIMS)

    dims = {"in_chans": 3, "num_classes": 1, "img_size": 64}
    assert check_model_contract(with_knob, "detection", dims=dims)["operating_point_knobs"] == [
        "score_thresh"]
    assert check_model_contract(without_knob, "detection", dims=dims)["operating_point_knobs"] == []
