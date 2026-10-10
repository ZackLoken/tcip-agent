"""Where a detector's operating-point knobs live: the module itself, its ``.detector.roi_heads``,
or its ``.detector``, resolved independently of what a given call actually applies.

A module exposing its knobs on itself, with no ``.detector`` to route through, reaches a validated
operating point via the ``getattr`` chain falling through to the module itself. The interface is
stated once (:func:`~tcip_mcp.pipelines.operating_point.detector_operating_point_holder`), and a
knob set on a model exposing none refuses naming it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests._chain_fixtures import (
    BARE_NO_KNOB_DETECTOR, BARE_SCORE_THRESH_DETECTOR, BESPOKE_MODELS,
)

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")

IMG = 32
# The width and count a three-band, one-class detector run is built at.
_DIMS = {"in_chans": 3, "num_classes": 1}


def _built(model_source: dict):
    """The model ``model_source`` builds at :data:`_DIMS`, imported from the staged sources of a
    run config over no data with this repository as its project."""
    from tcip_mcp.pipelines.model_build import build_from_model_source, staged_sources
    from tcip_mcp.pipelines.schemas import train_config
    from tests import REPO_ROOT
    from tests._chain_fixtures import training_config

    spec = train_config(training_config(model_source, {}))
    return build_from_model_source(spec.model_source,
                                   staged_sources(spec, REPO_ROOT).layout, _DIMS)


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
    from tests._verified_checkpoint_fixtures import BUILT_DETECTOR

    model = _built(BUILT_DETECTOR)
    assert hasattr(model.detector, "roi_heads")
    assert hasattr(model.detector.roi_heads, "score_thresh")
    model.score_thresh = 0.5  # restated on the wrapper itself, ambiguous with .detector.roi_heads

    from tcip_mcp.pipelines.operating_point import detector_operating_point_holder

    with pytest.raises(ValueError, match="more than one location"):
        detector_operating_point_holder(model)


def test_set_detector_operating_point_sets_each_knob_on_the_holder():
    from tcip_mcp.pipelines.operating_point import set_detector_operating_point

    model = SimpleNamespace(score_thresh=0.5, detections_per_img=100)
    set_detector_operating_point(model, score_thresh=0.2, detections_per_img=3)
    assert (model.score_thresh, model.detections_per_img) == (0.2, 3)


@pytest.mark.parametrize("model", [SimpleNamespace(unrelated=1),
                                   SimpleNamespace(score_thresh=0.5)],
                         ids=["no knob", "no cap knob"])
def test_set_detector_operating_point_refuses_a_knob_the_model_does_not_expose(model):
    from tcip_mcp.pipelines.operating_point import set_detector_operating_point

    with pytest.raises(ValueError, match="exposes no (score_thresh|detections_per_img)"):
        set_detector_operating_point(model, score_thresh=0.2, detections_per_img=3)


def _checkpoint(tmp_path, builder: str) -> str:
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    return registered_checkpoint(
        tmp_path,
        model_source={"builder": builder, "source_files": [BESPOKE_MODELS], "task": "detection"},
        data={"tiling": {"enabled": False}, "num_channels": 3,
              "scope": {"subject": "bud"}})


def _assessed(tmp_path: Path, builder: str) -> dict:
    """The assessment of ``builder``'s checkpoint over a drawn selection of images, each its own
    size, labeled at exactly the box ``BareScoreThreshDetector``/``BareNoKnobDetector`` always
    predict for that size, so every image matches and no two collide on content; its record, or
    the door's error."""
    from PIL import Image
    from tcip_annotation.state import Annotation, BBox

    from tcip_mcp.tools.data_tools import draw_splits
    from tests._chain_fixtures import assess, confirm_count_trait
    from tests._producer_fixtures import label_image

    root = tmp_path / "ds"
    images_dir = root / "images" / "2-11-26"
    images_dir.mkdir(parents=True)
    for i, size in enumerate(range(32, 96, 8)):
        Image.new("RGB", (size, size), (100, 100, 100)).save(images_dir / f"img{i}.png")
        box = BBox(size * 0.25, size * 0.25, size * 0.75, size * 0.75)
        label_image(images_dir / f"img{i}.png", [Annotation(subject="bud", geometry=box)],
                    size, size)
    selection_dir = tmp_path / "selection"
    drawn = draw_splits(tmp_path, str(root), output_path=str(selection_dir), subject="bud", seed=2,
                        val_ratio=0.25, calibration_ratio=0.25,
                        holdout_ratio=0.25)
    assert "error" not in drawn, drawn
    confirm_count_trait(tmp_path)
    return assess(tmp_path, _checkpoint(tmp_path, builder), selection_dir, device="cpu",
                  tile=False)


def test_a_bespoke_module_exposing_its_own_knobs_is_assessed_at_its_stated_floor(tmp_path):
    """A hand-rolled, non-torchvision module exposing both knobs on itself is assessed at the
    staged floor applied there, and its record names the attribute path it was applied on."""
    record = _assessed(tmp_path, BARE_SCORE_THRESH_DETECTOR)

    assert "error" not in record, record
    assert record["criterion"]["count"]["staged_conf_floor_attribute_path"] == "self"
    assert "conf_censored" not in record["failures"]


def test_a_module_exposing_no_knob_refuses_naming_the_knob_it_lacks(tmp_path):
    refused = _assessed(tmp_path, BARE_NO_KNOB_DETECTOR)

    assert "score_thresh" in refused.get("error", ""), refused
