"""Where a detector's operating-point knobs live, resolved apart from what a call applies.

The knobs live on the module itself, its ``.detector.roi_heads``, or its ``.detector``. A module
exposing its knobs on itself, with no ``.detector`` to route through, reaches a validated operating
point through :func:`~tcip_mcp.pipelines.operating_point.detector_operating_point_holder`, and a
knob set on a model exposing none refuses naming it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from tests._chain_fixtures import (
    BARE_NO_KNOB_DETECTOR,
    BARE_SCORE_THRESH_DETECTOR,
    BESPOKE_MODELS,
    FRAME_STATISTICS,
)
from tests._training_values import drawn_split

if TYPE_CHECKING:
    from torch import nn

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")

IMG = 32
# The width and count a three-band, one-class detector run is built at.
_DIMS = {"in_chans": 3, "num_classes": 1}


def _built(model_source: dict) -> nn.Module:
    """Return the model ``model_source`` builds at :data:`_DIMS` and a 32 px frame's geometry.

    The builder is imported from the staged sources of a run config over one noise frame of its
    own with this repository as its project.
    """
    import tempfile

    from tcip_mcp.pipelines.model_build import (
        build_from_model_source,
        staged_sources,
    )
    from tcip_mcp.pipelines.schemas import TrainGeometry, train_config
    from tests import REPO_ROOT
    from tests._chain_fixtures import (
        detector_geometry,
        frame_statistics,
        training_config,
    )
    from tests._image_fixtures import write_noise_image

    with tempfile.TemporaryDirectory() as frames:
        frame = Path(frames) / "frame.png"
        write_noise_image(frame, IMG)
        spec = train_config(
            training_config(
                Path(frames),
                {
                    **model_source,
                    "normalization": frame_statistics([str(frame)], 3),
                },
                {"images_dir": frames, "split": drawn_split()},
            )
        )
    return build_from_model_source(
        spec.model_source,
        staged_sources(spec, REPO_ROOT).layout,
        _DIMS,
        TrainGeometry(
            native_size=None,
            object_density=None,
            detector=detector_geometry((IMG, IMG)),
        ),
    )


def test_holder_resolves_to_the_module_itself_with_no_detector() -> None:
    from tcip_mcp.pipelines.operating_point import (
        detector_operating_point_holder,
    )

    model = SimpleNamespace(score_thresh=0.5)
    holder, path = detector_operating_point_holder(model)
    assert holder is model and path == "self"


def test_holder_resolves_to_detector_roi_heads_for_a_two_stage_wrapper() -> (
    None
):
    from tcip_mcp.pipelines.operating_point import (
        detector_operating_point_holder,
    )

    roi_heads = SimpleNamespace(score_thresh=0.5)
    model = SimpleNamespace(detector=SimpleNamespace(roi_heads=roi_heads))
    holder, path = detector_operating_point_holder(model)
    assert holder is roi_heads and path == "detector.roi_heads"


def test_holder_resolves_to_detector_itself_for_a_one_stage_wrapper() -> None:
    from tcip_mcp.pipelines.operating_point import (
        detector_operating_point_holder,
    )

    detector = SimpleNamespace(score_thresh=0.5)
    model = SimpleNamespace(detector=detector)
    holder, path = detector_operating_point_holder(model)
    assert holder is detector and path == "detector"


def test_holder_is_none_when_nothing_exposes_a_knob() -> None:
    from tcip_mcp.pipelines.operating_point import (
        detector_operating_point_holder,
    )

    model = SimpleNamespace(unrelated=1)
    holder, path = detector_operating_point_holder(model)
    assert holder is None and path is None


def test_holder_refuses_when_the_module_and_its_detectors_roi_heads_both_expose_a_knob() -> (
    None
):
    """The holder refuses a module whose knob sits both on itself and on its detector's heads.

    The module is a torchvision two-stage detector under ``.detector`` (whose ``roi_heads``
    exposes the knob) plus a knob restated on the wrapper itself, and the holder refuses naming
    more than one location.
    """
    from tcip_mcp.pipelines.operating_point import (
        SCORE_THRESH,
        detector_operating_point_holder,
    )
    from tests._verified_checkpoint_fixtures import BUILT_DETECTOR

    model = _built(BUILT_DETECTOR)
    assert hasattr(model.detector, "roi_heads")
    assert hasattr(model.detector.roi_heads, SCORE_THRESH)
    model.score_thresh = 0.5  # restated on the wrapper itself, ambiguous with .detector.roi_heads

    with pytest.raises(ValueError, match="more than one location"):
        detector_operating_point_holder(model)


def test_set_detector_operating_point_sets_each_knob_on_the_holder() -> None:
    """Each knob lands on its own attribute of a built detector's holder."""
    from tcip_mcp.pipelines.operating_point import set_detector_operating_point
    from tests._verified_checkpoint_fixtures import built_detector

    model = _built(built_detector())
    set_detector_operating_point(model, score_thresh=0.2, detections_per_img=3)
    holder = model.detector.roi_heads
    assert (holder.score_thresh, holder.detections_per_img) == (0.2, 3)


@pytest.mark.parametrize(
    "model, missing",
    [
        (SimpleNamespace(unrelated=1), "score_thresh"),
        (SimpleNamespace(score_thresh=0.5), "detections_per_img"),
        (SimpleNamespace(detections_per_img=100), "score_thresh"),
    ],
    ids=["no knob", "no cap knob", "no score knob"],
)
def test_set_detector_operating_point_refuses_a_knob_the_model_does_not_expose(
    model: SimpleNamespace, missing: str
) -> None:
    from tcip_mcp.pipelines.operating_point import set_detector_operating_point

    with pytest.raises(ValueError, match=f"exposes no {missing} "):
        set_detector_operating_point(
            model, score_thresh=0.2, detections_per_img=3
        )


def _checkpoint(tmp_path: Path, builder: str) -> str:
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    return registered_checkpoint(
        tmp_path,
        model_source={
            "builder": builder,
            "source_files": [BESPOKE_MODELS],
            "task": "detection",
            "normalization": FRAME_STATISTICS,
        },
        data={
            "num_channels": 3,
            "scope": {"subject": "bud"},
        },
    )


def _assessed(tmp_path: Path, builder: str) -> dict:
    """Assess ``builder``'s checkpoint over a drawn selection and return its record or error.

    The images are each their own size, each labeled at its central box spanning half its edge.
    """
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.tools.data_tools import draw_splits
    from tests._chain_fixtures import assess, confirm_count_trait
    from tests._producer_fixtures import label_image, write_image

    root = tmp_path / "ds"
    images_dir = image_dir(root, "2-11-26")
    for i, size in enumerate(range(32, 96, 8)):
        write_image(images_dir / f"img{i}.png", (size, size), (100, 100, 100))
        box = BBox(size * 0.25, size * 0.25, size * 0.75, size * 0.75)
        label_image(
            images_dir / f"img{i}.png",
            [Annotation(subject="bud", geometry=box)],
            size,
            size,
        )
    selection_dir = tmp_path / "selection"
    drawn = draw_splits(
        tmp_path,
        str(root),
        output_path=str(selection_dir),
        subject="bud",
        **drawn_split(
            seed=2,
            val_ratio=0.25,
            calibration_ratio=0.25,
            holdout_ratio=0.25,
            grouping="tile_prefix",
        ),
        actor=None,
    )
    assert "error" not in drawn, drawn
    confirm_count_trait(tmp_path)
    return assess(
        tmp_path,
        _checkpoint(tmp_path, builder),
        selection_dir,
        device="cpu",
        tile=False,
    )


def test_a_bespoke_module_exposing_its_own_knobs_is_assessed_at_its_stated_floor(
    tmp_path: Path,
) -> None:
    """A bespoke module exposing its own knobs is assessed, its floor path recorded as ``self``.

    A hand-rolled, non-torchvision module exposing both knobs on itself is assessed without
    error; the record names ``"self"`` as the floor's attribute path and reports no
    ``conf_censored`` failure.
    """
    record = _assessed(tmp_path, BARE_SCORE_THRESH_DETECTOR)

    assert "error" not in record, record
    assert (
        record["criterion"]["count"]["staged_conf_floor_attribute_path"]
        == "self"
    )
    assert "conf_censored" not in record["failures"]


def test_a_module_exposing_no_knob_refuses_naming_the_knob_it_lacks(
    tmp_path: Path,
) -> None:
    refused = _assessed(tmp_path, BARE_NO_KNOB_DETECTOR)

    assert "score_thresh" in refused.get("error", ""), refused
