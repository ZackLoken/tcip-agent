"""A torchvision detector's input scale and anchors: every builder takes them stated, the one
derivation gives them from a run's frame and its objects, the run's resolution records them,
and the platform hands them to the builder from that record alone.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tcip_mcp.pipelines.derivations import (  # noqa: E402
    DERIVATION_IMPLEMENTATIONS,
    Region,
    char_sizes,
    derive_detector_geometry,
    gt_aspect_ratios,
)
from tcip_mcp.pipelines.schemas import DetectorGeometry  # noqa: E402

GEOMETRY = DetectorGeometry.handed_names()


@pytest.mark.parametrize(
    "builder",
    [
        "_build_faster_rcnn",
        "_build_retinanet",
        "_build_mask_rcnn",
        "_build_fcos",
    ],
)
def test_every_detector_builder_takes_its_geometry_stated(builder):
    """No builder defaults its anchor base, its resize bounds or, but for the anchor-free FCOS,
    its aspect ratios, and a call leaving one out is refused by the signature.
    """
    from tcip_mcp.pipelines.components import detectors

    fn = getattr(detectors, builder)
    params = inspect.signature(fn).parameters
    taken = [name for name in GEOMETRY if name in params]
    assert taken == (
        [n for n in GEOMETRY if n != "aspect_ratios"]
        if builder == "_build_fcos"
        else list(GEOMETRY)
    )
    for name in taken:
        assert params[name].default is inspect.Parameter.empty, name
    assert (
        inspect.signature(detectors._anchor_sizes).parameters["base"].default
        is inspect.Parameter.empty
    )
    with pytest.raises(TypeError, match="min_size"):
        fn(
            object(),
            1,
            featmap_names=["0"],
            num_levels=1,
            **{n: 1 for n in taken if n != "min_size"},
        )


def _boxes(sizes_wh: list[tuple[float, float]]) -> np.ndarray:
    return np.array(
        [[2.0, 2.0, 2.0 + w, 2.0 + h] for w, h in sizes_wh], dtype=np.float32
    )


SIZES_WH = [
    (6, 9),
    (8, 12),
    (10, 10),
    (12, 20),
    (14, 14),
    (16, 30),
    (20, 24),
    (24, 40),
]
"""Object extents, width by height, spread across four doublings of their characteristic size."""


def test_the_geometry_agrees_with_numpy_and_the_aspect_ratio_derivation():
    """Agreement with numpy: the derived resize bounds are the frame's own edges, the four
    doubling anchor levels bracket the 10th to 90th percentile of the characteristic sizes
    ``numpy.quantile`` computes from the same boxes, and the ratios are ``gt_aspect_ratios``' own.
    """
    frame = (96, 64)
    regions = [
        Region(_boxes(SIZES_WH), np.ones(len(SIZES_WH), np.int64), 96 * 64)
    ]
    geometry = derive_detector_geometry(frame, regions)

    values = geometry.handed()
    assert (values["min_size"], values["max_size"]) == (64, 96)
    sizes = np.sqrt([w * h for w, h in SIZES_WH])
    low, high = np.quantile(sizes, [0.1, 0.9])
    levels = [geometry.anchor_base_size * 2**i for i in range(4)]
    assert levels[0] <= low and levels[-1] >= high
    assert char_sizes([regions[0].boxes]) == pytest.approx(sizes.tolist())
    assert list(values["aspect_ratios"]) == gt_aspect_ratios(SIZES_WH)
    assert DERIVATION_IMPLEMENTATIONS[str(geometry.source)] == (
        "tcip_mcp.pipelines.derivations.derive_detector_geometry"
    )


def test_a_detector_built_at_the_derived_geometry_sees_a_small_tile_at_its_native_scale():
    """A 64 px tile reaches the detector's transform at 64 px: the derived bounds are the
    tile's, where a fixed 800 px resize would scale it twelvefold.
    """
    from tests import bespoke_models

    tile = (64, 64)
    regions = [Region(_boxes(SIZES_WH[:4]), np.ones(4, np.int64), 64 * 64)]
    values = derive_detector_geometry(tile, regions).handed()
    model = bespoke_models.build_bespoke_detection(
        num_classes=1, image_mean=[0.5] * 3, image_std=[0.25] * 3, **values
    )
    model.eval()

    image_list, _ = model.detector.transform([torch.rand(3, 64, 64)])

    assert image_list.image_sizes == [(64, 64)]


def test_regions_holding_no_object_leave_a_geometry_its_type_refuses():
    """No object derives no anchor: the record's own type refuses the absence."""
    from pydantic import ValidationError

    empty = [Region(np.zeros((0, 4), np.float32), np.zeros(0, np.int64), 64.0)]
    with pytest.raises(ValidationError, match="anchor_base_size"):
        derive_detector_geometry((8, 8), empty)


def test_a_geometry_sourced_by_any_label_but_the_derivations_is_refused():
    """The derivation's own label is the one derived source the record admits: the record
    re-validates under it, and the same values under another label are refused by the type.
    """
    from pydantic import ValidationError

    from tests._chain_fixtures import detector_geometry

    record = detector_geometry().model_dump()
    assert DetectorGeometry.model_validate(record) == detector_geometry()
    with pytest.raises(ValidationError, match="source"):
        DetectorGeometry.model_validate(
            {**record, "source": "not the derivation's label"}
        )


def _preflight_config(tmp_path, **model_source):
    from tests._verified_checkpoint_fixtures import (
        BUILT_DETECTOR,
        detection_config,
    )

    return detection_config(
        tmp_path / "ds",
        project=tmp_path,
        model_source={**BUILT_DETECTOR, **model_source},
    )


@pytest.mark.parametrize("restated", GEOMETRY)
def test_preflight_refuses_a_builder_kwarg_restating_the_recorded_geometry(
    tmp_path, restated
):
    """The geometry reaches the builder from the run's resolution alone: ``builder_kwargs``
    naming any of the four refuses at the launch door, naming it.
    """
    from tcip_mcp.tools.training_tools import preflight_config

    r = preflight_config(
        tmp_path, _preflight_config(tmp_path, builder_kwargs={restated: 64})
    )

    assert r["valid"] is False
    assert any(
        restated in i and "detector's geometry" in i for i in r["issues"]
    ), r["issues"]


def test_the_launch_builds_the_detector_at_the_geometry_its_resolution_recorded(
    tmp_path, monkeypatch
):
    """Admits valid work: a config the chain fixtures build passes the launch door's smoke, the
    geometry its resolution derived is the one ``derive_detector_geometry`` gives at the frame its
    frames share over its train regions, and the model it builds holds that geometry's bounds and
    anchors.
    """
    from tcip_mcp.pipelines import model_build
    from tcip_mcp.tools.training_tools import preflight_config

    built: list = []
    real = model_build.build_from_model_source

    def record(source, layout, dims, geometry):
        built.append((real(source, layout, dims, geometry), geometry))
        return built[-1][0]

    monkeypatch.setattr(model_build, "build_from_model_source", record)

    r = preflight_config(tmp_path, _preflight_config(tmp_path), smoke=True)

    assert r["valid"] is True, r["issues"]
    model, geometry = built[-1]
    assert geometry.native_size == (64, 48)
    detector = geometry.detector
    assert (detector.min_size, detector.max_size) == (48, 64)
    transform = model.detector.transform
    assert (transform.min_size[0], transform.max_size) == (48, 64)
    sizes = model.detector.rpn.anchor_generator.sizes
    assert [level[0] for level in sizes] == [
        detector.anchor_base_size * 2**i for i in range(len(sizes))
    ]
