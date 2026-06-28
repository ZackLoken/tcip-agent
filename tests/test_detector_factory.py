"""Phase 2.1a — DETECTORS registry / DetectorFactory: registry-driven detector
construction, so a new detector is added by registering a builder, not by editing
DetectionModel."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


def _det_spec(detector):
    return {
        "backbone": {"name": "tv_resnet50"},
        "neck": {"name": "fpn", "out_channels": 64},
        "heads": [{"name": "anchor_detection", "num_classes": 1, "detector": detector}],
    }


def test_builtin_detectors_registered():
    import tcip_mcp.pipelines.components.detectors  # noqa: F401
    from tcip_mcp.pipelines.registry import DETECTORS
    for name in ("faster_rcnn", "fcos", "retinanet"):
        assert name in DETECTORS
        assert DETECTORS.describe(name)["valid_tasks"] == ["detection"]


@pytest.mark.parametrize("detector", ["faster_rcnn", "fcos", "retinanet"])
def test_compose_builds_each_builtin_detector(detector):
    from tcip_mcp.pipelines.composer import DetectionModel, compose_model
    model = compose_model(_det_spec(detector))
    assert isinstance(model, DetectionModel)
    model.eval()
    out = model([torch.rand(3, 64, 64)])
    assert isinstance(out, list) and "boxes" in out[0]


def test_register_external_detector_and_compose():
    from tcip_mcp.pipelines.components.detectors import _build_fcos, register_external_detector
    from tcip_mcp.pipelines.composer import DetectionModel, compose_model
    from tcip_mcp.pipelines.registry import restore_all, snapshot_all

    snap = snapshot_all()
    try:
        # An external detector registered from "outside" (here it delegates to FCOS) —
        # proves a new detector composes without touching DetectionModel.
        def my_detector(adapter, num_classes, *, featmap_names, num_levels, **kw):
            return _build_fcos(
                adapter, num_classes, featmap_names=featmap_names, num_levels=num_levels, **kw)

        register_external_detector(
            "my_detector", my_detector, metadata={"valid_tasks": ["detection"]})
        model = compose_model(_det_spec("my_detector"))
        assert isinstance(model, DetectionModel)
    finally:
        restore_all(snap)


def test_build_detector_unknown_name_raises():
    from tcip_mcp.pipelines.components.detectors import build_detector
    with pytest.raises(KeyError):
        build_detector("does_not_exist", object(), 1, featmap_names=["0"], num_levels=1)
