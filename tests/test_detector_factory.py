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


def test_detection_head_markers_raise_when_run_standalone():
    # 2.1b: the dead standalone-detector path is gone; the head markers (which exist only
    # for name validation / spec kwargs) fail loudly if anyone runs them directly.
    from tcip_mcp.pipelines.components.heads import AnchorDetectionHead, AnchorFreeDetectionHead
    for cls in (AnchorDetectionHead, AnchorFreeDetectionHead):
        with pytest.raises(NotImplementedError):
            cls(in_channels=256, num_classes=1).forward(None)


def test_composed_and_detection_models_share_base_and_inherit_freeze():
    from tcip_mcp.pipelines.composer import (
        ComposedModel, DetectionModel, _ComposedModule, compose_model,
    )
    det = compose_model(_det_spec("faster_rcnn"))
    assert isinstance(det, (_ComposedModule, DetectionModel))
    det.freeze_backbone(2)  # inherited from the shared base

    clf = compose_model({
        "backbone": {"name": "tv_resnet50"},
        "neck": {"name": "gap"},
        "heads": [{"name": "classification", "num_classes": 3}],
    })
    assert isinstance(clf, (_ComposedModule, ComposedModel))
    clf.freeze_backbone(1)
