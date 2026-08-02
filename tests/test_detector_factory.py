"""Detector builders: plain ``build_detector`` / ``_build_*`` factories, imported
directly by bespoke model code (no registry)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tests import bespoke_models  # noqa: E402


@pytest.mark.parametrize("detector", ["faster_rcnn", "fcos", "retinanet"])
def test_build_each_builtin_detector(detector):
    model = bespoke_models.build_bespoke_detection(
        num_classes=1, detector=detector, min_size=64, max_size=128)
    model.eval()
    out = model([torch.rand(3, 64, 64)])
    assert isinstance(out, list) and "boxes" in out[0]
    # train-forward: each detector returns a finite loss dict.
    model.train()
    target = [{"boxes": torch.tensor([[10.0, 10.0, 40.0, 40.0]]), "labels": torch.tensor([1])}]
    loss = model([torch.rand(3, 64, 64)], target)
    assert isinstance(loss, dict) and loss
    assert torch.isfinite(sum(loss.values()))


def test_build_detector_unknown_name_raises():
    from tcip_mcp.pipelines.components.detectors import build_detector
    with pytest.raises(KeyError):
        build_detector("does_not_exist", object(), 1, featmap_names=["0"], num_levels=1)
