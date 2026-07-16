"""Phase 3.1 — N-channel threading: sensor->in_chans, backbones accept in_chans, a
train-start channel guard, and dataset channel metadata. (Multi-channel *readers* are 3.2;
here the plumbing carries channel counts and validates them.)"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_recommend_sets_in_chans_from_sensor():
    from tcip_mcp.pipelines.composer import recommend_model_spec
    rgb = recommend_model_spec("classification", 300, sensor="rgb", num_classes=2)
    assert "in_chans" not in rgb["backbone"]                 # RGB stays the default (3)
    ms = recommend_model_spec("classification", 300, sensor="multispectral", num_classes=2)
    assert ms["backbone"]["in_chans"] == 5
    depth = recommend_model_spec("classification", 300, sensor="depth", num_classes=2)
    assert depth["backbone"]["in_chans"] == 1


def test_tv_backbone_accepts_in_chans():
    import tcip_mcp.pipelines.components.backbones  # noqa: F401
    from tcip_mcp.pipelines.registry import BACKBONES

    bb = BACKBONES.build("tv_resnet50", pretrained=False, in_chans=4)
    out = bb(torch.rand(1, 4, 64, 64))
    assert isinstance(out, dict) and len(out) == 4           # 4 feature stages
    assert BACKBONES.supports_channels("tv_resnet50", 4) is True


def test_compose_and_forward_4_channel_model():
    from tcip_mcp.pipelines.composer import compose_model
    model = compose_model({
        "backbone": {"name": "tv_resnet50", "pretrained": False, "in_chans": 4},
        "neck": {"name": "gap"},
        "heads": [{"name": "classification", "num_classes": 3}],
    })
    out = model(torch.rand(2, 4, 64, 64))
    assert isinstance(out, dict) and len(out) > 0            # forward runs on 4 channels


def test_train_start_channel_guard():
    from tcip_mcp.pipelines.training.generic_trainer import _validate_input_channels
    with pytest.raises(ValueError, match="channels"):
        _validate_input_channels({"backbone": {"name": "x", "in_chans": 4}},
                                 [(torch.rand(2, 3, 16, 16), {})])
    # matching channel counts -> no error
    _validate_input_channels({"backbone": {"name": "x", "in_chans": 3}},
                             [(torch.rand(2, 3, 16, 16), {})])


def test_build_dataset_sets_expected_channels(tmp_path):
    from PIL import Image

    from tcip_annotation import json_io
    from tcip_annotation.state import BBox
    from tcip_mcp.pipelines.data.datasets import build_dataset
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    Image.new("RGB", (16, 16)).save(images_dir / "a.png")
    json_io.write_detect(str(labels_dir / "a.json"),
                         [BBox(6.4, 6.4, 9.6, 9.6, 0)], 16, 16, keep_empty=True)

    ds = build_dataset("detection", images_dir=str(images_dir), labels_dir=str(labels_dir),
                       num_classes=1, num_channels=4)
    assert ds.expected_channels == 4
