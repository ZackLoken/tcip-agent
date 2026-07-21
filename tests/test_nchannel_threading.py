"""Phase 3.1 — N-channel threading: sensor->in_chans, backbones accept in_chans, a
train-start channel guard, and dataset channel metadata. (Multi-channel *readers* are 3.2;
here the plumbing carries channel counts and validates them.)"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_backbone_accepts_in_chans():
    import timm

    from tcip_mcp.pipelines.components.backbones import BackboneWrapper

    _m = timm.create_model("resnet50", pretrained=False, features_only=True,
                           out_indices=(1, 2, 3, 4), in_chans=4)
    bb = BackboneWrapper(_m, _m.feature_info.channels())
    out = bb(torch.rand(1, 4, 64, 64))
    assert isinstance(out, dict) and len(out) == 4           # 4 feature stages
    assert bb.out_channels == [256, 512, 1024, 2048]


def test_compose_and_forward_4_channel_model():
    from tests import bespoke_models
    model = bespoke_models.build_bespoke_classifier(num_classes=3, in_chans=4)
    out = model(torch.rand(2, 4, 64, 64))
    assert isinstance(out, dict) and len(out) > 0            # forward runs on 4 channels


def test_train_start_channel_guard():
    from tcip_mcp.pipelines.training.generic_trainer import _validate_input_channels
    with pytest.raises(ValueError, match="channels"):
        _validate_input_channels({"model_source": {"builder": "x:y", "in_chans": 4}},
                                 [(torch.rand(2, 3, 16, 16), {})])
    # matching channel counts -> no error
    _validate_input_channels({"model_source": {"builder": "x:y", "in_chans": 3}},
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
