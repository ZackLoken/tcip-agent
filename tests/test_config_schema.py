"""W7 — channel-compat validation + pydantic config schema."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")  # composer / validate_model_spec import torch

from tcip_mcp.pipelines.composer import validate_model_spec  # noqa: E402


def _spec(neck_name, head_name, head_extra=None, neck_extra=None):
    head = {"name": head_name, "num_classes": 2}
    head.update(head_extra or {})
    neck = {"name": neck_name}
    neck.update(neck_extra or {})
    return {"backbone": {"name": "tv_resnet50"}, "neck": neck, "heads": [head]}


# --------------------------------------------------------------------------
# Channel / format compatibility
# --------------------------------------------------------------------------

def test_channel_compat_gap_with_detection_rejected():
    issues = validate_model_spec(_spec("gap", "anchor_detection"))
    assert any("expects" in i.lower() for i in issues)  # flat-vector neck vs multi-scale head


def test_channel_compat_fpn_with_classification_rejected():
    assert validate_model_spec(_spec("fpn", "classification"))  # multi-scale neck vs flat head


def test_channel_compat_explicit_in_channels_mismatch():
    issues = validate_model_spec(_spec("fpn", "semantic_seg", head_extra={"in_channels": 128},
                                       neck_extra={"out_channels": 256}))
    assert any("in_channels" in i for i in issues)


def test_channel_compat_valid_specs_pass():
    assert validate_model_spec(_spec("gap", "classification")) == []
    assert validate_model_spec(_spec("fpn", "semantic_seg")) == []
    assert validate_model_spec({
        "backbone": {"name": "tv_resnet50"}, "neck": {"name": "fpn"},
        "heads": [{"name": "anchor_detection", "num_classes": 1}],
    }) == []


# --------------------------------------------------------------------------
# Pydantic schema
# --------------------------------------------------------------------------

def test_schema_rejects_bad_types_and_empty_heads():
    from tcip_mcp.pipelines.schemas import validate_train_config_schema
    good_spec = {"backbone": {"name": "tv_resnet50"}, "neck": {"name": "gap"},
                 "heads": [{"name": "classification", "num_classes": 2}]}
    assert validate_train_config_schema({"model_spec": good_spec, "training": {"batch_size": "big"}})
    assert validate_train_config_schema({"model_spec": {**good_spec, "heads": []}})
    assert validate_train_config_schema({"model_spec": good_spec, "training": {"batch_size": 4}}) == []


def test_validate_config_surfaces_channel_compat(tmp_path):
    from tcip_mcp.tools.training_tools import validate_config
    imgs = tmp_path / "images"
    lbls = tmp_path / "labels"
    imgs.mkdir()
    lbls.mkdir()
    cfg = {
        "model_spec": {"backbone": {"name": "tv_resnet50"}, "neck": {"name": "gap"},
                       "heads": [{"name": "anchor_detection", "num_classes": 1}]},
        "data": {"images_dir": str(imgs), "labels_dir": str(lbls)},
    }
    r = validate_config(cfg)
    assert r["valid"] is False
    assert any("expects" in i.lower() for i in r["issues"])

    r2 = validate_config({})  # empty config still flags model_spec
    assert any("model_spec" in i for i in r2["issues"])


def test_validate_pipeline_flags_bad_phase_model_spec():
    from tcip_mcp.pipelines.orchestrator import validate_pipeline
    bad = {"name": "p", "phases": [{
        "name": "train", "task": "detection",
        "model_spec": {"backbone": {"name": "tv_resnet50"}, "neck": {"name": "gap"},
                       "heads": [{"name": "anchor_detection", "num_classes": 1}]},
    }]}
    issues = validate_pipeline(bad)
    assert any("train" in i and "expects" in i.lower() for i in issues)

    # String-typed components must not crash the validator.
    string_spec = {"name": "p", "phases": [{
        "name": "train", "task": "detection",
        "model_spec": {"backbone": "resnet50", "neck": "fpn", "heads": []},
    }]}
    assert isinstance(validate_pipeline(string_spec), list)
