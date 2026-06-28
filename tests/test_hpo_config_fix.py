"""Phase 1.3 — HPO params reach the model_spec, model/model_spec alias, stage schema."""


def test_normalize_train_config_resolves_model_alias():
    from tcip_mcp.pipelines.schemas import normalize_train_config
    cfg = normalize_train_config({"model": {"backbone": {"name": "tv_resnet50"}}})
    assert cfg["model_spec"] == {"backbone": {"name": "tv_resnet50"}}
    # model_spec wins when both are present
    cfg2 = normalize_train_config({"model_spec": {"a": 1}, "model": {"b": 2}})
    assert cfg2["model_spec"] == {"a": 1}


def test_stage_spec_accepts_freeze_to_without_lr():
    from tcip_mcp.pipelines.schemas import StageSpec
    # The trainer's canonical stage shape (freeze_to + epochs, lr optional) must validate.
    s = StageSpec.model_validate({"freeze_to": -1, "epochs": 5})
    assert s.epochs == 5 and s.freeze_to == -1 and s.lr is None


def test_apply_hpo_params_varies_architecture_in_model_spec():
    from tcip_mcp.tools.training_tools import _apply_hpo_params
    base = {
        "model_spec": {
            "backbone": {"name": "tv_resnet50"},
            "neck": {"name": "fpn"},
            "heads": [{"name": "anchor_detection", "num_classes": 1, "detector": "faster_rcnn"}],
        },
        "training": {"batch_size": 2},
    }
    out = _apply_hpo_params(base, {
        "backbone": "tv_resnet101", "head": "fcos", "min_size": 800,
        "lr": 1e-3, "batch_size": 8, "weight_decay": 1e-3,
    })
    spec = out["model_spec"]
    assert spec["backbone"]["name"] == "tv_resnet101"      # backbone actually varied
    assert spec["heads"][0]["detector"] == "fcos"          # detector actually varied
    assert spec["heads"][0]["min_size"] == 800
    assert out["training"]["batch_size"] == 8
    assert out["training"]["weight_decay"] == 1e-3
    # lr builds a freeze_to schedule (the trainer reads freeze_to, not freeze_backbone)
    stages = out["training"]["stages"]
    assert [s["freeze_to"] for s in stages] == [-1, 2, 0]
    assert all("freeze_backbone" not in s for s in stages)
    # the base config must not be mutated
    assert base["model_spec"]["backbone"]["name"] == "tv_resnet50"


def test_apply_hpo_params_handles_string_backbone_and_model_alias():
    from tcip_mcp.tools.training_tools import _apply_hpo_params
    base = {"model": {"backbone": "tv_resnet50", "heads": [{"name": "anchor_detection"}]}}
    out = _apply_hpo_params(base, {"backbone": "tv_resnet101"})
    assert out["model_spec"]["backbone"] == {"name": "tv_resnet101"}
