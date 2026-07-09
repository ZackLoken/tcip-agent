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
    # The trainer's canonical stage shape is freeze_to + epochs. Per-stage lr was removed
    # (L1: the optimizer block sets LR), but extra="allow" keeps an old lr-carrying stage valid.
    s = StageSpec.model_validate({"freeze_to": -1, "epochs": 5})
    assert s.epochs == 5 and s.freeze_to == -1
    assert "lr" not in StageSpec.model_fields


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
    # lr/weight_decay land in top-level optimizer — the only keys the trainer reads
    assert out["optimizer"]["head_lr"] == 1e-3
    assert out["optimizer"]["backbone_lr"] == 1e-4
    assert out["optimizer"]["weight_decay"] == 1e-3
    # lr builds a TOP-LEVEL freeze_to schedule (train() reads config["stages"], never
    # training["stages"]) with no dead per-stage lr keys
    stages = out["stages"]
    assert [s["freeze_to"] for s in stages] == [-1, 2, 0]
    assert all("lr" not in s and "freeze_backbone" not in s for s in stages)
    assert "stages" not in out["training"]
    # the base config must not be mutated
    assert base["model_spec"]["backbone"]["name"] == "tv_resnet50"


def test_apply_hpo_params_handles_string_backbone_and_model_alias():
    from tcip_mcp.tools.training_tools import _apply_hpo_params
    base = {"model": {"backbone": "tv_resnet50", "heads": [{"name": "anchor_detection"}]}}
    out = _apply_hpo_params(base, {"backbone": "tv_resnet101"})
    assert out["model_spec"]["backbone"] == {"name": "tv_resnet101"}
