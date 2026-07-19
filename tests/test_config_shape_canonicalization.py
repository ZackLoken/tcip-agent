"""L1 — config-shape canonicalization. The GUI/validated schema nests
``stages``/``mixed_precision``/``batch_size`` under ``training``, but ``generic_trainer.train()``
reads them from the top level of ``run.config``. ``normalize_train_config`` must hoist them
(top-level-wins) so a GUI-launched run trains the configured schedule, not the default stage."""


GUI_CONFIG = {
    "model_source": {
        "builder": "module:build_net",
        "builder_kwargs": {"num_classes": 1},
        "task": "detection",
    },
    "data": {"images_dir": "", "labels_dir": "", "task": "detection"},
    "training": {
        "batch_size": 4,
        "num_workers": 0,
        "mixed_precision": True,
        "stages": [{"freeze_to": -1, "epochs": 5}, {"freeze_to": 2, "epochs": 10}],
    },
    "optimizer": {"name": "adamw", "backbone_lr": 1e-4, "head_lr": 1e-3, "weight_decay": 1e-4},
}


def test_normalize_hoists_training_section_to_top_level():
    from tcip_mcp.pipelines.schemas import normalize_train_config

    cfg = normalize_train_config(GUI_CONFIG)
    # train() reads these from the top level — they must equal the nested schedule/knobs.
    assert cfg["stages"] == GUI_CONFIG["training"]["stages"]
    assert cfg["batch_size"] == 4
    assert cfg["mixed_precision"] is True
    assert cfg["num_workers"] == 0
    # The nested section is preserved (validated schema + experiment-record snapshot).
    assert "training" in cfg


def test_normalize_top_level_wins_over_nested():
    from tcip_mcp.pipelines.schemas import normalize_train_config

    # The orchestrator writes a flat config; the HPO objective writes tuned params flat.
    # A pre-existing top-level key must never be clobbered by the nested value.
    cfg = normalize_train_config({
        "stages": [{"freeze_to": 0, "epochs": 3}],
        "training": {"stages": [{"freeze_to": -1, "epochs": 99}]},
    })
    assert cfg["stages"] == [{"freeze_to": 0, "epochs": 3}]


def test_run_config_exposes_top_level_stages_after_normalize(tmp_path, monkeypatch):
    """The characterization the fix is about: what train() reads (run.config['stages'])
    equals the GUI's configured schedule — not the default single stage."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.pipelines.schemas import normalize_train_config
    from tcip_mcp.pipelines.training.generic_trainer import create_run

    run = create_run(normalize_train_config(GUI_CONFIG), str(tmp_path))
    assert run.config["stages"] == GUI_CONFIG["training"]["stages"]
    assert len(run.config["stages"]) == 2  # not the default single 10-epoch stage


def test_stage_spec_tolerates_but_no_longer_declares_lr():
    # Per-stage lr was a lie (train() never reads it; the optimizer block sets LR). The field
    # is gone, but extra="allow" means an old config carrying stage lr still validates.
    from tcip_mcp.pipelines.schemas import StageSpec

    assert "lr" not in StageSpec.model_fields
    StageSpec.model_validate({"freeze_to": -1, "epochs": 5, "lr": 1e-3})  # tolerated
