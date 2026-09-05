"""Config-shape canonicalization. The GUI/validated schema nests
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
    # train() reads these from the top level: they must equal the nested schedule/knobs.
    assert cfg["stages"] == GUI_CONFIG["training"]["stages"]
    assert cfg["batch_size"] == 4
    assert cfg["mixed_precision"] is True
    assert cfg["num_workers"] == 0
    # The nested section is preserved (validated schema + experiment-record snapshot).
    assert "training" in cfg


def test_normalize_top_level_wins_over_nested():
    from tcip_mcp.pipelines.schemas import normalize_train_config

    # The HPO objective writes tuned params flat.
    # A pre-existing top-level key must never be clobbered by the nested value.
    cfg = normalize_train_config({
        "stages": [{"freeze_to": 0, "epochs": 3}],
        "training": {"stages": [{"freeze_to": -1, "epochs": 99}]},
    })
    assert cfg["stages"] == [{"freeze_to": 0, "epochs": 3}]


def test_run_config_exposes_top_level_stages_after_normalize(tmp_path, monkeypatch):
    """What train() reads (run.config['stages']) equals the GUI's configured schedule,
    not the default single stage."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.pipelines.schemas import normalize_train_config
    from tcip_mcp.pipelines.training.run_registry import create_run

    run = create_run(normalize_train_config(GUI_CONFIG), str(tmp_path))
    assert run.config["stages"] == GUI_CONFIG["training"]["stages"]
    assert len(run.config["stages"]) == 2  # not the default single 10-epoch stage


def test_evaluation_section_returns_the_top_level_block_when_both_are_present():
    from tcip_mcp.pipelines.schemas import evaluation_section

    cfg = {
        "evaluation": {"selection_metric": "f1"},
        "training": {"evaluation": {"selection_metric": "loss"}},
    }
    assert evaluation_section(cfg) == {"selection_metric": "f1"}


def test_evaluation_section_returns_the_nested_block_when_only_it_is_present():
    from tcip_mcp.pipelines.schemas import evaluation_section

    cfg = {"training": {"evaluation": {"selection_metric": "loss"}}}
    assert evaluation_section(cfg) == {"selection_metric": "loss"}


def test_evaluation_section_returns_an_empty_dict_when_neither_is_present():
    from tcip_mcp.pipelines.schemas import evaluation_section

    assert evaluation_section({}) == {}
    assert evaluation_section({"training": {"batch_size": 4}}) == {}


def test_evaluation_section_of_a_plain_dict_returns_the_original_nested_object():
    """A plain dict's own ``get`` returns the actual stored object, never a copy: a write
    through the returned block is visible on ``config`` afterward."""
    from tcip_mcp.pipelines.schemas import evaluation_section

    config = {"evaluation": {"selection_metric": "loss"}}
    evaluation_section(config)["selection_metric"] = "f1"

    assert config["evaluation"]["selection_metric"] == "f1"


def test_evaluation_section_of_an_access_tracking_config_returns_a_wrapped_copy():
    """An access-tracking config's own ``get`` wraps a nested read in a freshly constructed
    ``_AccessTrackingConfig``, a copy of that nested dict's items, never a live view over the
    original (``_AccessTrackingConfig``'s own stated limitation): a write through the returned
    block is lost, unlike the plain-dict case above."""
    from tcip_mcp.pipelines.schemas import evaluation_section
    from tcip_mcp.tools.training_tools import _AccessTrackingConfig

    config = _AccessTrackingConfig({"evaluation": {"selection_metric": "loss"}})
    evaluation_section(config)["selection_metric"] = "f1"

    assert config["evaluation"]["selection_metric"] == "loss"  # unchanged: the write hit a copy


def test_evaluation_section_is_idempotent_on_an_already_normalized_config():
    """Coverage, not a guard: this shape is stable under any implementation that agrees with
    itself, since normalize_train_config's hoist is a no-op once the top-level key already
    exists. The precedence itself is proven by
    test_evaluation_section_agrees_with_the_hoist_precedence_on_four_shapes below."""
    from tcip_mcp.pipelines.schemas import evaluation_section, normalize_train_config

    cfg = normalize_train_config({
        "evaluation": {"selection_metric": "f1"},
        "training": {"evaluation": {"selection_metric": "loss"}},
    })
    assert evaluation_section(cfg) == evaluation_section(normalize_train_config(cfg))


def test_evaluation_section_agrees_with_the_hoist_precedence_on_four_shapes():
    """A consistency test whose two sides are different implementations: evaluation_section
    reads config directly (no copy), normalize_train_config's own setdefault hoist reads through
    a shallow copy; the two must still agree on which block governs, on every shape a config may
    take (both placements present, a falsy top-level block beside a real nested one, nested
    only, neither)."""
    from tcip_mcp.pipelines.schemas import evaluation_section, normalize_train_config

    def via_hoist(cfg: dict) -> dict:
        return normalize_train_config(cfg).get("evaluation") or {}

    shapes = [
        {"evaluation": {"selection_metric": "f1"},
         "training": {"evaluation": {"selection_metric": "loss"}}},
        {"evaluation": {}, "training": {"evaluation": {"selection_metric": "loss"}}},
        {"training": {"evaluation": {"selection_metric": "loss"}}},
        {"training": {"batch_size": 4}},
    ]
    for cfg in shapes:
        assert evaluation_section(cfg) == via_hoist(cfg)


def test_stage_spec_tolerates_but_no_longer_declares_lr():
    # Per-stage lr was a lie (train() never reads it; the optimizer block sets LR). The field
    # is gone, but extra="allow" means an old config carrying stage lr still validates.
    from tcip_mcp.pipelines.schemas import StageSpec

    assert "lr" not in StageSpec.model_fields
    StageSpec.model_validate({"freeze_to": -1, "epochs": 5, "lr": 1e-3})  # tolerated


def test_model_source_refuses_an_undeclared_key_by_name():
    """A misspelled model_source key is dropped silently by every reader today; the schema names
    it rather than building the model at the builder's own defaults."""
    from tcip_mcp.pipelines.schemas import validate_train_config_schema

    issues = validate_train_config_schema({"model_source": {
        "builder": "m:f", "builder_kwargs": {}, "anchor_ratio": [0.5, 1.0, 2.0],
    }})
    assert any("anchor_ratio" in issue for issue in issues)


def test_model_source_admits_every_declared_key():
    """The rail admits valid work: every declared key, including the sixth
    (image_stats_sampling), validates with no issue."""
    from tcip_mcp.pipelines.schemas import validate_train_config_schema

    issues = validate_train_config_schema({"model_source": {
        "builder": "m:f", "builder_kwargs": {"image_mean": [0.1], "image_std": [0.2]},
        "task": "detection", "in_chans": 1, "source_files": ["m.py"],
        "image_stats_sampling": {"windows": [["a.tif", None]], "seed": None,
                                 "pixel_fraction": 1.0, "window_size": None,
                                 "max_windows_per_image": None},
    }})
    assert issues == []


def test_two_band_config_declaring_in_chans_only_in_builder_kwargs_is_checked_at_two():
    """declared_in_chans is the one fallback both the trainer and the predictor read; a config
    that only declares in_chans inside builder_kwargs must not silently check against 3."""
    from tcip_mcp.pipelines.model_build import declared_in_chans

    model_source = {"builder": "m:f", "builder_kwargs": {"num_classes": 1, "in_chans": 2}}
    assert declared_in_chans(model_source) == 2
