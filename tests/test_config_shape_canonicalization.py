"""A training config has one shape: every key ``generic_trainer.train()`` reads sits at the top
level beside ``model_source`` and ``data``. A nested ``training`` section is refused by name,
since a key under it would be read by nothing and the run would train at the trainer's own
defaults in silence; there is no hoist and no precedence rule between two placements."""

import pytest
from pydantic import ValidationError

from tcip_mcp.pipelines.schemas import (
    NESTED_TRAINING_SECTION_REFUSAL,
    StageSpec,
    validate_train_config_schema,
)

FLAT_CONFIG = {
    "model_source": {
        "builder": "module:build_net",
        "builder_kwargs": {"num_classes": 1},
        "task": "detection",
    },
    "data": {"images_dir": "", "labels_dir": "", "task": "detection"},
    "batch_size": 4,
    "num_workers": 0,
    "mixed_precision": True,
    "stages": [{"freeze_to": -1, "epochs": 5}, {"freeze_to": 2, "epochs": 10}],
    "optimizer": {"name": "adamw", "backbone_lr": 1e-4, "head_lr": 1e-3, "weight_decay": 1e-4},
    "evaluation": {"selection_metric": "f1"},
}


def test_a_flat_config_validates_with_no_issue():
    assert validate_train_config_schema(FLAT_CONFIG) == []


def test_a_nested_training_section_is_refused_by_name():
    nested = {k: v for k, v in FLAT_CONFIG.items() if k not in ("batch_size", "stages")}
    nested["training"] = {"batch_size": 4, "stages": FLAT_CONFIG["stages"]}

    issues = validate_train_config_schema(nested)

    assert issues == [NESTED_TRAINING_SECTION_REFUSAL]
    assert "top level" in NESTED_TRAINING_SECTION_REFUSAL


def test_an_empty_training_section_is_refused_too():
    """Even an empty section names a placement nothing reads; the refusal is about the key."""
    assert validate_train_config_schema({**FLAT_CONFIG, "training": {}}) == [
        NESTED_TRAINING_SECTION_REFUSAL]


def test_top_level_stages_are_typed_by_stage_spec():
    issues = validate_train_config_schema({**FLAT_CONFIG, "stages": [{"freeze_to": 0}]})
    assert any("stages.0.epochs" in issue for issue in issues)


def test_the_trainer_reads_the_flat_config_as_given(tmp_path, monkeypatch):
    """What train() reads (run.config['stages']) is the configured schedule itself, with no
    second placement to reconcile against."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.pipelines.training.run_registry import create_run

    run = create_run(dict(FLAT_CONFIG), str(tmp_path), id="auto-run-5")
    assert run.config["stages"] == FLAT_CONFIG["stages"]
    assert len(run.config["stages"]) == 2
    assert "training" not in run.config


def test_stage_spec_refuses_a_per_stage_lr():
    """train() reads learning rates from the optimizer block alone, so a stage lr is refused."""
    StageSpec.model_validate({"freeze_to": -1, "epochs": 5})
    with pytest.raises(ValidationError, match="lr"):
        StageSpec.model_validate({"freeze_to": -1, "epochs": 5, "lr": 1e-3})


def test_model_source_refuses_an_undeclared_key_by_name():
    """A misspelled model_source key is dropped silently by every reader today; the schema names
    it rather than building the model at the builder's own defaults."""
    issues = validate_train_config_schema({"model_source": {
        "builder": "m:f", "builder_kwargs": {}, "anchor_ratio": [0.5, 1.0, 2.0],
    }})
    assert any("anchor_ratio" in issue for issue in issues)


def test_model_source_admits_every_declared_key():
    """The rail admits valid work: every declared key, including the sixth
    (image_stats_sampling), validates with no issue."""
    issues = validate_train_config_schema({"model_source": {
        "builder": "m:f", "builder_kwargs": {"image_mean": [0.1], "image_std": [0.2]},
        "task": "detection", "in_chans": 1, "source_files": ["m.py"],
        "image_stats_sampling": {"windows": [["a.tif", None]], "seed": None,
                                 "pixel_fraction": 1.0, "window_size": None,
                                 "max_windows_per_image": None},
    }})
    assert issues == []


def test_two_band_config_declaring_in_chans_only_in_builder_kwargs_is_checked_at_two():
    """run_in_chans is the one reader the trainer, the predictor and the contract dims share; a
    config that only declares in_chans inside builder_kwargs must not silently check against 3.
    A run that declares none reads at the width its own data config recorded."""
    from tcip_mcp.pipelines.model_build import run_in_chans

    model_source = {"builder": "m:f", "builder_kwargs": {"num_classes": 1, "in_chans": 2}}
    assert run_in_chans(model_source, {"num_channels": 5}) == 2
    assert run_in_chans({"builder": "m:f"}, {"num_channels": 5}) == 5
    assert run_in_chans({"builder": "m:f"}, {}) is None
