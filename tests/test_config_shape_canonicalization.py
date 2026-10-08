"""A training config has one shape: every key ``generic_trainer.train()`` reads sits at the top
level beside ``model_source`` and ``data``. A nested ``training`` section is refused by name,
since a key under it would be read by nothing and the run would train at the trainer's own
defaults in silence; there is no hoist and no precedence rule between two placements."""

import copy

import pytest
from pydantic import ValidationError

from tcip_mcp.pipelines.schemas import (
    NESTED_TRAINING_SECTION_REFUSAL,
    SCHEDULER_TYPES,
    DefaultTrainerRegime,
    OptimizerSpec,
    StageSpec,
    TrainConfigSchema,
    checked_train_config,
    train_config,
)
from tests._chain_fixtures import training_config
from tests._training_values import (
    SCHEDULE_SETTINGS, adamw_optimizer, evaluation_block, schedule, sgd_optimizer,
)


def _issues_of(config):
    """The issues ``schemas.checked_train_config`` reports for ``config``."""
    return checked_train_config(config)[1]


def _without(config: dict, path: str) -> dict:
    """A copy of ``config`` with the key at the dotted ``path`` removed."""
    copied = copy.deepcopy(config)
    *parents, leaf = path.split(".")
    node = copied
    for part in parents:
        node = node[part]
    del node[leaf]
    return copied


def _names(issues: list[str], path: str) -> bool:
    """Whether one of ``issues`` names every part of the dotted ``path``."""
    return any(all(part in issue for part in path.split(".")) for issue in issues)


FLAT_CONFIG: dict = training_config(
    {"builder": "module:build_net", "builder_kwargs": {}, "task": "detection"},
    {"images_dir": ""},
    num_workers=0, mixed_precision=True, optimizer=adamw_optimizer(),
    stages=[{"freeze_to": -1, "epochs": 5}, {"freeze_to": 2, "epochs": 10}],
    evaluation=evaluation_block(selection_metric="f1"))


def _required_paths() -> list[str]:
    """Every tuned value the schema requires of a config the default trainer runs, by its dotted
    path, read off the schema itself: its required top-level fields, the default trainer's regime,
    and the required fields of the optimizer and of the stated schedule."""
    top = [n for n, f in TrainConfigSchema.model_fields.items() if f.is_required()]
    blocks = {"optimizer": OptimizerSpec, "scheduler": type(train_config(FLAT_CONFIG).scheduler)}
    nested = [f"{block}.{n}" for block, model in blocks.items()
              for n, f in model.model_fields.items() if f.is_required()]
    return [*top, *DefaultTrainerRegime._fields, *nested]


def test_a_flat_config_validates_with_no_issue():
    assert _issues_of(FLAT_CONFIG) == []


@pytest.mark.parametrize("path", _required_paths())
def test_a_config_stating_no_tuned_value_is_refused_naming_it(path):
    issues = _issues_of(_without(FLAT_CONFIG, path))
    assert _names(issues, path), issues


def test_a_config_naming_its_own_loop_states_only_what_its_loop_reads():
    """The default trainer's regime is required of a config the default trainer runs; a config
    naming its own ``training_source`` leaves unstated what its loop never reads."""
    custom = {k: v for k, v in FLAT_CONFIG.items() if k not in DefaultTrainerRegime._fields}
    assert _names(_issues_of(custom), "stages")
    assert _issues_of({**custom, "training_source": "my_loops:train"}) == []


def test_an_optimizer_naming_none_is_the_platforms_own():
    unnamed = _without(FLAT_CONFIG, "optimizer.name")
    optimizer = train_config(unnamed).optimizer
    assert optimizer is not None and optimizer.name == OptimizerSpec.model_fields["name"].default


@pytest.mark.parametrize("kind", SCHEDULER_TYPES)
def test_each_schedule_states_its_own_settings_and_no_other(kind):
    from typing import get_args

    from tcip_mcp.pipelines.schemas import SchedulerSpec

    assert set(SCHEDULE_SETTINGS) == set(SCHEDULER_TYPES)
    (schedule_class,) = [c for c in get_args(get_args(SchedulerSpec)[0])
                         if get_args(c.model_fields["type"].annotation) == (kind,)]
    settings = {name: field for name, field in schedule_class.model_fields.items()
                if name != "type"}
    assert set(settings) == set(SCHEDULE_SETTINGS[kind])
    assert all(field.is_required() for field in settings.values())
    stated = schedule(kind)
    assert _issues_of({**FLAT_CONFIG, "scheduler": stated}) == []
    for setting in SCHEDULE_SETTINGS[kind]:
        issues = _issues_of({**FLAT_CONFIG, "scheduler": _without(stated, setting)})
        assert _names(issues, f"scheduler.{setting}"), issues
    for other, settings in SCHEDULE_SETTINGS.items():
        if other == kind:
            continue
        stray = next(iter(settings))
        issues = _issues_of(
            {**FLAT_CONFIG, "scheduler": {**stated, stray: settings[stray]}})
        assert _names(issues, f"scheduler.{stray}"), issues


def test_sgd_states_its_momentum_and_no_other_optimizer_does():
    sgd = sgd_optimizer()
    adamw: dict = {**FLAT_CONFIG["optimizer"], "momentum": sgd["momentum"]}
    assert _issues_of({**FLAT_CONFIG, "optimizer": sgd}) == []
    assert _names(_issues_of({**FLAT_CONFIG, "optimizer": _without(sgd, "momentum")}),
                  "optimizer.momentum")
    assert _names(_issues_of({**FLAT_CONFIG, "optimizer": adamw}), "optimizer.momentum")


def test_a_sweep_base_config_may_omit_only_what_its_search_space_names():
    """A trial's config is validated once its point is applied, so a base config leaving the
    swept rate to the sweep validates, and one leaving a rate nothing sweeps is refused naming
    it."""
    from tcip_mcp.tools.training_tools import _apply_hpo_params

    base = _without(FLAT_CONFIG, "optimizer.head_lr")
    assert _issues_of(_apply_hpo_params(base, {"optimizer.head_lr": 3e-3})) == []
    unswept = _without(base, "optimizer.backbone_lr")
    issues = _issues_of(_apply_hpo_params(unswept, {"optimizer.head_lr": 3e-3}))
    assert _names(issues, "optimizer.backbone_lr"), issues


def test_a_nested_training_section_is_refused_by_name():
    nested = {k: v for k, v in FLAT_CONFIG.items() if k not in ("batch_size", "stages")}
    nested["training"] = {"batch_size": 4, "stages": FLAT_CONFIG["stages"]}

    issues = _issues_of(nested)

    assert issues == [NESTED_TRAINING_SECTION_REFUSAL]
    assert "top level" in NESTED_TRAINING_SECTION_REFUSAL


def test_an_empty_training_section_is_refused_too():
    """Even an empty section names a placement nothing reads; the refusal is about the key."""
    assert _issues_of({**FLAT_CONFIG, "training": {}}) == [
        NESTED_TRAINING_SECTION_REFUSAL]


def test_top_level_stages_are_typed_by_stage_spec():
    issues = _issues_of({**FLAT_CONFIG, "stages": [{"freeze_to": 0}]})
    assert any("stages.0.epochs" in issue for issue in issues)


def test_the_trainer_reads_the_flat_config_as_given(tmp_path, monkeypatch):
    """What train() reads (run.config['stages']) is the configured schedule itself, with no
    second placement to reconcile against."""
    monkeypatch.chdir(tmp_path)
    from tests.tiny_trainer_fixtures import trainer_run

    run = trainer_run(dict(FLAT_CONFIG), tmp_path, project=tmp_path, has_val_loader=True,
                      id="auto-run-5")
    assert run.config["stages"] == FLAT_CONFIG["stages"]
    assert len(run.config["stages"]) == 2
    assert "training" not in run.config


def test_stage_spec_refuses_a_per_stage_lr():
    """train() reads learning rates from the optimizer block alone, so a stage lr is refused."""
    StageSpec.model_validate({"freeze_to": -1, "epochs": 5})
    with pytest.raises(ValidationError, match="lr"):
        StageSpec.model_validate({"freeze_to": -1, "epochs": 5, "lr": 1e-3})


def test_a_stage_states_its_own_gradient_accumulation():
    """train() reads a stage's own accumulation, so the schema admits it."""
    stages = [{"freeze_to": -1, "epochs": 5},
              {"freeze_to": 0, "epochs": 5, "gradient_accumulation_steps": 4}]
    assert _issues_of({**FLAT_CONFIG, "stages": stages}) == []


def test_early_stopping_is_on_by_default_at_its_ruled_rule_and_opts_out_by_name():
    from tcip_mcp.pipelines.schemas import train_config

    unstated = train_config(_without(FLAT_CONFIG, "early_stopping")).early_stopping
    assert (unstated.enabled, unstated.patience, unstated.min_delta) == (True, 7, 1e-4)
    stated = train_config({**FLAT_CONFIG, "early_stopping": {"patience": 3}}).early_stopping
    assert (stated.enabled, stated.patience, stated.min_delta) == (True, 3, 1e-4)
    assert not train_config(
        {**FLAT_CONFIG, "early_stopping": {"enabled": False}}).early_stopping.enabled
    assert any("early_stopping.patience" in issue for issue in _issues_of(
        {**FLAT_CONFIG, "early_stopping": {"patience": 0}}))


def test_lr_scaling_requires_its_power_and_names_no_reference_batch():
    assert _issues_of(
        {**FLAT_CONFIG, "lr_scaling": {"scale_power": 0.5, "max_lr": 0.01}}) == []
    assert any("lr_scaling.scale_power" in issue for issue in _issues_of(
        {**FLAT_CONFIG, "lr_scaling": {}}))
    assert any("reference_effective_batch" in issue for issue in _issues_of(
        {**FLAT_CONFIG, "lr_scaling": {"scale_power": 0.5, "reference_effective_batch": 64}}))


def test_the_loaders_read_the_batch_size_the_schema_validated():
    """The batch size the schema admits is the one the loaders are built at, coerced the same
    way, and one it refuses builds no loader."""
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.schemas import train_config
    from tcip_mcp.pipelines.training.generic_trainer import run_loaders
    from tests.tiny_trainer_fixtures import ConstantImageDataset

    dataset = ConstantImageDataset([0.1, 0.2, 0.3], [0.2, 0.4, 0.6])
    assert _issues_of({**FLAT_CONFIG, "batch_size": "3"}) == []
    train_loader, _ = run_loaders(
        train_config({**FLAT_CONFIG, "batch_size": "3", "num_workers": "0"}),
        "regression", dataset, None)
    assert (train_loader.batch_size, train_loader.num_workers) == (3, 0)
    with pytest.raises(ValueError, match="batch_size"):
        train_config({**FLAT_CONFIG, "batch_size": 0})


def test_model_source_refuses_an_undeclared_key_by_name():
    """A misspelled model_source key is dropped silently by every reader today; the schema names
    it rather than building the model at the builder's own defaults."""
    issues = _issues_of({"model_source": {
        "builder": "m:f", "builder_kwargs": {}, "anchor_ratio": [0.5, 1.0, 2.0],
    }})
    assert any("anchor_ratio" in issue for issue in issues)


def test_model_source_admits_every_declared_key():
    """Every declared key, including the fifth (image_stats_sampling), validates with no issue."""
    issues = _issues_of({**FLAT_CONFIG, "model_source": {
        "builder": "m:f", "builder_kwargs": {"image_mean": [0.1], "image_std": [0.2]},
        "task": "detection", "source_files": ["m.py"],
        "image_stats_sampling": {"windows": [["a.tif", None]], "seed": None,
                                 "pixel_fraction": 1.0, "window_size": None,
                                 "max_windows_per_image": None},
    }})
    assert issues == []


def test_a_width_stated_on_the_model_source_is_refused_by_name():
    """The width is the run's own ``data.num_channels``, handed to the builder; a model source
    stating one beside it is refused by name rather than read as a second spelling."""
    issues = _issues_of(
        {"model_source": {"builder": "m:f", "task": "detection", "in_chans": 2}})
    assert any("model_source.in_chans" in issue for issue in issues), issues
