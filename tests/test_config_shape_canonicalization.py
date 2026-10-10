"""A training config has one shape: every key the trainer reads sits at the top level beside
``model_source`` and ``data``, and a nested ``training`` section is refused by name."""

import copy
from pathlib import Path

import pytest
from pydantic import ValidationError

from tcip_mcp.pipelines.schemas import (
    NESTED_TRAINING_SECTION_REFUSAL,
    SCHEDULER_TYPES,
    DefaultTrainerRegime,
    EarlyStoppingSpec,
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
    {"images_dir": str(Path(__file__).parent / "images")},
    num_workers=0, mixed_precision=True, optimizer=adamw_optimizer(),
    stages=[{"freeze_to": -1}, {"freeze_to": 2}],
    evaluation=evaluation_block(selection_metric="f1"))


def _required_paths() -> list[str]:
    """Every value the schema requires, by its dotted path, read off the schema itself: its
    required top-level fields and the required fields of the optimizer, of the stated schedule
    and of the stop rule."""
    top = [n for n, f in TrainConfigSchema.model_fields.items() if f.is_required()]
    blocks = {"optimizer": OptimizerSpec, "scheduler": type(train_config(FLAT_CONFIG).scheduler),
              "early_stopping": EarlyStoppingSpec}
    nested = [f"{block}.{n}" for block, model in blocks.items()
              for n, f in model.model_fields.items() if f.is_required()]
    return [*top, *nested]


TRAINER_READ_PATHS = ["batch_size", "evaluation.conf_threshold", *DefaultTrainerRegime._fields]
"""The values training reads under the default trainer that the schema leaves to the launch
door."""


def _door_issues(config: dict) -> list[str]:
    """The issues the launch door's structural check reports for ``config``, which the schema
    admits, for this repository as its project."""
    from tcip_mcp.tools.training_tools import _structural_issues
    from tests import REPO_ROOT

    return _structural_issues(train_config(config), REPO_ROOT)[1]


def test_a_flat_config_validates_with_no_issue():
    assert _issues_of(FLAT_CONFIG) == []


@pytest.mark.parametrize("path", _required_paths())
def test_a_config_stating_no_required_value_is_refused_naming_it(path):
    issues = _issues_of(_without(FLAT_CONFIG, path))
    assert _names(issues, path), issues


@pytest.mark.parametrize("path", TRAINER_READ_PATHS)
def test_a_config_stating_no_tuned_value_is_admitted_and_refused_at_the_door_naming_it(path):
    stated = _without(FLAT_CONFIG, path)
    assert _issues_of(stated) == []
    assert _names(_door_issues(stated), path)


def test_a_config_naming_its_own_loop_states_only_what_its_loop_reads():
    """The door requires the default trainer's regime of a config the default trainer runs; a
    config naming its own ``training_source`` leaves unstated what its loop never reads."""
    custom = {k: v for k, v in FLAT_CONFIG.items() if k not in DefaultTrainerRegime._fields}
    assert _names(_door_issues(custom), "stages")
    looped = _door_issues({**custom, "training_source": "bespoke_models:train_bespoke"})
    assert not any("unstated" in issue for issue in looped), looped


def test_an_optimizer_naming_no_family_is_refused_naming_it():
    """guard. The optimizer's family has no default: a block naming none is refused naming
    ``optimizer.name``, and the same block naming it is admitted."""
    assert _names(_issues_of(_without(FLAT_CONFIG, "optimizer.name")), "optimizer.name")
    assert _issues_of(FLAT_CONFIG) == []


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
    """A stage carries no epoch count: one stating ``epochs`` is refused by name."""
    issues = _issues_of({**FLAT_CONFIG, "stages": [{"freeze_to": 0, "epochs": 5}]})
    assert any("stages.0.epochs" in issue for issue in issues), issues


def test_the_trainer_reads_the_flat_config_as_given(tmp_path, monkeypatch):
    """What train() reads (run.spec.stages) is the configured schedule itself, with no
    second placement to reconcile against."""
    monkeypatch.chdir(tmp_path)
    from tests._chain_fixtures import DETECTION_SOURCE
    from tests.tiny_trainer_fixtures import trainer_run

    run = trainer_run({**FLAT_CONFIG, "model_source": DETECTION_SOURCE}, tmp_path,
                      project=tmp_path, has_val_loader=True, id="auto-run-5")
    assert run.spec.record()["stages"] == FLAT_CONFIG["stages"]
    assert "training" not in run.spec.record()


def test_stage_spec_refuses_a_per_stage_lr():
    """train() reads learning rates from the optimizer block alone, so a stage lr is refused."""
    StageSpec.model_validate({"freeze_to": -1})
    with pytest.raises(ValidationError, match="lr"):
        StageSpec.model_validate({"freeze_to": -1, "lr": 1e-3})


def test_a_stage_states_its_own_gradient_accumulation():
    """train() reads a stage's own accumulation, so the schema admits it."""
    stages = [{"freeze_to": -1}, {"freeze_to": 0, "gradient_accumulation_steps": 4}]
    assert _issues_of({**FLAT_CONFIG, "stages": stages}) == []


def test_the_stop_rule_validates_as_stated_and_refuses_a_zero_patience_and_an_on_switch():
    """The stop rule's two values are the config's own, read as stated, and neither has a
    default; a patience of zero and an ``enabled`` key, since nothing but the rule ends a stage,
    are refused by name."""
    stated = train_config({**FLAT_CONFIG, "early_stopping": {"patience": 3, "min_delta": 0.02}})
    assert stated.early_stopping is not None
    assert (stated.early_stopping.patience, stated.early_stopping.min_delta) == (3, 0.02)
    for value in ("patience", "min_delta"):
        issues = _issues_of(_without(FLAT_CONFIG, f"early_stopping.{value}"))
        assert any(f"early_stopping.{value}" in issue for issue in issues), issues
    assert any("early_stopping.patience" in issue for issue in _issues_of(
        {**FLAT_CONFIG, "early_stopping": {"patience": 0, "min_delta": 0.0}}))
    assert any("early_stopping.enabled" in issue for issue in _issues_of(
        {**FLAT_CONFIG, "early_stopping": {"enabled": True, "patience": 3, "min_delta": 0.0}}))


def test_lr_scaling_requires_its_power_and_names_no_reference_batch():
    assert _issues_of(
        {**FLAT_CONFIG, "lr_scaling": {"scale_power": 0.5, "max_lr": 0.01}}) == []
    assert any("lr_scaling.scale_power" in issue for issue in _issues_of(
        {**FLAT_CONFIG, "lr_scaling": {}}))
    assert any("reference_effective_batch" in issue for issue in _issues_of(
        {**FLAT_CONFIG, "lr_scaling": {"scale_power": 0.5, "reference_effective_batch": 64}}))


def test_the_loaders_read_the_batch_size_the_schema_validated(tmp_path):
    """The batch size the schema admits is the one the loaders are built at, coerced the same
    way, and one it refuses builds no loader."""
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.training.generic_trainer import run_loaders
    from tests.tiny_trainer_fixtures import ConstantImageDataset, regressor_config, trainer_run

    dataset = ConstantImageDataset([0.1, 0.2, 0.3], [0.2, 0.4, 0.6])
    assert _issues_of({**FLAT_CONFIG, "batch_size": "3"}) == []
    run = trainer_run(regressor_config(batch_size="3", num_workers="0"), tmp_path / "out",
                      project=tmp_path, has_val_loader=True)
    train_loader, _ = run_loaders(run, dataset, None)
    assert (train_loader.batch_size, train_loader.num_workers) == (3, 0)
    with pytest.raises(ValueError, match="batch_size"):
        train_config({**FLAT_CONFIG, "batch_size": 0})


def test_model_source_refuses_an_undeclared_key_by_name():
    """A misspelled model_source key is refused by name where the config is validated."""
    issues = _issues_of({"model_source": {
        "builder": "m:f", "builder_kwargs": {}, "anchor_ratio": [0.5, 1.0, 2.0],
    }})
    assert any("anchor_ratio" in issue for issue in issues)


def test_model_source_admits_every_declared_key(tmp_path):
    """Every declared key validates with no issue, image_stats_sampling as the normalization
    derivation's own producer (``derivations.image_stats_provenance``) renders it."""
    pytest.importorskip("torch")
    from PIL import Image

    from tcip_mcp.pipelines.derivations import band_normalization_stats, image_stats_provenance

    image = tmp_path / "a.png"
    Image.new("RGB", (8, 8), (20, 40, 60)).save(image)
    result = band_normalization_stats([image], 3)
    assert result is not None
    mean, std, _ = result
    issues = _issues_of({**FLAT_CONFIG, "model_source": {
        "builder": "m:f", "builder_kwargs": {"image_mean": mean, "image_std": std},
        "task": "detection", "source_files": [str(Path(__file__).parent / "m.py")],
        "image_stats_sampling": image_stats_provenance(result),
    }, "data": {}})
    assert issues == []


def test_a_width_stated_on_the_model_source_is_refused_by_name():
    """The width is the run's own ``data.num_channels``, handed to the builder; a model source
    stating one beside it is refused by name rather than read as a second spelling."""
    issues = _issues_of(
        {"model_source": {"builder": "m:f", "task": "detection", "in_chans": 2}, "data": {}})
    assert any("model_source.in_chans" in issue for issue in issues), issues


@pytest.mark.parametrize(("path", "config"), [
    ("model_source.task", {**FLAT_CONFIG, "model_source": {"builder": "module:build_net"}}),
    ("data.split.seed", {**FLAT_CONFIG, "data": {"split": {"seed": "first"}}}),
    ("data.split.selection_dirr", {**FLAT_CONFIG, "data": {"split": {"selection_dirr": "m"}}}),
    ("data.dataset_source.builder", {**FLAT_CONFIG, "data": {"dataset_source": {}}}),
    ("data.tiling.tile_sze", {**FLAT_CONFIG, "data": {"tiling": {"tile_sze": 64}}}),
    ("data.tiling.tile_size", {**FLAT_CONFIG, "data": {"tiling": {"tile_size": "large"}}}),
    ("data.split.spatial_manifest",
     {**FLAT_CONFIG, "data": {"split": {"spatial_manifest": {"stem": "a"}}}}),
])
def test_a_fact_a_reader_reads_is_refused_where_the_config_is_validated(path, config):
    """``model_source.task`` and every ``data`` key a resolver or a recorded-dimension reader
    reads are typed where the config enters: one missing, of the wrong type or misspelled is
    named there, before any reader below the boundary meets it."""
    issues = _issues_of(config)
    assert any(issue.startswith(path) for issue in issues), issues


def test_a_stated_tiling_block_reaches_the_tiler_as_stated():
    """Admits valid work: every tiler option a ``data.tiling`` block states validates and reaches
    the tiler's keyword arguments by name, and one left unstated is left to the tiler."""
    from tcip_mcp.pipelines.schemas import train_config

    stated = {"enabled": True, "tile_size": 64, "overlap": 0.25, "sliver_frac": 0.5,
              "dedup_iou": 0.7, "skip_empty": True, "keep_regions": [[0, 0, 64, 64]]}
    tiling = train_config({**FLAT_CONFIG, "data": {"tiling": stated}}).data.tiling
    assert tiling is not None
    assert tiling.tiler_options() == {**{k: v for k, v in stated.items() if k != "enabled"},
                                      "keep_regions": [(0, 0, 64, 64)]}
    assert tiling.tiler_options(frozenset({"keep_regions", "dedup_iou"})) == {
        "tile_size": 64, "overlap": 0.25, "sliver_frac": 0.5, "skip_empty": True}
