"""A training config has one shape, its training keys at the top level.

The schema validates the flat config with its training keys beside ``model_source`` and
``data``, and a nested ``training`` section is refused by name.
"""

import copy
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from tcip_mcp.pipelines.schemas import (
    NESTED_TRAINING_SECTION_REFUSAL,
    SCHEDULER_TYPES,
    DefaultTrainerRegime,
    EarlyStoppingSpec,
    EvaluationSpec,
    OptimizerSpec,
    StageSpec,
    TilingSpec,
    TrainConfigSchema,
    checked_train_config,
    train_config,
)
from tests import REPO_ROOT
from tests._chain_fixtures import (
    BESPOKE_DETECTION,
    BESPOKE_MODELS,
    FRAME_STATISTICS,
    training_config,
    weights_statistics,
)
from tests._training_values import (
    RATIONALE,
    SCHEDULE_SETTINGS,
    adamw_optimizer,
    augmentation,
    drawn_split,
    evaluation_block,
    sampler,
    schedule,
    sgd_optimizer,
    stage,
    stop_rule,
    tiling_block,
)


def _issues_of(config):
    """Return the issues ``schemas.checked_train_config`` reports for ``config``."""
    return checked_train_config(config)[1]


def _without(config: dict, path: str) -> dict:
    """Return a copy of ``config`` with the key at the dotted ``path`` removed.

    A list entry in the path is named by its index.
    """
    copied = copy.deepcopy(config)
    *parents, leaf = path.split(".")
    node: Any = copied
    for part in parents:
        node = node[int(part)] if isinstance(node, list) else node[part]
    del node[leaf]
    return copied


def _names(issues: list[str], path: str) -> bool:
    """Return whether one of ``issues`` names every part of the dotted ``path``."""
    return any(
        all(part in issue for part in path.split(".")) for issue in issues
    )


FLAT_CONFIG: dict = training_config(
    REPO_ROOT,
    weights_statistics(
        {
            "builder": "module:build_net",
            "builder_kwargs": {},
            "task": "detection",
        }
    ),
    {
        "images_dir": str(Path(__file__).parent / "images"),
        "split": drawn_split(),
        "tiling": tiling_block(tile_size=64, overlap=0.25),
    },
    num_workers=0,
    mixed_precision=True,
    optimizer=adamw_optimizer(),
    stages=[stage(-1), stage(2)],
    evaluation=evaluation_block(selection_metric="f1"),
    lr_scaling={"scale_power": 0.5, "rationale": RATIONALE},
    sampler=sampler("oversample", min_count=2),
    augmentation=augmentation(horizontal_flip={"p": 0.5}),
)


def _required_paths() -> list[str]:
    """Return the values the schema requires, by dotted path, read off the schema itself.

    The paths are its required top-level fields and the required fields of each stated block (a
    stage, the optimizer, the stated schedule, the stop rule, the evaluation and the tiling).
    """
    top = [
        n for n, f in TrainConfigSchema.model_fields.items() if f.is_required()
    ]
    blocks = {
        "stages.0": StageSpec,
        "optimizer": OptimizerSpec,
        "scheduler": type(train_config(FLAT_CONFIG).scheduler),
        "early_stopping": EarlyStoppingSpec,
        "evaluation": EvaluationSpec,
        "data.tiling": TilingSpec,
    }
    nested = [
        f"{block}.{n}"
        for block, model in blocks.items()
        for n, f in model.model_fields.items()
        if f.is_required()
    ]
    return [*top, *nested]


STATED_PATHS = [
    "seed",
    "deterministic",
    "mixed_precision",
    "evaluation",
    "data.split",
    "stages.0.freeze_to",
    "early_stopping.patience",
    "early_stopping.min_delta",
    "early_stopping.nonfinite_passes",
    "evaluation.selection_metric",
    "data.tiling.skip_empty",
]
"""Every value training or its loaders read that a config states itself, by its dotted path."""

DRAWN_SPLIT_PATHS = [
    f"data.split.{key}"
    for key in (
        "seed",
        "val_ratio",
        "calibration_ratio",
        "holdout_ratio",
        "stratify_foreground",
        "grouping",
        "rationale",
    )
]
"""Every key a drawn split states."""

SCHEDULE_PATHS = [
    (kind, f"scheduler.{key}")
    for kind, settings in SCHEDULE_SETTINGS.items()
    for key in settings
]
"""Each scheduler type beside each setting its sample block states."""

RATIONALE_PATHS = [
    "rationale",
    "stages.1.rationale",
    "optimizer.rationale",
    "scheduler.rationale",
    "early_stopping.rationale",
    "evaluation.rationale",
    "lr_scaling.rationale",
    "data.split.rationale",
    "data.tiling.rationale",
    "sampler.rationale",
    "augmentation.rationale",
]
"""The reason of every block :data:`FLAT_CONFIG` states, by its dotted path."""


TRAINER_READ_PATHS = [
    "batch_size",
    "evaluation.conf_threshold",
    *DefaultTrainerRegime._fields,
]
"""The values training reads under the default trainer that the schema leaves to the launch
door."""


def _door_issues(config: dict) -> list[str]:
    """Return the issues the launch door's structural check reports for ``config``.

    The config is one the schema admits, checked with this repository as its project.
    """
    from tcip_mcp.tools.training_tools import _structural_issues

    return _structural_issues(train_config(config), REPO_ROOT)[1]


def test_a_flat_config_validates_with_no_issue():
    assert _issues_of(FLAT_CONFIG) == []


def _refused_naming(config: dict, path: str, project: Path) -> None:
    """Assert ``config`` less the key at ``path`` is refused naming it and ``config`` is admitted.

    The refusal is asserted where a config is validated and at the launch door.
    """
    from tcip_mcp.tools.training_tools import launch_training

    unstated = _without(config, path)
    issues = _issues_of(unstated)
    assert _names(issues, path), issues
    refused = launch_training(project, unstated, actor=None)
    assert _names(refused["issues"], path), refused
    assert _issues_of(config) == []


@pytest.mark.parametrize(
    "path",
    sorted({*_required_paths(), *STATED_PATHS}) + DRAWN_SPLIT_PATHS,
)
def test_a_config_stating_no_required_value_is_refused_naming_it(
    path, tmp_path
):
    """A config stating no required value is refused naming it, and one stating it is admitted.

    Each required or stated value, a drawn split's shares, stratification and seed among them,
    is refused unstated by name where the config is validated and at the launch door.
    """
    _refused_naming(FLAT_CONFIG, path, tmp_path)


@pytest.mark.parametrize(("kind", "path"), SCHEDULE_PATHS)
def test_a_schedule_stating_no_setting_is_refused_naming_it(
    kind, path, tmp_path
):
    """Each setting a schedule kind states is refused unstated by name."""
    _refused_naming(
        {**FLAT_CONFIG, "scheduler": schedule(kind)}, path, tmp_path
    )


@pytest.mark.parametrize("path", DRAWN_SPLIT_PATHS)
def test_a_drawn_split_omitting_a_draw_key_is_refused_by_that_field(path):
    """A drawn split omitting a draw key is refused by that field's requiredness, located at it.

    One draw key left out is reported as ``data.split.drawn.<key>: Field required``.
    """
    key = path.removeprefix("data.split.")
    issues = _issues_of(_without(FLAT_CONFIG, path))
    assert f"data.split.drawn.{key}: Field required" in issues, issues


def test_a_drawn_split_groups_by_a_named_policy_or_an_explicit_map():
    """A drawn split's one ``grouping`` admits a named policy or an explicit map."""
    assert isinstance(FLAT_CONFIG["data"]["split"]["grouping"], str)
    assert _issues_of(FLAT_CONFIG) == []
    by_map = copy.deepcopy(FLAT_CONFIG)
    by_map["data"]["split"]["grouping"] = {"2-11-26/s00": "a"}
    assert _issues_of(by_map) == []


@pytest.mark.parametrize("path", RATIONALE_PATHS)
def test_a_block_stating_no_reason_is_refused_naming_it(path):
    """A stated block left without its ``rationale`` is refused naming the block.

    Each stated block carries its own ``rationale``, and the config stating each is admitted.
    """
    issues = _issues_of(_without(FLAT_CONFIG, path))
    assert _names(issues, path), issues
    assert _issues_of(FLAT_CONFIG) == []


def _least_admitted(config: dict, paths: list[str]) -> dict:
    """Return ``config`` less each of ``paths`` whose removal the schema and launch door admit.

    The paths are tried one at a time, each staying out when both still admit the config.
    """
    least = copy.deepcopy(config)
    for path in paths:
        fewer = _without(least, path)
        if not _issues_of(fewer) and not _door_issues(fewer):
            least = fewer
    return least


def _at(record: dict, path: str) -> Any:
    """Return the value at the dotted ``path`` of ``record``; ``KeyError`` naming it when absent."""
    node: Any = record
    for part in path.split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


RECORDED_PATHS = [
    "batch_size",
    *STATED_PATHS,
    *(f"scheduler.{key}" for key in SCHEDULE_SETTINGS["onecycle"]),
    *(
        f"data.split.{key}"
        for key in (
            "val_ratio",
            "calibration_ratio",
            "holdout_ratio",
            "stratify_foreground",
        )
    ),
    "rationale",
    *(
        f"{block}.rationale"
        for block in (
            "stages.0",
            "optimizer",
            "scheduler",
            "early_stopping",
            "evaluation",
            "data.split",
            "data.tiling",
        )
    ),
]
"""Every value training and its loaders read, by its dotted path in a run's recorded config."""


def test_a_runs_record_names_every_value_its_training_reads(tmp_path):
    """A run's record carries each recorded training value, as a run context reads it.

    The least config the platform admits for a tiled, drawn detection run under a one-cycle
    schedule, opened by the launcher's own producer, records each of the recorded paths in
    ``run.json``'s config, non-null, and a run context's spec record over the same config
    carries each at the recorded value.
    """
    from tcip_mcp.experiments import observe
    from tests._verified_checkpoint_fixtures import (
        SCOPED_DATA,
        detection_images,
        fixture_data_dir,
        opened_run,
    )
    from tests.tiny_trainer_fixtures import run_context

    config = training_config(
        tmp_path,
        {
            "builder": BESPOKE_DETECTION,
            "source_files": [BESPOKE_MODELS],
            "task": "detection",
            "normalization": FRAME_STATISTICS,
        },
        {
            **detection_images(
                fixture_data_dir(tmp_path, "least"), SCOPED_DATA["scope"], n=4
            ),
            **SCOPED_DATA,
            "split": drawn_split(),
            "tiling": tiling_block(tile_size=32, sliver_frac=0.5),
        },
        scheduler=schedule("onecycle"),
    )
    least = _least_admitted(config, RECORDED_PATHS)

    recorded = observe(opened_run(tmp_path, least)).record["config"]
    worker = run_context(tmp_path, least).run.spec.record()
    for path in RECORDED_PATHS:
        assert _at(recorded, path) == _at(worker, path) is not None, path


@pytest.mark.parametrize("path", TRAINER_READ_PATHS)
def test_a_config_stating_no_tuned_value_is_admitted_and_refused_at_the_door_naming_it(
    path,
):
    stated = _without(FLAT_CONFIG, path)
    assert _issues_of(stated) == []
    assert _names(_door_issues(stated), path)


def test_a_config_naming_its_own_loop_states_only_what_its_loop_reads():
    """A config naming its own training loop is not asked for the default trainer's regime.

    The door refuses a config naming no ``training_source`` that leaves the default regime's
    fields unstated, naming ``stages``; the same config naming its own ``training_source``
    reports nothing unstated.
    """
    custom = {
        k: v
        for k, v in FLAT_CONFIG.items()
        if k not in DefaultTrainerRegime._fields
    }
    assert _names(_door_issues(custom), "stages")
    looped = _door_issues(
        {**custom, "training_source": "bespoke_models:train_bespoke"}
    )
    assert not any("unstated" in issue for issue in looped), looped


def test_an_optimizer_naming_no_family_is_refused_naming_it():
    """An optimizer block naming no family is refused naming ``optimizer.name``.

    The optimizer's family has no default, and the same block naming it is admitted.
    """
    assert _names(
        _issues_of(_without(FLAT_CONFIG, "optimizer.name")), "optimizer.name"
    )
    assert _issues_of(FLAT_CONFIG) == []


@pytest.mark.parametrize("kind", SCHEDULER_TYPES)
def test_each_schedule_states_its_own_settings_and_no_other(kind):
    from typing import get_args

    from tcip_mcp.pipelines.schemas import SchedulerSpec

    assert set(SCHEDULE_SETTINGS) == set(SCHEDULER_TYPES)
    (schedule_class,) = [
        c
        for c in get_args(get_args(SchedulerSpec)[0])
        if get_args(c.model_fields["type"].annotation) == (kind,)
    ]
    settings = {
        name: field
        for name, field in schedule_class.model_fields.items()
        if name not in ("type", "rationale")
    }
    assert set(settings) == set(SCHEDULE_SETTINGS[kind])
    assert all(field.is_required() for field in settings.values())
    stated = schedule(kind)
    assert _issues_of({**FLAT_CONFIG, "scheduler": stated}) == []
    for setting in SCHEDULE_SETTINGS[kind]:
        issues = _issues_of(
            {**FLAT_CONFIG, "scheduler": _without(stated, setting)}
        )
        assert _names(issues, f"scheduler.{setting}"), issues
    for other, settings in SCHEDULE_SETTINGS.items():
        if other == kind:
            continue
        stray = next(iter(settings))
        issues = _issues_of(
            {**FLAT_CONFIG, "scheduler": {**stated, stray: settings[stray]}}
        )
        assert _names(issues, f"scheduler.{stray}"), issues


def test_sgd_states_its_momentum_and_no_other_optimizer_does():
    sgd = sgd_optimizer()
    adamw: dict = {**FLAT_CONFIG["optimizer"], "momentum": sgd["momentum"]}
    assert _issues_of({**FLAT_CONFIG, "optimizer": sgd}) == []
    assert _names(
        _issues_of({**FLAT_CONFIG, "optimizer": _without(sgd, "momentum")}),
        "optimizer.momentum",
    )
    assert _names(
        _issues_of({**FLAT_CONFIG, "optimizer": adamw}), "optimizer.momentum"
    )


def test_a_sweep_base_config_may_omit_only_what_its_search_space_names():
    """A sweep's base config may omit the rate its search space names.

    With the sweep's point applied, a base config leaving the swept rate to the sweep validates,
    and one leaving a rate nothing sweeps is refused naming it.
    """
    from tcip_mcp.tools.training_tools import _apply_hpo_params

    base = _without(FLAT_CONFIG, "optimizer.head_lr")
    assert (
        _issues_of(_apply_hpo_params(base, {"optimizer.head_lr": 3e-3})) == []
    )
    unswept = _without(base, "optimizer.backbone_lr")
    issues = _issues_of(
        _apply_hpo_params(unswept, {"optimizer.head_lr": 3e-3})
    )
    assert _names(issues, "optimizer.backbone_lr"), issues


def test_a_nested_training_section_is_refused_by_name():
    nested = {
        k: v
        for k, v in FLAT_CONFIG.items()
        if k not in ("batch_size", "stages")
    }
    nested["training"] = {"batch_size": 4, "stages": FLAT_CONFIG["stages"]}

    issues = _issues_of(nested)

    assert issues == [NESTED_TRAINING_SECTION_REFUSAL]
    assert "top level" in NESTED_TRAINING_SECTION_REFUSAL


def test_an_empty_training_section_is_refused_too():
    """An empty ``training`` section is refused by its key with the nested-section refusal."""
    assert _issues_of({**FLAT_CONFIG, "training": {}}) == [
        NESTED_TRAINING_SECTION_REFUSAL
    ]


def test_top_level_stages_are_typed_by_stage_spec():
    """A stage carries no epoch count: one stating ``epochs`` is refused by name."""
    issues = _issues_of({**FLAT_CONFIG, "stages": [stage(0, epochs=5)]})
    assert any("stages.0.epochs" in issue for issue in issues), issues


def test_the_trainer_reads_the_flat_config_as_given(tmp_path, monkeypatch):
    """An opened run's recorded spec carries the flat config's stages as given.

    The record holds the configured stages and no ``training`` key.
    """
    monkeypatch.chdir(tmp_path)
    from tests._chain_fixtures import DETECTION_SOURCE, own_statistics
    from tests._verified_checkpoint_fixtures import (
        SCOPED_DATA,
        detection_images,
    )
    from tests.tiny_trainer_fixtures import run_context

    data = {
        **detection_images(tmp_path / "ds", SCOPED_DATA["scope"]),
        **SCOPED_DATA,
    }
    config = own_statistics(
        {**FLAT_CONFIG, "model_source": DETECTION_SOURCE, "data": data},
        tmp_path,
    )
    run = run_context(tmp_path, config).run
    assert run.spec.record()["stages"] == FLAT_CONFIG["stages"]
    assert "training" not in run.spec.record()


def test_stage_spec_refuses_a_per_stage_lr():
    """A stage stating its own learning rate is refused naming ``lr``."""
    StageSpec.model_validate(stage(-1))
    with pytest.raises(ValidationError, match="lr"):
        StageSpec.model_validate(stage(-1, lr=1e-3))


def test_a_stage_states_its_own_gradient_accumulation():
    """A stage stating its own gradient accumulation is admitted by the schema."""
    stages = [stage(-1), stage(0, gradient_accumulation_steps=4)]
    assert _issues_of({**FLAT_CONFIG, "stages": stages}) == []


def test_the_stop_rule_validates_as_stated_and_refuses_a_zero_patience_and_an_on_switch():
    """The stop rule validates as stated and refuses a zero patience and an ``enabled`` key.

    The stop rule's values are the config's own, read as stated, and none has a default; a
    patience of zero and an ``enabled`` key are refused by name.
    """
    stated = train_config(
        {
            **FLAT_CONFIG,
            "early_stopping": stop_rule(3, 0.02, nonfinite_passes=4),
        }
    )
    assert stated.early_stopping is not None
    assert (
        stated.early_stopping.patience,
        stated.early_stopping.min_delta,
        stated.early_stopping.nonfinite_passes,
    ) == (3, 0.02, 4)
    for value in ("patience", "min_delta", "nonfinite_passes"):
        issues = _issues_of(_without(FLAT_CONFIG, f"early_stopping.{value}"))
        assert any(f"early_stopping.{value}" in issue for issue in issues), (
            issues
        )
    assert any(
        "early_stopping.patience" in issue
        for issue in _issues_of(
            {**FLAT_CONFIG, "early_stopping": stop_rule(0)}
        )
    )
    assert any(
        "early_stopping.enabled" in issue
        for issue in _issues_of(
            {
                **FLAT_CONFIG,
                "early_stopping": {**stop_rule(3), "enabled": True},
            }
        )
    )


def test_lr_scaling_requires_its_power_and_names_no_reference_batch():
    scaling = {"scale_power": 0.5, "rationale": RATIONALE}
    assert (
        _issues_of({**FLAT_CONFIG, "lr_scaling": {**scaling, "max_lr": 0.01}})
        == []
    )
    assert any(
        "lr_scaling.scale_power" in issue
        for issue in _issues_of(
            {**FLAT_CONFIG, "lr_scaling": {"rationale": RATIONALE}}
        )
    )
    assert any(
        "reference_effective_batch" in issue
        for issue in _issues_of(
            {
                **FLAT_CONFIG,
                "lr_scaling": {**scaling, "reference_effective_batch": 64},
            }
        )
    )


def test_the_loaders_read_the_batch_size_the_schema_validated(tmp_path):
    """The loaders are built at the batch size the schema validated, coerced the same way.

    The string ``"3"`` the schema admits builds loaders of batch size 3, and a batch size of 0
    is refused naming ``batch_size``.
    """
    pytest.importorskip("torch")
    from tests.tiny_trainer_fixtures import (
        frames_data,
        regressor_config,
        run_context,
    )

    assert _issues_of({**FLAT_CONFIG, "batch_size": "3"}) == []
    data = frames_data(tmp_path / "ds", [0.1, 0.2, 0.3], [0.2, 0.4, 0.6])
    train_loader = run_context(
        tmp_path,
        regressor_config(
            data=data, project=tmp_path, batch_size="3", num_workers="0"
        ),
    ).train_loader
    assert (train_loader.batch_size, train_loader.num_workers) == (3, 0)
    with pytest.raises(ValueError, match="batch_size"):
        train_config({**FLAT_CONFIG, "batch_size": 0})


def test_model_source_refuses_an_undeclared_key_by_name():
    """A misspelled model_source key is refused by name where the config is validated."""
    issues = _issues_of(
        {
            "model_source": {
                "builder": "m:f",
                "builder_kwargs": {},
                "anchor_ratio": [0.5, 1.0, 2.0],
            }
        }
    )
    assert any("anchor_ratio" in issue for issue in issues)


def test_model_source_admits_every_declared_key(tmp_path):
    """A model source stating builder, kwargs, task, sources, statistics and geometry validates.

    The statistics are stated as their derivation's own producer records them and the geometry
    is stated with its reason; the config reports no issue.
    """
    pytest.importorskip("torch")
    from tests._chain_fixtures import (
        frame_statistics,
        stated_detector_geometry,
    )
    from tests._image_fixtures import write_noise_image

    image = tmp_path / "a.png"
    write_noise_image(image, 8)
    issues = _issues_of(
        {
            **FLAT_CONFIG,
            "model_source": {
                "builder": "m:f",
                "builder_kwargs": {},
                "task": "detection",
                "source_files": [str(Path(__file__).parent / "m.py")],
                "normalization": frame_statistics([image], 3),
                "detector_geometry": stated_detector_geometry((8, 8)),
            },
        }
    )
    assert issues == []


def test_a_width_stated_on_the_model_source_is_refused_by_name():
    """A width stated on the model source is refused by name.

    The width is the run's own ``data.num_channels``; a model source stating ``in_chans`` is
    refused naming ``model_source.in_chans``.
    """
    issues = _issues_of(
        {
            "model_source": {
                "builder": "m:f",
                "task": "detection",
                "in_chans": 2,
            },
            "data": {},
        }
    )
    assert any("model_source.in_chans" in issue for issue in issues), issues


@pytest.mark.parametrize(
    ("path", "config"),
    [
        (
            "model_source.task",
            {**FLAT_CONFIG, "model_source": {"builder": "module:build_net"}},
        ),
        (
            "data.split.drawn.seed",
            {**FLAT_CONFIG, "data": {"split": {"seed": "first"}}},
        ),
        (
            "data.split.drawn.selection_dirr",
            {**FLAT_CONFIG, "data": {"split": {"selection_dirr": "m"}}},
        ),
        (
            "data.dataset_source.builder",
            {**FLAT_CONFIG, "data": {"dataset_source": {}}},
        ),
        (
            "data.tiling.tile_sze",
            {**FLAT_CONFIG, "data": {"tiling": {"tile_sze": 64}}},
        ),
        (
            "data.tiling.tile_size",
            {**FLAT_CONFIG, "data": {"tiling": {"tile_size": "large"}}},
        ),
        (
            "data.split.drawn.spatial_manifest",
            {
                **FLAT_CONFIG,
                "data": {"split": {"spatial_manifest": {"stem": "a"}}},
            },
        ),
    ],
)
def test_a_fact_a_reader_reads_is_refused_where_the_config_is_validated(
    path, config
):
    """A fact a reader reads that is missing, mistyped or misspelled is refused at validation.

    ``model_source.task`` and the listed ``data`` keys a resolver or a recorded-dimension reader
    reads are typed where the config enters: each one missing, of the wrong type or misspelled
    is named in the validation issues.
    """
    issues = _issues_of(config)
    assert any(issue.startswith(path) for issue in issues), issues


def test_a_stated_tiling_block_reaches_the_tiler_as_stated():
    """A stated ``data.tiling`` block reaches the tiler as stated.

    Every tiler option the block states validates and reaches the tiler's keyword arguments by
    name, its reason none of them, and one left unstated is left to the tiler.
    """
    from tcip_mcp.pipelines.schemas import train_config

    stated = tiling_block(
        tile_size=64,
        overlap=0.25,
        sliver_frac=0.5,
        skip_empty=True,
        keep_regions=[[0, 0, 64, 64]],
    )
    tiling = train_config(
        {**FLAT_CONFIG, "data": {**FLAT_CONFIG["data"], "tiling": stated}}
    ).data.tiling
    assert tiling is not None
    assert tiling.tiler_options() == {
        **{k: v for k, v in stated.items() if k != "rationale"},
        "keep_regions": [(0, 0, 64, 64)],
    }
    assert tiling.tiler_options(frozenset({"keep_regions"})) == {
        "tile_size": 64,
        "overlap": 0.25,
        "sliver_frac": 0.5,
        "skip_empty": True,
    }
