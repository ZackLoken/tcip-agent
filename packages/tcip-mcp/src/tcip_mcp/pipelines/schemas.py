"""Pydantic v2 config schemas. A training config's platform-owned keys are typed here, and the
trainer and its loaders read them from the validated model (:func:`train_config`); a bespoke
``model_source``/``dataset_source``/``training_source`` may read its own additional keys.

A training config has one shape: every key ``generic_trainer.train()`` reads (``batch_size``,
``stages``, ``mixed_precision``, ``device``, ``seed``, ``evaluation``, ...) sits at the top level
of the config, beside ``model_source`` and ``data``. A config carrying a ``training`` key is
refused by name.
"""

from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class StageSpec(BaseModel):
    """One progressive-unfreeze step: the trainer reads ``freeze_to``, ``epochs`` and the stage's
    own ``gradient_accumulation_steps``, while the learning rate comes from the top-level
    ``optimizer`` block, never per stage. Any other key, a per-stage ``lr`` included, is refused
    by name."""

    model_config = ConfigDict(extra="forbid")
    epochs: int = Field(ge=1)
    freeze_to: int = 0
    gradient_accumulation_steps: int | None = Field(None, ge=1)
    """``None``, stated or not, is the run's own top-level ``gradient_accumulation_steps``."""


class EarlyStoppingSpec(BaseModel):
    """A stage's plateau rule, on whenever the run has a validation loader unless ``enabled`` is
    false: a stage ends once its selection value has not improved by ``min_delta`` for
    ``patience`` epochs."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    patience: int = Field(7, ge=1)
    """An owner-ruled provisional default."""
    min_delta: float = Field(1e-4, ge=0)
    """An owner-ruled provisional default."""


DEFAULT_BACKBONE_LR = 1e-4
DEFAULT_HEAD_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
"""The optimizer block's rates and weight decay when a config states none: retained platform
defaults, provisional."""


class OptimizerSpec(BaseModel):
    """The optimizer every stage builds (``optimizer_factory.build_optimizer``)."""

    model_config = ConfigDict(extra="forbid")
    name: str = "adamw"
    backbone_lr: float = Field(DEFAULT_BACKBONE_LR, gt=0)
    head_lr: float = Field(DEFAULT_HEAD_LR, gt=0)
    weight_decay: float = Field(DEFAULT_WEIGHT_DECAY, ge=0)


SchedulerType = Literal["cosine", "plateau", "onecycle", "step"]
SCHEDULER_TYPES: tuple[str, ...] = get_args(SchedulerType)
"""The ``scheduler.type`` names ``generic_trainer._build_scheduler`` builds."""


class SchedulerSpec(BaseModel):
    """The learning-rate schedule each stage builds (``generic_trainer._build_scheduler``) and
    the settings its type reads; the defaults are retained platform defaults, provisional."""

    model_config = ConfigDict(extra="forbid")
    type: SchedulerType = "cosine"
    eta_min: float = 0.0
    factor: float = 0.5
    patience: int = 3
    max_lr: float | None = None
    step_size: int = 10
    gamma: float = 0.1


class EvaluationSpec(BaseModel):
    """What a run's validation pass scores against: the confirmed trait it selects for, the
    selection metric (resolved by ``generic_trainer.resolve_selection_metric``), and the
    thresholds and score weights the validation pass applies."""

    model_config = ConfigDict(extra="forbid")
    trait: str | None = None
    selection_metric: str | None = None
    conf_threshold: float | None = None
    iou_threshold: float = 0.5
    score_weights: dict | None = None


class LrScalingSpec(BaseModel):
    """Effective-batch LR scaling: each stage's learning rates times ``(its effective batch / the
    first stage's) ** scale_power``, capped at ``max_lr`` when one is stated."""

    model_config = ConfigDict(extra="forbid")
    scale_power: float
    max_lr: float | None = Field(None, gt=0)


class ImageStatsWindow(BaseModel):
    """One pixel rectangle a sampled band-normalization statistic read, in a raster's own grid."""

    model_config = ConfigDict(extra="forbid")
    x0: int
    y0: int
    x1: int
    y1: int


class ImageStatsSampling(BaseModel):
    """Provenance for ``model_source.builder_kwargs``'s per-band ``image_mean``/``image_std``.

    Rendered by ``derivations.image_stats_provenance`` from whichever of
    ``band_normalization_stats`` (the exact derivation) or ``band_normalization_stats_sampled``
    (the windowed one) produced the statistics; never hand-assembled. ``windows`` pairs each
    source's label with the rectangle read from it, or ``None`` for the exact derivation's own
    whole-image read (``pixel_fraction`` is then ``1.0`` and ``seed``/``window_size``/
    ``max_windows_per_image`` are ``None``, an exhaustive read fabricates no seed).
    """

    model_config = ConfigDict(extra="forbid")
    windows: list[tuple[str, ImageStatsWindow | None]]
    seed: int | None = None
    pixel_fraction: float
    window_size: int | None = None
    max_windows_per_image: int | None = None


class ModelSourceSchema(BaseModel):
    """The importable-builder reference; a key outside these fields refuses by name."""

    model_config = ConfigDict(extra="forbid")
    builder: str | None = None
    builder_kwargs: dict | None = None
    task: str | None = None
    source_files: list[str] | None = None
    image_stats_sampling: ImageStatsSampling | None = None


NESTED_TRAINING_SECTION_REFUSAL = (
    "'training' is not a config section: every key the trainer reads (batch_size, stages, "
    "mixed_precision, device, seed, evaluation, ...) sits at the top level of the config, "
    "beside model_source and data. Move the keys under 'training' up one level."
)


DEFAULT_BATCH_SIZE = 2
"""Images per training step when a config states no ``batch_size``."""


class TrainConfigSchema(BaseModel):
    """The one training config shape: trainer keys at the top level, no nested section. A run's
    config is validated once by its producer (:func:`checked_train_config`), and the trainer, its
    loaders, preflight and the model builder read that one validated model, ``model_source``
    included (``model_build.build_from_model_source`` builds from the ``ModelSourceSchema`` it
    holds). ``data``, typed here only as a mapping, is resolved by the dataset producers."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())
    model_source: ModelSourceSchema | None = None
    data: dict | None = None
    batch_size: int = Field(DEFAULT_BATCH_SIZE, ge=1)
    num_workers: int = Field(0, ge=0)
    sampler: str = "random"
    augmentation: dict | None = None
    device: str | None = None
    """``None``: cuda when available, else cpu."""
    seed: int | None = None
    deterministic: bool = False
    mixed_precision: bool = True
    stages: list[StageSpec] | None = Field(None, min_length=1)
    gradient_accumulation_steps: int = Field(1, ge=1)
    optimizer: OptimizerSpec = Field(default_factory=lambda: OptimizerSpec.model_validate({}))
    scheduler: SchedulerSpec = Field(default_factory=lambda: SchedulerSpec.model_validate({}))
    stage_warmup_epochs: int = Field(0, ge=0)
    enforce_monotonic_unfreeze: bool = True
    checkpoint_every_n_epochs: int = Field(5, ge=0)
    early_stopping: EarlyStoppingSpec = Field(
        default_factory=lambda: EarlyStoppingSpec.model_validate({}))
    lr_scaling: LrScalingSpec | None = None
    log_every_n_batches: int | None = Field(None, ge=1)
    """``None``: ``generic_trainer.BATCH_ROWS_PER_EPOCH`` rows an epoch
    (``generic_trainer.batch_row_ends``)."""
    evaluation: EvaluationSpec = Field(default_factory=lambda: EvaluationSpec.model_validate({}))

    @model_validator(mode="before")
    @classmethod
    def _refuse_nested_training_section(cls, values: object) -> object:
        if isinstance(values, dict) and "training" in values:
            raise ValueError(NESTED_TRAINING_SECTION_REFUSAL)
        return values


def checked_train_config(config: dict) -> tuple[TrainConfigSchema | None, list[str]]:
    """``config`` validated once against the schema: the validated config and no issues, or
    ``None`` and one issue string per type or structure error (e.g. ``batch_size="big"``, a stage
    missing ``epochs``, a nested ``training`` section). Does not require ``model_source``."""
    try:
        return TrainConfigSchema.model_validate(config), []
    except ValidationError as e:
        return None, _issues(e)


def train_config(config: dict) -> TrainConfigSchema:
    """``config`` validated as the trainer reads it (:func:`checked_train_config`). Refuses
    (``ValueError``) naming every issue it reports."""
    spec, issues = checked_train_config(config)
    if spec is None:
        raise ValueError(f"invalid training config: {'; '.join(issues)}")
    return spec


def _issues(error: ValidationError) -> list[str]:
    """One issue string per error ``error`` carries, located by its config path."""
    issues: list[str] = []
    for err in error.errors():
        loc = ".".join(str(x) for x in err.get("loc", ()))
        msg = err.get("msg", "invalid")
        if err.get("type") == "value_error":
            msg = str(err.get("ctx", {}).get("error", msg))
        issues.append(f"{loc}: {msg}" if loc else msg)
    return issues
