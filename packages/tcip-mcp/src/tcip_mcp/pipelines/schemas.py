"""Pydantic v2 config schemas. A training config's platform-owned keys are typed here, its
``model_source`` and ``data`` blocks included, and every reader takes them from the validated
model (:func:`train_config`); a bespoke ``training_source`` may read its own additional
top-level keys.

A training config has one shape: every key ``generic_trainer.train()`` reads (``batch_size``,
``stages``, ``mixed_precision``, ``device``, ``seed``, ``evaluation``, ...) sits at the top level
of the config, beside ``model_source`` and ``data``. A config carrying a ``training`` key is
refused by name.
"""

from __future__ import annotations

from typing import Annotated, Literal, NamedTuple, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from tcip_mcp.pipelines.data.selection import ClassScope


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
    """Owner ruling."""
    min_delta: float = Field(1e-4, ge=0)
    """Owner ruling."""


class OptimizerSpec(BaseModel):
    """The optimizer every stage builds (``optimizer_factory.build_optimizer``): its tuned
    settings required (both rates, the weight decay, and ``momentum`` for ``sgd`` and for no other
    optimizer), its identity defaulting to the platform's own."""

    model_config = ConfigDict(extra="forbid")
    name: str = "adamw"
    """The platform's optimizer when a config names none."""
    backbone_lr: float = Field(gt=0)
    head_lr: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    momentum: float | None = Field(None, ge=0)

    @model_validator(mode="after")
    def _momentum_belongs_to_sgd(self) -> OptimizerSpec:
        if (self.name == "sgd") != (self.momentum is not None):
            raise ValueError(f"optimizer.momentum is stated for sgd and for no other optimizer; "
                             f"this block names {self.name!r} and momentum {self.momentum!r}.")
        return self


class CosineSchedule(BaseModel):
    """Cosine annealing over a stage's epochs down to ``eta_min``."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["cosine"]
    eta_min: float


class PlateauSchedule(BaseModel):
    """The rate times ``factor`` once the validation loss has not improved for ``patience``
    epochs."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["plateau"]
    factor: float
    patience: int


class OneCycleSchedule(BaseModel):
    """One cycle over a stage's epochs peaking at ``max_lr``."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["onecycle"]
    max_lr: float


class StepSchedule(BaseModel):
    """The rate times ``gamma`` every ``step_size`` epochs."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["step"]
    step_size: int
    gamma: float


SchedulerSpec = Annotated[CosineSchedule | PlateauSchedule | OneCycleSchedule | StepSchedule,
                          Field(discriminator="type")]
"""The learning-rate schedule each stage builds (``generic_trainer._build_scheduler``): its
``type`` and that type's own settings, each stated, any other type's refused by name."""
SCHEDULER_TYPES: tuple[str, ...] = tuple(
    get_args(schedule.model_fields["type"].annotation)[0]
    for schedule in get_args(get_args(SchedulerSpec)[0]))
"""The ``scheduler.type`` names ``generic_trainer._build_scheduler`` builds."""


class EvaluationSpec(BaseModel):
    """What a run's validation pass scores against: the confirmed trait it selects for, the
    selection metric (resolved by ``generic_trainer.resolve_selection_metric``), and the
    thresholds and score weights the validation pass applies."""

    model_config = ConfigDict(extra="forbid")
    trait: str | None = None
    selection_metric: str | None = None
    conf_threshold: float | None = Field(None, ge=0, le=1)
    """The confidence a detector's validation counts boxes at, required of a detector run under
    every trainer by :class:`TrainConfigSchema`'s own validation."""
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
    """The importable-builder reference and the task its model is for; a key outside these
    fields refuses by name."""

    model_config = ConfigDict(extra="forbid")
    builder: str = Field(min_length=1)
    builder_kwargs: dict | None = None
    task: str = Field(min_length=1)
    """The run's task (detection, instance_seg, classification, ...)."""
    source_files: list[str] | None = None
    image_stats_sampling: ImageStatsSampling | None = None


class DatasetSourceSchema(BaseModel):
    """A bespoke dataset builder (``datasets.build_from_dataset_source``); a key outside these
    fields refuses by name."""

    model_config = ConfigDict(extra="forbid")
    builder: str = Field(min_length=1)
    builder_kwargs: dict | None = None
    source_files: list[str] | None = None


class SplitSpec(BaseModel):
    """``data.split``: a bound selection (``selection_dir``, ``redraw_within_selection``), or the
    parameters a run drawing its own split states."""

    model_config = ConfigDict(extra="forbid")
    selection_dir: str | None = None
    redraw_within_selection: bool = False
    seed: int | None = None
    group_by: str | None = None
    group_key_map: dict[str, str] | None = None
    stratify_foreground: bool = True
    val_ratio: float | None = None
    calibration_ratio: float | None = None
    holdout_ratio: float | None = None

    @property
    def stated(self) -> set[str]:
        """The keys this section states a value for."""
        return set(self.model_dump(exclude_unset=True, exclude_none=True))


Rect = tuple[int, int, int, int]


class SpatialManifest(BaseModel):
    """The within-image split a run's resolution drew over its one source's tile lattice
    (``split_construction.spatial_single_source_split``), recorded in its resolved partition:
    each side's half-open pixel regions, the region identities its train and val tiles carry,
    the lattice and buffer it was drawn at, what it requested and realized, and the raster it
    was drawn over (``split_construction.raster_identity``)."""

    model_config = ConfigDict(extra="forbid")
    stem: str
    train_identities: list[str]
    val_identities: list[str]
    train_region: list[Rect]
    val_region: list[Rect]
    holdout_region: list[Rect]
    calibration_region: list[Rect]
    kept_train_tiles: int
    kept_val_tiles: int
    kept_holdout_tiles: int
    kept_calibration_tiles: int
    width: int
    height: int
    tile_size: int
    overlap: float
    axis: str
    buffer: int
    requested_fractions: dict[str, float]
    realized_fractions: dict[str, float]
    realized_discard_fraction: float
    tiles_dropped_past_extent: int
    tiles_dropped_outside_regions: int
    raster_content_identity: dict


class TilingSpec(BaseModel):
    """``data.tiling``: whether a detection run tiles its loader, the
    ``datasets.TiledDetectionDataset`` options it states (``None``, the tiler's own default), and
    a within-image spatial split's ``buffer``. A key outside these fields refuses by name."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    tile_size: int | None = None
    overlap: float | None = None
    sliver_frac: float | None = None
    dedup_iou: float | None = None
    skip_empty: bool | None = None
    keep_regions: list[Rect] | None = None
    buffer: int | None = None

    def tiler_options(self, exclude: frozenset[str] = frozenset()) -> dict:
        """The ``TiledDetectionDataset`` keyword arguments this block states, but ``exclude``;
        an option it leaves ``None`` is omitted so the tiler's own default applies."""
        return self.model_dump(exclude_none=True, exclude={"enabled", "buffer", *exclude})


class _Recorded(BaseModel):
    """A validated block a run's records hold as :meth:`record` dumps it."""

    def record(self, exclude: dict | None = None) -> dict:
        """This block as a run's record holds it: every key stated or resolved, in JSON form,
        but the fields ``exclude`` names (``BaseModel.model_dump``'s nested form)."""
        return self.model_dump(mode="json", exclude_unset=True, exclude=exclude)


class DataSpec(_Recorded):
    """``data``: where a run's samples are and how they split, and what its resolution records
    beside them (``scope``, the sizes, the stamped geometry). A key outside these fields refuses
    by name."""

    model_config = ConfigDict(extra="forbid")
    images_dir: str | None = None
    labels_dir: str | None = None
    dataset_source: DatasetSourceSchema | None = None
    split: SplitSpec = Field(default_factory=lambda: SplitSpec.model_validate({}))
    auto_val: bool = True
    scope: ClassScope | None = None
    tiling: TilingSpec | None = None
    num_channels: int | None = None
    num_classes: int | None = None
    num_ranks: int | None = None
    train_native_size: list[int] | None = None
    plant_csv_paths: list[str] | None = None

    @property
    def recorded_scope(self) -> ClassScope:
        """The class space this block records (``ClassScope.recorded``, which refuses a block
        recording none)."""
        return ClassScope.recorded(self.scope)


NESTED_TRAINING_SECTION_REFUSAL = (
    "'training' is not a config section: every key the trainer reads (batch_size, stages, "
    "mixed_precision, device, seed, evaluation, ...) sits at the top level of the config, "
    "beside model_source and data. Move the keys under 'training' up one level."
)


class DefaultTrainerRegime(NamedTuple):
    """The blocks the default trainer (``generic_trainer.train``) reads, each stated."""

    stages: list[StageSpec]
    optimizer: OptimizerSpec
    scheduler: SchedulerSpec
    checkpoint_every_n_epochs: int


class TrainConfigSchema(_Recorded):
    """The one training config shape: trainer keys at the top level, no nested section. A run's
    config is validated once by its producer (:func:`checked_train_config`), and the trainer, its
    loaders, preflight, the model builder and every record of the run read that one validated
    model, ``model_source``, ``data`` and ``training_source`` included; a key outside these
    fields is a bespoke loop's own and is kept as stated. A config naming no ``training_source``
    runs the default trainer and states its regime (:meth:`default_trainer_regime`); one naming
    its own loop states what that loop reads."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())
    model_source: ModelSourceSchema
    data: DataSpec
    training_source: str | None = Field(None, min_length=1)
    """A bespoke ``train(ctx)`` loop, a dotted ``module:function``; ``None`` for the default
    trainer."""
    batch_size: int = Field(ge=1)
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
    optimizer: OptimizerSpec | None = None
    scheduler: SchedulerSpec | None = None
    stage_warmup_epochs: int = Field(0, ge=0)
    enforce_monotonic_unfreeze: bool = True
    checkpoint_every_n_epochs: int | None = Field(None, ge=0)
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

    @model_validator(mode="after")
    def _the_config_states_what_its_trainer_reads(self) -> TrainConfigSchema:
        from tcip_mcp.pipelines.model_contract import DETECTION_TASKS

        if (self.model_source.task in DETECTION_TASKS
                and self.evaluation.conf_threshold is None):
            raise ValueError("evaluation.conf_threshold unstated: a detector's validation counts "
                             "its boxes at a confidence the run states, under any trainer")
        if self.training_source is None:
            self.default_trainer_regime()
        return self

    def default_trainer_regime(self) -> DefaultTrainerRegime:
        """The regime the default trainer runs at. Refuses (``ValueError``) naming every block of
        it this config leaves unstated."""
        missing = [name for name in DefaultTrainerRegime._fields if getattr(self, name) is None]
        if missing:
            raise ValueError(f"{missing} unstated: a config naming no training_source trains "
                             "under the default trainer, which reads each of them")
        return DefaultTrainerRegime(*(getattr(self, name) for name in DefaultTrainerRegime._fields))


def checked_train_config(config: dict) -> tuple[TrainConfigSchema | None, list[str]]:
    """``config`` validated once against the schema: the validated config and no issues, or
    ``None`` and one issue string per type or structure error (e.g. ``batch_size="big"``, a stage
    missing ``epochs``, a nested ``training`` section, a missing ``model_source.task``)."""
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
