"""Pydantic v2 config schemas. A training config's platform-owned keys are typed here, its
``model_source`` and ``data`` blocks included (:func:`train_config`), each at the top level of
the config; a bespoke ``training_source`` may read its own additional top-level keys. A config
carrying a ``training`` key is refused by name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, NamedTuple, cast, get_args

from pydantic import (
    AfterValidator, BaseModel, ConfigDict, Field, ValidationError, ValidationInfo,
    model_validator,
)

from tcip_mcp.pipelines.data.selection import ClassScope


def _located(path: str, info: ValidationInfo) -> str:
    from tcip_mcp.registry_paths import located

    return str(located(path, (info.context or {}).get("project")))


ResolvedPath = Annotated[str, AfterValidator(_located)]
"""A path a config states, located once where the config enters
(:func:`~tcip_mcp.registry_paths.located` against the ``project`` its validation context
holds), and that location is read and recorded."""


class StageSpec(BaseModel):
    """One progressive-unfreeze step: the trainer reads ``freeze_to`` and the stage's own
    ``gradient_accumulation_steps``, while the learning rate comes from the top-level
    ``optimizer`` block, never per stage. Any other key, a per-stage ``lr`` or ``epochs``
    included, is refused by name."""

    model_config = ConfigDict(extra="forbid")
    freeze_to: int = 0
    gradient_accumulation_steps: int | None = Field(None, ge=1)
    """``None``, stated or not, is the run's own top-level ``gradient_accumulation_steps``."""


class EarlyStoppingSpec(BaseModel):
    """The stop rule that ends a stage and the run: a stage ends once its selection value has not
    improved by ``min_delta`` for ``patience`` epochs, and the run ends with the stage whose best
    does not improve by ``min_delta`` on the best of the stage before it."""

    model_config = ConfigDict(extra="forbid")
    patience: int = Field(ge=1)
    min_delta: float = Field(ge=0)


class OptimizerSpec(BaseModel):
    """The optimizer every stage builds (``optimizer_factory.build_optimizer``): its family and
    its tuned settings required (both rates, the weight decay, and ``momentum`` for ``sgd`` and
    for no other optimizer)."""

    model_config = ConfigDict(extra="forbid")
    name: str
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


class HorizonSchedule(BaseModel):
    """A schedule built over its own ``horizon_epochs`` scheduled epochs, the most a stage under
    it runs past its warmup."""

    model_config = ConfigDict(extra="forbid")
    horizon_epochs: int = Field(ge=1)


class CosineSchedule(HorizonSchedule):
    """Cosine annealing over the block's horizon down to ``eta_min``."""

    type: Literal["cosine"]
    eta_min: float


class PlateauSchedule(BaseModel):
    """The rate times ``factor`` once the validation loss has not improved for ``patience``
    epochs."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["plateau"]
    factor: float
    patience: int


class OneCycleSchedule(HorizonSchedule):
    """One cycle over the block's horizon peaking at ``max_lr``."""

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


class ScoreWeights(BaseModel):
    """The composite objective's weight on each of its terms
    (``evaluation.compute_composite_objective``), each non-negative so a better term never
    raises the lower-is-better objective."""

    model_config = ConfigDict(extra="forbid")
    loss: float = Field(ge=0)
    f1: float = Field(ge=0)
    map50: float = Field(ge=0)


class EvaluationSpec(BaseModel):
    """What a run's validation pass scores against: the confirmed trait it selects for, the
    selection metric (resolved by ``generic_trainer.resolve_selection_metric``), and the
    thresholds and score weights the validation pass applies."""

    model_config = ConfigDict(extra="forbid")
    trait: str | None = None
    selection_metric: str | None = None
    conf_threshold: float | None = Field(None, ge=0, le=1)
    """The confidence a detector's validation counts boxes at, read through
    :meth:`TrainConfigSchema.trainer_reads`."""
    iou_threshold: float = 0.5
    score_weights: ScoreWeights | None = None


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
    (the windowed one) produced the statistics. ``windows`` pairs each source's label with the
    rectangle read from it, or ``None`` for the exact derivation's own whole-image read
    (``pixel_fraction`` is then ``1.0`` and ``seed``/``window_size``/``max_windows_per_image``
    are ``None``).
    """

    model_config = ConfigDict(extra="forbid")
    windows: list[tuple[ResolvedPath, ImageStatsWindow | None]]
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
    source_files: list[ResolvedPath] | None = None
    image_stats_sampling: ImageStatsSampling | None = None


class DatasetSourceSchema(BaseModel):
    """A bespoke dataset builder; a key outside these fields refuses by name."""

    model_config = ConfigDict(extra="forbid")
    builder: str = Field(min_length=1)
    builder_kwargs: dict | None = None
    source_files: list[ResolvedPath] | None = None


class SplitSpec(BaseModel):
    """``data.split``: a bound selection (``selection_dir``, ``redraw_within_selection``), or the
    parameters a run drawing its own split states."""

    model_config = ConfigDict(extra="forbid")
    selection_dir: ResolvedPath | None = None
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
    the buffer it was drawn at (the lattice is the run's ``data.tiling``), what it requested and
    realized, and the raster it was drawn over (``split_construction.raster_identity``)."""

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
    ``datasets.TiledDetectionDataset`` options it states (an unstated edge and overlap derived,
    every other unstated option the tiler's default), and a within-image spatial split's
    ``buffer``. A key outside these fields refuses by name."""

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
        an option it leaves ``None`` is omitted."""
        return self.model_dump(exclude_none=True, exclude={"enabled", "buffer", *exclude})


class _Recorded(BaseModel):
    """A validated block a run's records hold as :meth:`record` dumps it."""

    def record(self, exclude: dict | None = None) -> dict:
        """This block as a run's record holds it: every key stated or resolved, in JSON form,
        but the fields ``exclude`` names (``BaseModel.model_dump``'s nested form)."""
        return self.model_dump(mode="json", exclude_unset=True, exclude=exclude)


class DataSpec(_Recorded):
    """``data``: where a run's samples are and how they split, and what its resolution records
    beside them (``scope``, the sizes, the stamped geometry and object density). A key outside
    these fields refuses by name, and so does a ``tiling`` beside a ``dataset_source``."""

    model_config = ConfigDict(extra="forbid")
    images_dir: ResolvedPath | None = None
    labels_dir: ResolvedPath | None = None
    dataset_source: DatasetSourceSchema | None = None
    split: SplitSpec = Field(default_factory=lambda: SplitSpec.model_validate({}))
    auto_val: bool = True
    scope: ClassScope | None = None
    tiling: TilingSpec | None = None
    num_channels: int | None = None
    num_classes: int | None = None
    num_ranks: int | None = None
    train_native_size: list[int] | None = None
    train_object_density: float | None = None
    plant_csv_paths: list[ResolvedPath] | None = None

    @model_validator(mode="after")
    def _no_tiling_beside_a_builder(self) -> DataSpec:
        if self.dataset_source is not None and self.tiling is not None:
            raise ValueError("data.tiling is stated beside data.dataset_source: the platform tiles "
                             "no dataset it does not build; a dataset_source builder's training "
                             "body composes its own tiler (ctx.tiled_dataset). Drop data.tiling.")
        return self

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
    """The blocks the default trainer reads, each stated."""

    stages: list[StageSpec]
    optimizer: OptimizerSpec
    scheduler: SchedulerSpec
    checkpoint_every_n_epochs: int
    early_stopping: EarlyStoppingSpec


class TrainerReads(NamedTuple):
    """What training under a config reads (:meth:`TrainConfigSchema.trainer_reads`): the loaders'
    batch size, the config's ``evaluation.conf_threshold`` (stated for a detector), and the
    default trainer's regime (``None`` for a config naming its own ``training_source``)."""

    batch_size: int
    conf_threshold: float | None
    regime: DefaultTrainerRegime | None


class TrainConfigSchema(_Recorded):
    """The one training config shape: trainer keys at the top level, no nested section; a key
    outside these fields is a bespoke loop's own and is kept as stated. It requires
    ``model_source`` and ``data`` alone; what training reads besides is
    :meth:`trainer_reads`."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())
    model_source: ModelSourceSchema
    data: DataSpec
    training_source: str | None = Field(None, min_length=1)
    """A bespoke ``train(ctx)`` loop, a dotted ``module:function``; ``None`` for the default
    trainer."""
    batch_size: int | None = Field(None, ge=1)
    num_workers: int = Field(0, ge=0)
    sampler: dict | None = None
    """The train loader's sampler (``samplers.build_sampler``), its ``name`` beside its own
    values; ``None`` draws every sample once an epoch in shuffled order."""
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
    early_stopping: EarlyStoppingSpec | None = None
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

    def default_trainer_regime(self) -> DefaultTrainerRegime:
        """The regime the default trainer runs at. Refuses (``ValueError``) naming every block of
        it this config leaves unstated."""
        missing = [name for name in DefaultTrainerRegime._fields if getattr(self, name) is None]
        if missing:
            raise ValueError(f"{missing} unstated: the default trainer reads each of them")
        return DefaultTrainerRegime(*(getattr(self, name) for name in DefaultTrainerRegime._fields))

    def trainer_reads(self) -> TrainerReads:
        """What training under this config reads, each stated: the loaders' ``batch_size``, a
        detector's ``evaluation.conf_threshold``, and, when the config names no
        ``training_source``, the default trainer's regime (:meth:`default_trainer_regime`).
        Refuses (``ValueError``) naming every one of them the config leaves unstated, and a
        ``sampler`` block its sampler refuses (``samplers.sampler_settings``)."""
        from tcip_mcp.pipelines.data.samplers import sampler_settings
        from tcip_mcp.pipelines.model_contract import DETECTION_TASKS

        unstated = []
        if self.sampler is not None:
            try:
                sampler_settings(self.sampler)
            except ValueError as exc:
                unstated.append(str(exc))
        if self.batch_size is None:
            unstated.append("batch_size unstated: a run's loaders draw batches of the size its "
                            "config states, under any trainer")
        if (self.model_source.task in DETECTION_TASKS
                and self.evaluation.conf_threshold is None):
            unstated.append("evaluation.conf_threshold unstated: a detector's validation counts "
                            "its boxes at a confidence the run states, under any trainer")
        regime = None
        if self.training_source is None:
            try:
                regime = self.default_trainer_regime()
            except ValueError as exc:
                unstated.append(str(exc))
        if unstated:
            raise ValueError("; ".join(unstated))
        return TrainerReads(cast(int, self.batch_size), self.evaluation.conf_threshold, regime)


def checked_train_config(config: dict, project: Path | None = None
                         ) -> tuple[TrainConfigSchema | None, list[str]]:
    """``config`` validated once against the schema, its paths located against ``project``
    (:data:`ResolvedPath`; ``None`` admits absolute paths only): the validated config and no
    issues, or ``None`` and one issue string per type or structure error (e.g.
    ``batch_size="big"``, a stage stating ``epochs``, a nested ``training`` section, a missing
    ``model_source.task``, a relative path with no project)."""
    try:
        return TrainConfigSchema.model_validate(config, context={"project": project}), []
    except ValidationError as e:
        return None, _issues(e)


def train_config(config: dict, project: Path | None = None) -> TrainConfigSchema:
    """``config`` validated against the schema (:func:`checked_train_config`). Refuses
    (``ValueError``) naming every issue it reports."""
    spec, issues = checked_train_config(config, project)
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
