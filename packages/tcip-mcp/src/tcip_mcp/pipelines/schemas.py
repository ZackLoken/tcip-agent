"""Pydantic v2 config schemas. A training config's platform-owned keys are typed here, its
``model_source`` and ``data`` blocks included (:func:`train_config`), each at the top level of
the config; a bespoke ``training_source`` may read its own additional top-level keys. A config
carrying a ``training`` key is refused by name.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    NamedTuple,
    cast,
    get_args,
)

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    PositiveInt,
    Tag,
    ValidationError,
    ValidationInfo,
    model_validator,
)

from tcip_mcp.pipelines.data.selection import ClassScope, Sample
from tcip_mcp.pipelines.derivations import DetectorGeometryDerivation
from tcip_mcp.pipelines.execution import MaskCut

if TYPE_CHECKING:
    from tcip_mcp.subject_registry import Attribute


def _located(path: str, info: ValidationInfo) -> str:
    from tcip_mcp.registry_paths import located

    return str(located(path, (info.context or {}).get("project")))


ResolvedPath = Annotated[str, AfterValidator(_located)]
"""A path a config states, located once where the config enters
(:func:`~tcip_mcp.registry_paths.located` against the ``project`` its validation context
holds), and that location is read and recorded."""


class _Stated(BaseModel):
    """A block a config states: a key outside its fields refuses by name, and ``rationale``
    carries why the block states what it does.
    """

    model_config = ConfigDict(extra="forbid")
    rationale: str = Field(min_length=1)


class SamplerSpec(_Stated):
    """``sampler``: the train loader's sampler ``name`` names (``samplers.build_sampler``), its
    own values kept as stated beside the block's ``rationale``.
    """

    model_config = ConfigDict(extra="allow")
    name: str


class AugmentationSpec(_Stated):
    """``augmentation``: each transform's name (``augmentations.build_augmentation``) to the
    values its constructor takes, kept as stated in order beside the block's ``rationale``.
    """

    model_config = ConfigDict(extra="allow")


class StageSpec(_Stated):
    """One progressive-unfreeze step: the trainer reads ``freeze_to`` and the stage's own
    ``gradient_accumulation_steps``, while the learning rate comes from the top-level
    ``optimizer`` block, never per stage. Any other key, a per-stage ``lr`` or ``epochs``
    included, is refused by name.
    """

    freeze_to: int
    """``0`` trains every parameter, ``-1`` freezes every backbone stage, a positive count
    freezes that many (``generic_trainer.apply_stage_freeze``)."""
    gradient_accumulation_steps: int | None = Field(None, ge=1)
    """``None``, stated or not, is the run's own top-level ``gradient_accumulation_steps``."""


class EarlyStoppingSpec(_Stated):
    """The stop rule that ends a stage and the run: a stage ends once its selection value has not
    improved by ``min_delta`` for ``patience`` epochs, the run ends with the stage whose best
    does not improve by ``min_delta`` on the best of the stage before it, and the run stops as
    diverged after ``nonfinite_passes`` consecutive full training passes with no finite batch
    loss.
    """

    patience: int = Field(ge=1)
    min_delta: float = Field(ge=0)
    nonfinite_passes: int = Field(ge=1)


class TraitRevisionRef(BaseModel):
    """One trait revision as every record names it: the trait, the revision's number and its
    entry's content hash (``traits.entry_sha256``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    trait: str
    trait_revision: int
    trait_revision_sha256: str


def revision_record(ref: TraitRevisionRef | None) -> dict | None:
    """``ref`` as every record stores it, ``None`` for a run whose evaluation names no trait."""
    return None if ref is None else ref.model_dump()


class OptimizerSpec(_Stated):
    """The optimizer every stage builds (``optimizer_factory.build_optimizer``): its family and
    its tuned settings required (both rates, the weight decay, and ``momentum`` for ``sgd`` and
    for no other optimizer).
    """

    name: str
    backbone_lr: float = Field(gt=0)
    head_lr: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    momentum: float | None = Field(None, ge=0)

    @model_validator(mode="after")
    def _momentum_belongs_to_sgd(self) -> OptimizerSpec:
        if (self.name == "sgd") != (self.momentum is not None):
            raise ValueError(
                f"optimizer.momentum is stated for sgd and for no other optimizer; "
                f"this block names {self.name!r} and momentum {self.momentum!r}."
            )
        return self


class HorizonSchedule(_Stated):
    """A schedule built over its own ``horizon_epochs`` scheduled epochs, the most a stage under
    it runs past its warmup.
    """

    horizon_epochs: int = Field(ge=1)


class CosineSchedule(HorizonSchedule):
    """Cosine annealing over the block's horizon down to ``eta_min``."""

    type: Literal["cosine"]
    eta_min: float


class PlateauSchedule(_Stated):
    """The rate times ``factor`` once the run's selection value has not improved in its
    metric's direction (:func:`selection_of`), by ``threshold`` in ``threshold_mode``, for
    ``patience`` epochs, then ``cooldown`` epochs before the count resumes, never below
    ``min_lr``, a change smaller than ``eps`` not applied
    (``torch.optim.lr_scheduler.ReduceLROnPlateau``).
    """

    type: Literal["plateau"]
    factor: float
    patience: int
    threshold: float
    threshold_mode: Literal["rel", "abs"]
    cooldown: int
    min_lr: float
    eps: float


class OneCycleSchedule(HorizonSchedule):
    """One cycle over the block's horizon peaking at ``max_lr``, each of
    ``torch.optim.lr_scheduler.OneCycleLR``'s own settings stated.
    """

    type: Literal["onecycle"]
    max_lr: float
    pct_start: float
    anneal_strategy: Literal["cos", "linear"]
    cycle_momentum: bool
    base_momentum: float
    max_momentum: float
    div_factor: float
    final_div_factor: float
    three_phase: bool


class StepSchedule(_Stated):
    """The rate times ``gamma`` every ``step_size`` epochs."""

    type: Literal["step"]
    step_size: int
    gamma: float


SchedulerSpec = Annotated[
    CosineSchedule | PlateauSchedule | OneCycleSchedule | StepSchedule,
    Field(discriminator="type"),
]
"""The learning-rate schedule each stage builds (``generic_trainer._build_scheduler``): its
``type`` and that type's own settings, each stated, any other type's refused by name."""
SCHEDULER_TYPES: tuple[str, ...] = tuple(
    get_args(schedule.model_fields["type"].annotation)[0]
    for schedule in get_args(get_args(SchedulerSpec)[0])
)
"""The ``scheduler.type`` names ``generic_trainer._build_scheduler`` builds."""


class ScoreWeights(BaseModel):
    """The composite objective's weight on each of its terms
    (``evaluation.compute_composite_objective``), each non-negative so a better term never
    raises the lower-is-better objective.
    """

    model_config = ConfigDict(extra="forbid")
    loss: float = Field(ge=0)
    f1: float = Field(ge=0)
    map50: float = Field(ge=0)


AP_IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
"""The IoU thresholds ``map`` averages average precision over, 0.50 to 0.95 by 0.05: the COCO
convention's own, so ``map`` stays comparable to a COCO-reported figure."""

COMPARABILITY_IOU = AP_IOU_THRESHOLDS[0]
"""The comparability convention: the IoU a match must reach where no trait's criterion governs,
COCO's first average-precision threshold, the one ``map50`` reports at."""

SelectionMetric = Literal[
    "loss",
    "objective",
    "mae",
    "rmse",
    "precision",
    "recall",
    "f1",
    "map",
    "map50",
    "iou_precision",
    "iou_recall",
    "iou_f1",
    "accuracy",
    "rank_acc",
    "quadratic_weighted_kappa",
    "r_squared",
    "mIoU",
    "dice",
    "pixel_acc",
]
"""Every directed scalar metric, by its bare (un-``val_``-prefixed) name; a raw count, a signed
bias and a non-finite state companion have no direction."""

_LOWER_IS_BETTER: frozenset[SelectionMetric] = frozenset(
    {"loss", "objective", "mae", "rmse"}
)

HIGHER_IS_BETTER_BY_METRIC: dict[str, bool] = {
    metric: metric not in _LOWER_IS_BETTER
    for metric in get_args(SelectionMetric)
}
"""Direction of a better value for each :data:`SelectionMetric`."""

CENTER_MATCH_COMPARABILITY_KEYS: frozenset[str] = frozenset(
    {
        "map50",
        "map",
        "iou_precision",
        "iou_recall",
        "iou_f1",
    }
)
"""The metric keys reported comparability-only (``map50_role``) under a trait's own governing
criterion: the IoU-convention average precision and counts."""


def selection_of(metric: str) -> tuple[str, bool]:
    """``metric``, a validated config's ``evaluation.selection_metric``, and whether a higher
    value of it ranks better (:data:`HIGHER_IS_BETTER_BY_METRIC`): the one read of a run's or a
    sweep's selection and its direction.
    """
    return metric, HIGHER_IS_BETTER_BY_METRIC[metric]


class EvaluationSpec(_Stated):
    """What a run's validation pass scores against: the confirmed trait it selects for, the
    selection metric, one with a declared ranking direction (:data:`SelectionMetric`, checked
    over the run by ``generic_trainer.resolve_selection_metric``), and the thresholds and score
    weights the validation pass applies.
    """

    trait: str | None = None
    selection_metric: SelectionMetric
    conf_threshold: float | None = Field(None, ge=0, le=1)
    """The confidence a detector's validation counts boxes at."""
    mask_cut: MaskCut | None = None
    """How an instance segmentation run's validation cuts its predicted masks and simplifies its
    predicted and ground-truth mask rings."""
    iou_threshold: float = COMPARABILITY_IOU
    """The IoU a match must reach where no trait's criterion governs; unstated,
    :data:`COMPARABILITY_IOU`."""
    score_weights: ScoreWeights | None = None


class LrScalingSpec(_Stated):
    """Effective-batch LR scaling: each stage's learning rates times ``(its effective batch / the
    first stage's) ** scale_power``, capped at ``max_lr`` when one is stated.
    """

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
    """What band statistics read: the sampling of whichever of
    ``derivations.band_normalization_stats`` (the exact derivation) or
    ``band_normalization_stats_sampled`` (the windowed one) produced them. ``windows`` pairs each
    source's label with the rectangle read from it, or ``None`` for the exact derivation's own
    whole-image read (``pixel_fraction`` is then ``1.0`` and
    ``window_size``/``max_windows_per_image`` are ``None``); ``max_images`` is the cap on rasters
    drawn and ``seed`` the seed that drew them.
    """

    model_config = ConfigDict(extra="forbid")
    windows: list[tuple[ResolvedPath, ImageStatsWindow | None]] = Field(
        min_length=1
    )
    seed: int
    max_images: int
    pixel_fraction: float
    window_size: int | None = None
    max_windows_per_image: int | None = None


class PretrainedWeights(BaseModel):
    """Pretrained weights a model's builder loads: the ``library`` publishing them and their
    ``name`` there, a timm pretrained tag (``resnet18.a1_in1k``) or a torchvision weights enum
    member (``ResNet18_Weights.IMAGENET1K_V1``).
    """

    model_config = ConfigDict(extra="forbid")
    library: Literal["timm", "torchvision"]
    name: str = Field(min_length=1)


class StatedBeyondWeights(_Stated):
    """Per-band statistics stated with ``rationale`` for a model reading more bands than the
    ``pretrained_weights`` it loads carry statistics for.
    """

    pretrained_weights: PretrainedWeights


def _present(record: Any, names: Sequence[str]) -> dict[str, Any]:
    """Each of ``names`` whose value on ``record`` is not ``None``, under its own name."""
    return {
        name: value
        for name in names
        if (value := getattr(record, name)) is not None
    }


class _Sourced(BaseModel):
    """Values a model builder is handed, beside the ``source`` that gave them."""

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def handed_names(cls) -> tuple[str, ...]:
        """The names of the values a builder is handed: every field but ``source``, and every
        property the class reads off its fields.
        """
        return (
            *(name for name in cls.model_fields if name != "source"),
            *(
                name
                for name, v in vars(cls).items()
                if isinstance(v, property)
            ),
        )

    def handed(self) -> dict[str, Any]:
        """Each value a builder is handed (:meth:`handed_names`) that is present."""
        return _present(self, self.handed_names())


class Normalization(_Sourced):
    """The per-band ``image_mean`` and ``image_std``, in [0, 1] tensor units, a model normalizes
    its input by, and their ``source``: the data's band statistics' own sampling record when no
    weights load (``derivations.band_normalization_stats``), the pretrained weights whose own
    statistics they are (``derivations.pretrained_normalization``), or the statement of every
    band for a model reading more bands than its weights carry statistics for.
    """

    image_mean: list[float]
    image_std: list[Annotated[float, Field(gt=0)]]
    """Each band's std, positive: a band of one value has none to normalize by."""
    source: ImageStatsSampling | PretrainedWeights | StatedBeyondWeights

    @property
    def pretrained_weights(self) -> PretrainedWeights | None:
        """The weights the model's builder loads, read off the source; ``None`` for none."""
        if isinstance(self.source, StatedBeyondWeights):
            return self.source.pretrained_weights
        if isinstance(self.source, PretrainedWeights):
            return self.source
        return None


class DetectorGeometry(_Sourced):
    """A detector's input scale and anchors, the four keyword values the platform hands a
    detection builder: ``min_size`` and ``max_size``, the resize bounds, ``anchor_base_size``,
    the anchor size of the finest pyramid level, each coarser level doubling it, and
    ``aspect_ratios`` (h/w), the ratios every level's anchors take; ``source``, the label of the
    derivation that gave them (``derivations.derive_detector_geometry``) or the statement of
    them.
    """

    min_size: PositiveInt
    max_size: PositiveInt
    anchor_base_size: PositiveInt
    aspect_ratios: tuple[float, ...] = Field(min_length=1)
    source: DetectorGeometryDerivation | _Stated


class StatedDetectorGeometry(DetectorGeometry):
    """A detector geometry a model source states, with its reason."""

    source: _Stated


class ModelDims(NamedTuple):
    """The dimensions a run's model is built at (``model_build.model_dims``): ``in_chans``, the
    band count its sources are read at; ``num_classes`` or ``num_ranks``, the count its ground
    truth derives; ``attributes``, the attribute records a document scope declares. ``None``
    for each the run has none of.
    """

    in_chans: int
    num_classes: int | None = None
    num_ranks: int | None = None
    attributes: tuple[Attribute, ...] | None = None

    def handed(self) -> dict[str, Any]:
        """Each dimension the run has, under its own name."""
        return _present(self, self._fields)


def _handed_once(
    door: str, handed: Sequence[str], kwargs: dict | None, why: str
) -> None:
    """Refuse (``ValueError``) a ``door.builder_kwargs`` naming any of ``handed``, the values
    the platform hands that door's builder, saying ``why`` they are the platform's.
    """
    restated = sorted(set(handed) & set(kwargs or {}))
    if restated:
        raise ValueError(
            f"{door}.builder_kwargs restates {restated}: {why}. Drop {restated} from "
            "builder_kwargs."
        )


class ModelSourceSchema(BaseModel):
    """The importable-builder reference, the task its model is for, the statistics it
    normalizes by (whose source names any pretrained weights its builder loads), and a
    detector's geometry stated for a run that resolves no frame to derive it from. A key
    outside these fields refuses by name, and so does a ``builder_kwargs`` naming a value the
    platform hands (:data:`HANDED_TO_BUILDER`).
    """

    model_config = ConfigDict(extra="forbid")
    builder: str = Field(min_length=1)
    builder_kwargs: dict | None = None
    task: str = Field(min_length=1)
    """The run's task (detection, instance_seg, classification, ...)."""
    source_files: list[ResolvedPath] | None = None
    normalization: Normalization
    detector_geometry: StatedDetectorGeometry | None = None

    @model_validator(mode="after")
    def _model_handed_once(self) -> ModelSourceSchema:
        _handed_once(
            "model_source",
            HANDED_TO_BUILDER,
            self.builder_kwargs,
            "the band count and head sizes a run derives, the statistics and weights "
            "model_source.normalization records and a detector's geometry are the platform's "
            "to hand the builder",
        )
        return self


HANDED_TO_BUILDER = (
    *ModelDims._fields,
    *Normalization.handed_names(),
    *DetectorGeometry.handed_names(),
)
"""What the platform hands a model builder, never its ``builder_kwargs``, read off each
producer: the run's dimensions (:class:`ModelDims`), its normalization's values and weights,
and a detector's geometry (:meth:`_Sourced.handed_names`)."""


class DatasetContext(NamedTuple):
    """What the platform hands a dataset builder, each under its field's name, never its
    ``builder_kwargs``: the ``task``, the ``samples`` of the side being built, the ``scope``
    they were admitted under, and the ``transforms``.
    """

    task: str
    samples: Sequence[Sample]
    scope: ClassScope
    transforms: Any


class DatasetSourceSchema(BaseModel):
    """A bespoke dataset builder; a key outside these fields refuses by name, and so does a
    ``builder_kwargs`` naming a value the platform hands (:class:`DatasetContext`).
    """

    model_config = ConfigDict(extra="forbid")
    builder: str = Field(min_length=1)
    builder_kwargs: dict | None = None
    source_files: list[ResolvedPath] | None = None

    @model_validator(mode="after")
    def _context_handed_once(self) -> DatasetSourceSchema:
        _handed_once(
            "dataset_source",
            DatasetContext._fields,
            self.builder_kwargs,
            "the samples this run trains on, the class space they were admitted under, the "
            "task and the augmentation are the platform's to state, and a builder given a "
            "second value for one of them would build over something the run's own record "
            "does not describe",
        )
        return self


class BoundSplit(BaseModel):
    """``data.split`` bound to the selection recorded at ``selection_dir``: its train and val
    sides as recorded, or, with a ``seed`` stated, redrawn inside them at it. The draw's reason
    is the selection's own record.
    """

    model_config = ConfigDict(extra="forbid")
    selection_dir: ResolvedPath
    seed: int | None = None


class DrawnSplit(_Stated):
    """``data.split`` a run draws itself at ``seed``: each side but ``train`` at its stated
    ``<side>_ratio`` (zero draws no such side, ``train`` the remainder), balanced by foreground
    count when ``stratify_foreground``, grouped by ``grouping``: a named policy, or an explicit
    ``{<capture>/<stem>: group_key}`` map covering every member.
    """

    seed: int
    val_ratio: float
    calibration_ratio: float
    holdout_ratio: float
    stratify_foreground: bool
    grouping: str | dict[str, str]


def _split_kind(value: Any) -> str:
    """``"bound"`` for a ``data.split`` naming a ``selection_dir``, else ``"drawn"``."""
    held = (
        value.get("selection_dir")
        if isinstance(value, dict)
        else getattr(value, "selection_dir", None)
    )
    return "drawn" if held is None else "bound"


SplitSpec = Annotated[
    Annotated[BoundSplit, Tag("bound")] | Annotated[DrawnSplit, Tag("drawn")],
    Discriminator(_split_kind),
]
"""``data.split``: :class:`BoundSplit` when it names a ``selection_dir``, else
:class:`DrawnSplit`."""


Rect = tuple[int, int, int, int]


class SpatialManifest(BaseModel):
    """The within-image split a run's resolution drew over its one source's tile lattice
    (``split_construction.spatial_single_source_split``), recorded in its resolved partition:
    each side's half-open pixel regions, the region identities its train and val tiles carry,
    the buffer it was drawn at (the lattice is the run's ``data.tiling``), what it requested and
    realized, and the raster it was drawn over (``split_construction.raster_identity``).
    """

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


class TilingSpec(_Stated):
    """``data.tiling``, stated for a run that tiles its loader (:func:`run_tiling`): the
    ``datasets.TiledDetectionDataset`` options it states (an unstated edge and overlap derived,
    an unstated sliver fraction derived by the tiler), whether a tile holding no object trains
    (``skip_empty``), and a within-image spatial split's ``buffer``.
    """

    tile_size: int | None = None
    overlap: float | None = None
    sliver_frac: float | None = None
    skip_empty: bool
    keep_regions: list[Rect] | None = None
    buffer: int | None = None

    def tiler_options(self, exclude: frozenset[str] = frozenset()) -> dict:
        """The ``TiledDetectionDataset`` keyword arguments this block states, but ``exclude``;
        an option it leaves ``None`` is omitted.
        """
        return self.model_dump(
            exclude_none=True, exclude={"rationale", "buffer", *exclude}
        )


def run_tiling(task: str, tiling: TilingSpec | None) -> TilingSpec | None:
    """The tiling a run of ``task`` builds its loaders at: ``tiling``, ``None`` for a run stating
    none. A task outside ``model_contract.TILING_TASKS`` trains whole frames, so a ``tiling``
    stated for one refuses (``ValueError``) naming the task and ``data.tiling``.
    """
    from tcip_mcp.pipelines.model_contract import TILING_TASKS

    if tiling is None:
        return None
    if task not in TILING_TASKS:
        raise ValueError(
            f"data.tiling is stated for a run of task {task!r}: only a "
            f"{sorted(TILING_TASKS)} run trains and predicts over tiles, its targets clipped "
            "to each one, so this run would never read the tiling it states. Drop data.tiling."
        )
    return tiling


def run_frame(
    task: str,
    tiling: TilingSpec | None,
    native_size: tuple[int, int] | None,
) -> tuple[int, int] | None:
    """The ``(width, height)`` a run of ``task`` trains each input at: the square tile its
    tiling resolved (:func:`run_tiling`), else ``native_size``, ``None`` for neither.
    """
    tiler = run_tiling(task, tiling)
    if tiler is None or tiler.tile_size is None:
        return native_size
    return (tiler.tile_size, tiler.tile_size)


class _Recorded(BaseModel):
    """A validated block a run's records hold as :meth:`record` dumps it."""

    def record(self, exclude: dict | None = None) -> dict:
        """This block as a run's record holds it: every key stated or resolved, in JSON form,
        but the fields ``exclude`` names (``BaseModel.model_dump``'s nested form).
        """
        return self.model_dump(
            mode="json", exclude_unset=True, exclude=exclude
        )


class DataSpec(_Recorded):
    """``data``: where a run's samples are and how they split, and what its resolution records
    beside them (``scope`` and the sizes). A key outside these fields refuses by name, and so
    does a ``tiling`` beside a ``dataset_source``.
    """

    model_config = ConfigDict(extra="forbid")
    images_dir: ResolvedPath | None = None
    labels_dir: ResolvedPath | None = None
    dataset_source: DatasetSourceSchema | None = None
    split: SplitSpec
    scope: ClassScope | None = None
    tiling: TilingSpec | None = None
    num_channels: int | None = None
    num_classes: int | None = None
    num_ranks: int | None = None
    plant_csv_paths: list[ResolvedPath] | None = None

    @model_validator(mode="after")
    def _no_tiling_beside_a_builder(self) -> DataSpec:
        if self.dataset_source is not None and self.tiling is not None:
            raise ValueError(
                "data.tiling is stated beside data.dataset_source: the platform tiles "
                "no dataset it does not build; a dataset_source builder's training "
                "body composes its own tiler (ctx.tiled_dataset). Drop data.tiling."
            )
        return self

    @property
    def recorded_scope(self) -> ClassScope:
        """The class space this block records (``ClassScope.recorded``, which refuses a block
        recording none).
        """
        return ClassScope.recorded(self.scope)


class TrainGeometry(BaseModel):
    """What a run's resolution read off the frames its train dataset serves: ``native_size``,
    the one ``(width, height)`` every training frame of a run the platform resolved untiled
    shares (``None`` when sizes differ or the run tiles), ``object_density``, a detector's
    objects per pixel over the regions it trains on, and ``detector``, the geometry its builder
    is handed (each ``None`` for any other head).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    native_size: tuple[PositiveInt, PositiveInt] | None
    object_density: float | None
    detector: DetectorGeometry | None


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
    batch size, the config's ``evaluation.conf_threshold`` (stated for a detector),
    ``evaluation.mask_cut`` (stated for instance segmentation), and the default trainer's regime
    (``None`` for a config naming its own ``training_source``).
    """

    batch_size: int
    conf_threshold: float | None
    mask_cut: MaskCut | None
    regime: DefaultTrainerRegime | None


class TrainConfigSchema(_Recorded):
    """The one training config shape: trainer keys at the top level, no nested section; a key
    outside these fields is a bespoke loop's own and is kept as stated. It requires
    ``model_source``, ``data``, ``evaluation``, the seeding and precision every trainer reads,
    and ``rationale``, the reason for the top-level values it states; what training reads
    besides is :meth:`trainer_reads`.
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())
    model_source: ModelSourceSchema
    data: DataSpec
    rationale: str = Field(min_length=1)
    training_source: str | None = Field(None, min_length=1)
    """A bespoke ``train(ctx)`` loop, a dotted ``module:function``; ``None`` for the default
    trainer."""
    batch_size: int | None = Field(None, ge=1)
    num_workers: int = Field(0, ge=0)
    sampler: SamplerSpec | None = None
    """``None`` draws every sample once an epoch in shuffled order."""
    augmentation: AugmentationSpec | None = None
    device: str | None = None
    """``None``: cuda when available, else cpu."""
    seed: int
    deterministic: bool
    """Whether cuDNN runs its deterministic algorithms (``generic_trainer.set_seed``)."""
    mixed_precision: bool
    """Whether a cuda run trains under automatic mixed precision."""
    stages: list[StageSpec] | None = Field(None, min_length=1)
    gradient_accumulation_steps: int = Field(1, ge=1)
    optimizer: OptimizerSpec | None = None
    scheduler: SchedulerSpec | None = None
    stage_warmup_epochs: int = Field(0, ge=0)
    checkpoint_every_n_epochs: int | None = Field(None, ge=0)
    early_stopping: EarlyStoppingSpec | None = None
    lr_scaling: LrScalingSpec | None = None
    log_every_n_batches: int | None = Field(None, ge=1)
    """``None``: ``generic_trainer.BATCH_ROWS_PER_EPOCH`` rows an epoch
    (``generic_trainer.batch_row_ends``)."""
    evaluation: EvaluationSpec

    @model_validator(mode="before")
    @classmethod
    def _refuse_nested_training_section(cls, values: object) -> object:
        if isinstance(values, dict) and "training" in values:
            raise ValueError(NESTED_TRAINING_SECTION_REFUSAL)
        return values

    @model_validator(mode="after")
    def _tiling_on_a_task_that_tiles(self) -> TrainConfigSchema:
        run_tiling(self.model_source.task, self.data.tiling)
        return self

    def default_trainer_regime(self) -> DefaultTrainerRegime:
        """The regime the default trainer runs at. Refuses (``ValueError``) naming every block of
        it this config leaves unstated.
        """
        missing = [
            name
            for name in DefaultTrainerRegime._fields
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(
                f"{missing} unstated: the default trainer reads each of them"
            )
        return DefaultTrainerRegime(
            *(getattr(self, name) for name in DefaultTrainerRegime._fields)
        )

    def trainer_reads(self) -> TrainerReads:
        """What training under this config reads, each stated: the loaders' ``batch_size``, a
        detector's ``evaluation.conf_threshold``, an instance segmentation run's
        ``evaluation.mask_cut``, and, when the config names no ``training_source``, the default
        trainer's regime (:meth:`default_trainer_regime`). Refuses (``ValueError``) naming every
        one of them the config leaves unstated, a ``mask_cut`` stated for a run cutting no mask,
        and a ``sampler`` block its sampler refuses
        (``samplers.sampler_settings``).
        """
        from tcip_mcp.pipelines.data.samplers import sampler_settings
        from tcip_mcp.pipelines.model_contract import DETECTOR_TASKS

        unstated = []
        if self.sampler is not None:
            try:
                sampler_settings(self.sampler)
            except ValueError as exc:
                unstated.append(str(exc))
        if self.batch_size is None:
            unstated.append(
                "batch_size unstated: a run's loaders draw batches of the size its "
                "config states, under any trainer"
            )
        if (
            self.model_source.task in DETECTOR_TASKS
            and self.evaluation.conf_threshold is None
        ):
            unstated.append(
                "evaluation.conf_threshold unstated: a detector's validation counts "
                "its boxes at a confidence the run states, under any trainer"
            )
        cut = self.evaluation.mask_cut
        if (self.model_source.task == "instance_seg") != (cut is not None):
            unstated.append(
                f"evaluation.mask_cut {cut!r} for a {self.model_source.task} run: an instance "
                "segmentation run's validation cuts its masks and simplifies their rings as the "
                "run states, and no other run's validation cuts a mask"
            )
        regime = None
        if self.training_source is None:
            try:
                regime = self.default_trainer_regime()
            except ValueError as exc:
                unstated.append(str(exc))
        if unstated:
            raise ValueError("; ".join(unstated))
        return TrainerReads(
            cast(int, self.batch_size),
            self.evaluation.conf_threshold,
            cut,
            regime,
        )


def checked_train_config(
    config: dict, project: Path | None = None
) -> tuple[TrainConfigSchema | None, list[str]]:
    """``config`` validated once against the schema, its paths located against ``project``
    (:data:`ResolvedPath`; ``None`` admits absolute paths only): the validated config and no
    issues, or ``None`` and one issue string per type or structure error (e.g.
    ``batch_size="big"``, a stage stating ``epochs``, a nested ``training`` section, a missing
    ``model_source.task``, a relative path with no project).
    """
    try:
        return TrainConfigSchema.model_validate(
            config, context={"project": project}
        ), []
    except ValidationError as e:
        return None, _issues(e)


def train_config(
    config: dict, project: Path | None = None
) -> TrainConfigSchema:
    """``config`` validated against the schema (:func:`checked_train_config`). Refuses
    (``ValueError``) naming every issue it reports.
    """
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
