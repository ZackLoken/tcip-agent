"""Pydantic v2 config schemas for structural/type validation; the runtime trainer reads the raw
config dict.

A training config has one shape: every key ``generic_trainer.train()`` reads (``batch_size``,
``stages``, ``mixed_precision``, ``device``, ``seed``, ``evaluation``, ...) sits at the top level
of the config, beside ``model_source`` and ``data``. A config carrying a ``training`` key is
refused by name.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class StageSpec(BaseModel):
    """One progressive-unfreeze step: the trainer reads ``freeze_to`` and ``epochs``, while the
    learning rate comes from the top-level ``optimizer`` block, never per stage. Any other key,
    a per-stage ``lr`` included, is refused by name."""

    model_config = ConfigDict(extra="forbid")
    epochs: int
    freeze_to: int | None = None


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
    """The importable-builder reference. ``extra="forbid"``: a misspelled key here is dropped
    silently by every reader, so it is refused by name instead of building at the builder's
    own defaults."""

    model_config = ConfigDict(extra="forbid")
    builder: str | None = None
    builder_kwargs: dict | None = None
    task: str | None = None
    in_chans: int | None = None
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
    """The one training config shape: trainer keys at the top level, no nested section."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())
    model_source: ModelSourceSchema | None = None
    data: dict | None = None
    batch_size: int = Field(DEFAULT_BATCH_SIZE, ge=1)
    stages: list[StageSpec] | None = None
    evaluation: dict | None = None

    @model_validator(mode="before")
    @classmethod
    def _refuse_nested_training_section(cls, values: object) -> object:
        if isinstance(values, dict) and "training" in values:
            raise ValueError(NESTED_TRAINING_SECTION_REFUSAL)
        return values


def validate_train_config_schema(config: dict) -> list[str]:
    """Validate a training config against the pydantic schema; return issue strings.

    Catches type/structure errors (e.g. ``batch_size="big"``, a stage missing ``epochs``, a
    nested ``training`` section). Does not enforce ``model_source`` presence
    (``preflight_config`` keeps its own check).
    """
    issues: list[str] = []
    try:
        TrainConfigSchema.model_validate(config)
    except ValidationError as e:
        for err in e.errors():
            loc = ".".join(str(x) for x in err.get("loc", ()))
            msg = err.get("msg", "invalid")
            if err.get("type") == "value_error":
                msg = str(err.get("ctx", {}).get("error", msg))
            issues.append(f"{loc}: {msg}" if loc else msg)
    return issues
