"""Pydantic v2 config schemas for structural/type validation.

Used by ``training_tools.preflight_config`` to surface type/structure errors. The
runtime trainer still reads the raw config dict; these schemas are validation-only.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, ValidationError


class StageSpec(BaseModel):
    # A stage is a progressive-unfreeze step. The trainer reads ``freeze_to`` and ``epochs``;
    # the optimizer LR comes from the top-level ``optimizer`` block, not per stage. extra is
    # allowed, so an old config still carrying a per-stage ``lr`` validates (it's ignored).
    model_config = ConfigDict(extra="allow")
    epochs: int
    freeze_to: int | None = None


class TrainingSection(BaseModel):
    model_config = ConfigDict(extra="allow")
    batch_size: int = 2
    stages: list[StageSpec] | None = None


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
    # extra="forbid": a misspelled key here is dropped silently by every reader today; refuse
    # it by name instead of building at the builder's own defaults.
    model_config = ConfigDict(extra="forbid")
    builder: str | None = None
    builder_kwargs: dict | None = None
    task: str | None = None
    in_chans: int | None = None
    source_files: list[str] | None = None
    image_stats_sampling: ImageStatsSampling | None = None


class TrainConfigSchema(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())
    model_source: ModelSourceSchema | None = None
    data: dict | None = None
    training: TrainingSection | None = None


def normalize_train_config(config: dict) -> dict:
    """Canonicalize a training config for ``generic_trainer.train()``.

    Hoist the ``training.*`` section onto the top level. The validated/GUI schema nests
    ``stages`` / ``mixed_precision`` / ``batch_size`` / … under ``training``, but ``train()``
    reads them from the top level of ``run.config``, so without this a GUI-launched run
    silently trains the default single stage instead of the configured schedule.

    Top-level wins: a key already present at the top level is never overwritten by the
    nested value: the HPO objective writes tuned params (lr, schedule) flat, and those must
    survive. The ``training`` section is left in
    place for the validated schema and the experiment-record snapshot. Shallow copy: nested
    dicts are shared, so callers must not mutate them in place after normalizing.
    """
    cfg = dict(config)
    training = cfg.get("training")
    if isinstance(training, dict):
        for key, value in training.items():
            cfg.setdefault(key, value)  # top-level wins
    return cfg


_NO_TOP_LEVEL_EVALUATION = object()


def evaluation_section(config: dict) -> dict:
    """The ``evaluation`` block that governs a run, read the one way every caller agrees on.

    A config may carry ``evaluation`` at the top level, nested under ``training.evaluation``, or
    both. Same precedence as ``normalize_train_config``'s hoist: a present top-level key wins,
    whatever its own value, and ``training.evaluation`` is honoured only when the top level
    carries no ``evaluation`` key at all. Read directly off ``config``, never through
    ``normalize_train_config(config).get(...)``: that helper's ``dict(config)`` copy is a
    CPython C-level copy that bypasses a dict-subclass's own ``__getitem__``/``get`` overrides,
    so a caller reading through the copy (an HPO trial's access-tracking config, in particular)
    would never see its own top-level ``evaluation`` read recorded. The trainer,
    ``preflight_config`` and the sweep's direction resolution all call this rather than each
    choosing between the two placements on its own.

    The returned block is the caller's own object, top-level or nested, never copied, when
    ``config`` is a plain dict. When ``config`` is a dict subclass whose ``get`` wraps a nested
    read (an HPO trial's access-tracking config), the returned block is that subclass's own
    freshly wrapped copy of the nested dict instead, per its own stated limitation: a write
    through it is never visible on ``config``. Either way, a caller that wants to keep the block
    must not mutate it in place and expect the mutation to be seen elsewhere.
    """
    top = config.get("evaluation", _NO_TOP_LEVEL_EVALUATION)
    if top is not _NO_TOP_LEVEL_EVALUATION:
        return top or {}
    training = config.get("training") or {}
    return training.get("evaluation") or {}


def validate_train_config_schema(config: dict) -> list[str]:
    """Validate a training config against the pydantic schema; return issue strings.

    Catches type/structure errors (e.g. ``batch_size="big"``, a stage missing ``epochs``).
    Does not enforce ``model_source`` presence (``preflight_config`` keeps its own check).
    """
    issues: list[str] = []
    try:
        TrainConfigSchema.model_validate(config)
    except ValidationError as e:
        for err in e.errors():
            loc = ".".join(str(x) for x in err.get("loc", ()))
            msg = err.get("msg", "invalid")
            issues.append(f"{loc}: {msg}" if loc else msg)
    return issues
