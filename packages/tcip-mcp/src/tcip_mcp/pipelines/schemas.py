"""Pydantic v2 config schemas for structural/type validation (W7).

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


class TrainConfigSchema(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())
    model_source: dict | None = None
    data: dict | None = None
    training: TrainingSection | None = None


def normalize_train_config(config: dict) -> dict:
    """Canonicalize a training config for ``generic_trainer.train()``.

    Hoist the ``training.*`` section onto the top level. The validated/GUI schema nests
    ``stages`` / ``mixed_precision`` / ``batch_size`` / … under ``training``, but ``train()``
    reads them from the top level of ``run.config`` — so without this a GUI-launched run
    silently trains the default single stage instead of the configured schedule.

    **Top-level wins**: a key already present at the top level is never overwritten by the
    nested value — the HPO objective writes tuned params (lr, schedule) flat, and those must
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
