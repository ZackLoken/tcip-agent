"""Pydantic v2 config schemas for structural/type validation (W7).

Used by ``training_tools.validate_config`` to surface type/structure errors and —
via ``ModelSpecSchema``'s validator — registry + channel-compatibility issues,
without duplicating composer logic. The runtime trainer still reads the raw config
dict; these schemas are validation-only.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class BackboneSpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str


class NeckSpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str = "gap"


class HeadSpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str


class ModelSpecSchema(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())
    backbone: BackboneSpec
    neck: NeckSpec = Field(default_factory=lambda: NeckSpec(name="gap"))
    heads: list[HeadSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_components(self):
        # Lazy import: avoid an import cycle / heavy torch import at module load.
        from tcip_mcp.pipelines.composer import validate_model_spec
        issues = validate_model_spec(self.model_dump())
        if issues:
            raise ValueError("; ".join(issues))
        return self


class StageSpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    lr: float
    epochs: int


class TrainingSection(BaseModel):
    model_config = ConfigDict(extra="allow")
    batch_size: int = 2
    stages: list[StageSpec] | None = None


class TrainConfigSchema(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())
    model_spec: ModelSpecSchema | None = None
    model: ModelSpecSchema | None = None
    data: dict | None = None
    training: TrainingSection | None = None


def validate_train_config_schema(config: dict) -> list[str]:
    """Validate a training config against the pydantic schema; return issue strings.

    Catches type errors (e.g. ``batch_size="big"``), empty ``heads``, and — via
    ``ModelSpecSchema`` — registry/channel-compat issues. Does NOT enforce
    ``model_spec`` presence (``validate_config`` keeps its own alias for that).
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
