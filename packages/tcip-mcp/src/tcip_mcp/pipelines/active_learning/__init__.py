"""Active learning pipeline: scorer and selector modules.

The package also holds the scorer block a review queue states and the catalog of the scorers a
block may name, both read through one model of each registered factory's values.
"""

from collections.abc import Callable
from functools import cache
from typing import Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    JsonValue,
    create_model,
    model_validator,
)


@cache
def values_model(factory: Callable[..., object]) -> type[BaseModel]:
    """Return the model of the values ``factory`` takes beside the task, built once per factory.

    Its fields are the factory's signature's parameters
    (:func:`~tcip_mcp.pipelines.model_build.signature_keywords`, annotations resolved) less the
    one the task is handed under (``scorer.TASK_PARAMETER``): each typed by its annotation
    (``Any`` where it has none) and defaulted where it has a default, extra values forbidden and
    every value validated strictly as the JSON type its annotation states. A factory naming no
    ``TASK_PARAMETER`` is refused (``TypeError``) naming it.
    """
    import inspect

    from tcip_mcp.pipelines.active_learning.scorer import TASK_PARAMETER
    from tcip_mcp.pipelines.model_build import signature_keywords

    parameters = signature_keywords(factory, resolved=True)[0]
    if TASK_PARAMETER not in {p.name for p in parameters}:
        raise TypeError(
            f"scorer factory {factory!r} takes no {TASK_PARAMETER!r}: a scorer factory is "
            f"handed the checkpoint's task as its {TASK_PARAMETER!r} keyword."
        )
    fields: dict[str, Any] = {
        p.name: (
            Any if p.annotation is inspect.Parameter.empty else p.annotation,
            ... if p.default is inspect.Parameter.empty else p.default,
        )
        for p in parameters
        if p.name != TASK_PARAMETER
    }
    return create_model(
        "ScorerValues",
        __config__=ConfigDict(extra="forbid", strict=True),
        **fields,
    )


class ScorerSpec(BaseModel):
    """A review queue's scorer: the factory ``name`` names (``scorer.resolve_scorer``).

    Its values (``model_extra``) are what its factory is handed and what it serializes. For a
    registered ``name`` they are validated by its factory's :func:`values_model`, whose missing,
    extra and type errors stand at each value's own place, and each stated value is replaced by
    the object that model validated it to; a dotted ``module:factory``'s values stay as stated
    and are refused by the factory's own call (``resolve_scorer``).
    """

    model_config = ConfigDict(extra="allow", frozen=True)
    name: str

    @model_validator(mode="after")
    def _admitted(self) -> Self:
        from tcip_mcp.pipelines.active_learning.scorer import SCORER_REGISTRY

        factory = SCORER_REGISTRY.get(self.name)
        if factory is not None and self.model_extra is not None:
            values = values_model(factory).model_validate(self.model_extra)
            self.model_extra.update(
                {
                    name: getattr(values, name)
                    for name in values.model_fields_set
                }
            )
        return self


class ScorerEntry(BaseModel):
    """One registered scorer: its name and the JSON Schema of the values a block naming it states.

    ``values_schema`` is its factory's :func:`values_model`'s, the model a :class:`ScorerSpec` is
    validated by: the values by name under ``properties``, the ones a block must state under
    ``required``, and any definition they reference under ``$defs``.
    """

    name: str
    values_schema: dict[str, JsonValue]


class ScorerCatalog(BaseModel):
    """Every scorer a block may name by its registered name."""

    scorers: list[ScorerEntry]


def scorer_catalog() -> ScorerCatalog:
    """List each registered scorer with the JSON Schema of the values a block naming it states."""
    from tcip_mcp.pipelines.active_learning.scorer import SCORER_REGISTRY

    return ScorerCatalog(
        scorers=[
            ScorerEntry(
                name=name,
                values_schema=values_model(factory).model_json_schema(),
            )
            for name, factory in SCORER_REGISTRY.items()
        ]
    )
