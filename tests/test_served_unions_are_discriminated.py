"""Every union an endpoint answers or takes is told apart by a discriminator, never by a key.

The unions are read off each endpoint's whole annotation through ``pydantic.TypeAdapter``: every
HTTP route's response model and body, every frame and refusal the generated frontend types list
beside the routes (``tools.generate_frontend_types.unrouted_models``), and every MCP tool's return
annotation. The nullable fields are read off the generated frontend types' definitions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, TypeAdapter

if TYPE_CHECKING:
    from collections.abc import Iterator

UNDISCRIMINATED = frozenset({"FlagRequest.properties.target"})
"""Object unions the label document's own shape carries, whose reshaping is the owner's to
decide (a flag's target)."""

ANSWERED_BESIDE_A_DICT = frozenset(
    {
        "cancel_training",
        "ingest_images",
        "launch_training",
        "prioritize_review_queue",
        "propose_annotations",
        "rank_registered_models",
        "register_plant_registry",
        "stage_proposals",
    }
)
"""The MCP tools whose answer beside its model is the ``{"error": ...}`` dict no model declares,
a refusal a typed one is to replace."""

NULLABLE_SETS = {
    "AnnotateFocus": {"bucket", "proposal"},
    "BuildMappingPayload": {"dates", "match_radius"},
    "CanvasBox": {"dash", "casing", "halo", "label"},
    "CanvasPoint": {"dash", "casing", "halo", "label"},
    "CanvasPolygon": {"dash", "casing", "halo", "label"},
    "CanvasPolyline": {"dash", "casing", "halo", "label"},
    "CanvasStatePayload": {"viewport", "active_subject", "shapes"},
    "CaptureChoice": {"subject", "bucket"},
    "ComparedRun": {
        "last_logged_metrics",
        "rows_after_end",
        "subject",
        "dataset_id",
        "dataset_fingerprint",
    },
    "CurveRow": {"accession", "ratio"},
    "DatasetSelection": {"subject", "bucket"},
    "DeliveryEventRecord": {"scale_assessment_id", "plant_mapping"},
    "DrawSpread": {"mean", "std", "min", "max"},
    "Execution": {
        "conf",
        "density",
        "mask_cut",
        "tile_size",
        "overlap",
        "tile_resize",
        "postprocess",
        "cross_tile_nms",
        "sahi_version",
        "rationale",
    },
    "FiniteBest": {"best_value_spread", "split_sensitivity"},
    "Gestures": {"decide", "rect"},
    "GuiState": {"active_subject", "dataset"},
    "ImageBand": {"wavelength_nm", "interpretation"},
    "LaunchedProgram": {"version", "confinement"},
    "NonFiniteBest": {"best_value_spread", "split_sensitivity"},
    "RasterIdentity": {"band_interpretations", "geotransform"},
    "RecordedIdentity": {
        "agent_client_name",
        "agent_client_version",
        "agent_session",
        "terminal_session",
    },
    "Sample": {"row_key", "ground_truth_digest"},
    "ServedAnnotation": {
        "bbox",
        "rings",
        "point",
        "score",
        "created_by",
        "created_at",
        "accepted_by",
        "accepted_at",
    },
    "ServedProposal": {
        "bbox",
        "rings",
        "point",
        "score",
        "created_by",
        "created_at",
        "accepted_by",
        "accepted_at",
        "paired",
        "decision",
    },
    "StandaloneRun": {
        "relaunched_from",
        "images_dir",
        "current_epoch",
        "best",
        "status_error",
    },
    "StateSnapshotMessage": {"project", "bundle"},
    "TrialRun": {
        "relaunched_from",
        "images_dir",
        "current_epoch",
        "best",
        "status_error",
    },
    "RangeTestRun": {
        "relaunched_from",
        "images_dir",
        "current_epoch",
        "best",
        "status_error",
    },
    "UnvalidatedBucket": {"date", "assessment_id"},
    "Stated": {
        "tile",
        "tile_size",
        "overlap",
        "postprocess",
        "cross_tile_nms",
        "conf",
        "masks",
        "mask_cut",
        "rationale",
    },
    "SweepInput": {
        "baseline_params",
        "resources_per_trial",
        "relaunched_from",
        "split_draw_seeds",
        "trial_budget",
    },
    "TraitEntry": {
        "positive_state",
        "iou_jitter_px",
        "iou_margin",
        "count_bias_tolerance_frac",
        "count_error_tolerance",
        "classifier_agreement_floor",
        "ordinal_agreement_floor",
        "regression_skill_floor",
        "scale_tolerance_frac",
        "holdout_match_quality_floor",
    },
}
"""Each served model holding two or more nullable fields, and those fields, each an absence its
reader shows or omits rather than a state it branches on: a model added to the browser's reach
with a nullable pair, or a field made nullable beside another, is a change to this list."""


def _schemas() -> dict[str, dict]:
    """Return every definition the generated frontend types render, by name."""
    from tools.generate_frontend_types import _schemas as rendered

    return rendered()[1]


def endpoint_annotations() -> dict[str, object]:
    """Return every endpoint's whole annotation, by where it is answered or taken.

    That is each HTTP route's response model and each body it takes, each model and named union
    :func:`~tools.generate_frontend_types.unrouted_models` lists, and each MCP tool's return
    annotation, the tools being every one the registry lists (``tcip_web.cli.tool_listing``).
    """
    from typing import get_type_hints

    from fastapi.routing import APIRoute

    from tcip_mcp.server import _TOOLS
    from tcip_web.app import app
    from tcip_web.cli import tool_listing
    from tcip_web.state import StateSnapshotMessage
    from tools.generate_frontend_routes import api_route_contexts
    from tools.generate_frontend_types import unrouted_models

    listed = {tool.name for tool in tool_listing()}
    tools = {fn.__name__: fn for fn, _description, _creates in _TOOLS}
    assert set(tools) == listed, set(tools) ^ listed
    found: dict[str, object] = {}
    for context in api_route_contexts(app):
        route = context.original_route
        if isinstance(route, APIRoute):
            where = f"{context.path_format} {route.name}"
            found[f"{where} answer"] = route.response_model
            for param in route.dependant.body_params:
                found[f"{where} {param.name}"] = param.field_info.annotation
    found |= {
        getattr(m, "__name__", str(m)): m
        for m in [*unrouted_models(), StateSnapshotMessage]
    }
    found |= {
        name: get_type_hints(fn, include_extras=True)["return"]
        for name, fn in tools.items()
    }
    return {where: a for where, a in found.items() if a is not None}


def endpoint_schemas(annotations: dict[str, object]) -> dict[str, dict]:
    """Return every definition the annotations nest, by name, and each annotation's own schema.

    An annotation's own schema is keyed by its key unless it is only a reference to one of the
    definitions.
    """
    refs, combined = TypeAdapter.json_schemas(
        [(w, "serialization", TypeAdapter(a)) for w, a in annotations.items()]
    )
    own = {w: refs[(w, "serialization")] for w in annotations}
    return combined.get("$defs", {}) | {
        w: s for w, s in own.items() if set(s) != {"$ref"}
    }


def _target(part: dict, defs: dict[str, dict]) -> dict:
    """Return the schema ``part`` is, its ``$ref``'s definition when it is one."""
    return defs[part["$ref"].rsplit("/", 1)[-1]] if "$ref" in part else part


def _is_model(part: dict, defs: dict[str, dict]) -> bool:
    """Return whether ``part`` is an object stating fields, inline or by ``$ref``, or a union."""
    target = _target(part, defs)
    return bool(target.get("properties")) or bool(
        {"anyOf", "oneOf"} & target.keys()
    )


def _is_free_form(part: dict, defs: dict[str, dict]) -> bool:
    """Return whether ``part`` is an object stating no field of its own, a ``dict``."""
    target = _target(part, defs)
    return target.get("type") == "object" and not target.get("properties")


def _unions(defs: dict[str, dict]) -> Iterator[tuple[str, dict, list[dict]]]:
    """Yield every union ``defs`` holds: where it is, its node and its members."""

    def walk(
        node: object, where: str
    ) -> Iterator[tuple[str, dict, list[dict]]]:
        if isinstance(node, dict):
            for key in ("anyOf", "oneOf"):
                if key in node:
                    yield where, node, node[key]
            for k, v in node.items():
                yield from walk(v, f"{where}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                yield from walk(v, f"{where}[{i}]")

    for name, schema in defs.items():
        yield from walk(schema, name)


def undiscriminated(defs: dict[str, dict]) -> list[str]:
    """Return where ``defs`` holds a union of two or more models no discriminator tells apart.

    A union naming a discriminator is held to it (:func:`_held`).
    """
    return [
        where
        for where, node, members in _unions(defs)
        if (
            not _held(node["discriminator"], members, defs)
            if "discriminator" in node
            else sum(_is_model(p, defs) for p in members) >= 2
        )
    ]


def beside_a_dict(defs: dict[str, dict]) -> list[str]:
    """Return where ``defs`` holds a union of a model and a ``dict`` stating no field."""
    return [
        where
        for where, _node, members in _unions(defs)
        if any(_is_model(p, defs) for p in members)
        and any(_is_free_form(p, defs) for p in members)
    ]


def _held(
    discriminator: dict, members: list[dict], defs: dict[str, dict]
) -> bool:
    """Return whether ``members`` each state ``discriminator`` as its mapping maps them.

    Every member is a model stating the field as its own ``const`` or ``enum`` values, no two
    members sharing a value, and the mapping maps exactly those values, each to its member.
    """
    field = discriminator["propertyName"]
    stated: dict[str, str | None] = {}
    for member in members:
        prop = _target(member, defs).get("properties", {}).get(field, {})
        values = prop.get("enum", [prop["const"]] if "const" in prop else [])
        if not values or stated.keys() & set(values):
            return False
        stated |= dict.fromkeys(values, member.get("$ref"))
    return discriminator.get("mapping", stated) == stated


def nullable_sets(defs: dict[str, dict]) -> dict[str, set[str]]:
    """Return each definition of ``defs`` holding two or more nullable fields, and those fields."""
    sets = {
        name: {
            prop
            for prop, schema in definition.get("properties", {}).items()
            if any(p.get("type") == "null" for p in schema.get("anyOf", []))
        }
        for name, definition in defs.items()
    }
    return {name: fields for name, fields in sets.items() if len(fields) >= 2}


def test_every_object_union_an_endpoint_carries_is_discriminated() -> None:
    """Every union of two or more models an endpoint carries is held to its discriminator.

    The answers that hold a model beside an untyped refusal dict are the ones enumerated.
    """
    defs = endpoint_schemas(endpoint_annotations())
    assert set(undiscriminated(defs)) == UNDISCRIMINATED
    assert set(beside_a_dict(defs)) == ANSWERED_BESIDE_A_DICT


def test_no_served_model_gains_a_nullable_pair_unnamed() -> None:
    """The served models holding two or more nullable fields are the ones enumerated."""
    assert nullable_sets(_schemas()) == NULLABLE_SETS


class _Kept(BaseModel):
    kind: Literal["kept"] = "kept"
    value: int


class _Other(BaseModel):
    kind: Literal["other"] = "other"
    reason: str


class _Untagged(BaseModel):
    reason: str


class _Holding(BaseModel):
    tagged: _Kept | _Other = Field(discriminator="kind")
    untagged: _Kept | _Untagged


class _Paired(BaseModel):
    first: str | None
    second: int | None


def test_the_walk_finds_an_untagged_union_and_a_nullable_pair() -> None:
    """A union pydantic renders with no discriminator, and a nullable pair, are each found.

    A bare union an endpoint answers directly is found as that endpoint's own schema.
    """
    defs = endpoint_schemas({"_Holding": _Holding, "bare": _Kept | _Untagged})
    assert sorted(undiscriminated(defs)) == [
        "_Holding.properties.untagged",
        "bare",
    ]
    assert nullable_sets({"_Paired": _Paired.model_json_schema()}) == {
        "_Paired": {"first", "second"}
    }


def test_the_walk_holds_a_discriminated_union_to_its_members() -> None:
    """A member whose schema lost its tag, or a mapping naming another member, is found."""
    defs = endpoint_schemas({"_Holding": _Holding})
    assert undiscriminated(defs) == ["_Holding.properties.untagged"]
    del defs["_Other"]["properties"]["kind"]
    assert "_Holding.properties.tagged" in undiscriminated(defs)
