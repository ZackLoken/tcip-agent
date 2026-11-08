"""A trait entry's milestone fields agree with each other or the entry's type refuses it.

Each entry is proposed through the MCP server's call path of ``propose_trait`` over a project the
platform made, and the answer the client receives is read: a value the type refuses answers as the
library's argument refusal, naming the argument model and carrying the type's sentence, before the
door's body runs; an entry the type admits reaches the door, which answers its own refusals as
``{"error": ...}``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import pytest

import tcip_store
from tcip_mcp import traits
from tests import _trait_fixtures as fx
from tests._server_calls import called_through_the_server

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import JsonValue

    from tcip_mcp.workspace import BoundProject

ARGUMENT_REFUSAL = "validation error for propose_traitArguments\nentry"
"""How the server's argument validation opens a refusal of the ``entry`` parameter."""


def _milestones(**fields: object) -> dict[str, Any]:
    """Return a count entry delivering ``leaf_length`` as JSON, the milestone ``fields`` over it."""
    return {
        **fx.entry("leaf", ("leaf_length",)).model_dump(mode="json"),
        **fields,
    }


def _proposed(entry: dict[str, Any], bound: BoundProject) -> JsonValue:
    """Return the client's answer to proposing ``entry`` through the server bound to ``bound``."""
    return called_through_the_server(
        "propose_trait",
        {"entry": entry, "rationale": "the breeder's own milestone terms"},
        bound=bound,
    )


def _refused(entry: dict[str, Any], bound: BoundProject, project: Path) -> str:
    """Return the argument refusal ``entry`` answers, asserting nothing was written."""
    answer = _proposed(entry, bound)
    assert isinstance(answer, str), answer
    assert ARGUMENT_REFUSAL in answer, answer
    assert traits.trait_names(project) == []
    return answer


def test_a_majority_that_is_not_one_of_the_fractions_is_refused_naming_both(
    bound: BoundProject, project: Path
) -> None:
    answer = _refused(
        _milestones(
            milestone_fractions=[0.1, 0.9],
            majority_fraction=0.95,
            majority_label="majority",
        ),
        bound,
        project,
    )

    assert (
        "majority_fraction 0.95 is not one of milestone_fractions [0.1, 0.9]"
        in answer
    )


def test_fractions_that_are_not_whole_percents_are_refused_before_any_collision(
    bound: BoundProject, project: Path
) -> None:
    answer = _refused(
        _milestones(milestone_fractions=[0.049, 0.051]), bound, project
    )

    assert (
        "milestone_fractions [0.049, 0.051] are not whole percents" in answer
    )
    assert "05per" not in answer


def test_equal_fractions_are_refused_as_one_key_naming_both(
    bound: BoundProject, project: Path
) -> None:
    answer = _refused(
        _milestones(milestone_fractions=[0.5, 0.5]), bound, project
    )

    assert (
        "milestone_fractions 0.5 and 0.5 both give the key '50per'" in answer
    )


@pytest.mark.parametrize(
    "fraction, diagnostic",
    [
        (0.0, "greater than 0"),
        (-0.05, "greater than 0"),
        (1.5, "less than or equal to 1"),
        (math.nan, "a finite number"),
        (math.inf, "a finite number"),
    ],
    ids=["zero", "negative", "above_one", "nan", "infinite"],
)
def test_a_fraction_outside_zero_to_one_or_not_finite_is_refused_by_its_field(
    bound: BoundProject, project: Path, fraction: float, diagnostic: str
) -> None:
    """A fraction outside (0, 1], NaN or infinity is refused by the field's own constraint.

    NaN and infinity are refused by the argument model before any write, not by the store's
    encoder on the proposal's write path.
    """
    answer = _refused(
        _milestones(milestone_fractions=[0.5, fraction]), bound, project
    )

    assert f"milestone_fractions.1\n  Input should be {diagnostic}" in answer


@pytest.mark.parametrize("fraction", [0.125, 0.005])
def test_a_fraction_that_is_not_a_whole_percent_is_refused(
    bound: BoundProject, project: Path, fraction: float
) -> None:
    answer = _refused(
        _milestones(milestone_fractions=[fraction]), bound, project
    )

    assert f"milestone_fractions [{fraction}] are not whole percents" in answer


@pytest.mark.parametrize(
    "fields, refusal",
    [
        (
            {"majority_fraction": 0.5, "majority_label": ""},
            "majority_label is empty while majority_fraction is 0.5",
        ),
        (
            {"majority_fraction": None, "majority_label": "peak"},
            "majority_label 'peak' is set while majority_fraction is None",
        ),
        (
            {"majority_fraction": 0.5, "majority_label": "50per"},
            "majority_label '50per' is shaped like a milestone key",
        ),
        (
            {"majority_fraction": 0.05, "majority_label": "95per"},
            "majority_label '95per' is shaped like a milestone key",
        ),
        (
            {"majority_fraction": 0.05, "majority_label": "05per "},
            "majority_label '05per ' is not a token free of whitespace",
        ),
        (
            {"majority_fraction": 0.05, "majority_label": " "},
            "majority_label ' ' is not a token free of whitespace",
        ),
        (
            {"phenology_prefix": " bud"},
            "phenology_prefix ' bud' is not a token free of whitespace",
        ),
    ],
    ids=[
        "empty_label_with_majority",
        "label_without_majority",
        "label_equal_to_a_key",
        "key_shaped_label_outside_the_fractions",
        "label_with_trailing_space",
        "blank_label",
        "prefix_with_leading_space",
    ],
)
def test_a_label_or_prefix_that_cannot_name_its_column_is_refused(
    bound: BoundProject,
    project: Path,
    fields: dict[str, object],
    refusal: str,
) -> None:
    answer = _refused(
        _milestones(milestone_fractions=[0.05, 0.5], **fields), bound, project
    )

    assert refusal in answer


@pytest.mark.parametrize(
    "majority, diagnostic",
    [
        (0.0, "Input should be greater than 0"),
        (1.5, "Input should be less than or equal to 1"),
        (math.nan, "Input should be a finite number"),
        (0.125, "majority_fraction 0.125 is not a whole percent"),
    ],
    ids=["zero", "above_one", "nan", "not_a_whole_percent"],
)
def test_a_majority_over_empty_fractions_is_refused_on_its_own_constraint(
    bound: BoundProject, project: Path, majority: float, diagnostic: str
) -> None:
    answer = _refused(
        _milestones(majority_fraction=majority, majority_label="majority"),
        bound,
        project,
    )

    assert diagnostic in answer
    assert "is not one of milestone_fractions" not in answer


def test_a_binary_float_that_is_a_whole_percent_is_accepted(
    bound: BoundProject, project: Path
) -> None:
    answer = _proposed(_milestones(milestone_fractions=[0.07, 1.0]), bound)

    assert isinstance(answer, dict) and "error" not in answer, answer
    assert traits.read_trait(
        "leaf", project
    ).latest.entry.milestone_fractions == (
        0.07,
        1.0,
    )


def _crossing(**fields: object) -> dict[str, Any]:
    """Return the fixture crossing entry stating its crossing kind as JSON, ``fields`` over."""
    stated = fx.with_operationalization(
        fx.CROSSING_SPEC,
        traits.STATE_CROSSING_DATES,
        measured_subject="flower",
        delivered_phenotypes=fx.CROSSING_SPEC.delivers,
    )
    return {**stated.model_dump(mode="json"), **fields}


def test_a_crossing_entry_with_a_majority_and_no_fractions_is_asked_for_its_fractions(
    bound: BoundProject, project: Path
) -> None:
    """The type never pre-empts the breeder's question for a crossing entry's unstated fractions."""
    answer = _proposed(
        _crossing(
            milestone_fractions=[],
            majority_fraction=0.5,
            majority_label="majority",
        ),
        bound,
    )

    assert isinstance(answer, dict), answer
    assert "['milestone_fractions'] unauthored" in answer["error"]
    assert traits.QUESTIONS["milestone_fractions"] in answer["error"]


def test_a_draft_with_a_majority_and_no_fractions_is_accepted(
    bound: BoundProject, project: Path
) -> None:
    answer = _proposed(
        _milestones(majority_fraction=0.5, majority_label="majority"), bound
    )

    assert isinstance(answer, dict) and "error" not in answer, answer
    assert (
        traits.read_trait("leaf", project).latest.entry.majority_fraction
        == 0.5
    )


def test_a_count_entry_with_no_milestone_fields_is_accepted(
    bound: BoundProject, project: Path
) -> None:
    answer = _proposed(
        fx.COUNT_SPEC.model_dump(mode="json"),
        bound,
    )

    assert isinstance(answer, dict) and "error" not in answer, answer
    assert traits.trait_names(project) == [fx.COUNT_TRAIT]


def test_an_entry_with_fractions_and_no_operationalization_is_accepted(
    bound: BoundProject, project: Path
) -> None:
    answer = _proposed(
        _milestones(
            milestone_fractions=[0.5, 0.95], milestone_on="positive_fraction"
        ),
        bound,
    )

    assert isinstance(answer, dict) and "error" not in answer, answer
    assert traits.read_trait(
        "leaf", project
    ).latest.entry.milestone_fractions == (
        0.5,
        0.95,
    )


def test_a_stored_entry_the_type_refuses_is_refused_on_read_naming_the_record(
    project: Path,
) -> None:
    fx.propose(
        project, fx.entry("leaf", ("leaf_length",), milestone_fractions=(0.5,))
    )
    stored = traits.read_trait("leaf", project).model_dump(mode="json")
    entry = stored["revisions"][0]["entry"]
    entry["majority_fraction"] = 0.3
    entry["majority_label"] = "majority"
    tcip_store.replace(traits.trait_key(project, "leaf"), stored)

    with pytest.raises(tcip_store.ReadRefusedError) as refused:
        traits.read_trait("leaf", project)

    assert "the record of trait 'leaf'" in str(refused.value)
    assert (
        "majority_fraction 0.3 is not one of milestone_fractions [0.5]"
        in str(refused.value)
    )
