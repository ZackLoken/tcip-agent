"""Write served trait listings as a fixture file, produced by the platform itself.

In projects created through ``tests._web_fixtures.named_project`` and opened through the picker's
route in a scratch session (``tools/fixture_session.py``), each trait entry is proposed through
``traits.propose_trait``, confirmed through ``POST /api/results/traits/confirm`` and served by
``GET /api/results/traits``. That route's answers are recorded beside the listings: a
confirmation naming a revision by another revision's hash refused (``revision_moved``), a
confirmation (``confirmed``), a withdrawal (``withdrawn``) and a withdrawal whose audit line
fails (``unrecorded``, ``tests._audit_fixtures.refuse_audit_appends``). A third project holds a
trait record whose entry states a localization rule no entry admits, so it will not read
(``tests._record_damage_fixtures.damage_record``). The
listings, keyed by fixture name under ``listings``, and the answers under ``answers``, each
refusal with its status, are written, normalized (``fixture_session.normalized``), to
:data:`GENERATED_PATH`.

    python -m tools.generate_trait_fixture
"""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any

import pytest

from tests import FRONTEND_SRC
from tools.fixture_session import (
    ACTOR,
    answered,
    normalized,
    opened,
    scratch_session,
    write_fixture,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import JsonValue

GENERATED_PATH = FRONTEND_SRC / "test" / "traitListings.json"

_STEM_ENTRY: dict[str, Any] = {
    "name": "stem",
    "delivers": ["stem_count"],
    "positive_state": None,
    "milestone_fractions": [],
    "milestone_on": "",
    "crossing_rule": "",
    "majority_milestone": "",
    "phenology_prefix": "",
    "majority_label": "",
    "count_objective": "count_unbiased",
    "localization": "center_match",
    "localization_tolerance": "half_class_avg_size",
    "localization_tolerance_frac": 0.5,
    "iou_jitter_px": None,
    "iou_margin": None,
    "count_bias_tolerance_frac": 0.1,
    "count_error_tolerance": 2,
    "classifier_agreement_floor": None,
    "ordinal_agreement_floor": None,
    "regression_skill_floor": None,
    "plant_aggregation": "",
    "regression_criterion": "",
    "scale_tolerance_frac": None,
    "holdout_match_quality_floor": 0.5,
    "notes": "",
    "operationalizations": {
        "per_image_count": {
            "statement": "One stem per detected box.",
            "mechanism": "the detector's boxes at the calibrated operating point",
            "measured_subject": "stem",
            "delivered_phenotypes": [],
            "delivered_value_keys": [],
        }
    },
}
_MILESTONE_ENTRY = {
    **_STEM_ENTRY,
    "name": "trait_a",
    "milestone_fractions": [0.5, 0.95],
    "milestone_on": "positive_fraction",
    "operationalizations": {},
}


UNKNOWN_LOCALIZATION = "anywhere"
"""A localization rule no trait entry admits, which the unreadable record's entry states."""


def produce() -> dict[str, JsonValue]:
    """Propose and confirm the fixture traits in a scratch session; return the served answers.

    The projects are created and opened in the session, the listing served at each stage beside
    the moved-revision refusal, and the result normalized.
    """
    from tcip_mcp import traits
    from tcip_web.routes.results import ConfirmRevisionPayload
    from tests._audit_fixtures import refuse_audit_appends
    from tests._record_damage_fixtures import damage_record
    from tests._web_fixtures import named_project

    def propose(project: Path, entry: dict) -> traits.TraitRevision:
        return traits.propose_trait(
            project,
            traits.TraitEntry.model_validate(entry),
            rationale="The breeder counts every stem.",
            relayed_note="",
            actor=None,
        )

    with scratch_session() as (workspace, client):

        def created(name: str) -> Path:
            project = named_project(workspace / name, name)
            opened(client, project)
            return project.root

        def decided(
            revision: traits.TraitRevision,
            status: int = 200,
            *,
            entry_sha256: str | None = None,
            confirmed: bool = True,
        ) -> JsonValue:
            return answered(
                client,
                "confirm_trait_revision",
                status,
                ConfirmRevisionPayload(
                    trait=revision.entry.name,
                    revision=revision.number,
                    entry_sha256=entry_sha256 or revision.entry_sha256,
                    user=ACTOR,
                    confirmed=confirmed,
                ),
            )

        def listing() -> JsonValue:
            return answered(client, "list_traits", 200)

        listings: dict[str, JsonValue] = {}
        answers: dict[str, JsonValue] = {}
        setup = created("setup")
        first = propose(setup, _STEM_ENTRY)
        decided(first)
        second = copy.deepcopy(_STEM_ENTRY)
        second["operationalizations"]["per_image_count"]["statement"] = (
            "Only stems above the graft union."
        )
        proposed = propose(setup, second)
        answers["revision_moved"] = decided(
            proposed, 409, entry_sha256=first.entry_sha256
        )
        listings["setup"] = listing()
        answers["confirmed"] = decided(proposed)
        answers["withdrawn"] = decided(first, confirmed=False)
        with pytest.MonkeyPatch.context() as patch:
            refuse_audit_appends(patch)
            answers["unrecorded"] = decided(proposed, confirmed=False)

        results = created("results")
        revision = propose(results, _MILESTONE_ENTRY)
        listings["unconfirmed"] = listing()
        decided(revision)
        listings["results"] = listing()

        unreadable = created("unreadable")
        name = propose(unreadable, _STEM_ENTRY).entry.name
        record = traits.read_trait(name, unreadable).model_dump(mode="json")
        record["revisions"][0]["entry"]["localization"] = UNKNOWN_LOCALIZATION
        damage_record(
            traits.trait_key(unreadable, name), json.dumps(record).encode()
        )
        listings["unreadable"] = listing()
        return normalized({"listings": listings, "answers": answers}, workspace)


if __name__ == "__main__":
    raise SystemExit(write_fixture(GENERATED_PATH, produce()))
