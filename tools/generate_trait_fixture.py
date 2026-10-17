"""Write the trait listings the frontend tests read, produced by the platform itself.

In projects created through ``tests._web_fixtures.named_project`` and opened through the picker's
route in a scratch session (``tools/fixture_session.py``), each trait entry is proposed through
``traits.propose_trait``, confirmed through ``POST /api/results/traits/confirm`` and served by
``GET /api/results/traits``; the served listings are written, :func:`normalized`, to
``frontend/src/test/traitListings.json`` under the names the tests import.

    python -m tools.generate_trait_fixture
"""

from __future__ import annotations

import json
from pathlib import Path

from tools.fixture_session import (
    opened,
    replaced,
    scratch_session,
    write_fixture,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATED_PATH = (
    _REPO_ROOT
    / "packages"
    / "tcip-web"
    / "frontend"
    / "src"
    / "test"
    / "traitListings.json"
)

_STEM_ENTRY = {
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
    "name": "subject_a",
    "milestone_fractions": [0.5, 0.95],
    "milestone_on": "positive_fraction",
    "operationalizations": {},
}


_FIXED_TIME = "1970-01-01T00:00:00+00:00"
"""What every set revision timestamp reads as in the fixture, so a regeneration compares equal."""


_SET_TIMES = ("proposed_at", "confirmed_at", "withdrawn_at")
"""The revision timestamps a run sets, each read as :data:`_FIXED_TIME` once set."""


def normalized(served: dict[str, dict]) -> dict[str, dict]:
    """Return ``served`` with each set revision timestamp replaced by ``_FIXED_TIME``.

    Those timestamps are the facts that differ between two otherwise identical runs.
    """
    return replaced(
        served,
        lambda key, value: (
            _FIXED_TIME if key in _SET_TIMES and value is not None else value
        ),
    )


def produce() -> dict[str, dict]:
    """Propose and confirm the fixture traits in a scratch session; return the served listings.

    The projects are created and opened in the session, the listing served at each stage, and
    the result :func:`normalized`.
    """
    from tcip_mcp import traits
    from tests._web_fixtures import named_project

    def propose(project: Path, entry: dict) -> traits.TraitRevision:
        return traits.propose_trait(
            project,
            traits.TraitEntry.model_validate(entry),
            rationale="The breeder counts every stem.",
            relayed_note="",
        )

    with scratch_session() as (workspace, client):

        def created(name: str) -> Path:
            project = named_project(workspace / name, name)
            opened(client, project)
            return project.root

        def confirm(revision: traits.TraitRevision) -> None:
            response = client.post(
                "/api/results/traits/confirm",
                json={
                    "trait": revision.entry.name,
                    "revision": revision.number,
                    "entry_sha256": revision.entry_sha256,
                    "user": "grower",
                    "confirmed": True,
                },
            )
            response.raise_for_status()

        def listing() -> dict:
            response = client.get("/api/results/traits")
            response.raise_for_status()
            return response.json()

        served: dict[str, dict] = {}
        setup = created("setup")
        confirm(propose(setup, _STEM_ENTRY))
        second = json.loads(json.dumps(_STEM_ENTRY))
        second["operationalizations"]["per_image_count"]["statement"] = (
            "Only stems above the graft union."
        )
        propose(setup, second)
        served["setup"] = listing()

        results = created("results")
        revision = propose(results, _MILESTONE_ENTRY)
        served["unconfirmed"] = listing()
        confirm(revision)
        served["results"] = listing()
        return normalized(served)


if __name__ == "__main__":
    raise SystemExit(write_fixture(GENERATED_PATH, produce()))
