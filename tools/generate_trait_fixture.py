"""Write the trait listings the frontend tests read, produced by the platform itself.

In a scratch project, each trait entry is proposed through ``traits.propose_trait``, confirmed
through ``POST /api/results/traits/confirm`` and served by ``GET /api/results/traits``; the served
listings are written, :func:`normalized`, to ``frontend/src/test/traitListings.json`` under the
names the tests import.

    python tools/generate_trait_fixture.py
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATED_PATH = (
    _REPO_ROOT / "packages" / "tcip-web" / "frontend" / "src" / "test" / "traitListings.json")

_STEM_ENTRY = {
    "name": "stem", "delivers": ["stem_count"], "positive_value": "", "milestone_fractions": [],
    "milestone_on": "", "majority_milestone": "", "phenology_prefix": "", "majority_label": "",
    "count_objective": "count_unbiased", "localization": "center_match",
    "localization_tolerance": "half_class_avg_size", "localization_tolerance_frac": 0.5,
    "count_bias_tolerance_frac": None, "count_error_tolerance": None,
    "classifier_agreement_floor": None, "ordinal_agreement_floor": None,
    "regression_skill_floor": None, "scale_tolerance_frac": None,
    "holdout_match_quality_floor": 0.5, "notes": "",
    "operationalizations": {"per_image_count": {
        "statement": "One stem per detected box.",
        "mechanism": "the detector's boxes at the calibrated operating point",
        "measured_subject": "stem", "delivered_phenotypes": [], "delivered_value_keys": []}},
}
_MILESTONE_ENTRY = {
    **_STEM_ENTRY, "name": "subject_a", "milestone_fractions": [0.5, 0.95],
    "milestone_on": "positive_fraction", "operationalizations": {},
}


_FIXED_TIME = "1970-01-01T00:00:00+00:00"
"""What every set revision timestamp reads as in the fixture, so a regeneration compares equal."""


def normalized(served: dict[str, dict]) -> dict[str, dict]:
    """``served`` with each revision's set timestamps replaced by ``_FIXED_TIME`` and its proposing
    agent's fields emptied, the facts that differ between two otherwise identical runs."""
    out = json.loads(json.dumps(served))
    for listing in out.values():
        for trait in listing["traits"]:
            for revision in trait["revisions"]:
                for key in ("proposed_at", "confirmed_at", "withdrawn_at"):
                    if revision[key] is not None:
                        revision[key] = _FIXED_TIME
                revision["proposing_agent"] = dict.fromkeys(revision["proposing_agent"])
    return out


def listings(root: Path) -> dict[str, dict]:
    """Propose and confirm the fixture traits under ``root``, serving the listing at each stage,
    :func:`normalized`. ``root`` must lie under the pinned workspace and state root."""
    from fastapi.testclient import TestClient
    from tcip_mcp import traits
    from tcip_web.app import app

    client = TestClient(app, base_url="http://127.0.0.1")

    def propose(project: Path, entry: dict) -> traits.TraitRevision:
        return traits.propose_trait(
            project, traits.TraitEntry.model_validate(entry),
            rationale="The breeder counts every stem.", relayed_note="")

    def confirm(project: Path, revision: traits.TraitRevision) -> None:
        response = client.post("/api/results/traits/confirm", json={
            "project_root": str(project), "trait": revision.entry.name,
            "revision": revision.number, "entry_sha256": revision.entry_sha256,
            "user": "grower", "confirmed": True})
        response.raise_for_status()

    def listing(project: Path) -> dict:
        response = client.get("/api/results/traits", params={"project_root": str(project)})
        response.raise_for_status()
        return response.json()

    served: dict[str, dict] = {}
    setup = root / "setup"
    setup.mkdir()
    confirm(setup, propose(setup, _STEM_ENTRY))
    second = json.loads(json.dumps(_STEM_ENTRY))
    second["operationalizations"]["per_image_count"]["statement"] = (
        "Only stems above the graft union.")
    propose(setup, second)
    served["setup"] = listing(setup)

    results = root / "results"
    results.mkdir()
    revision = propose(results, _MILESTONE_ENTRY)
    served["unconfirmed"] = listing(results)
    confirm(results, revision)
    served["results"] = listing(results)
    return normalized(served)


def main() -> int:
    with tempfile.TemporaryDirectory() as scratch:
        workspace = Path(scratch) / "workspace"
        workspace.mkdir()
        os.environ["TCIP_WORKSPACE"] = str(workspace)
        os.environ["TCIP_STATE_ROOT"] = str(workspace)
        from tcip_store.binding import bind_default

        bind_default()
        served = listings(workspace)
        import tcip_store

        tcip_store.close_connections()
    GENERATED_PATH.write_text(json.dumps(served, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {GENERATED_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
