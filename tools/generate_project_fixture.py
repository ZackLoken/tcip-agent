"""Write a served project listing and invariances correction as a fixture file.

Both are produced by the platform itself.

In a scratch session (``tools/fixture_session.py``), two projects are created through
``tests._web_fixtures.named_project`` and left unopened; the listing ``GET /api/projects`` serves
for them, the answer ``POST /api/projects/invariances`` gives for one correction of the second,
and that route's refusal of a block whose rotation is out of range are written, the listed
projects ordered by display name and normalized (``fixture_session.normalized``), to
:data:`GENERATED_PATH`.

    python -m tools.generate_project_fixture
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests import FRONTEND_SRC
from tools.fixture_session import normalized, scratch_session, write_fixture

if TYPE_CHECKING:
    from pydantic import JsonValue

    from tools.fixture_session import Stamp

GENERATED_PATH = FRONTEND_SRC / "test" / "projectFixtures.json"

STAMPED: dict[str, Stamp] = {"created": (float, 0.0), "modified": (float, 0.0)}
"""The listed times a run stamps, each read as zero."""


def produce() -> dict[str, JsonValue]:
    """Create, list and correct the fixture projects in a scratch session; return the answers.

    The second project's invariances are corrected and the route refuses an out-of-range block.
    The answers are normalized by :data:`STAMPED`, each minted project id read as its name and
    the workspace as ``<workspace>``.
    """
    from tcip_mcp.project_record import CaptureInvariances
    from tcip_web.routes.projects import InvariancesRequest
    from tests._web_fixtures import INVARIANCES, named_project

    def answered(answers: dict) -> CaptureInvariances:
        return CaptureInvariances.model_validate(
            INVARIANCES.model_dump() | answers
        )

    with scratch_session() as (workspace, client):
        other = named_project(workspace / "other", "other")
        valley = named_project(
            workspace / "valley",
            "valley",
            invariances=answered({"vertical_flip": False, "rotation": 5}),
        )
        listing = client.get("/api/projects")
        listing.raise_for_status()

        def corrected(answers: dict) -> dict:
            return InvariancesRequest(
                id=valley.id, invariances=answered(answers), user="grower"
            ).model_dump()

        correction = client.post(
            "/api/projects/invariances", json=corrected({"rotation": 10})
        )
        correction.raise_for_status()
        request = corrected({})
        request["invariances"]["rotation"] = 400
        refusal = client.post("/api/projects/invariances", json=request)
        assert refusal.status_code == 422, refusal.text
        listed = listing.json()
        listed["projects"].sort(key=lambda p: p["display_name"])
        return normalized(
            {
                "listing": listed,
                "correction": correction.json(),
                "refusal": refusal.json(),
            },
            stamped=STAMPED,
            minted={other.id: "<other id>", valley.id: "<valley id>"},
            places={workspace: "<workspace>"},
        )


if __name__ == "__main__":
    raise SystemExit(write_fixture(GENERATED_PATH, produce()))
