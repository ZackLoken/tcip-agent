"""Write the project listing and invariances correction the frontend tests read.

Both are produced by the platform itself.

In a scratch session (``tools/fixture_session.py``), two projects are created through
``tests._web_fixtures.named_project`` and left unopened; the listing ``GET /api/projects`` serves
for them, the answer ``POST /api/projects/invariances`` gives for one correction of the second,
and that route's refusal of a block whose rotation is out of range are written,
:func:`normalized`, to ``frontend/src/test/projectFixtures.json``.

    python -m tools.generate_project_fixture
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.fixture_session import replaced, scratch_session, write_fixture

_REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATED_PATH = (
    _REPO_ROOT
    / "packages"
    / "tcip-web"
    / "frontend"
    / "src"
    / "test"
    / "projectFixtures.json"
)


def normalized(served: dict, workspace: Path, ids: dict[str, str]) -> dict:
    """Return ``served`` with the values a run fills differently replaced by what they stand for.

    Each minted project id in ``ids`` becomes its project's name, each listed path the workspace
    placeholder and its folder name, the workspace the placeholder, and each listed time zero;
    the listed projects are ordered by display name.
    """

    def leaf(key: str | None, value: Any) -> Any:
        if key in ("created", "modified"):
            return 0.0
        if key == "workspace":
            return "<workspace>"
        if key == "path":
            return f"<workspace>/{Path(value).relative_to(workspace)}"
        if isinstance(value, str) and value in ids:
            return f"<{ids[value]} id>"
        return value

    out = replaced(served, leaf)
    out["listing"]["projects"].sort(key=lambda p: p["display_name"])
    return out


def produce() -> dict:
    """Create, list and correct the fixture projects in a scratch session; return the answers.

    The second project's invariances are corrected and the route refuses an out-of-range block;
    the answers are :func:`normalized`.
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
        return normalized(
            {
                "listing": listing.json(),
                "correction": correction.json(),
                "refusal": refusal.json(),
            },
            workspace,
            {other.id: "other", valley.id: "valley"},
        )


if __name__ == "__main__":
    raise SystemExit(write_fixture(GENERATED_PATH, produce()))
