"""Write the meta routes' answers as a fixture file, produced by the platform itself.

In a scratch session (``tools/fixture_session.py``), a project created through
``tests._web_fixtures.named_project`` and opened through the picker's route is given one friction
report (``meta_tools.report_friction``) of :data:`REPORT` and two retrospectives
(``meta_tools.write_retrospective``) of :data:`RETROSPECTIVE`, the second then stored as a
document no retrospective reads as (``tests._record_damage_fixtures.damage_record``). The pages
``GET /api/meta/reports`` and ``GET /api/meta/retrospectives`` serve are written, each record's
time read as an instant and the id each write answered read under the scene's name for it
(``fixture_session.normalized``), to :data:`GENERATED_PATH`, keyed ``reports`` and
``retrospectives``.

    python -m tools.generate_meta_fixture
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests import FRONTEND_SRC
from tools.fixture_session import (
    answered,
    normalized,
    opened,
    scratch_session,
    write_fixture,
)

if TYPE_CHECKING:
    from pydantic import JsonValue

GENERATED_PATH = FRONTEND_SRC / "test" / "metaAnswers.json"

REPORT = {
    "category": "missing_tool",
    "detail": "no lister of assessments",
    "context": {"route": "/api/results/assessments", "attempts": 2},
}
"""The friction report the scene records, sample wording with a structured context."""

RETROSPECTIVE = {
    "project_id": "sample-trial",
    "task": "train the sample counter",
    "worked": "tiling held",
    "did_not_work": "dense clusters merged",
}
"""The stated sections of each retrospective the scene writes, sample wording; the rest are left
unstated."""


def produce() -> dict[str, JsonValue]:
    """Record the report and the retrospectives in a scratch session; return the served pages.

    A route refusing what it should admit raises.
    """
    from tcip_mcp.tools.meta_tools import (
        Retrospective,
        memory_key,
        report_friction,
        write_retrospective,
    )
    from tests._record_damage_fixtures import damage_record
    from tests._web_fixtures import named_project

    with scratch_session() as (workspace, client):
        root = workspace / "meta"
        opened(client, named_project(root, "Meta"))
        report = report_friction(root, **REPORT, actor=None)  # type: ignore[arg-type]
        kept = write_retrospective(root, **RETROSPECTIVE, actor=None)["id"]
        damaged = write_retrospective(root, **RETROSPECTIVE, actor=None)["id"]
        damage_record(memory_key(Retrospective, str(root), damaged), b"[]")
        return normalized(
            {
                "reports": answered(client, "meta_reports", 200),
                "retrospectives": answered(client, "meta_retrospectives", 200),
            },
            workspace,
            minted={
                report["id"]: "<report id>",
                kept: "<retrospective id>",
                damaged: "<damaged retrospective id>",
            },
        )


if __name__ == "__main__":
    raise SystemExit(write_fixture(GENERATED_PATH, produce()))
