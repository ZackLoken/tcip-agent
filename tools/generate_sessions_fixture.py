"""Write the sessions route's answers as a fixture file, produced by the platform itself.

In a scratch session (``tools/fixture_session.py``), a project created through
``tests._web_fixtures.named_project`` and opened through the picker's route is given one
contribution through ``POST /api/sessions/image_event``, built through the route's request
model; its session is then ended through ``POST /api/sessions/end`` and the same contribution
sent again, a contribution with no time, no annotations and no confirmed negative is sent, and
a new contribution is sent while every audit append fails, as
``tests._audit_fixtures.refuse_audit_appends`` makes them fail; a contribution naming the ended
session, and one by :data:`ANOTHER` naming the session the unlogged one landed in, are then sent.
The answers are written, each session's start and end stamps read as instants
(``fixture_session.normalized``), to :data:`GENERATED_PATH`, keyed ``landed``,
the contribution
landing in a new session, ``repeated``, the repeat answering the session it first landed in,
ended, ``empty``, the contribution with nothing to record, ``audit_gap``, the refusal of the
contribution that landed without its line, ``ended_named`` and ``another_named``, the
refusals of the two naming a session, each refusal with its status, ``listing``, the sessions
``GET /api/sessions/load`` then lists, and ``unopened``, its refusal before any project was open.

    python -m tools.generate_sessions_fixture
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests import FRONTEND_SRC
from tools.fixture_session import (
    ACTOR,
    answered,
    detail_of,
    normalized,
    opened,
    scratch_session,
    write_fixture,
)

if TYPE_CHECKING:
    from pydantic import JsonValue

GENERATED_PATH = FRONTEND_SRC / "test" / "sessionAnswers.json"

ANOTHER = "another"
"""A person other than :data:`~tools.fixture_session.ACTOR`, whose contribution names their
session."""


def produce() -> dict[str, JsonValue]:
    """Land, end and repeat a contribution, send an empty, an unlogged and two refused; return them.

    The contributions go to a scratch session's sessions route; a route refusing what it should
    admit, or admitting what it should refuse, raises.
    """
    from tcip_web.routes.audit_gap import AuditGapRefusal
    from tcip_web.routes.sessions import (
        ImageEventPayload,
        SessionRef,
        SessionWrite,
    )
    from tests._audit_fixtures import refuse_audit_appends
    from tests._web_fixtures import named_project

    with scratch_session() as (workspace, client):
        unopened = answered(client, "load_sessions", 409)
        project = named_project(workspace / "sessions", "Sessions")
        opened(client, project)
        contribution = ImageEventPayload(
            contribution_id="c1",
            image_name="a.jpg",
            seconds=12.0,
            annotations_added=2,
            activity="new_annotation",
            user=ACTOR,
            project_id=project.id,
            started=None,
        )
        landed = answered(client, "image_event", 200, contribution)
        session = SessionWrite.model_validate(landed).session
        assert session is not None, landed
        end = SessionRef(project_id=project.id, started=session.started)
        answered(client, "close_session", 200, end)
        repeated = answered(client, "image_event", 200, contribution)
        nothing = contribution.model_copy(
            update={
                "contribution_id": "c2",
                "seconds": 0.0,
                "annotations_added": 0,
                "activity": "review",
            }
        )
        empty = answered(client, "image_event", 200, nothing)
        unlogged = contribution.model_copy(update={"contribution_id": "c3"})
        with pytest.MonkeyPatch.context() as patch:
            refuse_audit_appends(patch)
            gap = answered(client, "image_event", 409, unlogged)
        held = AuditGapRefusal.model_validate(detail_of(gap)).committed
        held_session = SessionWrite.model_validate(held).session
        assert held_session is not None, gap
        ended_named = answered(
            client,
            "image_event",
            409,
            contribution.model_copy(
                update={"contribution_id": "c4", "started": session.started}
            ),
        )
        another_named = answered(
            client,
            "image_event",
            409,
            contribution.model_copy(
                update={
                    "contribution_id": "c5",
                    "user": ANOTHER,
                    "started": held_session.started,
                }
            ),
        )
        return normalized(
            {
                "landed": landed,
                "repeated": repeated,
                "empty": empty,
                "audit_gap": gap,
                "ended_named": ended_named,
                "another_named": another_named,
                "listing": answered(client, "load_sessions", 200),
                "unopened": unopened,
            },
            workspace,
        )


if __name__ == "__main__":
    raise SystemExit(write_fixture(GENERATED_PATH, produce()))
