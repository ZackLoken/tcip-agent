"""Write the agent terminal's route answers as a fixture file, produced by the platform itself.

In a scratch session (``tools/fixture_session.py``) each session runs the fake agent
``tests/fake_terminal_app.py`` named through ``tcip_web.terminal.TERMINAL_CMD_ENV``, which never
turns bracketed paste on, so a request submitted through
``POST /api/terminal/sessions/{id}/submit`` is never given to it. With no project open, a session
of the first provider row is launched through ``POST /api/terminal/sessions``, given a request,
and asked through ``POST .../hand-off`` for its hand-off, which ends it at once
(``handed_off_at_once``). With a project created through ``tests._web_fixtures.named_project``
opened, ``GET /api/terminal/status`` lists every row available (``status``); a session is
launched (``launch``) and ended through ``POST .../end`` (``ended``), another launched
(``relaunch``), given :data:`REQUESTS` and ended (``ended_undelivered``); the first is ended and
asked for a hand-off again, each refused for a session the backend no longer holds
(``end_unknown``, ``hand_off_unknown``); a third, asked for its hand-off, answers it is handing
off (``handed_off``); a fourth, given a request, is ended while every audit append fails
(``end_audit_gap``, ``tests._audit_fixtures.refuse_audit_appends``); and a fifth, whose process
survives its termination (``tests._terminal_fixtures.StubPty``), is refused its end
(``end_survived``). Then, with no override, the status of rows whose executables run the fake
(``tests._terminal_fixtures.fake_executable``) but the third, which names
``tests._terminal_fixtures.ABSENT`` (``status_mixed``), and of rows all naming it
(``status_absent``), are listed. The answers, each refusal with its status, are written,
normalized (``fixture_session.normalized``) with each session id minted read as its order, the
interpreter as ``<python>`` and the fake as ``<fake agent>``, to :data:`GENERATED_PATH`.

    python -m tools.generate_terminal_fixture
"""

from __future__ import annotations

import dataclasses
import sys
from typing import TYPE_CHECKING

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
    from pydantic import JsonValue

GENERATED_PATH = FRONTEND_SRC / "test" / "terminal.json"

UNDELIVERED = "never given"
"""The request an ended session's agent is never given, which the end answers."""

REQUESTS = ("repeat", "middle", "repeat")
"""Requests a session is given in order, two equal, which its end answers in that order."""


def produce() -> dict[str, JsonValue]:
    """Launch, end, refuse and list the terminal's sessions in a scratch session; return answers.

    A route refusing what it should admit, or admitting what it should refuse, raises.
    """
    from tcip_web import terminal as pty_host
    from tcip_web.routes import terminal as terminal_routes
    from tcip_web.routes.terminal import (
        CreateSessionRequest,
        EndSessionRequest,
        SubmitRequest,
        TerminalLaunch,
    )
    from tests._audit_fixtures import refuse_audit_appends
    from tests._terminal_fixtures import (
        ABSENT,
        FAKE,
        StubPty,
        fake_executable,
        wire_stub_spawn,
    )
    from tests._web_fixtures import named_project

    served: dict[str, JsonValue] = {}
    sessions: list[str] = []
    with (
        scratch_session() as (workspace, client),
        pytest.MonkeyPatch.context() as patch,
    ):

        def create() -> tuple[JsonValue, dict[str, str]]:
            launch = answered(
                client,
                "create_session",
                200,
                CreateSessionRequest(
                    provider=pty_host.PROVIDERS[0].id, user=ACTOR
                ),
            )
            sessions.append(TerminalLaunch.model_validate(launch).session_id)
            return launch, {"session_id": sessions[-1]}

        def submit(session: dict[str, str], text: str) -> None:
            answered(
                client,
                "submit_to_session",
                200,
                SubmitRequest(text=text),
                path=session,
            )

        def acted(
            route: str, session: dict[str, str], status: int
        ) -> JsonValue:
            return answered(
                client,
                route,
                status,
                EndSessionRequest(user=ACTOR),
                path=session,
            )

        patch.setenv(pty_host.TERMINAL_CMD_ENV, f"{sys.executable} -u {FAKE}")
        _, unowned = create()
        submit(unowned, UNDELIVERED)
        served["handed_off_at_once"] = acted("hand_off_session", unowned, 200)

        opened(client, named_project(workspace / "terminal", "Terminal"))
        served["status"] = answered(client, "get_status", 200)
        served["launch"], first = create()
        served["ended"] = acted("end_session", first, 200)
        served["relaunch"], second = create()
        for text in REQUESTS:
            submit(second, text)
        served["ended_undelivered"] = acted("end_session", second, 200)
        served["end_unknown"] = acted("end_session", first, 404)
        served["hand_off_unknown"] = acted("hand_off_session", first, 404)
        _, third = create()
        served["handed_off"] = acted("hand_off_session", third, 200)
        acted("end_session", third, 200)
        _, fourth = create()
        submit(fourth, UNDELIVERED)
        with pytest.MonkeyPatch.context() as refusing:
            refuse_audit_appends(refusing)
            served["end_audit_gap"] = acted("end_session", fourth, 409)
        with pytest.MonkeyPatch.context() as surviving:
            survivor = StubPty(survives=True)
            wire_stub_spawn(surviving, survivor)
            _, fifth = create()
            served["end_survived"] = acted("end_session", fifth, 503)
            survivor.exit()
            terminal_routes.shutdown_all()

        patch.delenv(pty_host.TERMINAL_CMD_ENV)
        fake = str(fake_executable(workspace / "bin"))
        rows = pty_host.PROVIDERS
        patch.setattr(
            pty_host,
            "PROVIDERS",
            tuple(
                dataclasses.replace(row, executable=ABSENT if i == 2 else fake)
                for i, row in enumerate(rows)
            ),
        )
        served["status_mixed"] = answered(client, "get_status", 200)
        patch.setattr(
            pty_host,
            "PROVIDERS",
            tuple(dataclasses.replace(row, executable=ABSENT) for row in rows),
        )
        served["status_absent"] = answered(client, "get_status", 200)
        return normalized(
            served,
            workspace,
            minted={
                **{
                    session: f"<session {n}>"
                    for n, session in enumerate(sessions)
                },
                sys.executable: "<python>",
                str(FAKE): "<fake agent>",
            },
        )


if __name__ == "__main__":
    raise SystemExit(write_fixture(GENERATED_PATH, produce()))
