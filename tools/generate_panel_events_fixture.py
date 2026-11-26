"""Write the panel events the backend broadcasts as a fixture file, produced by the platform itself.

In a scratch session (``tools/fixture_session.py``), the project ``tools.generate_open_fixture``
opens is laid down again by its own layout (``generate_open_fixture.captures``) and opened through
the picker's route, and each panel event is posted through ``POST /api/events/{panel}`` as
``web_client.PanelPost``: two banners to the Results panel (``banner``, and ``banner_second``, a
note of two sentences) and one whose text is blank (``banner_blank``), the labels of the
first and of the second photograph of its capture of photographs written to the Annotate panel
(``labels_written``, ``labels_written_second``), and to the ``app`` panel a banner
(``app_banner``), an annotate focus for ``generate_open_fixture.FOCUSED_SUBJECT`` on that
capture (``annotate_focus``) and on the capture whose label document will not read
(``annotate_focus_troubled``) and a canvas state request (``canvas_state_request``); last, the
labels of the first photograph written to the Annotate panel again, posted with the identity
headers of an agent declaring ``generate_training_fixture.DECLARED_CLIENT`` and no terminal
session, whatever the calling shell's environment holds (``labels_written_declared``). Each
panel's socket is opened before the posts, which delivers each event live, and again after them,
which sends each event at connect.

The first Results banner is then dismissed through ``POST /api/events/{panel}/dismissals``: by
:data:`~tools.fixture_session.ACTOR` for a project made beside the open one
(``another_project``), by a person named blank (``no_one``), and by
:data:`~tools.fixture_session.ACTOR` for the open project (``dismissed``). Each panel's socket is
opened once more, which sends what is held after that dismissal, and last the second Results
banner is dismissed while every audit append fails, as
``tests._audit_fixtures.refuse_audit_appends`` makes them fail (``audit_gap``).

The envelopes are written, each event id and the declared agent's minted session numbered
(``fixture_session.normalized``) so every delivery of an event keeps one id, and the open
project's id named, to :data:`GENERATED_PATH`, as ``live``, ``at_connect`` and
``after_dismissal``, each keyed by scene, beside the dismissals' answers under ``dismissals``,
each refusal with its status.

    python -m tools.generate_panel_events_fixture
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING

from tests import FRONTEND_SRC
from tools.fixture_session import (
    ACTOR,
    answered,
    normalized,
    opened,
    scratch_session,
    socket_url,
    write_fixture,
)
from tools.generate_open_fixture import (
    FOCUSED_SUBJECT,
    PROJECT_ID_STAND_IN,
    RAW_IMAGES,
    captures,
)
from tools.generate_training_fixture import DECLARED_CLIENT

if TYPE_CHECKING:
    from pydantic import JsonValue
    from starlette.testclient import WebSocketTestSession

GENERATED_PATH = FRONTEND_SRC / "test" / "panelEvents.json"


def produce() -> dict[str, JsonValue]:
    """Post and dismiss panel events in a scratch session; return what each panel sent and answered.

    The result holds ``live``, what a socket open at the posts received, ``at_connect``, what a
    socket connecting after them received, and ``after_dismissal``, what a socket connecting after
    the first Results banner's dismissal received, each keyed by scene in posting order, and
    ``dismissals``, the dismissal route's answers keyed as the module states.

    A route refusing what it should admit, admitting what it should refuse, or a socket sending
    an event other than its scenes', raises.
    """
    import pytest

    from tcip_mcp import agent_identity
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.web_client import (
        AnnotateFocus,
        Banner,
        CanvasStateRequest,
        LabelsWritten,
        Panel,
        PanelEvent,
        PanelPost,
    )
    from tcip_web.app import PanelDismissal
    from tests._audit_fixtures import refuse_audit_appends
    from tests._web_fixtures import named_project

    with scratch_session() as (workspace, client):
        another = named_project(workspace / "ridge", "ridge")
        project = named_project(workspace / "valley", "valley")
        opened(client, project)
        written = captures(project.root)
        root, date = written["raw"]

        def focus(capture: str) -> AnnotateFocus:
            dataset_root, capture_date = written[capture]
            return AnnotateFocus(
                dataset_root=str(dataset_root),
                subject=FOCUSED_SUBJECT,
                date=capture_date,
                image_index=0,
                mode="box",
                proposal=None,
            )

        def labels_of(image: str) -> LabelsWritten:
            return LabelsWritten(image_path=str(image_dir(root, date) / image))

        posted: list[tuple[str, Panel, PanelEvent]] = [
            (
                "banner",
                "results",
                Banner(text="The agent is computing the counts."),
            ),
            (
                "banner_second",
                "results",
                Banner(
                    text=(
                        "The counts are written to the results table, one row"
                        " for each image of the capture. The images whose"
                        " labels are not finished are listed under the table,"
                        " each with the subject it lacks, so they can be"
                        " labeled and counted again before the table is"
                        " exported."
                    )
                ),
            ),
            ("banner_blank", "results", Banner(text=" ")),
            ("labels_written", "annotate", labels_of(RAW_IMAGES[0])),
            ("labels_written_second", "annotate", labels_of(RAW_IMAGES[1])),
            (
                "app_banner",
                "app",
                Banner(text="The agent moved the canvas to another capture."),
            ),
            ("annotate_focus", "app", focus("raw")),
            ("annotate_focus_troubled", "app", focus("troubled")),
            ("canvas_state_request", "app", CanvasStateRequest()),
        ]
        declared: tuple[str, Panel, PanelEvent] = (
            "labels_written_declared",
            "annotate",
            labels_of(RAW_IMAGES[0]),
        )
        scenes = [*posted, declared]
        panel_of = {name: panel for name, panel, _ in scenes}
        panels = list(dict.fromkeys(panel_of.values()))

        def post(
            panel: Panel, event: PanelEvent, headers: dict[str, str]
        ) -> None:
            answered(
                client,
                "post_panel_event",
                200,
                PanelPost(event=event, project_id=project.id),
                path={"panel": panel},
                headers=headers,
            )

        def received(
            sockets: dict[Panel, WebSocketTestSession],
            among: list[tuple[str, Panel, PanelEvent]],
        ) -> dict[str, JsonValue]:
            return {
                name: sockets[panel].receive_json() for name, panel, _ in among
            }

        def event_id(envelope: JsonValue) -> str:
            assert isinstance(envelope, dict), envelope
            held = envelope["event_id"]
            assert isinstance(held, str), envelope
            return held

        def sent_again(
            view: dict[str, JsonValue], live: dict[str, JsonValue]
        ) -> dict[str, JsonValue]:
            assert {name: event_id(sent) for name, sent in view.items()} == {
                name: event_id(live[name]) for name in view
            }, view
            return view

        def connect_each(
            stack: ExitStack,
        ) -> dict[Panel, WebSocketTestSession]:
            return {
                panel: stack.enter_context(
                    client.websocket_connect(
                        socket_url(client, "panel_ws", panel=panel)
                    )
                )
                for panel in panels
            }

        with ExitStack() as stack:
            open_at_posts = connect_each(stack)
            for _, panel, event in posted:
                post(panel, event, {})
            _, panel, event = declared
            with pytest.MonkeyPatch.context() as patch:
                patch.delenv(
                    agent_identity.TERMINAL_SESSION_ENV, raising=False
                )
                agent_identity.begin(*DECLARED_CLIENT)
            try:
                post(panel, event, agent_identity.http_headers())
            finally:
                agent_identity.end()
            live = received(open_at_posts, scenes)
        with ExitStack() as stack:
            at_connect = sent_again(
                received(connect_each(stack), scenes), live
            )

        def dismiss(
            status: int,
            scene: str,
            *,
            user: str = ACTOR,
            project_id: str = project.id,
        ) -> JsonValue:
            return answered(
                client,
                "dismiss_panel_event",
                status,
                PanelDismissal(
                    user=user,
                    project_id=project_id,
                    event_id=event_id(live[scene]),
                ),
                path={"panel": panel_of[scene]},
            )

        dismissals = {
            "another_project": dismiss(409, "banner", project_id=another.id),
            "no_one": dismiss(400, "banner", user=""),
            "dismissed": dismiss(200, "banner"),
        }
        kept = [scene for scene in scenes if scene[0] != "banner"]
        with ExitStack() as stack:
            after_dismissal = sent_again(
                received(connect_each(stack), kept), live
            )
        with pytest.MonkeyPatch.context() as patch:
            refuse_audit_appends(patch)
            dismissals["audit_gap"] = dismiss(409, "banner_second")
        return normalized(
            {
                "live": live,
                "at_connect": at_connect,
                "after_dismissal": after_dismissal,
                "dismissals": dismissals,
            },
            workspace,
            numbered=frozenset({"event_id", "agent_session"}),
            minted={project.id: PROJECT_ID_STAND_IN},
        )


if __name__ == "__main__":
    raise SystemExit(write_fixture(GENERATED_PATH, produce()))
