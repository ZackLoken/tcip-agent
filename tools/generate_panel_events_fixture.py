"""Write the panel events the backend broadcasts as a fixture file, produced by the platform itself.

In a scratch session (``tools/fixture_session.py``), the project ``tools.generate_open_fixture``
opens is laid down again by its own layout (``generate_open_fixture.captures``) and opened through
the picker's route, and each panel event is posted through ``POST /api/events/{panel}`` as
``web_client.PanelPost``: two banners to the Results panel (``banner``, ``banner_second``), the
labels of the first and of the second photograph of its capture of photographs written to the
Annotate panel (``labels_written``, ``labels_written_second``), and to the ``app`` panel a banner
(``app_banner``), an annotate focus for ``generate_open_fixture.FOCUSED_SUBJECT`` on that capture
(``annotate_focus``) and on the capture whose label document will not read
(``annotate_focus_troubled``) and a canvas state request (``canvas_state_request``). Each panel's
socket is opened before the posts, which delivers each event live, and again after them, which
sends each event at connect. The envelopes are written, each event id an arrival stamped numbered
(``fixture_session.normalized``) so both deliveries of an event keep one id and the open
project's id named, to :data:`GENERATED_PATH`, as ``live`` and ``at_connect``, each keyed by scene.

    python -m tools.generate_panel_events_fixture
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING

from tests import FRONTEND_SRC
from tools.fixture_session import (
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

if TYPE_CHECKING:
    from pydantic import JsonValue
    from starlette.testclient import WebSocketTestSession

GENERATED_PATH = FRONTEND_SRC / "test" / "panelEvents.json"


def produce() -> dict[str, JsonValue]:
    """Post the panel events in a scratch session; return the envelopes each panel sent twice.

    The result holds ``live``, what a socket open at the posts received, and ``at_connect``, what
    a socket connecting after them received, each keyed by scene in posting order.

    A route refusing what it should admit raises.
    """
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
    from tests._web_fixtures import named_project

    with scratch_session() as (workspace, client):
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
                Banner(text="The counts are written to the results table."),
            ),
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
        panels = list(dict.fromkeys(panel for _, panel, _ in posted))

        def received(
            sockets: dict[Panel, WebSocketTestSession],
        ) -> dict[str, JsonValue]:
            return {
                name: sockets[panel].receive_json()
                for name, panel, _ in posted
            }

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
                answered(
                    client,
                    "post_panel_event",
                    200,
                    PanelPost(event=event, project_id=project.id),
                    path={"panel": panel},
                )
            live = received(open_at_posts)
        with ExitStack() as stack:
            at_connect = received(connect_each(stack))
        return normalized(
            {"live": live, "at_connect": at_connect},
            workspace,
            numbered=frozenset({"event_id"}),
            minted={project.id: PROJECT_ID_STAND_IN},
        )


if __name__ == "__main__":
    raise SystemExit(write_fixture(GENERATED_PATH, produce()))
