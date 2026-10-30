"""Write the panel events the backend broadcasts as a fixture file, produced by the platform itself.

In a scratch session (``tools/fixture_session.py``), the project ``tools.generate_open_fixture``
opens is laid down again by its own layout (``generate_open_fixture.captures``) and opened through
the picker's route, and each panel event is posted through ``POST /api/events/{panel}`` as
``web_client.PanelPost``: a banner to the Results panel (``banner``), the labels of the first
photograph of its capture of photographs written to the Annotate panel (``labels_written``), and
to the ``app`` panel an annotate focus for ``generate_open_fixture.FOCUSED_SUBJECT`` on that
capture (``annotate_focus``) and on the capture whose label document will not read
(``annotate_focus_troubled``) and a canvas state request (``canvas_state_request``). Each
panel's socket then replays what it retained. The envelopes are written, each event id an
arrival stamped numbered (``fixture_session.normalized``), to :data:`GENERATED_PATH`, keyed by
scene.

    python -m tools.generate_panel_events_fixture
"""

from __future__ import annotations

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
from tools.generate_open_fixture import FOCUSED_SUBJECT, RAW_IMAGES, captures

if TYPE_CHECKING:
    from pydantic import JsonValue

GENERATED_PATH = FRONTEND_SRC / "test" / "panelEvents.json"


def produce() -> dict[str, JsonValue]:
    """Post the panel events in a scratch session; return the envelopes each panel replays.

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

        posted: list[tuple[str, Panel, PanelEvent]] = [
            (
                "banner",
                "results",
                Banner(text="The agent is computing the counts."),
            ),
            (
                "labels_written",
                "annotate",
                LabelsWritten(image_path=str(image_dir(root, date) / RAW_IMAGES[0])),
            ),
            ("annotate_focus", "app", focus("raw")),
            ("annotate_focus_troubled", "app", focus("troubled")),
            ("canvas_state_request", "app", CanvasStateRequest()),
        ]
        for _, panel, event in posted:
            answered(
                client,
                "post_panel_event",
                200,
                PanelPost(event=event, project_id=project.id),
                path={"panel": panel},
            )
        envelopes: dict[str, JsonValue] = {}
        for panel in dict.fromkeys(panel for _, panel, _ in posted):
            with client.websocket_connect(
                socket_url(client, "panel_ws", panel=panel)
            ) as ws:
                for name in (name for name, p, _ in posted if p == panel):
                    envelopes[name] = ws.receive_json()
        return normalized(
            envelopes, workspace, numbered=frozenset({"event_id"})
        )


if __name__ == "__main__":
    raise SystemExit(write_fixture(GENERATED_PATH, produce()))
